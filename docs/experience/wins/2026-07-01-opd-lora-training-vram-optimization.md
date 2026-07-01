# OPD LoRA writeback VRAM optimization on Qwen3.6-27B-FP8

## Context

Masked-writeback OPD steps on a Qwen3.5 / Qwen3.6 LoRA student burned
peak VRAM in two places: (1) the tape backward retained every op's
grad in the returned map for the whole walk, (2) LoRA matmul and MoE
grouped-linear backwards materialized full-shape transients before
merging, (3) SDPA recompute-backward went CPU-round-trip on
`head_dim > 256` and allocated many chunked device transients, and
(4) `AdamW::zero_grad` kept the freed grads pinned by zero-filling
in place. Between writebacks the cudarc allocator retained pages the
next step then had to re-alloc from the OS.

`docs/experience/errors/2026-06-24-agent-opd-writeback-oom-*.md` and
follow-ups repeatedly landed here.

## What Worked

Landed in `8ba80cc6` + `a1c6af4a` + `ad8d6d0e`:

* `Tape::backward_accumulate_only[/_profiled]` — walks the tape
  without building the returned `grads` map, drops each op's output
  grad the moment its inputs merge. `masked_writeback_ce_step[/
  _frozen_prompt_kv]` switched to it. Unit tests cover the invariant.
* `TensorStore::offload_checkpoint_to_host` — pooled Vec<f32> buffer
  reuse, min-bytes threshold (`ARLE_OPD_CHECKPOINT_OFFLOAD_MIN_BYTES`
  default 2 MiB) so we don't offload sub-page tensors that cost more
  in syscalls than they save. Recycles on `free` / `ensure_device`.
* `Backend::readback_into` — dtoh into a caller-provided slice; the
  CUDA impl takes the `memcpy_dtoh + sync` fast lane, no extra Vec.
* `Backend::trim_memory_pool` — `CudaBackend` runs
  `cuMemPoolTrimTo(pool, 0)` after stream sync, so retained
  allocator pages come back between OPD steps. Wired into
  `masked_writeback_ce_step` via `ARLE_OPD_TRIM_AFTER_WRITEBACK`
  (default on).
* Fused `causal_sdpa_recompute_backward` CUDA kernel — one thread
  block per row, warp-reduced dot products, three device grad
  allocations total. Legacy chunked device path stays behind
  `ARLE_OPD_LEGACY_SDPA_BWD` as a kill-switch. `head_dim > 256`
  still routes CPU (kernel uses stack-local dq accumulator sized
  for 256 elements at 32-lane stride).
* `matmul_bt` LoRA sites (`.lora_a` / `.lora_b`) — row-tile the
  A × Gᵀ backward at `ARLE_OPD_LORA_LINEAR_BWD_TILE_ROWS` (default
  1024) so the per-tile transient stays bounded on long
  trajectories. Legacy behind `ARLE_OPD_LEGACY_LORA_LINEAR_BWD`.
* MoE grouped-linear LoRA backward — tile active-expert axis at
  `ARLE_OPD_MOE_LORA_BWD_EXPERT_TILE` (default 16). Same
  kill-switch. Previously packed all active experts at once.
* `AdamW::zero_grad` — frees the grad tensor via
  `TensorStore::free` instead of zero-filling, releasing memory
  back to the pool between steps.
* `AutogradError::CudaAllocFailed { op, shape, bytes }` — CUDA
  matmul and transpose alloc sites now report which op / shape /
  bytes tripped OOM instead of a generic "cuda alloc_zeros failed".
* `argmax_last_dim_f32` — tie-break on the largest index (matches
  CPU `max_by` ordering; the previous "smallest index" tie-break
  diverged silently from CPU).

Behavior stays byte-identical on the baseline path unless one of
`ARLE_OPD_LEGACY_*` is set; the tiling / trim / fused-SDPA switches
are all opt-out via env.

## Verification

Local (Mac, no CUDA / Metal):

```bash
CUDARC_CUDA_VERSION=12040 cargo check -p autograd --release --no-default-features --features cuda,no-cuda
CUDARC_CUDA_VERSION=12040 cargo check -p train --release --no-default-features --features cuda,no-cuda
cargo test -p autograd --release
cargo test -p train --release --lib -- --skip sandbox --skip score
```

Results: autograd 28/28 lib + full suite pass (new tests:
`backward_does_not_persist_intermediate_grads`,
`backward_collect_targets_only_drops_unrequested_leaf_grads`,
`backward_accumulate_only_persists_leaf_grads_without_return_map`,
`cuda_causal_sdpa_recompute_backward_matches_cpu`). Train lib 152/152
pass (sandbox/score tests skipped — pre-existing bash-not-found under
`cargo test`, not related). Pre-existing test-only build error on
`crates/train/src/qwen35.rs:6361` (77aac056 added `lora_skip_experts`
arg) fixed in `ad8d6d0e`.

## Pod perf ledger — Qwen3.6-27B-FP8, seq=512, attention-qv LoRA (r=16)

H20 pod (94 GB HBM per GPU, CUDA 12.9, sm_90). Same binary, same
dataset stub, same synthetic 512-token trajectory (256 prompt / 256
response, all masked). Optimized run on GPU 4, legacy run on GPU 5
in parallel — same-binary matched A/B, `--writeback-window 256`.

`/host/Qwen3.6-27B-FP8` — Qwen3.6 hybrid (linear + full attention),
64 layers, head_dim 256, FP8 base, `attn_output_gate=true`. Manifest
built via `arle model index-shards` (walk every `layers-N.safetensors`
into a `model.safetensors.index.json`; upstream loader expects HF
layout).

| phase / metric | optimized | legacy | Δ |
|---|---|---|---|
| base_used_mib (autograd student + AdamW resident) | 33707 | 33707 | 0 |
| post_forward_used_mib | 34415 | 34415 | 0 |
| **post_backward_used_mib (peak)** | **36367** | **36399** | **−32 MiB (−0.09 %)** |
| post_cleanup_used_mib | 36367 | 36399 | −32 |
| allocator_retained_delta_mib | 2660 | 2692 | −32 |
| **post_trim_used_mib** | **34223** | **36399 (no trim)** | **−2176 MiB (−6.0 %)** |
| forward_hidden_states seconds | 61.62 | 61.67 | wash |
| backward seconds | 154.11 | 153.99 | wash |
| **loss** | **3.819344** | **3.819344** | **byte-identical** |
| synthetic-writeback wall (s) | 215.78 | 215.71 | wash |

Env flips (matched A/B, one variable per axis, seq=512 held constant):

* Optimized: `ARLE_OPD_VRAM_TRACE=1 ARLE_OPD_TRIM_AFTER_WRITEBACK=1
  ARLE_OPD_LEGACY_LORA_LINEAR_BWD=0 ARLE_OPD_LEGACY_SDPA_BWD=0`.
* Legacy: same seq, `ARLE_OPD_TRIM_AFTER_WRITEBACK=0
  ARLE_OPD_LEGACY_LORA_LINEAR_BWD=1 ARLE_OPD_LEGACY_SDPA_BWD=1`.

Read: at seq=512 with attention-qv LoRA on a 27B hybrid, the peak
backward gain is small — the live grad set was already bounded by
checkpointing + a shallow trainable tape (attention-qv only). The
big signal is the post-step *residual*: `trim_memory_pool` released
2176 MiB (6 %) of allocator-retained pages back to the pool, so the
next step starts against the base floor rather than the previous
peak. Correctness is confirmed by byte-identical loss between the
two paths (both `--lora_skip_experts` implicitly, no MoE experts).

Attention: LoRA row-tiling did NOT bind at this shape — the
attention-qv LoRA on Qwen3.6 is 5120 × 16 (q) + 5120 × 16 (v), well
below the 1024-row tile threshold. Expected: the row-tile pays on
long trajectories (rows == seq_len at LM head), which the writeback
window bounds at 256 here; a longer window or `all-linear` target
set would surface it.

## Pod 训推一体 (train-infer unified) end-to-end — Qwen3.6-27B-FP8, single GPU

Same GPU 4 hosts BOTH the rollout engine (FP8 27B, generates tokens
for the agent bash-tool loop) AND the autograd student (bf16
materialization + LoRA r=16 attention-qv + AdamW). `--share-frozen-
base` (default) zero-copies the FP8 base pointers from the rollout
engine into the student — ONE copy of the 27B lives on GPU.

Toy SWE-bench-Pro-shaped task (`/tmp/aopd-smoke/staged/toy-1/`,
`foo(x) = x + 0` bug, `fail_to_pass = test_foo.py::test_foo_doubles`
asserting `foo(2) == 4`). One rollout, `--max-turns 6 --max-tokens
256 --rollout-temperature 0.0`.

The agent solved it in 2 turns (read → replace → stop) with a clean
`git diff`; `test_patch` applied, pytest exit 0. That trajectory then
drove ONE real masked-CE writeback step:

```
[agent-opd-debug] toy-1 terminal=Stop turns=2 sub_turns=3
[agent-opd] toy-1 sample 0: passed=true (turns=2) :: [exit 0]
[agent-opd] released inference scratch
[agent-opd] released rollout KV pool
[opd-vram-ledger] masked-writeback
  base_used_mib=33835 post_forward_used_mib=34991
  post_backward_used_mib=37519 post_cleanup_used_mib=37519
  allocator_retained_delta_mib=3684 post_trim_used_mib=34415
[masked-writeback] phase=forward_hidden_states seconds=122.802
[masked-writeback] phase=backward seconds=315.403
[masked-writeback] DONE loss=0.283432 total_targets=122
[agent-opd] round 0: tasks=1 rollouts=1 passed=1 distinct=1
                    trained_pairs=1 mean_loss=0.2834
```

Read (seq_len=1011, 122 masked targets, chunk_rows=512):

| phase | used_mib | Δ vs base |
|---|---|---|
| autograd student resident floor | 35723 | (before rollout release) |
| pre-writeback (after rollout release) | 33835 | 0 (base) |
| post_forward | 34991 | +1156 |
| **post_backward (peak)** | **37519** | **+3684** |
| post_cleanup | 37519 | +3684 |
| **post_trim** | **34415** | **+580 (−3104 MiB released to pool)** |

The `trim_memory_pool` released **3104 MiB (8.3 %) of allocator-
retained pages** — the next round starts from 34.4 GiB, not 37.5 GiB.
On a multi-round agent-OPD loop this is the difference between "peak
grows monotonically until OOM" and "peak decays to base between
rounds".

## Deferred to a follow-up bench

* Longer trajectories (`--synthetic-writeback-seq {2048, 4096}` +
  `--writeback-window 1024`) — where the row-tile in
  `matmul_bt_lora_backward_tiled` first binds.
* MoE grouped-LoRA (`--lora-target-set all-linear` on a Qwen3.5-MoE
  base) — where `grouped_lora_backward_tiled` bounds the active-
  expert axis. The 27B-FP8 is dense; needs `/host/Qwen3.5-122B-A10B`
  with TP=8, or a stubbed MoE small model.
* Fused SDPA backward wall-clock on `head_dim > 128` — Qwen3.6 has
  `head_dim=256` full-attention layers; the fused kernel is on the
  active path here (154 s backward) but not measurably faster than
  legacy chunked — the chunked path also runs on-device for
  `head_dim ≤ 256`, so the kernel-launch amortization is the only
  saving and it disappears in the noise for 4 full-attn layers /
  64 total. Signal would show on a dense long-context student.

## Rule

Backward VRAM = *live grad set*, not just live activations. The
optimizations that pay in a masked-writeback are exactly the ones
that shrink the *live* set: drop grads at last-consumer, tile so a
transient stays bounded, free the grad in `zero_grad`, hand pages
back to the allocator between steps. Byte-identity gates on the
baseline path (kill-switch env vars) keep the shipping bar low —
the new paths are opt-out, not "trust me".

At seq=512 attention-qv LoRA on 27B hybrid the *peak* delta is
noise-floor (32 MiB, 0.1 %) but the *residual* delta is
6 % (2.1 GiB) — the trim is what turns "peak = residual" into
"peak decays back to base between steps", which is exactly the
grinding-OOM failure mode `errors/2026-06-24` was hitting. Peak
gain from row-tiling / MoE-tiling / fused-SDPA-bwd is expected to
bind on longer sequences or MoE — the current bench doesn't
exercise those axes.
