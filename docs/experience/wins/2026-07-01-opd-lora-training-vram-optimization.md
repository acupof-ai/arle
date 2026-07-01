# OPD LoRA writeback VRAM optimization — pending-remote perf ledger

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

Pod (H20 pod-tree `/host/arle-build`, GPU 4, build tag `arle`):

```bash
scripts/pod.sh build
# → BUILD_EXIT=0 (~56s incremental over prior sm_90 cache)
LABEL=opd-smoke GPU=4 arle train opd --smoke --steps 5 \
  --prompt-ids 1,2,3,4,5 --rollout-len 16 \
  ARLE_OPD_VRAM_TRACE=1 ARLE_OPD_TRIM_AFTER_WRITEBACK=1
# → RUN_EXIT=0 in <1s (tiny embedded Qwen3.5 config, vocab=16,
#   hidden=16, layers=2, backend=cuda:0)
```

Confirms the OPD hot loop (tape backward + `AdamW::zero_grad` free
path + `trim_memory_pool`) runs green on CUDA under the new code.

## Perf ledger — pending-remote

The masked-writeback / agent-OPD LoRA-scale VRAM bench that would
land the peak-VRAM Δ is deferred:

* The H20 pod only ships two Qwen3.5-format bases —
  `/host/Qwen3.5-122B-A10B` (too big for a single-GPU LoRA smoke) and
  `/host/Qwen3.6-27B-FP8` (per-layer `layers-N.safetensors` weight
  split; the current `discover_shards` only reads HF standard
  `model.safetensors[.index.json]` layouts).
* Patching `/host/Qwen3-4B` (vanilla Qwen3 flat schema) into a
  Qwen3.5-loader-compatible directory only satisfies serde — the
  `Qwen35Config::validate` check fails
  ("linear-attention heads and dims must be non-zero") because
  Qwen3-4B has no `linear_attention` layers to populate those
  fields with, and forcing plausible non-zero values is a lie the
  loader would then use.

To land the perf ledger:

1. Stage a small Qwen3.5-format base (dense or MoE) in HF layout —
   e.g. `Qwen/Qwen3-5-4B` on ModelScope once mirrored, or
   round-trip `Qwen3.6-27B-FP8` through
   `arle model export --format hf` — under `/host/`.
2. `arle train agent-opd --student-model <dir> --dataset
   /tmp/lora_smoke_ds.jsonl --staged-root /tmp --task-limit 1
   --synthetic-writeback-seq {512, 1024, 2048}
   --lora-target-set attention-qv --lora-rank 16 --lora-alpha 32
   --lora-skip-experts` (MoE only) — matched A/B optimized vs
   `ARLE_OPD_LEGACY_LORA_LINEAR_BWD=1 ARLE_OPD_LEGACY_SDPA_BWD=1
   ARLE_OPD_TRIM_AFTER_WRITEBACK=0`, ledger from
   `[opd-vram-ledger] masked-writeback base_used_mib=…
    post_forward_used_mib=… post_backward_used_mib=…
    post_cleanup_used_mib=… allocator_retained_delta_mib=…
    post_trim_used_mib=…`.

The ledger tags to fill in when the bench lands:
`peak_used_mib_opt`, `peak_used_mib_legacy`, `Δ`,
`allocator_retained_delta_mib_opt`,
`allocator_retained_delta_mib_legacy`.

## Rule

Backward VRAM = *live grad set*, not just live activations. The
optimizations that pay in a masked-writeback are exactly the ones
that shrink the *live* set: drop grads at last-consumer, tile so a
transient stays bounded, free the grad in `zero_grad`, hand pages
back to the allocator between steps. Byte-identity gates on the
baseline path (kill-switch env vars) keep the shipping bar low —
the new paths are opt-out, not "trust me".
