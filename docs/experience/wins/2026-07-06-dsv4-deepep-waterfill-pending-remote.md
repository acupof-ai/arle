# DSv4 shared-expert static token sharding ("Waterfill", static-mode only) - 2026-07-06

## Goal

- Close part of gap #1 from the DSv4 MoE audit against SGLang's DeepSeek-V4
  reference: the shared expert is computed redundantly on every rank (`normed`
  is TP-replicated, so all `world_size` ranks run the identical dense
  shared-expert FFN over the full `[N]` batch). The reference's Waterfill
  dispatches the shared-expert token to the least-loaded rank as a 9th routed
  expert. This pass ships a narrower, verifiable substitute.

## Hypothesis

- Each rank already knows, before any dispatch, that `normed` at the
  shared-expert call site (`forward_decode_batch_stream_impl`, DSv4 batched
  decode) is bit-identical across ranks. Splitting the `N` tokens into
  `world_size` contiguous, evenly-sized shards (one per rank), running the
  existing single-group shared-expert GEMM on only the owned shard, and
  combining via one `all_reduce_sum` should produce the same result as today
  while cutting shared-expert GEMM FLOPs by `(world_size-1)/world_size`.
- **Scope decision (descoped from literal Waterfill):** the reference's actual
  algorithm is a weighted-random per-token dispatch to the least-loaded rank
  (by real routed-expert load), which would require a new gather/scatter-by-
  arbitrary-token-index CUDA kernel. That kernel cannot be compiled or tested
  on this machine (no CUDA toolchain, no GPU). Rather than hand-write two new
  unverifiable kernel bodies for a correctness-critical MoE combine path, this
  pass ships the **static even-count split**, which is expressible entirely
  with existing, already-proven primitives (contiguous `.slice()`,
  `memcpy_dtod`, `all_reduce_sum` — the same trio the existing DeepEP
  low-latency "owned column range" path already uses). This is NOT
  load-aware and is a materially simpler feature than literal Waterfill;
  see Follow-ups.
- **No throughput or latency improvement is claimed** — this local host has no
  GPU and cannot execute, profile, or correctness-check the kernels.

## Params

- New env flag: `ARLE_DSV4_MOE_WATERFILL=1` (default OFF — falls through
  byte-for-byte to today's full-redundant shared-expert path).
- Gate: `world_size > 1 && seq_len >= 64` (`DSV4_MOE_WATERFILL_MIN_TOKENS`) —
  below that, the extra all-reduce isn't worth it for a handful of decode
  tokens; decode (typically N < 64) falls through unconditionally.
- Only wired into `forward_decode_batch_stream_impl` (the batched, non-CUDA-
  -graph decode lane). Deliberately NOT touched: the single-token CUDA-graph
  decode path (`dsv4_shared_expert_forward_decode_graph`/`_decode_scratch`),
  since graph capture requires fixed shapes and this lever's shard sizes vary
  with `seq_len` — and N=1 is below the MIN_TOKENS gate anyway.

## Environment

- **Backend:** local Rust typecheck + clippy (Mac) **+ pod-verified CUDA
  runtime, serial-only (2026-07-06)**.
- **Model:** `/host/DeepSeek-V4-Flash-FP8` (274 GB FP8).
- **Hardware:** 8×H20 pod (115.190.184.36), **TP=4/world_size=4 on GPUs
  2,3,4,5** (GPU 1 occupied by another tenant at probe time — forced off the
  model's native TP=8/EP=8; `world_size > 1` is still satisfied at 4).
- **Non-default env:** `ARLE_DSV4_MOE_WATERFILL=1`, `INFER_CUDA_DEVICES=2,3,4,5
  INFER_TP_SIZE=4 ARLE_DSV4_MOE_BACKEND=allreduce ARLE_DSV4_INCREMENTAL_KV=1
  --max-total-tokens 8192`.

## Command

```bash
CUDARC_CUDA_VERSION=12080 \
cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib

CUDARC_CUDA_VERSION=12080 \
cargo clippy -p infer-cuda -p cuda-kernels --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
```

GPU verification TODO for a CUDA host:

```bash
# Correctness: needle ladder must be unaffected with the lever ON, multi-rank.
ARLE_DSV4_MOE_WATERFILL=1 CUDA_HOME=/usr/local/cuda \
  scripts/needle_gate.py --model <DSv4 checkpoint> --backend cuda

# Perf: same-binary, same-shell A/B, lever OFF vs ON, decode batch >= 64 so the
# lever actually engages (world_size > 1 required — multi-GPU only).
scripts/bench_guidellm.sh dsv4-moe-waterfill-off ...
ARLE_DSV4_MOE_WATERFILL=1 scripts/bench_guidellm.sh dsv4-moe-waterfill-on ...
```

## Results

- `cargo check` (cuda,no-cuda): clean.
- `cargo clippy -D warnings`: 23 findings before and after (git-stash A/B) —
  diff adds zero new warnings.
- Pod build (`cargo build --release --features cuda,nccl`): BUILD_EXIT=0;
  `ARLE_DSV4_MOE_WATERFILL` symbol present in the binary.
- **Serial correctness gate (n=1, world_size=4), flag OFF vs ON**
  (`lever_gate.sh`, RAW=1 TEMPLATE=dsv4, lengths 115/300/446/2000/8000 ×3):
  ON matches OFF exactly — exact=3/3 every length, DET at every length
  (the OFF run's own len=8000 NONDET is a pre-existing bold/plain-formatting
  floor, not reproduced here) → **PASS on the reachable path**.
- **The lever's own code path was NEVER reached.** `dsv4_moe_waterfill_active`
  additionally gates on `seq_len >= DSV4_MOE_WATERFILL_MIN_TOKENS` (64
  concurrent decode rows). This box's current TP=4/24 GB-free-per-rank config
  supports at most ~5 concurrent decode slots before a pre-existing
  `HostPagedKvPool` capacity bug crashes the coordinator (see Problems) — an
  order of magnitude short of 64. **No A/B of the actual static-split /
  all-reduce logic was possible**; the OFF-vs-ON match above only proves the
  gate correctly no-ops below threshold, which was already guaranteed by
  inspection (`world > 1 && seq_len >= 64`).
- Perf (guidellm): **BLOCKED**, unrelated to concurrency — guidellm 0.6.0's
  synthetic-text generator throws `AttributeError: 'PreTrainedConfig' object
  has no attribute 'max_position_embeddings'` against this checkpoint's custom
  `rope_parameters` HF config, regardless of `--data` spec or concurrency.

**Verdict (superseded by the re-verification below): gate-off-path PASS
(byte-identical, matches baseline envelope); on-path UNVERIFIED.**

## Re-verification after the KV-pool sizing fix (2026-07-06, round 2)

**Root cause, corrected.** Not a `HostPagedKvPool` bug (that struct only
stores whatever `total_pages`/`fixed_pages_per_slot` it's given). The actual
gap was in `Dsv4::kv_budget_plan` (`infer-cuda/src/dsv4.rs`): it planned
`num_slots` from per-slot-STATE affordability alone, never checking whether
the shared FlashMLA pool's "coherent remainder" could actually back that many
whole fixed bands — so the pool ran out of pages well before `num_slots`
concurrent requests arrived, and the `(N+1)`th's `alloc_fixed_band` crashed
the coordinator. Fix (commits `51c31b44f`, `77d60fd4d`): `kv_budget_plan`
additionally computes `pool_affordable_slots = pool_budget_bytes_per_layer /
(flashmla_slot_pages × flashmla_page_bytes)` and clamps `num_slots =
min(num_slots, pool_affordable_slots)`, reusing the function's existing
NCCL-min-reduced clamp pattern; the reduced value flows through the
already-existing `loaded.rs` scheduler-sync unchanged.

**Reachability: FIXED — the MIN_TOKENS=64 gate now fires.** Same TP=4/GPUs
2-5 config, binary @ commit `77d60fd4d`. At `--max-total-tokens 8192` the pool
affords only 1 slot (`pool-band-affordable=1`, logged); at `--max-total-tokens
2048` it affords the full requested 256 (`pool-band-affordable=387`). At 2048,
n=64 concurrent decode (`world_size=4 > 1` and `seq_len=64 ≥
DSV4_MOE_WATERFILL_MIN_TOKENS`, both conditions satisfied) completed with
**zero coordinator crashes**, across repeated trials — previously this
threshold was completely unreachable (crash at n=2).

**Correctness: serial clean; concurrent confounded by a separate, pre-existing
bug (same as the multistream-overlap doc).** Serial (n=1), flag ON:
exact=3/3 at every length (115/300/446/1000), fully DET —
`needle_gate_v2_waterfill.log`. Concurrent n=64, flag ON: two trials,
`exact=22/64` and `exact=30/64`. **Zero-flag baseline, same server
generation, same prompt set:** `exact=25/64` (one trial). All three runs show
the identical failure signature — truncation to a numeric needle-prefix
(`738`, `7382`, …), rarely a single corrupted digit, never new garbage/looping
output. This matches the same pre-existing DSv4 batched (n>1) decode
correctness bug documented in the multistream-overlap doc (reproduces with
zero `ARLE_DSV4_*` flags) — **not attributable to Waterfill**, but it means
the n≥64 envelope isn't clean enough to certify the sharding logic's own
effect on this box today.

**Perf: no clean signal.** guidellm remains incompatible with this
checkpoint. A same-arm back-to-back repeat showed wall-clock drop from ~22-23s
(cold) to ~4-9s (warm) for a 64-way batch — but a **matched zero-flag repeat
showed the identical cold→warm drop** (23s → 6-9s), proving this is a
prefix-cache-reuse artifact of hitting the same prompt set twice against a
long-lived server, not a Waterfill effect. Cold-vs-cold (the only fair
comparison): OFF ~21-23s, ON ~22-23s for the 64-way batch — no distinguishable
delta.

**Verdict: DEFER (revised reason).** The `seq_len >= 64` gate is now reachable
and exercised (previously impossible). Serial correctness is clean. The
concurrent (n≥64) correctness read is confounded by an independent,
pre-existing DSv4 batched-decode bug at n>1 that affects the OFF arm equally —
the ON envelope is statistically indistinguishable from the (already broken)
OFF envelope, i.e. "no new regression detected," not a correctness license for
the static-split/all-reduce logic itself. No perf delta detected (cold-vs-cold
control), and no perf claim is made. Re-verify once the pre-existing n>1
decode bug is fixed, or with enough repeats to resolve the lever's effect
against the baseline's own noise floor.

## Problems

- **The MIN_TOKENS=64 gate was never exercised in round 1.** See Results. This
  box's DSv4 shared FlashMLA pool capacity (`Dsv4::kv_budget_plan`,
  `infer-cuda/src/dsv4.rs` — NOT `HostPagedKvPool`, see the Re-verification
  section below for the corrected root cause) crashed the coordinator on
  concurrency well below 64, independent of this lever (reproduced on plain
  baseline, zero `ARLE_DSV4_*` flags: a 2-concurrent-request burst already
  crashed at `--max-total-tokens 16384`; raising the ceiling to `--max-total-
  tokens 8192` only got to ~5 concurrent slots before the same crash class).
  **Fixed 2026-07-06 round 2** — see below.
- The literal DeepSeek-V4 Waterfill (weighted-random dispatch to the
  least-loaded rank, shared expert as a 9th routed expert through the DeepEP
  dispatch/combine pipeline) was scoped out. It requires new
  gather/scatter-by-arbitrary-token-index CUDA kernels touching the
  count/pack/scatter path (`dsv4_count_local_experts`,
  `dsv4_pack_local_experts_with_slots`, `dsv4_scatter_all_route_slots`), which
  this pass declined to hand-write without any way to compile or test them.

## Delta vs baseline

- **Baseline:** `lever_gate.sh baseline` (flag unset), TP=4/world_size=4, GPUs
  2-5, same binary/shell/prompts as the ON run.
- **Delta, correctness (n=1, off-path):** zero.
- **Delta, correctness/perf (n≥64, on-path):** not measured — unreachable this
  pass.

## Artefacts

- Pod logs: `/root/needle_gate_waterfill.log` (pod-local, gitignored
  bench-output convention, not committed).
- GuideLLM: not produced (guidellm/DSv4-config incompatibility, see Problems).

## Learnings

- ARLE's actual topology for DSv4 decode: attention is TP-replicated (every
  rank holds the identical token batch after attention + allreduce), and only
  the MoE experts are EP-sharded — confirmed by the existing DeepEP
  low-latency branch's own comment ("this rank owns the contiguous token cols
  [start..end) of the **replicated** `normed`"). This makes the static
  even-split substitute correct-by-construction: every token's shared-expert
  contribution is computed by exactly one rank and summed exactly once.

## Follow-ups

- Real (load-weighted) Waterfill: needs a gather-by-mask + scatter-by-index
  kernel pair, authored and tested on a CUDA host, then wired as the 9th
  routed-expert slot into `dsv4_moe_forward_deepep`'s existing
  `topk_idx_i64`/`route_weights` dispatch buffers (the per-token target-rank
  assignment itself is derivable with zero extra communication from the
  already-local `routing.indices`, since expert-id → owning-rank is a pure
  function of `experts_per_rank`).
- Dynamic mode (EP all-reduce for global load) — not attempted this pass.
- Extending the static-split lever to the "intranode normal DeepEP" transport
  branch specifically (today it applies uniformly post-combine regardless of
  transport, adding one extra all-reduce even where the routed-expert combine
  already provided one to piggyback on) — a targeted follow-up once a CUDA
  host can measure whether the extra collective is a net win.
