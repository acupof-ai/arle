# M_e.1 — Wire `MetalKVPool` onto the Qwen3.5 decode hot path

> Empirical motivation:
> [`docs/experience/wins/2026-05-07-bench-guidellm-metal-c-sweep-m4pro.md`](../experience/wins/2026-05-07-bench-guidellm-metal-c-sweep-m4pro.md)
> — at c=4 ARLE-Metal sustains 19 ms ITL median; at c=16 ITL collapses to
> 82 ms and output tok/s **drops** from 158 → 78. Same workload, mlx-lm
> sustains 19 ms ITL at c=16 with 467 tok/s. The 3× output gap is 100%
> in the decode kernel, not the scheduler.
>
> Tier B#1 from the morning gap analysis
> ([`2026-05-07-metal-world-first-gap-analysis.md`](../projects/2026-05-07-metal-world-first-gap-analysis.md))
> and the unification recalibration
> ([`2026-05-07-metal-world-first-recalibration-vs-unification.md`](../projects/2026-05-07-metal-world-first-recalibration-vs-unification.md))
> both pointed at paged-KV; this plan turns it into ordered commits.

## 0. Goal

Replace the contiguous `[B, n_kv_heads, S, head_dim]` per-request cache
under `Qwen35PackedDecodeBatch` (left-padding + additive mask + per-row
RoPE offsets per `infer/src/backend/metal/AGENTS.md` invariant 7) with
token-slot KV writes into `MetalKVPool` plus `gather_kv_rows` at
attention time. This eliminates left-padding overhead at c≥4 and is
the only known route to closing the 3× output-tok/s gap vs mlx-lm on
M-series.

## 1. Substrate already in tree

`infer/src/backend/metal/kv_pool.rs` — fully built but unused on the
hot path:

- `MetalKVPool::new(num_layers, num_kv_heads, head_dim, max_total_tokens, dtype)`
- `alloc_tokens`, `alloc_slots`, `share_slots`, `share_prefix_from`
- `write_kv(layer, request_id, k, v)`, `write_kv_slots(layer, slots, k, v)`
- `gather_kv(layer, request_id) -> (MlxArray, MlxArray)`
- `gather_kv_rows(layer, requests) -> (MlxArray, MlxArray)`
- `release_slots`, `select_eviction_candidates`, `reclaim_target_tokens`

Only call site in production: `runtime.rs:2863` reads `kv_pool_usage()`
for the pressure-metric report (the M2 KvTierAdapter pressure feed).
Zero KV writes / reads pass through the pool today.

## 2. Why mlx-lm wins at c=16

mlx-lm decode does NOT left-pad. Each request has its own KV cache
maintained via slice-append on a per-row buffer; attention compute
reads each request's K/V as a slice. The packed batch is built only
at the SDPA step, with no padding because each query has exactly
`current_seq_len` keys.

ARLE's `Qwen35PackedDecodeBatch` left-pads the entire batch to the
longest in-flight prompt. With variable-length prompts this wastes
compute proportional to length variance. At c=4 the variance fits in
one tick's compute envelope; at c=16 the wasted compute dominates.

## 3. Atomic commit sequence

Each commit lands a bench entry per CLAUDE.md §Benchmarks. The cap
default of 4 stays until commit 5 lands.

### Commit 1 — wire `MetalKVPool::new` into the runtime startup

- Allocate one `MetalKVPool` per active scheduler runtime, sized by
  `MetalSchedulerConfig::max_running_requests * METAL_PREFIX_BLOCK_SIZE
  * <max_seq_len>` (with a sensible cap).
- Plumb dtype from the loaded weights (Qwen3.5 quant currently means
  the K/V activation dtype is BF16 or F16).
- No behavior change. The pool is allocated, the pressure-metric path
  feeds from it, but writes continue through the legacy concat path.
- Effort: **S**. Touches: `metal/runtime.rs`.
- Bench: regression-only — confirm c=4 baseline at ITL 19 ms median.

### Commit 2 — slot allocation lifecycle on prefill

- On prefill admit: allocate slots equal to prompt length via
  `pool.alloc_tokens(request_id, prompt_len)`.
- On request finish: `pool.free_request(request_id)`.
- Read the slot indices from the request state but do NOT yet write
  K/V into them — concat path still owns numerics.
- Effort: **S**. Touches: `metal/request_state.rs`, `metal/runtime.rs`.
- Bench: regression-only — confirm allocation overhead is sub-1 ms
  per prefill (bench c=4 baseline still at ~19 ms ITL).

### Commit 3 — dual-write K/V (concat + pool) under `--metal-kv-pool`

- Behind the existing `--kv-pool` flag (currently a no-op for Qwen3.5):
  on each decode step, write the new K/V vectors to BOTH the legacy
  concat cache AND the pool via `pool.write_kv_slots(layer, slots, k, v)`.
- Attention still reads from concat (legacy correctness).
- Property test: the gathered K/V via `pool.gather_kv` matches the
  concat cache slice for the same request.
- Effort: **M**. Touches: `metal/qwen35.rs`, `metal/request_state.rs`.
- Bench: confirm dual-write does NOT regress ITL beyond 1 ms (bench
  with `--max-running-requests 4 --kv-pool` and without).

### Commit 4 — kernel switches read path to `gather_kv_rows`

- Under `--kv-pool`: SDPA input K/V comes from `gather_kv_rows(layer,
  request_ids)` instead of the concat cache.
- The gather path produces tensors of exact length per request — no
  left-pad. Attention mask and RoPE offsets become per-request scalars
  again (not per-row arrays).
- Concat path stays as the `--kv-pool=false` fallback for one release.
- Effort: **L**. Touches: `metal/qwen35.rs`, `metal/runtime.rs`,
  possibly `crates/mlx-sys/src/mlx_qwen35_model.cpp`.
- Bench: this is the unlock. Expect c=8 ITL ≤ 25 ms (vs current 40
  ms), c=16 ITL ≤ 30 ms (vs current 82 ms), output tok/s at c=16
  ≥ 300 (vs current 78). Acceptance: ≥ 2× output tok/s vs c=4
  baseline at c=16 same workload.

### Commit 5 — flip default `max_running_requests` from 4 to 16

- After commit 4 lands and the bench shows scaling reverses to
  monotonic, change `MetalSchedulerConfig::default().max_running_requests`
  from 4 to 16.
- The new flag (commit landed bbc484c) lets operators tune both
  directions; the default was empirically calibrated at c=4 only
  because of the left-pad collapse.
- Effort: **S**. Touches: `metal/scheduler.rs`.
- Bench: full c-sweep + matched-A/B vs mlx-lm at c=16; expect parity
  on output tok/s, ARLE wins on ITL p95 stability per the morning's
  evidence.

### Commit 6 — retire the concat path

- Once commit 5 has shipped one bench window without rollback, drop
  the legacy concat KV cache code path from `Qwen35PackedDecodeBatch`
  entirely. Reduces hot-path branches and `request_state.rs` size.
- Effort: **S**. Touches: `metal/request_state.rs`, `metal/qwen35.rs`,
  `metal/runtime.rs`.
- Bench: no expected delta; pure deletion-style refactor per
  `feedback_no_half_states.md` ("finish a refactor unit fully").

## 4. Acceptance gates (whole plan)

- `cargo test -p infer --lib` continues at 556+ passing post-each-
  commit (no scheduler regressions).
- `--fast` bench `metal-m-paged-kv-c16` after commit 4 vs the recorded
  c-sweep baseline:
  - output tok/s c=16 ≥ 300 (was 78)
  - ITL p95 c=16 ≤ 35 ms (was 84 ms)
  - peer with mlx-lm c=16 within ±10% on output tok/s
- `cargo check -p infer --no-default-features --features cuda,no-cuda`
  remains green throughout (CUDA-Rust drift gate; no CUDA hot-path
  changes are introduced by this plan).
- ELI Layer-1 smoke (curl /v1/chat/completions with tool_choice +
  response_format) returns HTTP 200 with valid completion after each
  commit; the request shape contract is unchanged.

## 5. Out of scope

- **Full vLLM-style block-table-indirect-attention.** The token-slot
  pool stops short of letting the attention kernel itself walk a
  block table; it materializes contiguous K/V via `gather_kv_rows`
  before SDPA. This is enough to close the left-pad gap; true
  block-table-aware attention is M_e.2.
- **KV quantization.** Q8 / FP8 KV is Tier A#2 and lives on the
  shared `infer/src/ops/kv_ops.rs` per the M4 unification frame
  ([`recalibration doc`](../projects/2026-05-07-metal-world-first-
  recalibration-vs-unification.md)). It composes orthogonally on top
  of this plan.
- **Multi-LoRA.** Punica/S-LoRA on Metal is Tier D frontier work, no
  upstream baseline to compete with — separate plan.
- **CUDA path.** Out of scope per user directive ("metal 自主迭代",
  "不影响 cuda 的性能"). CUDA already has paged-KV via
  `cuda_kernels::PagedKVPool`; this plan is the Metal sibling, not a
  unification.

## 6. References

- Bench evidence:
  [`2026-05-07-bench-guidellm-metal-c-sweep-m4pro.md`](../experience/wins/2026-05-07-bench-guidellm-metal-c-sweep-m4pro.md)
- Tier ranking source:
  [`2026-05-07-metal-world-first-gap-analysis.md`](../projects/2026-05-07-metal-world-first-gap-analysis.md)
- Unification frame for any cross-backend op work:
  [`2026-05-07-metal-world-first-recalibration-vs-unification.md`](../projects/2026-05-07-metal-world-first-recalibration-vs-unification.md)
- Substrate to wire:
  `infer/src/backend/metal/kv_pool.rs` (fully built, unused on hot path)
- Hot-path invariants:
  [`infer/src/backend/metal/AGENTS.md`](../../infer/src/backend/metal/AGENTS.md)
  invariants 4 + 7 (variable-length packed decode contract that this
  plan supersedes for `--kv-pool`).
- Bench protocol:
  [`docs/plans/M6-metal-world-rank-snapshot.md`](M6-metal-world-rank-snapshot.md)
  (this plan's commit 4+5 acceptance run)
- Bench-and-trace spec:
  [`docs/bench-and-trace-spec.md`](../bench-and-trace-spec.md)
