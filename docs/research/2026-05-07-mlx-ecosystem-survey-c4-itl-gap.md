# MLX ecosystem survey — c=4 ITL gap (Task #16) — 2026-05-07

Subagent-driven survey of `ml-explore/mlx`, `ml-explore/mlx-lm`,
`ml-explore/mlx-swift-examples`, `jundot/omlx`, and `ggerganov/llama.cpp`
Metal patterns to identify techniques NOT in ARLE that could close the
c=4 ITL gap (ARLE 19ms vs mlx-lm 7ms).

Stack with [`docs/plans/M_e1-omlx-deep-dive-strategy-correction.md`](../plans/M_e1-omlx-deep-dive-strategy-correction.md).

## Already in ARLE (no chase)

- `BatchKVCache` + left-padding + per-row RoPE offsets — matches
  `mlx_lm.models.cache.BatchKVCache` semantics
  ([`request_state.rs:2428-2570`](../../infer/src/backend/metal/request_state.rs))
- Custom 2-pass batched SDPA (Q-len=16) for verify path —
  [`mlx_bridge.cpp:1024-1115`](../../crates/mlx-sys/src/mlx_bridge.cpp)
- `mx.fast.scaled_dot_product_attention` masked variant for c≥2 decode
  ([`lib.rs:1014`](../../crates/mlx-sys/src/lib.rs))
- `async_eval`/`eval` ping-pong overlap on c=1 path
  ([`mlx_qwen35_model.cpp:3274-3282`](../../crates/mlx-sys/src/mlx_qwen35_model.cpp))
- `mx::compile(..., shapeless=true)` for SiLU/SwiGLU/qk-norm/RoPE
  side-paths ([`mlx_qwen35_model.cpp:314-394`](../../crates/mlx-sys/src/mlx_qwen35_model.cpp))
- Faster two-pass SDPA (MLX #3023, Jan 2026) and fused-attn fp32-scale
  fix (#3119, Feb 2026) — pulled in transitively via MLX upgrade

## Top 3 ROI techniques NOT in ARLE — ranked by likely c=4 ITL impact

### 1. Multi-step async pipelining at the c≥2 packed-batch path

> sample-overlap-with-next-forward, the same trick the c=1 driver uses

**Upstream:** `mlx-lm/mlx_lm/generate.py:2050-2078` (`GenerationBatch._step`).
Pattern: build the **next** forward graph, call
`mx.async_eval(self._next_tokens, ...)`, and only then `mx.eval(inputs,
self._current_logprobs)` on the **previous** step's outputs.

**Why c=4 specifically:** ARLE's c≥2 path at
[`request_state.rs:2529`](../../infer/src/backend/metal/request_state.rs)
issues `async_eval(eval_refs)` and **immediately** does a synchronous
`eval(&[&sampled])` at line 2545 inside the same step. That sync wait
scales with batch (more KV writes + bigger logits), so c=4 sees ~4× the
host-blocking latency of c=1 — exactly the shape of the 19 vs 7 ms gap.
The c=1 driver already does next-step-overlap (line 3274); the c≥2
path does not. Implementable with the existing `async_eval` FFI
([`lib.rs:1095`](../../crates/mlx-sys/src/lib.rs)) — keep last-step
`sampled` resident, build next forward, `async_eval` it, then `eval()`
last-step's `sampled` to extract i32 tokens.

**Compatible with sync `mx.eval()` at step boundary?** Yes — `async_eval`
IS the sync-eval contract. mlx-lm itself uses this exact mlx primitive
on M4. The crash documented by oMLX issues #300/#888 is for
`async_eval` of partial graphs that depend on un-evaluated state; the
mlx-lm pattern only async_evals fully-resolved tensors. ARLE's c=1 path
proves this works on the same hardware.

**Effort:** S–M. Restructure of the in-loop `decode_qwen35_packed_batch`
(or its scheduler caller). Keep flush()-style sync at request
boundaries.

### 2. Per-sequence early-termination compaction inside the c≥2 batch

> avoid wasted decode work when one row stops

**Upstream:** mlx-lm PR #1072 (`mlx_lm/generate.py` BatchGenerator,
2026-04-01). Quote: "Mixed batches (100 + 10,000 tokens) now process
~12,000 tokens instead of ~20,000." Implementation: track per-row
`lengths` in `ArraysCache`, drop completed rows from the live batch
immediately, repack.

**Why c=4 specifically:** at c=4, one row stopping early continues to
consume 25% of every step's compute until natural batch turnover.
ARLE's `decode_qwen35_packed_batch` takes
`&mut [&mut ResumableRequestState]` but does not appear to compact
mid-batch on stop-token hit; see line 2541's monolithic
`batch_cache_len += 1`. Compacting the four arrays — `packed_kv_flat`,
`packed_gdr_flat`, `left_padding`, `rope_offsets` — at row-finish
would directly cut wall-time per step on heterogeneous-length
workloads.

**Effort:** M. Touches `Qwen35PackedDecodeBatch` admit/evict paths.

### 3. PromptTrie-backed prefix cache reuse for incoming requests in a continuous batch

> shared-prefix dedup for chat workloads

**Upstream:** `mlx-lm/mlx_lm/models/cache.py` (PromptTrie + LRUPromptCache,
PR #1019 2026-03-26) and `BatchKVCache.extend()` (PR #1141 2026-04-15).
The `extend()` method is specifically for **continuous batching**: when
a new request joins an in-flight batch, it concatenates a fresh empty
cache with cached-prefix caches by zero-filling along batch dim and
reconciling `lengths`/`left_padding`. Combined with PromptTrie matching
at submission time, repeated system prompts and chat-template prefixes
hit instantly.

**Why c=4 specifically:** c=4 sweeps in production are dominated by
chat-style requests sharing long system prompts. Skipping prefill on
the shared prefix collapses the per-request prefill→first-decode
transition that adds tail latency to the c=4 ITL distribution.

**Effort:** L. Scheduler-layer addition (`infer/src/scheduler/`); also
needs `MetalKVPool.append_request_to_batch` to follow the upstream
`lengths` reconciliation convention.

## Out of scope (for clarity)

- ChunkedKVCache / sliding-window — Qwen3.5 not a sliding-window model.
- NAX (Apple Neural Accelerator) attention path — picked up
  automatically via MLX SDPA on supported hardware tiers.
- mlx-swift-examples — same `BatchKVCache` pattern, no novel tricks.
- llama.cpp Metal — no `flash_attn_varlen` Metal kernel beating MLX SDPA.
- Speculative / DFlash draft glue — ARLE already ahead of upstream.

## Recommended next-step ordering

1. **Wire multi-step async pipelining** into `decode_qwen35_packed_batch`
   (Task #16 candidate fix — profile-validate the 19→<10 ms target
   before committing technique 2).
2. If technique 1 closes only ~50% of the gap, add per-row
   early-termination compaction (technique 2).
3. PromptTrie prefix cache (technique 3) is independent — schedule
   separately as a scheduler-layer plan since its ROI scales with
   workload skew, not single-bench c=4 numbers.

## Phase-timing prerequisite (this commit)

Before adopting technique 1, the per-phase wall-time data needs to
land. This commit adds `INFER_PHASE_TIMING=1` env-gated logging in
`decode_qwen35_packed_batch` that reports five wall-time deltas per
step:

- `prep_us` — host-side input + mask + RoPE-offset MlxArray builds
- `build_graph_us` — `step_batch_packed` graph-build call
- `async_eval_kickoff_us` — async_eval kick (should be near-zero
  since async_eval returns immediately)
- `sample_us` — sample + sync-`eval()` wait — **this is where the
  current implementation pays the host-block tax**
- `pool_dual_write_us` — per-row pool writes + sync `flush()`

The `sample_us` line is the candidate target for technique 1: when
multi-step pipelining lands, the sync wait on `sampled` should overlap
with the next step's forward graph, dropping `sample_us` from
batch-scaling to constant.

## References

- ARLE c=1 async pattern (existing prior art):
  [`mlx_qwen35_model.cpp:3231-3320`](../../crates/mlx-sys/src/mlx_qwen35_model.cpp)
- ARLE c≥2 path under instrumentation:
  [`request_state.rs:2417-2630`](../../infer/src/backend/metal/request_state.rs)
- mlx-lm BatchGenerator: `mlx-lm/mlx_lm/generate.py:2050-2078`
- mlx-lm PromptTrie + extend: `mlx-lm/mlx_lm/models/cache.py`,
  PRs #1019 / #1072 / #1141
- oMLX deep-dive (paged-KV scope correction):
  [`docs/plans/M_e1-omlx-deep-dive-strategy-correction.md`](../plans/M_e1-omlx-deep-dive-strategy-correction.md)
