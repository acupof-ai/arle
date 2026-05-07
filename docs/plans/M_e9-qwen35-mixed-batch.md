# M_e.9 — Generalize `try_mixed_batch` to Qwen3.5/3.6 (A3B MoE)

**Owner:** ckl · **Status:** designed 2026-05-07 (subagent), awaiting impl tick
**Track:** Metal scheduler · **Predecessor:** M_e.4 SwiGLU compile-fusion,
  oMLX-C v3 host pipelining

## Goal

ARLE's `try_mixed_batch` ([`infer/src/backend/metal/request_state.rs:1321-1399`](../../infer/src/backend/metal/request_state.rs))
fuses decode and prefill into **one** `mx::async_eval` graph for
Qwen3 — but returns `Ok(None)` for Qwen3.5/3.6, forcing two
sequential graphs per tick on the canonical Metal model. Per the
2026-05-07 encoder-bound diagnosis (95% of step time = MLX encoder
work for ~600-1000 primitives), **collapsing two graphs into one
should yield roughly 1.7× speedup on mixed prefill+decode workloads.**

This is the **largest single ROI scheduler-side lever** still on the
table per the parallel research subagent (this date).

## Background — why Qwen3.5 falls back today

When the scheduler tick has both prefill and decode work:

- **Qwen3 path** (`request_state.rs:1381`): `try_mixed_batch` calls
  `execute_qwen3_packed_batch` which composes both into one variable-
  length forward. ONE async_eval per tick.
- **Qwen3.5/3.6 path** (`request_state.rs:1344`): `try_mixed_batch`
  matches `MetalRequestStateInner::Qwen35 => return Ok(None)` —
  hard-coded fallback. The dispatcher at `runtime.rs:1514-1548`
  detects this `None` and falls through to `guard_decode_batch` →
  `guard_prefill_chunk` sequentially. **TWO async_eval per tick.**

The capability *exists* — Qwen3.5 has variable-length packed decode
([`backend/metal/AGENTS.md`](../../infer/src/backend/metal/AGENTS.md) §7:
left-padding + additive mask + per-row RoPE offsets). The C++ side
already handles arbitrary-length rows in `step_batch_packed`. The
gap is purely Rust-side wiring: build a row vector that includes
both decode rows (1 token each) AND prefill rows (N tokens each)
then dispatch through the same `step_batch_packed` entry.

## Design

### Entry point change

`request_state.rs:1321-1399` (`try_mixed_batch`):

```rust
match &mut state_ref.inner {
    MetalRequestStateInner::Qwen3(_) => { /* existing code */ }
    MetalRequestStateInner::Qwen35(qwen35) => {
        // M_e.9: route through generalized variable-length packed
        // batch that accepts mixed (decode_row + prefill_row) input.
        try_mixed_batch_qwen35(states, batch, scheduler_cfg)
    }
}
```

### `try_mixed_batch_qwen35` shape

Mirror `try_decode_qwen35_packed_batch` ([`request_state.rs:1465`](../../infer/src/backend/metal/request_state.rs))
but with per-row token counts allowed to be > 1:

1. For each scheduled row, build `Qwen35PackedDecodeBatch` row entry
   - decode rows: `[last_token]` (length 1)
   - prefill rows: `[next_chunk_tokens]` (length 1..max_chunk_budget)
2. Compute `left_padding[row]` to align all rows to the longest
   row in the batch.
3. Build the additive attention mask + RoPE offsets that handle
   mixed-length sequences (partially exists for varlen decode;
   needs extension for new query positions in prefill rows).
4. Single C++ FFI call: `step_batch_packed` with combined input.
5. Per-row output split: prefill rows commit their chunk and check
   for emit-token; decode rows commit their next sample.

### Mask construction

`build_varlen_decode_mask` ([`infer/src/backend/metal/mlx.rs:854`](../../infer/src/backend/metal/mlx.rs))
already handles per-row left-padding. Need an extended variant for
mixed batch that also masks **future query positions within each
prefill row** (causal within that row's new positions). The
`build_varlen_verify_mask` at the same file demonstrates this
pattern for DFlash; can be adapted.

### KV cache writes

Each row appends N tokens to its own packed_kv_flat slab; the
existing `extend_kv_cache` machinery handles capacity. The
prefill rows write N columns; decode rows write 1. The C++
forward already sets row-aware `cache_pos_arr` and per-row RoPE
offsets — no new C++ work, just Rust-side row aggregation.

## Implementation steps

1. **Read precondition gate**: per `feedback_path_probe_before_perf_claim.md`,
   first measure how often the Qwen3.5 path actually has mixed
   decode+prefill in a tick. Add tracing at `runtime.rs:1514-1548`
   counting `metal_mixed_batch_qwen35_eligible_total` /
   `metal_mixed_batch_qwen35_fallback_total`. If <30% of ticks
   would benefit, deprioritize.
2. **Extend `Qwen35PackedDecodeBatch`** ([`request_state.rs:773`](../../infer/src/backend/metal/request_state.rs))
   to track per-row `prefill_chunk_len: Vec<i32>` (currently all
   rows are 1 implicit; this is required for prefill rows >1).
3. **Extend mask construction** in `mlx.rs` — new function
   `build_varlen_mixed_mask(left_padding, per_row_query_lens,
   batch_cache_len)` that combines the varlen decode mask with the
   prefill causal-block mask.
4. **Implement `try_mixed_batch_qwen35`** following the design
   above.
5. **Wire the dispatcher** at `runtime.rs:1514-1548` to actually
   call the new entry instead of falling through to sequential
   paths.
6. **Path probe** at the new entry: `M_E9_MIXED_BATCH_PROBE` →
   `metal_path_probe: Qwen3.5 mixed batch FIRED`.
7. **Bench**: longctx workload at c=4 with new requests joining
   mid-decode. Compare against M_e.4-baseline:
   - Mixed-fallback rate → mixed-batch rate (should be near 100%
     when prefill+decode coexist)
   - p50 ITL over the bench window → expect ~30-40% reduction
     on the ticks that previously fell back

## Composability

| Lever | Composable | Notes |
|-------|-----------|-------|
| oMLX-C v3 (host pipelining) | Yes | One async_eval per tick is exactly what oMLX-C v3 wants — preserves the prev_sampled handoff. |
| auto-wired-limit | Yes | Memory pinning unaffected. |
| M_e.4 SwiGLU compile-fusion | Yes | Compile-fused kernel applies to mixed graph the same way. |
| INFER_MOE_TOP_K=N | Yes | Both rows go through the same MoE block; top_k applies to all. |
| oMLX-A pool flush | Yes | Pool dual-write fires per row regardless of mixed. |

## Risks

| ID | Risk | Mitigation |
|----|------|------------|
| R1 | The fall-through path is ALSO a correctness fallback for cases where the variable-length packed batch can't handle mixed shapes (e.g. attention mask shape mismatch) | The pre-condition counter (step 1) tells us when the simple cases dominate; gate the new path behind eligibility checks (all rows have valid left_padding ≤ batch_cache_len, etc.) and fall back on misalignment. |
| R2 | Prefill row in mixed batch causes per-row RoPE offset / mask construction to grow O(max_prefill_chunk × batch_size) | Cap prefill chunk per row at a small budget (e.g. 256) when mixed; let pure-prefill ticks process larger chunks. The scheduler config `max_batch_tokens=512` already implicitly bounds this. |
| R3 | KV cache slab capacity overflow when a prefill row appends N>>1 tokens — `extend_kv_cache` allocates new MlxArrays mid-tick | `ensure_capacity_for_states` already handles this for decode rows; just generalize to take per-row needed_tokens. |
| R4 | DFlash speculative path (`try_decode_qwen35_dflash_speculative_batch`) has its own batch-build that conflicts with mixed | DFlash path is mutually exclusive with mixed (gated separately at `runtime.rs:2293`). v1 of M_e.9 keeps that exclusion; v2 could integrate. |

## Acceptance bench

`scripts/bench_guidellm.sh qwen36-mixed-batch-c4-longctx --workload longctx-32k`:

- Pre-condition (step 1 instrumentation): `metal_mixed_batch_qwen35_fallback_total /
  total_qwen35_ticks ≥ 0.30` on the bench (otherwise this isn't on the hot path
  for this workload).
- Win predicate: `c=4` p50 ITL drops by ≥15% on the ticks that exhibit
  mixed prefill+decode. Aggregate ITL p50 may drop less (since pure-
  decode ticks are unaffected).
- Path probe `M_E9_MIXED_BATCH_PROBE` fires.
- Matched A/B per `feedback_matched_ab_for_small_bench_effects.md`.

## Companion: KV-aware admission (S effort, free)

The same research dispatch surfaced a low-cost companion. Today
`MetalScheduler::admit_next_waiting_request`
([`scheduler.rs:487-500`](../../infer/src/backend/metal/scheduler.rs))
admits one request iff `running_len() < max_running_requests`. No
KV check. When KV runs short mid-decode, the request gets aborted.
A simple S-effort fix: refuse admission if free KV < estimated peak
KV need for the prompt. Converts hard failures into queueing.

File: `scheduler.rs:487-500`, gate on `kv_pool.free_blocks() >=
estimated_kv_blocks(prompt_len + max_new_tokens)`. Reuses existing
`MetalQwen35PrefixRuntime` knowledge of `cached_tokens` /
`max_cached_tokens` (`runtime.rs:441-444`). Land alongside or before
M_e.9 to remove the cancel-mid-decode footgun.

## References

- Predecessor (the encoder bottleneck this attacks):
  [`docs/experience/wins/2026-05-07-bench-qwen36-encode-bottleneck.md`](../experience/wins/2026-05-07-bench-qwen36-encode-bottleneck.md)
- Existing Qwen3 mixed-batch implementation:
  [`infer/src/backend/metal/runtime.rs:1655-1798`](../../infer/src/backend/metal/runtime.rs)
- Existing Qwen3.5 packed-batch entry to mirror:
  [`infer/src/backend/metal/request_state.rs:1465`](../../infer/src/backend/metal/request_state.rs)
- vLLM v1 unified-step pattern (research subagent quoted):
  `vllm/v1/core/sched/scheduler.py::Scheduler.schedule()`
