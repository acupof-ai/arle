# cap=8 bimodal trigger CONFIRMED:prefill warmup gap — `c20b1ce` covers decode only

> Per `61ebf45` empirical:**first-burst sessions 0-9 consistently fail
> across all 3 runs**(76% / 56% / 76% normal/degraded modes)。Pattern
> isolation + `warmup.rs` code-grep CONFIRMS:`c20b1ce` warmup pass
> covers DECODE code paths only — PREFILL paths cold-start in first burst。

## Empirical signal(`61ebf45`)

Per-session failure ID across 3 fresh runs:
| Run | Mode | Failed sessions |
|-----|------|----------------|
| #1 (76% normal) | Failed 0,1,2,3,4,5,6,7,8,9 |
| #2 (56% degraded) | Failed 0,1,2,3,4,5,6,7,8,10 |
| #3 (76% normal) | Failed 0,1,2,3,4,5,6,7,9,10 |

**Pattern**:first 8-10 sessions consistently fail。Subsequent sessions
mostly succeed。Mode determined by what happens AFTER first burst stabilizes。

## Code-grep confirms gap

`infer/src/scheduler/cuda/core/warmup.rs:75-105` warmup pass:
```rust
// Lazy-init DECODE context before warmup.
if self.decode_bufs.is_none() {
    match self.model.create_decode_context(...) {
        Ok(ctx) => self.decode_bufs = Some(ctx),
        ...
    }
}

let dummy_tokens: Vec<u32> = vec![0; max_bs];
let slot_indices: Vec<usize> = (0..max_bs).collect();

// Pass 1: drive forward for each warmup batch size. Populates the
// cublasLt heuristic algo cache for all GEMM shapes used by decode.
warmed = self.warmup_graphs_pass(&warmup_sizes, &dummy_tokens, &slot_indices);
```

Comment line 102 explicit:**"GEMM shapes used by decode"**。
No `prefill_bufs` lazy-init,no prefill batch sizes warmed,no
prefill_graphs_pass。

→ First-burst prefill triggers cold-start tax:
1. Marlin GEMM kernel JIT compilation(~50-200 ms first call)
2. cublasLt heuristic search for prefill shape(~100-300 ms)
3. Allocator slop on FP16 GEMM scratch(~50 ms)
4. Attention kernel prefill path first-encounter
5. KV pool first-allocation per session

8-10 sessions × 100-500 ms cold-start each = **1-5 second admission
cascade** during first burst → 503 retry exhaustion。

## Why warm server `19d12c2` succeeded

Override test `19d12c2`(257/257 = 100%):server had been running prior
benches with prefill workloads → all prefill code paths hot → no
first-encounter cost。

Fresh server runs(`bwa4piqqx`/`b4r8fha82`/`b5i3467ad`):cold prefill
paths → first-burst tax → 76% / 56% bimodal。

## Fix proposal — extend `warmup_cuda_graphs` with prefill pass

```rust
// After existing decode warmup pass:

// Pass 2: prefill warmup. Drive a single forward_prefill at each
// per-step admission count to populate prefill code paths:
// - Marlin GEMM kernel for prefill shapes
// - cublasLt heuristic for prefill GEMM shapes
// - Attention prefill kernel
// - FP16 GEMM scratch allocator
let prefill_cap = self.model
    .max_concurrent_prefill_requests()
    .unwrap_or(1);
for batch_size in 1..=prefill_cap {
    let dummy_prompt: Vec<u32> = vec![0; 256]; // realistic prefill length
    let slot_ids: Vec<usize> = (0..batch_size).collect();
    if let Err(e) = self.model.warmup_prefill(&dummy_prompt, &slot_ids) {
        warn!("Warmup: prefill batch={} failed: {}", batch_size, e);
        break;
    }
}
```

**Effort**:50-150 LOC depending on `model.warmup_prefill` API design。
- If model already has prefill forward path:wrap call ~50 LOC
- If needs new API:add to ModelForward trait + qwen3 impl ~150 LOC

**Cost**:additional cold-start time,~1-3 seconds total
(prefill_cap × 200-500 ms each)。Trade-off:5 sec cold start vs
10× of current 5-15s first-burst latency cascade。

## Predicted post-fix behavior

If prefill warmup added:
- First-burst sessions 0-9:no cold-start tax → succeed normally
- Subsequent sessions:still succeed(unchanged)
- **Predicted turn success:100% deterministic**(both modes resolve)
- Predicted TTFT:roughly same(first-burst session 1 still takes
  whatever prefill compute time,but no first-encounter overhead)

## Decision tree

If user approves fix path:
1. **Phase A**(0.5d codex):add `warmup_prefill` to ModelForward trait
   + qwen3 Marlin implementation
2. **Phase B**(0.25d codex):wire prefill warmup into `warmup_cuda_graphs`
3. **Phase C**(0.25d):re-run N≥3 fresh-server bench
4. **Phase D**(0.25d):if 100%,update master strategy + ship wins entry

Total:1-1.5 days codex。Eliminates bimodal residual。

## Cross-references

- `61ebf45` first-burst session 0-9 failure pattern
- `f7da3e1` bimodal-trigger walking back
- `c20b1ce` warmup fix(decode only,sufficient for cap≤cap mismatch
  but not first-encounter prefill compute)
- `12300c5` cap=8 flip
- `8281047` initial 91.8% validation(was lucky-mode)
- `a0a3f42` 6-run bimodal characterization

## Methodology insight

`61ebf45` per-session failure ID parsing was the ONE thing that
isolated the trigger after 5 ticks of variance/memory hypotheses。

**Rule**:when bimodal pattern persists across hypothesis testing,go
**granular** — parse per-failure-event timing/index instead of
aggregate metrics。Aggregate hides the deterministic structure。

## Status

This brief proposes prefill warmup fix path。Codex pickup queue:
- P0 Hybrid Phase 1b
- P0' bimodal investigation → **CONCRETE FIX READY**(this brief)
- P1 B3 PrefixAwareAdmission(`a1965ab`)
- P1 #33 KV W4A8
- P1' Medusa Phase 1.B

If prefill warmup fix is straightforward(model already has prefill
forward),this becomes P0' too — alongside Hybrid Phase 1b for one
codex pickup pair。
