# M_ttft-p99 Phase 0 — H1 Sequential prefill HOL admission CONFIRMED via code-grep

> Per `a25416b` plan(M_ttft-p99 D4)Phase 0 reconnaissance(0.5d Claude)。
> H1 sequential prefill HOL admission was strongest hypothesis by signal:
> 6.2× p50/p99 spread at c=8 matched 7-session sequential pattern。
> **Code-grep CONFIRMS H1**:Qwen3 model hard-caps `max_concurrent_prefill_requests = Some(4)`。
> But H1 alone doesn't fully explain 6.2× — needs additional factor。

## H1 mechanism via direct code-grep

### Cap at model level
`infer/src/model/qwen3/forward.rs:310-320`:
```rust
fn max_concurrent_prefill_requests(&self) -> Option<usize> {
    if self.uses_marlin_prefill_gemm() {
        // Marlin prefill GEMM converts BF16 activations to FP16 and
        // allocates a FP16 output scratch per projection. A 16-slot burst
        // can otherwise fit the token budget but still OOM the temporary
        // GEMM scratch, which used to panic the scheduler thread.
        Some(4)
    } else {
        None
    }
}
```

This was introduced by `b708e00` admission fix to avoid Marlin scratch
OOM(W3 c=16 deadlock fix)。Per-step prefill admission capped at 4。

### Cap propagation to PrefillBudget
`infer/src/scheduler/cuda/execution.rs:174-183`:
```rust
token_budget: StepTokenBudget::for_prefill(
    scheduler.config.max_num_batched_tokens,
    scheduler.config.max_prefill_tokens,
    decode_slots.len(),
    scheduler.config.prefill_max_requests
        .unwrap_or(usize::MAX)
        .min(scheduler.model.max_concurrent_prefill_requests().unwrap_or(usize::MAX)),
),
```

So the effective cap = `min(config.prefill_max_requests, model.max_concurrent_prefill_requests)`。
- Config default(`types.rs:200`):`prefill_max_requests: None` = `usize::MAX`
- Model:`Some(4)` for Marlin path
- → Effective:**4 prefills per scheduler step** for Marlin Linear paths

### Cap config CLI flag
`infer/src/main.rs:111`:`prefill_max_requests: Option<usize>` is CLI exposed
but not user-set in current bench → defaults to None → falls through to
model cap = 4。

## c=8 burst staircase math

For c=8 W4 8k longctx workload:
- Per-prefill latency estimate(per `c4fae17` c=4 8k):4079 ms
- 8 sessions ÷ 4 per step = **2 step cycles** to fully admit
- Session 1-4:enter prefill at t=0,finish ~4079 ms
- Session 5-8:wait until step 1 done(t≈4079 ms),enter prefill at t=4079,finish ~8158 ms

**Predicted TTFT spread**:p50 ≈ midpoint ≈ 6118 ms,p99 ≈ session 8 ≈ 8158 ms = **1.33× p99/p50**

Observed `f5cf829`:**11768 ms p50,72515 ms p99 = 6.2× spread**

→ **H1 alone doesn't explain 6.2×**。Multiple admissions cycle expected
but not >6× spread。Other factors present。

## Additional factors hypothesis

H1' (sub-hypothesis):eviction/preemption of in-flight prefills causes
some sessions to wait multiple step cycles。

H1'' (sub-hypothesis):page_budget limit at c=8 forces some prefills
to be deferred even when prefill_max_requests=4 not hit。

H1''' (sub-hypothesis):session prefill chunk size auto-clamps low at
high concurrency,causing per-prefill latency to grow non-linearly。

H4 sub-hypothesis(per `a25416b` plan):CUDA Graph not pre-captured for
batch=5-8 — first encounter of each batch size triggers graph capture
overhead = ~100-500 ms tax on tail sessions。

## Phase 0 conclusion

**H1 partially confirmed**:
- Cap-of-4 mechanism is real(model.rs:316)and would create 2-cycle
  admission for c=8 → ~1.33× spread。
- This explains the FLOOR of TTFT increase from c=4 → c=8 but not the p99 tail。

**Still unexplained**:additional 4-5× spread suggests preemption /
chunking / graph-warmup interaction。

## Next steps(per plan §Phase 0 → Phase 1)

### Phase 0.5 — Targeted /v1/stats trace at c=8
- [ ] Run W4 c=8 with logging:`step_phase_us{prefill,decode}`,
      `prefill_queue` length over time,`active` slot count
- [ ] Trace specific session 6/7/8 timing:
  - Step admission time
  - Prefill chunk count and size
  - Decode start time
  - First token emitted time
- [ ] Identify if any session sees > 1 admission attempt(eviction signal)

### Phase 0.6 — `select_launch_prefill_candidates` deeper read
- [ ] `infer/src/scheduler/cuda/execution.rs:345 select_launch_prefill_candidates` —
      verify only 4 admitted per call
- [ ] Check if subsequent sessions are pushed back to `prefill_queue` or
      dropped+re-admitted(latter would cause cascade)

### Phase 1 — Test prefill_max_requests=8 override
- [ ] Single-variable A/B:run W4 c=8 bench with `--prefill-max-requests 8`
      override(if Marlin scratch OOM doesn't trigger)
- [ ] Compare TTFT p50/p99 vs `f5cf829` baseline
- [ ] If reduces p99 → H1 confirmed,investigate raising cap or improving
      Marlin scratch reuse
- [ ] If no change → kill H1,move to H2-H5

## Cross-references

- Plan: [`a25416b`](../plans/M_ttft-p99-tail-latency.md)
- Source-of-cap: `b708e00` admission fix
- Empirical signal: `f5cf829` W4 c=8 bench
- Cap location: `infer/src/model/qwen3/forward.rs:310-320`
- Budget propagation: `infer/src/scheduler/cuda/execution.rs:174-183`
- Config default: `infer/src/scheduler/types.rs:200`
- 3-shape grid: `c4fae17`

## Status

H1 mechanism CONFIRMED but not sole explanation of 6.2× p99/p50 spread。
Phase 0.5 trace + 0.6 read + Phase 1 A/B needed for completion。

Codex action(or Claude continue):0.5h Phase 0.5 trace setup,then
Phase 1 A/B(needs GPU)。Estimate 1-2h to converge on root cause(or
identify multi-factor combination)。

## Rule

When tail-latency spread(p99/p50)exceeds simple staircase math
(e.g. cap=N admission predicts log_N(c)× spread but observed is much
higher),investigate **secondary admission paths**:
- Preemption + re-admission cascade
- Page budget contention with decode growth
- Kernel warmup tax on first-encounter batch sizes
- Stream blocking between prefill and decode phases

These factors compound multiplicatively with the primary cap,creating
spreads larger than naive sequential analysis predicts。
