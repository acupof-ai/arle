# 2026-05-07 · F4-Small inadvertently disabled Mixed path at long-ctx steady state

## Discovery (parallel work during codex M3.9 Phase 0)

While codex implements M3.9 Phase 0 instrumentation, source-survey
of `plan_step` (execution.rs:393-414) revealed:

```rust
} else if self.deferred_decode_emit.is_none()
    && self.model.supports_mixed_batch(self.paged_kv_pool.format)
{
    StepPlan::Mixed(candidates)
} else if /* short_prompt_bypass */ {
    StepPlan::Prefill(candidates)
} else {
    // Keep the legacy split launches for models that do not have a
    // real single-launch mixed lowering yet.
    StepPlan::Split(candidates)
}
```

**`git log -L 396,400:execution.rs`** confirms the
`deferred_decode_emit.is_none() &&` was added by F4-Small commit
`2a534c4`. Before F4-Small the condition was just
`supports_mixed_batch(...)`.

## Why this matters

F4-Small's async readback architecture means
`deferred_decode_emit = Some(...)` is COMMON in steady state — the
copy-stream event hasn't fired yet, so the decode emit is deferred
to next tick. Under that regime:

- Mixed path is **disabled** (the F4-Small-added precondition fails)
- Plan falls through to **Split** path
- Split has the **10× per-token tax** measured in
  [`4a3612b`](2026-05-07-split-plan-10x-tax-confirmed.md)

This is the SOURCE of the 10× tax. F4-Small fixed decode-axis
sync but inadvertently routed long-ctx mixed workloads to the
slow Split path.

## Why F4-Small added the precondition

Codex was protecting against a potential race: with deferred decode
emit outstanding (older launch's D2H copy in flight), launching a
new Mixed batch would write to the same `argmax_out` GPU buffer.
If the new launch's argmax overwrites `argmax_out` before the
deferred's D2H copy completes, deferred reads the new tokens for
the old request — **silent corruption**.

The Split path apparently doesn't trigger this protection (or
F4-Small didn't apply it consistently), perhaps because Split
workloads don't typically overlap with deferred reads in the same
way.

## Fix design (verification needed)

Two paths to restore Mixed at long-ctx:

### A) Double-buffer `argmax_out`
Add a 2-deep ring of GPU `argmax_out` buffers. Each launch writes
to a different buffer slot. Deferred and new readbacks reference
their own buffers. No race. ~30-50 LOC in `BatchDecodeBuffers` +
`sample_batch_greedy_launch/readback`.

### B) Drain deferred BEFORE new launch (kills async benefit)
At launch time, sync-wait for deferred's event then process. This
re-introduces the per-tick sync F4-Small fixed → would regress
high-conc by ~50%. **Not viable**.

### C) Validate the race is REAL
Race may not actually fire — the D2H copy is small (B×4 bytes for
argmax + B×4 bytes for logprobs). Copy-stream completion may
always finish before the next compute-stream argmax overwrites
the source buffer (compute kernel takes ~10s of µs; copy completes
in similar time).

If race doesn't fire in practice → simply remove the
`deferred_decode_emit.is_none() &&` precondition. 1-line fix.

Test: greedy_consistency + e2e at long-ctx mixed workload (NOT
the current standard high-conc which never hits Split).

## Recommended next steps

This finding **supersedes** M3.9 Phase 1B (the "scheduler policy
delay split") — the split-path was never the right design, it was
just the fall-through when Mixed was blocked.

**M3.9 Phase 1A v2 (revised)**:

1. Verify race is real or not. Read `BatchDecodeBuffers::argmax_out`
   lifetime + sample_batch_greedy_launch ordering.
2. If NOT real: 1-line fix removing the precondition. Bench
   long-ctx 4k/c=4 → expect TTFT cuts to ~half (mixed path is
   ~3× faster than split based on the 252 vs 787ms data from
   c=8/2k).
3. If real: implement double-buffered argmax_out (~50 LOC). Same
   bench validation.

## Process rule

- **F4-Small's added precondition is exactly the kind of
  "play-it-safe guard" that has performance cost which was
  invisible at landing time** — F4-Small's bench was high-conc
  (which never hits the Mixed/Split branch). Long-ctx exposes the
  cost.
- **Each performance fix should test at multiple workload shapes**
  before declaring victory. F4-Small's +82.5% at high-conc was
  real; F4-Small's regression at long-ctx mixed workloads was
  hidden until the longctx-vs-vLLM bench surfaced it.
- This finding supersedes my earlier H2 attention-scaling analysis
  (`f5d0fd8`) — the 10× tax has a CONCRETE structural cause
  (Mixed disabled), not a math/kernel cause. Update M3.9 plan.

## Bench Status

No new bench. Source-survey + git-blame provide the evidence.
Verification bench follows after the fix design is confirmed.

## Cross-references

- F4-Small commit: `2a534c4`
- Plan_step source: `infer/src/scheduler/cuda/execution.rs:393-414`
- Deferred-decode-emit set sites: `decode.rs:804, 832`
  (`Ok(None)` from `sample_batch_greedy_readback` when event not ready)
- M3.9 plan: `63af21f` — to be revised given this finding
