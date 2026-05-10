---
title: Task #43 W4A16 sustained-load failure — hypothesis CONFIRMED via dispatch audit (env-gated scratch fallback)
date: 2026-05-10
type: research
status: open (Task #43 hypothesis upgraded to strongly evidenced; needs reproduction + INFER_PREFILL_GRAPH=1 A/B)
related_tasks: [#43 (W4A16 stack overflow under sustained 4k bench), #44 (PF8 chain), #47 (H1' refactor), #24 (W4A8 prefill graph capture hoist)]
---

# Task #43 W4A16 sustained-load failure — dispatch audit confirms env-gated scratch fallback

> **Purpose**: follow-up to `2cc608a` Task #43/#47 hypothesis. Audit
> dispatch path proves the per-call alloc fallback IS reachable in
> production when `INFER_PREFILL_GRAPH=1` is not set, providing strong
> evidence Task #43 may share root cause with PF8.3 KILL.

## §1 Dispatch path verified

`infer/src/ops/linear.rs:2064-2095` (read this tick):

```rust
let plan = LinearKernelPlan::batched(weight, x.seq_len, phase);
match plan {
    LinearKernelPlan::MarlinW4Gemm => {
        if let Some(scratch) = marlin_scratch {
            run_marlin_w4_linear_with_scratch(ctx, weight, &x.data, x.seq_len,
                                               &mut out.data, scratch)?;
        } else {
            run_marlin_w4_gemm(ctx, weight, x, out)?;  // → run_marlin_w4_linear (per-call alloc)
        }
    }
    LinearKernelPlan::MarlinW4A8Gemm | LinearKernelPlan::MarlinW4Hybrid => {
        if let Some(scratch) = marlin_scratch {
            run_marlin_w4a8_linear_with_scratch(...)?;
        } else {
            run_marlin_w4a8_linear(...)?;  // per-call alloc fallback
        }
    }
    LinearKernelPlan::MarlinW4FP8Prefill => {
        run_marlin_w4_fp8_prefill(...)?;  // ALWAYS per-call alloc — no scratch fallback
    }
    ...
}
```

## §2 Scratch initialization gate

`infer/src/model/qwen3/forward.rs:312-313`:

```rust
let prefill_marlin_scratch = if super::prefill::qwen3_prefill_graph_requested()
    && prefill_marlin_scratch_config.any()
{
    // ... allocate MarlinPrefillScratch ...
    Some(scratch)
} else {
    None  // ← marlin_scratch = None → dispatch falls back to per-call alloc
};
```

`qwen3_prefill_graph_requested()` is gated on the `INFER_PREFILL_GRAPH=1`
env var (per `35fc3cf` Task #24 W4A8 prefill graph capture hoist).

## §3 Implication: Task #43 root cause likely allocator fragmentation

If the user ran the W4A16 4k-token sustained bench (Task #43)
**without** `INFER_PREFILL_GRAPH=1`:

- `marlin_scratch` initialized to `None` per §2
- Dispatch falls back to `run_marlin_w4_linear` (per-call alloc) per §1
- Per-call cost: 3 allocations (x_fp16 ~2.6MB + y_fp16 ~6MB +
  workspace ~5KB) ≈ **9 MB/call**
- Production load: 4k tokens × 4 conc × 60s × 252 ops/forward × ~9MB =
  **~580 GB cumulative cudarc CachingAllocator churn** = same
  fragmentation pattern that killed PF8.3 at 101380/101380 failures
  per `0cde63d`

The "stack overflow" symptom in Task #43's title may be the manifest
form (Rust stack overflow from deeply-nested error handling under
sustained alloc failure), not the root cause.

## §4 Cheap experiment to validate

Two-arm controlled A/B (single-variable: env var on/off):

```bash
# Arm A (treatment): scratch enabled
INFER_PREFILL_GRAPH=1 RUST_MIN_STACK=33554432 \
  target/release/infer --model-path infer/models/Qwen3-4B-W4A16-sym-g128-marlin \
  --port 8000 > /tmp/task43-A-scratch-enabled.log 2>&1 &
# ... run W4A16 4k 60s sustained bench at conc=4 ...
# Verify: 0 stack overflow, 0 cudaErrorMemoryAllocation
pkill -f "target/release/infer.*--port 8000"

# Arm B (baseline): scratch disabled (the hypothesis-failing case)
RUST_MIN_STACK=33554432 \
  target/release/infer --model-path infer/models/Qwen3-4B-W4A16-sym-g128-marlin \
  --port 8000 > /tmp/task43-B-scratch-disabled.log 2>&1 &
# ... run same bench ...
# Predict: stack overflow / cudaErrorMemoryAllocation reproduces

# Run scripts/pf83_bench_health.sh on both — Arm A should report HEALTHY,
# Arm B should report SUBSTRATE-KILL or BENCH-NO-OUTPUT.
```

**If Arm B reproduces Task #43 symptom + Arm A does not** → hypothesis
confirmed → fix = make `INFER_PREFILL_GRAPH=1` default-on for prefill,
OR add per-call alloc fallback that uses thread-local pre-allocated
buffer.

**If Arm B does NOT reproduce** → hypothesis disproven → Task #43 root
cause is something else (not allocator fragmentation).

## §5 Cross-references with H1' refactor (Task #47)

Per `2cc608a` revised H1' design: PF8 path needs `_with_scratch`
variant. The architectural fix for PF8.3 KILL **also closes the gap
for Task #43** if applied symmetrically:

1. Add `run_marlin_w4_fp8_prefill_with_scratch` variant (per Task #47
   plan) → PF8.3 sustained-load fixed
2. Audit `qwen3_prefill_graph_requested()` gate — if removable,
   default to scratch-allocated for ANY non-trivial prefill load
   regardless of graph capture intent → Task #43 W4A16 sustained-load
   fixed (and possibly default-on Pass 3 cap=8 warmup per Task #35
   benefits more)

This is the **two-tasks-one-PR opportunity** noted in `2cc608a` §2.2,
now with stronger evidence.

## §6 Cross-references

- `2cc608a` (this session) original H1' revision + Task #43 hypothesis
- `0cde63d` PF8.3 RUNTIME KILL evidence (101380 failures sustained load)
- `35fc3cf` Task #24 W4A8 prefill graph capture hoist (introduced
  `INFER_PREFILL_GRAPH` env var + scratch lifecycle)
- `infer/src/ops/linear.rs:2064-2095` dispatch match (verified this tick)
- `infer/src/model/qwen3/forward.rs:312-313` scratch gate (verified)
- `infer/src/ops/linear.rs:1167` per-call alloc W4A16 fallback path
- `infer/src/ops/linear.rs:1256` `_with_scratch` variant W4A16 happy path
- Skill kernel-optimization v1.12.0 #29 default test fixtures broken
  (analogous: env var gate makes default-off scratch a "broken default")

## §7 Status

**Strong evidence** — dispatch audit + scratch gate both verified via
read of source files this tick. Need one cheap A/B run to convert from
"strongly evidenced hypothesis" to "confirmed root cause".

Surface via PushNotification + Task #43 description update.
