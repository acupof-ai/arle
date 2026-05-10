---
title: M_pf83 H1' v2 redesign brief — extend MarlinScratch (do NOT create PF8Scratch); fix per-call alloc that fails on FIRST request
date: 2026-05-10
type: plan
status: ready-for-codex-pickup (replaces blocked M_pf83_h1prime_static_scratch.md original §3)
related_docs: [`M_pf83_h1prime_static_scratch.md` blocked plan, `0be278f` PF8.5 KILL, `7ed8160` Arm B REFUTE, `2b956ce` sm_89 W4 alternatives §4.3]
---

# M_pf83 H1' v2 — redesign brief (Task #47 unblock)

> **Why this brief**: original `M_pf83_h1prime_static_scratch.md` §1 root
> cause hypothesis ("per-call alloc fragmentation under sustained load")
> was REFUTED by Arm B (`7ed8160`): with `INFER_PREFILL_WARMUP=0`, kernel
> still fails 5959 times starting at Request 0 — fragmentation cannot
> happen on FIRST call. The bug is per-call alloc PERIOD, not fragmentation.
>
> This brief redirects the redesign: use the EXISTING `MarlinScratch`
> struct (already pre-allocates W4/W4A8 workspaces at engine init in
> `infer/src/ops/linear.rs:319`) instead of designing a separate
> `PF8Scratch`. Extend it, don't fork it.

## §1 Refined root cause

Per `0be278f` + `7ed8160`:
- code 2 = `cudaErrorMemoryAllocation`
- Fails on Pass 3 warmup B=1 1540 tokens (`INFER_PREFILL_WARMUP=1`)
- Fails on Request 0 with no warmup (`INFER_PREFILL_WARMUP=0`)
- VRAM utilization < 50% at failure point — NOT a real OOM
- Arms C (W4A16) + D (W4A8-zpfix) HEALTHY → not a global allocator issue

Likely actual mechanism (hypothesis, MUST be A/B verified):
1. **cudarc CachingAllocator per-stream pool isolation**: PF8 path may
   allocate on a different stream than main forward, hitting an empty
   pool on first call
2. **Specific size class unsupported**: `s_activation` (just `m × 4 B`,
   tiny) hits a degenerate path in cudarc's pool
3. **Stream context issue**: alloc context not initialized before kernel
   launch on PF8 path

The §1 hypothesis in the original plan ("fragmentation under sustained
load") was wrong because Arm B happens BEFORE any sustained load.

## §2 Solution: extend MarlinScratch (NOT new struct)

`infer/src/ops/linear.rs` already has:

```rust
struct MarlinScratch {
    // existing fields (lines 319-410):
    a_fp16: CudaSlice<Half>,
    a_fp8: Option<CudaSlice<u8>>,             // W4A8 path
    s_activation: Option<CudaSlice<f32>>,     // W4A8 path
    reduce: Option<CudaSlice<f32>>,           // W4A8 path
    w4_workspace: Option<CudaSlice<i32>>,     // W4A16 path (PRE-ALLOCATED ✓)
    w4a8_workspace: Option<CudaSlice<i32>>,   // W4A8 path (PRE-ALLOCATED ✓)
    out_fp16: CudaSlice<Half>,
    // ... capacity_m, capacity_n, capacity_k, sm_count
}
```

PF8 path (`run_marlin_w4_fp8_prefill`, lines 1637-1693) currently does
PER-CALL allocs via `CudaSlice::alloc_zeros` instead of using
MarlinScratch. **This is the bug.** The MarlinScratch pattern already
proven (W4A16 + W4A8 paths use it, both HEALTHY in Arms C+D).

### §2.1 Diff scope (~50 LOC)

1. Add to MarlinScratch struct (line ~319): no new fields needed if
   existing `a_fp8` + `s_activation` + `reduce` + `w4a8_workspace`
   buffers can be SHARED between W4A8 and PF8 paths (verify shapes
   match).
2. If shapes differ slightly, add PF8-specific fields: `pf8_x_fp8`,
   `pf8_workspace`. Reuse `a_fp8` etc. where shape matches W4A8.
3. In `run_marlin_w4_fp8_prefill` (line 1637), replace 5 per-call
   `alloc_zeros` calls with `scratch.pf8_x_fp8.slice(0..m*k)` etc.
4. Add capacity check at function entry (matches existing
   W4A8/W4A16 pattern).

### §2.2 Acceptance gates (per blocked plan §0)

A/B at conc=1 prompt=512 60s sustained:
- **Old path** (per-call alloc, broken): 5959 kernel failures, TTFT 0ms
  artifact, kill verdict
- **New path** (MarlinScratch-routed): MUST be 0 kernel failures, TTFT
  ≤ Arm D W4A8-zpfix bar (54.2ms), 60s clean run

A/B at conc=4 prompt=4096 60s (sustained-load gate, per da7f5a2):
- 0 kernel failures
- VRAM stable (no leak)
- TTFT/tok-s within ±5% of W4A16 + W4A8 controls

## §3 Risk: even MarlinScratch route may fail at first call

If the underlying mechanism is stream-isolation or specific size-class
issue (per §1 hypothesis 1 or 2), even pre-allocated MarlinScratch
buffers might still hit code 2 if the path uses a different stream
than allocation context.

**Mitigation**: codex's first task BEFORE re-implementing is to add
diagnostic logging at the failing alloc site:
```rust
// in run_marlin_w4_fp8_prefill alloc_zeros call sites
let result = ctx.stream.alloc_zeros::<u8>(m * k);
if let Err(e) = &result {
    eprintln!("PF8 alloc failure: m={} k={} stream_id={:?} pool_state={:?}",
              m, k, ctx.stream.id(), /* cudarc pool stats */);
}
```

Run bench v11 with this logging FIRST to confirm whether failure mode
matches §1 hypothesis. THEN implement MarlinScratch route. Don't
implement blind.

## §4 Wall-clock budget

- Diagnostic logging + repro bench: 1-2 hr (codex)
- MarlinScratch struct extension: 1-2 hr (codex)
- `run_marlin_w4_fp8_prefill` rewire: 1 hr (codex)
- Conc=1 + conc=4 A/B bench: 1 hr (claude)
- Code review + commit: 0.5 hr (claude codex review)
- **TOTAL**: ~5-7 hr (~1 day)

## §5 Position in priority order (per `2b956ce` §5)

This is **P3** in the post-Machete pivot ordering:
- P1: A+B combined (Medusa + Hybrid) — highest ROI
- P2: vLLM upstream Marlin diff-port — quickest win
- **P3: PF8.3 H1' v2 redesign** ← THIS BRIEF
- P4: Cutlass FP8 direct mma sm_89
- P5: Wait sm_100

Not blocking A+B. Pick up if A+B falls short of expected gains, or as
parallel codex track if multiple agents are available.

## §6 Cross-references

- Original blocked plan: `M_pf83_h1prime_static_scratch.md`
  (§3 PF8Scratch struct sketch — superseded by §2 here)
- `0be278f` PF8.5 KILL evidence
- `7ed8160` Arm B REFUTES warmup-DEPENDENT framing (key insight for §1)
- `06b7437` Arm C HEALTHY (W4A16 control, validates MarlinScratch pattern)
- `d8b2870` Arm D HEALTHY (W4A8-zpfix control, validates MarlinScratch pattern)
- `infer/src/ops/linear.rs:319` MarlinScratch struct definition
- `infer/src/ops/linear.rs:1637` run_marlin_w4_fp8_prefill (the broken path)
- `crates/cuda-kernels/csrc/gemm/marlin_w4_fp8_kernel.cu` (kernel — works at conc=1 per H8 DISPROVEN `57c37b5`)
- `2b956ce` sm_89 W4 alternatives §4.3 (P3 priority context)
