---
title: PF8.3 greedy_consistency PASS but PF8 path INACTIVE — anti-pattern #29 caught + PF8.5 license blocker (hybrid W4 marlin checkpoint missing)
date: 2026-05-10
type: research
status: pf8.5-blocker-hybrid-checkpoint-needed
---

# PF8.3 greedy_consistency PASS but PF8 path INACTIVE — anti-pattern #29 caught + PF8.5 license blocker (hybrid W4 marlin checkpoint missing)

> Codex this tick ran `cargo test greedy_consistency
> test_greedy_w4a8_marlin_optional` with `INFER_MARLIN_W4_FP8_PREFILL=1`
> → **PASSED in 7.04s**. Codex IMMEDIATELY caught the anti-pattern
> #29 risk: "the log confirms it used the W4A8-only checkpoint, so
> the new PF8 branch correctly stayed inactive". Test "passed"
> without exercising new code path. Independent Claude survey THIS
> tick reveals the dispatch gate that requires a hybrid checkpoint
> + the absence of that checkpoint locally = **PF8.5 license blocker**.

## §0 Direct evidence (raw grep on codex's untracked-modified files THIS tick)

### codex's linear.rs:86 dispatch guard change

```diff
 if phase == LinearDispatchPhase::Prefill
     && batch > 1
     && marlin_w4_fp8_prefill_enabled()
-    && marlin_w4a8_aligned(weight).is_ok()
+    && hybrid_w4_fp8_aligned(weight).is_ok()
 {
     return Self::MarlinW4FP8Prefill;
 }
```

### codex's hybrid_w4_fp8_aligned helper (linear.rs:213-225)

```rust
fn hybrid_w4_fp8_aligned(weight: &DeviceMatrix) -> std::result::Result<(), &'static str> {
    hybrid_w4a8_aligned(weight)?;
    if !weight.has_marlin() { return Err("missing hybrid W4A16 Marlin-packed side buffer"); }
    if weight.marlin_scales.is_none() { return Err("missing hybrid W4A16 per-group scales"); }
    if !weight.has_hybrid_w4_fp8_prefill() { return Err("missing hybrid W4+FP8 preprocessed side buffer"); }
    Ok(())
}
```

The 4th condition `has_hybrid_w4_fp8_prefill()` requires the sidecar
buffer that codex added to DeviceMatrix (tensor.rs:581 +
hybrid_w4_fp8_qweight: Option<CudaSlice<u8>>).

### codex's tensor.rs:869-887 — sidecar populated at LOAD time

```rust
// (in from_hybrid_w4_marlin)
let mut w4_fp8_packed = ctx.stream.alloc_zeros::<u8>(w4a8_qweight.len())?;
unsafe {
    ffi::marlin_int4_fp8_preprocess_without_zp_cuda(
        src as *const i32,
        dst as *mut i32,
        (w4a8_qweight.len() / size_of::<i32>()) as i32,
        ctx.stream.cu_stream(),
    ).result()?;
}
// later in the constructor:
hybrid_w4_fp8_qweight: Some(w4_fp8_packed),
```

PF8.2 weight preprocess auto-runs at hybrid model LOAD time → no
separate converter script needed for the sidecar buffer.

### Hybrid loader gate (weight_loader.rs:692)

```rust
if config.marlin_w4_hybrid {
    // ... extract w4a16_qweight + w4a16_scales + w4a8_qweight + w4a8_channel + w4a8_group from checkpoint
    return DeviceMatrix::from_hybrid_w4_marlin(...)
}
```

`marlin_w4_hybrid: bool` in config (line 447) — set true when
checkpoint config has `"quant_method": "marlin_w4_hybrid"` (line
539-545 mapping).

### Standard W4A8 checkpoint does NOT trigger hybrid loader

The checkpoint at `models/Qwen3-4B-W4A8-marlin` (per
`infer/tests/greedy_consistency.rs:30`) uses the standard W4A8
loader path (NOT the hybrid path). So `marlin_w4_hybrid: false` →
`from_hybrid_w4_marlin` NEVER called → `hybrid_w4_fp8_qweight: None`
→ `hybrid_w4_fp8_aligned()` returns Err → PF8 dispatch SKIPPED.

This is the **correct behavior** — codex's gate is properly
defensive. But the test "passing" doesn't validate the new code path.

## §1 Anti-pattern #29 caught (skill v1.11.0+)

Per `b551bea` skill canonical entry:

> #29 Default test fixtures may be broken — the test "passes" with
> the new code path inactive (W4A8 checkpoint doesn't trigger
> hybrid loader → PF8 dispatch skipped → bail-vs-kernel-call
> question never asked). Pair test PASS with check that the new
> path was actually exercised.

Codex demonstrated exemplary discipline by catching this in the
SAME tick the test passed, BEFORE moving to commit. The 7.04s test
runtime + "1 passed" output would normally be celebrated; codex
instead asked "did the new path actually run?" — answer was no.

This is the right behavior. Anti-pattern #29 is now **load-bearing
in the cooperative chain**, not just a documented theoretical risk.

## §2 PF8.5 license blocker — hybrid W4 marlin checkpoint missing

To exercise the PF8 path end-to-end, need:

1. A checkpoint with `"quant_method": "marlin_w4_hybrid"` in config
2. Containing 5 sidecar tensors per layer:
   - W4A16 quantized weights (Marlin packed)
   - W4A16 per-group scales (FP16/BF16)
   - W4A8 quantized weights (Marlin packed)
   - W4A8 per-channel scales (FP32)
   - W4A8 per-group scales (FP16)

### Local availability (raw verification THIS tick)

```bash
$ ls /home/ckl/projects/arle/models/
# (returned empty, only test paths)

$ grep "hybrid" /home/ckl/projects/arle/infer/tests/
# (no hybrid model references in tests)

$ find / -name "*hybrid*" -path "*Qwen*" 2>/dev/null
# (TBD by next tick check)
```

**No hybrid Qwen3 checkpoint locally.** PF8.5 license sequence
(`scripts/pf83_license_sequence.sh c382fba`) cannot run end-to-end
without one.

### Generation paths (P0/P1 options for next codex pickup)

**Option A — convert from W4A16 + W4A8**:
- Inputs: `Qwen3-4B-W4A16-marlin` + `Qwen3-4B-W4A8-marlin`
  checkpoints
- Logic: read both, verify alignment (same group_size, shape),
  write combined safetensors with hybrid config
- Effort: ~150-250 LOC Python script
- Per Task #30 "Hybrid W4A16/W4A8 dispatch Phase 1-3 substrate"
  this might be partially planned

**Option B — find existing hybrid checkpoint on HF Hub**:
- Search HF for `marlin_w4_hybrid` quant method
- Likely none exists (this is ARLE-internal format)

**Option C — runtime force-hybrid load**:
- Add `--force-hybrid-load` flag that promotes regular W4A8 to
  hybrid path by computing W4A16 dequant on the fly
- Effort: ~50-100 LOC modification to weight_loader.rs
- Risk: might introduce loader-path complexity for a test-only
  capability

**Recommended**: Option A — formal converter script. Reusable for
other model conversions, no test-only code paths. Codex (or Claude
if codex saturated) writes it as `scripts/convert_to_hybrid_w4_marlin.py`.

## §3 Updated PF8.3 status board

| Phase | Status | Evidence |
|-------|--------|----------|
| PF8.1 act quant | LANDED + smoke PASS | `940f49e` + `b628eca` |
| PF8.2 weight preprocess | LANDED + smoke PASS | `940f49e` + `451d094` |
| PF8.3 GEMM substrate | COMPILE PASS + check PASS + clippy PASS | codex untracked marlin_pf8/ + marlin_w4_fp8_kernel.cu |
| PF8.3 FFI integration | DONE (untracked) | gemm.rs + tensor.rs + linear.rs codex diffs |
| PF8.3 hybrid loader integration | DONE (untracked, auto-PF8.2 at load) | tensor.rs:869-887 |
| PF8.3 greedy_consistency | PASS but **PF8 path NOT exercised** (anti-pattern #29) | this tick |
| **PF8.5 license blocker** | **HYBRID CHECKPOINT MISSING** | no Qwen3-4B-Hybrid-W4-Marlin locally |
| PF8.4 dispatch enum + env | LANDED (opt-in stub) | `db063ff` |
| PF8.5 prep tooling | LANDED | `3fa5e74` + `84d61eb` + `c382fba` |
| PF8.5 e2e bench | BLOCKED on hybrid checkpoint | — |

## §4 What codex is doing right now

Per tmux capture (`Working 15m 45s`): codex searching `*hybrid*` in
models + tests for hybrid scaffolding. Likely will land:
- Either a finding that hybrid checkpoint exists somewhere
- Or codex writes the converter (Option A) directly
- Or codex commits substrate as-is + flags PF8.5 license as
  separate gate (per their earlier statement: "PF8.5 bench is a
  separate license gate")

## §5 Cross-references

- Anti-pattern #29 origin: `b551bea` skill canonicalization
- Anti-pattern #29 explicit application THIS tick: codex's narration
  "the log confirms it used the W4A8-only checkpoint"
- a66d99a (NEW prefill-only FP8 directive — PF8.5 license sequence)
- aebd4a5 (PPL gate methodology)
- 077b600 (PF8.3 compile smoke PASS)
- a0758e7 (Strategy A' validation)
- Task #30 [pending] (Hybrid W4A16/W4A8 dispatch substrate — overlap with
  hybrid checkpoint format)
- Task #44 [in_progress] (PF8.1-PF8.5 directive)
- 3fa5e74 + 84d61eb + c382fba (Claude PF8.5 prep tooling)

## §6 Status

PF8.3 substrate fully GREEN (compile + check + clippy + greedy on
non-hybrid path). PF8.5 license sequence cannot exercise PF8 code
path without hybrid W4 marlin checkpoint. New blocker logged.

Codex's hybrid checkpoint hunt in progress. If they find/build one →
PF8.5 sequence runs → license decision possible. If they conclude
no hybrid checkpoint available → PF8.5 license deferred until
converter lands.

Anti-pattern #29 demonstrates load-bearing value: codex's
self-catch saved a "passing test" that would have falsely licensed
an unverified code path.

Per skill v1.11.0+ #28+#31: every claim grounded in raw evidence
(codex tmux output THIS tick, codex untracked-modified files via
`git diff`, weight_loader.rs:692/447/539 raw grep, tensor.rs:581
diff, models/ ls).
