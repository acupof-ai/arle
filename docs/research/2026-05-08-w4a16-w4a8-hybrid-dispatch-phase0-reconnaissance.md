# M_quant W4A16/W4A8 hybrid dispatch Phase 0 reconnaissance

> Codex `128fe32` plan Phase 0(0.25d Claude reconnaissance)。Verifies
> scheduler phase boundary detection,Linear weight loading flow,and
> memory cost analysis for hybrid prefill-W4A8 + decode-W4A16 dispatch。
>
> **Findings**:scheduler boundary CLEAN(StepPlan enum already
> discriminated),Linear dispatch needs **TWO DeviceMatrix per layer**
> (current `WeightFormat` enum is single-format-per-tensor),memory cost
> **45% GPU**(7.15/16 GB)— tight but feasible on Qwen3-4B + sm_89。

## §1 Scheduler phase boundary — CLEAN

`infer/src/scheduler/cuda/execution.rs:411` `Scheduler::plan_step()`
returns `StepPlan` enum:
```rust
StepPlan::Decode                  // pure decode step (all batched decode rows)
StepPlan::Prefill(candidates)     // pure prefill step
StepPlan::Mixed(candidates)       // mixed prefill + decode in same step
StepPlan::Split(candidates)       // chunked-prefill split
```

Decode-only and Prefill-only paths are **already cleanly discriminated**
at scheduler level。Hybrid dispatch decision = trivial match on
`StepPlan` variant:
```rust
match plan {
    StepPlan::Decode => PhaseHint::Decode,        // → W4A16 path
    StepPlan::Prefill(_) | StepPlan::Split(_) => PhaseHint::Prefill,  // → W4A8 path
    StepPlan::Mixed(_) => /* TBD: prefill priority? hybrid mid-step? */
}
```

**Mixed step is the design question**:if a step has both prefill rows
AND decode rows,which kernel to use? Options:
- A:Use W4A8 for ALL Linear calls in mixed step(prefill priority,decode pays activation overhead)
- B:Use W4A16 for ALL(decode priority,prefill loses TTFT advantage)
- C:Per-row dispatch within step(complex,multiple kernel launches per Linear)

**Recommendation**:Option A(prefill priority)— ARLE step times are
dominated by prefill anyway when prefill rows present;decode performance
in mixed steps is already capped by prefill chunking。

## §2 DeviceMatrix struct — single-format per tensor

`crates/cuda-kernels/src/tensor.rs:398` `WeightFormat` enum:
```rust
pub enum WeightFormat {
    DenseBf16,
    W8A16,
    W4A16,        // ← naive sym W4A16-marlin path
    MarlinW4A8,   // ← W4A8 path
    W2A16,
    GgufQ3K,GgufQ4K,GgufQ5K,GgufQ6K,
    TurboQuant,
}
```

Each `DeviceMatrix` carries ONE `WeightFormat`。To hold both W4A16 and
W4A8 packed bytes for the same Linear layer,we need either:

### Option A — Two DeviceMatrix per Linear

```rust
pub struct HybridLinear {
    weight_w4a16: DeviceMatrix,  // for decode
    weight_w4a8:  DeviceMatrix,  // for prefill
}
```

LOC:~50 LOC in Linear container + ~20 LOC in loader to read both side
tensors。Memory:**2× weight pool**(both formats resident)。

### Option B — New WeightFormat variant

```rust
pub enum WeightFormat {
    ...,
    MarlinW4Hybrid,  // contains both packed buffers
}
```

Requires new struct fields + kernel selection logic inside dispatch。
LOC:~100 in tensor.rs + ~50 in dispatch。Memory:same as Option A。

### Option C — Runtime conversion

Store W4A8 only;dequant to BF16 on-demand for decode then re-quant
to W4A16。**REJECTED**:dequant + re-quant per Linear call =
prohibitive overhead,defeats purpose of hybrid。

**Recommendation**:Option A(simpler,1-day codex impl)。Option B is
nicer architecturally but more LOC for marginal benefit。

## §3 Memory cost analysis

For Qwen3-4B(252 Linear layers,total 4B params):
- W4A16 checkpoint(`Qwen3-4B-GPTQ-W4A16-marlin-zpfix`):**4.5 GB**(scales fp16 + qweight int4 + extra Marlin layout overhead)
- W4A8 checkpoint(`Qwen3-4B-GPTQ-W4A8-zpfix`):**2.65 GB**(qweight int4 + s_channel fp32 + s_group fp16)
- Sum if both resident:**7.15 GB / 16 GB = 45%**
- Plus KV cache(~5 GB at typical max-seq-len 5120 × 4 slots):**12.15 GB / 16 GB = 76%**
- Plus activations + workspaces(~1-1.5 GB):**~13-14 GB = 81-87%**

**Verdict**:Feasible but TIGHT。Production deployment will hit OOM
margin at higher concurrencies(c=8 → 8 slots × KV ≈ 10 GB,plus 7 GB
weights = 17 GB > 16 GB)。

**Mitigation options**:
- Reduce max-seq-len for production hybrid(e.g., 4096 instead of 5120)
- Skip hybrid for c≥8(at higher conc,prefill TTFT is queue-bound
  not kernel-bound,W4A8 advantage diminishes)
- Quantize KV(W4A8 KV cache,paired master strategy §1.2.1.B)

For Qwen3.6 35B-A3B MoE on Apple sm_metal — different memory budget;
hybrid may not fit。Restrict hybrid to Qwen3-4B + sm_89 production for
this cycle。

## §4 Phase 1 implementation scope refinement(codex)

Per codex `128fe32` Phase 1 estimate(0.5d):

### Loader storage augmentation
- `weight_loader.rs`:read both `marlin_qweight + marlin_scales`(W4A16
  side)AND `marlin_w4a8_qweight + marlin_w4a8_s_channel + marlin_w4a8_s_group`
  (W4A8 side)from same checkpoint dir
- Detection priority:if both files present → hybrid;if only one →
  fallback to that format only;config.json `quantization_config.quant_type`
  could be `marlin_w4_hybrid` to opt in
- LOC:~80(branching + new tensor field + Option<W4A8> handling)

### DeviceMatrix container
- Per Option A:`HybridLinear { w4a16: DeviceMatrix, w4a8: DeviceMatrix }`
- LOC:~50(struct + accessors + serde if checkpoint format)

### run_linear PhaseHint dispatch
- Add `PhaseHint::Prefill` and `PhaseHint::Decode` enum
- Caller(scheduler step plan)passes hint based on `StepPlan` variant
- LOC:~30(enum + match + dispatch)

### Single-checkpoint hybrid generation script
- New `scripts/convert_to_hybrid_w4_marlin.py`:reads GPTQ-Int4-converted-zpfix
  source and outputs both side-tensors in one safetensors
- Reuses `marlin_repack.py`(W4A16 path)+ `convert_gptq_w4a16_to_w4a8_marlin.py`
  (W4A8 path)logic
- LOC:~100(combine both pack flows)

**Total Phase 1**:~260 LOC(within codex `128fe32` estimate of 150-300)

## §5 Skill v1.4.0 anti-pattern check

Per anti-pattern #12(decode-vs-prefill duality)— this hybrid
implementation is the ARCHITECTURAL response to the duality:
- W4A8 wins prefill(empirical `b5889b3`,c=4 + c=8)
- W4A16 wins decode(empirical `bc15eca` + `8588f6a`)
- Hybrid = use specialized kernel per phase(not within-kernel branching
  like R4 #6 KILL)

Per anti-pattern #14(upstream parser silent corruption)— pre-condition
for hybrid is BOTH formats hitting their bandwidth ceilings,which
requires zpfix-corrected source(`2a3a6f0`)。**Hybrid evaluation pre
qzeros fix would have been wrong-layer debugging**。

## §6 Phase 0 deliverables checklist

- ✅ Scheduler phase boundary detection clean(StepPlan enum)
- ✅ Linear weight storage analysis(two DeviceMatrix recommended)
- ✅ Memory cost analysis(45% GPU,tight but feasible at c=4)
- ✅ Mixed-step dispatch question answered(Option A prefill priority)
- ✅ Phase 1 LOC scope refinement(~260 LOC total)
- ✅ Anti-pattern audit(this is NOT R4 #6 KILL pattern)

## Action

Codex Phase 1 unblocked。Recommended order:
1. Loader storage augmentation(codex,0.25d)
2. DeviceMatrix HybridLinear container(codex,0.25d)
3. run_linear PhaseHint dispatch(codex,0.25d)
4. Hybrid checkpoint generation script(Claude,0.25d ~100 LOC)
5. greedy_consistency + bench gate(codex,0.25d)

Total wall-time:1.25 days(within `128fe32` estimate)

## Cross-references

- Codex hybrid plan: `128fe32`(`docs/plans/M_quant-w4a16-w4a8-hybrid-prefill-decode.md`)
- W4A8 prefill LICENSED:`b5889b3`(c=4)+ `8588f6a`(c=4 + c=8 multi-conc)
- W4A16 LICENSED 1.64×:`bc15eca`(GPTQ-zpfix matches naive sym)
- R4 #6 KILL precedent: `r4-hybrid-dispatch-killed-batch4-decode-regression.md`(different pattern,does NOT apply)
- Skill v1.4.0 anti-pattern #12+#14:`6c627c4`
- Scheduler StepPlan: `infer/src/scheduler/cuda/execution.rs:411`
- WeightFormat enum: `crates/cuda-kernels/src/tensor.rs:398`
- Linear dispatch: `infer/src/ops/linear.rs:50-100`

## Status

- ✅ Phase 0 reconnaissance complete(this entry)
- ⏳ Phase 1 loader + container + dispatch(codex,~0.75d)
- ⏳ Phase 4 hybrid checkpoint generation(Claude,~0.25d 100 LOC)
- ⏳ Phase 3 e2e gate + Phase 4 production bench(codex,~0.5d)

## Rule

**For dual-kernel hybrid dispatch via phase boundary,verify scheduler
discrimination is clean BEFORE committing to dispatch implementation**。
ARLE's StepPlan enum already discriminates Decode/Prefill/Mixed/Split;
no scheduler refactor needed。

If a system lacks clean phase discrimination(e.g., monolithic step
loop),hybrid dispatch by phase requires schedule refactor first —
gating cost。This Phase 0 confirms ARLE doesn't pay that cost。

Memory cost rule:hybrid dispatch with both formats resident requires
production memory budget audit at TARGET concurrency。Don't ship
hybrid without verifying it fits at production c。45% GPU at c=4 is
acceptable;c=16 may need KV-format quantization first。
