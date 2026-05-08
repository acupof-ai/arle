# M_quant Round 4 #6 — `W4A16BatchGemv` dispatch override Phase 0 audit

> Per directive §7 binding work pool + `2026-05-08-marlin-w4a16-bench-implementation-gap.md`
> Round 4 #6:override dispatch for `batch>1` to use `W4A16BatchGemv`(BF16-native,
> 1 launch)instead of `MarlinW4Gemm`(3 launches:bf16→fp16 + gemm + fp16→bf16)。
>
> **Phase 0 audit verdict**:CLEAN 5-7 LOC env-gated override possible at
> `linear.rs:67-99`。Both arms accept `WeightFormat::W4A16` weights — no
> tensor-format incompatibility。Implementation ready for codex pickup or
> Claude single-tick(small enough)。

## Phase 0 source audit(post-P0.2 main verified)

### Current dispatch site

`infer/src/ops/linear.rs:67-99` `LinearKernelPlan::batched()`:

```rust
fn batched(weight: &DeviceMatrix, batch: usize) -> Self {
    if marlin_w4a8_aligned(weight).is_ok() {
        return Self::MarlinW4A8Gemm;
    }
    if batch > 1 && marlin_prefill_aligned(weight).is_ok() {
        return Self::MarlinW4Gemm;  // ← R4#6 override target(3 launches)
    }
    // ... fallthrough match
    match (batch, weight.weight_format()) {
        ...
        (_, WeightFormat::W4A16) => Self::W4A16BatchGemv,  // ← R4#6 prefers this(1 launch)
        ...
    }
}
```

### Format compatibility verification

`marlin_prefill_aligned()`(line 102-116):
```rust
weight.has_marlin()  // Marlin-packed side buffer present
&& weight.weight_format() == WeightFormat::W4A16  // Same W4A16 format
&& K % 16 == 0
&& N % 64 == 0
```

→ When `marlin_prefill_aligned()` is `Ok`,`weight.weight_format()` is **already**
`W4A16`。Therefore the W4A16BatchGemv arm at line 86 can dispatch on the SAME
weight without tensor format conversion or fallback。

**Conclusion**:override is **type-safe**。No new fields,no new conversions。

## Concrete override implementation(~5-7 LOC + env var)

```rust
fn batched(weight: &DeviceMatrix, batch: usize) -> Self {
    if marlin_w4a8_aligned(weight).is_ok() {
        return Self::MarlinW4A8Gemm;
    }
    // R4#6 override: prefer W4A16BatchGemv (BF16-native, 1 launch) over
    // MarlinW4Gemm (3 launches: bf16→fp16 + gemm + fp16→bf16) when override
    // env var is set. Safe rollout gate.
    if batch > 1
        && weight.weight_format() == WeightFormat::W4A16
        && std::env::var("INFER_R4_W4A16_GEMV_OVERRIDE").is_ok()
    {
        return Self::W4A16BatchGemv;
    }
    if batch > 1 && marlin_prefill_aligned(weight).is_ok() {
        return Self::MarlinW4Gemm;
    }
    // ... rest unchanged
}
```

LOC delta:**+6 lines**(if-block insertion before existing Marlin path)。

### Why env-gated rollout

- Default OFF preserves production Marlin path(Round 1 measured 1.06× still
  positive vs BF16,no regression risk)
- Bench env can flip ON to compare ITL paired
- If license-pass(≥1.5×),flip default ON in next iteration
- If kill-band(<1.0× or worse),trivial revert by removing env var

## Phase 4 formula(per `marlin-w4a16-bench-implementation-gap.md` Round 4)

```
Per linear call surplus = 2 elementwise conversion launches × ~5-10us = 10-20us
Per token decode: 252 GEMMs × 15us launch overhead saved = 3.8 ms (low end)
                                                          to 5.0 ms (high end)
Predicted ITL = 18.13 ms (Round 1 Marlin actual) - 3.8/5.0 ms saved
              = 14.1 ms (low) → 1.37× vs BF16 baseline 19.27ms
              = 12.1 ms (high) → 1.59× vs BF16

License band: 1.5× (M_quant §9.2)
→ License-band straddler。Worth running。
```

## Bench protocol(matched-control)

Same as Round 1 baseline(`marlin-w4a16-c4-4k`):
- Model:`infer/models/Qwen3-4B-GPTQ-Int4-marlin`(NOT bare GPTQ-Int4 per
  loader limitation noted in Round 1 prep)
- Workload:`prompt_tokens=4096,output_tokens=256,c=4,max-seconds=120,warmup=10`
- KV dtype:`--kv-cache-dtype bf16`(matched per skill checklist)
- Slots:`--num-slots 8 --max-seq-len 5120`

Override-bench command:
```bash
INFER_R4_W4A16_GEMV_OVERRIDE=1 \
CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  ./target/release/infer --model-path infer/models/Qwen3-4B-GPTQ-Int4-marlin \
  --port 8000 --num-slots 8 --max-seq-len 5120 --kv-cache-dtype bf16

# Then matching bench script
scripts/bench_guidellm.sh r4-w4a16-gemv-override \
  --model Qwen3-4B-GPTQ-Int4-marlin \
  --processor /home/ckl/projects/arle/infer/models/Qwen3-4B-GPTQ-Int4-marlin \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_min=256,output_tokens_max=256'
```

Compare to:
- BF16 baseline:`786a20a` 19.27 ms ITL p50
- Marlin Round 1 baseline:18.13 ms ITL p50
- License threshold:12.85 ms ITL p50(≥1.5× vs BF16)

## §0 SOLID gates

### Gate 1 — single-variable A/B(skill anti-pattern #2)

Override change is single-variable(only dispatch routing changes)。Same model,
same KV dtype,same workload,same slots/seqlen。Matched-control rigid。

### Gate 2 — bench shape validation(per B3 Step 2 wins entry rule)

`prompt_tokens=4096 + output_tokens=256` total = 4352 tokens。Server `max-seq-len 5120`。Headroom 768 tokens。**No overflow risk**(unlike B3 Step 2 turns=3 trap)。

### Gate 3 — Layer-8 num_slots gate(per `655accf`)

`--num-slots 8` constant across baseline + override bench。Variable changes:**only
the env var**。No multi-variable confound。

### Gate 4 — n≥3,σ < 5%(skill rule 6)

License criterion:n≥3 paired runs,ITL σ/mean < 5%,Δ% ≥ 10% relative to baseline,or kill-band(< 1.0× regression)。

## Risk mitigation

### Risk 1 — batch range specificity

**Concern**:`W4A16BatchGemv` may only win at batch=2-8 but lose at batch>8(higher
contention)。

**Mitigation**:current batch=4 bench is in the predicted-win range。If license-pass
at batch=4,bench batch=8 + batch=16 separately to find break-even。

### Risk 2 — Marlin-packed weight memory waste

**Concern**:if override is permanent,Marlin-packed side tensors become unused
~2GB VRAM waste。

**Mitigation**:override is env-gated for rollout。If license-pass adopted as
default,follow-up cleanup commit can drop Marlin pack from loader for W4A16-only
checkpoints(separate refactor,not this commit)。

### Risk 3 — Numerical correctness divergence

**Concern**:`W4A16BatchGemv` vs `MarlinW4Gemm` may produce slightly different
greedy outputs due to FP precision path differences。

**Mitigation**:run `cargo test --release -p infer --features cuda --test greedy_consistency`
in BOTH override-OFF and override-ON modes。Verify byte-identical greedy or
license greedy diff per existing W4A8 accuracy gate criteria。

## Phase 7 tradeoffs(skill rule 7 — explicit enumeration)

| Axis | Status | Note |
|---|---|---|
| LOC complexity | ✅ +6 LOC | Single dispatch override + env gate |
| Hardware specificity | ✅ sm_80+ | Both paths support same range |
| Compiler/runtime version | ✅ no new dep | Existing W4A16BatchGemv kernel |
| Maintainability | ⚠ env var | Adds 1 env var to `docs/environment.md` |
| Numerical correctness | ❌ NOT verified yet | Phase 8 gate via greedy_consistency |
| Generality | ⚠ batch-range untested | Predicted win at batch=2-8;>8 unknown |
| Memory budget | ⚠ Marlin pack unused | If override default ON,~2GB waste(follow-up cleanup) |
| Scheduling impact | ✅ none | No envelope or admission change |

## Pickup priority

This is **truly orthogonal work**(per natural-closure heuristic option (b))to
the c20b1ce/12300c5 audit chain that closed at stage 30。Round 4 #6 was
queued from 2026-05-08 and remains unpicked。Implementation is now Phase-0-audited
+ ready。

**Effort**:~10 LOC code + bench(~30 min Claude OR codex)。

**Alternative ordering**:can be picked up alongside cell (d) experiment(both
~30 min wall-clock,both bench-driven,both gated on Layer-8 num_slots=8
constant)。

## Cross-references

- Round 1-3 evidence:`docs/experience/errors/2026-05-08-marlin-w4a16-bench-implementation-gap.md`
- Round 4 prep with W4A16BatchGemv source survey:same doc §"Round 4 prep"(2026-05-08)
- Skill v1.7.0 anti-pattern #18(Phase 0 substrate audit)+ #19(path verification):applied here
- Layer-8 num_slots=8 gate:`655accf`(multi-variable confound prevention)
- Natural-closure heuristic:`memory/feedback_bidirectional_audit_cycle.md`

## Status

**READY FOR PICKUP**。Code change小(~6 LOC env-gated),bench protocol matched-control
rigid(same Round 1 baseline),license band straddles 1.5× target,§0 SOLID gates
identified,risk mitigations explicit。

Rule:**Phase 0 audit reduces pickup-day execution from ~60 min audit + implement
+ bench to ~30 min implement + bench**。Same pattern as `78ccbb6` task #24
pre-audit:pre-staging high-leverage Claude work。
