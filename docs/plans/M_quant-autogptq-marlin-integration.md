# M_quant — AutoGPTQ → Marlin integration(axis 1 W4 production path)

**Status:** P0 plan — ready for codex pickup after current admission-fix landing
**Owner:** TBD(codex implementation,Claude planning)
**Trigger:** `39237b9` finding — naive max-scale W4 too lossy,calibration required
**Master strategy ref:** §1.2.1.A weight axis FP8 + §1.2.1.B KV W4A8

## §1 Why

`39237b9` empirically established:
- ARLE W4A8 pack matches PR #31 W4A8Layer byte-for-byte across 8/8 shapes
- Kernel recovers weights with 0.8% rel error from packed bytes
- 100%-token-diff in greedy_consistency is **inherent quant noise of naive max-scale**,not a bug

→ Production W4 accuracy requires **calibration**(GPTQ / AWQ family)。

**Goal**:integrate AutoGPTQ-produced GPTQ-Marlin checkpoints into ARLE,
unblock W4A8 default-on accuracy gate per master `2026-05-07-arle-master-strategy.md` §1.2.1.A。

## §2 Math / why GPTQ helps

Naive max-scale W4 quant noise per element:
```
ε_naive ≈ scale / 2 = max(|w|) / (2 × 7) ≈ 7% relative
```

For 36-layer Qwen3-4B with 4 GEMMs/layer = 144 W4A8 ops:
- Per-op noise compounds: (1+ε)^144 ≈ 25× std growth
- argmax stability fails at ~5-10 layers → 100% token diff

GPTQ uses **inverse Hessian** to update remaining columns when
quantizing column j,minimizing reconstruction error:
```
ε_GPTQ ≈ 2-4× lower than naive max-scale
≈ 1.5-3% relative per element
→ (1+ε)^144 ≈ 2-5× std growth
→ argmax stability holds for full 36 layers
```

Empirical:GPTQ Qwen3-4B W4 typically loses ≤1% on MMLU vs FP16 baseline。
AutoGPTQ + Marlin combined reportedly achieves <0.5% MMLU degradation。

## §3 Production path inventory

### 3.1 AutoGPTQ checkpoint sources

| Path | Status | Notes |
|------|--------|-------|
| HuggingFace Hub `Qwen/Qwen3-4B-GPTQ` | TBD existence check | Search HF for official Qwen3-4B GPTQ |
| Generate locally via AutoGPTQ | Always available | ~30-60 min on 4070Ti SUPER + calibration dataset |
| vLLM-compatible GPTQ-Marlin | If exists | byte-compat with our `marlin_w4a8_kernel.cu` |

### 3.2 ARLE loader compatibility

Existing loader at `infer/src/weight_loader.rs:663-715`:
- Reads:`.marlin_w4a8_qweight` / `.marlin_w4a8_s_channel` / `.marlin_w4a8_s_group`
- Layout:int32 packed,f32 channel scales,f16 group scales
- Calls into `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu`(0-diff vs PR #31)

AutoGPTQ output naming convention:
- `.qweight`(int32 packed)
- `.qzeros`(int32 zero points)
- `.scales`(f16 group scales)
- `.g_idx`(int32 group indices,sometimes optional)

→ **Naming mismatch needs adapter**:converter from AutoGPTQ → ARLE
naming convention,OR loader reads both naming styles。

## §4 Implementation phases

### Phase 0 — Reconnaissance(0.5 day,Claude)
- [ ] Check HF Hub for existing `Qwen/Qwen3-4B-GPTQ-Int4` checkpoints
- [ ] Compare AutoGPTQ packed weight layout to ARLE marlin_w4a8 byte order
- [ ] Verify AutoGPTQ uses same group_size = 128 + sym=True convention
- [ ] Gate criterion:if compatible AutoGPTQ checkpoint exists,proceed to Phase 1;
      else Phase 1 generates locally via `auto-gptq` package

### Phase 1 — Local AutoGPTQ generation(1 day,codex)
```bash
pip install auto-gptq optimum
python scripts/quantize_qwen3_w4a8_gptq.py \
    --src infer/models/Qwen3-4B \
    --dst infer/models/Qwen3-4B-GPTQ-Int4 \
    --bits 4 --group-size 128 --sym \
    --calibration-dataset c4 --calibration-samples 128
```
- Generates GPTQ-Marlin compatible safetensors
- ~30-60 min wall time(includes calibration forward passes)
- Output:safetensors with AutoGPTQ naming convention

### Phase 2 — Loader adapter(1-2 days,codex)
- [ ] Add `weight_loader.rs` branch:detect AutoGPTQ naming convention via config or tensor name probe
- [ ] Map AutoGPTQ tensors to ARLE conventions:
  - `.qweight` → `.marlin_w4a8_qweight`(byte-compat after permute,if needed)
  - `.scales` + `.qzeros` → `.marlin_w4a8_s_channel` + `.marlin_w4a8_s_group`(scale convention reconciliation)
- [ ] Gate criterion:`cargo test --release -p infer --features cuda --test greedy_consistency::test_w4a8_vs_bf16_token_diff` PASSES with new checkpoint

### Phase 3 — Bench + default-on flip(1 day)
- [ ] Run `scripts/bench_guidellm.sh m_quant-w4a8-gptq` for production numbers
- [ ] Compare TTFT/ITL/throughput vs:
  - BF16 baseline
  - W4A16 Marlin(`f6f3af3` license)
  - W4A8 naive max-scale(known-bad,for noise verification)
  - SGLang / vLLM W4A8 (if those have GPTQ-Marlin too)
- [ ] If accuracy + perf both good → flip default-on(unblocks `62e75ee` graph capture)

### Phase 4 — Documentation(0.5 day,Claude)
- [ ] Update `docs/support-matrix.md` with W4A8 GPTQ status
- [ ] Add `docs/experience/wins/2026-05-XX-w4a8-gptq-production-bench.md`
- [ ] Master strategy §1.2.1.A update:W4A8 line `❌` → `✅ via AutoGPTQ`

## §5 Risk + KILL criteria

### Risks
1. **AutoGPTQ output not byte-compatible with our Marlin kernel**
   - Mitigation:Phase 2 byte-level audit + adapter
   - Fallback:tweak AutoGPTQ source to emit ARLE-compatible packing
2. **GPTQ calibration produces lower accuracy than expected on Qwen3.6 MoE**
   - Calibration may struggle with router decisions
   - Mitigation:test on Qwen3-4B(dense)first;defer Qwen3.6 to phase 5
3. **Performance regression vs naive max-scale**
   - GPTQ-quantized weights might dispatch slower if extra ops needed
   - Mitigation:Phase 3 bench will catch;if regression > 5%,investigate kernel path

### KILL criteria
- **Phase 1**:if local AutoGPTQ generation fails / produces nan / crashes for >2 hours debug → Phase 1 KILL,fall back to W4A16 + FP8 hybrid
- **Phase 2**:if loader adapter requires kernel changes(would invalidate audit) → KILL,re-evaluate by writing our own GPTQ
- **Phase 3**:if W4A8-GPTQ accuracy regresses > 2% on lm-eval MMLU vs BF16 → KILL,defer until calibration improvement

## §6 Estimates

| Phase | Wall time | LOC | Risk |
|-------|-----------|-----|------|
| 0 — Reconnaissance | 0.5 day | 0 | Low |
| 1 — AutoGPTQ generate | 1 day | ~150(script) | Low |
| 2 — Loader adapter | 1-2 days | ~200(rust) | Med |
| 3 — Bench + flip | 1 day | 0(uses scripts/bench_guidellm.sh) | Low |
| 4 — Docs | 0.5 day | 0(docs only) | Low |
| **Total** | **4-5 days** | **~350** | **Med** |

## §7 Decision points

D1. **Generate locally vs reuse HF checkpoint**:
   - Reuse if exists(faster,validated)
   - Generate if not(needed in any case for later models like Qwen3.6 35B)

D2. **Loader adapter direction**:
   - Adapter at loader(non-invasive,read both styles): preferred
   - Adapter at quant script(re-pack from AutoGPTQ output): alternative if loader changes risky

D3. **Default-on flip timing**:
   - After Phase 3 bench passes:flip immediately
   - Hold flip until graph capture also wired(`62e75ee`):safer but more delay
   - Recommendation:flip after Phase 3,treat graph capture as separate ROI

## §8 Cross-references

- W4A8 finding: [`39237b9`](../research/2026-05-08-w4a8-naive-max-scale-too-lossy-need-calibration.md)
- W4A16 license: [`f6f3af3`](../experience/wins/...marlin license entry)
- Master strategy: [`docs/projects/2026-05-07-arle-master-strategy.md`](../projects/2026-05-07-arle-master-strategy.md) §1.2.1.A
- Kernel substrate: `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu`(0-diff PR #31)
- Loader: `infer/src/weight_loader.rs:663-715`
- Linear FFI: `infer/src/ops/linear.rs:777-859`
- AutoGPTQ: https://github.com/AutoGPTQ/AutoGPTQ
- vLLM GPTQ-Marlin: vllm/model_executor/layers/quantization/gptq_marlin.py
- PR #31 GPTQ ref: `/tmp/marlin-w4a8/gptq/`

## §9 Methodology validation

This plan ships AFTER 5+ iterations of code-side W4A8 narrowing
(EOD+22 → EOD+30) which were investigating the wrong problem。Per
`39237b9` rule:**when investigating "quantization produces wrong output",
FIRST diagnostic should be round-trip pack/unpack vs upstream reference
test data**。Had we run that diagnostic on day 1,we'd have pivoted to
this calibration plan ~2 weeks earlier。

The plan itself is short and concrete because:
- Code substrate(kernel + pack + loader)is verified correct
- Calibration is a well-known standard solution(AutoGPTQ + GPTQ-Marlin)
- The unknown is byte-compatibility,which Phase 2 explicitly resolves
