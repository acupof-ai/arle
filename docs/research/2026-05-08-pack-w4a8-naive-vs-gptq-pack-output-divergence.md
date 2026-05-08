# pack_w4a8 naive vs gptq_scales — empirical pack output divergence isolated

> Direct comparison of `pack_w4a8(naive)` vs `pack_w4a8(gptq_scales=...)`
> on identical input weights from `Qwen3-4B-GPTQ-Int4-marlin`。Refutes my
> earlier "s_channel 18× overflow" hypothesis(`592b80c`)— **s_channel
> is IDENTICAL between modes**。Real divergence is in `s_group`(17% rel
> max)and `qweight`(~7% positions differ)。
>
> Kernel was developed/audited(`01ace86`)against naive pack output。
> GPTQ-aware mode produces structurally similar but quantitatively
> different output → kernel may hold an implicit magnitude / distribution
> assumption that GPTQ-aware path violates。

## Test setup

`scripts/diff_pack_w4a8_naive_vs_gptq.py`(108 LOC)— independent of
codex's investigation,runs both pack modes on same decoded GPTQ weights
+ element-wise diff。

```bash
$ python scripts/diff_pack_w4a8_naive_vs_gptq.py --layer 0 --proj self_attn.q_proj
```

## Empirical results — 2 layers

### Layer 0 self_attn.q_proj

| Tensor | NAIVE | GPTQ-aware | Δ |
|---|---|---|---|
| s_channel mean | 6.71e-4 | 6.71e-4 | **0.0(IDENTICAL)** |
| s_channel max | 4.37e-3 | 4.37e-3 | **0.0** |
| s_group mean | 13.45 | 13.54 | +0.01% |
| s_group max | **18.16** | **21.25** | **+17%** |
| s_group rel max diff | — | — | 17.13% |
| qweight diff > 0 positions | — | — | **84,387 / 1,310,720(6.4%)** |
| qweight max diff(int32) | — | — | 286,330,880 ≈ `0x11111100` = 4 nibbles ±1 |

### Layer 5 mlp.gate_proj

| Tensor | NAIVE | GPTQ-aware | Δ |
|---|---|---|---|
| s_channel | 9.71e-4 mean | 9.71e-4 | **0.0(IDENTICAL)** |
| s_group max | 18.16 | 21.25 | +17% |
| qweight diff > 0 | — | — | **211,124 / 3,112,960(6.8%)** |

**Pattern consistent across 2 layers.**

## Hypothesis revision

### Refuted

- `592b80c` "s_channel 18× overflow"(`max/127` vs `max/7`)— **WRONG**。
  Both modes derive s_channel from data via `max(|w|)/127`(line 113-115
  in `quantize_qwen3_w4a8.py`)。

### Surviving hypotheses

1. **GPTQ groups have weights at non-integer-7 levels** → s_group max
   17% larger because GPTQ Hessian-aware quant doesn't always push max
   to level 7。Kernel was tested against naive(s_group max ≤ 18.16
   bounded by `max/7`),GPTQ-aware allows larger values。
2. **Kernel hardcodes scale distribution / range** specific to naive
   pack — possibly an integer-domain fast path that assumes scales fit
   in a specific bit width。Worth checking
   `marlin_w4a8_kernel.cu` for any int8/uint8 scale conversion or
   bit-shift on s_group values。
3. **`s_group_stored` reshape / permutation** may behave differently
   when underlying values exceed naive range,due to `scale_perm`
   precomputed for naive distribution。

### What this DOESN'T explain

If kernel just does `(q-8) * s_group * s_channel` element-wise,GPTQ-aware
should give CORRECT(GPTQ-calibrated)output,not garbage。The garbage
(token id 0)suggests **NaN or denormal**,which only happens if:
- s_group values overflow some intermediate buffer
- or kernel's reshape/permute breaks at non-naive distributions

## Action — codex own,kernel-side

Codex investigation tasks(based on this evidence):
1. **Inspect `marlin_w4a8_kernel.cu`** for any:
   - INT8/UINT8 scale conversion(would clip > 18.16 to 0)
   - Bit-shift on s_group values
   - Hard-coded normalization assumptions
2. **Run single-layer kernel forward** with naive vs gptq_scales packed weights
   + same activation → compare output element-wise to localize divergence
3. **If kernel is fine,problem is `pack_w4a8(gptq_scales=...)` itself**:
   - Maybe `s_group` should be normalized differently when not derived from data
   - Or `s_channel` derivation should be aware of `gptq_scales` for consistency

## Skill v1.3.0 anti-pattern caught

**Hypothesis without empirical verification**:my `592b80c` "18× overflow"
hypothesis was numerically wrong(both paths use `max/127` for s_channel)。
Anti-pattern:**guessing magnitudes from formulae alone before running
the actual code**。

Per skill v1.3.0 rule:**when reasoning about pack output divergence,
RUN both packs and DIFF the outputs**(this entry's approach)before
hypothesizing magnitude problems。Empirical refutation in 50 LOC + 2
layers > theoretical overflow speculation。

## Status

- ✅ s_channel identical refuted "18× overflow" hypothesis
- ✅ s_group / qweight divergence localized at element level
- ✅ Pattern consistent across 2 layers
- ⏳ Kernel-side investigation(codex,`marlin_w4a8_kernel.cu`)
- ⏳ Single-layer naive vs gptq forward compare(codex)

## Cross-references

- e2e fail finding: `592b80c`
- pack_w4a8 GPTQ-aware: `12a54da`
- Diff script: `scripts/diff_pack_w4a8_naive_vs_gptq.py`
- Verify script: `scripts/verify_gptq_w4a8_repack_quality.py`
- Kernel: `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu`
- Codex audit kernel CLEAN(naive path):`01ace86`
- pack_w4a8 source: `scripts/quantize_qwen3_w4a8.py:93-145`

## Rule

When two pack modes produce visibly different output tensors but both
round-trip cleanly through their own Python unpack:**the kernel
(third party reader)is the divergence point**,not the math。Test by
running the kernel against BOTH pack outputs and comparing to FP
reference forward。Skill v1.3.0:**three-way verification(pack A,pack
B,kernel)is required for cross-mode quant validation**,not just
two-way A↔A round-trip。

## 2026-05-08 Update - Superseded by Converter Source Bug

The direct naive-vs-GPTQ pack diff remains factual, but it was not the
cause of the token-id-0 garbage. A follow-up kernel-side check showed
PR #31 W4A8 Marlin reads the GPTQ-aware packed tensors correctly for
single-layer projections. The end-to-end failure came from the W4A16
source checkpoint produced by `scripts/convert_gptq.py`: GPTQ `qzeros`
were decoded as stored zero-points instead of `(stored + 1)`.

With the corrected converter:
- W4A16 source no longer emits `!`/garbage and passes
  `test_greedy_solo_vs_concurrent`.
- W4A8 repacked from that source passes
  `test_w4a8_vs_bf16_token_diff` with 32/32 token match against BF16.

Rule refinement: pack-mode tensor diffs are not sufficient to blame the
kernel. Always first validate the upstream converted W4A16 source quality
before debugging a downstream W4A8 repack.
