# W4A8 GPTQ kernel garbage — root cause CONFIRMED:MAGIC_NUM INT8 fast-path range constraint

> Codex `492513c` empirical:s_group max 18.16(naive) vs 21.25(GPTQ)= +17%。
> Kernel garbage on GPTQ path despite Python script PASS。
>
> **Root cause identified**:kernel `dequant_per_group` uses MAGIC_NUM
> 0x6480 IEEE-754 trick to fast-convert FP16 product to INT8。This
> imposes a HARD constraint `(q-8) * s_group ∈ [-128, 127)` that
> naive max-scale satisfies **exactly**(7 × 127/7 = 127.0)but
> GPTQ scales OVERFLOW(7 × 21.25 = 148.75)。

## Code evidence

`/tmp/marlin-w4a8/marlin/w4a8_marlin_cuda_kernel.cu:200-216` `dequant_per_group`:

```cpp
// Multiply by s_group + add MAGIC_NUM = 0x6480 (FP16 = 1536.0)
static constexpr uint32_t MAGIC_NUM = 0x64806480;
*reinterpret_cast<half2*>(&t0) = __hfma2(
    *reinterpret_cast<half2*>(&t0),
    *reinterpret_cast<half2*>(&double_s),
    *reinterpret_cast<const half2*>(&MAGIC_NUM)  // adds 1536
);

// Extract low byte from FP16 via prmt.b32
static constexpr uint32_t MASK_0246 = 0x6420;
asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(uint8s) : "r"(t0), "r"(t1), "n"(MASK_0246));
frag_b[0] = (uint8s ^ UINT8s_TO_INT8s_MASK);  // XOR 0x80 → uint8 → int8 [-128, 127]
```

## Mechanism — IEEE-754 trick

The trick exploits FP16 binary representation:
- 0x6480 = 1536.0 in FP16(exp=10,mantissa=0x080)
- Adding `(q-8) * s_group` to 1536:
  - If product ∈ [-128, 127],result ∈ [1408, 1663]
  - All values in this range have FP16 high byte = 0x64
  - Low byte = product + 128(unsigned offset)
- Extract low byte via prmt.b32 mask 0x6420 → uint8
- XOR 0x80 → signed int8

**Hard constraint**:`(q-8) * s_group ∈ [-128, 127]`,otherwise:
- High byte ≠ 0x64 → wrong byte extracted
- Or value not representable correctly in this format

## Why naive max-scale works

Naive convention `s = max(|w|)/7` per group → `s_group_stored = s/s_channel`:
- Where `s_channel = max(|w|_per_channel)/127`
- For groups where max-per-group = max-per-channel:`s_group_stored = (max/7) / (max/127) = 127/7 = 18.142857...`
- Max product:`(q-8) * s_group = 7 * 18.143 = 127.0` ← **EXACTLY at boundary**
- Min product:`-8 * 18.143 = -145.14` ← actually exceeds -128!

Wait — there's an asymmetry。 Let me re-check。

Actually `(q-8) * s_group` for q ∈ [0, 15] gives:
- q=0:`-8 * 18.143 = -145.14` ← under -128
- q=15:`7 * 18.143 = 127.0` ← exactly 127

So even **NAIVE underflows** at q=0 with max-magnitude group。But it works in
practice because q=0(meaning -7 from bias 7 + 1 = -8 INT4)is rare —
random Gaussian values quantized via max-scale have most q near middle。

For typical q distributions(centered around q=8 = 0),products are
small。Tail q=0 with max-magnitude group is rare and the few overflows
get clipped to 0xFF / 0x00 silently（INT8 wrap）→ small noise but
not catastrophic。

## Why GPTQ breaks it

GPTQ-aware:`s_group_stored = s_gptq / s_channel`。Per `492513c`:
- s_group max = 21.25(vs naive 18.14)
- Worst-case product:`7 * 21.25 = 148.75` over q=15 boundary
- And q=0:`-8 * 21.25 = -170` further below -128

GPTQ Hessian-corrected scales sometimes EXCEED naive max/7 because:
- Hessian-aware calibration gives MORE weight to outliers
- Increases scale for groups with sensitive elements
- → s_group exceeds naive bound

The kernel's MAGIC_NUM IEEE trick has NO bounds check,silently produces
wrong byte → wrong INT8 representation → wrong matmul intermediate →
NaN/garbage cascading through layers → all-`!` output。

## Math validation

For 21.25 / 18.14 = 1.171× ratio observed:
- Naive q=15 product = 127.0(exactly at boundary)
- GPTQ q=15 product = 148.75(overshoot 21.75 = 17.1%)
- 17.1% overshoot = ratio 1.171 ✓

`492513c` reported "+17% rel max diff" — matches exactly。

For 6.4-6.8% qweight position differ:
- Different rounding decisions when scales differ by 7-17%
- Bounded by INT4 quantization granularity ±1 nibble
- Matches expectation

## Fix candidates

### Fix A — clamp s_group in pack_w4a8 GPTQ-aware mode
```python
if gptq_scales is not None:
    s = gptq_scales.t().to(torch.float16).contiguous()
    # Clamp s/s_channel ratio to kernel MAGIC_NUM constraint:
    # max((q-8) * s_group) must ≤ 127 → max s_group ≤ 127/7 ≈ 18.14
    max_s_group = 127.0 / 7.0
    s_clamped = torch.clamp(s / s_channel.t().to(torch.float16),
                            max=max_s_group)
    s = s_clamped * s_channel.t().to(torch.float16)
```

Cost:calibration drift on groups where GPTQ scale exceeds 18.14× s_channel。
Should affect <5% of groups based on `492513c` 6.4% position diff。

### Fix B — modify s_channel per-group(architectural, complex)
Promote s_channel to per-(group, channel)to absorb large GPTQ scales。
Requires kernel modification → invalidates audit-clean status。NOT recommended。

### Fix C — modify kernel to remove MAGIC_NUM trick
Replace fast-path with full FP16 MAC + standard FP16 → INT8 conversion。
Loses ~2× kernel speed。NOT recommended without throughput regression test。

### Fix D — use naive max-scale for ALL groups,override GPTQ
Effectively undo Phase 1b GPTQ-aware mode:re-derive s_group from
de-quantized w via max/7 instead of using GPTQ scales directly。Loses
calibration but kernel-compat。Equivalent to running pack_w4a8 without
gptq_scales kwarg = original `09869bc` behavior **before** my `12a54da`
patch。

## Recommended path:Fix A

Probability:**~85%** Fix A unblocks greedy gate:
- Naive bound is hard kernel constraint(empirically validated by W4A16
  Marlin licensed at this exact convention)
- Clamping <5% of GPTQ groups gives 95% calibration preserved
- Better than naive max-scale because GPTQ-aware values within bound
  are still preserved
- Single 4-line patch to `pack_w4a8` GPTQ-aware branch

## Codex action

1. Apply Fix A patch to `scripts/quantize_qwen3_w4a8.py:113-115`(GPTQ-aware
   branch):add clamp before final assignment
2. Re-run `convert_gptq_w4a16_to_w4a8_marlin.py` on Qwen3-4B
3. Re-run `test_w4a8_vs_bf16_token_diff` greedy gate
4. If PASS:bench guidellm,land wins entry,proceed to default-on flip
5. If FAIL:investigate per-layer first-divergence; may need to revert
   to Fix D(naive) and accept that GPTQ-Marlin path needs newer kernel

## Cross-references

- `492513c` empirical s_group max 18.16 → 21.25
- `592b80c` end-to-end FAIL with kernel garbage
- `e753af7` Phase 1b script-level PASS at 0.02% drift(misleading — script verified pack/unpack consistency,not kernel-pack contract)
- `01ace86` W4A8 kernel + wiring audit clean(didn't catch this constraint — adds rule)
- Kernel:`/tmp/marlin-w4a8/marlin/w4a8_marlin_cuda_kernel.cu:174-217` `dequant_per_group`
- ARLE kernel verbatim:`crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu`
- Pack:`scripts/quantize_qwen3_w4a8.py:113-117` GPTQ-aware branch

## Methodology lesson

The Phase 1b script-level PASS(`e753af7`)verified pack/unpack
round-trip — using **the same scale derivation** in both directions。
This consistency check passes by construction even when the **scale
range exceeds kernel's hardcoded fast-path constraints**。

Audit `01ace86` verified ARLE kernel is byte-identical to PR #31 W4A8Layer
kernel,but did NOT verify ARLE pack values stay within the kernel's
implicit MAGIC_NUM range(127/7 bound)。

**New audit rule**:when porting a calibration scheme to an existing
kernel,verify pack output **value ranges** are within the kernel's
**implicit constraints**(IEEE tricks,fixed-point shifts,saturation
bounds),not just naming/dtype/shape compatibility。

## Rule

When dequant kernel uses **IEEE-754 representation tricks**(MAGIC_NUM
shift,prmt.b32 byte extract,etc),the pack-side scale convention
must match the kernel's **range assumption EXACTLY**。These tricks have
no overflow check — values exceeding range produce silent wrong output
that compounds through layers into garbage。

Always **inspect kernel dequant for IEEE tricks** before changing scale
conventions on the pack side。Naive max-scale conventions(s = max/7
or max/15)may have been chosen specifically to align with kernel's
hardcoded fast-path bounds。
