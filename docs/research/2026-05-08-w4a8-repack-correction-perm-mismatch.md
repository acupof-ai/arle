# W4A8 re-pack correction:`marlin_repack.py` perms ≠ W4A8 perms — use original `qweight` not `marlin_qweight`

> Continues `da19d71` Phase 0 reconnaissance(reframes M_quant Phase 1 from
> AutoGPTQ generate to "re-pack existing GPTQ checkpoint")。
>
> **Correction**:da19d71 claims "qweight bit layout IDENTICAL between
> W4A16-marlin and W4A8-marlin"。Direct code-grep finds this claim is
> **wrong**:`scripts/marlin_repack.py:32-37` uses the W4A16 **skip-8**
> row pattern,but W4A8 requires the **4-consecutive** pattern per
> `3cee2f0` retrospective + PR #31 W4A8Layer.
>
> The shortcut path needs adjustment:re-pack from the **original `qweight`
> [N, K/2] U8 tensor**(pre-Marlin GPTQ-calibrated weights),NOT from the
> `marlin_qweight` [K/16, N*16/8] I32 tensor。

## Code evidence

`scripts/marlin_repack.py:25-53`(W4A16 production path):
```python
def get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),         # ⛔ skip-8 pattern (W4A16, NOT W4A8)
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
```

`scripts/quantize_qwen3_w4a8.py:59-64`(W4A8 path,corrected per `3cee2f0`):
```python
for row in [
    4 * (i % 4),         # ✅ 4-consecutive pattern (W4A8)
    4 * (i % 4) + 1,
    4 * (i % 4) + 2,
    4 * (i % 4) + 3,
]:
```

→ `marlin_qweight` produced by `marlin_repack.py` uses **W4A16 byte layout**。
Loading directly into ARLE's `marlin_w4a8_kernel.cu`(verbatim from PR #31
W4A8Layer)would produce **garbage output** because the kernel expects
4-consecutive-pattern bytes。

## Why da19d71 was misled

da19d71 inspected tensor SHAPES:
- W4A16 `marlin_qweight`:[K/16, N*16/8] I32
- W4A8 `marlin_w4a8_qweight`:[K/16, N*16/8] I32

Identical shapes ✓。But shapes don't determine byte content。Within each
int32,the 8 nibbles are arranged according to the perm pattern,which
**differs** between W4A16 and W4A8 paths。

## Corrected Phase 1 path

Don't re-pack from `marlin_qweight`(W4A16 byte layout)。Instead,re-pack
from the **original `qweight` [N, K/2] U8 tensor**(pre-Marlin GPTQ
calibrated weights),which the existing GPTQ-Int4 checkpoint also stores:

```bash
$ inspect infer/models/Qwen3-4B-GPTQ-Int4-marlin/
*.qweight        shape=[N, K/2] dtype=U8     ← USE THIS (pre-Marlin)
*.scales         shape=[N, K/groupsize] dtype=BF16
*.marlin_qweight shape=[K/16, N*16/8] I32    ← W4A16-perm bytes (DON'T USE for W4A8)
*.marlin_scales  shape=[K/groupsize, N] F16  ← already permuted for W4A16
```

Re-pack flow:
```python
def repack_w4a16_to_w4a8(qweight_u8, scales_bf16, n, k, groupsize=128):
    """Re-pack GPTQ-calibrated W4A16 weights into ARLE W4A8 format.

    Uses the original (N, K/2) U8 qweight + BF16 scales from the GPTQ
    checkpoint, NOT the marlin_qweight which has W4A16 perms baked in.

    Output format matches scripts/quantize_qwen3_w4a8.py::pack_w4a8.
    """
    # 1. Decode qweight_u8 [N, K/2] → unsigned int4 weights [N, K] in [0, 15]
    lo = (qweight_u8 & 0x0F).to(torch.int32)
    hi = ((qweight_u8 >> 4) & 0x0F).to(torch.int32)
    w_int = torch.zeros(n, k, dtype=torch.int32)
    w_int[:, 0::2] = lo
    w_int[:, 1::2] = hi

    # 2. Convert to dequantized FP16 weights (using GPTQ scales)
    # GPTQ symmetric: zeros = 8 (per quantize_config sym=True with bias 8)
    # w_real = (q - 8) * scales_bf16
    scales_per_element = scales_bf16.repeat_interleave(groupsize, dim=1)  # [N, K]
    w_real = (w_int - 8).float() * scales_per_element.float()

    # 3. Re-pack via existing pack_w4a8 (uses W4A8 4-consecutive perms)
    from scripts.quantize_qwen3_w4a8 import pack_w4a8
    qweight, s_channel, s_group = pack_w4a8(w_real.to(torch.bfloat16))
    return qweight, s_channel, s_group
```

## Quality preservation note

Re-packing via `pack_w4a8(w_real)` runs naive max-scale W4 quantization
over GPTQ-calibrated weights。Since `w_real` lives at GPTQ-quant levels
already(integer multiples of GPTQ scale),and pack_w4a8 uses
`round(w / s)` with `s ≈ max(|w|)/7`:
- If GPTQ groupsize=128 matches our pack groupsize=128,scales align
- Max-abs of dequantized GPTQ weights = max-abs of original × scale
- Re-quantization SHOULD recover same integer levels(near-zero added noise)
- → Calibration preserved through re-pack

This is testable via the existing `scripts/diag_w4a8_pack_roundtrip.py`
diagnostic adapted to take pre-quantized GPTQ inputs。

## Updated Phase 1 estimate

- 50 LOC re-pack script(versus 150 LOC AutoGPTQ generate)
- 5-10 minute re-pack runtime per checkpoint(no calibration forward passes)
- Total Phase 1:0.5 day(versus 1 day)
- **Phase 1 net saving over AutoGPTQ-from-scratch**:0.5 day + 30-60 min
  GPU time

If re-pack works,total M_quant timeline becomes 2-2.5 days(versus
4-5 days original plan,2.5-3 days da19d71 estimate)。

## KILL criteria for re-pack path

- **If re-quantization noise > 5%** measured via diag_w4a8_pack_roundtrip
  on dequantized-GPTQ weights → quality regression,fall back to
  AutoGPTQ-direct or implement custom GPTQ-aware pack
- **If marlin_w4a8 kernel still produces token diff** with re-packed
  weights → bug in pack scale derivation,investigate per-group / per-
  channel scale split mismatch
- **If GPTQ-Int4 checkpoint lacks raw `qweight`** field(only
  `marlin_qweight`)→ revert to AutoGPTQ-direct path,no shortcut available

## Cross-references

- da19d71 Phase 0 reconnaissance: [`docs/research/...`]
- M_quant plan: [`662cbbb`](../plans/M_quant-autogptq-marlin-integration.md)
- W4A8 finding: [`39237b9`](2026-05-08-w4a8-naive-max-scale-too-lossy-need-calibration.md)
- W4A16 vs W4A8 perm distinction: [`3cee2f0`](2026-05-08-w4a8-methodology-retrospective-wrong-class.md)
- ARLE W4A16 path: `scripts/marlin_repack.py:25-53`(skip-8)
- ARLE W4A8 path: `scripts/quantize_qwen3_w4a8.py:50-81`(4-consecutive)
- Existing GPTQ checkpoint: `infer/models/Qwen3-4B-GPTQ-Int4-marlin/`

## Rule

When two related quant paths share **storage shapes** but differ in
**byte layout**(perm pattern),never assume "same shape = compatible
bytes"。Verify the perm pattern definition source(class hierarchy,
function signature,callsite)before claiming byte-compat。Tensor shape
comparison is necessary but **not sufficient** for serialization
interchange。
