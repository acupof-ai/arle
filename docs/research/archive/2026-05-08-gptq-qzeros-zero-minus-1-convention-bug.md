# GPTQ `qzeros` "zero - 1" convention bug — REAL root cause behind W4A16 marginal accuracy + W4A8 garbage

> Codex pane EOD+40 finding(2026-05-08):
>
> > 找到实际 bug 了:scripts/convert_gptq.py 解 GPTQ qzeros 少了 +1。
> > 官方 GPTQ 的 qzeros 全是 7,这是常见的 "zero point stored as zero-1"
> > 格式;脚本直接 subtract 7,把所有权重整体偏了一个 scale。难怪 W4A16
> > public GPTQ 和 W4A8 re-pack 都坏。
>
> Codex applied 1-line patch at `scripts/convert_gptq.py:54-56`:
> `zeros_unpacked = z_expanded.reshape(...) + 1`
>
> This fix likely supersedes Fix A(`163c8ee` clamp s ≤ 16)as the REAL
> root cause of W4A8 GPTQ greedy gate failure。Fix A may still be needed
> as a kernel safety bound but the calibration drift was secondary。

## Why this is hidden

GPTQ stores `zero_point - 1` in `qzeros` to fit `[0, 15]` 4-bit unsigned
when actual zero is 8(symmetric quant convention)。Per AutoGPTQ source
this is documented but not loud — many Marlin-port implementations miss it。

Symptoms:
- `qzeros` tensor when dumped shows all values = 7(actual zero is 8)
- Without `+1` correction:dequant uses 7 as zero,off-by-one for every
  weight element
- Effect:every weight magnitude multiplied/shifted by `~scale_per_group`
  worth of bias
- Cumulative through 36 layers → wrong logits → wrong tokens

## Empirical evidence(codex pane)

```
qzeros shape torch.Size([20, 512]) torch.int32 2004318071 2004318071
unpacked minmax 7 7 unique first [7]
counts [0, 0, 0, 0, 0, 0, 0, 81920, 0, 0, 0, 0, 0, 0, 0, 0]
```

All 81920 unpacked nibbles = 7。This is the AutoGPTQ "zero-1" pattern。
Without the `+1` fix,the converter computes:
```
weight_signed = weight_unpacked - zeros_expanded  # = w - 7
                                                  # but should be w - 8
```

So every weight is shifted by +1 quantization unit。Per group of 128
elements with scale ≈ max/7,this introduces ~max/7 ≈ 14% systematic
bias per element。

## Why I missed it

My audit `01ace86` checked:
- Pack matches PR #31 W4A8Layer ✓
- Kernel byte-identical ✓
- FFI dtypes match ✓
- Loader naming match ✓
- Activation quant convention match ✓

But assumed the GPTQ Int4 checkpoint was correctly decoded by
`convert_gptq.py`。Did NOT audit that script's qzeros handling against
AutoGPTQ source spec。

This is the **upstream-data-correctness blindspot**:audit treated the
GPTQ checkpoint as a trusted input,when in fact ARLE's parser had a
silent +1 off-by-one。

## Connection to Fix A(`163c8ee`)

Fix A clamps `s_group_stored ≤ 16`(kernel MAGIC_NUM bound)。The
`+1 qzeros` bug was creating wrong `s_group_stored` values too:
- Wrong zeros → wrong `(q - zero) * s` decoded weights
- pack_w4a8 GPTQ-aware uses `gptq_scales` directly,assumes weights
  are at integer multiples of those scales
- But weights were shifted by 1 zero-unit due to the +1 bug
- → re-pack with `gptq_scales` produces NaN-prone values

Fix A clamping bounds the kernel range,but the underlying weights are
still wrong-by-1 → garbage output regardless of clamp。

After codex's `convert_gptq.py +1` fix:
- Weights correctly decoded
- `gptq_scales` correctly aligned
- Re-pack via Fix A(or original GPTQ-aware)should produce correct output
- Likely greedy gate PASSES now

## Cumulative bug stack(post-discovery)

| Bug | Where | Fix | Effect |
|-----|-------|-----|--------|
| 1. Wrong-class perm(skip-8 vs 4-cons) | quantize_qwen3_w4a8.py | reverted | Perm pattern alignment(`3cee2f0`)|
| 2. Redundant `s_pack=s.t()` | quantize_qwen3_w4a8.py | removed | Broadcast index alignment |
| 3. scale_perm_single placement | quantize_qwen3_w4a8.py | applied AFTER division | s_channel layout |
| 4. naive max/7 not GPTQ scales | pack_w4a8 GPTQ-aware | gptq_scales kwarg | Calibration preservation |
| 5. MAGIC_NUM kernel bound | pack_w4a8 GPTQ branch | clamp s ≤ 16 | Kernel range constraint |
| **6. qzeros +1 missing** | **convert_gptq.py** | **+1 in decode** | **Zero-point alignment** ← **REAL ROOT CAUSE** |

Bugs 1-5 were real but secondary。Bug 6 was the upstream data
corruption that propagated through everything。Once #6 is fixed,#1-#5
may not even need their respective fixes for greedy gate to pass(though
#5 MAGIC_NUM bound is still a kernel safety constraint that should be
kept)。

## Methodology lesson

When investigating "checkpoint produces wrong output",**audit the
upstream parser FIRST** before any internal pack/kernel diff:
1. Dump `qweight` / `qzeros` / `scales` raw values
2. Compare against AutoGPTQ source code spec exactly
3. Verify each tensor convention(zero-stored-as-zero-minus-1,sym vs asym,
   scale magnitude convention)
4. Only then iterate on internal pack/kernel layers

ARLE's `convert_gptq.py` was an UPSTREAM-DATA parser that we trusted。
This trust was misplaced for ~1 year(silent W4A16 marginal accuracy)
and only surfaced when W4A8 re-pack made the bug catastrophic enough
to force investigation。

## New audit rule(adds to 4-rule list per `36830bf`)

**RULE 5: Audit upstream-data parsers BEFORE internal kernel logic**。
When porting a quant format,the parser that decodes upstream packed
tensors(qweight,qzeros,scales,zeros)is the FIRST suspect for
"output looks slightly off" symptoms。Verify:
- Bit-extraction order
- Sign extension(signed vs unsigned)
- Zero-point convention(stored as zero,or zero-1,or other)
- Scale magnitude convention(max/7,max/15,max/127,etc)
- g_idx interpretation(per-token vs per-group)

These are the "hidden contracts" that don't appear in the kernel and
don't appear in the pack — they're entirely in the parser → kernel
expectation chain。

## Cross-references

- `163c8ee` Fix A patch(MAGIC_NUM clamp)
- `b255828` MAGIC_NUM root cause
- `570e04e` MAGIC_NUM bound distribution(misleading per-bound 18.14 vs corrected 16)
- `6137a0e` MAGIC_NUM bound corrected to 16
- `492513c` pack divergence isolated
- `592b80c` W4A8 e2e FAIL
- `e753af7` Phase 1b script-level PASS(misleading — script consistency check passes but actual values were +1 wrong)
- `39237b9` "naive max-scale W4 too lossy" finding(may have been mostly the +1 bug,not actually noise compounding)
- `b7176d3` ~4% drift baseline(may also have been +1 bug)
- AutoGPTQ source: `gptq/quant.py` (zero stored as zero-1 convention)
- ARLE parser: `scripts/convert_gptq.py:53-56`

## Strategic implication

If codex's regenerated checkpoint passes greedy gate post-`+1` fix:
- Multiple briefs(`b7176d3`,`e753af7`,`39237b9`,`592b80c`)need
  re-interpretation:the underlying numbers were always corrupted by `+1`
- W4A8 + W4A16 production accuracy unblocked simultaneously
- The "naive max-scale 36-layer noise compounding" hypothesis may have
  been wrong — **it was actually the +1 zero-point bug all along**

Probability `+1` fix unblocks greedy gate:**~85%** based on:
- Direct math evidence(`(w - 7)` vs `(w - 8)` per element bias = ~14%)
- Cumulative through 36 layers explains all-`!` output character
- AutoGPTQ source spec match

If still fails post-`+1` fix:back to investigating MAGIC_NUM bound
clamp interaction or per-layer cumulative noise(but with correctly
decoded weights to test from)。

## Rule

When a quant pipeline has been "almost working" for a while(W4A16
"marginal accuracy" hand-wave),the most likely explanation is an
upstream parser silent corruption,not internal kernel/pack issues。
Iterating on the kernel/pack will produce diminishing returns until
the upstream parser is audited byte-for-byte against the source format
specification。
