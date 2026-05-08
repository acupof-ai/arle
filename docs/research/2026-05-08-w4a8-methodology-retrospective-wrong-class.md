# W4A8 5-iteration retrospective — H3 was wrong-class identification,user/codex caught & corrected

> Continues `0be5967`(diag confirms pack consistent at production shape)+
> user/codex revert of H3 row pattern in `scripts/quantize_qwen3_w4a8.py`。
>
> **Major methodology error**:my `25391f3` H3 brief compared ARLE's
> `get_perms()` against PR #31 **top-level `_get_perms()` at line 58**,
> not `W4A8Layer._get_perms()` at line 231。These use DIFFERENT row
> patterns。ARLE's original 4-consecutive WAS correct for W4A8;**my
> "H3 fix" introduced a bug**。User/codex caught it and reverted。

## PR #31 has TWO classes with DIFFERENT perms

`/tmp/marlin-w4a8/marlin/__init__.py`:

**`Layer` top-level** (line 89, used for non-W4A8 Marlin):
- `_get_perms()` at line 58:**skip-8 pattern**`[2k, 2k+1, 2(k+4), 2(k+4)+1]`
- Used for plain W4A16 layout

**`W4A8Layer`** (line 160, used for W4A8 specifically):
- `_get_perms()` at line 231:**4-consecutive pattern**`[4k, 4k+1, 4k+2, 4k+3]`
- This is what ARLE quantize_qwen3_w4a8.py should match

```python
# W4A8Layer._get_perms() line 237-242 (CORRECT for W4A8)
for row in [
    4 * (i % 4),
    4 * (i % 4) + 1,
    4 * (i % 4) + 2,
    4 * (i % 4) + 3
]:
```

## What went wrong in H3

`25391f3` H3 brief compared ARLE row pattern `[4k, 4k+1, 4k+2, 4k+3]`
against PR #31 top-level `Layer._get_perms()` `[2k, 2k+1, 2(k+4), 2(k+4)+1]`,
concluding ARLE's 4-consecutive was wrong。Reasoning was about INT8 vs FP16
fragment layout — superficially plausible BUT the actual upstream W4A8
class uses 4-consecutive。

I should have checked which class the kernel `marlin_w4a8_kernel.cu`
expects perms from。The kernel matches `W4A8Layer` since:
- Both use `_get_perms()` defined in `W4A8Layer` instance
- Both interlock through `Layer.pack` invocation paired with the right class
- The kernel's PTX `mma.sync.s32.s8.s8.s32` works with `W4A8Layer`'s
  perm convention

## Methodology lesson

When porting kernel code from upstream:
1. **Identify the EXACT class hierarchy** before comparing perm functions
2. Distinguish between "module-level helper" and "instance method" with
   the same name(`_get_perms`)
3. Trace **which class the kernel signature is paired with** — not all
   methods named `_get_perms` are interchangeable
4. Run round-trip diagnostic at **production-relevant shape**(e.g.
   k=2560 vs k=128 edge case)before making concrete claims

## Current canonical state(scripts/quantize_qwen3_w4a8.py)

After 5 iterations + user/codex correction,canonical pack:
1. **4-consecutive row pattern**(W4A8Layer correct,reverted from "H3 fix")
2. **H3b applied**:`scale_perm_single` applied to `s_channel`
3. **H3c reinstated**:`s_channel` permutation AFTER s_group division
4. **H4 applied**:no `s_pack = s.t()` redundant transpose
5. **Inline comments** document W4A8Layer vs Layer class distinction

Per `0be5967`:diagnostic on production shape (256, 2560) PASSES。Pack
is internally consistent for production layers。

## What this means for greedy_consistency 100% diff

If pack is now correctly matching PR #31 W4A8Layer + audit `01ace86`
confirmed kernel + wiring 0-diff,but `greedy_consistency` still 100%
diff,then bug must be in:

A. **Activation INT8 quantizer** (`w4a8_activation_quant.cu`):
   Maybe per-token scale convention or storage layout differs from what
   kernel expects when paired with W4A8Layer's perm。

B. **Storage byte interpretation**:Maybe ARLE writes qweight to
   safetensors with int32 storage but kernel reads as int4 packed in
   some other endian/order。

C. **m dimension padding**:Maybe ARLE doesn't pad M to 16-row boundary
   but kernel assumes it does。

D. **Kernel autotune dispatch**:`thread_k=-1, thread_n=-1` defaults
   might differ between PR #31 testbench and ARLE invocation context。

Codex action (currently Working per pane):continue investigating B / C /
D paths。If empirical signal continues,extract one Linear layer from
quantized Qwen3-4B,run kernel directly on known input,compare to
PR #31 reference output。

## Probability re-estimate

After H3 correction:
- P(pack still has bug)= 15%(round-trip passes at production shape but
  may have subtle issue we missed at non-prod edge)
- P(activation quantizer mismatch)= 35%
- P(storage byte interpretation)= 20%
- P(kernel/wiring subtle issue audit missed)= 20%
- P(other)= 10%

## Cross-references

- Diag confirms pack consistent at prod: [`0be5967`](2026-05-08-w4a8-pack-roundtrip-diag-confirms-broken.md)(but actually says PASS at prod)
- Diag tool: `scripts/diag_w4a8_pack_roundtrip.py`(`ab43959`)
- Canonical pack: `scripts/quantize_qwen3_w4a8.py`(W4A8Layer-matching)
- Audit clean: [`01ace86`](2026-05-08-w4a8-kernel-and-wiring-audit-clean.md)
- W4A8 garbage gate: `81b6481`
- PR #31 W4A8Layer: `/tmp/marlin-w4a8/marlin/__init__.py:160-261`
- PR #31 top-level Layer: `/tmp/marlin-w4a8/marlin/__init__.py:89-323`

## Rule

Before drawing conclusions from upstream-vs-local code diff:
- **Identify which class/function defines the contract**(perm,layout,
  kernel binding)
- Verify the LOCAL code matches that SPECIFIC contract,not a
  similarly-named alternative
- For kernels with multiple per-class implementations,trace **which
  pack path is paired with the kernel signature**

Failing to do this leads to "fix" introductions that revert correct
code,prolonging investigation and confusing the elimination chain。
