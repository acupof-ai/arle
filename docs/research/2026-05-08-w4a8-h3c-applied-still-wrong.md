# W4A8 H3c applied — still 100% diff,4th-layer bug remains

> Applied codex `d0f030b` H3c fix(scale_perm_single AFTER division per PR
> #31)+ re-quantize + re-test。Output 100% diff persists,but qualitative
> character changed AGAIN。Pattern suggests **4th-layer bug** OR H3c
> interaction with another bug。

## Setup

Applied H3c patch to `/tmp/quantize_qwen3_w4a8.py`:

```python
# Removed premature s_channel permutation (line 90-95 raw)
# Kept division s_work / raw s_channel (line 106)
# After s_group permutation (line 108-109), apply scale_perm_single to s_channel:
s_channel = s_channel.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
s_channel = s_channel.reshape((-1, n)).contiguous()
```

Re-quantized + re-tested。

## Result

```
prompt: "The capital of France is" (max_tokens=32)
BF16:  " Paris. The capital of Germany..."

W4A8 progression:
(1) Pre-fix (81b6481):    ".........11.1..." (period bias)
(2) Post-H3 row:          " Gamb Cocktail zipトル..." (multilingual)
(3) Post-H3b:             "ing bootstrap recipients prime bargaining
                          indefinite/bootstrap natural odd 一主义者..."
                          (English-frag + code-like — CLOSEST so far)
(4) Post-H3c (this run):  "处分违舒适的婿 withStyle suppy umba 什么呢
                          ...続きを読む ras/MIT pics 舣 ynos 作业..."
                          (back to multilingual mix,Chinese/Japanese
                          dominant)
```

Token diff still 100% in all states。

## Qualitative pattern observation

| State | Output character | Distance to "Paris" |
|---|---|---|
| Pre | period spam | farthest |
| H3 | multilingual | far |
| H3+H3b | English-frag + code | closest seen |
| **H3+H3b+H3c** | **multilingual + Chinese+Japanese** | **regressed?** |

H3c was supposed to be structurally correct per direct PR #31 source diff
(`d0f030b` 85% confidence)。But output character moved AWAY from English
toward more multilingual mix。

## Possible interpretations

1. **H3c is correct,but interacts with another bug**:
   - PR #31 line-by-line correct now,but `pack_w4a8` lines 112-115 (final
     tile permutation reshape)or bit-packing(line 117 `q |= res_np[:, i::8] << (4*i)`)still differs
   - H3c "correctness" exposes the ANOTHER bug that was masked by H3b's
     incorrect-but-coincidentally-aligned permutation

2. **H3c hypothesis was wrong**:
   - Maybe ARLE's Marlin kernel actually expects PRE-permuted s_channel
     (different from PR #31 reference)
   - In that case H3b was actually closer to correct
   - Codex's source-diff analysis missed an ARLE-specific kernel quirk

3. **Qualitative metric is not monotonic**:
   - Output character differences across iterations may be noise
   - Token-level greedy outputs at extreme quant could randomly land in
     different vocab regions even with similar magnitude error
   - "English-frag" ≠ "closer to Paris" semantically

## Cumulative state — 4-layer perm bug pattern likely

| Layer | Bug | Status |
|---|---|---|
| 1. Per-thread row stride | `[4k]` → `[2k skip-8]` | ✅ FIXED H3 |
| 2. Per-channel scale missing | `del scale_perm_single` removed | ✅ FIXED H3b |
| 3. Per-channel scale order | apply AFTER division | ⚠ APPLIED H3c (output regressed?) |
| 4. ??? | TBD | ❌ |

Suspect 4 candidates:
- A. **Tile permute lines 112-115** — `w.permute((0, 2, 1, 3)).reshape((k // tile, n * tile))` order
- B. **Final tile interleave `perm`** — `res = w.reshape((-1, perm.numel()))[:, perm].reshape(w.shape)`
- C. **Bit-packing stride** — `q |= res_np[:, i::8] << (4 * i)` 8-stride
- D. **weight_loader.rs:663-715** — tensor naming OR scale loading bytes assumption mismatch with pack
- E. **ARLE kernel internal** — actual kernel does NOT match PR #31 expectations (cherry-pick mismatch)

## Codex action recommended

1. **Verify H3c regression is real** — run again to rule out bench noise (single test, σ unknown)
2. **Compare lines 112-115 verbatim** with PR #31 `Layer.pack` lines 304-310 (tile permute / final interleave)
3. **Compare line 117 bit-packing** with PR #31 (should be identical — codex marked OK earlier)
4. **OR revert to H3+H3b state** (commit `03178cf`) which produced English-frag output (closest qualitative state) and investigate from there

If H3c is ARLE-incompatible (case 2 above), H3+H3b stays as production
quantize script while codex investigates kernel-side mismatch.

## Probability re-estimate

After 3 fix attempts (H3 row, H3b scale_perm_single, H3c order):
- Token diff 100% sustained — quant accuracy stuck
- Output character non-monotonic — methodology limit reached
- Need either: (a) deeper kernel-side audit, or (b) try H3c revert + alternate fix path

**Probability classical iteration-fix gets to passing test: ~30%**
(declining because each fix narrows space but doesn't converge)

## Skill methodology lesson

Per anti-pattern #13 (NULL = real elimination): 4 iterations is approaching
the limit of "iterative narrowing without external reference". Without
reading the ARLE Marlin kernel C++ in detail (codex own substrate), Claude
cannot independently distinguish:
- "Implementation matches PR #31 reference but kernel expects something else"
- "Implementation deviates from reference in a missed way"

Recommend codex action: **direct kernel-internal audit** of how `s2` and `s3`
are consumed in `dequant_per_group` + downstream ops (`marlin_w4a8_kernel.cu`).
The real test: compute kernel-internal recovered weight vs BF16 reference
on a single linear layer (unit test with known input).

## Cross-references

- H3 row stride: [`2026-05-08-w4a8-bug-h3-confirmed-perms-row-stride.md`](2026-05-08-w4a8-bug-h3-confirmed-perms-row-stride.md) (`25391f3`)
- H3b scale_perm_single missing: [`2026-05-08-w4a8-bug-h3b-confirmed-scale-perm-single-deleted.md`](2026-05-08-w4a8-bug-h3b-confirmed-scale-perm-single-deleted.md) (`3479a87`)
- H3 row-fix partial: [`2026-05-08-w4a8-h3-row-fix-partial.md`](2026-05-08-w4a8-h3-row-fix-partial.md) (`62f885d`)
- H3b applied still partial: [`2026-05-08-w4a8-h3b-fix-applied-still-partial.md`](2026-05-08-w4a8-h3b-fix-applied-still-partial.md) (`03178cf`)
- H3c source diff: [`2026-05-08-w4a8-bug-h3c-confirmed-permute-before-divide.md`](2026-05-08-w4a8-bug-h3c-confirmed-permute-before-divide.md) (`d0f030b`)
- W4A8 garbage gate: [`../experience/errors/2026-05-08-w4a8-quantize-broken-100pct-token-diff.md`](../experience/errors/2026-05-08-w4a8-quantize-broken-100pct-token-diff.md) (`81b6481`)
- ARLE script (current state with H3 + H3b + H3c): `/tmp/quantize_qwen3_w4a8.py`
