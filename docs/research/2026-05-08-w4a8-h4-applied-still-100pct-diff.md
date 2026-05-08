# W4A8 H4 (`s_pack = s.t()` removed) applied — still 100% diff,more layers remain

> Codex `592779a` confirmed H4: redundant `s_pack = s.t()` transpose
> mis-aligned broadcast during division。`945df02` promoted
> `/tmp/quantize_qwen3_w4a8.py` to repo `scripts/quantize_qwen3_w4a8.py`
> with H4 fix applied。
>
> Claude this tick:re-quantize Qwen3-4B with promoted script + run
> `test_w4a8_vs_bf16_token_diff`。**Result: still 100% diff,output
> character similar to H3+H3b "multilingual gibberish + code-like"
> state**。Real progress empirically but bug pattern is now 5+ layer。

## Test result

```
prompt: "The capital of France is" (max_tokens=32)

BF16 baseline: " Paris. The capital of Germany..."
                first token 12095 ("Paris")

W4A8 post-H4: "Compact prime Bootstrap|,\n half kick|\n并不说 Boot谷 kick
              primeстьobjectId|\n |\n lắm使人 scalar受primekickchilds
              并且 a closering primailsingUpdates"
              first token 98335 (different again)

Token diff: 100% (idx=0 onward)
```

## Output character progression(5 iterations)

| State | Output | Empirical character |
|---|---|---|
| Pre-fix(`81b6481`) | ".........11.1.11111111 baudaskan1..." | period/digit bias(weights near-zero) |
| H3 row only(`62f885d`) | " Gamb Cocktail zipトル..." | multilingual gibberish |
| H3+H3b(`03178cf`) | "ing bootstrap recipients prime bargaining indefinite/bootstrap natural odd 一主义者..." | **English-frag + code-like(closest qualitative)** |
| H3+H3b+H3c(`4dea952`) | "处分违舒适的婿 withStyle suppy umba..." | regressed multilingual + Chinese-Japanese |
| **H4 only(`945df02`,this run)** | **"Compact prime Bootstrap\| half kick 并不说 Boot 谷..."** | **multilingual + English + code mix(similar to H3+H3b)** |

H4 fix(`s_pack = s.t()` removed)corrected broadcast-alignment in
division but **did NOT recover Paris**。Output is qualitatively similar
to H3+H3b state — both have English/code-like + Chinese mixed。

## Interpretation

Per `01ace86` codex audit: kernel + linear.rs FFI wiring + weight_loader
naming all 0-diff vs PR #31。**Bug surface is 100% in
`scripts/quantize_qwen3_w4a8.py`**。

5 fix iterations across multiple bug layers:
- H3 row stride ✅(was W4A16-FP16 layout vs W4A8-INT8)
- H3b scale_perm_single missing ✅(was deleted before save)
- H3c scale_perm_single position ⚠(applied but order doesn't fully resolve)
- H4 `s_pack = s.t()` redundant transpose ✅(broadcast misalignment per codex)
- **Layer 5+:additional bug remaining**

Hypothesis space remaining(per codex `01ace86` audit):

1. **Tile permute lines 110-115 reshape order** — `w.permute((0, 2, 1, 3))`
   may need different
2. **Bit-packing stride `i::8`** in `q |= res_np[:, i::8] << (4*i)`
3. **`scale_perm` shape mismatch on s_group post other-fixes** — interaction
4. **PR #31 documentation gap** — codex copying source verbatim may have
   missed a test invocation difference
5. **Known-good checkpoint test**(audit Option 3)— bypass quant script
   entirely

## Skill methodology limit

Per anti-pattern #13:5 NULL eliminations real progress(narrowed bug
landscape from "kernel + wiring + script unknown" to "script-only with
4 layers fixed,1+ remaining")。But **iterative quant script tweaks
without unit-test isolation** is at methodology limit per codex audit Rule。

**Next-step recommended**:
1. Codex own — Option 3:try PR #31 reference Qwen-7B-W4A8 checkpoint
   (or any known-good W4A8 produced by reference impl)through ARLE
   Marlin path。If passes → bug 100% in script,not kernel。If fails →
   ARLE script may be missing kernel-specific quirk。
2. Codex own — Option 1:single-layer unit test with known input ×
   known output = bug pattern localizes layer

## Cross-references

- Codex H4 confirmed: [`2026-05-08-w4a8-bug-h4-confirmed-spack-redundant-transpose.md`](2026-05-08-w4a8-bug-h4-confirmed-spack-redundant-transpose.md) (`592779a`)
- Codex H4 fix promoted: `scripts/quantize_qwen3_w4a8.py` (`945df02`)
- Codex audit: [`2026-05-08-w4a8-kernel-and-wiring-audit-clean.md`](2026-05-08-w4a8-kernel-and-wiring-audit-clean.md) (`01ace86`)
- 5-iteration narrowing chain:
  - H3 row stride: `25391f3` `62f885d`
  - H3b scale_perm_single: `3479a87` `03178cf`
  - H3c order regressed: `d0f030b` `4dea952` reverted `06193eb`
  - H4 broadcast: `592779a` (this entry tests post-fix)
- Skill v1.3.0 anti-pattern #13: NULL elimination
- Failing test: `infer/tests/greedy_consistency.rs::test_w4a8_vs_bf16_token_diff`

## Rule

5 perm/quant-script bug layers found over 30+ commits via Claude+codex
collaborative iteration。Each fix produces qualitative output progression
but token diff persists 100% — production decode default REMAINS W4A16
Marlin until W4A8 quant correctness passes。

Methodology lesson:**when ≥4 fix iterations don't converge,non-iterative
escalation(unit test / known-good checkpoint)is the prescribed next
step**(codex audit Rule from `01ace86`)。Ad-hoc PR #31 source-diffing
has reached diminishing returns — empirical isolation needed。
