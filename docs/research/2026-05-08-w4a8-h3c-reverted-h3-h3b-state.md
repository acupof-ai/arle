# W4A8 H3c reverted to H3+H3b state — methodology pivot per codex audit

> Per codex `01ace86` audit:**kernel + wiring + dtype 0-diff vs PR #31**;
> **bug 100% in quant script**(remaining ~3 candidate sites in `pack_w4a8`)。
> Codex recommended:**stop iterating script blind**;run single-layer
> unit test to isolate。
>
> H3c regression(`4dea952` output went BACK to multilingual gibberish)
> validated via codex audit interpretation:H3c was structurally aligned
> with PR #31 source diff,but interaction with another script-side bug
> exposed worse output。**H3+H3b state(`03178cf` English-frag output)
> remains "closest yet" empirically**。
>
> Operational action this tick:**revert H3c in `/tmp/quantize_qwen3_w4a8.py`**
> back to H3+H3b state for next iteration baseline。Per audit Option 2
> recommendation。

## Operational changes(`/tmp/` untracked,no git diff)

```python
# Reverted /tmp/quantize_qwen3_w4a8.py to H3+H3b state:
#
# 1. Re-applied scale_perm_single permutation BEFORE division (line ~94,
#    H3b place) — was the post-`03178cf` state which produced "English-
#    frag + code-like" output (closest qualitative state observed).
#
# 2. Removed the post-division scale_perm_single permutation (added at
#    H3c, regressed output to multilingual mix per `4dea952`).
#
# Script now matches commit-state at `03178cf` (post-H3b applied).
```

## Codex audit synthesis(post-iteration limit)

Per `01ace86` audit:

| Layer | Status | Confidence |
|---|---|---|
| ARLE Marlin kernel C source | 0-diff vs PR #31(961 LOC + 26-line FFI wrapper) | 100% |
| `linear.rs::run_marlin_w4a8_linear` FFI call | arg ordering + dtypes match PR #31 signature | 100% |
| `weight_loader.rs:669-671` tensor naming | matches `marlin_w4a8_qweight/s_channel/s_group` script writes | 100% |
| **`/tmp/quantize_qwen3_w4a8.py`** | **bug surface concentrated here** | **bug 100% in script** |

Remaining script-side candidate sites(per `01ace86` + `4dea952` synthesis):

1. **Tile permute lines 112-115**(reshape order):
   ```python
   tile = 16
   w = w.reshape((k // tile, tile, n // tile, tile))
   w = w.permute((0, 2, 1, 3)).reshape((k // tile, n * tile))
   res = w.reshape((-1, perm.numel()))[:, perm].reshape(w.shape)
   ```
2. **Bit-packing stride line 117** `q |= res_np[:, i::8] << (4 * i)` — PR #31 marked OK earlier but worth re-verify post-row-fix
3. **scale_perm vs scale_perm_single timing on s_group**(line 108-109)— may need different post row-fix

## Methodology pivot — Option 1 unit test next

Per audit "stop iterating script blind",next-step recommendation is
**Option 1:single-layer unit test**:

- Pick one `q_proj` weight from layer 0 of Qwen3-4B BF16
- Pack with current(H3+H3b)script → INT4 + scales
- Run through ARLE `run_marlin_w4a8_linear` with known input vector
- Compare output to BF16 reference(matmul with full precision)
- Element-wise diff localizes the bug:
  - "first 8 channels right,others wrong" → tile permute issue
  - "every 4th channel right" → bit-packing stride issue
  - "magnitude right but wrong N-position" → scale_perm reshape order
  - "all wrong by constant factor" → s_channel permutation issue

Codex own / Claude pickup TBD。LOC ~50 Rust test or ~50 Python via PyO3。

## Skill methodology validation

Per anti-pattern #13:NULL elimination still real progress(narrowed 5→1.5→3-layer pattern,then audit confirmed kernel+wiring clean = bug surface from 5 candidates → ~3 candidate script sites)。

The 4-iteration-without-convergence pattern triggered codex's audit
escalation,exactly as audit's Rule prescribes:**3rd iteration without
convergence = methodology limit signal**,escalate to non-iterative
diagnostic(unit test / known-good reference checkpoint)。

## Cross-references

- Codex audit: [`2026-05-08-w4a8-kernel-and-wiring-audit-clean.md`](2026-05-08-w4a8-kernel-and-wiring-audit-clean.md) (`01ace86`)
- H3c applied still wrong: [`2026-05-08-w4a8-h3c-applied-still-wrong.md`](2026-05-08-w4a8-h3c-applied-still-wrong.md) (`4dea952`)
- H3+H3b state baseline: [`2026-05-08-w4a8-h3b-fix-applied-still-partial.md`](2026-05-08-w4a8-h3b-fix-applied-still-partial.md) (`03178cf`)
- Skill v1.3.0 anti-pattern #13: NULL elimination
- ARLE quant script(reverted to H3+H3b): `/tmp/quantize_qwen3_w4a8.py`(untracked)

## Rule

When kernel + wiring + dtype audit comes back CLEAN(0-diff vs reference),
**stop iterating the script side**;the remaining bug is a script-side
order-of-operations issue but iterative-without-isolation can not converge
on it。Single-layer unit test(known input → element-wise compare to BF16
reference)is the prescribed next step,not more PR-source-diffing。

H3c "structurally aligned with PR #31 line-by-line diff" but produced
WORSE output empirically — proving that line-by-line equivalence is
necessary but not sufficient when ops have positional coupling。Empirical
gate(unit test)is the only way past the methodology limit。
