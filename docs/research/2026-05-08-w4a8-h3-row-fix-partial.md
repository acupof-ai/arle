# W4A8 H3 row-stride fix PARTIAL — output magnitude corrected, semantic still wrong

> Continues elimination chain through `25391f3` H3 confirmed → 4-line patch
> applied → re-quantize → re-test. **Result: 100% token diff persists**, but
> qualitative output character changed from period-spam to real-text
> gibberish — H3 row pattern was real but is **not the only bug**.

## Setup

Applied the codex-verified 4-line patch to `/tmp/quantize_qwen3_w4a8.py`:

```python
# Before (W4A16 FP16 32-byte/thread layout):
for row in [4*(i%4), 4*(i%4)+1, 4*(i%4)+2, 4*(i%4)+3]:

# After (W4A8 INT8 16-byte/thread layout, per PR #31):
for row in [2*(i%4), 2*(i%4)+1, 2*(i%4+4), 2*(i%4+4)+1]:
```

Re-quantized `infer/models/Qwen3-4B → Qwen3-4B-W4A8-marlin` (252 linear
tensors, 2.6 GB safetensors). Removed stale `model.safetensors.index.json`
(multi-shard reference from BF16 source). ARLE loaded checkpoint OK.

Ran `cargo test test_w4a8_vs_bf16_token_diff` after rebuild.

## Result

```
prompt: "The capital of France is"
max_tokens: 32

BF16 baseline:
  " Paris. The capital of Germany is Berlin. The capital of Italy is Rome..."
  first token: 12095 ("Paris")

W4A8 PRE-fix (`81b6481`):
  ".........11.1.11111111 baudaskan1 baud111askan11"
  first token: 13 (".")

W4A8 POST-fix (this run):
  " Gamb Cocktail zipトル Turbobsites+xmlval dephenperate.zip
   castingtraitsintendoeah指拜登国ensis fraction tur烦恼ondacticismobé
   ogi pastdet hope-fi"
  first token: 66789 (different token, mid/high vocab range)

Token diff vs BF16: 100% (idx=0 onward)
```

## Diagnostic — qualitative output changed

| Property | Pre-fix | Post-fix | Interpretation |
|---|---|---|---|
| Token magnitude bias | extreme (period/digit) | spread across vocab | weights now in correct order of magnitude |
| Token semantic | unrelated bias | unrelated real words | weights placed in approximately right thread fragments, but cross-tile positioning still wrong |
| Multilingual mix | no (English bias) | yes (en/jp/zh) | embedding lookup OK, GEMM partial-correct |

**Interpretation**: H3 row pattern was a real bug, but **not THE bug**. Fix
restores per-thread fragment ordering (so weights map to correct mma lanes
within a tile), but tile-level positioning is still off → cross-tile mma
sums produce semantically-wrong logits → wrong tokens.

## Remaining suspects (post-row-fix)

The `get_perms()` function has more parts that codex identified as
"identical to PR #31" but may NOT actually be:

1. **`256 * j` block stride** in inner loop:
   ```python
   for j in range(4):
       perm.extend([p + 256 * j for p in perm1])
   ```
   PR #31 reference also uses `256 * j` per the codex comparison — should
   be correct. But "identical syntactically" ≠ "identical semantically"
   if the surrounding `perm1` shape changed (which it did, post row-fix).

2. **`interleave` array `[0, 2, 4, 6, 1, 3, 5, 7]`**:
   Same in both per codex `25391f3`. But INT8 mma may need different.

3. **Reshape `(-1, 8)` then `[:, interleave]` then `ravel()`**:
   The 8-column structure may correspond to a different unit count post
   row-fix.

4. **`scale_perm` / `scale_perm_single`** — codex marked `✅ identical to
   PR #31`. Worth re-checking with row-fix's new perm shape.

5. **Pack function `pack_w4a8()` byte packing** — `q |= res_np[:, i::8] << (4 * i)`
   may need different for INT8 layout.

## Codex action recommended

Compare ALL of `pack_w4a8()` (lines 73-111 of `/tmp/quantize_qwen3_w4a8.py`)
with PR #31 reference, not just `get_perms()`. The fact that output went
from period-spam to multilingual gibberish suggests the per-tile ordering
is mostly right but cross-tile or scale ordering is still off.

Specific suspect lines:
- Line 87: `reshaped = ref.reshape(k // groupsize, groupsize, n)` — group
  reshape; PR #31 uses different?
- Line 91: `w = ref.reshape((-1, groupsize, n)).permute(1, 0, 2).reshape((groupsize, -1))`
  — permute order; PR #31 may differ
- Line 102-105: tile + perm application
- Line 108-110: bit packing into uint32

## Probability re-estimate

After row-fix partial success:
- H3 (perms) was the right HYPOTHESIS but only partially fixed
- Remaining bug likely in same `pack_w4a8()` function — different sub-step
- Probability "next-fix works": ~50-60% (we know we're in the right area,
  but more places to check)

## Skill methodology validation

Per anti-pattern #13: NULL-or-partial result is **also real elimination**.
Pre-fix output character (period spam) eliminated. Post-fix output character
(real words) remains. That qualitative character change IS evidence — narrows
the search to "tile/scale-level mis-positioning, not per-thread bit-level".

Without methodology, "test still fails 100%, gave up" would lose the
qualitative information. With methodology, partial-fix is real progress.

## Cross-references

- H3 confirmed: [`2026-05-08-w4a8-bug-h3-confirmed-perms-row-stride.md`](2026-05-08-w4a8-bug-h3-confirmed-perms-row-stride.md) (`25391f3`)
- H3 mechanism: [`2026-05-08-w4a8-bug-h3-mechanism.md`](2026-05-08-w4a8-bug-h3-mechanism.md) (`e3ca4d8`)
- W4A8 garbage gate: [`../experience/errors/2026-05-08-w4a8-quantize-broken-100pct-token-diff.md`](../experience/errors/2026-05-08-w4a8-quantize-broken-100pct-token-diff.md) (`81b6481`)
- PR #31 reference (per codex): see `25391f3`
- Failing test: `infer/tests/greedy_consistency.rs::test_w4a8_vs_bf16_token_diff`
- Re-quantize log: `/tmp/w4a8-requantize.log`
