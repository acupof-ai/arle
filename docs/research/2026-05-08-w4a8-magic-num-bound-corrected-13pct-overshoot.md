# W4A8 MAGIC_NUM bound CORRECTED to 16 — 13.6% groups affected,Fix A calibration loss ~1.4%

> Codex `163c8ee` corrected the MAGIC_NUM bound derivation:
> `s_group_stored ≤ 16` not `≤ 18.143`(my earlier `570e04e` analysis
> wrong)。Re-running the empirical diag with the correct bound shows
> **13.6% groups affected**(9× more than my earlier 1.5% estimate),
> with aggregate calibration loss ~1.4%。Fix A still wins by ~70× vs
> naive but the margin is tighter than I thought。

## Codex correction(`163c8ee`)

```
Bound derivation (corrected from initial 127/7=18.14 estimate):
  Kernel dequant_per_group MAGIC_NUM=0x6480 (FP16 1152.0) requires
  result = (q-8)*s_group_stored + 1152 ∈ [1024, 1280)
  → (q-8)*s_group_stored ∈ [-128, 128)
  → For q=0 (t0=-8):  s ≤ 16     (BINDING constraint)
  → For q=15 (t0=7):  s < 18.286
  → Effective bound:  s_group_stored ≤ 16
```

My `570e04e` analysis used 18.143 = 127/7,which was the q=15 case
ONLY。The q=0 case is tighter:`-8 * 16 = -128`,exactly at the
[-128, 128) range bound。So the binding constraint is `s ≤ 16`,not
`s ≤ 18.143`。

I had MAGIC_NUM as 1536 in my analysis;codex correctly identified it
as 1152(0x6480 in FP16 IEEE-754)。

## Re-run empirical results(corrected bound 16)

```
Found 252 GPTQ Linear layers
Kernel MAGIC_NUM bound: |s_group_stored| ≤ 16.000000

Total: 3,870,191 / 28,385,280 groups exceed bound (13.6345%)
Worst overshoot: max s_group_stored = 21.167 (bound 16.000, +32.3%)
```

Top affected layers(~19% groups affected):
- `layers.{29-32}.mlp.up_proj` 18.97-19.21%
- `layers.0.mlp.up_proj` 19.05%

Bottom layers(~5.3-5.6% affected):
- `layers.{14,15,17,18}.mlp.down_proj` 5.25-5.56%

Distribution(over correct bound 16):
- [16.00, 16.16):   1.05% — just over,minimal clamp impact
- [16.16, 16.80):   3.54% — ~1-5% clamp
- [16.80, 17.60):   3.14% — ~5-10% clamp
- [17.60, 19.20):   5.49% — ~10-20% clamp **largest mid-range**
- [19.20, 24.00):   0.42% — ~20-32% clamp **biggest impact per group**

## Updated calibration loss estimate

| Estimate | Value | Source |
|---|---|---|
| **Earlier(wrong bound 18.143)** | **0.075% aggregate** | `570e04e`(my mistake) |
| **Corrected(bound 16)** | **~1.36% aggregate** | this entry |
|   - 13.63% groups affected × ~10% mean overshoot | | |
| Fix D(naive max-scale,no GPTQ) | **100%** | (alternative path) |

**Fix A still strongly wins**:1.36% << 100% naive。Margin is ~73×
calibration preservation。But my earlier 1330× margin was wrong by 18×。

## Updated Fix A probability

Codex's `b255828` original estimate:**~85%**。My `570e04e`
update(based on wrong bound):~95%。

**Re-corrected estimate:~85-90%**:
- +5% for bounded distribution(no >32% outliers)
- −5% from ~1.36% calibration loss vs ~0.075% earlier estimate
- → net ~85-90%

Greedy gate result will reveal:
- **PASS**:Fix A LICENSED,~1.36% calibration loss is tolerable for
  W4A8 production
- **FAIL with improved character**:1.36% loss too much,need finer
  fix(per-group scale tier or GPTQ-aware kernel rewrite)
- **FAIL same garbage**:bound is wrong-derivation again(e.g., q=15
  case OR additional kernel constraint not yet found)

## Cross-references

- Codex Fix A apply: `163c8ee`(this commit clamps at 16,not 18.143)
- Codex MAGIC_NUM root cause: `b255828`
- My earlier wrong-bound diag: `570e04e`(based on 18.143)
- This corrected diag re-run with bound 16
- Diag script updated: `scripts/diag_gptq_w4a8_magic_num_bound.py`(this commit)

## Skill v1.3.0 anti-pattern caught(self-correction)

**Bound derivation from incomplete cases**:my `570e04e` checked only
q=15 case for the MAGIC_NUM constraint。Should have enumerated ALL
q ∈ [0, 15] cases(or at least both q=0 and q=15 boundary cases)
to find the BINDING constraint。

Per skill v1.3.0:**when deriving a numerical constraint from a
hardware/kernel trick,always check both numerical extremes**(min and
max representable q,or otherwise full input domain)。Single-extreme
analysis missed the q=0 binding case。

## Methodology lesson

Self-correction within Claude+codex collaboration:
- Claude:570e04e analyzed bound at 18.143(wrong,partial)
- Codex:163c8ee corrected to 16 with full derivation
- Claude:re-validates empirically(this entry)— 13.6% not 1.5%

This is the methodology working — empirical re-run after correction
catches the impact magnitude difference。Three-axis check
(my analysis → codex correction → my empirical) > single-axis。

## Status

- ✅ Codex Fix A patch applied(`163c8ee`)
- ✅ Empirical re-validated with correct bound(this entry)
- ⏳ Codex re-running greedy gate(25m31s building cargo test)
- ⏳ If PASS:wins entry + bench guidellm + master strategy update

## Rule

**When relaying empirical numbers to drive a fix decision**,verify
the bound/threshold derivation independently before publishing。
570e04e was 18× off in calibration loss estimate because the bound
was wrong。Re-derive from first principles + check ALL boundary cases
of the constraint(min and max input,not just one)。
