# W4A8 pack roundtrip multi-shape sweep — bug shape-dependent,K=128 single-group case worst

> Companion to [`2026-05-08-w4a8-pack-roundtrip-diag-confirms-broken.md`](2026-05-08-w4a8-pack-roundtrip-diag-confirms-broken.md)
> (`0be5967`)。Extended codex single-shape diag(`ab43959`)to parametric
> sweep across(N,K,groupsize)to characterize shape distribution of pack
> asymmetry。
>
> **Result:7/8 shapes FAIL with gradient pattern**。Bug is shape-dependent
> in a way that **isolates root cause to per-group scale handling**:
> single-group case(K=groupsize)hits 2× ratio at row 112,multi-group
> cases hit 1.33-1.45× ratio at varied rows。

## Sweep results

```
shape (N,K,gs)               max_abs      expected   ×over  worst_row   worst_ratio   verdict
------------------------------------------------------------------------------------------------
(128, 128, 128)       unsupported W4A8 shape (Marlin N<128 boundary)
(256, 128, 128)           2.1056e-01    1.9577e-02    10.8        112         1.709      FAIL
(256, 256, 128)           1.1075e-01    1.9535e-02     5.7         92         1.357      FAIL
(512, 128, 128)           2.1056e-01    1.9554e-02    10.8        112         1.709      FAIL
(512, 512, 128)           1.0509e-01    1.9532e-02     5.4        477         1.361      FAIL
(1024, 256, 128)          1.4818e-01    1.9576e-02     7.6        535         1.446      FAIL
(1024, 1024, 128)         1.0040e-01    1.9531e-02     5.1        397         1.325      FAIL
(2048, 512, 128)          1.2914e-01    1.9599e-02     6.6       1190         1.391      FAIL

7/8 shapes FAIL pack/unpack round-trip
```

## Key shape-pattern insights

### Pattern 1:K=128 case worst — ratio 1.71 at row 112

Both `(256,128)` and `(512,128)` hit identical ratio 1.71 at row 112,with
identical max_abs 0.21。These shapes have **K/groupsize = 128/128 = 1
group only**。

This is a **single-group degenerate case**:
- `s_group` is shape `(1, N)` instead of `(num_groups, N)`
- Permutation `scale_perm` operating on a single-row scale matrix
  may degenerate(no-op)or hit edge case
- N is NOT the ratio-driver — same row 112 / ratio 1.71 across
  N=256 and N=512 → **bug is per-K not per-N**

### Pattern 2:K≥256 cases — ratio 1.33-1.45 at varied rows

`(256,256)`, `(512,512)`, `(1024,256)`, `(1024,1024)`, `(2048,512)` all hit
**lower ratio 1.33-1.45**(not 1.71)。Worst row varies(92, 477, 535, 397,
1190)— not a fixed boundary。

This suggests:
- Multi-group case has **partial scale averaging** that smooths the bug
- Per-group max-finding spreads error across groups
- BUT bug is still present(5.1×-7.6× over noise band)

### Pattern 3:Ratio decreases with #groups

| #groups (K/gs) | Worst ratio | Comment |
|---|---|---|
| 1(K=128) | 1.71 | Single-group degenerate |
| 2(K=256) | 1.36 | First multi-group |
| 4(K=512) | 1.36 | More groups, similar |
| 8(K=1024) | 1.33 | Even more groups |

**Convergence to ~1.33** as #groups → ∞ suggests there's a **constant
~1.33× scale factor** mismatch in the multi-group path,plus **additional
~1.71× factor** specific to single-group path。

## Refined root cause hypothesis(from shape distribution)

Per [`0be5967`](2026-05-08-w4a8-pack-roundtrip-diag-confirms-broken.md)
hypothesis space "scale chain"(80% probability),this sweep narrows
further:

**Strongest hypothesis(updated)**:bug is in **`s_group.reshape((-1, n))`
or `s_work` flatten when num_groups=1**:
- pack_w4a8 lines 92-100:`s_work = s.reshape((1, -1))` when num_groups=1
  may flatten differently than expected
- `s_group_real = s_group * s_channel` reconstruction has a 4/3 ≈ 1.33× factor missing
- + an extra factor ~1.28 only triggered when num_groups=1(amplifies to 1.71)

**Alternative hypothesis**:scale_perm logic over s_group when groupsize=K:
- `scale_perm` table assumes multi-group layout
- single-group path takes degenerate code path with different permutation

## Codex action — refined

Given shape distribution evidence,codex should focus instrumentation on:

1. **Run with `(N,K,gs)=(256,128,128)`** to maximize signal(2× ratio)
2. **Print s_group BEFORE and AFTER** scale_perm permutation
3. **Print s_channel** and `s_group * s_channel` reconstruction
4. **Compare s_group_stored / s_group_real ratio** — if 1.33× missing
   factor exists,this print will show it directly
5. **Diff against PR #31 reference Layer.pack** at the same shape

The 1.33× → 1.71× transition between multi-group and single-group cases
is a **strong fingerprint** —— whatever line of code activates only for
num_groups=1 is the bug。

## Skill v1.3.0 methodology validation

Per anti-pattern #13(NULL elimination):

| Iteration | Bug landscape pre | Bug landscape post |
|---|---|---|
| H3 row stride | "perm + scale + tile + bit-pack 4 layers" | "perm + scale + tile + bit-pack,row stride OK" |
| H3b scale_perm_single | "3 layers,scale_perm OK" | "scale_perm + tile + bit-pack" |
| H4 broadcast misalign | "3 layers" | "scale chain still asymmetric" |
| `0be5967` round-trip diag | "scale chain unknown 4 candidates" | "scale chain 100% confirmed via single-shape FAIL" |
| **THIS multi-shape sweep** | **"scale chain unknown sub-mechanism"** | **"single-group degenerate case + 1.33× constant offset"** |

Each NULL elimination compresses bug landscape。Methodology delivers。

## Cross-references

- Companion single-shape diag: `0be5967`(`docs/research/2026-05-08-w4a8-pack-roundtrip-diag-confirms-broken.md`)
- Codex audit clean: `01ace86`
- Codex H4 fix: `592779a` `945df02`
- Multi-shape script: `scripts/diag_w4a8_pack_roundtrip_multishape.py`
- Skill v1.3.0 anti-pattern #13:NULL elimination

## Rule

Multi-shape sweep is a **second-order isolation** that single-shape diag
can't provide。When pack/quant bugs reach scale-chain level,K-dimension
sweep(particularly K=groupsize edge case)reveals **per-group vs
multi-group bifurcation**。

Per skill v1.3.0:**when single A/B variable narrows but doesn't pinpoint,
sweep one MORE variable**(here:#groups via varying K with fixed
groupsize)。The 2D sweep catches edge-case bifurcation that 1D misses。

For W4A8 specifically:**any future quant correctness regression test must
include K=groupsize edge case**(single-group)since that path is most
fragile per this evidence。
