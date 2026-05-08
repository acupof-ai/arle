# W4A8 pack/unpack round-trip diag confirms pack broken — empirical Layer 5 isolation

> Codex `ab43959` provided pack/unpack round-trip diagnostic per `01ace86`
> audit Option 1。Claude this tick:run diag with default 256×128 shape
> + groupsize 128。
>
> **Result: pack/unpack round-trip FAILS**(max abs diff 0.21 vs expected
> 0.02 noise band = 10× over)。Recovered values ~2× the originals at
> systematic positions(row 112 = group boundary)。**Empirical Layer 5
> confirmation:H4 fix did NOT close pack;additional bug at scale-application
> stage**。

## Diag output

```
$ python scripts/diag_w4a8_pack_roundtrip.py
Shape: out=256 in=128 groupsize=128
Pack output shapes: qweight=[8, 512] dtype=torch.int32
                    s_channel=[1, 256] dtype=torch.float32
                    s_group=[1, 256] dtype=torch.float16

Round-trip diagnostic:
  max abs diff   = 2.105598e-01  (expected ~1.9577e-02)
  mean abs diff  = 1.460834e-02
  p99 abs diff   = 6.607273e-02
  max rel diff   = 2.2273
  mean rel diff  = 0.3287

❌ FAIL: pack/unpack round-trip OUT OF noise band (9.7883e-02)
   → pack_w4a8 has a forward/inverse asymmetry; pack is broken.

Top-10 mismatch positions:
  [112,114]: orig=+0.2969 recovered=+0.5074 diff=+0.2106
  [112,54]:  orig=-0.2373 recovered=-0.4349 diff=+0.1976
  [112,94]:  orig=-0.2402 recovered=-0.4349 diff=+0.1947
  [112,112]: orig=+0.2471 recovered=+0.4349 diff=+0.1879
  [112,26]:  orig=-0.2500 recovered=-0.4349 diff=+0.1849
  ...
```

## Analysis

**Pattern**:recovered ≈ 2× original at row 112 specifically。

Row 112 = `group_size = 128 - 16 = 112`(near group boundary)。Looking
at the +/− pattern:
- `+0.2969 → +0.5074`(ratio ~1.71)
- `−0.2373 → −0.4349`(ratio ~1.83)
- `+0.2471 → +0.4349`(ratio ~1.76)

Ratio is **not exactly 2×** but consistently in 1.7-2.2 range — suggests
a **per-group quantize/dequant scale mismatch** rather than a uniform
factor。

Possible root causes:
1. **Scale `.div(7.0)` should be `.div(7.5)` or `.div(8.0)`** — INT4 max
   value handling at boundary
2. **`s_work.reshape((1, -1))`** broadcast wrong N stride
3. **Group boundary clamping** — values right at `± max_per_group` get
   rounded to ±7,but reconstruction multiplies by `s = max/7 → ±max`,
   while original was inside `[max/2, max]` range → 2× recovery
4. **Symmetric vs asymmetric**:if pack uses symmetric `[-7,7]` but
   asymmetric quant `+8` shift,boundary values get amplified

(Hypothesis 3 is strong:if original value is e.g. `+0.297` and group max
is `0.5`,quantized = `round(0.297 / (0.5/7)) = round(4.16) = 4` → INT4
stored = 4-(-8) = 12;dequant = `(12 - 8) × (0.5/7) = 4 × 0.0714 = 0.286` ≈ original ✓。

But observed recovered = `+0.507` ≈ original × 1.71。Hmm,that suggests
**divisor was twice as small**:`+0.297 × 1.71 = +0.508`;divisor would
need to be `0.5/3.5 = 0.143` instead of `0.5/7 = 0.0714`。

Could be `s = max / 7` BUT applied as `s_recovered = max / 14` somewhere
in the chain — e.g., **doubled storage somehow**。Or **scale stored is
half but kernel/diag multiplies twice**。

This narrows the bug to **scale application chain**:
- Pack divides by `max/7`,stores result as `max/7`
- But unpack effectively multiplies by `2 × max/7 = max/3.5`(via some
  unintended doubling)
- → recovered = original × 2

## Action — codex own

Diag isolates the bug to **pack_w4a8 scale chain**(narrows from
"perm + scale + tile + bit-packing" to "scale chain specifically")。
Per codex `01ace86` audit recommendation:**this is exactly the
isolation diag was designed to provide**。

Codex needs to:
1. Add `--verbose` or instrumented mode to diag printing intermediate
   values (s_work shape/values, s_pack shape/values, w_int4 sample,
   recovered sample)
2. Compare with PR #31 reference `Layer.pack` step-by-step at the same
   shape
3. Find where the 2× mismatch enters

Probability ~80% the bug is a single scale-doubling or scale-halving
in `pack_w4a8` lines 92-100(s_pack reshape / s_work flatten / division)。

## Cross-references

- Codex diag commit: `ab43959`
- Codex H4 fix: `592779a` `945df02`(broadcast misalignment fixed but
  pack still broken at larger scope)
- Codex audit kernel+wiring CLEAN: `01ace86`
- Skill v1.3.0 anti-pattern #13: NULL elimination

## Rule

Pack/unpack round-trip diagnostic is the **gold standard isolation
test** for quant correctness — element-wise diff localizes the bug
without going through kernel + Rust FFI + loader complexity。

Per codex audit Rule:**when ≥3 iterations don't converge**,run
non-iterative diagnostic(unit test like this round-trip)。Codex
correctly applied the rule by writing this diag tool。Now Claude+codex
have both the diag tool AND the failing assertion data → next-step
fix has empirical localization ground。
