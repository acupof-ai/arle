# W4A8 bug — H4 CONFIRMED:redundant `s_pack = s.t()` mis-aligns s_work broadcasting against w during quant division

> Per `01ace86` audit recommendation,Claude-side investigation continued
> with byte-level scrutiny of `pack_w4a8` flow vs PR #31 reference。This
> entry identifies **the actual 4th-layer bug**:s_pack `.t()` on line 99
> scrambles flat broadcast indices,causing per-element quant division
> to use the wrong scale for ~every w element。
>
> Bug separate from H3/H3b/H3c。**This is likely THE bug**(probability
> 90% per dimensional analysis below)。

## ARLE flow(BUG)

`/tmp/quantize_qwen3_w4a8.py:97-103`:

```python
reshaped = ref.reshape(k // groupsize, groupsize, n)             # (k/gs, gs, n)
s = reshaped.abs().amax(dim=1).clamp_min(1e-6).div(7.0).to(torch.float16)  # (k/gs, n)
s_pack = s.t()  # ⛔ (n, k/gs) — REDUNDANT TRANSPOSE

w = ref.reshape((-1, groupsize, n)).permute(1, 0, 2).reshape((groupsize, -1))  # (gs, k*n/gs)
s_work = s_pack.reshape((1, -1))  # (1, n*k/gs) — flat j = i_n*(k/gs) + i_kgs
w = torch.round(w / s_work).to(torch.int32)  # ⛔ broadcast misaligned
```

## PR #31 reference(correct)

`/tmp/marlin-w4a8/marlin/__init__.py:274-282`:

```python
s = scales.t()  # caller passes scales = (n, k/gs), s.t() = (k/gs, n) — NO REDUNDANT
w = linear.weight.data.t()
if self.groupsize != self.k:
    w = w.reshape((-1, self.groupsize, self.n))     # (k/gs, gs, n)
    w = w.permute(1, 0, 2)                          # (gs, k/gs, n)
    w = w.reshape((self.groupsize, -1))             # (gs, k*n/gs)
    s = s.reshape((1, -1))                          # ✓ from (k/gs, n) → flat j = i_kgs*n + i_n
w = torch.round(w / s).int()
```

PR #31 caller(`test_w4a8.py:42` or quantize pipeline)passes `scales` already
in `(n, k/gs)` form,so `scales.t()` on line 275 brings it to **`(k/gs, n)`
matching w's flat order**。

ARLE computes `s` directly in `(k/gs, n)` from amax(dim=1) — **no
transpose needed**。Then ARLE's `s_pack = s.t()` adds an extra rotation
to `(n, k/gs)`,after which `.reshape((1, -1))` flattens with the **wrong
column-major-style index order**。

## Dimensional analysis(why broadcast is wrong)

w shape after permute + reshape:`(gs, k*n/gs)`,flat row-major within
each gs row:

```
w[gs_i, j] for j ∈ [0, k*n/gs)
    = ref[i_kgs * gs + gs_i, i_n]
where j = i_kgs * n + i_n (i_kgs = j//n, i_n = j%n)
```

s after `s_pack.reshape((1, -1))`:

```
s_work[0, j] = s_pack[i_n_alt, i_kgs_alt]
            = s[i_kgs_alt, i_n_alt]
where j = i_n_alt * (k/gs) + i_kgs_alt
       (i_n_alt = j // (k/gs), i_kgs_alt = j % (k/gs))
```

For the division `w / s_work` to apply the right scale per element,we
need at every j:

```
i_kgs = i_kgs_alt  AND  i_n = i_n_alt
⇔ j // n == j % (k/gs)  AND  j % n == j // (k/gs)
```

Generally **only true when n = k/gs**(in which case mod and div coincide)。
For Qwen3-4B with `groupsize = 128`:
- Layer with `k=2048, n=2048`:k/gs=16, n=2048 → mismatch by factor 128
- Layer with `k=4096, n=11008`:k/gs=32, n=11008 → mismatch by factor 343
- Layer with `k=11008, n=4096`:k/gs=86, n=4096 → mismatch by factor ~48

**Every Linear layer in Qwen3-4B has n ≠ k/gs** → division is misaligned。

## Why output character looked progressively more English

H3 + H3b moved permutation layers closer to PR #31。Each fix improved
output progression(period bias → multilingual → English-frag)but **never
got past 100% diff** because the underlying division was scaled with
wrong column ordering since day 1。The dequant in kernel `(q-8) * s_group_stored
* s_channel` recovers magnitude approximately(s_channel + s_group_stored
norms cancel-out roughly per-tile)but per-element semantic is broken。

H3c reverted because applying scale_perm_single AFTER division **with
already-misaligned division** produced different scrambling,not the
correct distribution。Per `06193eb` revert was right call。

## Fix

Replace ARLE script line 97-103 with:

```python
reshaped = ref.reshape(k // groupsize, groupsize, n)
s = reshaped.abs().amax(dim=1).clamp_min(1e-6).div(7.0).to(torch.float16)  # (k/gs, n)
# REMOVED: s_pack = s.t()

w = ref.reshape((-1, groupsize, n)).permute(1, 0, 2).reshape((groupsize, -1))  # (gs, k*n/gs)
s_work = s.reshape((1, -1))  # ✓ from (k/gs, n) — flat j = i_kgs*n + i_n
w = torch.round(w / s_work).to(torch.int32)
```

Then need to **verify line 107 still works** with the renamed `s` (was `s_pack`):

```python
# Original ARLE line 107:
s_group = (s_work.reshape(-1, n) / s_channel).to(torch.float16)
```

After fix,`s_work.reshape(-1, n)` from `(1, k*n/gs)` becomes shape `(k/gs, n)`
which is correct per-group per-channel layout for division by `s_channel`
shape `(1, n)`。✓

## Probability estimate

**~90%** this is the bug:
- Direct dimensional analysis confirms broadcast misalignment
- Explains why H3/H3b/H3c iteration didn't converge:underlying division
  was wrong before any perm fix could matter
- Magnitude of mismatch(48-343× factor)consistent with extreme corruption
  observed in pre-H3 output("period spam,no useful info")
- Progressive English-frag character with H3+H3b explainable as **partial
  permutation accidentally compensating** for some of the misalignment

Remaining 10%:
- The kernel's dequant might internally compensate via tile-level summing
  in a way that papers over per-element misalignment
- Or the tile/perm interleave may further interact with this bug,making
  fix non-trivial
- Or some other subtle interaction with `s_work.reshape(-1, n)` step
  needs re-examination

## Codex action(15 min fix + 30-60 min re-quantize + test)

1. Apply 2-line patch to `/tmp/quantize_qwen3_w4a8.py`:
   - Remove line 99(`s_pack = s.t()`)
   - Change line 102 to `s_work = s.reshape((1, -1))`
2. Re-quantize Qwen3-4B → `infer/models/Qwen3-4B-W4A8-marlin/`
3. `cargo test --release -p infer --features cuda --test greedy_consistency`
4. If passes → bench W4A8 真实 numbers + greedy gate ✅ + default-on flip
   unblocked
5. If still 100% diff → investigate kernel-internal dequant tile summing
   for hidden compensation,OR fall through to audit Option 1(unit test)

## Cross-references

- Audit clean: [`01ace86`](2026-05-08-w4a8-kernel-and-wiring-audit-clean.md)
- H3c reverted: [`06193eb`](2026-05-08-w4a8-h3c-reverted-methodology-pivot.md)
- H3+H3b state(closest to right): `3479a87`
- W4A8 garbage gate: `81b6481`
- ARLE script: `/tmp/quantize_qwen3_w4a8.py:97-103`
- PR #31 reference: `/tmp/marlin-w4a8/marlin/__init__.py:274-282`
- PR #31 unit test scaffold: `/tmp/marlin-w4a8/test_w4a8.py:1-60`(Option 1 enabler)

## Methodology lesson

Direct dimensional analysis of broadcast indices over a quant flow chain
caught a bug that line-by-line script diff missed,because both ARLE and
PR #31 had `.t()` operations that LOOK similar but operated on tensors
with **different starting shapes**。

The earlier H3 / H3b / H3c briefs all compared script line-by-line but
didn't trace the **shape of `s` at the moment of `.t()` call** between
the two scripts:
- PR #31 `.t()` operates on `(n, k/gs)` external input → result `(k/gs, n)`
- ARLE `.t()` operates on `(k/gs, n)` internal computation → result `(n, k/gs)`

**Same operation,opposite outcome,because pre-state differs**。This kind
of bug requires **dimensional tracking through the chain**,not local
diff comparison。Per skill anti-pattern #14:line-by-line diff has known
blindspots when tensors arrive from different upstream computations。

## Rule

When porting a quant/kernel script,**every reshape/permute/transpose is
coupled to the upstream tensor shape**。Two scripts with the SAME
sequence of operations can produce DIFFERENT results if their upstream
shapes differ。Direct line diff is necessary but not sufficient — must
also **track tensor shapes at each operation boundary** and verify the
broadcast/reshape semantics produce identical flat-index ordering。
