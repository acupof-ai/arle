# W4A8 bug — H3c CONFIRMED:`scale_perm_single` applied to s_channel BEFORE division(should be AFTER)

> Continues 3-layer perm bug elimination chain:
> `25391f3` H3 row stride → `62f885d` row-fix partial → `3479a87` H3b
> scale_perm_single missing → `03178cf` H3b applied still partial(3rd
> layer remaining)→ this entry。
>
> Direct source diff between `/tmp/quantize_qwen3_w4a8.py` (ARLE post-H3b)
> and `/tmp/marlin-w4a8/marlin/__init__.py` (PR #31 reference) reveals
> **order-of-operations bug**:ARLE permutes s_channel BEFORE the division
> `s_work / s_channel`,PR #31 permutes AFTER。Both produce shape (1, n)
> tensors,but the division semantics differ。

## ARLE flow(BUG)

`/tmp/quantize_qwen3_w4a8.py:90-110`:

```python
# Line 90-95: produce + ⛔ PREMATURELY permute s_channel
s_channel = ref.t().abs().amax(dim=-1, keepdim=True).div(127.0).to(torch.float32)
s_channel = torch.where(s_channel == 0, torch.ones_like(s_channel), s_channel)
s_channel = s_channel.reshape(1, n)
s_channel = s_channel.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]  # ⛔
s_channel = s_channel.reshape((-1, n)).contiguous()

# Line 97-105: quantize w using s_pack (per-group)
reshaped = ref.reshape(k // groupsize, groupsize, n)
s = reshaped.abs().amax(dim=1).clamp_min(1e-6).div(7.0).to(torch.float16)
s_pack = s.t()
w = ref.reshape((-1, groupsize, n)).permute(1, 0, 2).reshape((groupsize, -1))
s_work = s_pack.reshape((1, -1))
w = torch.round(w / s_work).to(torch.int32)
w += 8
w = torch.clamp(w, 0, 15)

# Line 107: ⛔ DIVIDE BY PERMUTED s_channel
s_group = (s_work.reshape(-1, n) / s_channel).to(torch.float16)
# Then line 109: permute s_group by scale_perm
s_group = s_group.reshape((-1, len(scale_perm)))[:, scale_perm]
```

## PR #31 reference(correct)

`/tmp/marlin-w4a8/marlin/__init__.py:288-299`:

```python
if self.groupsize != self.k:
    s_extra = s_extra.reshape(1, -1).to(dtype=torch.float)  # raw shape, NOT permuted yet
    s = (
        s.reshape(-1, self.n) / (s_extra)              # ⭐ divide BEFORE permute
    ).to(dtype=torch.half)
    w = w.reshape((self.groupsize, -1, self.n))
    w = w.permute(1, 0, 2)
    w = w.reshape((self.k, self.n)).contiguous()
    s = s.reshape((-1, len(self._scale_perm)))[:, self._scale_perm]
    # ⭐ s_extra permuted AFTER division
    s_extra = s_extra.reshape((-1, len(self._scale_perm_single)))[
        :, self._scale_perm_single
    ]
    s_extra = s_extra.reshape((-1, self.n)).contiguous()
```

## Why order matters

Division `s_work[group, j] / s_channel[0, j]` requires both tensors to align
column N → output channel j。

- **PR #31**:s_channel raw col j == output channel j。Division correctly
  computes `s_group[group, j] = s_work[group, j] / s_channel[0, j]`
- **ARLE post-H3b**:s_channel_permuted col j == s_channel raw at index
  `scale_perm_single[j]`,which is some OTHER output channel。Division
  computes `s_group[group, j] = s_work[group, j] / s_channel[0, scale_perm_single[j]]`
  → **wrong denominator** for every column

After division wrong,line 109 permutes s_group by scale_perm independently。
Now kernel reads:
- s_channel(permuted by scale_perm_single)at thread fragment positions
- s_group(permuted by scale_perm,but values were divided by WRONG s_channel)

→ multiplication `s_group * s_channel` 在 kernel 内 partially "cancels"
the wrong division but introduces **systematic per-channel error**(real
math:`s_work / s_channel[A] * s_channel[B]`,期望 `s_work`)→ **每个
output channel 的 logit 被 ratio `s_channel[B] / s_channel[A]` 乘错**。

This matches qualitative observation in `03178cf`:
- "English-fragmented + code-like" output
- Token diff still 100% but distribution shape closer to natural English
- Per-channel scale **partially right magnitude** but **wrong feature
  channel routing**

## Fix

Move scale_perm_single permutation to AFTER the division。Suggested patch:

```python
# Replace line 90-95:
s_channel = ref.t().abs().amax(dim=-1, keepdim=True).div(127.0).to(torch.float32)
s_channel = torch.where(s_channel == 0, torch.ones_like(s_channel), s_channel)
s_channel = s_channel.reshape(1, n)
# ❌ DON'T permute here

# Keep line 97-105 unchanged
# Keep line 107 unchanged (divides by raw s_channel — now correct)

# After line 109 (s_group scale_perm applied), apply scale_perm_single to s_channel:
# (insert before line 110)
s_channel = s_channel.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
s_channel = s_channel.reshape((-1, n)).contiguous()
```

Equivalently:hoist the s_channel permutation to immediately before
`return qweight, s_channel.contiguous(), s_group.contiguous()`(line 121),
matching PR #31 line 296-299 timing。

## Probability estimate

**~85%** this is the 3rd-layer bug:
- Direct source diff confirms order difference
- Math analysis shows division produces wrong per-channel ratio
- Qualitative output progression matches predicted character(English-frag
  + per-channel routing error)
- All 3 perm-script bugs sourced from same `del scale_perm_single` mistake
  (3479a87)— author likely reordered ops thinking permutation could be
  applied "anywhere",missing that division creates positional dependency

Remaining 15%:
- Maybe additional subtle bug in tile permute lines 112-115 (reshape order)
- Or weight_loader.rs:663-715 reads scales from wrong tensor field name
- Or pack final bit-packing `q |= res_np[:, i::8]` stride differs(but
  PR #31 line 312 identical)

## Codex action(15 min fix + 30-60 min re-quantize + test)

1. Apply patch:remove premature s_channel permutation(line 94-95),
   add it after s_group permutation
2. Re-quantize Qwen3-4B → `infer/models/Qwen3-4B-W4A8-marlin/`
3. `cargo test --release -p infer --features cuda --test greedy_consistency`
4. If passes → bench W4A8 真实 numbers + greedy gate ✅ + default-on flip
   unblocked
5. If still 100% diff but output character still progressed → investigate
   tile permute lines 112-115 (next-layer suspect)

## Cross-references

- H3c source diff: this entry
- H3b confirmed: [`2026-05-08-w4a8-bug-h3b-confirmed-scale-perm-single-deleted.md`](2026-05-08-w4a8-bug-h3b-confirmed-scale-perm-single-deleted.md) (`3479a87`)
- H3b applied still partial: [`2026-05-08-w4a8-h3b-fix-applied-still-partial.md`](2026-05-08-w4a8-h3b-fix-applied-still-partial.md) (`03178cf`)
- H3 row stride confirmed: [`2026-05-08-w4a8-bug-h3-confirmed-perms-row-stride.md`](2026-05-08-w4a8-bug-h3-confirmed-perms-row-stride.md) (`25391f3`)
- ARLE script: `/tmp/quantize_qwen3_w4a8.py:90-110`
- PR #31 reference: `/tmp/marlin-w4a8/marlin/__init__.py:288-299`
- W4A8 garbage gate: `81b6481`
- Failing test: `infer/tests/greedy_consistency.rs::test_w4a8_vs_bf16_token_diff`

## Rule

When porting a quantize script,**every reshape/permute/divide/multiply is
positionally coupled** — changing operation order without rederiving the
position math creates per-channel routing bugs that produce 100%-diff
output that's qualitatively close to right but wrong on every token。
Direct line-by-line diff against the upstream reference is the only
reliable verification — algebraic-equivalence reasoning fails here。
