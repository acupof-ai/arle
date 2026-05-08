# W4A8 bug — H3 CONFIRMED:row stride mismatch in get_perms()

> Continues elimination chain:H1 algebra OK / H2/H4/H5 ruled out by code read
> ([`b65c8c6`](2026-05-08-w4a8-bug-h2-ruled-out.md)
> [`88dfafc`](2026-05-08-w4a8-bug-h4-h5-ruled-out.md))/ H3 mechanism brief
> ([`e3ca4d8`](2026-05-08-w4a8-bug-h3-mechanism.md))。
>
> This entry **directly compares** `/tmp/quantize_qwen3_w4a8.py::get_perms()`
> vs PR #31 reference `marlin/__init__.py::_get_perms()` — bug 100%
> confirmed。

## Diff

### PR #31 reference(correct,W4A8 INT8 fragment layout)

```python
def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),           # 0
                2 * (i % 4) + 1,       # 1
                2 * (i % 4 + 4),       # 8  ⭐ skip to row 8
                2 * (i % 4 + 4) + 1,   # 9  ⭐
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])
    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])  # always
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single
```

### ARLE `/tmp/quantize_qwen3_w4a8.py` 当前实现(BUG)

```python
def get_perms(groupsize: int, k: int):
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                4 * (i % 4),       # 0      ⛔ wrong stride
                4 * (i % 4) + 1,   # 1      ⛔
                4 * (i % 4) + 2,   # 2      ⛔
                4 * (i % 4) + 3,   # 3      ⛔
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    if groupsize == k:                              # ⛔ extra branch from W4A16 path
        interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
    else:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    # scale_perm + scale_perm_single 部分 identical to PR #31 ✅
    ...
```

## 两 bug 点

**Bug 1**:row pattern。
- PR #31 (correct):rows `{2k, 2k+1, 2(k+4), 2(k+4)+1}` = sparse skip-8 pattern → matches W4A8 INT8 mma fragment layout per-thread byte interleave
- ARLE (wrong):rows `{4k, 4k+1, 4k+2, 4k+3}` = consecutive 4-row pattern → matches W4A16 FP16 layout

**Bug 2**:groupsize == k branch。
- PR #31 only handles `groupsize != k`(per-group quant,正常 case)
- ARLE adds branch `groupsize == k → interleave = [4,0,5,1,...]`(per-tensor quant 退化 case)
- Qwen3-4B uses groupsize=128, k 通常更大 → falls into else branch → interleave 跟 PR #31 一致 ✓
- 但 extra branch 不会触发当前 codepath,**not the bug**

→ **唯一真 bug = row pattern**。

## Fix

修改 `/tmp/quantize_qwen3_w4a8.py::get_perms()` 把 row generation 替换为 PR #31 verbatim:

```python
# Before
for row in [
    4 * (i % 4), 4 * (i % 4) + 1,
    4 * (i % 4) + 2, 4 * (i % 4) + 3,
]:

# After
for row in [
    2 * (i % 4), 2 * (i % 4) + 1,
    2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1,
]:
```

然后 re-quantize Qwen3-4B → re-run `cargo test --release -p infer --features cuda --test greedy_consistency::test_w4a8_vs_bf16_token_diff`。

## Why row pattern matters(per H3 mechanism brief)

W4A8 mma `m16n8k16` with INT8:per-thread A fragment loads 16 elements over 16-byte chunks。Row stride determines which physical rows的 elements 落到 single thread fragment。

- 4-stride consecutive pattern(W4A16 FP16 32-byte/thread):rows 0-3 in same thread
- 2-stride skip-8 pattern(W4A8 INT8 16-byte/thread):rows {0,1,8,9} in same thread

Wrong stride → permuted weights placed at wrong thread-local positions → mma reads wrong data → **garbage output(token diff 100%)**。

## Codex action(15 min fix + 30-60 min re-quantize)

1. Apply 4-line patch to `/tmp/quantize_qwen3_w4a8.py`(or save as `scripts/quantize_qwen3_w4a8.py` in repo for repro)
2. Re-quantize Qwen3-4B → `infer/models/Qwen3-4B-W4A8-marlin/`
3. `cargo test --release -p infer --features cuda --test greedy_consistency`
4. If passes → bench longctx 4k/c=4 again(W4A8 真实 number,not garbage speed)
5. If fails → expand investigation to scale_perm or block stride

Probability fix works:**~90%**(row pattern bug is verbatim, mechanism aligns with PTX byte-layout difference)。
