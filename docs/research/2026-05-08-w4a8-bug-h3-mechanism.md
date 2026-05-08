# W4A8 bug — H3 perms mismatch mechanism(INT8 vs FP16 fragment layout)

> Follow-up to elimination chain:
> [`e20f24c`](2026-05-08-w4a8-bug-claude-investigation.md)(5→2)→
> [`b65c8c6`](2026-05-08-w4a8-bug-h2-ruled-out.md)(H2 out)→
> [`88dfafc`](2026-05-08-w4a8-bug-h4-h5-ruled-out.md)(H4+H5 out)→ this entry。
>
> H3 (perms / scale_perm mismatch) is now THE remaining suspect。This entry
> identifies the **mechanism** for codex to verify(15 min cheap):INT8 vs
> FP16 element width differs → per-thread mma fragment layout differs。

## Kernel mma shape

`crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu`:

```cpp
// Line 50 comment:
//   matrix-fragments-for-mma-m16n8k16-with-integer-type

// Line 111 comment:
//   m16n8k16 tensor core mma instruction with int8 inputs and int32 output

// Line 117 PTX:
mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.s8.s32
```

→ **mma shape M=16, N=8, K=16** with **INT8 inputs**, **INT32 accumulator**。

## W4A16 Marlin uses same shape but FP16 inputs

`crates/cuda-kernels/csrc/gemm/marlin_kernel.cu`(production W4A16,license `f6f3af3`):

PR #31 says "supported w4a8 marlin based your code" — directly derived from Elias Frantar's W4A16 Marlin。Both use **m16n8k16** mma shape。

**But element width differs**:
- W4A16: A is FP16 (16-bit), B is INT4 packed(stored as 16-bit)
- W4A8: A is **INT8 (8-bit)**, B is INT4 packed (stored as 16-bit), accumulator INT32

## Per-thread mma fragment layout differs

For PTX `mma.sync.aligned.m16n8k16` per [NVIDIA PTX docs](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-integer-type):

| matrix | dtype | per-thread elements | per-thread bytes |
|---|---|---|---|
| A FP16 (W4A16)| f16 | 4 elements ×4 chunks = 16 elements | 32 bytes |
| **A INT8 (W4A8)** | s8 | 4 elements ×4 chunks = 16 elements | **16 bytes** |
| B FP16(W4A16)| f16 | 2 elements ×2 chunks = 4 elements | 8 bytes |
| **B INT8(W4A8)** | s8 | 2 elements ×4 chunks = 8 elements | **8 bytes** |

**A INT8 per-thread bytes is half of A FP16**(16 vs 32),meaning **fragment register layout is different**。

If `/tmp/quantize_qwen3_w4a8.py` `get_perms()` was copied verbatim from a W4A16 quantize script(Marlin codebase originally written for FP16),the permutation **assumes 32-byte per-thread A fragment**(FP16 layout),but kernel **actually loads 16-byte per-thread A fragment**(INT8 layout)。

→ Permuted weights end up in wrong thread slots → mma reads wrong data → garbage output。

## Concrete diagnostic for codex(15 min)

1. **Find a known-working W4A8 Marlin quantize script** — PR #31 `marlin/__init__.py` 应该包含 the **right** perm function for INT8 fragment。
2. Compare `/tmp/quantize_qwen3_w4a8.py::get_perms()` vs PR #31 reference:
   - Stride pattern(256 vs 128 vs other)
   - Interleave indices order
   - Block step
3. If different → bug confirmed。Cherry-pick PR #31 perms verbatim。
4. If identical(same FP16-style perms script copied)→ bug is in the assumption that PR #31 itself published wrong perms → cross-check QQQ paper §3.3 fragment layout。

## Probability estimate

H3 perms mismatch:**high(~70%)** based on:
- All other 4 hypotheses ruled out by direct kernel read
- INT8 vs FP16 element width DOES change per-thread layout(verified PTX docs)
- Quantize scripts are notoriously version-specific between Marlin variants
- Marlin perms are "load-bearing"(per `e20f24c` original investigation)

Remaining 30%:
- PR #31 published correct INT8 perms but quantize script had subtle bug elsewhere
- Or:bug in `cp_async4_stream` byte-level loading 跟 perm 互动
