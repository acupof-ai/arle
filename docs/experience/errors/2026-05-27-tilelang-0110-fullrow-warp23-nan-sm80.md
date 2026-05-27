# TileLang 0.1.10 FullRow + sm_80 — warps 2/3 produce NaN output

## Context

A100 80GB SXM4, CUDA 12.8, TileLang 0.1.10 (pip), Qwen3-4B BF16
prompt qlen=125, page_size=16 paged KV pool.

Running `kv_precision_parity` audit (`infer/tests/kv_precision_parity.rs`):

```
kv-parity: bf16   prompt0 first8 tokens: ref=[0, 0, 0, 0, 0, 0, 0, 0]
kv-parity: int8   prompt0 first8 tokens: ref=[0, ...] cand=[0, 0, 0, 0, 0, 0, 0, 0]
kv-parity: fp8    prompt0 first8 tokens: ref=[0, ...] cand=[0, 1183, 30, 21990, ...]
kv-parity: tq4    prompt0 first8 tokens: ref=[0, ...] cand=[151667, 13, 13, 22, ...]
```

BF16/INT8/FP8 paged prefill all argmax to token 0 (the `!` character),
which is the NaN-argmax fallback inside our argmax kernel. TQ4 (which
routes through `forward_prefill_batch` contig path because page_size=1)
produces token 151667 (`<think>`), matching the HF reference output —
correct. Contig BF16 prefill on the same prompt (via
`forward_token_logits` in `kv_fp8_prefill_logit_parity.rs`) also
produces 151667. So the bug is specifically in the **paged** prefill
path, which routes through TileLang HD128.

Distinct from
[`2026-05-27-tilelang-0110-cuda122-cutlass-incompat.md`](2026-05-27-tilelang-0110-cuda122-cutlass-incompat.md)
which is a TileLang 0.1.10 **build-time** cutlass C++20 / CUDA 12.2
mismatch. Our A100 CUDA 12.8 build of the same kernel compiles cleanly
— the bug here is a **runtime** miscompilation that produces NaN
attention output for specific warps.

## Root Cause

`crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py` has
identical functional code to commit `47bad713` (2026-04-28, last
known-working bench) — diff is comments only. The bug is in TileLang
0.1.10's codegen for `T.gemm(... policy=T.GemmWarpPolicy.FullRow)`
on `sm_80` with 4 warps × m=16 mma:

Probe via `INFER_TILELANG_PROBE2=1` after running the kernel on Qwen3-4B
(36 q heads / 8 kv heads, head_dim=128, BLOCK_M=64, NUM_THREADS=128 =
4 warps, NUM_STAGES=2, qlen=125):

```
[probe2-post-NaN-rows] layer=0 row=32 nan_count=4096/4096 max_abs=0
[probe2-post-NaN-rows] layer=0 row=35 nan_count=3970/4096 max_abs=0.057
[probe2-post-NaN-rows] layer=0 row=36 nan_count=3970/4096 max_abs=3.4e33
[probe2-post-NaN-rows] layer=0 row=42 nan_count=4096/4096 max_abs=0
[probe2-post-NaN-rows] layer=0 row=48 nan_count=4096/4096 max_abs=0
...
[probe2-post-NaN-rows] layer=0 row=100..124 nan_count=4096/4096 max_abs=0
```

Rows 0-31 clean (warps 0, 1 in FullRow partition: each owns BLOCK_M/4=16
rows). Rows 32+ exhibit NaN (warps 2, 3). The `max_abs=3.4e33` overflow
is `exp2(~111)` ≈ catastrophic softmax denominator collapse → division
in the kernel's final write produces inf/NaN.

The post-NaN distribution is identical with BLOCK_M=128 (1-tile case) —
only the warp boundary shifts: rows 0-63 clean (warps 0, 1 each owning
32 rows via 2 m-iters), rows 64-127 NaN (warps 2, 3).

Inspecting the generated `device_kernel.cu` from the TileLang AOT
output dir (`target/release/build/cuda-kernels-*/out/tilelang_aot/
batch_prefill_paged_hd128_q32_kv8_sm80/`) the Q `ldmatrix_x4` offset
for the second m-iter is hardcoded as `(ki >> 2) * 8192 +
(threadIdx.x >> 5) * 2048 + i_4 * 1024` — this offset increments warp
stride by 16 rows per warp (`2048 bf16 = 16 rows × 128 dims`). For
BLOCK_M=64 / 4 warps = 16 rows per warp, this is correct in arithmetic
but the resulting fragment layout for warps with index ≥ 2 reads from
shared-memory positions that don't map to a valid 16x16 A operand for
mma.sync m16n8k16. The exact mechanism is in TileLang's fragment
layout inferencer for sm_80 + 4-warp FullRow — not in our kernel
Python source.

## Workarounds attempted (all failed)

1. **Padding-row m_new clamp** (my hypothesis: `(-inf)-(-inf)=NaN` from
   the partial last tile's padding rows leaking into adjacent real rows
   via mma fragment): no effect — bug exists in **full** tiles (tile 0
   rows 0-63, no padding) too, so padding NaN is not the cause.
2. **BLOCK_M=128**: NaN pattern shifts to rows 64-127, same warp-2/3
   pattern. Disproves "last partial tile only" hypothesis.
3. **NUM_THREADS=64** (2 warps, no warps 2/3): TileLang host
   `cuLaunchKernel` block dim hardcoded to 128 regardless of
   `threads=` parameter → `CUDA_ERROR_INVALID_VALUE` on launch. Even
   if the device kernel was regenerated for 2-warp layout, the host
   launch config wasn't updated.
4. **FullCol policy** (warp partition across N axis instead of M):
   TileLang's data-race checker rejects the kernel with
   `Output(q_start + row, by * 128 + d) is written by multiple threads
   in loop (i,)` — FullCol is incompatible with this kernel structure.
5. **NUM_THREADS=256 + BLOCK_M=128** (8 warps × m=16 single mma per
   warp, FA-2 canonical layout): partial fix — warps 0-1 produce rows
   0-7 / 16-23 clean, partial NaN at rows 8-15 / 24-31; warps 2-3
   fully NaN at rows 32-63; warps 4-7 wrote nothing (rows 64-127
   acc_o stayed at the `T.fill(acc_o, 0)` initial value, presents as
   "clean" in the NaN probe but is functionally wrong = output 0).
   BF16 then produces first token = 262 (real but wrong) followed by
   `[0, 0, 0]` from cascading NaN through subsequent decode steps.
6. **NUM_STAGES=1** (disable software pipelining): no effect.
7. **Causal-bound disable** (`kv_visible_end = kv_total_len`): no
   effect.

All workarounds confirmed via cubin diff — `device_kernel.cu` does
regenerate per Python change after `rm -rf ~/.tilelang/cache`.

## Fix

**Recommended:** Pin `tilelang>=0.1,<0.1.10` for any environment
running paged prefill on sm_80. Pair with the existing
`scripts/sm70_tilelang.patch` (V100 SM70 FMA fallback) using a
compatible upstream tag.

This is consistent with the parallel finding in
[`2026-05-27-tilelang-0110-cuda122-cutlass-incompat.md`](2026-05-27-tilelang-0110-cuda122-cutlass-incompat.md):
both bugs surface only on 0.1.10, both resolve by pinning to 0.1.9.

**Short-term route-around (if 0.1.10 must stay):** route BF16/INT8/FP8
paged-pool prefill requests through the contig prefill kernel
(`process_all_layers_batch`, already known-correct via the
`kv_fp8_prefill_logit_parity` test), then migrate contig K/V into the
paged pool post-prefill before decode (TQ4 already does an analogous
contig→paged migration). Env-gated `INFER_BYPASS_TILELANG_PREFILL=1`.

**Long-term:** Migrate paged prefill+decode to FlashInfer C++ (whose
ABI — `qo_indptr` / `kv_indptr` / `kv_indices` / `kv_last_page_len` —
the current kernel already mirrors). Removes the TileLang AOT
codegen + fork-and-patch maintenance burden for this surface.

## Rule

**TileLang 0.1.10 has two independent regressions vs 0.1.9** (cutlass
build + sm_80 runtime). Both surfaced in the same week. Pin to 0.1.9
unless explicitly testing 0.1.10 on a single SKU you can audit
end-to-end. Bump only after running `kv_precision_parity` (Qwen3-4B
on sm_80) AND the V100 sm_70 audit AND a CUDA 12.2 box, and verifying
ALL THREE produce non-degenerate baselines with `first8 tokens` dumped
explicitly.

**When per-warp NaN patterns appear in a TileLang kernel output**,
inspect the generated `device_kernel.cu` directly — the AOT output
dir under `target/release/build/cuda-kernels-*/out/tilelang_aot/<kernel>/`
contains the lowered CUDA source. mma fragment layout bugs are not
visible from the Python kernel definition; they require the generated
.cu to diagnose.

## Related

- [`2026-05-27-tilelang-0110-cuda122-cutlass-incompat.md`](2026-05-27-tilelang-0110-cuda122-cutlass-incompat.md)
  — TileLang 0.1.10 build-time cutlass C++20 / CUDA 12.2 break, also
  fixed by 0.1.9 pin.
- [`2026-05-26-fp8-kv-catastrophic-was-test-artifact.md`](2026-05-26-fp8-kv-catastrophic-was-test-artifact.md)
  — earlier "FP8 KV catastrophic divergence" was a `mean_match`-vs-
  degenerate-baseline test-methodology bug. This new entry is a
  *real* kernel bug, distinct from that test artifact, and is
  reproducible on a non-degenerate prompt as well.
- Commits 25c7d409 (FP8 quant scale floor), 73a72615 (KIVI partial
  kernel normalize), e0c283d1 (recursive rerun-if-changed) — kept,
  unrelated to this bug.
- `scripts/sm70_tilelang.patch` + `scripts/patch_tilelang_sm70.sh` —
  the V100 SM70 PR #2279 patch, already in repo; needs version
  alignment if we pin 0.1.9 (`git apply --check` against 0.1.9
  source).
