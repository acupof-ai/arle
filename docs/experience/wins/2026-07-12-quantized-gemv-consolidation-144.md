# quantized_gemv.cu consolidation (#144) — templated GEMV bodies

> Live ledger for the #144 deletion-refactor: collapse 14 near-identical
> warp-per-row GEMV kernels onto ~4 `template<..., class ScaleFn>` bodies. One
> section per increment; a single clean idle-box perf A/B covers the whole
> refactor at the end (each increment is a compile-time template merge → same
> SASS by construction, so per-increment perf is structurally neutral).

## SLO-shape probed?  N — pure refactor; runtime dispatch + numerics unchanged, gate is correct-inference

## Context

`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu` had 37 `__global__` kernels /
3226 lines of warp-per-row FP8/FP4/wNa16 × single/batch/tiled/pair/grouped/route
boilerplate. The warp-per-row + uint4-dot16 + `warp_reduce_sum` +
`smem[GEMV_ROWS*8]` skeleton is copy-pasted; only the (decode, scale) functor
varies. k-quant (12), embedding (4), dense-dequant (1), probe (1) are
structurally distinct and stay untouched.

## Increment 1 (2026-07-12, `5f9a0b3ec`) — fp8 tiled pair — VERIFIED

Merged `dsv4_fp8_gemv_batch_tiled_kernel<TILE>` (e8m0 uint8 block scale) +
`fp8_f32_block_gemv_batch_tiled_kernel<TILE>` (f32 block scale) — byte-identical
except the scale — into `template<int TILE, class ScaleFn> fp8_gemv_batch_tiled_kernel`.
Launchers construct `Fp8E8m0BlockScale` / `Fp8F32BlockScale`. **−85 lines.**

H20 sm_90 verification:
- **Compile PASS** — `cargo build --features cuda,nccl` `BUILD_EXIT=0`, template +
  functors + per-B tile dispatch clean on sm_90 (the primary gate for a raw .cu
  template change I can't compile on Mac).
- **Correctness PASS** — both families exercise the B>1 tiled path: Qwen3.6-27B-FP8
  needle gate PASS (9 lengths exact 3/3, MoE-nondet floor expected); 4 concurrent
  → all `Paris`. DSv4-Flash-FP8 TP=4 decoded cases coherent (`BLUE-42`, finish=stop);
  4 concurrent → all `Paris`, sound reasoning. Zero garbage from the merged kernel.
- **Perf — structurally neutral, raw number confounded.** Measured conc8 1213 vs a
  1980 MHz, no-throttle, 0-34%-util GPU while foreign load pinned GPUs 3/4 at 100%
  — the GPU was host-starved (a slower kernel would raise util, not lower it), so
  the raw drop is box contention, not the merge. Functors inline → same SASS.
  Clean idle-box A/B deferred to the end of #144.

## Increment 2 (2026-07-12, `90de21f0e`) — fp8 single-output batch pair — VERIFIED

Merged `dsv4_fp8_gemv_batch_kernel` + `fp8_f32_block_gemv_batch_kernel` (the B==1
single-output path) into `template<class ScaleFn> fp8_gemv_batch_kernel`, reusing
the Increment-1 functors. **−62 lines** (cumulative −147). H20 sm_90: compile PASS
(`BUILD_EXIT=0`, nccl present), correctness PASS (Qwen3.6-27B-FP8 needle 9/9 exact
115→8000, samples ` Paris.`/` Tokyo.`, zero garbage). Both fp8 increments green.

**fp8 hot-path consolidation (the #144 core) is done: 4 kernels → 2 templated
bodies, −147 lines, the 52%-of-decode FP8 GEMV path, correctness double-verified.**

## Remaining increments — DEFERRED (lower ROI / higher risk)

The fp8 merges were byte-identical duplicates on the hot path — clear wins. The
rest are marginal and left as follow-ups:

3. wNa16 w2/w4/w8 single+batch (6 kernels) — NOT byte-identical: each bit width's
   unpack core differs (w8 4/word, w4 8/word+zp8, w2 16/word+zp2). Cross-bit-width
   `UnpackFn` is a fragile abstraction, and wNa16 is the weight-only/GGUF cold path,
   not FP8 decode. Marginal line savings for real risk on a kernel I can't compile
   locally. If done, do the safe single+batch merge per bit width only.
4. fp4 route/grouped (4 kernels) — higher risk (route_meta + ptr-table indirection),
   also off the FP8 hot path.

A clean idle-box perf A/B for the fp8 merges (inc 1+2) is still owed once the box
is uncontended.

## Rule

**A compile-time template merge of byte-identical kernels is gated by
correct-inference, not a perf number** — functors inline to the same SASS, so
perf-neutrality is structural. Verify the merge serves every affected FFI export
correctly (needle + concurrent B>1 to hit the tiled path); batch the clean
idle-box perf A/B once for the whole refactor. Don't pass off a contended-box
throughput number as a perf pass — a starved GPU (low util, full clock) exonerates
the kernel more than a raw tok/s would.
