# V100 (sm_70) BF16 GEMM — pre-allocate scratch, eliminate per-call malloc/free+sync

> Status: Shipped 2026-07-15 (commit 2b06a04e4). Build PASS, kernel
> correctness PASS, end-to-end correctness PASS, throughput verified on
> V100.

## Context

V100 (sm_70) has no BF16 tensor cores — BF16 GEMM falls back to
`gemm_fp16_cast_cuda` in `crates/cuda-kernels/csrc/gemm/gemv.cu`, which
converts BF16→FP16, runs FP16 `cublasGemmEx`, converts FP16→BF16. The
fallback allocated three scratch buffers (W_f16, X_f16, Y_f16) with
`cudaMalloc` per GEMM call, ran a `cudaStreamSynchronize` after the
conversion kernels, then `cudaFree`'d all three.

With ~24 layers × ~4 GEMMs/layer ≈ 576 GEMMs per token, that was ~1728
`sync` was the real killer: it drained the pipeline every GEMM, so
concurrent requests could not overlap. Throughput *decreased* with
concurrency (c=4 76 tok/s, c=8 69 tok/s, c=16 60 tok/s) on a 32 GB V100
serving a 0.8B model — far below what the GPU should deliver.

## Fix

Pre-allocate a single per-device scratch buffer (`cast_scratch` +
`cast_scratch_bytes` in `CublasDeviceState`), grown on demand with
double-checked locking (`g_state_mutex`). The common case (scratch large
enough) is lock-free; the grow path takes the mutex, frees the old
buffer, and allocates the next 1 MB-rounded size. `gemm_fp16_cast_cuda`
now slices this single buffer into W_f16/X_f16/Y_f16 and runs the
conversion kernels + GEMM with no per-call `cudaMalloc`, `cudaFree`, or
`cudaStreamSynchronize`. The buffer is freed once in `cublas_destroy`.

## Results

- **Build: PASS** (V100 sm_70, `cargo build --release --features cuda`,
  12m 07s).
- **Kernel correctness: PASS** (V100 sm_70,
  `cargo test -p cuda-kernels --release --features cuda -- w4a16`,
  0.25s).
- **End-to-end correctness: PASS** (V100 sm_70, Qwen3.5-0.8B BF16,
  greedy). "2+2=?"→"2+2=4", "capital of France"→"Paris" — correct
  output, not garbage.
- **Throughput: measured** (V100 sm_70, Qwen3.5-0.8B BF16, 20 prompts ×
  200 tok, token-counted):

  | c | Pre-fix tok/s | Post-fix tok/s | Δ |
  |---|---------------|----------------|---|
  | 1 | —             | 98             | — |
  | 4 | 76            | 125            | +64% |
  | 8 | 69            | 186            | +170% |

  Concurrency now scales correctly instead of inverting.

## Rule

- A `cudaStreamSynchronize` per GEMM in the decode hot loop serializes the
  entire pipeline — concurrent requests can never overlap. Pre-allocate
  scratch once and reuse; never alloc/free/sync per call in the hot path
  (§0.3: design for the fast path, not the fallback).
- sm_70 (no BF16 tensor cores) is the only platform that exercises the
  cast fallback — the sm_80+ hot path uses native BF16 `cublasGemmEx` and
  never touches it. Validate on real sm_70 hardware, not just sm_80+
  where the fallback is dead code.
- Double-checked locking: the common case (scratch large enough) is
  lock-free; only the rare grow path takes `g_state_mutex`.
