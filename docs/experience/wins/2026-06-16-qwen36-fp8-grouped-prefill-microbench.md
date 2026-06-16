# Qwen3.6 FP8 grouped prefill microbench isolates JIT, not kernel throughput

## SLO-shape probed? N

This is an isolated kernel-chain probe, not an HTTP/guidellm SLO run. It
directly calls the FP8 grouped prefill path once on dummy large-M inputs and
compares Qwen3.6 against the working DSv4 grouped path.

## Goal

Stop diagnosing the Qwen3.6 FP8 SLO slowdown through slow e2e serve sweeps.
Measure the grouped FP8 prefill chain itself: dispatch, BF16->FP8 activation
quantize, DeepGEMM cold-vs-cached JIT, grouped GEMM time, and row-packing
parameters.

## Implementation

Added `crates/infer-cuda/examples/fp8_grouped_prefill_probe.rs`.

The probe bypasses HTTP, scheduler, tokenizer, loader, routing, and KV state. It
calls the same production wrappers used by the large routed-row FP8 MoE lane:

1. `dsv4_deepgemm_pack_quantize_bf16_to_fp8`
2. `dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous` for fused w13
3. `dsv4_deepgemm_swiglu_quantize_w13`
4. `dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous` for down

It logs host and CUDA-event time for cold and cached passes in the same process.
The JIT cache is controlled with `DG_JIT_CACHE_DIR`.

## Environment

- Remote tree: `/data01/arle-qwenfp8-smoke`.
- Binary: `target/release/examples/fp8_grouped_prefill_probe`.
- Build: `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`,
  `ARLE_CUDA_KERNEL_SET=dsv4_flash`, `ARLE_CUDA_DISABLE_FLASHMLA=1`.
- Runtime CUDA: `/usr/local/cuda`; DeepGEMM JIT CUDA:
  `/usr/local/cuda-12.9`; `NVCC_CCBIN=/usr/bin/clang++-11`.
- Qwen probe: GPU2, `tokens=4096`, `topk=8`, `experts=256`,
  `hidden=2048`, `intermediate=512`.
- DSv4 probe: GPU3, `tokens=4096`, `topk=6`, `experts=32`,
  `hidden=4096`, `intermediate=2048`.

## Results

Both probes dispatched to `DeepGEMM_contiguous_fp8`; no scalar/hand fallback
occurred.

| Kind | Phase | pack ms | w13 GEMM ms | swiglu ms | down GEMM ms | total CUDA ms |
|---|---|---:|---:|---:|---:|---:|
| Qwen | cold | 1.754 | 4352.023 | 0.669 | 2661.449 | 7015.895 |
| Qwen | cached | 1.636 | 0.724 | 0.641 | 0.377 | 3.378 |
| DSv4 | cold | 1.564 | 4287.933 | 1.143 | 4019.240 | 8309.879 |
| DSv4 | cached | 1.435 | 3.211 | 1.115 | 1.640 | 7.401 |

Qwen row-packing parameters:

| tokens | topk | routes | rows_cap | aligned_total | pad rows | max_count | scale_stride_m |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 8 | 32768 | 65536 | 32768 | 32768 | 128 | 65536 |

DSv4 row-packing parameters:

| tokens | topk | routes | rows_cap | aligned_total | pad rows | max_count | scale_stride_m |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 6 | 24576 | 28672 | 24576 | 4096 | 768 | 28672 |

## Root Cause

The grouped FP8 DeepGEMM kernel is not the hot cached bottleneck: Qwen's full
grouped chain is 3.378 ms cached for the large dummy shape. Pack/requant are
small, and the GEMMs are sub-millisecond cached. The pathological multi-second
part is cold JIT compilation of the two grouped GEMM shapes.

This mirrors the earlier dense-projection JIT misuse: a genuinely empty
DeepGEMM cache can still put grouped w13/down JIT on the first long-prefill
request unless the grouped shapes are warmed before serving traffic.

## Fix

Added Qwen boot-time grouped DeepGEMM warmup:

- `Qwen35Model::warm_fp8_deepgemm_grouped_prefill`
- `Qwen35CudaExecutor::warmup` now calls it after dense FP8 DeepGEMM warmup.
- Warmup compiles only the two JIT-backed grouped GEMMs (`w13`, `down`) for the
  default CUDA prefill chunk shape and its page-tail neighbor (`2048` and
  `2032` tokens). Pack/requant kernels are static CUDA kernels and are not
  warmed.
- The warmup does not touch KV, slot state, request state, or real routing.

## Warm-cache verification

With a fresh `DG_JIT_CACHE_DIR`, the Qwen production chunk shape
(`tokens=2048`, `rows=49152`) paid cold grouped JIT once:

| Run | w13 cold ms | down cold ms | total cold CUDA ms | cached CUDA ms |
|---|---:|---:|---:|---:|
| Fill fresh cache, 2048 tokens | 4537.600 | 2381.892 | 6921.379 | 2.362 |
| Reuse same cache, 2048 tokens, new process | 7.812 | 5.791 | 15.480 | 2.335 |
| Fill fresh cache, 2032 tokens | 4882.411 | 2392.926 | 7277.260 | 2.342 |
| Reuse same cache, 2032 tokens, new process | 7.534 | 5.637 | 15.025 | 2.323 |

The same cache contained two compiled cubins, matching the two grouped GEMM
shapes for that one token count. The production warmup covers both common 4K
prefill chunks (`2048` and `2032`), so it compiles four cubins total for Qwen3.6
FP8 grouped MoE. In the real serve process, the boot warmup keeps those runtimes
resident after compilation, so the first request should hit the cached path
rather than the multi-second JIT path.

## Problems

A direct 35B serve-startup verification with fresh `DG_JIT_CACHE_DIR` was not a
cheap check: it timed out after 300 seconds before model warmup, only reaching
tokenizer/model startup logs. No request was sent. The isolated probe remains
the decisive diagnostic for this change; one e2e SLO A/B is still required after
the full serving startup path can be exercised cheaply.

## Rule

For DeepGEMM ports, resident weights and correct dispatch are not enough. Every
request-reachable JIT shape needs an explicit boot-time warmup or a measured
proof that the first production request cannot hit an empty JIT cache.
