# DSv4 Single Decode Token MoE Scratch Uninit

This artifact profiles one generated decode token on the real 8xH20
`/root/DeepSeek-V4-Flash` checkpoint after switching additional MoE dispatch,
receive, local-route, payload, and combine scratch allocations from zeroed
allocation to uninitialized allocation.

The request was:

```text
Compute 137 + 269. Answer with the number only.
```

The HTTP response was `406`.

Compared with
[`../nsys-single-decode-token-expanded-uninit/`](../nsys-single-decode-token-expanded-uninit/):

- Single decode wave: 88.554 ms -> 87.667 ms.
- `cuMemsetD8Async`: 1,920 calls / 2.839 ms per rank range -> 1,232 calls /
  1.558 ms.
- `cuMemAllocAsync`: 5,040 calls / 5.611 ms -> 5,040 calls / 4.992 ms.
- D2H activity remains tiny: 344 calls / 44,032 bytes.

The remaining slow stack is unchanged:

- `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL`: 20.503 ms per rank range.
- Local expert GEMV: FP8 11.470 ms plus FP4 11.101 ms per rank range.
- Attention/MHC/route kernels: 7.393 ms, 5.501 ms, and 5.659 ms.
- CUDA runtime launch overhead: 16,177 `cudaLaunchKernel_v7000` calls taking
  28.329 ms per rank range.

This is a scratch/lifetime cleanup. It does not replace the core need for
DeepEP combine reduction improvements and true grouped GEMM/DeepGEMM expert
execution.
