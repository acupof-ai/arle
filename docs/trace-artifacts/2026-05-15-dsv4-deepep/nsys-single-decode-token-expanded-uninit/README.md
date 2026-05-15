# DSv4 Single Decode Token Expanded Uninit

This artifact profiles one generated decode token on the real 8xH20
`/root/DeepSeek-V4-Flash` checkpoint after switching additional full-write
runtime scratch buffers from zeroed allocation to uninitialized allocation.

The request was:

```text
Compute 137 + 269. Answer with the number only.
```

The HTTP response was `406`.

Compared with
[`../nsys-single-decode-token-current-breakdown/`](../nsys-single-decode-token-current-breakdown/):

- Single decode wave: 105.205 ms -> 88.554 ms.
- `cuMemsetD8Async`: 3,640 calls / 6.932 ms per rank range -> 1,920 calls /
  2.839 ms.
- `cuMemAllocAsync`: 5,040 calls / 6.897 ms -> 5,040 calls / 5.611 ms.
- D2H activity remains tiny: 344 calls / 44,032 bytes.

The remaining slow stack is still not sampler time:

- `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL`: 20.342 ms per rank range.
- Local expert GEMV: FP8 11.477 ms plus FP4 11.108 ms per rank range.
- Attention/MHC/route kernels: 7.393 ms, 5.502 ms, and 5.659 ms.
- CUDA runtime launch overhead: 16,177 `cudaLaunchKernel_v7000` calls taking
  29.349 ms per rank range.

This confirms scratch zeroing was real overhead, but the main decode target
remains DeepEP combine reduction plus replacing route/local expert GEMV with
true grouped GEMM/DeepGEMM-style execution.
