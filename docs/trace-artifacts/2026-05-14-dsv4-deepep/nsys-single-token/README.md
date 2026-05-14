# DSv4 Nsight decode trace

Captured with `nsys launch/start/stop` around a warmed HTTP request on the remote H20 box. The profiled request used `max_tokens=8` and returned 7 completion tokens: `静水流深云淡风清`.

## Key result

Decode waves are 257-270 ms wall each. Per GPU, summed CUDA kernel time is only about 81-102 ms per wave, so the remaining wall time is dominated by host/API sync, async alloc/free, launch and small-message communication boundaries.

Top decode CUDA kernels per token/rank: NCCL SendRecv 28.858 ms, FP8 GEMV 11.474 ms, FP4 tiled GEMV 10.871 ms, NCCL AllReduce 7.934 ms, hybrid attention 7.406 ms, NCCL AllGather 6.026 ms.

Top CUDA runtime API time per token/rank: cuStreamSynchronize 92.605 ms, cuMemFreeAsync 42.053 ms, cuMemAllocAsync 20.331 ms, cudaLaunchKernel 19.384 ms, cuMemsetD8Async 16.968 ms.

The high-level layer trace still points to FFN/MoE as the largest model phase: ffn_total p50 2.881 ms/layer/rank, ffn_deepep_dispatch_combine p50 2.298 ms, attention total p50 1.396 ms.
