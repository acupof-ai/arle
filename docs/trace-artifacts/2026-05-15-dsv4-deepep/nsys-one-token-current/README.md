# DSv4 Single-Token Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. The remote source workspace was `/root/arle-perf-recv-route-scratch`
with the route-logits scratch cleanup build, and the service was launched under
`nsys profile` with CUDA profiler API start/stop.

The profiled request used streaming `max_tokens=2` and returned two completion
tokens: `霓灯`. The Nsight decode filter found one
`step_decode_kernel_launch` wave across 8 rank threads, so the tables below are
for one generated decode token, normalized per rank/range.

## Run Shape

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 \
ARLE_DSV4_MOE_BACKEND=deepep \
ARLE_DSV4_INCREMENTAL_KV=1 \
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1 \
nsys profile \
  --trace cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --export=sqlite \
  /root/arle/target/release/infer \
    --model-path /root/DeepSeek-V4-Flash \
    --port 18131 \
    --num-slots 1 \
    --max-seq-len 4096 \
    --mem-fraction-static 0.10 \
    --kv-cache-dtype fp8 \
    --deepseek-distributed-layers 43
```

## Result

| Metric | Value |
| --- | ---: |
| Prompt tokens | 16 |
| Completion tokens | 2 |
| Captured decode waves | 1 |
| Decode ranges | 8 |
| Decode wave wall time | 158.439 ms |
| Per-rank decode range p50 | 158.161 ms |

Top CUDA runtime API time inside the decode-token NVTX ranges:

| API | Time per rank range | Calls |
| --- | ---: | ---: |
| `cuMemFreeAsync` | 26.393 ms | 9144 |
| `cuMemAllocAsync` | 26.160 ms | 9136 |
| `cudaLaunchKernel_v7000` | 21.778 ms | 15056 |
| `cuMemcpyDtoHAsync_v2` | 20.196 ms | 871 |
| `cuMemsetD8Async` | 14.458 ms | 10182 |
| `cudaEventRecord_v3020` | 5.133 ms | 4136 |
| `cudaStreamWaitEvent_v3020` | 4.394 ms | 3440 |

Top CUDA kernels inside the same decode-token window:

| Kernel | Time per rank range | Calls |
| --- | ---: | ---: |
| `ncclDevKernel_SendRecv` | 29.018 ms | 1032 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 19.996 ms | 344 |
| `dsv4_fp8_gemv_batch_kernel` | 11.475 ms | 2920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.870 ms | 774 |
| `dsv4_hybrid_attention_kernel` | 7.355 ms | 328 |
| `dsv4_route_kernel` | 5.659 ms | 344 |
| `dsv4_mhc_params_kernel` | 5.503 ms | 688 |

## Bottleneck

The isolated token is still not sampler-bound and not missing KV-cache reads.
The DSv4 hybrid attention kernel is visible, but it is smaller than the combined
runtime overhead and communication work.

The concrete bottleneck stack in this capture is:

1. temporary allocation/free churn, about 52.6 ms per rank range;
2. small-kernel launch and memset overhead, about 36.2 ms per rank range;
3. D2H routing readbacks, about 20.2 ms per rank range;
4. NCCL SendRecv and AllReduce kernels, about 49.0 ms per rank range;
5. local expert FP8/FP4 GEMV kernels, about 22.3 ms per rank range;
6. attention and MHC kernels, about 17.0 ms per rank range.

Compared with the first committed single-token trace, scratch reuse reduced the
isolated decode wave from 266.020 ms to 158.439 ms and allocator calls from
11,980/11,988 `cuMemAllocAsync`/`cuMemFreeAsync` to 9,136/9,144. The remaining
performance work is still allocator lifetime or graph capture, fewer D2H route
decisions, DeepEP-style overlap or lower-latency exchange, and replacing the
per-expert GEMV path with true grouped GEMM/DeepGEMM.

Raw trace files are committed here as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`

