# DSv4 recv/local route scratch cleanup

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. This run validates the cleanup that reuses DeepEP decode
recv/local route buffers:

- received hidden rows and metadata;
- local expert packed hidden rows, weights, and route slots;
- per-rank route outputs before the return-side combine exchange.

The scratch is only used for B=1 decode. During prefill, the runtime allocates
a small decode scratch capacity of `ep_world * topk` routes, not a capacity
derived from prompt length, so long-prefill requests do not keep a large route
buffer resident.

## Functional smoke

Trace-off DeepEP serving command shape:

```bash
ARLE_DSV4_MOE_BACKEND=deepep \
ARLE_DSV4_INCREMENTAL_KV=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 \
/root/arle/target/release/infer \
  --model-path /root/DeepSeek-V4-Flash \
  --port 18118 \
  --num-slots 1 \
  --max-seq-len 4096 \
  --mem-fraction-static 0.10 \
  --kv-cache-dtype fp8 \
  --deepseek-distributed-layers 43
```

| Case | Prompt tokens | Completion tokens | Latency | Completion tok/s | Output |
| --- | ---: | ---: | ---: | ---: | --- |
| warmup | 13 | 2 | 0.457 s | 4.38 | `1+` |
| `37*29` | 19 | 12 | 1.366 s | 8.79 | `37 × 29 = 1073。  \n解释：` |
| `58+67` | 19 | 12 | 1.397 s | 8.59 | `58 + 67 = 125。  \n解释：先` |
| writing | 20 | 10 | 1.214 s | 8.24 | `毫秒级洞察，极致算力释放。` |

## Single-token nsys

The profiled streaming request used `max_tokens=2`, returned `好的，`, and
captured one `step_decode_kernel_launch` wave across 8 rank threads. Grouped
expert mode was disabled to match the previous default DeepEP trace.

| Metric | Send-route scratch | Recv/local route scratch |
| --- | ---: | ---: |
| Decode wave wall time | 191.152 ms | 148.253 ms |
| Per-rank decode range p50 | 190.959 ms | 147.188 ms |
| `cuMemAllocAsync` calls | 11097 | 9480 |
| `cuMemFreeAsync` calls | 11105 | 9488 |
| `cuMemsetD8Async` calls | 12167 | 10554 |
| `cudaLaunchKernel_v7000` calls | 15080 | 15084 |

Top decode runtime APIs after this cleanup:

| API | Time per rank range | Calls |
| --- | ---: | ---: |
| `cuMemFreeAsync` | 27.707 ms | 9488 |
| `cuMemAllocAsync` | 27.145 ms | 9480 |
| `cuMemcpyDtoHAsync_v2` | 20.153 ms | 885 |
| `cudaLaunchKernel_v7000` | 18.188 ms | 15084 |
| `cuMemsetD8Async` | 12.172 ms | 10554 |

Top decode kernels after this cleanup:

| Kernel | Time per rank range | Calls |
| --- | ---: | ---: |
| `ncclDevKernel_SendRecv` | 26.189 ms | 1032 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 18.786 ms | 344 |
| `dsv4_fp8_gemv_batch_kernel` | 11.476 ms | 2920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.871 ms | 774 |
| `dsv4_hybrid_attention_kernel` | 7.110 ms | 328 |

The remaining decode bottleneck is still not KV or sampler. KV-backed hybrid
attention is visible and smaller than the combined runtime and communication
cost. The largest remaining work is allocator lifetime, host D2H routing
decisions, many small launches/memsets, NCCL send/recv and all-reduce
boundaries, then the per-expert FP8/FP4 GEMV kernels.

Raw trace files are committed here as compressed artifacts:

- `nsys/trace.nsys-rep.gz`
- `nsys/trace.sqlite.gz`

The uncompressed copies also remain on the remote host under:

`/root/arle-perf-recv-route-scratch/docs/trace-artifacts/2026-05-14-dsv4-deepep/recv-route-scratch/nsys/`
