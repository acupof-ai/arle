# DSv4 send-route scratch cleanup

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. This run validates the cleanup that moves DeepEP send-token and
send-route-slot buffers into reusable per-layer scratch and removes the unused
`expert_token` output from `dsv4_pack_received_experts_cuda`.

## Functional smoke

Trace-off DeepEP serving command shape:

```bash
ARLE_DSV4_MOE_BACKEND=deepep \
ARLE_DSV4_INCREMENTAL_KV=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 \
/root/arle/target/release/infer \
  --model-path /root/DeepSeek-V4-Flash \
  --port 18111 \
  --num-slots 1 \
  --max-seq-len 4096 \
  --mem-fraction-static 0.10 \
  --kv-cache-dtype fp8 \
  --deepseek-distributed-layers 43
```

| Case | Prompt tokens | Completion tokens | Latency | Completion tok/s | Output |
| --- | ---: | ---: | ---: | ---: | --- |
| warmup | 12 | 2 | 0.459 s | 4.36 | `2` |
| `37*29` | 17 | 12 | 1.506 s | 7.97 | `37 × 29 = 1073。  \n解释：` |
| `58+67` | 17 | 12 | 1.511 s | 7.94 | `答案：125  \n解释：先算 50+60` |
| writing | 13 | 10 | 1.236 s | 8.09 | `算力奔涌，逻辑如刃，在` |

## Single-token nsys

The profiled streaming request used `max_tokens=2`, returned `霓灯`, and
captured one `step_decode_kernel_launch` wave across 8 rank threads.

| Metric | Value |
| --- | ---: |
| Decode wave wall time | 191.152 ms |
| Per-rank decode range p50 | 190.959 ms |
| `cuMemAllocAsync` calls | 11097 |
| `cuMemFreeAsync` calls | 11105 |
| `cudaLaunchKernel_v7000` calls | 15080 |

The previous current single-token artifact recorded `cuMemAllocAsync=11980`
and `cuMemFreeAsync=11988` in the same 8-rank decode-wave shape. This cleanup
therefore removes roughly 883 allocator calls from the isolated token window.
Allocator elapsed time is still noisy under nsys, so call count is the stable
signal for this patch; the remaining allocator pressure is dominated by other
per-layer buffers that still need scratch reuse or graph-safe lifetime work.

Top decode kernels after this cleanup:

| Kernel | Time per rank range | Calls |
| --- | ---: | ---: |
| `ncclDevKernel_SendRecv` | 33.657 ms | 1032 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 28.449 ms | 344 |
| `dsv4_fp8_gemv_batch_kernel` | 11.470 ms | 2920 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.875 ms | 774 |
| `dsv4_hybrid_attention_kernel` | 7.842 ms | 328 |

Raw `.nsys-rep` and `.sqlite` stay on the remote host under:

`/root/arle-perf-send-slot-scratch/nsys-send-slot-scratch/`

Committed local artifacts include smoke JSON/logs, single-token nsys summaries,
decode-only CSVs, request profiles, and command logs.
