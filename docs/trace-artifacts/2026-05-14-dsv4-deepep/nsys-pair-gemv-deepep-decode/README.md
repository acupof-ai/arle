# DSv4 Grouped Pair GEMV Decode Nsys

Date: 2026-05-14

Remote workspace: `/root/arle-perf-pair-gemv`

Raw Nsight files retained remotely:

- `/root/arle-perf-pair-gemv/docs/trace-artifacts/2026-05-14-dsv4-deepep/nsys-pair-gemv-deepep-decode/trace.nsys-rep`
- `/root/arle-perf-pair-gemv/docs/trace-artifacts/2026-05-14-dsv4-deepep/nsys-pair-gemv-deepep-decode/trace.sqlite`

The committed files are the light `nsys stats` exports plus client JSON. The
server was launched with:

```bash
ARLE_DSV4_MOE_BACKEND=deepep \
ARLE_DSV4_INCREMENTAL_KV=1 \
ARLE_DSV4_GROUPED_EXPERTS=1 \
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 \
/root/arle/target/release/infer \
  --model-path /root/DeepSeek-V4-Flash \
  --port 18099 \
  --num-slots 1 \
  --max-seq-len 4096 \
  --mem-fraction-static 0.10 \
  --kv-cache-dtype fp8 \
  --deepseek-distributed-layers 43
```

## Window

The client first issued an unprofiled warmup request, then used streaming to
start `cuProfilerStart` after the first emitted token and stop after the next
emitted token. The streamed chunks were:

| Chunk | Text |
| --- | --- |
| First, before profiler start | `一` |
| Second, before profiler stop | `,` |

The measured client gap between those chunks was 0.320807 s. Nsight captured
16 `step_decode_kernel_launch` ranges, which corresponds to 8 ranks and about
two decode scheduler steps in this external signal window.

## Result

The trace confirms the new opt-in pair kernel is actually running on the
DeepEP grouped expert route:

| Kernel | GPU time share | Instances | Median |
| --- | ---: | ---: | ---: |
| `ncclDevKernel_SendRecv` | 46.9% | 1025 | 40.704 us |
| `dsv4_fp4_grouped_gemv_pair_batch_kernel` | 14.6% | 195 | 726.400 us |
| `dsv4_fp4_grouped_gemv_batch_kernel` | 7.2% | 194 | 358.433 us |
| `dsv4_fp8_gemv_batch_kernel` | 7.1% | 2900 | 27.392 us |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 5.1% | 339 | 23.553 us |
| `dsv4_hybrid_attention_kernel` | 4.6% | 309 | 232.736 us |
| `ncclDevKernel_AllGather_RING_LL` | 1.7% | 336 | 50.033 us |

Top CUDA API time is still dominated by temporary allocation, free, D2H, launch,
and memset overhead:

| CUDA API | API time share | Calls |
| --- | ---: | ---: |
| `cuMemAllocAsync` | 26.6% | 12720 |
| `cuMemFreeAsync` | 24.9% | 12717 |
| `cuMemcpyDtoHAsync_v2` | 23.6% | 940 |
| `cudaLaunchKernel` | 7.9% | 15461 |
| `cuMemsetD8Async` | 7.5% | 13836 |

## Conclusion

Fusing grouped `w1` and `w3` into one pair GEMV launch works, but it is not a
production win yet. The opt-in grouped path remains slower than the default
scratch-reuse DeepEP path because decode is still dominated by return-side
NCCL send/recv, allocation/free churn, small launch overheads, and raw GEMV
local expert compute. The next useful replacement is still true grouped
GEMM/DeepGEMM with communication overlap, not enabling the raw grouped GEMV
harness by default.
