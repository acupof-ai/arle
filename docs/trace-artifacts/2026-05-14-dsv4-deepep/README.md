# DSv4 DeepEP Decode Trace

Date: 2026-05-14

Remote workspace: `/root/arle`

Model: `/root/DeepSeek-V4-Flash`

Runtime command shape:

```bash
ARLE_DSV4_TRACE_LAYER=1 \
ARLE_DSV4_MOE_BACKEND=deepep \
ARLE_DSV4_INCREMENTAL_KV=1 \
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7 \
./target/release/infer \
  --model-path /root/DeepSeek-V4-Flash \
  --port 18084 \
  --max-seq-len 900000 \
  --kv-cache-dtype fp8 \
  --num-slots 1 \
  --deepseek-distributed-layers 43 \
  --mem-fraction-static 0.1
```

DSv4 now disables startup prefill warmup through the model capability hook, so
`INFER_PREFILL_WARMUP=0` is no longer required for this run shape.

## Functional Result

- 8 CUDA workers loaded the real DSv4 checkpoint with `tp_rank=0..7`, `ep_rank=0..7`, `experts_per_rank=32`.
- HTTP `/v1/chat/completions` returned normal multi-token content.
- Math smoke: `37*29` returned `1073` with the expected explanation.
- Writing smoke returned non-repetitive Chinese prose.
- 900K context capacity was configured at startup: `max_seq_len=900000`, paged KV format `FP8E4M3`.

## MoE Route

The traced run confirms the MoE path is DeepEP-style dispatch/combine:

- `ffn_deepep_dispatch_combine` count: 1720 before metadata packing, 1720 after metadata packing.
- `ffn_deepep_dispatch` count: 3440 before metadata packing, 3440 after metadata packing.
- `ffn_deepep_combine` count: 1720 before metadata packing, 1720 after metadata packing.
- `ffn_all_reduce` count: 0.

## Performance Snapshot

Warm steady-state request, 14 prompt tokens + 8 generated tokens:

| Metric | Before route metadata packing | After route metadata packing |
| --- | ---: | ---: |
| End-to-end latency | 1.981 s | 1.888 s |
| Decode phase per step | 151.809 ms | 142.818 ms |
| `ffn_deepep_dispatch` p50 | 0.135 ms | 0.069 ms |
| `ffn_deepep_dispatch_combine` p50 | 2.030 ms | 1.865 ms |
| `ffn_total` p50 | 2.706 ms | 2.520 ms |
| Completion throughput | 4.038 tok/s | 4.238 tok/s |

The first traced request after process start had a one-time NCCL send/recv connection cost:
`layer=0 phase=ffn_deepep_count_exchange elapsed_ms=4689`. The warm request above excludes that
first-use initialization spike.

## Scratch Reuse Snapshot

After adding per-layer incremental MoE expert scratch reuse, a clean three-request
window produced normal content:

| Case | Prompt tokens | Completion tokens | Latency | Output |
| --- | ---: | ---: | ---: | --- |
| `37*29` | 17 | 8 | 1.979 s | `37乘以29等于1073。计算` |
| `58+67` | 17 | 8 | 1.988 s | `58加67等于125。计算过程` |
| writing | 17 | 10 | 2.528 s | `霓灯织夜，车流如河。` |

Clean trace phase medians from the same window:

| Phase | p50 | p95 |
| --- | ---: | ---: |
| `ffn_deepep_dispatch_combine` | 1.834 ms | 2.281 ms |
| `ffn_deepep_dispatch` | 0.058 ms | 0.115 ms |
| `ffn_deepep_count_exchange` | 0.119 ms | 0.297 ms |
| `ffn_deepep_local_experts` | 0.464 ms | 0.886 ms |
| `ffn_deepep_combine` | 0.723 ms | 1.286 ms |
| `ffn_total` | 2.519 ms | 2.989 ms |
| `attn_total` | 1.187 ms | 1.687 ms |

## Grouped Expert Experiment

A raw grouped FP8/FP4 GEMV expert path is available behind
`ARLE_DSV4_GROUPED_EXPERTS=1`. It packs active expert weight pointers and uses
one grouped launch per `w1`, `w3`, and `w2`, then scatters all route slots in one
kernel. The gate is off by default because the current raw GEMV prototype is
slower than the scratch-reuse per-expert path on B=1 decode.

| Path | `ARLE_DSV4_GROUPED_EXPERTS` | Math latency | Writing latency | `ffn_deepep_local_experts` p50 | `ffn_total` p50 |
| --- | --- | ---: | ---: | ---: | ---: |
| Scratch reuse baseline | unset | 1.98-2.53 s | 2.53 s | 0.464 ms | 2.519 ms |
| Grouped raw GEMV, all local experts | `1` | 3.58 s | 4.27 s | 1.549 ms | 4.181 ms |
| Grouped raw GEMV, active experts only | `1` | 3.40-3.48 s | 4.56 s | 1.301 ms | 4.085 ms |
| Gated default validation | unset | 2.04-2.16 s | 2.51 s | 0.463 ms | 2.478 ms |

The grouped-kernel harness is kept for the next replacement step: swap the raw
GEMV implementation for real grouped GEMM/DeepGEMM rather than enabling the
slower prototype.

## Count Exchange Optimization

The tiny per-layer `i32[ep_world]` route-count exchange now defaults to NCCL
all-gather. The previous grouped send/recv path is still available with
`ARLE_DSV4_COUNT_EXCHANGE=sendrecv`.

Matched same-build A/B on 8xH20:

| Count exchange | Math latency | Writing latency | `ffn_deepep_count_exchange` p50 | `ffn_deepep_count_exchange` p95 | `ffn_total` p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all-gather default | 1.86-1.94 s | 2.40 s | 0.115 ms | 0.271 ms | 2.334 ms |
| grouped send/recv fallback | 2.07-2.13 s | 2.58 s | 0.176 ms | 0.419 ms | 2.590 ms |

The output content was identical across both modes for the math and writing
smokes.

## Current Throughput and 900K Context

With per-layer trace disabled, the current default route (`DeepEP` MoE,
incremental KV, count all-gather) returns normal multi-token content:

| Case | Prompt tokens | Completion tokens | Latency | Completion throughput | Output |
| --- | ---: | ---: | ---: | ---: | --- |
| math + summary | 21 | 21 | 3.135 s | 6.70 tok/s | `计算结果：579  \n总结：三位数加法，个十百位分别相加，满十进位。` |
| writing | 18 | 45 | 6.127 s | 7.35 tok/s | `高性能推理系统，如疾风掠影，毫秒间解析海数据。...` |

A true long-context request with a 2,699,957-byte prompt admitted as 899,965
tokens and entered chunked prefill with `chunk_size=16384`, but did not finish
before the 300s non-streaming HTTP timeout. The service remained alive and all
8 GPUs were at 100% utilization after the client timeout, so this is now a
long-prefill throughput blocker rather than a sampler, HTTP fanout, or decode
synchronization blocker.

## Prefill Bottleneck Trace

A 1,039-token prompt with per-layer trace enabled completed in 17.178 s and
returned `37 × 29`. The trace separates the true prefill row (`tokens=1039`)
from the subsequent decode rows (`tokens=1`).

Top prefill phase totals across 43 layers and 8 ranks:

| Phase | Count | Sum | Avg per layer/rank | Max |
| --- | ---: | ---: | ---: | ---: |
| `ffn_total` | 344 | 114269.243 ms | 332.178 ms | 595.966 ms |
| `ffn_deepep_dispatch_combine` | 344 | 106179.338 ms | 308.661 ms | 571.507 ms |
| `ffn_deepep_combine` | 344 | 58949.560 ms | 171.365 ms | 560.340 ms |
| `ffn_deepep_local_experts` | 344 | 44305.389 ms | 128.795 ms | 549.584 ms |
| `attn_total` | 344 | 15854.440 ms | 46.088 ms | 143.814 ms |
| `ffn_deepep_dispatch` | 344 | 293.982 ms | 0.855 ms | 1.637 ms |
| `ffn_deepep_count_exchange` | 344 | 182.072 ms | 0.529 ms | 3.023 ms |

This makes the next prefill optimization target concrete: count exchange and
token dispatch are no longer first-order; prefill needs real grouped
GEMM/DeepGEMM for local experts plus a more efficient combine path.

## Route-Slot Prefill Combine

The combine path now preserves each packed route's original top-k slot during
rank dispatch and uses that slot for prefill output aggregation. B=1 decode
keeps the previous direct combine kernel to avoid adding extra launches.

Matched 1,039-token request with layer trace:

| Metric | Baseline | Route-slot combine | Delta |
| --- | ---: | ---: | ---: |
| End-to-end latency | 17.178 s | 16.826 s | -2.0% |
| `ffn_deepep_combine` avg | 171.365 ms | 160.129 ms | -6.6% |
| `ffn_deepep_dispatch_combine` avg | 308.661 ms | 299.119 ms | -3.1% |
| `ffn_total` avg | 332.178 ms | 322.577 ms | -2.9% |
| Output | `37 × 29` | `37 × 29` | correct |

This removes an avoidable route scan, but it is not the main prefill fix: the
phase remains dominated by MoE return all-to-all synchronization and local
expert GEMMs. The next optimization has to target DeepEP-style communication
overlap and real grouped GEMM/DeepGEMM rather than only the final combine
scatter.

## Combine Scratch and Trace Split

The route-combine path now reuses per-layer MoE scratch for `combine_recv` and
prefill `route_slot_out`. The same patch splits `ffn_deepep_combine` into:

- `ffn_deepep_combine_exchange`: return-side grouped BF16 send/recv.
- `ffn_deepep_combine_kernel`: route-slot zero/scatter/final sum.

The compressed full trace log is committed as
`arle-http-dsv4-combine-scratch-trace.log.gz`.

Trace-off functional smoke:

| Case | Prompt tokens | Completion tokens | Latency | Completion throughput | Output |
| --- | ---: | ---: | ---: | ---: | --- |
| `37*29` | 17 | 11 | 1.968 s | 5.59 tok/s | `37 乘以 29 等于 1073。` |
| `58+67` | 17 | 10 | 1.741 s | 5.75 tok/s | `58 加 67 等于 125。` |
| writing | 22 | 34 | 5.141 s | 6.61 tok/s | `毫秒级响应，千亿级吞吐。智能调度算力...` |

Matched 1,039-token trace request:

| Metric | Route-slot combine | Scratch + split trace | Delta |
| --- | ---: | ---: | ---: |
| End-to-end latency | 16.826 s | 16.488 s | -2.0% |
| `ffn_total` avg | 322.577 ms | 316.543 ms | -1.9% |
| `ffn_deepep_combine` avg | 160.129 ms | 157.944 ms | -1.4% |
| `ffn_deepep_combine_exchange` avg | n/a | 157.052 ms | dominant |
| `ffn_deepep_combine_kernel` avg | n/a | 0.438 ms | not bottleneck |
| Output | `37 × 29` | `37 × 29` | correct |

This confirms the current prefill combine bottleneck is not the final route
aggregation kernel. It is the return all-to-all exchange/synchronization.

## FP8 Combine Exchange Experiment

The return-side MoE combine exchange has a gated FP8 E4M3 experiment behind
`ARLE_DSV4_COMBINE_DTYPE=fp8`. It quantizes each BF16 route-output row with a
per-row FP32 scale, exchanges the FP8 payload plus scales, then dequantizes back
to BF16 before the existing route-slot combine kernel. The default remains the
BF16 exchange path.

The full FP8 trace log is committed as
`arle-http-dsv4-combine-fp8-trace.log.gz`; the parsed record is
`dsv4-combine-fp8-experiment-summary.json`.

Matched 1,039-token trace request:

| Metric | BF16 scratch default | FP8 combine experiment | Delta |
| --- | ---: | ---: | ---: |
| End-to-end latency | 16.488 s | 17.505 s | +6.2% |
| `ffn_total` avg | 316.543 ms | 318.873 ms | +0.7% |
| `ffn_deepep_dispatch_combine` avg | 293.370 ms | 295.244 ms | +0.6% |
| `ffn_deepep_combine` avg | 157.944 ms | 158.361 ms | +0.3% |
| `ffn_deepep_combine_exchange` avg | 157.052 ms | 156.936 ms | -0.1% |
| `ffn_deepep_combine_kernel` avg | 0.438 ms | 0.362 ms | -17.4% |
| Output | `37 × 29` | `37 × 29 = 1073` | correct |

Trace-off default post-check after this change:

| Case | Prompt tokens | Completion tokens | Latency | Completion throughput | Output |
| --- | ---: | ---: | ---: | ---: | --- |
| `37*29` | 19 | 11 | 1.997 s | 5.51 tok/s | `37 乘以 29 等于 1073。` |
| `58+67` | 19 | 10 | 1.795 s | 5.57 tok/s | `58 加 67 等于 125。` |
| writing | 20 | 19 | 3.142 s | 6.05 tok/s | `毫秒级响应，万亿级参数。算力极致调度，智能无界延伸。` |

Conclusion: FP8 combine is functionally correct but not a throughput win on the
current 8xH20 shape. The smaller payload is offset by the extra FP32 scale
exchange plus quantize/dequantize kernels, so the path stays opt-in. The
highest-value work remains real grouped GEMM/DeepGEMM for local experts and a
combine strategy that reduces or overlaps the return all-to-all synchronization.

## MHC Scratch Reuse

Incremental DSv4 attention and FFN now reuse per-layer HyperConnection/MHC
temporary buffers for the mix projection output plus `pre` / `post` / `comb`
parameter tensors. This removes repeated small allocations from every
attention/FFN layer after the first matching shape.

The full trace log is committed as `arle-http-dsv4-mhc-scratch-trace.log.gz`;
the parsed record is `dsv4-mhc-scratch-summary.json`.

Trace-off default route, same smoke set:

| Case | Before latency | After latency | Before throughput | After throughput | Output |
| --- | ---: | ---: | ---: | ---: | --- |
| `37*29` | 1.997 s | 1.733 s | 5.51 tok/s | 6.35 tok/s | `37 乘以 29 等于 1073。` |
| `58+67` | 1.795 s | 1.612 s | 5.57 tok/s | 6.20 tok/s | `58 加 67 等于 125。` |
| writing | 3.142 s | 2.590 s | 6.05 tok/s | 7.34 tok/s | `毫秒级响应，万亿级参数。算力极致调度，智能无界延伸。` |

Layer-trace phase comparison:

| Phase | Shape | Before p50 | After p50 | Before p95 | After p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| `attn_mhc` | decode `tokens=1` | 0.152 ms | 0.088 ms | 0.254 ms | 0.185 ms |
| `ffn_mhc` | decode `tokens=1` | 0.134 ms | 0.085 ms | 0.194 ms | 0.096 ms |
| `ffn_mhc` | prefill `tokens=1039` | 0.274 ms | 0.255 ms | 0.317 ms | 0.270 ms |

The 1,039-token traced request still returns `37 × 29 = 1073`. MHC scratch
reuse is a real decode cleanup, but it does not change the long-prefill
bottleneck: return-side MoE combine exchange and local expert GEMMs still
dominate.

## Current Bottleneck

The current decode bottleneck is still model compute and per-layer routing/GEMM orchestration, not
the old full hidden all-reduce. In the warm trace, median `ffn_deepep_dispatch_combine` is roughly
1.865 ms per layer/rank, while the overall decode step is roughly 143 ms/token. Remaining high-value
work:

- Replace chunked 900K prefill with a real high-throughput paged/varlen prefill
  path for DSv4; current 900K true prompt exceeds the 300s HTTP timeout.
- Replace per-expert looped GEMMs with grouped GEMM/DeepGEMM.
- Enable CUDA Graph/PDL after allocation and dynamic NCCL paths are made graph-safe.
- Add B>1 vectorized decode after B=1 dispatch/combine is stable.
- Add MTP after the decode path is vectorized and the sampler path is stable.

## Nsight Decode Trace

`nsys-single-token/` records a warmed 8xH20 HTTP decode profile captured with
`nsys launch/start/stop`. A first `max_tokens=2` attempt only produced one
completion token, so the actionable capture uses `max_tokens=8` and returns
seven completion tokens: `静水流深云淡风清`.

Decode waves in the Nsight capture are 257-270 ms wall each. Per GPU, summed
CUDA kernel time is only about 81-102 ms per wave, so the remaining decode wall
time is mostly host/runtime overhead: `cuStreamSynchronize`, async allocation
and free, kernel launch, memset, and small-message communication boundaries.

Top decode CUDA runtime API time per token/rank:

| API | Time |
| --- | ---: |
| `cuStreamSynchronize` | 92.605 ms |
| `cuMemFreeAsync` | 42.053 ms |
| `cuMemAllocAsync` | 20.331 ms |
| `cudaLaunchKernel` | 19.384 ms |
| `cuMemsetD8Async` | 16.968 ms |

Top decode CUDA kernel time per token/rank:

| Kernel | Time |
| --- | ---: |
| `ncclDevKernel_SendRecv` | 28.858 ms |
| `dsv4_fp8_gemv_batch_kernel` | 11.474 ms |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 10.871 ms |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 7.934 ms |
| `dsv4_hybrid_attention_kernel` | 7.406 ms |
| `ncclDevKernel_AllGather_RING_LL` | 6.026 ms |

The trace confirms the slow single-token decode is not explained by missing KV
reads alone. The attention kernel is present and costs about 7.4 ms/token/rank,
but the larger issue is that B=1 decode spends most wall time in synchronization,
temporary allocation/free, launch overhead, and MoE/NCCL boundaries around many
small kernels.
