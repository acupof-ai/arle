# DSv4 Pair GEMV Nsight Trace

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash`
checkpoint on 8xH20. This trace tested a single-expert `w1`/`w3` pair GEMV
kernel in the default local expert loop. The capture was taken before the
experiment was gated; in current code the same path is equivalent to setting
`ARLE_DSV4_PAIR_EXPERT_GEMV=1`.

```text
prompt: 用两个字形容彩虹。
output: 霓彩
```

## Result

| Metric | Value |
| --- | ---: |
| Captured decode waves | 1 |
| Decode ranges | 8 |
| Decode wave wall time | 127.412 ms |
| Per-rank decode range p50 | 127.213 ms |
| Profiled request wall time | 1.421 s |
| Returned text | `霓彩` |

Top CUDA kernels inside the decode-token window:

| Kernel | Time per rank range | Calls |
| --- | ---: | ---: |
| `ncclDevKernel_SendRecv` | 45.794 ms | 1032 |
| `dsv4_fp4_gemv_pair_batch_kernel` | 23.207 ms | 258 |
| `dsv4_fp8_gemv_batch_kernel` | 11.473 ms | 2920 |
| `dsv4_hybrid_attention_kernel` | 6.953 ms | 328 |
| `dsv4_route_kernel` | 5.657 ms | 344 |
| `dsv4_mhc_params_kernel` | 5.498 ms | 688 |
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 3.549 ms | 344 |
| `dsv4_fp4_gemv_batch_tiled_kernel` | 3.349 ms | 258 |

## Decision

This is a negative experiment and is not enabled by default. The pair kernel
does reduce the number of FP4 GEMV launches, but the fused FP4 kernel costs
more than the default split GEMV work on this B=1 decode shape. It also does
not address the larger communication/runtime stack. The shipped path keeps the
old split GEMV behavior unless `ARLE_DSV4_PAIR_EXPERT_GEMV=1` is set.

The useful signal from this trace is that simple gate/up fusion is not enough;
the next compute-side path still needs real grouped GEMM/DeepGEMM or a
different decode-specialized expert kernel, plus DeepEP-style overlap for the
NCCL exchange.

Raw trace files are committed as compressed artifacts:

- `trace.nsys-rep.gz`
- `trace.sqlite.gz`
- `server.log.gz`
