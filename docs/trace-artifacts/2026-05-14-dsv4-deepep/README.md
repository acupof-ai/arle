# DSv4 DeepEP Decode Trace

Date: 2026-05-14

Remote workspace: `/root/arle`

Model: `/root/DeepSeek-V4-Flash`

Runtime command shape:

```bash
INFER_PREFILL_WARMUP=0 \
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

## Current Bottleneck

The current decode bottleneck is still model compute and per-layer routing/GEMM orchestration, not
the old full hidden all-reduce. In the warm trace, median `ffn_deepep_dispatch_combine` is roughly
1.865 ms per layer/rank, while the overall decode step is roughly 143 ms/token. Remaining high-value
work:

- Replace per-expert looped GEMMs with grouped GEMM/DeepGEMM.
- Reuse DSv4 MoE scratch buffers across layers/decode steps to remove allocator churn.
- Enable CUDA Graph/PDL after allocation and dynamic NCCL paths are made graph-safe.
- Add B>1 vectorized decode after B=1 dispatch/combine is stable.
- Add MTP after the decode path is vectorized and the sampler path is stable.

