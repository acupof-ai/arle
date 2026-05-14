# DSv4 route-logits scratch cleanup

Captured on 2026-05-15 against the real `/root/DeepSeek-V4-Flash` checkpoint
on 8xH20. This run validates the follow-up cleanup that reuses the B=1 decode
MoE gate-logits output buffer. Prefill preallocates only a one-token route
logits buffer, so the first generated decode token does not allocate it inside
the decode NVTX range.

## Functional smoke

Trace-off DeepEP serving kept grouped experts disabled to match the default
path.

| Case | Prompt tokens | Completion tokens | Latency | Completion tok/s | Output |
| --- | ---: | ---: | ---: | ---: | --- |
| warmup | 13 | 2 | 0.460 s | 4.35 | `1+` |
| `37*29` | 19 | 12 | 1.392 s | 8.62 | `37 × 29 = 1073。  \n解释：` |
| `58+67` | 19 | 12 | 1.392 s | 8.62 | `58 + 67 = 125。  \n解释：先` |
| writing | 20 | 10 | 1.223 s | 8.18 | `毫秒级洞察，极致算力释放。` |

## Single-token nsys

The profiled streaming request used `max_tokens=2`, returned `好的，`, and
captured one `step_decode_kernel_launch` wave across 8 rank threads.

| Metric | Recv/local route scratch | Route-logits scratch |
| --- | ---: | ---: |
| Decode wave wall time | 148.253 ms | 162.062 ms |
| Per-rank decode range p50 | 147.188 ms | 160.841 ms |
| `cuMemAllocAsync` calls | 9480 | 9136 |
| `cuMemFreeAsync` calls | 9488 | 9144 |
| `cuMemsetD8Async` calls | 10554 | 10210 |

This is an allocator-count cleanup, not a confirmed wall-time win. The decoded
token window is still dominated by runtime overhead and communication variance:
`cuMemAllocAsync`, `cuMemFreeAsync`, `cuMemcpyDtoHAsync_v2`, launch/memset
calls, and NCCL send/recv plus all-reduce remain ahead of the attention kernel.

Raw trace files are committed here as compressed artifacts:

- `nsys/trace.nsys-rep.gz`
- `nsys/trace.sqlite.gz`
