# W4AFP8 TP=4 decode: 47.7 tok/s — CUDA, 2026-08-19

> Status: Shipped

## Context

W4AFP8 TP=2 serves DSv4-Flash-0731 at 36.9 tok/s decode (same bench, same
day). FP8 TP=8 serves the same model at 53 tok/s. The user asked whether
TP=4 closes the gap.

## Result

Same binary (`w4afp8-tp4`, built from `8c1a54948`), same bench script
(`bench_dsv4_trace_http.py`), same model, same pod:

| Config | GPUs | decode64 tok/s | write_zh tok/s |
|--------|------|---------------:|---------------:|
| W4AFP8 TP=2 | 2 | 36.9 | 37.3 |
| W4AFP8 TP=4 | 4 | **47.7** | **48.5** |
| FP8 TP=8 | 8 | 53 | — |

TP=4 is 1.29× over TP=2, 10% below FP8 TP=8 in absolute tok/s. Per-GPU
efficiency: W4AFP8 TP=4 = 11.9 tok/s/GPU vs FP8 TP=8 = 6.6 tok/s/GPU.

Prefill at TP=4 is slower than TP=2 (4-way NCCL overhead on small M):
prefill4k 1327 tok/s vs TP=2's 3673 tok/s. Decode is the target workload.

## Rule

W4AFP8 decode scales sub-linearly with TP (1.29× per 2× GPU) because the
CUTLASS grouped GEMM is M=1 tile-bound, not weight-bandwidth-bound. More
TP buys less than a faster decode kernel would.
