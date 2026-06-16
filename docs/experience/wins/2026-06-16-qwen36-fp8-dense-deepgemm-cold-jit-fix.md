# Qwen3.6 FP8 dense DeepGEMM cold-JIT fix

## Goal

Root-cause the 4096-token Qwen3.6 FP8 SLO regression where FP8 TTFT was
20.7s vs BF16 1.76s, without blaming H20 or FP8 as a method.

## Hypothesis

The Qwen FP8 port diverged from the working DSv4 DeepGEMM usage. The likely
misuse was one of: fallback to scalar FP8 GEMV, per-call BF16->FP8 quantization
dominating, DeepGEMM JIT compiling on the request path, max-m padding misuse, or
grouped/scale-layout mismatch.

## Environment

- Remote tree: `/data01/arle-qwenfp8-smoke`.
- Binary: `/data01/arle-qwenfp8-smoke/target/release/arle`.
- Hardware: H20, single GPU.
- Build: `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`,
  `ARLE_CUDA_KERNEL_SET=dsv4_flash`, `ARLE_CUDA_DISABLE_FLASHMLA=1`.
- Serve: `--num-slots 8 --total-pages 272 --page-size 16
  --max-total-tokens 4352 --max-prompt-tokens 4096`.
- Prompt smoke: 4093 prompt tokens, 4 greedy output tokens, `ignore_eos=true`.

## Profile

The Qwen FP8 MoE grouped path was not the 20s wall:

| Probe | Evidence |
|---|---:|
| MoE grouped dispatch | `tokens=2048/2032`, `topk=8`, `rows=49152/49024`, `masked=false` |
| FP8 MoE grouped inner kernels | about 272ms total on the 4K request |
| Dense FP8 projection path before warm | `qwen/fp8/dense_deepgemm` 500 calls, 17.68s total |
| Dense activation pack | 500 calls, 26.3ms total |
| Cold JIT shape costs | first calls at 4.14s, 3.95s, 4.99s, 1.79s, 2.76s |

Root cause: Qwen dense FP8 projections were using DeepGEMM without a
DSv4-style boot-time warm of the dense projection shapes. The first real user
request paid one DeepGEMM JIT compile per unique `(M,N,K)` projection shape.

## Fix

- Route large Qwen FP8 dense projections (`M >= 1024`, 128x128 block-scaled
  weights) through native DeepGEMM dense FP8 GEMM instead of the scalar FP8
  GEMV batch path.
- Keep decode/small `M` on the existing FP8 GEMV path.
- Add Qwen boot-time warmup for the five unique dense FP8 projection shapes at
  `M=2048`, before the server opens:
  `8192x2048`, `4096x2048`, `2048x4096`, `512x2048`, `2048x512`.
- Tried request-local scratch reuse for the dense FP8 activation/scales buffers;
  it measured neutral (2.727s -> 2.725s), so it is not counted as the win.

## Results

| Run | FP8 elapsed | BF16 elapsed | Verdict |
|---|---:|---:|---|
| Before fix, profiled first 4K request | 20.45s | n/a | pathological |
| Same process, second 4K request after JIT cache warm | 2.747s | n/a | JIT hypothesis confirmed |
| After boot warmup, first 4K request with profiling | 2.933s | n/a | cold-JIT removed |
| After boot warmup, no profiling | 2.727s | 1.972s | FP8 still slower |
| After scratch reuse attempt, no profiling | 2.725s | 1.972s | neutral |

All smoke requests decoded coherently: `" hello hello hello hello"`.

## Verdict

The 12x SLO slowdown was our Qwen DeepGEMM misuse: dense FP8 DeepGEMM JIT was
left on the request path. That specific bug is fixed; the first 4K FP8 request
dropped from about 20.45s to about 2.73s.

The final throughput gate is still not a PASS: FP8 remains slower than BF16 on
this raw c=1 SLO smoke (2.725s vs 1.972s, about +38%). Do not claim FP8
throughput/default on H20 from this result. The memory/slot license from the
SLO sweep remains valid, but raw throughput still needs a separate follow-up if
we want FP8 to beat BF16.

## Rule

DeepGEMM correctness is not enough. Any Qwen port of a working DSv4 DeepGEMM
path must also mirror DSv4's operational discipline: resident weights, no scalar
fallback for large prefill, and JIT warmed before the server accepts traffic.
