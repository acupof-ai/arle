# M_quant CUTLASS FP8 Smoke Killed on sm_89

## Context

M_quant Phase 0 needed a cheap hardware-path sanity check before any
runtime implementation. The first cuBLASLt FP8 smoke on this RTX 4070 Ti
SUPER (`sm_89`, CUDA 13.2) measured only 1.83-1.88x over BF16, so the
follow-up was to test whether CUTLASS direct FP8 MMA could bypass a poor
cuBLASLt heuristic.

Plan: [`M_quant-fp8-w4-magnitude-path.md`](../../plans/M_quant-fp8-w4-magnitude-path.md)
§9.1.

## Setup

The smoke was kept outside the build graph at `/tmp/fp8_cutlass_smoke.cu`
and was not committed. It used TileLang's vendored CUTLASS headers:

```bash
/opt/cuda/bin/nvcc -ccbin /usr/bin/g++-14 -arch=sm_89 -O3 -std=c++17 \
  -I /opt/cuda/include \
  -I .venv/lib/python3.14/site-packages/tilelang/3rdparty/cutlass/include \
  -I .venv/lib/python3.14/site-packages/tilelang/3rdparty/cutlass/tools/util/include \
  /tmp/fp8_cutlass_smoke.cu \
  -o /tmp/fp8_cutlass_smoke \
  -lcudart -lcublasLt

/tmp/fp8_cutlass_smoke
```

Shape matched the cuBLASLt smoke: `M=2048, N=2560, K=2560`, 100 warmup
iterations, 100 timed iterations with CUDA events.

## Results

| Path | Mean / iter | Std | Speedup vs BF16 |
|---|---:|---:|---:|
| cuBLASLt BF16 control | 0.325 ms | 0.003 | 1.00x |
| cuBLASLt FP8 E4M3 TN | 0.177 ms | 0.001 | 1.84x |
| CUTLASS FP8 default `OpMultiplyAdd` | 0.510 ms | 0.002 | 0.64x |
| **CUTLASS FP8 `OpMultiplyAddFastAccum`** | **0.203 ms** | **0.000** | **1.60x** |

Additional sanity:

- CUTLASS `A=RowMajor, B=ColumnMajor` is the supported fast path for the
  vendored Sm89 FP8 kernel.
- Changing C output layout between row-major and column-major did not move
  the result (`0.203 ms` both ways).
- `A=ColumnMajor, B=RowMajor` failed to compile in this CUTLASS
  specialization, so it is not a viable direct path here.

## Root Cause

The direct CUTLASS path did not expose the predicted Ada FP8 throughput.
Even after switching to `OpMultiplyAddFastAccum`, measured FP8 speedup is
only 1.60x against the same BF16 control and is slower than cuBLASLt FP8.
This fails the <3x kill bucket from §9.1.

The evidence now points away from "cuBLASLt chose a bad algo" and toward
"this sm_89 + CUDA 13.2 + available CUTLASS stack does not provide a
high-utilization W8A8 FP8 GEMM path for the target Qwen3 shape."

## Fix

Do not implement the M_quant W8A8 FP8 runtime path on this machine. The
next quantization work should pivot to:

1. W4A16 Marlin as the sm_89 weight-bandwidth path.
2. KV W4A8 as an orthogonal cache-bandwidth path.
3. Re-open W8A8 only with a new CUDA/CUTLASS stack, different hardware, or
   a separate smoke that clears the 3x/6x gates.

## Rule

For quantization milestones, keep the first license gate at a single-GEMM
wall-clock measurement. Whitepaper TFLOP ratios and trace framing are not
enough to license runtime implementation when the measured primitive is
below the kill threshold.
