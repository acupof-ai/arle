# KV W4A8 FP4 sm89 scan smoke kill

## Context

User focus: FP8 / FP4 quantization operators and KV quantization operators.
After landing the Qwen3.5 FP8 KV HND refill pairwise-store win, the next
question was whether FP4/W4A8 KV has enough operator-level magnitude to
license a runtime implementation.

This was a Phase 0a standalone CUDA smoke, not a runtime change:

```bash
nvcc -O3 -std=c++17 -arch=sm_89 scripts/kv_w4a8_smoke.cu -o /tmp/kv_w4a8_smoke
for i in 1 2 3 4; do /tmp/kv_w4a8_smoke; done
```

Shape:

| Param | Value |
|---|---:|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| SM | 89 |
| batch | 4 |
| seq_len | 65536 |
| kv_heads | 8 |
| head_dim | 80 |
| elements | 167,772,160 |
| scale elements | 2,097,152 |
| iterations | 50 |

The working set is intentionally larger than the 48 MiB L2 on the tested GPU:
BF16 KV is ~320 MiB, FP8 KV+scales is ~168 MiB, and FP4 KV+scales is ~88 MiB.

The smoke measures KV read + scale + dequant scan only. It does not implement
softmax attention, FP8 MMA, runtime KV packing, or an accuracy gate.

## Root Cause

On sm_89, NVFP4/E2M1 has no native Blackwell conversion path. MLX's CUDA
NVFP4 helper only uses native `cvt.*.e2m1*` instructions for CUDA >= 12.8 and
`__CUDA_ARCH__ >= 1000`; sm_89 uses fallback conversion. ARLE's Marlin code
has a bit-dequant path, so the smoke tested both a naive LUT path and a
bit-dequant path before drawing a conclusion.

Steady-state hot runs:

| Kernel | time_us | effective read GB/s | vs BF16 time | vs FP8 time |
|---|---:|---:|---:|---:|
| BF16 scan | 732.119-732.694 | 457.96-458.32 | 1.000x | n/a |
| FP8 E4M3 + f32 scale scan | 694.009-694.682 | 253.58-253.83 | 1.054-1.055x | 1.000x |
| FP4 E2M1 LUT + f32 scale scan | 948.426-949.616 | 97.17-97.29 | 0.771-0.773x | 0.731-0.732x |
| FP4 E2M1 bit-dequant + f32 scale scan | 492.872-494.797 | 186.49-187.22 | 1.480-1.485x | 1.403-1.409x |

Cold first process results were discarded for the license decision because GPU
P-state/clocks made timings unstable and changed relative ordering. Hot
repeated runs were stable.

## Fix

Killed the naive runtime direction for sm_89 FP4/W4A8 KV. The LUT path is
slower than FP8, and the faster bit-dequant scan is only 1.48x faster than
BF16 and 1.40x faster than FP8, below the Phase 0a license floor in
[`docs/plans/M_quant-kv-w4a8.md`](../../plans/M_quant-kv-w4a8.md). Do not add
`--kv-cache-dtype w4a8`, new KV enum variants, or runtime attention dispatch
from this evidence.

The only licensed next step is a narrower follow-up smoke if we want to keep
the axis alive:

1. Build a full attention microkernel with register-resident FP4->FP8 unpack
   and FP8 MMA, not a BF16/float scan.
2. Run it in the long-context regime where KV read bandwidth is actually the
   dominant wall-clock term.
3. Require wall-clock ITL or full-attention speedup evidence, plus a KV
   accuracy gate, before touching runtime state or CLI modes.

## Rule

Memory capacity math is not a license. On sm_89, FP4's 2x smaller KV bytes do
not automatically beat FP8 because unpack/dequant overhead can dominate. For
FP4/W4A8 KV, require a full-attention Phase 0 pass before runtime integration;
operator scan results alone are enough to reject naive paths but not enough to
claim production support.
