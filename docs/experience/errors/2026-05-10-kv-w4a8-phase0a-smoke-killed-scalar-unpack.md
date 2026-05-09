# KV W4A8 Phase 0a Smoke Killed — Scalar INT4 Unpack Is Not Enough

## Context

Task #33 tested the cheapest KV W4A8 license gate before wiring a production
KV format. The smoke binary stayed outside the repo at `/tmp/kv_w4a8_smoke.cu`
and did not touch Cargo, FFI, kernels, or runtime dispatch.

Command:

```bash
nvcc -arch=sm_89 -O3 -std=c++17 /tmp/kv_w4a8_smoke.cu -o /tmp/kv_w4a8_smoke
/tmp/kv_w4a8_smoke
```

Shape:

| Field | Value |
|---|---:|
| Batch | 4 |
| Q heads | 32 |
| KV heads | 8 |
| Seq len | 4096 |
| Head dim | 128 |
| Iterations | 100 |

Result:

| Path | Mean ms / iter |
|---|---:|
| BF16 scalar decode attention | 1.7080 |
| W4 scalar decode attention | 1.5189 |
| W4 speedup vs BF16 | 1.12x |

License gate from `docs/plans/M_quant-kv-w4a8.md`:

| Speedup | Decision |
|---|---|
| `>= 2x` | proceed to Phase 0b |
| `1.5x..2x` | borderline |
| `< 1.5x` | kill |

## Root Cause

The simple scalar online-softmax W4 path saves memory traffic but spends much
of that gain unpacking INT4 values and multiplying scales inside the attention
loop. At 4096-token decode, the measured end-to-end kernel speedup is only
1.12x, far below the 1.5x minimum license threshold.

This does not disprove a more aggressive W4A8 KV design. It specifically kills
the low-complexity "inline scalar unpack in the current quantized-decode style"
path as a basis for Phase 0b.

## Fix

Do not wire `--kv-cache-dtype w4a8` from this scalar-unpack substrate.

If KV W4A8 is revisited, the next attempt should start from a real throughput
design: FP8-MMA/TileLang attention, vectorized dequant staging, or another
kernel structure that can recover enough bandwidth to beat the unpack cost.

## Rule

For KV compression, byte savings alone are not a license. The first gate must
be a wall-clock kernel smoke at the target decode shape; if unpack overhead
eats the bandwidth gain, stop before adding runtime surface area.
