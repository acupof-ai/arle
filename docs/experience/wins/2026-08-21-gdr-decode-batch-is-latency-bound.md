# `gdr_decode_batch_kernel` is latency-bound, not at a ceiling — CUDA, 2026-08-21

> Status: Measured, no code change. Backlog item 2 of
> [the NVFP4 decode plan](../../plans/2026-08-21-nvfp4-decode-lever-backlog.md).

## Context

`nsys cuda_gpu_kern_sum` put this kernel at **13.0% of decode GPU time** — 398.7
ms over 9,792 launches, 40.7 µs each — and it had never been profiled. The
question was only whether it sits near a hardware ceiling (like Marlin at 87% of
SM peak) or has the kind of headroom the NVFP4 widen kernel turned out to have.

The kernel is the batched gated-delta-rule recurrence,
`csrc/recurrent/gdr_decode_batch.cu`: grid `(num_value_heads, B)`, 512 threads
per block, 5,712 B of static shared memory.

## What the measurement says

`ncu --set full`, H20 (sm_90, 78 SMs), 5 launches. **Neither pipe is close to
binding:**

| | measured |
|---|---:|
| Compute (SM) throughput | 17.5 - 19.4% of peak |
| Memory throughput (SOL) | 20.2 - 22.5% of peak |
| DRAM throughput | 1.7 - 2.3% |
| Executed IPC (active) | 0.66 |
| Issue slots busy | 16.7% |
| Achieved occupancy | 30.9% against a theoretical 100% |

Nothing caps theoretical occupancy: 32 registers per thread and the warp limit
tie at 4 blocks per SM, which is exactly 64 warps. Shared memory allows 9.

**The time goes to waiting for global loads.** 29.53 cycles between issues, and
the stall breakdown:

| stall reason | cycles / issued inst | share |
|---|---:|---:|
| **Long Scoreboard** (L1TEX data dependency) | 18.19 | **61.6%** |
| Barrier | 3.62 | 12.3% |
| Short Scoreboard (shared / MIO) | 2.49 | 8.4% |
| Wait (fixed-latency) | 1.91 | 6.5% |
| everything else | ≤1.00 each | ≤3.4% each |

83.3% of cycles have no eligible warp, with 4.94 active warps per scheduler and
0.25 eligible. Bank conflicts are negligible — 188-262 shared load conflicts
against 1,034,016 executed instructions. The recurrent state is effectively L2
resident (L2 hit rate 92.3-94.3%, ~1.3 MB of DRAM traffic per launch), so this
is not a bandwidth problem either.

ncu's own top rule, verbatim:

```
OPT   Est. Speedup: 61.61%
      On average, each warp of this workload spends 18.2 cycles being stalled
      waiting for a scoreboard dependency on a L1TEX (local, global, surface,
      texture) operation.
```

## The profiled point is not the production point

The captured grid is `(48, 2, 1)` — 48 value heads x **B=2**, 96 blocks on 78
SMs, 0.31 waves per SM. Under ncu's replay serialisation only two requests
stayed concurrently resident, and duration was 12.9-16.3 µs against the 40.7 µs
nsys average. So ncu's *second* rule — "this kernel grid is too small to fill
the device" — is an artifact of the profiling condition, not a property of
serving: at B=16 the grid is 768 blocks, 2.5 waves against the 4-blocks-per-SM
limit.

What does carry across batch size is per-warp: the 61.6% Long Scoreboard stall,
IPC 0.66, and 16.7% issue-slot occupancy are properties of the instruction
stream, and more blocks do not shorten a global-load dependency.

## Result

The kernel is **latency-bound on global loads**, at roughly a fifth of both the
compute and the memory ceiling. It is not near a ceiling, which is the question
that was asked. Whether the remaining 13% is worth taking depends on what
overlaps the stalls — more warps in flight per block, or fewer dependent global
loads on the critical path — and that needs a B=16 capture before any edit.

## Rule

Profile at the batch the workload runs. ncu's replay serialises requests, so a
kernel whose grid is `(heads, B)` gets captured at a smaller B than serving
uses, and every grid-shaped rule it prints is then about the profiler rather
than about the code. Separate the per-warp findings (which transfer) from the
grid-shaped ones (which do not) before acting on either.
