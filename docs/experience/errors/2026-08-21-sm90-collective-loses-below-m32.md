# The sm_90 mixed-input collective loses to Marlin below M=32 — CUDA, 2026-08-21

> Status: Hypothesis refuted at decode M, and the measurement relocated the
> lever. Backlog item 1 of
> [the NVFP4 decode plan](../../plans/2026-08-21-nvfp4-decode-lever-backlog.md).
> Instrument: a throwaway dev probe (removed once the verdict landed; the
> sweep table below is the record).

## Context

Marlin is 68.3% of decode GPU time and `ncu` puts it at 87% of SM peak —
issue-bound. Its inner loop is `mma.sync.aligned.m16n8k16` with `cp_async` and
its only architecture guard is `__CUDA_ARCH__ < 800`, so it is an sm_80 kernel
running unmodified on H20: no `wgmma`, no TMA. The prediction was that a
Hopper-native mixed-input GEMM, which issues one warpgroup instruction where
Marlin issues four warp instructions, would cut the issue count that is binding.

ARLE already vendors CUTLASS's sm_90 mixed-input warp-specialised collective for
the DSv4 W4A8 MoE path. Driven with one group it is a dense GEMM, so the
prediction could be tested before writing any kernel.

## Result — Marlin wins below M=32 and loses above it

1xH20, GPU idle, both arms one process, `gate_up [34816, 5120]`, 100 warmup +
100 timed launches. Reproduced on a second GPU within 0.3%.

| M | Marlin ms | collective ms | ratio |
|---:|---:|---:|---:|
| 1 | **0.0629** | 0.0965 | 0.65x |
| 4 | **0.0629** | 0.0973 | 0.65x |
| 8 | **0.0629** | 0.0975 | 0.65x |
| 16 | **0.0880** | 0.0973 | 0.90x |
| 32 | 0.1731 | **0.1603** | 1.08x |
| 48 | 0.2442 | **0.1676** | 1.46x |
| 64 | 0.3177 | **0.1676** | 1.90x |
| 128 | 0.5487 | **0.3230** | 1.70x |

`down [5120, 17408]` is the same shape of answer: 0.52x at M=1, 0.70x at M=16.

The gap is not tile waste. Counting the bytes each kernel actually reads —
Marlin 100.27 MB (nibbles plus S0E5M3 group scales), the collective 91.9 MB
(INT4 plus BF16 block scales) — at M=1 Marlin achieves **1,594 GB/s** against
the collective's **952 GB/s**. Marlin moves more bytes per second while reading
more of them. `wgmma` and TMA do not help at one row.

**Serving decode runs M=1..16 at `--max-running-requests 16`, so the untuned
collective is the wrong kernel there.** The axis is demoted, with one caveat
kept open: this is the array/grouped variant, which pays per-group pointer
indirection a dense Machete-style kernel would not.

## What the measurement actually found

**Marlin costs the same at M=8 as at M=1.** 0.0629 ms for 1, 4 and 8 rows —
identical to four digits. It first moves at M=16 (1.40x) and then grows roughly
linearly: 2.75x at M=32, 5.05x at M=64.

So the lever on the 68.3% is not a faster kernel at M=1. It is **more rows per
call**, and rows 2 through 8 are free.

Per row, the difference is an order of magnitude: 0.0629 ms for one row against
0.0880 ms for sixteen, which is **0.0055 ms/row — 11.4x better**.

That also explains the issue-bound reading without appealing to the dequant.
Marlin's E2M1 unpack is already four integer instructions for four values, table
free and branch free (`marlin/dequant.h:391`), so it is not what fills the issue
slots. The `mma.sync.aligned.m16n8k16` is: at `thread_m_blocks == 1` it computes
16 rows whatever M is, so at M=1 **93.75% of the tensor-core issue is discarded**.
87% of SM peak and 39.8% of HBM are the same fact seen twice — the machine is
busy computing rows nobody asked for.

That makes speculative decode a measured mechanism rather than a feature to
re-try: a verify step of `b` requests by `d` draft tokens presents `M = b*(d+1)`
to the same GEMM. At c=1 with `d=3`, M goes 1 to 4 at **zero** extra GEMM time,
so every accepted draft token is free on the kernel that dominates decode. The
recorded MTP loss (-77%) was measured on a build whose prefill path has since
been rewritten, and it never had this number to explain itself with.

The two levers also compose rather than compete. Spec decode multiplies rows; at
`b=16, d=3` the GEMM sees M=64, which is exactly where the collective is 1.90x
Marlin. **A kernel switch is worth wiring only once something pushes M past 32,
and spec decode is that something.**

## Two alignment bugs found on the way

Driving the grouped kernel with one group exposed a latent crash for any odd
expert count, fixed in `8778391f5` and `7806c01ec`:

- CUTLASS builds the prototype TMA descriptor from the **pointer array's own
  address**, not from a pointer read out of it, and `make_tma_copy_desc` asserts
  that address is 16B aligned. The arrays were spaced `num_experts * 8` apart,
  so an odd count lands the A array on an 8B boundary and aborts the process.
- Padding the arrays to 16B then moved the CUTLASS workspace that follows the
  metadata from 128B to 160B, which the kernels reject device-side with
  `CUDA_ERROR_MISALIGNED_ADDRESS`. The metadata block is now rounded to 256B.

Even expert counts land on `E*128` either way, so DSv4-Flash is unchanged.

## Rule

A kernel comparison at one M is not a comparison. Both kernels here are flat in
M over part of the range and linear over the rest, and they cross — quoting
either endpoint alone gives the opposite conclusion. Sweep until the curves
cross or provably do not.

And read the flat part. Marlin being free from M=1 to M=8 is a bigger fact than
either kernel's speed at M=1, and it was visible only because the sweep included
rows the workload does not currently produce.
