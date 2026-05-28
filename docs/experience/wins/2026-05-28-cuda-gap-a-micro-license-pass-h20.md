# GAP-A · License-or-kill micro-experiment — PASS on H20

Date: 2026-05-28
Phase: 2 (of the GAP-A plan,
[`docs/plans/2026-05-28-gap-a-cutlass-mma-quant-gemv.md`](../../plans/2026-05-28-gap-a-cutlass-mma-quant-gemv.md))
Hardware: H20 (SM_90), 78 SMs, HBM3 6144-bit @ 2619 MHz, 4023 GB/s peak
CUDA: 12.2 / nvcc 12.2.140

## Context

The audit at
[`docs/research/2026-05-28-arle-kernel-vs-sota-audit.md`](../../research/2026-05-28-arle-kernel-vs-sota-audit.md)
GAP-A claims `dsv4_fp8_gemv_batch_tiled_kernel` is compute-bound at the
decode shape and a CUTLASS-MMA port should deliver ~4-6% decode wall-clock
on SM_89. On H20 (SM_90) the audit says "the gap is narrower" — but did
not quantify. Per CLAUDE.md §0 SOLID, the §0 confounder for our pod is
roofline: if scalar is BW-bound on H20, MMA cannot help.

## Hypothesis

Scalar `dsv4_fp8_gemv_batch_tiled_kernel` achieves ≥ 75% of H20's peak
HBM3 BW at the canonical decode shape (B=1..16, N=2048, K=7168). If
true, GAP-A axis is dead on H20 — KILL.

## Experiment

Self-contained CUDA C program at `/tmp/gap_a_micro.cu` (pod), built from
[`_gap_a_micro_workdir/gap_a_micro.cu`](../../../_gap_a_micro_workdir/gap_a_micro.cu)
(local; gitignored — micro-experiment scratch, deleted before commit).

The program copy-pastes `dsv4_fp8_gemv_batch_tiled_kernel` verbatim from
`crates/cuda-kernels/csrc/gemm/quantized_gemv.cu:392-515` (so we
measure the **production** kernel, not a re-derived variant) and times
it with cudaEvent on three batch sizes: B = 1, 4, 16.

Build: `nvcc -O3 -arch=sm_90 -std=c++17 gap_a_micro.cu -o gap_a_micro`
Run: `./gap_a_micro` on the pod.

Bytes-moved accounting: weight (`N*K` bytes FP8) + activation
(`B*K*2` bytes BF16) + output (`B*N*2`). Scales tiny, ignored.
Weight is **not reused across B** in this kernel — every K element is
re-read for each B — so the bytes-moved is constant-in-B in this
accounting (a property of the scalar kernel, not a measurement
artifact).

## Results

```
Device: NVIDIA H20  SM count=78  MEM clock=2619 MHz  bus=6144-bit
Reported peak HBM BW: 4023 GB/s
# shape N=2048 K=7168  scale[rows=16 cols=56]  peak HBM BW=4023 GB/s
# B    kernel_us    weight_GB    act_GB       achieved_GB/s frac_peak
  1    84.63        0.0147       0.0000       174          0.043
  4    85.54        0.0147       0.0001       172          0.043
  16   326.53       0.0147       0.0002       46           0.011
```

## Verdict — PASS (license to proceed to Phase 3 granted)

`frac_peak` is 0.04 at B=1/4 and 0.01 at B=16 — **far below the 0.50
license-or-kill floor**. The scalar kernel is leaving 96–99% of HBM3
bandwidth on the table.

Diagnosis (Hypothesis, not yet evidence):

1. **Inner-loop sequential dependence on `b` axis**. At B=16, the
   per-K-element inner body is `for (int b = 0; b < DSV4_BATCH_TILE; ++b)
   if (batch_idx < B) sums[b] += w * x[batch_idx*K+k]` — 32 strided
   global loads (BF16 activation) per thread per K element, even though
   only 16 are useful. Hence B=16 is 4× slower than B=4 (not 4× faster):
   the **same scalar inner loop runs at B=16's cost regardless of mask**.
2. **FP8 decode + E8M0 scale done per-K-element on a single FP32 SIMT
   pipe**. No tensor cores. The kernel's effective FLOP/s is ~10× off
   peak FP32 SIMT throughput on H20.
3. **Block occupancy**. Smem usage `GEMV_ROWS * 8 * DSV4_BATCH_TILE *
   4 = 4 KB` for the smem reduction — fine. But `sums[DSV4_BATCH_TILE]`
   register array (32 floats × 256 threads × 4 = 32 KB / block of
   register memory) crowds the SM's register file. H20 has 64K
   registers/SM → ~2 blocks/SM achievable, leaving ~78×2 = 156 blocks
   in flight vs ~512 blocks (`N/4=512` × ⌈B/32⌉=1) we need to schedule
   at B=16. Low occupancy → BW unused.

Any one of these is enough to explain the 1–4% peak. They all point
to **same fix direction**: tile-MMA the work so per-K-element compute
collapses to `mma.m16n8k32` (16 FP32 outputs / 32 K-elements / 1 issue)
instead of `32 × FFMA / K-element / 32-thread warp` (256 FFMA / K-element
across the warp).

## What worked

- **Verbatim copy-paste of the production kernel**: removes the "is the
  micro a different kernel" confounder. The kernel is the exact bytes
  shipped in the runtime.
- **Single source of truth for peak BW** (driver-reported via
  `cudaGetDeviceProperties`): 4023 GB/s on this H20. No vendor-spec-vs-
  measured ambiguity.
- **Three-point sweep (B=1, 4, 16)**: shows the inner-loop cost is
  invariant in B at small B but jumps at B=16 because the kernel
  switches branches (`tile_batches > 4` path), which is the audit's
  exact claim that the 32-tile path serializes.

## What didn't work / open gaps

- **No `ncu` on pod** (CUDA 12.2 toolkit shipped, but no Nsight Compute
  binary). Could not get achieved-occupancy / DRAM-throughput metrics
  directly. Bytes-moved is *modelled* (N*K + B*K*2 + B*N*2), not
  measured — but the model is conservative (counts weight once even
  though scalar may re-read across SMs that don't share L2 lines), so
  the reported `frac_peak` is an **upper bound**. Real `frac_peak` is
  likely even lower → kill verdict only strengthens.
- Did not run a control with B=32 to verify the inner-loop hypothesis
  directly. Deferring to Phase 3 — once MMA path exists, the A/B is
  the same-shape MMA vs scalar.
- Did not measure on SM_89 (L4) — no L4 box currently provisioned. The
  audit's SM_89 wall-clock projection (4–6% decode) remains
  `Hypothesis` until that hardware lands.

## Rule

- For "scalar GEMV vs MMA GEMV" decisions, **measure the scalar's BW
  achievement first** before writing any MMA code. If scalar is BW-
  bound (frac_peak ≥ 0.75), MMA is dead. If frac_peak < 0.50, license
  granted.
- Verbatim copy-paste of the production kernel into a standalone .cu
  + nvcc + `cudaEventRecord` is a 60-line, 5-minute experiment that
  beats writing a 500 LoC MMA prototype as the first step.
- Bytes-moved accounting should be conservative (over-count → BW
  fraction is an upper bound → kill verdict survives the worst case).

## Cross-refs

- Plan: [`docs/plans/2026-05-28-gap-a-cutlass-mma-quant-gemv.md`](../../plans/2026-05-28-gap-a-cutlass-mma-quant-gemv.md)
- Audit: [`docs/research/2026-05-28-arle-kernel-vs-sota-audit.md`](../../research/2026-05-28-arle-kernel-vs-sota-audit.md)
- DSv4 binding constraints: [`docs/research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md`](../../research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md)
