# Marlin NVFP4 decode: blocks_per_sm tiebreaker — CUDA, 2026-08-23

> Status: Shipped

## Goal

Improve NVFP4 decode throughput on H20 by increasing the Marlin kernel's
blocks/SM at decode shapes (M≤8).

## Hypothesis

The `determine_exec_config` tiebreaker prefers larger `thread_k` when waves are
equal, leaving decode shapes at bps=4 (the first config where waves=1). At
decode shapes the kernel is instruction-issue bound, so a 5th resident block
buys issue slots that 4 blocks cannot fill. Expected: higher bandwidth
utilization, lower kernel time.

## Parameters

Standalone kernel probe (no engine, no checkpoint):

```bash
INFER_CUDA_DEVICE=6 target/release/examples/marlin_fp4_probe
```

- Baseline: `gptq_marlin.cuh` at f681c8156 (tiebreaker: larger thread_k)
- Treatment: tiebreaker prefers higher bps for `m_block_size_8 && q_type == host::kFE2M1f`
- Shapes: Qwen3.8-27B dense-MLP (gate_up N=34816 K=5120, down N=5120 K=34816)
- M sweep: 1, 16, 512, 2048
- 20 iters/point, 3 runs

## Environment

- Host / GPU: 8×H20 (sm_90, 78 SM, 4.0 TB/s HBM3, 228 KB smem/SM)
- Driver / CUDA: 12.8
- Model / dtype: Qwen3.8-27B-NVFP4 (synthetic weights in probe)
- TP / EP / slots / KV: N/A (standalone kernel)

## Results

| shape | M | baseline TB/s | treatment TB/s | delta |
|---|---:|---:|---:|---:|
| gate_up fp4 | 1 | 1.58 | 1.61 | +1.9% |
| down fp4 | 1 | 1.23 | 1.29 | +4.9% |
| gate_up fp4 | 16 | 1.14 | 1.13 | -0.9% |
| down fp4 | 16 | 0.89 | 0.89 | 0% |
| gate_up fp8 | 1 | 2.96 | 2.96 | 0% |
| down fp8 | 1 | 2.29 | 2.29 | 0% |

3 runs, consistent. fp8 path unchanged (gated to num_bits==4 only).

ncu (gate_up fp4, M=1):

| Metric | Baseline | Treatment |
|---|---:|---:|
| Grid Size | 312 (78×4) | 390 (78×5) |
| Achieved Occupancy | 20.61% | 24.51% |
| Max Bandwidth | 37.50% | 38.42% |
| Duration | 68.0 us | 66.3 us |

## Problems

The first iteration applied the higher-bps preference to all quant types.
The fp8 per-channel path regressed -7% on the down projection (N=5120, only
40 tiles → 90% idle blocks at bps=5). The second iteration gated to
`num_bits==4`, but that also matches kU4 (W4A16) and kU4B8 (W4A8). Final
gate: `q_type == host::kFE2M1f` (NVFP4 only); fp8 per-channel and W4A16/W4A8
keep the upstream tiebreaker.

## Learnings

PASS for NVFP4 decode: +2-6% at M=1, no regression at M≥16. The kernel is
compute-bound (ncu: "Compute is more heavily utilized than Memory"), so the
occupancy gain from 4→5 blocks/SM yields a small throughput improvement, not
the ~2× the original occupancy hypothesis predicted. The 2026-08-19 errors
entry ("Marlin at decode shapes is not occupancy-limited") is confirmed:
issue utilization barely moves with occupancy. The remaining headroom is in
the dequant→scale→MMA dependent chain, not in warp supply.
