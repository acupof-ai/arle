# Marlin at decode shapes is not occupancy-limited — three reverted attempts, 2026-08-19

> Status: Confirmed. All three changes reverted; the probe that measured them is kept.

## Context

After [`1da4e0422`](../wins/2026-08-19-nvfp4-marlin-remaining-load-sites.md) put
every quantised GEMM of Qwen3.8-27B-NVFP4 on Marlin, c=16 sat at +13.3% over
Qwen3.6-27B-FP8 while c=1..8 were all above +30%. A per-op profile put the whole
residue in `dense_ffn`, and an in-tree comment recorded the kernel as
issue-bound with "occupancy 12.5% capped by the opt-in shared-memory request".
That reading is correct and the cap is real. Removing it does not help.

## Root cause of the wrong hypothesis

The gap was framed in bytes per second: Marlin's NVFP4 arm at ~1.04 TB/s against
its per-channel FP8 arm at ~1.95 TB/s on the same card, so NVFP4 looked 2x
inefficient with 2x of headroom to recover.

**Bytes is the wrong denominator for a kernel that is not bandwidth-bound.** Per
weight *value* — the quantity the dequant, the scale multiply and the MMA all
scale with — the two arms are 12% apart, not 2x:

| gate_up 34816x5120, M=16 | FP4 | per-channel FP8 |
|---|---:|---:|
| ms | 0.093 | 0.083 |
| values | 178.3 M | 178.3 M |

The 12% is the NVFP4 group-scale multiply: one E4M3 scale per 16 elements means
`dequant_fp8_scales` per k-step plus `scale<scalar_t>` per fragment
(`marlin_template.h:1076`, `:1132`), which the `group_blocks == -1` per-channel
path skips entirely. It is intrinsic to the format, not a defect.

## What was tried, and what each measured

`crates/infer-cuda/examples/marlin_fp4_probe.rs` (kept), 1xH20, 50 iters/point.
Baseline is `1da4e0422`. ncu metrics from the gate_up FP4 kernel.

| | regs/thread | warps active | issue | gate_up M=16 | down M=16 | gate_up M=32 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 96 | 20.70% | 65.6% | **1.14** | 0.89 | **0.58** |
| 1. config search by resident threads | 96 | 21.36% | 62.2% | 1.00 | 0.70 | 0.62 |
| 2. + launch asks for the tile, not the budget | 96 | 24.70% | 66.8% | 1.08 | **0.97** | 0.62 |
| 3. + `__launch_bounds__` at 80 regs | **80** | **30.72%** | 69.9% | 1.02 | 0.86 | 0.25 |

TB/s columns are weight bytes over kernel time.

1. **Selecting the config by resident threads made it worse.** A hand-rolled
   `max_shared_mem / (cache_size + 1024)` said the 256-thread tile would hold 3
   blocks; the driver placed 2, because registers bind before shared memory.
   The estimate picked a bigger tile that never got its residency.
2. **The over-request was real and fixing it did nothing net.** The launch asked
   for the whole opt-in budget instead of the tile's own footprint, so
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor` is the honest source for
   residency and warp occupancy rose 20.7% -> 24.7%. Byte-weighted over the
   56-layer MLP at M=16 the throughput was 8.08 ms before and 8.09 ms after.
   gate_up lost what down gained.
3. **Buying occupancy with registers loses.** 96 -> 80 registers put warp
   occupancy at 30.7%, the highest of the four, and throughput at the lowest.
   M=32 (`thread_m_blocks=2`, more accumulators) collapsed 2.4x — the spill
   signature.

## Rule

Issue utilisation that will not move when occupancy rises 50% is a dependent-chain
latency problem inside the warp, not a warp-supply problem. Marlin's inner loop is
`dequant -> scale -> mma` on the same registers; the registers that look like
excess are what hides that chain, and taking them away costs more than the extra
warps return. Before spending builds on occupancy, check whether issue moves with
it — here it went 65.6% -> 66.8% -> 69.9% while throughput fell.

Corollary for the +30% goal at c=16: with this checkpoint the GEMM path has no
tuning headroom left. NVFP4's GEMM subtotal is 15.103 ms against the FP8
checkpoint's 19.474, the remaining 7.351 ms of the step is code both checkpoints
share, and closing to +30% needs 2.11 ms that only NVFP4-specific work can
supply. See [`../wins/2026-08-19-nvfp4-marlin-remaining-load-sites.md`](../wins/2026-08-19-nvfp4-marlin-remaining-load-sites.md).
