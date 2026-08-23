# Paged Attention Optimization TODO

Baseline: kernel at ab8c96200 (pre-fp8-MMA); ncu data below is that baseline.

Kernel: `crates/cuda-kernels/csrc/attention/paged_attention_quantized_fa3.cu`
Hardware: H20 (sm_90, 78 SM, 4.0 TB/s HBM3, 228 KB smem/SM, 65536 regs/SM)

## ncu Baseline (2026-08-23, c=16/32, ctx=32K, fp8 KV, hd256)

| Metric | c=16 | c=32 |
|---|---|---|
| regs/thread | 254 | 254 |
| CTA/SM (reg limit) | 2 | 2 |
| warps active | 11.7% | 11.9% |
| long scoreboard stall | 46.8% | 46.8% |
| math pipe throttle | 13.2% | 13.2% |
| DRAM read | 1.09 GB | 2.19 GB |
| bandwidth | 0.85 TB/s (21%) | 0.94 TB/s (24%) |
| latency | 1.285 ms | 2.334 ms |

DRAM read = theoretical floor. Kernel is **instruction-issue bound** (CUDA core pipe saturated at 13.2%, tensor pipe nearly idle at 8.4M vs 1.54B instructions).

## Root Cause

254 regs → 2 CTA/SM → 8 warps/SM. The 46.8% long scoreboard is from the dequant + softmax register dependency chain, NOT HBM latency. The 13.2% math pipe throttle confirms CUDA core pipes are the bottleneck. Tensor pipe is underutilized.

## Optimization Items

### 1. ~~o_acc → smem~~ — RULED OUT

- [x] Investigate smem feasibility
- **Finding**: 64 KB smem → 2 CTA/SM (same as register limit). No improvement.

### 2. ~~Launch bounds 3~~ — RULED OUT

- [x] `__launch_bounds__(128, 3)` → 168 regs, 3 CTA/SM
- **Finding**: long scoreboard ROSE 46.8→52.2%. Latency +15.3% c16 / +8.4% c32. 254 regs / 2 CTA/SM is optimal.

### 3. ~~K+V parallel load~~ — RULED OUT

- [x] Split smem_kv into smem_k + smem_v, issue both loads at tile start
- **Finding**: latency +45.7% c16 / +56.8% c32. Duplicated dequant code → register pressure → occupancy drop. 4 warps already cross-hide K/V latency at hardware level. Kernel is compute-bound, not memory-latency bound.

### 4. ~~fp8 tensor core Q·Kᵀ~~ — BLOCKED on sm_90 (sm_100 only)

- [x] Keep K in fp8 (skip dequant for K) — saves ~192 instructions/lane/tile
- [x] Quantize Q per CTA to e4m3 (once, not per tile)
- [x] Use `mma.sync.m16n8k32.e4m3.e4m3.f32` for S = Q·Kᵀ
- [ ] K scale applied to S columns (existing code at L296)
- [ ] V path unchanged (still dequant to bf16 for O = P·V)
- [ ] Expected: +20-35% (frees CUDA core pipe, uses idle tensor pipe)
- [ ] Risk: numerical — needs needle ladder + GSM8K gate
- [ ] ncu + bench A/B on pod

**Blocker**: `ldmatrix.b8` (m16n16/m8n16 shapes) is **sm_100 (Blackwell) only** — confirmed by CuTe source (`copy_sm100.hpp`, guarded by `CUTE_ARCH_LDSM_SM100A_ENABLED`). The `mma.sync.m16n8k32.e4m3` MMA itself is sm_89+ (available on H20), but without ldmatrix.b8 the fragments must be loaded manually. Decoded the full fragment layout from CuTe `mma_traits_sm89.hpp`:

- A (16×32, 4×uint32/lane): thread t (t0=t/8, t1=t%8) holds rows {2t0, 2t0+1, 2t0+8, 2t0+9} × cols {t1, t1+8, t1+16, t1+24}, K interleaved {k, k+16}
- B (32×8, 2×uint32/lane): thread t holds cols {t0, t0+4} × K {t1, t1+8, t1+16, t1+24}

Manual load cost: ~128 LDS.U8/lane/tile for A + ~64 for B = 192 load instructions, vs 32 ldmatrix in the bf16 path. The K-dequant savings (~192 instructions) are entirely offset by the extra loads — **net loss on sm_90**. Viable only on Blackwell where ldmatrix.b8 loads 16 bytes/lane in one instruction.

**Verdict**: Do not pursue on H20. Revisit on sm_100 hardware.

### 5. Marlin GEMV M-split occupancy — PLANNED (separate kernel)

- [ ] M-split grid (n_tiles × m_tiles) in marlin_fp4_gemm
- [ ] Target: 4-8 blocks/SM (from 1-4)
- [ ] Expected: ~2× kernel → +26% decode on NVFP4/27B
- [ ] Baseline: 1.29 TB/s (32% of 4 TB/s), 1-4 blocks/SM, warp active 12-22%
- [ ] Kernel correctness verified (20/20 tests pass)

## Progress Log

- 2026-08-23: ncu baseline. Three approaches ruled out (o_acc→smem, launch bounds, K+V parallel). Kernel is instruction-issue bound. fp8 tensor core Q·Kᵀ is the correct direction.
- 2026-08-23: fp8 MMA blocked on sm_90 — ldmatrix.b8 is sm_100 only. Manual fragment load cost offsets dequant savings. Direction viable only on Blackwell. Next: Marlin GEMV M-split (item 5).
