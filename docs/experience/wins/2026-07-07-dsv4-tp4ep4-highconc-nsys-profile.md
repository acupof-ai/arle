# DSv4 TP4/EP4 High-Concurrency Throughput + nsys Profile

> Status: Active
> Date: 2026-07-07

## Context

DSv4 decode runs 25-38× above HBM bandwidth roofline at B=1. First-principles
diagnosis: single token per expert → GEMV latency-bound, no weight amortization.
This experiment measures actual throughput scaling and kernel-level attribution
under real concurrency to find the optimization space.

## Environment

| Item | Value |
|------|-------|
| Model | DeepSeek-V4-Flash-FP8 (`/host/DeepSeek-V4-Flash-FP8`, 274GB, 46 shards) |
| GPUs | 4× H20 (indices 1-4) |
| Topology | TP=4, EP=4, MoE backend = NCCL allreduce (no DeepEP) |
| FlashMLA decode | enabled |
| KV format | FP8 packed (584 bytes/token, `KVFormat::PackedBytes`) |
| KV slots | 105 per GPU |
| Binary | `/host/arle-build/target/release/arle` (release, cuda+nccl, no deepep) |
| Probe tool | `scripts/dsv4_concurrent_probe.py` (aiohttp, ~512-token prompt) |

## 1. Throughput Sweep (clean, no tracing)

| c | reqs | errs | wall_s | out_tok | agg_tps | per_req_tps | mean_lat_s | p50 | p99 |
|---|------|------|--------|---------|---------|-------------|------------|-----|-----|
| 1 | 2 | 0 | 9.73 | 201 | 20.7 | 26.3 | 4.87 | 6.38 | 6.38 |
| 2 | 4 | 0 | 16.93 | 512 | 30.2 | 15.2 | 8.41 | 8.45 | 8.48 |
| 4 | 8 | 0 | 21.63 | 1024 | 47.3 | 11.9 | 10.76 | 10.81 | 10.82 |
| 8 | 16 | 0 | 33.27 | 2048 | 61.6 | 7.7 | 16.58 | 16.55 | 16.72 |
| 16 | 32 | 0 | 42.02 | 3690 | 87.8 | 6.7 | 19.05 | 20.46 | 21.56 |
| 32 | 32 | 32 | 4.55 | 0 | — | — | — | — | — |
| 64 | 0 | 128 | 0.02 | 0 | — | — | — | — | — |

**c=32/64**: serve OOM'd (KV slot budget exceeded). c=16 is the operational peak.

**Derived per-batch-step wall time** (from per_req_tps, assuming each step produces 1 token per active request):

| c | step_wall_ms | tokens/step | efficiency vs c=1 |
|---|-------------|-------------|-------------------|
| 1 | ~38 | 1 | 1.0× |
| 2 | ~66 | 2 | 1.15× |
| 4 | ~84 | 4 | 1.81× |
| 8 | ~130 | 8 | 2.35× |
| 16 | ~149 | 16 | 3.42× |

Efficiency = (c / step_wall_ratio). c=16 gives 3.42× throughput gain on 16× concurrency
= 21% parallel efficiency. Sub-linear because per-expert token count stays ≪ 1
(see §4).

## 2. Decode-Phase Breakdown (c=8, PHASE_TIME=1)

Absolute timing inflated by ~3× from `cuEventSynchronize` + `cuStreamSynchronize`
(see §5). **Relative proportions are reliable.**

Steady-state, rank 0, n=8 batched decode step:

| Phase | ms (traced) | % of forward |
|-------|------------|-------------|
| sw_attn prep (proj + compidx) | ~26 | 31% |
| └ QKV proj | ~7 | 8% |
| └ compressor index build | ~17 | 20% |
| sw_attn fwd (FlashMLA kernel) | ~2.7 | 3% |
| sw_attn finish (combine + wo_a) | ~18 | 21% |
| moe (allgather + expert FFN + reduce) | ~31 | 37% |
| lm_head + sampling | ~2 | 2% |
| **Total forward** | **~80** | **100%** |

First-step penalty: +15-20ms in `prep` (compidx index building not yet cached).

## 3. Linear Profile (c=8, LINEAR_PROFILE=1)

| Kernel | avg us | calls/step | % linear time |
|--------|--------|------------|---------------|
| `wo_a` (MLA output proj A) | 325 | 43 | **57-62%** |
| `wqkv_a_fused_batched` (QKV proj A) | 45 | 43 | 7-8% |
| `compressor_wkv_batched` | 31 | 62 | 7-8% |
| `compressor_wgate_batched` | 30 | 62 | 7-8% |
| `wo_b` (output proj B) | 39 | 43 | 6-7% |
| `wq_b_batched` (Q proj B) | 33 | 43 | 5-6% |
| `indexer_weights_batched` | 30 | 21 | 2-3% |
| `indexer_wq_b_batched` | 31 | 21 | 2-3% |

`wo_a` alone = >50% of all linear time. 43 layers × 1 call/step. Shape:
MLA down-projection (low-rank A path of the decomposed output projection).

## 4. nsys Kernel Attribution (c=8, 90s system capture)

Top kernels by total GPU time:

| % GPU | Total | Calls | avg us | Kernel |
|-------|-------|-------|--------|--------|
| 21.9 | 22.5s | 1.21M | 18.6 | `gemv_handwritten_kernel` (bf16 GEMV) |
| 10.0 | 10.3s | 94K | 109 | `dsv4_fp8_grouped_swiglu_decode_kernel` |
| 8.1 | 8.3s | 189K | 44 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` |
| 8.0 | 8.2s | 404K | 20 | `ncclDevKernel_AllGather_RING_LL` |
| 7.5 | 7.7s | 94K | 82 | `dsv4_fp8_grouped_down_decode_kernel` |
| 4.8 | 5.0s | 312K | 16 | `dsv4_fp8_gemv_batch_kernel` |
| 3.7 | 3.8s | 189K | 20 | `dsv4_mhc_params_kernel` |
| 3.5 | 3.6s | 35K | 102 | `dsv4_block_scaled_to_fp8_cache_scales_kernel` |
| 3.4 | 3.5s | 35K | 101 | `dsv4_block_scaled_to_fp8_cache_values_kernel` |
| 3.4 | 3.5s | 85K | 40 | `dsv4_fp8_gemv_batch_tiled_kernel<8>` |
| 2.0 | 2.1s | 2.5K | 842 | `ncclDevKernel_AllReduce_Sum_u32_RING_LL` |
| 2.0 | 2.0s | 550K | 3.7 | `dsv4_deepgemm_pack_quantize_bf16_to_fp8_kernel` |
| 1.9 | 1.9s | 95K | 21 | `deep_gemm::sm90_fp8_gemm_1d2d_impl` |
| 1.5 | 1.6s | 3.7K | 425 | `dsv4_fp8_gemv_batch_tiled_kernel<32>` |
| 1.5 | 1.5s | 94K | 16 | `flash_fwd_splitkv_mla_fp8_sparse_kernel` (FlashMLA) |

**Category totals:**

| Category | % GPU time |
|----------|-----------|
| GEMV (all variants) | **31.6%** |
| NCCL communication | **18.1%** |
| MoE compute (grouped kernels) | **17.5%** |
| FP8 KV cache conversion | **6.9%** |
| DeepGEMM (all tiles) | **~6-7%** |
| FlashMLA attention | **1.5%** |
| Other (mhc_params, pack_quantize, etc.) | ~18% |

**Attention compute (FlashMLA) is NOT the bottleneck** at 1.5%. The surrounding
projections (`wo_a` via GEMV) and prep (compidx) dominate the attention path.

## 5. CUDA API Overhead (c=8, 90s, WITH tracing)

| API Call | % API time | Total | Calls | avg |
|----------|-----------|-------|-------|-----|
| `cuMemcpyHtoDAsync` | 28.2% | 56.3s | 2.15M | 26us |
| `cuEventSynchronize` | 16.7% | 33.3s | 2.23M | 15us |
| `cuStreamSynchronize` | 16.2% | 32.4s | 720K | 45us |
| `cudaLaunchKernel` | 11.2% | 22.3s | 5.94M | 3.75us |
| `cudaGetDeviceProperties` | 9.1% | 18.1s | 18K | 1ms |
| `cuMemcpyDtoHAsync` | 3.4% | 6.7s | 67K | 100us |

Notes:
- `cuMemcpyHtoDAsync` 28% = model weight H2D load (294GB / 4 GPUs during init).
  Not a steady-state decode cost.
- `cuEventSynchronize` + `cuStreamSynchronize` = 33% = **PHASE_TIME/LINEAR_PROFILE
  instrumentation overhead**. Clean runs don't have this.
- `cudaGetDeviceProperties` at 9.1% with 1ms avg: the static SM-count cache fix
  may not be covering all call sites. 200 calls/s during decode is suspicious.
- Kernel launch rate: ~110K launches/s. 5.94M launches across 90s.

## 6. Expert Loading Analysis (why sub-linear scaling)

DSv4: 384 global experts, top-8 routing. EP=4 → 96 local experts per rank.

| c | total routes/step | routes per local expert | GEMV or GEMM? |
|---|-------------------|------------------------|---------------|
| 1 | 8 | 0.08 | pure GEMV |
| 2 | 16 | 0.17 | pure GEMV |
| 4 | 32 | 0.33 | pure GEMV |
| 8 | 64 | 0.67 | pure GEMV |
| 16 | 128 | 1.33 | GEMV (barely >1) |
| 48 | 384 | 4.0 | approaching GEMM regime |
| 96 | 768 | 8.0 | weight amortization starts to matter |

DeepGEMM threshold (`QWEN35_DEEPGEMM_MIN_ROUTES`): 1024 total routes.
At c=16: 128 routes → **8× below threshold**. To trigger DeepGEMM tensor cores
for MoE, need c=128 (1024 routes). Current KV slots cap at 105 → unreachable
with BF16 KV.

## 7. Artifacts

- nsys capture: `/tmp/arle_experiment/dsv4_c8_nsys.nsys-rep` (pod, 1.1GB)
- Throughput log: `/tmp/arle_experiment/throughput_sweep.log` (pod)
- Trace log: `/tmp/arle_experiment/serve_trace.log` (pod, 646KB)
- Probe script: `scripts/dsv4_concurrent_probe.py`

## Problems

- c=32 OOM: KV slot budget exceeded. BF16 KV at 105 slots/GPU is the hard cap.
- Tracing overhead (PHASE_TIME + LINEAR_PROFILE) inflates wall-clock by ~3×.
  Clean throughput numbers were measured separately without tracing.
- nsys was system-wide for 90s; model init H2D transfers dominate early window.
  Steady-state decode-only attribution is approximate (extrapolated from kernel
  proportions after warmup).

## Rule

- Per-expert token count is the fundamental knob for MoE weight amortization.
  At c=16 with EP=4, average is 1.33 tokens/expert — still deep in GEMV regime.
  FP8 KV → 2× slots → c up to 210 → 2.3 tokens/expert. Still GEMV.
  Weight amortization for MoE requires either DP-attn batching (c ≫ 64) or
  expert consolidation (fewer local experts). Neither is free.
- Dense projections (`wo_a`, `wqkv_a`) don't depend on expert routing. They
  see all B tokens directly. These are the highest-ROI targets for batched GEMM
  (DeepGEMM) replacement of the GEMV path, because B=8 is already enough for
  weight amortization on a 7168×7168 matrix.
