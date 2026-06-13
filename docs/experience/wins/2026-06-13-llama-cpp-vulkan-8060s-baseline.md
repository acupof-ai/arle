# llama.cpp Vulkan reference baseline — Radeon 8060S (Strix Halo, gfx1151)

The performance bar for the ARLE `infer-vulkan` bring-up. ARLE must reach "at
least not worse than llama.cpp" on this box (goal, ckl 2026-06-13). Numbers are
from the user's on-box `llama-bench` logs (`C:\Users\Asus\models\qwen3.6\bench_*.log`,
`Downloads\...` for 122B) — llama.cpp build `726704a16 (9204)`, Vulkan backend,
`KHR_coopmat`, `-ngl 99` (all layers on GPU), flash-attn on.

## Context

- GPU: AMD Radeon 8060S, RDNA 3.5, gfx1151, unified memory (`uma:1`), 96 GB VRAM.
- Device caps (Vulkan probe): `fp16:1 | bf16:1 | int dot:1 | matrix cores: KHR_coopmat`.
  No FP8 matrix support (RDNA3.5). Mem BW ~256 GB/s LPDDR5X-8533.

## Reference numbers (Vulkan, fp16 KV unless noted)

| Model | Active | Weight quant | Size | pp512 (t/s) | tg128 (t/s) |
| --- | --- | --- | --- | ---: | ---: |
| Qwen3.5-122B-A10B (qwen35moe) | 10B | Q4_K (XL, MXFP4 experts) | 63.65 GiB | **205** | **23.4** |
| Qwen3.6-35B-A3B (qwen35moe) | 3B | Q4_K_M | 20.60 GiB | **822** | **47.3** |
| Qwen3.6-35B-A3B (qwen35moe) | 3B | Q8_0 | 34.36 GiB | 735 | 42.9 |
| Qwen3.6-27B (dense) | 27B | Q8_0 | 26.62 GiB | 141 | 7.2 |

Long-context throughput (`pp4096+tg128`): 35B Q8_0 kvq8 = 445 t/s; 122B Q4_K = 163 t/s.

Live `llama-cli` sanity check on the 122B (interactive, with thinking) generated
coherent text at ~12.5 t/s prompt / 16.1 t/s gen (lower than the clean bench
because of chat template + sampling; the 23.4 tg bench is the reference).

## Precision findings (what's fastest on this GPU)

- **Weights: Q4_K (4-bit) is fastest AND ~lossless.** Q4_K_M vs Q8_0 on the 35B:
  47.3 vs 42.9 tg, 822 vs 735 pp, and PPL 5.3825 vs 5.3646 (c8192, Δ0.3%). Use
  4-bit weights; Q8 only buys negligible quality at 2× memory + slower.
- **Decode is memory-bandwidth-bound**: speed ≈ f(active-params × bytes/weight).
  27B dense Q8 = 7.2 tg (reads ~26.6 GB/tok) vs 35B-A3B MoE Q4 = 47.3 tg (~3B
  active, 4-bit). MoE + 4-bit dominates.
- **Prefill is compute-bound** → the `coopmat`/WMMA FP16 GEMM path is what gets
  llama.cpp to 800+ pp on the 35B.
- **Compute ladder (RDNA3.5):** INT8 (WMMA/DP4A, ~2× FP16) > FP16 = BF16 (WMMA,
  FP32 accumulate) > FP32 (½) > FP8 (no HW → emulated, avoid).
- **KV cache:** fp16 KV is fastest at short ctx (q8 KV 40 vs 43 tg, slightly
  slower); use q8_0 KV only to save memory at long context.

## Implication for ARLE

ARLE `vulkan-kernels` currently registers only scalar paths and **no `mul_mm`**
(dense GEMM), **no `mul_mmq`** (quantized GEMM), **no `mul_mat_id`** (MoE expert
GEMM). To not be "an order of magnitude slower than llama.cpp", Phase 1 must wire
the **coopmat FP16 `mul_mm` / `mul_mat_id`** path (+ INT8 `mul_mmq`, Q4→FP16
dequant) from `vendor/llama.cpp/vulkan-shaders/`. See
[`docs/plans/amd-vulkan-strix-halo-bringup.md`](../../plans/amd-vulkan-strix-halo-bringup.md).
