# Qwen3.5/3.6 post-license re-profile — board re-rank by measured share

**Date:** 2026-06-11. **Binary:** `9e37bc77` (decode MoE kernel licensed,
default-ON), H20 single GPU, Qwen3.6-35B-A3B. Two nsys passes
(`/data01/build/qwen35_{decode2,prefill2}.nsys-rep`): decode band = 256-token
generation from a tiny prompt (83.6 tok/s under profiler vs 92.0 unprofiled);
prefill band = 3072-token prompt ×3, 8 tokens out (1.43 s warm = 2150 tok/s).

## Decode band (B=1, ~10.1 ms GPU / 11.95 ms wall per token — 85% GPU-busy)

| Path | % GPU | per token | ideal @ 4 TB/s | efficiency |
|---|---:|---:|---:|---:|
| MoE down kernel (`moe_bf16_grouped_gemm_decode`) | 27.7 | 69.3 µs/layer × 40 = 2.77 ms | 4.2 µs/layer (16.8 MB) | **6%** |
| MoE swiglu kernel (gate+up fused) | 13.2 | 32.9 µs/layer × 40 = 1.32 ms | 8.4 µs/layer (33.6 MB) | 25% |
| Dense cuBLASLt band (7 nvjet GEMV shapes) | 26.4 | 2.63 ms | ~1.5 ms (≈6 GB dense) | ~50% |
| GDR decode (30 GDN layers) | 5.1 | 16.9 µs × 30 = 0.51 ms | — | — |
| lm_head `gemv_handwritten` | 4.9 | 491 µs | 156 µs (0.62 GB) | 32% |
| Full-attn decode via `nonpaged_prefill_attention` | 4.5 | 44.7 µs × 10 = 0.45 ms | ~10–15 µs/layer | — |
| `dsv4_route` | 3.8 | 9.5 µs × 40 = 0.38 ms | — | — |
| MoE routing orchestration (pack/scan/scatter/combine/renorm/count) | ~4.5 | ~0.45 ms | — | — |
| Norms / adds / silu | ~3.7 | ~0.37 ms | — | — |

The down kernel is the outlier: it reads **half** the swiglu kernel's bytes in
**twice** the time, σ = 0.6% (deterministic, not contention). Mechanism:
K = 512 → `kv = 64` → each lane runs the v-loop only 2 iterations →
one-row-per-warp keeps ~2 weight loads in flight; the swiglu kernel at
K = 2048 × 2 matrices runs ~16 in flight. Binding constraint =
**memory-level parallelism**, not DRAM bandwidth. Fix shipped as `30611ad4`
(4-row warp tile); A/B LICENSED — 69.3 → 45.7 µs/layer, c=1 +6.6% / c≥2 +11%
(`docs/experience/wins/2026-06-11-qwen35-moe-down-kernel-row-tile.md`).

## Prefill band (3072 tokens, chunked 2048)

| Path | % GPU | shape note |
|---|---:|---|
| `nonpaged_prefill_attention` (10 full-attn layers) | **42.1** | avg 5.0 ms, max 30.8 ms/launch; ~0.5 ms/layer roofline at 148 TFLOPS → ~10× off, no tensor cores |
| `gated_delta_rule_prefill_recurrent` (30 GDN layers) | **28.0** | 4.43 ms avg ≈ 13.3 ms/layer-run; SGLang chunked GDR class does this <1 ms |
| DeepGEMM contiguous MoE (gate/up + down) | 12.0 | healthy; not a lever |
| MoE decode-band kernels (tail chunks ≤256 routes) | 3.6 | covered by the down-tile fix |
| Everything else | <3 each | — |

## Re-ranked board

1. **Decode MoE down kernel MLP fix** — in flight (`30611ad4`), predicted
   92 → ~108–112 tok/s. Cheapest, formula-backed.
2. **#4 full-attention kernel** — 42.1% of prefill + 4.5% of decode. The top
   structural lever. Adoption-first survey (FA3 / FlashInfer HD256 GQA vs the
   in-tree TileLang HD256 paged pipeline) before writing anything.
3. **#3 GDR chunked prefill (FlashQLA)** — 28.0% of prefill. Survey already
   done; ~3–4 days incl. the TileLang-skew pod spike.
4. **lm_head GEMV** — 491 → ~160 µs available (cuBLASLt or vectorized
   rewrite); +~3% decode for an afternoon. Fold into the next decode tranche.
5. **Routing orchestration band** (~0.83 ms/token with `dsv4_route`) — only
   reachable via graph capture or aggressive fusion; the whole-step graph
   measured +5.5% (below the 10% bar) — stays parked.

Wall-clock framing (agent shape 3k in / 256 out): today 1.43 s prefill +
2.78 s decode = 4.2 s/request. Down-tile fix → ~3.8 s; attention fix →
~3.2 s; GDR fix → ~2.8 s. Decode and prefill levers are now the same order —
neither side dominates the request anymore.

## Rule

- A licensed kernel is not a finished kernel: re-profile splits the licensed
  aggregate (102 µs/layer "MoE") into healthy (swiglu 25%) and broken
  (down 6%) halves that the A/B alone could not separate.
- σ < 1% on a slow kernel means deterministic structural limit (here MLP),
  not noise or contention — go read the loop trip counts before the grid math.
