# Qwen3.6-35B-A3B MoE CUDA — e2e VALIDATED on H20 (real model, coherent output)

## Context

First end-to-end run of the Qwen3.5-MoE / Qwen3.6 CUDA grouped-MoE forward
(commit `3fe673b7`) on a **real production model**: `Qwen3.6-35B-A3B` (BF16,
67 GB, 40 layers, **256 experts, top-8**, hidden=2048, moe_inter=512, hybrid
3:1 linear/full attention, head_dim=256). Hardware: one **H20 (sm_90, 102 GB)**.
The earlier `tiny-random/qwen3.5-moe` smoke target was unusable (degenerate
head_dim=32, no matching paged-attention kernel), so this is the first time the
MoE forward kernels actually executed.

## What Worked

**Coherent generation from the real model** (greedy, 40-token prompt, H20):

> prompt: "…In this short essay I will describe"
> → `" the history of artificial intelligence and the current approaches.\n\nThe history of artificial intelligence began in the nineteen fifties when"`
> (24 completion tokens, `finish_reason=length`, sensible English.)

The full MoE GPU path is proven on the production shape — route
(`dsv4_route_cuda`, 256 experts / top-8) → count → exclusive-scan → pack →
**BF16 grouped pair GEMM (gate+up)** → SwiGLU → **BF16 grouped down GEMM** →
scatter+scale → gated shared expert — across all 40 layers with real weights.
A per-stage probe trace under `CUDA_LAUNCH_BLOCKING=1` confirmed every MoE
kernel completes for the 2048-token warmup forward.

Two enabling fixes landed with this entry (both REQUIRED to reach the MoE; **neither
is the MoE kernel itself** — the MoE forward was correct as written in `3fe673b7`):

1. **Stacked+fused expert loader** (`weight_loader::load_stacked_expert_2d` +
   the `moe::load_moe_mlp` layout branch). The real checkpoint ships routed
   experts as `experts.gate_up_proj [E, 2·moe_inter, hidden]` (gate‖up fused on
   the output axis) + `experts.down_proj [E, hidden, moe_inter]` — **not** the
   per-expert `experts.{i}.{gate,up,down}_proj` layout the tiny-random smoke
   used. The loader now slices each expert out of the stacked tensor (and splits
   gate/up) into the same `Vec<DeviceMatrix>`, bit-identical to the per-expert
   path. Per-expert and stacked+fused are both supported; legacy `switch_mlp.*`
   still rejected.
2. **TileLang AOT sm_90a** (`gen_tilelang_aot.py`). Hopper TileLang attention
   kernels emit CUTLASS WGMMA (`wgmma.fence`), which nvcc only enables under the
   architecture-accelerated `sm_90a` target (defines
   `CUTE_ARCH_MMA_SM90A_ENABLED`). The generator compiled with plain `sm_90`
   → runtime `cute::warpgroup_arrive(): Attempting to use wgmma.fence without
   CUTE_ARCH_MMA_SM90A_ENABLED` device assert. **TileLang attention had never
   run on H20 before** (DSv4 uses FlashMLA, V100 uses the patched sm_70
   TileLang), so this latent build bug surfaced on the first Qwen3.x-on-H20 run.

H20 build recipe (sm_90): `CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0
ARLE_CUDA_DISABLE_FLASHMLA=1 cargo build --release -p infer --features cuda
--bin infer`. Model fetched via `oniond download model Qwen3.6-35B-A3B` (TOS
bucket, ~9 s for 67 GB). Loadability was source-verified first: arch
`Qwen3_5MoeForConditionalGeneration` → `Qwen3_5_Moe` (model_registry:142),
config nested in `text_config` (qwen35-spec accepts both layouts), tensor prefix
`model.language_model.*` matches the loader, expert dims pass the W4A8 shape gate
(in%128==0 && out%256==0, g128) for the planned quant follow-up.

## Known limitation (filed separately)

Prompts with `seq_len` below the gated-delta chunk span (≈32 tokens) **hang** in
`gated_delta_rule_prefill_chunkwise_batch` (linear-attention prefill,
partial-chunk-only path) — a pre-existing linear-attn issue, **not** the MoE
kernels (decode single-step **and** ≥40-token prefill both produce coherent
output). 2-token and 11-token prompts hang as the first request (fresh state);
40-token works. See
[`errors/2026-05-30-gated-delta-short-seq-prefill-hang-h20.md`](../errors/2026-05-30-gated-delta-short-seq-prefill-hang-h20.md).
This blocks a default-shape `guidellm` perf sweep (short prompts); **perf
numbers pending the short-seq fix** (`pending-remote`).

## Rule

The real production checkpoint's expert layout (stacked+fused `gate_up_proj`) and
the H20 WGMMA arch (`sm_90a`) are BOTH things a tiny-random / sm_70 smoke cannot
surface. Validate a new model path on the **real checkpoint + the real target
SKU** before claiming "supported" — the tiny-random smoke "passed loading" but
the real model needed a different loader entirely, and the V100/sm_70 build never
exercised the Hopper WGMMA attention path.
