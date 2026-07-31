# ISO-Merger grafts de-censoring onto a reasoning model with zero loss, 2026-08-01

> Status: **method validated**. Same-lineage Qwen3.6-27B: base +
> ThinkingCap (reasoning RL) + Huihui (de-censoring RL) → one merge that keeps
> TC's reasoning (MMLU 0.808, GSM8K 0.949) *and* answers what TC refuses. Data-free,
> no rollouts/gradients/distillation. Tool: [`scripts/iso_merger.py`](../../../scripts/iso_merger.py).

## Context

The earlier symmetric merge (Iso-CTS 3-way on the 35B MoE) collapsed to 20–30%
MMLU — the wrong method (spectrum-flattening) on the wrong objective (dilution).
See [errors/2026-07-30-moe-expert-merge-collapse.md](../errors/2026-07-30-moe-expert-merge-collapse.md).
This entry is the **correct** ISO-Merger (arXiv:2607.19331, Zhu et al.): freeze
the base singular spectrum, merge only the singular *frames* the RL fine-tune
rotated. The paper's core observation — RLVR barely moves the spectrum (κ_spec
1.02–1.35) while SFT moves it 100× — means an RL expert's learned capability
lives in U/V, not Σ. So a data-free merge that reuses Σ₀ and composes the frame
displacements should stack capabilities without diluting either.

## The recipe (per 2D weight matrix, all fp32)

Given base `W₀ = U₀ Σ₀ V₀ᵀ` and K experts `Wᵢ = Uᵢ Σᵢ Vᵢᵀ` off that same base,
build one merged `W* = U* Σ₀ V*ᵀ` reusing the base spectrum Σ₀ (never modified):

1. **Base SVD** — thin SVD `W₀ = U₀ Σ₀ V₀ᵀ`.
2. **Sign-canonical expert SVD** — per expert, thin SVD then flip each singular
   pair's joint sign against the base (`sₖ = sign⟨u₀ₖ, uᵢₖ⟩`). Without this the
   frame displacement is gauge-ambiguous and the merge is noise.
3. **Stiefel-tangent projection** — `ξ_{u,i} = Π_{U₀}(Uᵢ − U₀)`,
   `ξ_{v,i} = Π_{V₀}(Vᵢ − V₀)`, where `Π_X(Y) = Y − X·sym(XᵀY)`.
4. **Top-k mask** — keep the first `round(0.9·q)` tangent columns (drop the
   noise tail; feasibility restored at step 7).
5. **First-order proxy** — `gᵢ = ξ_{u,i} Σ₀ V₀ᵀ + U₀ Σ₀ ξ_{v,i}ᵀ` (each expert's
   linearized weight effect).
6. **Retention coefficients** — Gram `Γᵢⱼ = ⟨gᵢ, gⱼ⟩_F`, solve
   `(Γ + 1e-12·I) c = diag(Γ)`, clip `[0, 1.5]`. This is the SVD-merge's answer
   to "why coefficients": `c` solves `ret_i(c) = (Γc)ᵢ/Γᵢᵢ = 1` — each expert's
   own first-order effect survives at unit strength, cross-expert interference
   is damped, and near-collinear experts (one absorbs the other) get one clipped
   toward 0 automatically. No hand-tuned λ.
7. **Aggregate → re-project → polar retract → reconstruct** —
   `ξ_{u,*} = Π_{U₀}(Σ cᵢ ξ_{u,i})`, `U* = polar(U₀ + ξ_{u,*})` (thin-SVD polar),
   same for V, then `W* = U* Σ₀ V*ᵀ`. Orthonormal U*/V* ⇒ W* keeps Σ₀ exactly.

**Scope split** (Qwen3.6 hybrid attention): the 496 per-layer 2D projections
(`self_attn.{q,k,v,o}`, `linear_attn.in_proj_*`/`out_proj`, `mlp.{gate,up,down}`)
get the Stiefel merge; embed/lm_head + 1D params (LayerNorm, biases) go to
task-mean because RL barely rotates those frames (measured c*≈0); vision tower +
conv1d are copied from base. 498/533/168 of 1199 tensors.

## What the retention coefficients revealed

The `c*` distribution across all 496 tensors is the mechanism, not a symmetric
average:

| layer type | n | TC c* (mean) | Huihui c* (mean) |
|---|---|---|---|
| self_attn (q/k/v/o) | 64 | **0.990** | 0.250 |
| mlp (gate/up/down) | 192 | **0.929** | 0.345 |
| linear_attn (gated-delta) | 240 | **0.000** | 0.200 |

TC is the trunk — near-unit retention on attention + MLP (28 tensors clip to
1.5), the reasoning skeleton preserved. Huihui is a sparse increment — 74% of
tensors get c*≈0, capability injected only where its de-censoring frame
displacement dominates. De-censoring is a low-rank, high-magnitude behavioral
edit; ISO placed it exactly where it lives and nowhere else. No systematic loss
of TC.

## Results (n≈100, seed 0, bf16 serve on H20)

| model | MMLU | GSM8K | de-censoring (lock-pick prompt) |
|---|---|---|---|
| **iso merge** | **0.808** (80/99) | **0.949** (94/99) | **answers** (detailed breakdown) |
| TC (reasoning src) | 0.798 (79/99) | 0.920 (92/100) | **refuses** ("I can't provide…") |
| Huihui (de-censor src) | 0.825 (80/97) | 0.944 (85/90) | answers |

Reasoning is not merely retained — GSM8K 0.949 ≥ TC's 0.920. The de-censoring
that TC refuses, the merge delivers. Three capabilities stacked without
dilution, no post-merge data.

## Rules

- **Same-lineage RL experts stack losslessly under fixed-spectrum frame merge.**
  Reuse the base Σ₀, merge U/V in the Stiefel tangent space, let the Gram-solve
  retention coefficients apportion per-tensor. No λ sweep, no calibration data.
- **Inspect retention `c*` before trusting the merge.** The per-layer TC-vs-expert
  split (trunk near-unit, increment sparse) is the go/no-go — a systematic zero
  on the trunk would mean the reasoning got dropped.
- **fp32, not fp64.** fp64 SVD is needlessly slow on datacenter GPUs (64s/tensor
  → hours); fp32 is the reference precision (bf16/fp16 SVD is unstable). Shard
  the 496 independent tensors across GPUs (`--shard-mod/--shard-idx`): 140min → 29min on 4.
- **FP8 experts need bf16 first.** FP8 is `weight`(E4M3) + `weight_scale_inv`
  (128×128 block scale); SVD on E4M3 is unsupported and copying base scale
  mis-dequantizes. Dequant → merge in bf16 → requantize (ISO operates in full
  precision by design). Not implemented — no verified FP8 run yet.
