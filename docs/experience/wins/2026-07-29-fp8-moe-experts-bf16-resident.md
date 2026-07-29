# FP8 grouped-MoE LoRA re-merge unblocked — BF16-resident experts, CUDA, 2026-07-29

> Status: functional gate PASSED on H20 (3 steps, finite loss); MMLU curve pending

## Context

`arle train opd` with `--lora-target-set all-linear` on an FP8 MoE student
(Qwen3.6-35B-A3B-FP8-iso64) died at step 3: the per-step rollout re-merge folds
LoRA into experts, but the loader routes FP8 experts into the fused grouped-FP8
buffer (`w13_fp8_grouped`/`down_fp8_grouped`) and leaves the per-expert
`Vec<DeviceMatrix>` empty, so `lora_matrix_mut` errored `grouped/FP8 MoE LoRA
sync is not supported`.

## What Worked

Load-time residency switch (`infer_api::set_qwen35_moe_experts_bf16_resident`),
flipped by the trainer when the target set covers experts. When set, the loader
skips the grouped-FP8 concat, keeps experts per-expert, and dequantizes each to
dense BF16 in place — so the whole layer is one uniform `moe_bf16_grouped_gemm`
kernel over a stable pointer table. Per-step merge then rides the proven dense
lane (pristine base D2D cache + B·A GEMM + in-place scaled-add); no realloc, so
the static ptr table the forward reads stays valid.

Rejected the alternative (slice/dequant/merge/requant back into the fused FP8
buffer): reintroduces the 60-83s/round host requant lane the promote-to-bf16
refactor deleted, plus w13 gate‖up fusion and sm120 SFB-transpose handling —
more code, more fragile, no benefit for a short-rollout student.

Per-expert lazy promote (`promote_lora_target_to_bf16`, works for attn/dense)
can NOT apply to grouped MoE: the forward dispatches one kernel per layer keyed
on the per-layer `expert_weight_format`, and reads a static device pointer
table; promoting experts one at a time would leave the layer half-FP8/half-BF16
with no runnable kernel and a ptr table pointing at freed memory. Experts must
convert together, at load.

## Cost

Routed experts resident as BF16 ~16 GB vs ~8 GB grouped FP8 (+8 GB, iso64,
affordable on 96 GB H20); rollout drops to the hand grouped-GEMM path (slower
than DeepGEMM) — acceptable for a short-rollout student. Scope-gated: off =
serving default (grouped-FP8), so serve/teacher and attention-only training are
byte-identical to before.

## Rule

Grouped MoE forward reads a load-time static pointer table + one per-layer
kernel format; any per-step weight mutation must write in place and all experts
in a layer must share one format. Lazy per-tensor promote is an attn/dense-only
trick — it does not generalize to grouped experts.

## Verification (functional gate PASSED)

Mac has no nvcc; typecheck via CI Lint mirror passed
(`CUDARC_CUDA_VERSION=12080 cargo check --no-default-features
--features cuda,no-cuda` for infer-api + cli). H20 functional gate: `arle train
opd --lora-target-set all-linear` on iso64 cleared step 3 (the original failure
point) and ran 3 steps to completion with finite loss `[0.53, 2.21, 0.63]`.

The residency switch alone was not enough — it took two companion fixes to
actually run end-to-end, each surfaced only at runtime on the H20:

1. **Empty per-expert Vec** (commit 227790953): the BF16 dequant flips
   `expert_weight_format` to `DenseBf16`, which made `deepgemm_ready=true` on
   Hopper — the BF16 grouped concat then cleared the per-expert `gate/up/down`
   Vecs the re-merge needs. Fix: gate `deepgemm_ready` on
   `!experts_bf16_resident` (see errors/2026-07-29-bf16-resident-experts-cleared-by-grouped-cache).
2. **VRAM budget** (commit 5563ba032 + ARLE_FORCE_CHECKPOINT): the in-process
   rollout engine profiled its KV pool at `mem_fraction_static=0.9` of
   post-weights free VRAM, and BF16-resident experts doubled routed VRAM — so
   the per-step re-merge scratch OOM'd (step 3), and once the KV pool shrank the
   ckpt-gate false-non-engaged and the backward tape OOM'd (step 1). Fix: new
   `--rollout-mem-fraction 0.1` (single-sequence rollout needs a tiny KV pool)
   + `ARLE_FORCE_CHECKPOINT=1` to force gradient checkpointing the gate's memory
   model underestimated for the MoE all-linear backward.

MMLU curve still pending: serve `step_000050` + aligned MMLU n=100 seed=0 vs the
29-30% merged baseline (>37% = on-policy KL meaningfully recovers accuracy).
