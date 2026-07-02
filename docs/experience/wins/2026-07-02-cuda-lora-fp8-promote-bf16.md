# Qwen3.5/3.6 LoRA sync: promote FP8 targets to dense BF16 on first touch

**Date:** 2026-07-02. **Backend:** CUDA, Qwen3.6-35B-A3B-FP8 (agent-OPD rollout
engine). **Scope:** `infer-cuda/src/qwen35.rs` LoRA re-merge path only.
**Status: VERIFIED on H20** (run-tail3-promote, 3-round A/B vs run-tail3-probe):
`phase=sync_lora` 83.2/60.9/60.9 s → **0.035/0.018/0.019 s** (~3000×; round 0
includes the one-time device dequant promotion). All rounds `passed=1` (rollout
rides the promoted dense-BF16 matrices), loss 0.2849/0.3437/0.2706 — in band,
and now numerically HONEST: the old host lane rewrote the shared FP8 base in
place each round, so the autograd student's "frozen" base drifted with the
adapter delta; promotion + keepalive reads pristine base forever.

## Context

Agent-OPD spent 60-83 s PER ROUND in `sync_lora_from_store` →
`remerge_student_lora` → `merge_lora_proj`. FP8-block-scaled weight targets took
a host lane: D2H snapshot + scalar `W = base + scale·B·A` triple loop over
rows×cols×rank (~1e9 MACs per matrix, single-threaded, ×~128 matrices/round) +
host re-quant + full-weight upload. The dense-BF16 lane
(`merge_lora_proj_device`) was already fully on-device and takes milliseconds.

## What Worked

Promote-on-first-touch: `promote_lora_target_to_bf16` dequantizes the FP8
target on device (`dequantize_fp8_block_scaled_to_bf16_cuda`, one-time) and
swaps the matrix's resident storage to dense BF16 in place; every later
re-merge rides the single on-device dense lane (device base cache → rank-r GEMM
→ scaled-add). The entire host lane was deleted (~1.1 kLOC): host base
snapshots, host triple loop, host FP8 quant/dequant, grouped-FP8 slice uploads,
and the `LoraWeightTarget{,Mut}` fork — grouped DeepGEMM expert targets now
error at target resolution (they were only reachable through the deleted host
lane; per-expert matrices still merge).

VRAM: FP8→BF16 doubles storage for the *touched* projections only. When
`--share-frozen-base` exported the FP8 base pointers (the autograd student
holds non-owning views), the retired FP8 buffers are kept alive in
`lora_promoted_fp8_keepalive` — freeing them would dangle the student's aliased
frozen base; without sharing they are freed on promotion.

## Rule

A one-time on-device format promotion beats a per-round host requant loop
whenever both formats are first-class in serving; on first touch, retire (don't
free) buffers whose raw pointers were exported to a co-resident consumer.
