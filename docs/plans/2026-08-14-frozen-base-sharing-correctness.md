# Frozen-base sharing correctness (OPD single-GPU)

Status: implemented and verified 2026-08-14 (`7c4c9082f`). Follow-up to the
alias-UAF fix (`a1a3fda92`, wins entry 2026-08-14-opd-offload-student-alias-uaf).

Verification result: 5-step arms (offload=student vs off, same seed) track
within ±0.5 loss with no divergence trend (D1 closed); layer-0 hidden
sum-squares are digit-identical across arms (167.8612689715945), so the D2
step-1 loss spread is teacher-engine run-to-run nondeterminism, and the same
spread appears between two runs of one arm. Weights carry no defect.

## Problem

`sync_lora_from_store` re-points the trainer's frozen-base tensors at the
engine's resident BF16 projection buffers (`frozen_base_bf16_pointers` →
`import_bf16_device_ptr` → `replace_device_handle`), saving a second base
copy (~28 GB on the 27B). Two defects remain on this path after the UAF fix:

**D1 — merged-alias double-apply.** The exported buffers hold MERGED bytes
(base + LoRA delta; `remerge_student_lora` merges in place, `ef486bd86` keeps
a pristine window only for the next re-merge). `LinearWithLora::forward`
(`crates/train/src/lora.rs:242-249`) computes `base @ x + scale·B(Ax)`
unconditionally when an adapter exists. A LoRA-targeted projection whose base
was re-pointed therefore applies the delta twice from the first step where
B ≠ 0. Untargeted projections are unaffected (merged bytes equal base bytes).
Whether a given config is bitten depends on suffix overlap between the export
table (`qwen35_lora.rs:50-66`) and the trainer's adapter names.

**D2 — step-1 parity anomaly, cause unknown.** With B = 0 the aliased and
owned bases hold numerically identical weights, yet the verification arms
diverged at step 1 (offload=student loss 26.870852 vs offload=off 27.695499,
same seed, same rollout_len 2171). The two arms must agree at B = 0; until
the divergence is explained, neither arm's loss can be called the reference.

## Design

One rule closes D1: **re-point only LoRA-untargeted projections.** In the
re-point loop (`crates/train/src/infer_student.rs:322-352`), skip any entry
whose trainer param has an adapter (`adapter_map` contains
`"{name}.lora_a"`). The trainer keeps its own owned copies exactly for the
targeted projections; for attention-qv on the 27B that is q/v of the
full-attention layers — a fraction of a percent of the base — so the VRAM
saving that motivates sharing is preserved.

Rejected alternatives:
- Trainer skips its delta when the base is a merged alias: couples trainer
  forward semantics to engine merge state; silently wrong the day the engine
  changes its merge policy.
- Export pristine pointers: the pristine cache is a per-projection window
  kept only for re-merge, not a full resident matrix; exporting it would
  require a second full base copy, defeating the sharing.

The offload=student gate and the `frozen_base_ptrs_exported` fail-fast from
`a1a3fda92` stay unchanged.

## Verification

1. **B = 0 parity probe (also resolves D2).** One-step run per arm at the
   verification config, `ARLE_OPD_STEP_TRACE=1`; compare the probe-forward /
   layer-trace hidden sum-squares between the aliased and owned arms. They
   must match to bf16-identical bytes. A mismatch localizes D2's real cause
   before the fix is trusted.
2. **Warmed-B divergence check (D1).** 5 steps per arm, same seed. Before
   the fix, per-step losses diverge from step 2 in any config where the
   export table overlaps LoRA targets; after the fix the arms track within
   noise.
3. Both Mac typecheck lanes; 3-step pod runs of both offload arms stay
   finite.
