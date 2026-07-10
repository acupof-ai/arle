# OPD P2 — in-memory student LoRA re-merge sync into InferStudent

**Date**: 2026-05-29
**Track**: CUDA / train (OPD throughput enabler)
**Plan**: [`docs/plans/2026-05-29-opd-student-rollout-via-infer.md`](../../plans/2026-05-29-opd-student-rollout-via-infer.md) §P2
**Status**: PASS — sync correctness canary 100% / 100%

## Context

P1 brought up `InferStudent` (`crates/train/src/infer_student.rs`) routing the
OPD student rollout through the in-process infer engine. P2 adds the one thing
the teacher path doesn't need: **per-step LoRA sync**. The student's LoRA
updates every training step, but infer's only Qwen3.5 LoRA path is
merge-at-load (`infer/src/model/qwen35/weights.rs` `load_and_attach_lora` →
`merge_lora_into_dense_matrix`), which mutates q/v in place and keeps **no base
copy** — naive re-merge accumulates deltas.

Chosen mechanism (plan design-subagent verdict): **B1.5 in-memory re-merge from
a cached base**. Cache the 6×(q,v)=12 pristine BF16 matrices at LoRA-enabled
load; each step restore base + merge the fresh adapter + re-upload.

## What Worked

**Infer side** (`Qwen35Model`):
- New `lora_base_cache: Option<Vec<LoraBaseLayer>>` field; `cache_lora_base`
  snapshots pristine q/v host BF16 for every full-attention layer **before the
  first merge** (idempotent — never overwrites once set).
- `remerge_lora(&StudentLoraUpdate)`: `restore_lora_base` → `merge_lora`.
  Idempotent across steps (deltas never accumulate; each call starts from the
  same snapshot).
- Public data type `StudentLoraUpdate` (raw un-scaled A `[r,in]` / B `[out,r]`
  + `r`/`alpha`); `Qwen35LoRA::from_student_update` reuses the existing merge
  kernel so `scale = alpha/r` is applied **exactly once**.
- Routed up the stack mirroring `forward_token_logits`: new `RemergeLoraRequest`
  channel → `SchedulerHandle::remerge_student_lora` →
  `RequestHandleInferenceEngine` → `LoadedInferenceEngine::remerge_student_lora`.
  Re-merge runs on the single-writer scheduler thread that owns the model.
  Default-impl `ModelForward::remerge_student_lora` bails; only `Qwen35Model`
  overrides — no cfg-leak across backends.

**Train side**:
- `InferStudent::sync_lora_from_store(store, adapter_map, lora_config)` D2H's
  the 24 F32 adapter tensors (q/v A/B × 6 full-attn layers) from the train
  `TensorStore`, groups by absolute layer index, and pushes a raw
  `StudentLoraUpdate`. Recognizes q/v adapters only (target set `AttentionQv`).

## Results — canary (`test_infer_student_lora_sync`, RTX 4070 Ti SUPER, sm89)

Qwen3.5-0.8B-Base, r=8 α=16 AttentionQv, 64-token greedy rollout, c=1,
`mem_fraction_static=0.05`. Agreement = infer-student argmax vs train-student
(F32) argmax over the generated tail.

| step | LoRA B | agreement | note |
|---|---|---:|---|
| **A** floor | zero-init (== base) | **100.0%** | BF16(infer)-vs-F32(train) numeric floor |
| **B** sync | non-zero deterministic | **100.0%** | post-`sync_lora_from_store` |

Non-zero adapter changed the train tail vs Step A (`760,4128,…` → `40,1044,…`),
so the sync test is not vacuous: the infer student tracked the perturbed
train student exactly.

**Verdict: PASS** (≥90% threshold). Sync is correct and idempotent. 24 adapter
tensors confirmed (6 full-attn layers, the 18 linear-attn layers carry none).

## Rule

- Per-step LoRA reuse on a merge-at-load backend **requires a cached pristine
  base + restore-then-merge**; in-place re-merge silently accumulates deltas.
- Export **raw** (un-scaled) A/B + `r`/`alpha`; let the one merge path apply
  `scale = alpha/r` once. Pre-scaling double-applies.
- The reference-rollout half of a cross-engine canary must `retain_ids(params)`
  after each forward, or the O(n²) train path OOMs the second in-process CUDA
  engine (first canary run died on `cuda alloc_zeros failed` before this).

## Scope / next

- Build only: opd.rs rollout loop swap is **P3** (this lands the sync API +
  canary, not the production wiring). Train-crate decode-path deletion is P4.
- Infer lib + train lib compile clean under `--features cuda`; the new test +
  touched files are clippy-clean. (Pre-existing `infer` bin nccl errors and
  ~2 pre-existing train-lib clippy warnings are untouched.)
