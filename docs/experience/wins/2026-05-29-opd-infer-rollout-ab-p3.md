# OPD P3 — route student rollout through InferStudent (flagged) + A/B

**Date**: 2026-05-29
**Track**: CUDA / train (OPD throughput enabler)
**Plan**: [`docs/plans/2026-05-29-opd-student-rollout-via-infer.md`](../../plans/2026-05-29-opd-student-rollout-via-infer.md) §P3
**Status**: PASS — 5.0× step / 60.9× rollout, KL matched, no OOM

## Context

P1 brought up `InferStudent` (`crates/train/src/infer_student.rs`,
`decode_next_token`); P2 added `sync_lora_from_store` (100% canary, bit-exact
re-merge). P3 wires the rollout swap into the OPD step behind a flag and runs a
matched A/B.

The train-crate hand-written greedy rollout is 67% of step time at rollout=128
(O(n²) per-token decode, no fast kernel). P3 routes the rollout through the
in-process infer engine (CUDA graph + paged KV) — the path that already has the
fast kernel — keeping the train-crate path as the default for A/B.

## What Worked

**`crates/train/src/opd.rs`**:
- `InferRolloutCtx<'a>` (cuda-gated): `{ student: &InferStudent, lora_config }`.
- `infer_rollout_flag_enabled()` reads `ARLE_OPD_INFER_ROLLOUT` (`1`/`true`).
- `opd_step_with_teacher_forward_profiled_gkd_anchor` takes a cuda-gated
  `Option<InferRolloutCtx>`. When `Some`: once per step `sync_lora_from_store`
  mirrors the train LoRA into the infer student, then greedily decodes
  `rollout_len` tokens via `decode_next_token` (P1's re-prefill pattern),
  producing the **same** `rollout: Vec<u32>` the train path would — downstream
  `backward_chunked_kl_rollout` is byte-for-byte unchanged. Train-crate path
  (flag off / `None`) is the default A/B baseline; both coexist (no half-state).
- The new param is `#[cfg(feature = "cuda")]` on the fn signature so the
  non-cuda public API and CPU tests (`test_opd_step.rs`) are unchanged.

**`crates/train/examples/opd_step_cuda_infer_teacher_train.rs`**:
- When the flag is on, writes the train student's current q/v adapter to a temp
  PEFT dir, sets `INFER_LORA_PATH`, loads a **second** infer engine from the
  student dir (so `cache_lora_base` snapshots the pristine BF16 base at load —
  the prerequisite `remerge_lora` bails without), then unsets the env. Threads
  `infer_student.as_ref()` into `run_training` → `opd_step`. No VRAM paid when
  the flag is off.

### Gotcha (one-strike fix)

First infer-arm run died: `remerge_lora requires a cached LoRA base; load the
model with an adapter (INFER_LORA_PATH)`. The infer base cache is populated
**only** by `load_and_attach_lora` (i.e. only when `INFER_LORA_PATH` is set at
load). Fix mirrors P2's canary: seed a PEFT adapter dir + `INFER_LORA_PATH`
before loading the student engine.

## Results — matched A/B (RTX 4070 Ti SUPER, sm89, CUDA 13.2)

Same binary, same prompts (`examples/opd/sample-prompts.jsonl`, 20 rows),
Qwen3.5-0.8B-Base teacher+student, rollout=128, 3 steps, r=8 α=16 AttentionQv,
lr=1e-5, `mem_fraction_static=0.05`, CUDA graph on. Two arms back-to-back:
BASELINE (flag off) then INFER (`ARLE_OPD_INFER_ROLLOUT=1`).

| metric | BASELINE | INFER | Δ |
|---|---:|---:|---:|
| **mean step (s)** | **249.88** | **50.07** | **−79.96% / 4.99×** |
| median step (s) | 248.19 | 49.54 | 5.01× |
| total wall (3 steps, s) | 752.76 | 153.61 | 4.90× |
| **student_rollout mean (s)** | **203.27** | **3.34** | **−98.4% / 60.9×** |
| backward mean (s) | 38.70 | 38.89 | unchanged (now dominant, 78%) |
| student_forward mean (s) | 7.04 | 7.06 | unchanged |
| teacher_forward mean (s) | 0.63 | 0.51 | — |
| infer_sync mean (s) | — | 0.20 | per-step LoRA re-merge |

Per-step student_rollout: BASELINE 204.7 / 203.5 / 201.6 s; INFER 3.38 / 3.19 /
3.45 s.

### KL / loss sanity (finite + comparable magnitude — not bit-parity)

| step | BASELINE loss | INFER loss |
|---|---:|---:|
| 1 | 1.0384e-4 | 1.0362e-4 |
| 2 | 1.0575e-4 | 1.0461e-4 |
| 3 | 0.9658e-4 | 0.9358e-4 |

All finite, same order of magnitude, <3% apart per step. The slight drift is
expected: BF16(infer) vs F32(train) argmax can flip, producing a slightly
different on-policy rollout — but the plan established bit-parity is **not**
required (`backward_chunked_kl_rollout` consumes only the token sequence). No
NaN, no order-of-magnitude divergence → no red flag.

### Peak VRAM (16 GB card, `nvidia-smi` 2 Hz)

| arm | engines resident | peak |
|---|---|---:|
| BASELINE | 1 infer (teacher) + train student | **7234 MiB** |
| INFER | 2 infer (teacher + student) + train student | **7588 MiB** |

The second in-process engine added only **+354 MiB** (student is 0.8B BF16 +
small KV at `mem_fraction_static=0.05`). The OOM risk flagged for P3 did **not**
materialize at rollout=128; ~8.8 GB headroom remains.

## Verdict: **LICENSE (PASS)**

- step-time ≥2× → **4.99×** ✅
- student_rollout ~24× projected → **60.9×** ✅ (re-prefill at the infer
  engine's fast kernel beats the train O(n²) decode by far more than projected)
- KL finite + comparable ✅
- no OOM ✅

Backward (38.9 s, 78% of step) is now the bottleneck — the next OPD throughput
axis, as the plan anticipated.

## Rule

- Routing a hand-written O(n²) decode through the production infer engine
  (CUDA graph + paged KV) is the right escape hatch when the slow path is
  structurally beaten — don't keep tuning the dead kernel's constant.
- A second in-process infer engine for the student loading from a merge-at-load
  backend **must** be seeded with `INFER_LORA_PATH` at load so `cache_lora_base`
  snapshots the pristine base; otherwise per-step `remerge_lora` bails.
- bit-parity is not required when the downstream consumer takes only the token
  sequence; gate on finite + comparable-magnitude KL, not exact match.

## Scope / next

- Flag default **OFF**; both rollout paths coexist intentionally. P4 deletes the
  train-crate decode path (`forward_rollout_cached*` + online_f32) on this
  confirmed pass.
- Next axis: backward (now 78% of step).
