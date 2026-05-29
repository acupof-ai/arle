# Plan: route OPD student rollout through the infer engine

**Date**: 2026-05-29
**Status**: approach (pre-implementation; >5 files, crosses train↔infer boundary)
**Supersedes the BF16-KV-cache direction** in
[`docs/research/2026-05-29-online-softmax-null-at-rollout64.md`](../research/2026-05-29-online-softmax-null-at-rollout64.md)
and the rollout-perf hypothesis in
[`docs/research/2026-05-28-opd-rollout-perf-208s-bottleneck.md`](../research/2026-05-28-opd-rollout-perf-208s-bottleneck.md).

## Priority & ROI

**P0 for OPD throughput.** student_rollout is 67% of step time at rollout=128
(208 s / 310 s). The fix unblocks (a) ≥10× more training steps per wall-clock,
(b) rollout=256+ (currently 19 min/step → killed), (c) multi-seed-from-start
training to actually test the null-effect hypothesis.

**ROI anchor (measured, not hypothesis):**

| path | per-token decode | source |
|---|---:|---|
| train-crate hand-written rollout | **1600–2880 ms/tok** (O(n²), grows with ctx) | `docs/research/2026-05-28-opd-rollout-perf-208s-bottleneck.md` |
| **infer engine, Qwen3.5-0.8B BF16, c=1** | **3.5 ms/tok** | `docs/experience/wins/2026-05-25-*` GuideLLM, RTX 4070 Ti SUPER |

Infer is **~500× faster at the decode level**. Even with per-step LoRA-sync
overhead and D2D logit bridging, projected student_rollout at rollout=128:
**208 s → ~1–3 s**. Step time **310 s → ~100 s** (then backward-bound at 78 s,
the next axis). This is far above the original research-doc projection of "5×"
(which conservatively assumed infer kept O(n²) — it does not at this scale).

## Why not BF16 KV cache (the abandoned direction)

The BF16 KV cache / online-softmax work optimized the **train-crate
hand-written decode kernel** — the wrong layer. Even a successful 2× on its
O(n²) constant leaves it at ~800 ms/tok, still ~230× slower than infer.
kernel-optimization skill Phase 1.5 escape hatch: stop tuning a path that is
structurally beaten; route through the path that already has the fast kernel
(CUDA graph + paged KV + no autograd machinery). The dead BF16 kernels were
deleted in `03cf1bc8`; the active `online_f32` path and the whole train-crate
decode path get deleted as one unit when this lands (no half-states).

## Key de-risking finding — bit-parity is NOT required

`opd.rs:1790` `backward_chunked_kl_rollout(student, teacher, &rollout, ...)`
consumes **only the token sequence** `rollout: Vec<u32>`. Rollout logits are
used for argmax (`opd.rs:1694`) then discarded; KL gradients **recompute**
student logits with tape enabled on the fixed sequence. Therefore the rollout
only needs to produce a *plausible on-policy* sequence — infer-student and
train-student share LoRA-synced weights, so BF16-vs-F32 argmax flips do not
break correctness. This collapses the research doc's "must be bit-identical"
hazard to a much weaker canary (see kill criteria).

## Architecture — mirror the teacher

The teacher already runs through infer in-process:
`InferTeacher` (`crates/train/src/teacher_infer.rs:617`) holds
`Arc<Mutex<LoadedInferenceEngine>>` + a `train_backend` bridge, and calls
`engine.forward_token_logits(input_ids, positions)`
(`infer/src/server_engine/loaded.rs:127` → `scheduler/types.rs:1134`),
bridging BF16 device logits → F32 via `import_bf16_device_ptr_as_f32`.

The student mirrors this with one addition the teacher doesn't need:
**per-step LoRA sync** (teacher weights are frozen; student LoRA updates every
step).

### Per-step loop
1. **LoRA sync**: push student LoRA params (train `TensorStore` TensorIds,
   `qwen35.rs:1026` `adapter_names`) into the infer student engine's adapter.
2. **Rollout**: greedy single-token decode via infer `forward_token_logits`,
   argmax on host or device, append to `rollout: Vec<u32>`.
3. **KL/backward**: unchanged — `backward_chunked_kl_rollout` on `rollout`.

## The one hard problem — per-step LoRA sync

Infer currently merges LoRA into dense weights **once at load**
(`infer/src/backend/cuda/bootstrap.rs:221` `load_and_attach_lora`, disk via
`INFER_LORA_PATH`). Two implementation options:

- **Option A — un-merged in-memory delta (preferred).** Infer has
  `decode_batch_lora_body` + `apply_lora_{gemv,gemm}_add` hooks
  (`infer/src/model/qwen35/lora.rs`, `forward.rs`) that suggest an un-merged
  LoRA application path exists. If the student engine keeps LoRA as a separate
  device delta (not merged), per-step sync = a D2D copy of the A/B matrices
  (rank≪hidden, tiny). Needs a new in-memory adapter-update entry point in
  infer. **Cleanest; ~µs sync.**
- **Option B — disk reload (fallback v1).** Write adapter safetensors, reload
  the student engine adapter. ~10–100 ms/step but **once per step**, not per
  token → amortizes to <1 ms/tok over 130 tokens. Trivial to implement if
  re-merge doesn't require a full base-weight reload (verify). Use as the
  bring-up path if Option A's hooks aren't ready.

**Open question to resolve in implementation:** does infer's un-merged LoRA
path (Option A) already run end-to-end for decode, or is merge-at-load the
only working path? (Workflow agent flagged hooks exist but unconfirmed.)

## Implementation phases

- **P1** — `InferStudent` engine bring-up: load Qwen3.5-0.8B-Base into a second
  in-process infer engine alongside the teacher; smoke `forward_token_logits`.
- **P2** — LoRA sync (Option A if viable, else B): train LoRA → infer student
  adapter; canary that infer-student logits track train-student.
- **P3** — swap the rollout loop (`opd.rs:1654-1776`) to call infer student;
  keep the train-crate path behind a flag for A/B.
- **P4** — bench + gate; on pass, **delete the train-crate decode path**
  (online_f32 + legacy + `forward_rollout_cached*`) as one deletion unit.

## Kill criteria (license-or-kill, explicit thresholds)

- **PASS**: step-time at rollout=128 drops ≥2× (310 s → ≤155 s) AND the
  rolled sequence is on-policy — step-1 token agreement vs train-crate argmax
  **≥90%** (canary: <90% means LoRA sync is wrong or numerics diverge badly,
  not just BF16 rounding).
- **KILL**: <1.5× step-time OR step-1 agreement <60% (LoRA-sync bug) →
  keep train-crate rollout, investigate per-step LoRA-sync cost / correctness
  before retrying.
- Multi-seed capability re-eval (the actual OPD-effect question) is downstream
  and gated separately — this plan is a **throughput enabler, not a capability
  claim**.

## Out of scope

- Backward optimization (78 s, 25% of step) — next axis after rollout.
- Quantized (W4A8) student — train loads only F32/BF16; defer.
- Sampling/beam rollout — OPD uses greedy.
