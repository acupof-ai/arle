# OPD Teacher Expansion: API Teachers + Multi-Teacher Routing

> Status: added to OPD task queue on 2026-05-21 after the local 9B checkpoint
> availability check. This plan is intentionally separate from the 9B→0.8B
> distillation plan because API-backed teachers and multi-teacher routing change
> the OPD teacher contract.

## Current State

Train-side OPD already has a teacher abstraction:

- `crates/train/src/teacher_infer.rs::TeacherForward`
- `InProcessTeacher`: wraps a frozen train-side `Qwen35Model`
- `InferTeacher`: wraps an in-process `infer::LoadedInferenceEngine` and
  returns raw logits through the device bridge

That means ARLE can use four teacher classes today:

1. A train-side frozen `Qwen35Model`
2. A local `infer` runtime teacher through `InferTeacher`
3. A full-logits HTTP teacher through `ApiTeacher`
4. A deterministic token-prefix multi-teacher router through `MultiTeacher`

`ApiTeacher` is intentionally a full-logits API, not an ordinary chat API:
it imports f32/bf16 logits into the caller `TensorStore` and can be routed
through the same OPD KL path as local teachers. It is correct-first and
profiled, but not claimed as a latency win over in-process `InferTeacher`.

## Why Ordinary Chat/Completions APIs Are Not Enough

OPD's current loss is forward KL distillation:

```text
KL(teacher_distribution || student_distribution)
```

That requires teacher logits, or at least a numerically faithful teacher token
distribution, over the same vocabulary as the student. A normal chat API that
returns only generated text is not a valid substitute for the teacher forward
pass. `top_logprobs` is also insufficient for the current full-vocab KL unless
we deliberately change the objective to top-k distillation.

Therefore an API-backed OPD teacher needs one of these contracts:

1. **Full-logits API**: `input_ids + positions -> [seq_len, vocab_size]`
   logits, preferred for correctness.
2. **Top-k distribution API**: `input_ids + positions -> top_k token ids +
   logprobs`, only after adding a separate top-k distillation loss.
3. **Text-only API**: not accepted for OPD KL; usable only for prompt/answer
   data generation, not this loss path.

## Local 9B Availability Result

**2026-05-22 update — local 9B weights deleted post-cycle.** The 2026-05-22
BF16 frozen-base cycle proved same-card 9B OPD is hardware-blocked on
16 GB (teacher+student+tape co-residency, peak 15871/16384 MiB before
backward). Cycle wrap:
[`docs/projects/2026-05-22-opd-9b-teacher-bf16-cycle-wrap.md`](../projects/2026-05-22-opd-9b-teacher-bf16-cycle-wrap.md).
~28 GB reclaimed. Re-download from ModelScope if/when a 9B retry on
≥24 GB hardware or via remote `ApiTeacher` becomes viable.

The historical inventory below records what was on disk during the cycle:

| Checkpoint | Local status (pre-cleanup) | Runtime status |
| --- | --- | --- |
| `Qwen/Qwen3.5-9B` BF16 | Complete, 18.0 GiB files (DELETED) | Fails ARLE CUDA serve on 16 GB: H2D OOM before readiness |
| `Qwen/Qwen3.5-9B-Instruct` | Directory exists, no weights (DELETED) | Not usable |
| `DavidWen2025/Qwen3.5-9B-GPTQ-4bit` | Complete, 10.35 GiB safetensors (DELETED) | Experimental GPTQModel W4 loader reaches HTTP readiness behind `INFER_EXPERIMENTAL_GPTQMODEL_W4=1`; generation quality gate fails with repeated `!`, so not OPD-licensed |
| `RedHatAI/Qwen3.5-9B-FP8-dynamic` | Metadata only (DELETED) | Loader blocked: compressed-tensors `.weight_scale` unsupported |

Raw evidence:

```text
bench-output/2026-05-21-qwen35-9b-local-availability/
```

Research note:

```text
docs/research/2026-05-21-arle-qwen35-9b-local-availability.md
```

## Work Queue

### T1 — External Full-Logits Teacher API — DONE

Add an `ApiTeacher` implementation of `TeacherForward`.

Minimum wire contract:

```text
POST /v1/token_logits
{
  "input_ids": [u32],
  "positions": [u32],
  "dtype": "bf16" | "fp32"
}

response:
{
  "shape": [seq_len, vocab_size],
  "dtype": "bf16" | "f32",
  "logits_b64": "..."
}
```

Implementation landed:

- Commit: `c0a2975 feat(opd): add external API teacher logits bridge`
- Code: `crates/train/src/teacher_infer.rs::ApiTeacher`
- Supported response formats:
  - JSON `logits: Vec<f32>`
  - little-endian `logits_b64` with dtype `f32` / `float32`
  - little-endian `logits_b64` with dtype `bf16` / `bfloat16`
- Profile counters: HTTP, decode, upload, total.

Constraints:

- Same tokenizer/vocab as the student.
- Same token ids as the student prompt path.
- Full-vocab logits are the correctness baseline.
- Host transfer is acceptable for v1: `8 * 248k * 2 bytes ~= 4 MB` per
  rollout scoring call at Qwen3.5 vocab size.

License gates:

- Self-teach parity: API teacher vs in-process teacher top-64 dominant-logit
  relerr <= 5e-2 on Qwen3.5-0.8B.
- OPD correctness: held-out KL decreases over a 100-step smoke.
- Perf: API teacher wall-clock is reported honestly; no win claim unless it
  beats the local `InferTeacher` baseline for the same model.

Kill gates:

- Missing full logits or incompatible tokenizer: KILL for OPD KL.
- Top-k only: route to T2 top-k loss first, not the full-KL path.

### T2 — Optional Top-K Distillation Loss

If the only available teacher API exposes top-k logprobs, add a separate loss
instead of pretending it is full KL.

Candidate objective:

```text
sum_{token in teacher_top_k} p_teacher(token) *
  (log p_teacher(token) - log p_student(token))
```

Open issues:

- Renormalization over top-k changes the objective.
- Missing tail probability can bias gradients.
- Needs an A/B against full-logits KL on the same local model before use.

License gate:

- On a local model where full logits are available, top-k loss must preserve
  the direction of improvement: held-out KL/NLL improves within 10% of the
  full-logits run over 100 steps.

### T3 — Multi-Teacher Router — PROMPT ROUTER DONE

Add a `MultiTeacher` implementation of `TeacherForward`.

Supported modes, in implementation order:

1. **Prompt-router mode**: one teacher chosen per prompt via metadata
   (`domain`, `skill`, or explicit `teacher_id`).
2. **Weighted ensemble mode**: aggregate multiple same-vocab teachers into one
   teacher distribution.
3. **Confidence-router mode**: choose teacher by entropy / max probability /
   verifier score.

Prompt-router is first because it preserves the current KL path exactly: each
OPD step still has one teacher distribution.

Implementation landed:

- Commit: `0bfa852 feat(opd): add multi-teacher routing abstraction`
- Code: `crates/train/src/teacher_infer.rs::MultiTeacher`
- Routing mode: deterministic longest token-prefix match with configured
  default teacher.
- Safety: validates all teachers share `vocab_size`; unions all local teacher
  parameter ids so cleanup does not free in-process teacher tensors.

Still open:

- CLI / JSON config surface for named teachers and token-prefix routes.
- Per-step route logging in the real OPD harness.
- Weighted ensemble and confidence-router modes.

Weighted ensemble requires a stable distribution aggregation:

```text
p_ensemble = sum_i w_i * softmax(logits_i)
teacher_logits_for_kl = log(p_ensemble)
```

Do not average raw logits directly unless an A/B proves it is equivalent enough
for the target models.

Constraints:

- All teachers must share tokenizer and vocab size for v1.
- Cross-tokenizer teachers are deferred; they require retokenization and a
  different loss contract.
- The router decision must be logged per step for attribution.

License gates:

- Router determinism: same prompt + seed chooses the same teacher.
- Single-teacher equivalence: with one teacher configured, `MultiTeacher`
  matches `TeacherForward` output within the existing BF16 gate.
- Multi-teacher A/B: specialist routing improves at least one held-out metric
  versus the strongest single teacher without regressing global held-out KL.

Kill gates:

- If routing metadata is missing or ambiguous, fail closed with an actionable
  error instead of silently choosing teacher 0.
- If ensemble mode causes non-finite logits or KL, KILL and keep router-only.

### T4 — 9B Runtime Teacher Unblock

A 9B runtime teacher remains useful, but it is a loader tranche, not an OPD
algorithm tranche.

Viable implementation axes:

1. GPTQModel physical layout loader for `qweight [K/8, N]`, scales
   `[K/group_size, N]`, optional `qzeros`, and `g_idx`.
2. Compressed-tensors FP8 loader for `.weight_scale`.
3. A new ARLE-native quantized 9B artifact with tensor-local parity before
   full-model serve.

Acceptance order:

1. Tensor-local dequant parity.
2. Layer-local matmul parity.
3. Full-model logits parity.
4. Multi-token generation coherence.
5. OPD KL trajectory.

Do not skip directly to serve smoke; previous 9B quant attempts proved that
`loads and decodes one token` is not a sufficient quality gate.

Current TODO after the 2026-05-22 DavidWen GPTQModel probe:

1. Keep the GPTQModel W4 physical-layout branch behind
   `INFER_EXPERIMENTAL_GPTQMODEL_W4=1` until quality is licensed.
2. Done: layer-local projection parity harness landed as
   `infer/examples/gptqmodel_w4_gemv_parity.rs`. Sampled W4 projections pass
   ARLE CUDA W4A16 GEMV vs faithful GPTQ reference at <=0.25% RMSE/reference-RMS.
3. Done: dense fallback scan found that the checkpoint stores
   `linear_attn.A_log` and `linear_attn.norm.weight` as BF16. ARLE's f32 1D
   loader was fixed to convert by dtype; layer-0 `linear_attention` now passes
   the BF16-realistic gate at `4.07%` RMSE/reference-RMS with no NaNs.
4. Done: full-model single-token logits after the f32-load fix pass the
   pragmatic 9B GPTQModel envelope: top-64 dominant relerr `0.124`, top-64
   RMSE/reference-RMS `0.043`, ARLE argmax `11`, PyTorch BF16 argmax `11`.
5. Killed: 9B GPTQModel -> 0.8B LoRA OPD on the old train-side f32
   student base. Both the 100-step real-prompt bench and a single-token,
   rollout-1 control fail before `eval_summary step=0` with
   `cuda htod copy failed`; live memory reached `14399 MiB / 16376 MiB`.
   Upload diagnostics narrowed the failing allocation to a small
   `[1024, 3584]` f32 projection upload (`14.68 MB`) returning
   `CUDA_ERROR_OUT_OF_MEMORY`, confirming whole-runtime memory exhaustion
   rather than a single oversized tensor.
6. Done, then killed: BF16 frozen-base substrate and Qwen3.5 LoRA loader
   selection landed, but the same-card 9B GPTQModel -> 0.8B LoRA control still
   OOMs during the first backward pass. The strictest no-eval control used
   `rollout_len=1`, `prompt_max_tokens=1`, and hit `15871 MiB used / 73 MiB
   free` immediately before `cuda alloc_zeros failed (mean_backward_device)`.
   This rules out the original f32-loader bug as the remaining blocker and
   reclassifies the issue as same-card teacher/student/tape co-residency on a
   16 GB GPU.
7. Current DX axis: add CLI / JSON config for `ApiTeacher` + `MultiTeacher` so
   `arle train opd` can select local infer, external API, or routed specialist
   teachers without editing examples.
   - Example-level surface: `opd_step_cuda_infer_teacher_train` accepts
     `--teacher-api-url` for a single full-logits API teacher and
     `--teacher-config` for deterministic token-prefix routing across multiple
     API teachers.
   - Sample config: `examples/opd/multi-teacher-config.example.json`.
   - Verified gate: `cargo test -p train --lib --release teacher_infer` passes
     `ApiTeacher` BF16/f32 decode, HTTP full-logits upload, and
     `MultiTeacher` longest-prefix routing tests.
   - Still open: wire the same config surface into `arle train opd` once the
     current CLI dirty work is reconciled.
   - Memory note: a same-GPU local API teacher does not solve the 9B OOM,
     because teacher+student still co-reside. API/multi-teacher is the right
     9B path when the teacher runs on another GPU/host or separate service
     memory budget.

Evidence:

```text
docs/experience/errors/2026-05-22-arle-qwen35-9b-gptqmodel-generation-kill.md
docs/research/2026-05-22-arle-qwen35-9b-gptqmodel-w4-gemv-parity.md
docs/experience/errors/2026-05-22-arle-qwen35-9b-gptqmodel-dense-tensor-kill.md
docs/research/2026-05-22-qwen35-9b-gptqmodel-linear-attn-f32load-fix.md
docs/research/2026-05-22-qwen35-9b-gptqmodel-full-logits-after-f32load-fix.md
docs/experience/errors/2026-05-22-qwen35-9b-gptqmodel-08b-opd-memory-kill.md
bench-output/2026-05-22-qwen35-9b-gptq-int4-loader/
bench-output/2026-05-22-qwen35-9b-gptqmodel-layerlocal/
bench-output/2026-05-22-qwen35-9b-gptqmodel-dense-parity/
bench-output/2026-05-22-qwen35-9b-gptqmodel-full-logits-after-f32load-fix/
bench-output/2026-05-22-qwen35-9b-gptqmodel-08b-opd-infer-teacher/
bench-output/2026-05-22-qwen35-9b-gptqmodel-08b-opd-infer-teacher-smoke-minimal/
bench-output/2026-05-22-qwen35-9b-gptqmodel-08b-opd-infer-teacher-bf16-student-smoke-noeval-memtrace/
```

## Implementation Order

1. Done: local 9B availability note and raw artifacts.
2. Done: `ApiTeacher` full-logits client behind `TeacherForward`.
3. Done: prompt-router `MultiTeacher`.
4. Next: add CLI / JSON config for `ApiTeacher` + `MultiTeacher` so `arle train
   opd` can select local infer, external API, or routed specialists without
   editing examples.
5. Done: add train-side BF16 frozen-base support for LoRA students.
6. Killed: same-card 9B GPTQModel -> 0.8B LoRA OPD still exceeds 16 GB even
   after BF16 frozen-base support.
7. Add top-k loss only if a real target API cannot provide full logits.
8. Return to 9B headline work only after one of these passes: remote/API
   teacher no-OOM + monotonic KL, smaller/quantized train-side student, or a
   larger-memory GPU local-teacher bench.

## Cross-Links

- 9B local availability:
  [`../research/2026-05-21-arle-qwen35-9b-local-availability.md`](../research/2026-05-21-arle-qwen35-9b-local-availability.md)
- GPTQ loader kill:
  [`../experience/errors/2026-05-21-arle-qwen35-9b-gptq-int4-loader-kill.md`](../experience/errors/2026-05-21-arle-qwen35-9b-gptq-int4-loader-kill.md)
- FP8 compressed-tensors kill:
  [`../experience/errors/2026-05-21-arle-qwen35-9b-fp8-compressed-tensors-layout-kill.md`](../experience/errors/2026-05-21-arle-qwen35-9b-fp8-compressed-tensors-layout-kill.md)
- GPTQModel W4 layer-local parity:
  [`../research/2026-05-22-arle-qwen35-9b-gptqmodel-w4-gemv-parity.md`](../research/2026-05-22-arle-qwen35-9b-gptqmodel-w4-gemv-parity.md)
- GPTQModel dense tensor kill:
  [`../experience/errors/2026-05-22-arle-qwen35-9b-gptqmodel-dense-tensor-kill.md`](../experience/errors/2026-05-22-arle-qwen35-9b-gptqmodel-dense-tensor-kill.md)
- GPTQModel linear-attention f32-load fix:
  [`../research/2026-05-22-qwen35-9b-gptqmodel-linear-attn-f32load-fix.md`](../research/2026-05-22-qwen35-9b-gptqmodel-linear-attn-f32load-fix.md)
- GPTQModel full-logits gate after f32-load fix:
  [`../research/2026-05-22-qwen35-9b-gptqmodel-full-logits-after-f32load-fix.md`](../research/2026-05-22-qwen35-9b-gptqmodel-full-logits-after-f32load-fix.md)
- GPTQModel 9B -> 0.8B OPD memory kill:
  [`../experience/errors/2026-05-22-qwen35-9b-gptqmodel-08b-opd-memory-kill.md`](../experience/errors/2026-05-22-qwen35-9b-gptqmodel-08b-opd-memory-kill.md)
- Frozen-BF16 LoRA student Step 0 audit:
  [`../research/2026-05-22-opd-frozen-bf16-lora-student-step0-audit.md`](../research/2026-05-22-opd-frozen-bf16-lora-student-step0-audit.md)
- Infer-teacher adapter:
  [`../../crates/train/src/teacher_infer.rs`](../../crates/train/src/teacher_infer.rs)
