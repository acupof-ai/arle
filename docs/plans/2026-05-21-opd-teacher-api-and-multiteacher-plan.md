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

That means ARLE can use two local teacher classes today:

1. A train-side frozen `Qwen35Model`
2. A local `infer` runtime teacher through `InferTeacher`

It does **not** yet support an arbitrary external HTTP/OpenAI-style API as the
OPD teacher.

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

Latest local inventory:

| Checkpoint | Local status | Runtime status |
| --- | --- | --- |
| `Qwen/Qwen3.5-9B` BF16 | Complete, 18.0 GiB files | Fails ARLE CUDA serve on 16 GB: H2D OOM before readiness |
| `Qwen/Qwen3.5-9B-Instruct` | Directory exists, no weights | Not usable |
| `DavidWen2025/Qwen3.5-9B-GPTQ-4bit` | Complete, 10.35 GiB safetensors | Experimental GPTQModel W4 loader reaches HTTP readiness behind `INFER_EXPERIMENTAL_GPTQMODEL_W4=1`; generation quality gate fails with repeated `!`, so not OPD-licensed |
| `RedHatAI/Qwen3.5-9B-FP8-dynamic` | Metadata only | Loader blocked: compressed-tensors `.weight_scale` unsupported |

Raw evidence:

```text
bench-output/2026-05-21-qwen35-9b-local-availability/
```

Research note:

```text
docs/research/2026-05-21-arle-qwen35-9b-local-availability.md
```

## Work Queue

### T1 — External Full-Logits Teacher API

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
  "vocab_size": usize,
  "logits_b64": "..."
}
```

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

### T3 — Multi-Teacher Router

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
3. Next: projection parity passed, so scan dense fallback modules under the same
   checkpoint: embedding, linear-attention dense tensors, final norm, and
   untied lm_head.
4. Re-run multi-token generation before any OPD bench or headline switch.

Evidence:

```text
docs/experience/errors/2026-05-22-arle-qwen35-9b-gptqmodel-generation-kill.md
docs/research/2026-05-22-arle-qwen35-9b-gptqmodel-w4-gemv-parity.md
bench-output/2026-05-22-qwen35-9b-gptq-int4-loader/
bench-output/2026-05-22-qwen35-9b-gptqmodel-layerlocal/
```

## Implementation Order

1. Land the local 9B availability note and raw artifacts.
2. Add `ApiTeacher` full-logits client behind `TeacherForward`.
3. Add prompt-router `MultiTeacher`.
4. Add top-k loss only if a real target API cannot provide full logits.
5. Return to 9B loader work after the teacher abstraction is stable.

## Cross-Links

- 9B local availability:
  [`../research/2026-05-21-arle-qwen35-9b-local-availability.md`](../research/2026-05-21-arle-qwen35-9b-local-availability.md)
- GPTQ loader kill:
  [`../experience/errors/2026-05-21-arle-qwen35-9b-gptq-int4-loader-kill.md`](../experience/errors/2026-05-21-arle-qwen35-9b-gptq-int4-loader-kill.md)
- FP8 compressed-tensors kill:
  [`../experience/errors/2026-05-21-arle-qwen35-9b-fp8-compressed-tensors-layout-kill.md`](../experience/errors/2026-05-21-arle-qwen35-9b-fp8-compressed-tensors-layout-kill.md)
- GPTQModel W4 layer-local parity:
  [`../research/2026-05-22-arle-qwen35-9b-gptqmodel-w4-gemv-parity.md`](../research/2026-05-22-arle-qwen35-9b-gptqmodel-w4-gemv-parity.md)
- Infer-teacher adapter:
  [`../../crates/train/src/teacher_infer.rs`](../../crates/train/src/teacher_infer.rs)
