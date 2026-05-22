# 2026-05-22 — OPD methodology, industry best practices, and ARLE pipeline

> **Status:** research note. Synthesizes On-Policy Distillation (OPD) industry
> references + ARLE-specific pipeline state + capability taxonomy + how-to
> methodology ("posture"). Cross-links the active ARLE OPD plans and the
> 2026-05-22 BF16 cycle wrap.

## TL;DR

- **OPD is "teacher provides logits at each token the *student* generated"**, not "teacher generates and student imitates". The on-policy part is what distinguishes it from static knowledge distillation (KD).
- **Industry standard is GKD** (Generalized Knowledge Distillation, Agarwal et al. 2024) which interpolates between SFT and pure on-policy via a `lambda` knob. Forward-KL is the default loss.
- **OPD works best for knowledge-heavy, distribution-close tasks** (factuality, instruction following, structured output). It works poorly for tasks requiring novel reasoning beyond the teacher, long-form creativity, or capabilities the teacher itself lacks.
- **The right "posture" for ARLE**: 4-10× teacher/student ratio, on-policy student rollouts, forward-KL with T=2-3, ~10k–100k steps, eval on capability harness (MMLU/IFEval) — not just KL. Current ARLE pipeline shape (`4B → 0.8B, KL loss, BF16 LoRA student`) is positioned correctly; the missing piece is a capability-eval baseline + a long-horizon run.

## 1. What OPD actually is

| Variant | Data source | Teacher's role |
|---|---|---|
| **SFT** | Pre-recorded teacher outputs (golden text) | Provides the ground-truth sequence |
| **Static KD** | Pre-recorded teacher outputs | Provides ground-truth + soft logits |
| **OPD (this project's term)** | **Student's own rollouts** (on-policy sampling) | Provides logits **conditioned on the student's generated prefix** |
| **GKD** | Mix of teacher samples + student rollouts via `lambda` | Same as OPD when `lambda=1.0`, same as static KD when `lambda=0.0` |
| **RLHF / DPO** | Student rollouts | Preference signal, not logits |

The key insight (Agarwal et al. 2024, "Generalized Knowledge Distillation"):
**static KD suffers from train/inference distribution mismatch** — student is trained on teacher's sequence distribution but deployed on its own. OPD eliminates this by always conditioning the teacher on what the *student* would actually generate.

ARLE's `train::OpdStep` implements **pure OPD** (`lambda=1.0`): every prefix
fed to the teacher is a student rollout. There is no SFT mixing knob yet.

## 2. Industry references

### Foundational papers

| Paper | Year | Contribution |
|---|---|---|
| "Distilling the Knowledge in a Neural Network" (Hinton et al.) | 2015 | Original KD: soft logits, temperature scaling |
| "DistilBERT" (Sanh et al.) | 2019 | First widely-adopted LM distillation (encoder) |
| "On-Policy Distillation of Language Models" (Lin et al.) | 2023 | Established on-policy as key for autoregressive LMs |
| "Generalized Knowledge Distillation" (GKD, Agarwal et al.) | 2024 | SFT↔OPD interpolation, forward/reverse KL analysis |
| "MiniLLM" (Gu et al., Tsinghua) | 2024 | Reverse-KL for mode-seeking, exposure-bias analysis |
| "Distilling Step-by-Step" (Hsieh et al., CMU+Google) | 2023 | Rationale distillation (teacher's chain-of-thought as auxiliary target) |

### Production frameworks (in priority order for ARLE comparison)

| Framework | Stack | Notes |
|---|---|---|
| **HuggingFace TRL `GKDTrainer`** | PyTorch + HF Transformers | Most accessible reference impl. ARLE's `2026-05-21-arle-vs-trl-gkd-head-to-head.md` wins entry shows ARLE beats TRL 2.04× at matched setup. |
| **vLLM + verl** | vLLM teacher + verl trainer | Industrial-scale. Teacher serves via vLLM API, verl runs student SFT/PPO/distill. This is the "remote ApiTeacher" pattern ARLE is now positioning toward. |
| **NVIDIA NeMo-Aligner / NeMo Distill** | NeMo + Megatron-LM | Megatron-scale (≥7B teachers). Static KD primarily; on-policy is an add-on. |
| **NVIDIA DistillKit** | NeMo | Toolkit-oriented; reference for capability eval integration. |
| **DeepSpeed-Chat** | DeepSpeed | Has a distillation phase between SFT and RLHF. Less on-policy than verl. |

### Real-world "made by distillation" models (public)

- **Anthropic Claude Haiku 4.5** — model card explicitly cites distillation from Claude Sonnet/Opus family.
- **OpenAI GPT-4 Turbo / GPT-4o Mini** — public statements imply distillation-style compression.
- **DeepSeek V2 / V3 Lite variants** — MoE → dense distillation.
- **Google Gemini Flash** — public Gemini paper section on Flash describes distillation from Pro.
- **Meta Llama 3.x 8B / 70B teachers → smaller students** in OSS community (e.g., NeuralChat, Hermes-Llama-Mini).

Pattern: every major lab now ships distilled "small model" variants. ARLE's
positioning as a Rust-native runtime + OPD train integration aims to make
this pipeline reproducible by single individuals on consumer GPUs.

## 3. ARLE OPD pipeline today (concrete)

```
┌─────────────────────────────────────────────────────────────┐
│ infer runtime (LoadedInferenceEngine)                        │
│   - Loads teacher (Qwen3.5-4B BF16 today)                    │
│   - forward_token_logits(prefix) → BF16 logits               │
└─────────────────────────────────────────────────────────────┘
                          ▲
                  HTTP (ApiTeacher) or
                  D2D bridge (InferTeacher, current default)
                          │
┌─────────────────────────────────────────────────────────────┐
│ train::OpdStep                                               │
│   1. student greedy/sampling rollout (rollout_len tokens)    │
│   2. tokenize student output                                 │
│   3. call teacher on (prompt + student_rollout)              │
│   4. forward-KL loss between student logits and teacher      │
│   5. backward + LoRA adapter update                          │
│   6. AdamW step                                              │
└─────────────────────────────────────────────────────────────┘
```

**Current shape**: 4B BF16 teacher → 0.8B BF16 base + LoRA student
(rank 16, alpha 32, attention-qv target).
**Hardware fit**: 16 GB GPU after 2026-05-22 BF16 frozen-base substrate
landed (peak 13.4 GB at rollout=8).
**Speed**: 5.44 s/step mean (200 steps in 1111 s).
**KL decrease**: -2% held-out KL over 200 steps (directional, not
convergence).

**Missing pieces for capability claim**:
1. **Checkpoint save** during training (current bench discards student weights at end).
2. **Capability eval harness** (lm-evaluation-harness or simple-evals integration).
3. **Long horizon run** (200 steps is not nearly enough for capability transfer; literature suggests 10k–100k steps).
4. **Baseline accuracy** of `Qwen3.5-0.8B-Base` measured on the same harness, to give us a delta to claim.
5. **SFT-warmup option** (GKD `lambda < 1.0`) — pure OPD on a base model often struggles in early steps; literature recommends SFT warmup or lambda-scheduling.

## 4. Capability taxonomy — what OPD does well / poorly

### Strong fit (high ROI from OPD)

| Capability | Why OPD helps |
|---|---|
| **Factual knowledge / closed-book QA** (MMLU, TriviaQA, NaturalQuestions) | Knowledge lives in weights; KL transfer compresses it effectively. |
| **Instruction following** (IFEval, simple chat) | Teacher provides aligned distribution; student inherits alignment shape. |
| **Structured output** (JSON, function calling) | Distribution over format tokens is sharp; KL captures it well. |
| **Short-form summarization / paraphrase** | Teacher's distribution is concentrated; student learns it stably. |
| **Domain-specialized models** (medical, legal, code) | Use a domain-finetuned teacher; OPD compresses it into a smaller deployable. |
| **Calibration / refusal behavior** | Teacher's "I don't know" or refusal patterns transfer cleanly via KL. |

### Weak fit (OPD helps marginally or not at all)

| Capability | Why OPD struggles |
|---|---|
| **Long-form generation** (creative writing, long essays) | Mode collapse: KL with T=2-3 favors high-probability paths; student becomes repetitive. Use rejection sampling or RLHF instead. |
| **Multi-turn dialog with persona** | Persona / context dependency is hard to capture with token-level KL. |
| **Mathematical reasoning beyond teacher** | Student can't exceed teacher's accuracy; if teacher is 60% on GSM8K, student caps at ~60%. |
| **Novel code generation** | Same upper bound as math. Plus: code requires exact correctness; KL doesn't optimize for it. |
| **Robustness / adversarial** | Teacher's mistakes transfer too. |

### Poor fit (use a different technique)

| Goal | Use instead |
|---|---|
| **Student needs to be *safer* than teacher** | RLHF or Constitutional AI, not distillation. |
| **Student needs to explore / discover new strategies** | RL (PPO, GRPO), not distillation. |
| **Compress a frontier model down 100×** | Quantization + pruning first, then maybe distillation as polish. Pure distillation from 100× larger teacher tends to collapse. |
| **Cross-architecture transfer** (e.g., Transformer → Mamba) | Distillation works but is much harder; usually needs feature-level alignment (DistillKit-style) rather than just KL. |

## 5. The methodology / posture (how to do it right)

### Setup

1. **Teacher/student size ratio: 4×–10×**. Bigger ratio → harder transfer (more capability gap). Smaller → less savings. ARLE's 4B/0.8B = 5× is in the sweet spot.
2. **Teacher quality matters more than size**. A 4B aligned model is better than a 13B unaligned model for OPD purposes.
3. **Same tokenizer**. Cross-tokenizer distillation needs alignment tricks (token-level matching, segment-level) and adds noise.
4. **Same architecture family** preferred (Llama-style → Llama-style, Qwen → Qwen). Cross-arch is possible but harder.
5. **Student starts from a pre-trained checkpoint** (base model), not random init. LoRA on top is fine for capability transfer up to ~5% accuracy delta; for >5% delta, full-finetune the student.

### Loss and sampling

6. **Forward KL (teacher → student) is the default**. Mode-covering: student learns the full distribution. Use this if you want a general-purpose student.
7. **Reverse KL (student → teacher)** is mode-seeking: student picks a "tall" mode of the teacher. Use this for task-specialized students where you want consistency over coverage. MiniLLM uses this.
8. **Temperature T=2.0–3.0** for softening logits. Higher T = more transfer of "dark knowledge" (relative ordering of low-probability tokens). T=1.0 is barely different from cross-entropy.
9. **On-policy rollouts always**. Pre-recorded teacher data → static KD, which underperforms OPD by 2-5% on capability tasks (Agarwal et al. 2024).
10. **Rollout length: 8–32 tokens typical**. Longer = more on-policy signal, but slower and more memory. ARLE's `rollout_len=8` is conservative; the in-flight `rollout_len=16` stretch will tell us if doubling helps.

### Training schedule

11. **Steps: 10k–100k for capability transfer**. 200 steps (current ARLE bench) is enough to validate the pipeline; **not enough to claim capability**.
12. **Learning rate: 1e-5 to 5e-5** for LoRA adapters; 5e-6 to 1e-5 for full-finetune.
13. **Warmup**: 100–500 steps linear warmup to peak LR, then cosine decay.
14. **Eval every N steps** on a small held-out capability set + on KL.
15. **GKD `lambda` scheduling** (optional): start with `lambda=0.3` (more SFT-like), anneal to `lambda=1.0` (pure OPD) over warmup. Helps stability when student is far from teacher initially.

### Evaluation

16. **KL alone is not capability**. KL going down means student matches teacher distribution; it does NOT prove student is better at any user-visible task. **You MUST run a capability harness** (MMLU, IFEval, HellaSwag, GSM8K, HumanEval-style).
17. **Baseline triplet**: HF base / your distilled / teacher. Without all three, you can't separate "OPD didn't help" from "OPD broke the model".
18. **Eval shape matters**: 5-shot MMLU vs 0-shot, with-CoT vs without — fix one shape and stick with it across runs.

### Cost expectations (consumer hardware reference)

- 4B teacher + 0.8B student on RTX 4070 Ti SUPER (16 GB):
  - 5.44 s/step (rollout=8, prompt_max=16)
  - 10k steps ≈ 15 hours
  - 100k steps ≈ 6 days
- 7B teacher + 1.5B student: needs ≥24 GB (RTX 3090 or 4090 24GB).
- 13B+ teacher: needs A100/H100 or remote `ApiTeacher` path.

## 6. Recommended next steps for ARLE (in priority order)

1. **Add `--save-student-checkpoint <path>` to `opd_step_cuda_infer_teacher_train`** so we can keep trained weights for eval. (Small CLI patch.)
2. **Pick + integrate an eval harness**. Options:
   - **lm-evaluation-harness** (EleutherAI): widest task coverage, Python-only, runs against any HF model directory. Heaviest dep.
   - **simple-evals** (OpenAI): minimal, MMLU/IFEval/MATH/HumanEval/GPQA. Lightest dep. Recommended for ARLE.
   - **ARLE-native eval**: write a thin Rust harness that calls `infer::serve` and scores against MMLU/IFEval prompts. Maximum integration, most work.
3. **Measure baseline** `Qwen3.5-0.8B-Base` on chosen harness. This is the floor we have to beat.
4. **Long-horizon distillation run**: 10k steps, save checkpoint every 1k, eval on harness every 2k. Use Stretch shape (`rollout=16, prompt_max=16`) if BF16 cycle's stretch passes.
5. **Plot KL vs capability**: per-checkpoint, scatter KL-decrease vs accuracy-delta. This is the "is OPD actually working?" smoking gun. If KL goes down but accuracy stays flat → KL is decoupled from capability, change loss or teacher.
6. **Compare to TRL**: with the same teacher/student/shape, run TRL `GKDTrainer` for 10k steps too. ARLE wins entry already shows 2× perf advantage; now show **same or better capability** at that speed.

## 7. Open research questions for ARLE specifically

- **Should we mix in SFT?** (GKD `lambda < 1.0`) Current pure-OPD is the harder learning task; adding SFT mix might be necessary for base-model students.
- **Reverse KL vs forward KL** — should we expose this as a config flag?
- **Rationale distillation** — for reasoning tasks (GSM8K, MATH), should we capture teacher's CoT and add as auxiliary loss?
- **Quantization-aware distillation** — can we distill *into* a quantized student directly, so the deployment target matches the training target?
- **MoE student** — can the OPD signal stably train a MoE student? (Most current OPD literature is dense → dense.)

## Cross-links

- ARLE pipeline: [`docs/projects/2026-05-21-arle-opd-cuda-usage-manual.md`](../projects/2026-05-21-arle-opd-cuda-usage-manual.md)
- Industry positioning: [`docs/projects/2026-05-21-opd-industry-positioning-best-framework.md`](../projects/2026-05-21-opd-industry-positioning-best-framework.md)
- 2026-05-22 BF16 cycle wrap: [`docs/projects/2026-05-22-opd-9b-teacher-bf16-cycle-wrap.md`](../projects/2026-05-22-opd-9b-teacher-bf16-cycle-wrap.md)
- TRL head-to-head: [`docs/experience/wins/2026-05-21-arle-vs-trl-gkd-head-to-head.md`](../experience/wins/2026-05-21-arle-vs-trl-gkd-head-to-head.md)
- 4B-teacher baseline OPD: [`docs/experience/wins/2026-05-21-qwen35-4b-08b-opd-infer-teacher.md`](../experience/wins/2026-05-21-qwen35-4b-08b-opd-infer-teacher.md)
- BF16 substrate memory savings: [`docs/experience/wins/2026-05-22-bf16-substrate-4b-opd-memory-savings.md`](../experience/wins/2026-05-22-bf16-substrate-4b-opd-memory-savings.md)
