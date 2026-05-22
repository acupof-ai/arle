# 2026-05-22 — ARLE OPD capability eval plan

> **Status:** plan / coordination doc. Defines the first end-to-end
> "ARLE OPD actually improves student capability" data point: baseline
> eval → long-horizon distillation → post-distill eval, all on the
> 4B teacher → 0.8B LoRA student pipeline validated by the
> [2026-05-22 BF16 cycle wrap](../projects/2026-05-22-opd-9b-teacher-bf16-cycle-wrap.md).
> Builds on the recommendations section of the
> [OPD methodology research note](../research/2026-05-22-opd-methodology-and-industry-best-practices.md).

## Why this plan

The 2026-05-22 BF16 cycle proved:

- ARLE OPD pipeline runs on 16 GB (4B teacher → 0.8B LoRA student, 5.44 s/step)
- KL trajectory decreases monotonically (-2% held-out KL @ step 200)
- Memory + perf are real wins

**What is still unproven**: whether KL decrease translates to actual
*capability* improvement on user-visible tasks. As the research note
states: "**KL alone is not capability**". Without a capability eval
triplet (HF base / our distilled / teacher), we can't claim ARLE OPD
makes anything better — only that the pipeline runs and KL goes down.

This plan executes the smallest credible experiment that produces a
real capability-delta number.

## Goal

Produce a defensible answer to: "Does ARLE OPD at the 4B → 0.8B shape
move Qwen3.5-0.8B-Base by a measurable amount on any standard
capability benchmark, within reasonable compute budget on 16 GB?"

PASS criterion (the smallest meaningful claim): **distilled student
shows ≥ +1 percentage point on at least one of MMLU-5-shot accuracy,
IFEval strict-followed-pct, or GSM8K exact-match accuracy** vs the HF base
checkpoint, with the teacher serving as upper bound.

KILL criteria (any of):

- 0 ± noise delta across all three tasks after ≥ 5k training steps
- Distilled student *regresses* on any task by ≥ 2 percentage points (over-distillation, mode collapse)
- Training run cannot reach 5k steps in 16 h compute budget

Both PASS and KILL are wins for the project — PASS validates the
runtime, KILL identifies the loss function / schedule / shape issue
that needs fixing.

## Approach

ARLE's `infer` HTTP server exposes the OpenAI v1 surface
(`/v1/completions`, `/v1/chat/completions`, `logprobs` parameter — see
`infer/src/http_server/router.rs`). The eval harness can talk to it as
if it were the OpenAI API.

**ARLE HTTP surface audit (2026-05-22)**:

| Field | Status | Impacted tasks |
|---|---|---|
| `/v1/completions`, `/v1/chat/completions` | ✅ | All generation-based eval |
| `logprobs` (top-N) | ✅ | Calibration tasks |
| `token_logprobs` in response | ✅ | Per-token analysis |
| `temperature`, `max_tokens`, `top_p` | ✅ | Sampling control |
| **`echo`** (return prompt logprobs) | ❌ | HellaSwag (standard scoring), Lambada |
| Batched completions | Check before P0 | Throughput-bound tasks |

For tasks that need `echo` (HellaSwag-style logprob-of-candidate-sequence
scoring), use a generation-based fallback ("Which is most plausible? A/B/C/D")
for P0, and add `echo` support to ARLE as a separate small tranche if the
signal-quality difference matters.

Recommended task subset (revised): **MMLU + IFEval + GSM8K** at 500
samples each. All three are generation-based, work with current ARLE.
HellaSwag deferred to a v2 cycle that lands `echo`.

Eval harness choice: **simple-evals** (OpenAI, MIT license, ~2k LOC
Python).

Why simple-evals over alternatives:

| Harness | Pros | Cons | Verdict |
|---|---|---|---|
| **simple-evals** | 2k LOC, MMLU/IFEval/MATH/HumanEval/GPQA, talks to OpenAI API directly | Smaller task coverage | **Pick this** |
| lm-evaluation-harness (EleutherAI) | 100+ tasks, widely cited | 50k LOC, heavy deps, slower to wire | Overkill for first cut |
| ARLE-native Rust harness | Maximum integration | Significant new code | Defer to v2 |
| TRL evaluate / lighteval | Decent task set | Tied to HF model loading; we need API mode | No |

## Phases

### P0 — Baseline (no training, ~1 h compute)

1. Spin up `arle serve` with Qwen3.5-0.8B-Base.
2. Install simple-evals, point `OPENAI_BASE_URL=http://localhost:8123/v1`.
3. Run MMLU-5-shot (subset 500 questions), IFEval (500 prompts), GSM8K (500 problems). Total ~1500 prompts.
4. Record per-task accuracy + total wall-clock.
5. Re-run same three tasks against Qwen3.5-4B (teacher) for upper-bound reference.
6. Land wins entry: `docs/experience/wins/2026-05-22-baseline-08b-4b-capability-eval.md` with the two-row table.

**Acceptance**: harness wires up cleanly, two-row baseline table exists, no crashes.

### P1 — Distillation run (~16 h compute)

1. Patch `crates/train/examples/opd_step_cuda_infer_teacher_train.rs` to add `--save-student-checkpoint <dir>` flag (saves LoRA adapter + base merge every `--save-every` steps).
2. Run: 4B teacher → 0.8B LoRA student, 10k steps, rollout=8 or 16 (whichever passed Stretch), prompt_max=16 or 32, lr=2e-5 (slightly higher than current 1e-5 for capability transfer), warmup 500 steps, cosine decay, save checkpoint every 2k steps.
3. Log KL trajectory + step time + nvidia-smi monitor as usual.
4. Land wins entry: `docs/experience/wins/2026-05-22-10k-step-distill-run.md` with KL curve, peak mem, step time, checkpoint paths.

**Acceptance**: 10k steps complete without OOM/NaN; ≥ 5 checkpoints saved (steps 2k, 4k, 6k, 8k, 10k).

### P2 — Post-distill eval (~5 h compute)

1. For each saved checkpoint (5 total), load with `arle serve` and run the same 3 tasks from P0.
2. Build a 6-row table: HF base / step 2k / step 4k / step 6k / step 8k / step 10k / teacher.
3. Plot accuracy vs step for each task. Plot KL-delta vs accuracy-delta scatter.
4. Land wins entry: `docs/experience/wins/2026-05-22-distill-capability-delta.md` (PASS) or `docs/experience/errors/2026-05-22-distill-capability-flat-kill.md` (KILL).

**Acceptance**: 6×3 = 18 accuracy numbers in the report; PASS/KILL verdict explicit.

## Compute budget

| Phase | Wall-clock estimate (RTX 4070 Ti SUPER 16 GB) |
|---|---|
| P0 baseline | 1 h (0.8B + 4B at 2000 prompts each, batched) |
| P1 10k-step distill | ~15 h (5.44 s/step × 10000 = 15.1 h at rollout=8; ~30 h at rollout=16) |
| P2 5-checkpoint eval | 5 h (5 × 1 h per checkpoint) |
| **Total** | **~22 h at rollout=8, ~36 h at rollout=16** |

Decision rule: **use rollout=8 unless Stretch A shows meaningful KL improvement at rollout=16**. The 2× wall-clock cost is hard to justify without evidence.

## Codex execution plan (when this plan is licensed)

### Phase 0 brief (smallest, runs first)

```
== Goal ==
P0 — baseline capability eval on Qwen3.5-0.8B-Base + Qwen3.5-4B
via arle serve + simple-evals.

== Steps ==
1. pip install simple-evals (or git clone openai/simple-evals if not on PyPI)
2. arle serve --model Qwen3___5-0___8B-Base --port 8123
3. OPENAI_BASE_URL=http://localhost:8123/v1 python -m simple_evals \
   --models qwen35-08b-base --tasks mmlu,ifeval,gsm8k --n_samples 500
4. Repeat with Qwen3___5-4B on port 8123 (teacher)
5. Record results in bench-output/2026-05-22-baseline-capability/
6. Write wins entry with the two-row table

== License-or-kill ==
PASS: 6 accuracy numbers (2 models × 3 tasks) recorded with proper sample counts.
KILL: any task can't run (simple-evals API mismatch, ARLE missing OpenAI feature). Document gap in errors entry.

== Constraints ==
- Don't touch parallel dirty files
- Standard gate discipline
```

### Phase 1 + 2 briefs: drafted after P0 lands, contingent on P0 PASS.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| simple-evals expects HTTP features ARLE doesn't have | P0 acceptance is "harness wires up"; if it fails, errors entry documents which feature gap; we patch ARLE or fall back to lm-evaluation-harness |
| 10k steps insufficient to move accuracy | Stop early at 5k for first data point; literature suggests visible deltas at 5k-20k for narrow tasks |
| Eval too slow on 0.8B model via ARLE serve | Increase batch size on the eval side; if still slow, reduce per-task sample count to 200 |
| Checkpoint save bloat (5 × ~3 GB = 15 GB disk) | Save LoRA adapter only (not merged), keep base reference; ~50 MB per ckpt |
| Compute budget overrun | Cap each phase with `timeout`; report partial results if hit |

## Cross-links

- Methodology research note: [`docs/research/2026-05-22-opd-methodology-and-industry-best-practices.md`](../research/2026-05-22-opd-methodology-and-industry-best-practices.md)
- BF16 cycle wrap: [`docs/projects/2026-05-22-opd-9b-teacher-bf16-cycle-wrap.md`](../projects/2026-05-22-opd-9b-teacher-bf16-cycle-wrap.md)
- BF16 4B OPD wins: [`docs/experience/wins/2026-05-22-bf16-substrate-4b-opd-memory-savings.md`](../experience/wins/2026-05-22-bf16-substrate-4b-opd-memory-savings.md)
- Original 4B-teacher baseline: [`docs/experience/wins/2026-05-21-qwen35-4b-08b-opd-infer-teacher.md`](../experience/wins/2026-05-21-qwen35-4b-08b-opd-infer-teacher.md)
- ARLE OPD usage manual: [`docs/projects/2026-05-21-arle-opd-cuda-usage-manual.md`](../projects/2026-05-21-arle-opd-cuda-usage-manual.md)

## Open decisions for user

Before P0 briefs to codex:

1. **Confirm simple-evals as the harness** (vs lm-eval / native).
2. **Confirm task subset** (MMLU + IFEval + GSM8K at 500 samples each, vs full sets or swapping in HumanEval / MATH). HellaSwag deferred until ARLE adds `echo` support.
3. **Confirm rollout choice for P1** — wait for Stretch A/B results before committing to rollout=8 vs 16.
4. **Compute budget tolerance** — 22 h vs 36 h vs cap at 5k steps for fast first data point.
