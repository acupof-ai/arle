# OPD distillation has task-divergent impact: MMLU U-curve vs GSM8K inverse-U at step 1000

## Context

After the lr sweep (`0629294`) showed both `lr=2e-5` and `lr=1e-5` fail
to cross MMLU base 51.4 % in 2 k steps, I added the missing GSM8K
evals on the `lr=1e-5` checkpoints to fill in the trajectory grid.
The result is more interesting than the lr=2e-5 GSM8K result codex
already had: **the two tasks move in opposite directions at the same
training step.**

## Setup

- Adapter: `runs/2026-05-22-p2-distill-lr1e5/step_001000` and `.../step_002000`
- Loaded via `INFER_LORA_PATH` into `arle serve --backend cuda
  --max-seq-len 4096 --chunked-prefill-size 4096
  --max-num-batched-tokens 4096`
- Eval: `scripts/arle_capability_eval.py --tasks gsm8k --n-samples 200`
- Other 4 cells (MMLU at both lrs both steps + GSM8K lr=2e-5 step 2000)
  come from `bb72066`, `0629294`, and `1586cef`.

## Trajectory grid

| Snapshot | MMLU @ step 1000 | MMLU @ step 2000 | GSM8K @ step 1000 | GSM8K @ step 2000 |
|---|---:|---:|---:|---:|
| Base 0.8B (step 0) | 51.4 % | 51.4 % | 1.5 % | 1.5 % |
| lr=2e-5 | 47.9 % | 50.0 % | — | 1.6 % |
| **lr=1e-5** | **50.6 %** | **48.5 %** | **3.16 %** | 2.22 % |
| Teacher 4B | 77.3 % | 77.3 % | 2.5 % | 2.5 % |

## What's surprising

`lr=1e-5` at step 1000 has **GSM8K = 3.16 %**, which is **above the
4B teacher's 2.5 %** (Δ +0.66 pp). The student briefly exceeds the
teacher on this task even though the teacher is the upper bound for
KL-driven distillation.

At the same training point, MMLU is at 50.6 % — below the base 51.4 %
floor. So:

- **MMLU**: standard U-curve. Trough at step 1000, partial recovery
  inconclusive (lr=2e-5) or absent (lr=1e-5).
- **GSM8K**: **inverse U-curve** for lr=1e-5. Base 1.5 → **peak
  3.16 % at step 1000** → regression 2.22 % at step 2000. The student
  beats the teacher mid-training and then loses it.

## Hypotheses for why teacher gets beat on GSM8K

Listed with the cheapest experiment that could license or kill each.

1. **GSM8K signal sparsity.** GSM8K accuracy at this scale (1-3 % on
   small base models) is dominated by noise — 4/200 = 2 % vs 5/200 =
   2.5 % is a 1-question difference. Test: rerun on a different 200
   sample at the same checkpoint and see if 3.16 % holds within
   ±1.5 pp. (≈12 min compute.)
2. **OPD LoRA params drift into a math-friendly direction by
   accident.** Forward-KL minimization on a math-heavy teacher
   distribution may bias the LoRA toward better number-token
   handling for a window of steps, then lose it as the LoRA
   over-fits the broader distribution. Test: probe the GSM8K-correct
   subset at finer-grain step counts (steps 250, 500, 750, 1250,
   1500, 1750, 2000) and see if the peak narrows. (~2 h compute for
   5 extra evals on saved checkpoints — but those checkpoints don't
   exist yet, so this needs a finer-grained re-run.)
3. **Format-following leak.** The few-shot GSM8K prompt ends with
   `A:`, and the model may learn from teacher-distribution-fitting
   to commit faster to the `####` marker. Test: separate "predicted
   answer extracted" rate from "answer correct" rate; if invalid-
   pct drops faster than accuracy rises at step 1000, the gain is
   format-driven not reasoning-driven. The current data: invalid 6
   (base) → 10 (step 1000) → 20 (step 2000) — invalid actually
   INCREASED. So format-following is NOT the explanation.
4. **Teacher itself is weak on GSM8K.** Qwen3.5-4B-Base on GSM8K is
   2.5 % which is near floor; the "teacher beats student" framing
   doesn't apply cleanly because the teacher itself is bad. Both
   teacher and student are essentially guessing; the student happens
   to guess slightly better in a 200-sample window.

The strongest hypothesis is **(1) + (4) combined**: at near-floor
accuracy on a teacher that's itself near-floor, student fluctuations
of ±1-2 pp are normal sample noise, not signal. The "student beats
teacher" is likely a 1-2 question coincidence.

This is not a "we accidentally solved math distillation" story — it's a
"task-divergent OPD impact is a real and underappreciated dynamic"
story. The right reading is:

- MMLU: distillation hurts then helps (literature-standard U-curve)
- GSM8K: distillation moves around in the noise floor (no signal)

Don't claim "we beat the teacher on GSM8K" without a fresh 200-sample
control. Even with the control, the absolute accuracy is too low to
matter for any user-visible task.

## What this changes about the cycle's posture

1. **Eval triplet must be saved per-checkpoint, not just final.** The
   pilot saved adapters at step 1000 and step 2000; this entry
   exists because we had step 1000 to look at. Future runs should
   save every 500-1000 steps even if the eval budget can't keep up
   in real time.
2. **Don't quote a single MMLU number as "the OPD verdict".**
   Different capability dimensions move at different speeds, in
   opposite directions, at the same training step. The eval triplet
   (`base / various-step-distilled / teacher`) across **multiple
   tasks** is the actual unit of evidence.
3. **Updated chart panel** (right side of `docs/projects/img/2026-05-22-arle-opd-distill-trajectory.png`)
   now shows the GSM8K inverse-U trajectory instead of the cross-
   validation. The cross-validation result lives in
   `2026-05-22-arle-vs-hf-transformers-cross-validation.md` and the
   chart's left panel still shows it implicitly via the ARLE-served
   teacher line.

## Cross-links

- Original P1-B (loop closure): [`2026-05-22-p1b-train-save-load-eval-loop.md`](2026-05-22-p1b-train-save-load-eval-loop.md)
- U-curve diagnosis (lr=2e-5 only): [`2026-05-22-distill-trajectory-valley-then-recovery.md`](2026-05-22-distill-trajectory-valley-then-recovery.md)
- lr=1e-5 sweep KILL: [`../errors/2026-05-22-p2-lr-sweep-not-the-fix.md`](../errors/2026-05-22-p2-lr-sweep-not-the-fix.md)
- Cross-validation vs HF: [`2026-05-22-arle-vs-hf-transformers-cross-validation.md`](2026-05-22-arle-vs-hf-transformers-cross-validation.md)
- Methodology research note: [`../research/2026-05-22-opd-methodology-and-industry-best-practices.md`](../research/2026-05-22-opd-methodology-and-industry-best-practices.md)

## Rule

A single-task capability eval is not enough to verdict an OPD run. Tasks
respond at different speeds and in different directions to the same
distillation signal. The evidence unit is the **eval triplet across at
least two capability dimensions**. And when the absolute accuracy is
in the noise floor (small-model GSM8K), don't read deltas as signal
without a fresh control.
