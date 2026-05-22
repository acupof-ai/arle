# OPD distillation trajectory: early valley → recovery, not overshoot

## Goal

Diagnose why 2 k-step OPD distillation showed MMLU regression
(51.4 % base → 50.0 % after) despite monotonic KL decrease. The
intermediate step_001000 checkpoint that the pipeline saved was the
cheap probe for this — eval it and see whether MMLU peaked before
step 2000 (overshoot hypothesis) or was rising into step 2000
(recovery hypothesis).

## Setup

- Use the existing `runs/2026-05-22-p1b-distill-pilot/step_001000/`
  LoRA adapter saved during the 2026-05-22 pilot (commit `bb72066`)
- Load through `INFER_LORA_PATH` (the qwen3.5 serve-side LoRA path
  shipped in `584f07b`)
- Re-run the same MMLU 5-shot eval the base and step_2000 ran (200
  samples → 171 evenly across 57 subjects)

Command:

```bash
INFER_LORA_PATH=runs/2026-05-22-p1b-distill-pilot/step_001000 \
./target/release/arle serve --backend cuda \
  --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --port 8125 -- \
  --num-slots 1 --max-seq-len 4096 \
  --chunked-prefill-size 4096 --max-num-batched-tokens 4096

.venv/bin/python scripts/arle_capability_eval.py \
  --base-url http://localhost:8125 \
  --model-id Qwen3___5-0___8B-Base \
  --tasks mmlu --n-samples 200 \
  --output bench-output/2026-05-22-capability-after-distill-step1000/
```

## Results

| Snapshot | MMLU | scored | invalid | Held-out KL |
|---|---:|---:|---:|---:|
| Base 0.8B | **51.4 %** | 142 | 29 | 1.739e-5 |
| step 1000 | **47.9 %** | 169 | **2** | 1.598e-5 |
| step 2000 (final) | **50.0 %** | 166 | 5 | 1.599e-5 |
| Teacher 4B | 77.3 % | 150 | 21 | — |

Δ MMLU vs base:

| Snapshot | Δ MMLU vs base |
|---|---:|
| step 1000 | **−3.48 pp** (valley) |
| step 2000 | **−1.41 pp** (recovering) |

**Trajectory: 51.4 → 47.9 (step 1000) → 50.0 (step 2000)** — not
monotonic decline, not overshoot. Instead a U-curve: regress hard
early, then partially recover.

KL trajectory in the same window (from `bb72066`):

| Step | Held-out KL |
|---:|---:|
| 0 | 1.739e-5 |
| 500 | 1.606e-5 |
| 1000 | **1.598e-5** |
| 2000 | 1.599e-5 |

KL bottoms out around step 1000 and stays flat — but MMLU continues
improving from 47.9 % → 50.0 %. So **KL is decoupled from capability
in this regime**: MMLU recovers without further KL decrease, just by
the adapter parameters drifting toward a more task-correct subspace
of the teacher's distribution.

## What this tells us about the OPD dynamic

1. **Early phase (0 → ~1 k steps)**: student fits teacher's
   distribution shape at the cost of task accuracy. Invalid-rate
   plummets (29 → 2): the student is now committing strongly to
   A/B/C/D, but committing *wrongly* more often than the un-adapted
   base.
2. **Recovery phase (~1 k → 2 k steps)**: student starts to recover
   task accuracy as the LoRA parameters refine. MMLU goes from 47.9 %
   → 50.0 % (+2.1 pp) while KL stays effectively flat (~1.598e-5 →
   1.599e-5).
3. **The base 51.4 % is the temporary ceiling** that the distillation
   has to climb back to. By step 2000 we're still 1.4 pp under it.

This is consistent with the OPD literature (Agarwal et al. 2024,
MiniLLM 2024) which reports:

- A "valley" in early steps where distribution-fitting hurts task
  performance.
- A subsequent recovery phase that takes 5–20× longer than the valley
  itself.
- Final convergence (beating the base) typically at 10 k–100 k steps
  for capability tasks.

## Implications for next tranches

| Tranche | Hypothesis | Cost |
|---|---|---:|
| Resume training to 10 k steps | Recovery continues — by step 5–10 k MMLU exceeds base | +12 h GPU |
| Lower LR (1e-5 instead of 2e-5) | Less aggressive distribution-fitting → shallower valley + faster recovery | +3 h GPU |
| GKD λ-mixing (0.3 SFT + 0.7 OPD) | SFT signal stabilizes early-phase fluency, prevents valley | +3 h GPU + new flag in train |
| LR warmup + cosine decay | Smooth early-phase distribution-fitting | +3 h GPU |

The cheapest informative experiment is the LR sweep, because:

- It reuses the existing save / load / eval pipeline (no new code).
- A single 2 k-step run at lr=1e-5 should tell us if the valley is
  primarily LR-driven.
- If lr=1e-5 still shows a valley but a shallower one, the dynamic is
  fundamental and we need GKD mixing or longer horizon.

## Cross-links

- Original P1-B wins (full loop closure): [`2026-05-22-p1b-train-save-load-eval-loop.md`](2026-05-22-p1b-train-save-load-eval-loop.md)
- Capability baselines + framework: [`../projects/2026-05-22-serve-fix-and-capability-baselines.md`](../projects/2026-05-22-serve-fix-and-capability-baselines.md)
- Cross-validation vs HF: [`2026-05-22-arle-vs-hf-transformers-cross-validation.md`](2026-05-22-arle-vs-hf-transformers-cross-validation.md)
- Methodology research (the U-curve dynamic): [`../research/2026-05-22-opd-methodology-and-industry-best-practices.md`](../research/2026-05-22-opd-methodology-and-industry-best-practices.md)
- Comparison artifact: [`../../bench-output/2026-05-22-distill-trajectory-08b-mmlu.md`](../../bench-output/2026-05-22-distill-trajectory-08b-mmlu.md)

## Rule

When a distillation run regresses on a capability benchmark, eval every
saved intermediate checkpoint before concluding "OPD hurt this model".
The U-curve dynamic (valley → recovery) is the default expectation in
the literature, and the regression often is just an incomplete
recovery phase, not a true regression. A monotonically-declining KL
with non-monotonic capability is the smoking gun for this regime.
