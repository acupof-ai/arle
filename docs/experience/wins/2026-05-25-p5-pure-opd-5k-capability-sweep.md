# P5 Pure OPD 5k Capability Sweep

Related: `docs/projects/2026-05-24-opd-mainline-task-backlog.md` T14,
`docs/experience/wins/2026-05-25-capability-eval-preflight.md`,
`docs/experience/wins/2026-05-22-distill-trajectory-valley-then-recovery.md`,
and `docs/experience/wins/2026-05-22-opd-task-divergent-impact.md`.

## Context

P5 finished the pure-OPD 5k run at
`runs/2026-05-24-p5-pure-opd-5k/` with checkpoints every 1000 steps. The train
KL kept improving, but heldout KL formed a shallow V: best at step 1000, then
slowly worse through step 5000. T14 checks whether visible capability follows
that heldout-KL winner.

The eval uses the same lightweight harness and sampling convention as the
2026-05-22 OPD cycle:

- MMLU 5-shot through `scripts/arle_capability_eval.py --tasks mmlu,gsm8k`
- `--n-samples 200`, which means 171 MMLU questions evenly across 57 subjects
  and 200 GSM8K questions
- Baseline reused from the 2026-05-22 no-LoRA Qwen3.5-0.8B-Base run:
  `bench-output/2026-05-22-capability-baseline-08b-retry-after-longprompt-fix/summary.json`

## Method

The first attempt used the T12 `arle serve` wrapper shape and found a lifecycle
hazard: killing the wrapper process can leave the backend `infer` child alive,
which makes a sequential checkpoint sweep risk hitting the previous adapter's
server. That attempt was discarded and its output directory was removed.

The accepted sweep launched `target/release/infer` directly, one checkpoint at
a time. After every eval, the script killed the process group, verified port
8125 closed, and checked GPU memory returned to the Edge-only baseline
(`1093 MiB used`, `14851 MiB free`).

Artifacts:

```text
bench-output/2026-05-25-p5-pure-opd-5k-capability-sweep/step_001000/summary.json
bench-output/2026-05-25-p5-pure-opd-5k-capability-sweep/step_002000/summary.json
bench-output/2026-05-25-p5-pure-opd-5k-capability-sweep/step_003000/summary.json
bench-output/2026-05-25-p5-pure-opd-5k-capability-sweep/step_004000/summary.json
bench-output/2026-05-25-p5-pure-opd-5k-capability-sweep/step_005000/summary.json
```

## Results

| checkpoint | train_kl | heldout_kl | MMLU | MMLU scored/invalid | GSM8K | GSM8K scored/invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | n/a | 1.739e-5 | **51.41%** | 73/142 inv 29 | 1.55% | 3/194 inv 6 |
| step_001000 | 1.357e-5 | **1.598e-5** | 47.93% | 81/169 inv 2 | 2.22% | 4/180 inv 20 |
| step_002000 | 1.318e-5 | 1.599e-5 | **50.00%** | 83/166 inv 5 | 1.60% | 3/188 inv 12 |
| step_003000 | 1.299e-5 | 1.603e-5 | 49.40% | 82/166 inv 5 | **3.76%** | 7/186 inv 14 |
| step_004000 | 1.288e-5 | 1.611e-5 | 45.78% | 76/166 inv 5 | 2.73% | 5/183 inv 17 |
| step_005000 | **1.281e-5** | 1.618e-5 | 42.26% | 71/168 inv 3 | 1.09% | 2/183 inv 17 |

Delta vs base:

| checkpoint | MMLU delta | GSM8K delta |
| --- | ---: | ---: |
| step_001000 | -3.48 pp | +0.68 pp |
| step_002000 | -1.41 pp | +0.05 pp |
| step_003000 | -2.01 pp | +2.21 pp |
| step_004000 | -5.63 pp | +1.18 pp |
| step_005000 | -9.15 pp | -0.45 pp |

## KL vs Capability

The heldout-KL winner and MMLU winner are not the same checkpoint:

- Heldout KL best: step 1000 (`1.598e-5`)
- MMLU best: step 2000 (`50.00%`)
- Train KL best: step 5000 (`1.281e-5`)
- GSM8K best: step 3000 (`3.76%`)

Correlation over the five checkpoints:

| Pair | Pearson | Spearman | Reading |
| --- | ---: | ---: | --- |
| heldout KL vs MMLU | -0.919 | -0.700 | Later overfit broadly hurts MMLU, but the exact heldout-KL minimum is not the MMLU optimum. |
| heldout KL vs GSM8K | -0.319 | -0.200 | GSM8K remains too near the floor for this 200-sample run to support a strong trajectory claim. |

This is another data point for the 2026-05-22 rule: KL is useful substrate
evidence, not a capability verdict. It is directionally helpful after step
2000, but it does not select the best checkpoint by itself.

## Errors-Style Reflection

The industry-headline hypothesis did not pass on MMLU. No P5 checkpoint beats
the no-LoRA 0.8B base; the best MMLU row is still -1.41 pp vs base.

The GSM8K step_003000 row is numerically above both the base (1.55%) and the
historical 4B teacher row (2.5%), but it is only 7 correct answers out of 186
scored. Per
`docs/experience/wins/2026-05-22-opd-task-divergent-impact.md`, this is a
near-floor fluctuation until a fresh sample or larger benchmark confirms it.
Do not put this in README as an achievement headline.

New research direction: the capability optimum for this pure-OPD recipe likely
sits earlier than 5k and is task-dependent. The next informative run is not
"more steps at the same lr"; it is a controlled recipe change:

- lower LR or decay after the step-1000 heldout-KL minimum
- SFT/GKD anchor that prevents the late MMLU collapse
- finer-grain checkpoints around 1000-3000 if the same pure-OPD recipe is
  rerun

## Verification

```bash
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
target/release/infer --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base --port 8125 ...
.venv/bin/python scripts/arle_capability_eval.py --backend arle --base-url http://127.0.0.1:8125 --model-id Qwen3___5-0___8B-Base --tasks mmlu,gsm8k --n-samples 200 --output <step-dir>
```

- Five checkpoint evals completed sequentially.
- No concurrent capability evals were run.
- Final GPU state after sweep: RTX 4070 Ti SUPER, `1093 MiB used`,
  `14851 MiB free`, `0%` GPU utilization.

## Rule

For OPD checkpoint sweeps, launch the backend process directly or kill the full
process group. Wrapper-level `arle serve` is fine for a single manual session,
but a sequential sweep must prove the old backend is gone before the next
adapter starts.

## Verdict

PASS for T14 measurement. KILL/defer for the README capability headline:
P5 pure OPD 5k did not beat base on MMLU, and the GSM8K bump is too close to
the floor to treat as an industry-recognized result.
