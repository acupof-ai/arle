# ARLE vs TRL GKD Head-to-Head

## Goal

Measure the industry-facing comparison point for OPD: ARLE's Qwen3-0.6B
CUDA OPD harness against HuggingFace TRL's closest equivalent,
`GKDTrainer`, on the same checkpoint shape, same 32 train prompts, same
4 held-out prompts, same `rollout_len=8`, same `lr=1e-7`, and 500 steps.

## Hypothesis

ARLE should win step wall-clock because the OPD loop is device-resident and
uses the fused CUDA path from the 2026-05-21 OPD CUDA cycle. TRL should be a
useful credibility baseline if its `GKDTrainer` reaches similar held-out KL
under the matched recipe.

## Params

ARLE reference:

- Artefact: `bench-output/2026-05-21-arle-cuda-opd-prompts-32-a-b/run.txt`
- Commit lineage: OPD CUDA stack through the 32-prompt run
- Model: `~/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B/`
- Teacher: frozen Qwen3-0.6B
- Student: same checkpoint, perturbed by amplitude `1e-3`
- Training: full-finetune AdamW, `lr=1e-7`, `rollout_len=8`, 500 steps
- Prompts: built-in 32 train prompts + same 4 held-out prompts

TRL reference:

- Script: `bench-output/2026-05-21-trl-gkd-baseline/trl_gkd_baseline.py`
- Artefacts:
  - `bench-output/2026-05-21-trl-gkd-baseline/run.txt`
  - `bench-output/2026-05-21-trl-gkd-baseline/metrics.jsonl`
  - `bench-output/2026-05-21-trl-gkd-baseline/summary.json`
  - `bench-output/2026-05-21-trl-gkd-baseline/nvidia-smi-before.txt`
  - `bench-output/2026-05-21-trl-gkd-baseline/nvidia-smi-after.txt`
- Python: project `.venv/bin/python`
- Torch: `2.11.0+cu130`
- Transformers: `5.8.0`
- TRL: `1.4.0`, `trl.experimental.gkd.GKDTrainer`
- GKD settings: `lmbda=1.0`, `beta=0.0`, greedy generation
  (`do_sample=false`), `max_new_tokens=8`
- Optimizer: AdamW via HF Trainer, betas `(0.9, 0.999)`, epsilon `1e-8`,
  weight decay `0`, grad clip `1.0`, constant LR `1e-7`

Important control note: the first runnable TRL pass accidentally used HF
Trainer's default linear LR decay, which drove the LR near zero by step 500.
That was rejected as a confounder. The committed result below uses constant
LR to match the ARLE harness.

## Env

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- CUDA build: torch `cu130`
- GPU snapshot before TRL run:
  `2026/05/21 10:54:20.369`, used `955 MiB`, free `14989 MiB`
- GPU snapshot after TRL run:
  `2026/05/21 11:00:26.631`, used `955 MiB`, free `14989 MiB`

## Results

Step wall-clock:

| Runner | Trainable params | Mean step seconds | Median step seconds | Sigma % | Peak memory evidence |
|---|---:|---:|---:|---:|---|
| ARLE CUDA OPD, matched 500-step harness | ~596M | `0.200370` | `0.200648` | n/a | full-finetune peak observed `15358 MiB` in 10k monitor |
| TRL `GKDTrainer`, constant LR | `596,049,920` | `0.408634` | `0.407652` | `1.327238` | torch peak allocated `12575.7 MiB`, reserved `12642.0 MiB` |

Wall-clock ratio:

| Comparison | Ratio |
|---|---:|
| TRL / ARLE matched harness | `2.04x` slower |
| TRL / ARLE 0.164s bare-step frontier | `2.49x` slower |

Held-out trajectory:

| Step | ARLE held-out exact % | ARLE held-out KL | TRL held-out exact % | TRL held-out KL |
|---:|---:|---:|---:|---:|
| 0 | 50.000000 | `2.172812e-2` | 64.062500 | `2.159872e-2` |
| 50 | 51.562500 | `2.115785e-2` | 64.062500 | `2.044910e-2` |
| 100 | 51.562500 | `2.052191e-2` | 64.062500 | `2.109818e-2` |
| 250 | 64.062500 | `1.913815e-2` | 64.062500 | `2.119996e-2` |
| 500 | 64.062500 | `1.771600e-2` | 100.000000 | `2.041379e-2` |

Train trajectory:

| Step | ARLE train exact % | ARLE train KL | TRL train exact % | TRL train KL |
|---:|---:|---:|---:|---:|
| 0 | 59.765625 | `1.415704e-2` | 48.437500 | `3.419336e-2` |
| 50 | 60.937500 | `1.267499e-2` | 63.867188 | `2.265303e-2` |
| 100 | 61.328125 | `1.154891e-2` | 66.015625 | `1.703158e-2` |
| 250 | 69.335938 | `9.527480e-3` | 74.609375 | `1.097483e-2` |
| 500 | 73.437500 | `7.759120e-3` | 87.500000 | `8.971481e-3` |

Derived deltas:

| Metric | ARLE 0 -> 500 | TRL 0 -> 500 |
|---|---:|---:|
| Train KL | `-45.19%` | `-73.76%` |
| Held-out KL | `-18.47%` | `-5.49%` |
| Held-out teacher-token NLL | n/a in the older ARLE 32-prompt run | `-2.49%` |

## Problems

The held-out KL acceptance target was not fully met. The final TRL held-out KL
is `2.041379e-2`, while the matched ARLE final held-out KL is
`1.771600e-2`; TRL is about `15.23%` higher. That misses the requested
roughly-10% comparability band.

This does not invalidate the wall-clock comparison, because the two runners
started from nearly identical held-out KL (`2.172812e-2` vs `2.159872e-2`) and
used the same prompt IDs and LR. It does mean the quality comparison should be
read as "same nominal recipe, different trainer implementation", not a proof
that the two optimizers are byte-equivalent.

The most likely remaining confounders are:

- TRL's `GKDTrainer` perturbation uses the same amplitude and seed, but the
  exact parameter traversal and RNG stream are HF/PyTorch-native, not ARLE's
  Rust tensor-store order.
- TRL's generation and loss path are the experimental `GKDTrainer` path. With
  `beta=0`, the loss is forward `KL(teacher || student)`, but the surrounding
  Trainer data flow is still not the same implementation as
  `train::opd::opd_step`.
- Held-out exact overlap is coarse on four prompts. TRL reaches `100%` exact
  held-out overlap at step 500 while still having worse full-distribution KL,
  matching the earlier ARLE lesson that exact-token overlap is not the primary
  metric.

## Learnings

ARLE wins the industry-facing wall-clock comparison for the matched 500-step
full-finetune setup: `2.04x` faster than TRL `GKDTrainer` on the same
Qwen3-0.6B prompt workload.

TRL uses less memory in this run (`12.6 GiB` torch reserved) than ARLE's
full-finetune peak observed during the 10k run (`15.4 GiB`). ARLE's LoRA path
is the user-facing answer for memory-constrained cards: the LoRA OPD bench
landed at `0.140092s`/step and `3934 MiB` peak, but that is a different
trainable-parameter regime and not the full-finetune TRL comparison above.

Quality-wise, ARLE's matched 500-step run generalizes better by held-out KL:
`-18.47%` for ARLE vs `-5.49%` for TRL. The exact-overlap result alone would
mislead in the opposite direction, so future head-to-head claims should lead
with held-out KL / teacher-token NLL and keep exact-token overlap secondary.

## Rule

For OPD framework comparisons, lock Trainer scheduler semantics before running
the bench. HF Trainer defaults such as linear LR decay are not neutral; they
change convergence over only 500 steps and must be matched explicitly.
