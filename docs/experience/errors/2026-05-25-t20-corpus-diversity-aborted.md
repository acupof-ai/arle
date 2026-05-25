# T20 Corpus Diversity Aborted

Status: aborted before capability sweep. Not a KILL verdict.

Related:
`docs/research/2026-05-25-opd-methodology-audit.md`,
`docs/experience/wins/2026-05-25-t18-recipe-variant-result.md`, and
`examples/opd/opd-diverse-1k.jsonl`.

## Context

T20 tested the hypothesis that the 20-row `sample-prompts.jsonl` corpus was the
main reason OPD capability stayed near the base model. The launch changed only
the prompts file, using the tokenizer-filtered 1000-row MMLU-derived corpus from
commit `1292db5`.

That was the wrong next priority after the methodology audit. The audit ranks
`rollout_len=8` as the largest method gap versus TRL/GKD defaults, while corpus
diversity is orthogonal to rollout horizon.

## Evidence Preserved

The run was stopped with SIGTERM after step 599. The partial log is preserved at
`runs/2026-05-25-t20-corpus-diversity/run.txt`; no capability sweep was run.

| step | train_kl | heldout_kl | checkpoint |
| ---: | ---: | ---: | --- |
| 0 | 1.360075994565e-5 | 1.385939549436e-5 | n/a |
| 500 | 1.184418480457e-5 | 1.195986737912e-5 | `step_000500` |

The partial data says the wider corpus still provides a KL training signal. It
does not answer whether corpus diversity improves MMLU/GSM8K capability.

## Root Cause

The next experiment should isolate the highest-ranked method gap first. Running
T20 first would spend GPU time on a lower-priority, orthogonal variable while
`rollout_len=8` remained unchanged.

## Fix

Stop T20 and run T21: keep the original 20-prompt corpus and change only
`--rollout-len` from 8 to 32, then optionally 64 if variant A is PASS/PARTIAL.

## Rule

When a methodology audit ranks root-cause hypotheses, run the highest-ranked
single-variable experiment first unless hardware constraints make it impossible.
