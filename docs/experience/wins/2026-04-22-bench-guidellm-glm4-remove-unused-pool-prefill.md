# GLM4 remove unused pool-prefill path — guidellm sweep, cuda, 2026-04-22

## Goal

- Regression-check a cleanup-only change that removes GLM4's unused `forward_prefill_with_pool` implementation while the SGLang gap-closure work standardizes paged-prefill on Qwen-only paths.

## Hypothesis

- No measurable runtime change. The removed GLM4 pool-prefill path is not exercised by the current scheduler flow.

## Command

```bash
scripts/bench_guidellm.sh cuda \
  --model THUDM/GLM-4 \
  --processor models/GLM-4
```

Status: `pending-remote` (CUDA bench host required).

## Environment

- **Backend:** `cuda`
- **Model:** `THUDM/GLM-4`
- **Hardware:** pending remote CUDA host
- **Commit:** pending local commit for this micro-tranche
- **Feature set:** `cargo build --release`
- **Non-default flags / env vars:** none
- **Server launch:** pending remote validation

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh cuda`

## Results — sweep headline table

Pending remote run.

## Problems

- Local environment is not a CUDA bench host, so no in-repo guidellm sweep was run before commit.

## Learnings

- Cleanup-only hot-path commits still need an explicit pending-remote bench stub; do not silently skip runtime validation.

## Δ vs baseline

- **Baseline:** first run for this micro-tranche

## Artefacts

- Pending remote run

## Notes

- What changed in the code since baseline: removed GLM4's dead `forward_prefill_with_pool` implementation
- Suspected cause of any regression: n/a
- Follow-ups: run the canonical CUDA guidellm sweep on the remote bench host after the SGLang-gap closure series lands
