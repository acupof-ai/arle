# Scheduler two-pass budget knobs — guidellm sweep, cuda, 2026-04-22

## Goal

- Regression-check the scheduler config tranche that formalizes whole-step token budgeting and decode-active per-request prefill caps.

## Hypothesis

- No regression from config-surface cleanup alone. These knobs make the live two-pass planner explicit without changing the already-landed planner shape.

## Command

```bash
scripts/bench_guidellm.sh cuda \
  --model Qwen/Qwen3.5-4B \
  --processor models/Qwen3.5-4B
```

Status: `pending-remote` (CUDA bench host required).

## Environment

- **Backend:** `cuda`
- **Model:** `Qwen/Qwen3.5-4B`
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

- Once the step planner is already consuming a whole-step token budget, the config surface should name those invariants directly instead of relying on implicit `chunked_prefill_size/max_prefill_tokens` combinations.

## Δ vs baseline

- **Baseline:** first run for this micro-tranche

## Artefacts

- Pending remote run

## Notes

- What changed in the code since baseline: formalized `max_num_batched_tokens` and `long_prefill_token_threshold` in `SchedulerConfig`, defaults, validation, and tests
- Suspected cause of any regression: n/a
- Follow-ups: run the canonical CUDA guidellm sweep on the remote bench host after the SGLang-gap closure series lands
