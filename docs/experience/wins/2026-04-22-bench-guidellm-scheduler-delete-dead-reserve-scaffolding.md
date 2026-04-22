# Scheduler delete dead reserve scaffolding — guidellm sweep, cuda, 2026-04-22

## Goal

- Regression-check a cleanup-only scheduler change that deletes unused reserve-ratio scaffolding and redundant prefill reservation state while the larger SGLang gap-closure refactor is still in progress.

## Hypothesis

- No measurable runtime change. The removed fields and helpers were not read by the live scheduler path.

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

- Dead scheduler scaffolding should be deleted as a dedicated micro-tranche so `cargo check`/`clippy` can keep the hot path honest during larger refactors.

## Δ vs baseline

- **Baseline:** first run for this micro-tranche

## Artefacts

- Pending remote run

## Notes

- What changed in the code since baseline: removed unused reserve-ratio fields/helpers from `Scheduler` and dropped the redundant `PrefillReservation::slot_idx`
- Suspected cause of any regression: n/a
- Follow-ups: run the canonical CUDA guidellm sweep on the remote bench host after the SGLang-gap closure series lands
