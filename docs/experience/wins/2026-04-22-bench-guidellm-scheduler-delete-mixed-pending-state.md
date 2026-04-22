# Scheduler delete mixed pending state — guidellm sweep, cuda, 2026-04-22

## Goal

- Regression-check the deletion of the last scheduler-side mixed pending state so the CUDA scheduler keeps a single decode launch/readback shape.

## Hypothesis

- No regression. This change removes stale bookkeeping after the step planner stopped relying on scheduler-side mixed decode launches.

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

- Once the scheduler planner has one canonical decode launch shape, stale mixed-only pending state should be deleted immediately rather than kept as inert baggage.

## Δ vs baseline

- **Baseline:** first run for this micro-tranche

## Artefacts

- Pending remote run

## Notes

- What changed in the code since baseline: removed `PendingDecode::mixed_prefill` and the redundant `PrefillReservation::slot_idx`
- Suspected cause of any regression: n/a
- Follow-ups: run the canonical CUDA guidellm sweep on the remote bench host after the SGLang-gap closure series lands
