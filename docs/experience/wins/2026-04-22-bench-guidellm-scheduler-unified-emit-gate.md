# Scheduler unified emit gate — guidellm sweep, cuda, 2026-04-22

## Goal

- Regression-check the scheduler refactor that routes both stopless and
  stop-sensitive streaming emit through one emit worker, leaving the scheduler
  to consume only gate results before the next decode launch.

## Hypothesis

- High-concurrency throughput and TTFT tail should improve modestly because
  textual stop checking, UTF-8 decode, and stream delta emission are no longer
  executed directly on the scheduler thread.

## Command

```bash
scripts/bench_guidellm.sh cuda \
  --model Qwen/Qwen3-4B \
  --processor models/Qwen3-4B
```

Status: `pending-remote` (CUDA bench host required).

## Environment

- **Backend:** `cuda`
- **Model:** `Qwen/Qwen3-4B`
- **Hardware:** pending remote CUDA host
- **Commit:** pending local commit for unified emit-gate tranche
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

- Local environment is not a CUDA bench host, so no in-repo guidellm sweep was
  run before commit.

## Learnings

- The clean overlap seam is not “async only for stopless requests”. A single
  emit worker can own both delta emission and stop-sequence scanning as long as
  the scheduler only waits on explicit gate results before launching the next
  decode step.

## Δ vs baseline

- **Baseline:** [2026-04-22-bench-guidellm-scheduler-async-emit-worker.md](./2026-04-22-bench-guidellm-scheduler-async-emit-worker.md)

## Artefacts

- Pending remote run

## Notes

- What changed in the code since baseline: unified stopless and stop-sensitive
  streaming emit under one worker channel; removed scheduler-thread
  `emit_delta()` control-flow branch
- Suspected cause of any regression: n/a
- Follow-ups: confirm whether c8/c16 TTFT tail drops once the CUDA sweep is rerun
