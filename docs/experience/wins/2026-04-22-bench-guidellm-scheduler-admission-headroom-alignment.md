# Scheduler admission headroom alignment — guidellm sweep, cuda, 2026-04-22

## Goal

- Regression-check the scheduler fix that aligns admission page budgeting with
  the future page headroom already claimed by active prefill/decode slots.

## Hypothesis

- High-concurrency TTFT should improve because `assign_slots()` will stop
  over-admitting long prompts that execution cannot actually advance with the
  available KV pages.

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

- Local environment is not a CUDA bench host, so no in-repo guidellm sweep was
  run before commit.

## Learnings

- Full-ISL reservation has to survive past the initial slot-admission pass.
  Reconstructing admission budget from raw `seq_len` alone lets later requests
  borrow pages already promised to active long prompts and decode tails.

## Δ vs baseline

- **Baseline:** first run for this micro-tranche

## Artefacts

- Pending remote run

## Notes

- What changed in the code since baseline: `AdmissionPageBudget::from_scheduler()`
  now reserves active-slot future headroom instead of only the pages currently
  materialized in `seq_len`
- Suspected cause of any regression: n/a
- Follow-ups: run the canonical CUDA `c4/c8/c16` sweep and re-check the
  admission-vs-execution mismatch noted in the `ede0daa` trace
