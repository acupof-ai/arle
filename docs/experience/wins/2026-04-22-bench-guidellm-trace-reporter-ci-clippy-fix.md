# Trace reporter CI clippy fix — guidellm sweep, cuda, 2026-04-22

## Goal

- Regression-check the `trace_reporter` cleanup that unblocked CI without
  changing trace semantics or steady-state serving behavior.

## Hypothesis

- Rewriting the Option defaulting code to satisfy clippy should be behavior
  preserving and should not move serving latency or throughput.

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
- **Commit:** working tree atop `39152ac`
- **Feature set:** `cargo build --release --no-default-features --features cuda,no-cuda`
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

- This workspace is not a CUDA bench host, so the canonical `guidellm` sweep
  was not executed locally for this CI-only fix.

## Learnings

- CI-only hot-path cleanups still need a bench paper trail in this repo, even
  when the expected runtime delta is zero.
- The exact CI failures here were both downstream of `cargo clippy` on
  `infer/src/trace_reporter.rs`, so one local clippy reproduction covered both
  Linux and macOS job failures.

## Δ vs baseline

- **Baseline:** first run for this CI-fix tranche

## Artefacts

- Pending remote run

## Notes

- What changed in the code since baseline: replaced unnecessary lazy Option
  defaults in `infer/src/trace_reporter.rs` with clippy-clean equivalents
- Suspected cause of any regression: n/a; logic is intended to be identical
- Follow-ups: execute the canonical CUDA sweep remotely if this trace plumbing
  is bundled into a larger serving change set
