# Qwen3.5 paged-prefill graph compile-fix — guidellm sweep, cuda, 2026-04-22

## Goal

- Regression-check the compile-fix tranche that makes the Qwen3.5 paged-prefill / full-forward graph path borrow-correct and self-contained.

## Hypothesis

- No immediate throughput claim yet; this tranche should preserve behavior while making the new paged-prefill graph path compile and ready for follow-on validation.

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

- The paged-prefill graph path needs to land as a self-contained compile-fix tranche: state buffer lifecycle, graph metadata buffers, and graph-safe batched GEMM must ship together.

## Δ vs baseline

- **Baseline:** first run for this micro-tranche

## Artefacts

- Pending remote run

## Notes

- What changed in the code since baseline: made the Qwen3.5 paged-prefill graph path compile-clean by introducing stable paged-prefill state buffers and graph-safe batched GEMM support
- Suspected cause of any regression: n/a
- Follow-ups: run the canonical CUDA guidellm sweep on the remote bench host after the graph path is functionally exercised end-to-end
