# Qwen3.5 paged-prefill graph on non-zero start positions — guidellm sweep, cuda, 2026-04-22

## Goal

- Regression-check the qwen35 paged-prefill graph tranche that allows graph capture/replay on canonical paged-prefill chunks beyond `start_pos == 0`.

## Hypothesis

- No correctness change on the shipped path; later paged-prefill chunks should now stay on the same full-forward graph path instead of eagerly falling back only because the chunk starts after an attached prefix or earlier chunk.

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

- Qwen3.5 paged-prefill graph replay is gated by stable buffers and metadata invalidation, not by `start_pos == 0`. Keeping the old "first chunk only" gate silently left later paged-prefill chunks on the eager path.

## Δ vs baseline

- **Baseline:** [2026-04-22-bench-guidellm-qwen35-paged-prefill-graph-compile-fix.md](./2026-04-22-bench-guidellm-qwen35-paged-prefill-graph-compile-fix.md)
- Delta table pending remote run.

## Artefacts

- Pending remote run

## Notes

- What changed in the code since baseline: paged-prefill graph replay is no longer limited to `start_pos == 0`
- Suspected cause of any regression: n/a
- Follow-ups: run the canonical CUDA guidellm sweep on the remote bench host after finishing the remaining gap-closure tranches
