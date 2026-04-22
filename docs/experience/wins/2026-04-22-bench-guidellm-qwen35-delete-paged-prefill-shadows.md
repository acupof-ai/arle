# Qwen3.5 paged-prefill dead-shadow cleanup — guidellm sweep, cuda, 2026-04-22

## Goal

- Regression-check the deletion tranche that removes dead HD256 paged-prefill shadows left behind after Qwen3.5 moved to state-owned paged-prefill buffers.

## Hypothesis

- No throughput claim for this tranche. The goal is to keep the live paged-prefill path compiling cleanly while deleting unused shared-plan and helper shadows.

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

- Once Qwen3.5 paged-prefill owns its per-forward plan/buffers in state, the old model-level shared plan cache and the unused HD256 helper path should be deleted together to avoid leaving a fake parallel surface behind.

## Δ vs baseline

- **Baseline:** first run for this micro-tranche

## Artefacts

- Pending remote run

## Notes

- What changed in the code since baseline: removed the unused model-level shared HD256 paged-prefill plan cache and deleted the orphaned HD256 paged-prefill helper in `ops/attention.rs`
- Suspected cause of any regression: n/a
- Follow-ups: run the canonical CUDA guidellm sweep on the remote bench host after the Qwen3.5 paged-prefill path is exercised end-to-end
