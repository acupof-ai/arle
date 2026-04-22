# GLM4 delete dual-write prefill path — guidellm sweep, cuda, 2026-04-22

## Goal

- Regression-check the deletion of GLM4's dead dual-write prefill path and the matching CUDA FFI/kernel so the remaining paged-prefill work can converge on the live Qwen paths only.

## Hypothesis

- No measurable runtime change. The removed GLM4 dual-write path was not exercised by the current scheduler path.

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

- Local environment is not a CUDA bench host, so no in-repo guidellm sweep or nvcc build was run before commit.

## Learnings

- When a hot-path cleanup deletes a stale model-specific kernel entrypoint, remove the model code, FFI surface, and CUDA implementation in one micro-tranche.

## Δ vs baseline

- **Baseline:** first run for this micro-tranche

## Artefacts

- Pending remote run

## Notes

- What changed in the code since baseline: deleted GLM4's dead dual-write prefill implementation plus the matching CUDA FFI/kernel
- Suspected cause of any regression: n/a
- Follow-ups: run the canonical CUDA guidellm sweep on the remote bench host after the SGLang-gap closure series lands
