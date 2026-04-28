# Multi-SM Phase D — guidellm sweep on sm_86 (A10/3090), 2026-04-28 (pending-remote)

> **Status: pending-remote.** Commit A landed the tier policy +
> `TORCH_CUDA_ARCH_LIST` migration; this entry retires when the multi-SM
> binary (commits B + C) is bench-validated on a real sm_86 box per
> [`docs/plans/sm-coverage.md`](../../plans/sm-coverage.md) §5.

## Goal

- Validate that the T1 fat binary built with
  `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"` runs the canonical
  guidellm sweep on **sm_86 (A10 or RTX 3090)** with TTFT and out_tok/s
  within ±5 % of the most recent same-SM baseline.

## Hypothesis

- A10 / 3090 baseline does not exist yet; first multi-SM run on sm_86 *is*
  the baseline. Subsequent runs delta against this.

## Command

```bash
scripts/bench_guidellm.sh cuda-multi-sm-86 \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-8B \
  --processor models/Qwen3-8B
```

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-8B (fits in 24 GB on RTX 3090 / A10)
- **Hardware:** A10 24 GB or RTX 3090 24 GB (record exact SKU when filled)
- **Commit:** TBD — multi-SM commit B + C merge sha
- **Feature set:** `cargo build --release --features cuda`
- **Non-default flags / env vars:** `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"`
- **Server launch:** `scripts/start_infer.sh Qwen3-8B 8000`

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`

## Results — sweep headline table

_Pending. Expected first run on remote A10 / 3090._

## Δ vs baseline

- **Baseline:** none yet (first sm_86 multi-SM run is the baseline).

## Notes

- Bench gate context: see
  [`docs/plans/sm-coverage.md`](../../plans/sm-coverage.md) §5.
- sm_86 is the lowest-priced T1 row; expect this to be the easiest of the
  four to schedule.
