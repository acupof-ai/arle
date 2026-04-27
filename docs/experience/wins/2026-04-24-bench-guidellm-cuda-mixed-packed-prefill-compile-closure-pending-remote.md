# CUDA mixed packed prefill compile closure pending remote verification

## Goal

- Regression-check the compile closure for the CUDA Qwen3 mixed decode+packed-prefill contract after the scheduler-side contract moved from one prefill row to row slices.

## Hypothesis

- The Qwen3 model mixed-forward implementation should accept the new slice-shaped `MixedBatchRequest` contract for the currently supported single prefill row and continue to fall back for unsupported shapes, while CUDA runtime performance still needs remote measurement.

## Command

```bash
scripts/bench_guidellm.sh cuda-qwen3-mixed-packed-prefill
```

Invoked via: pending remote CUDA host.

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** pending remote NVIDIA host
- **Commit:** pending; local workspace dirty on top of `9caaab4`
- **Feature set:** `CUDA_HOME=/usr/local/cuda cargo build --release`
- **Non-default flags / env vars:** none expected
- **Server launch:** `scripts/start_infer.sh models/Qwen3-4B 8000` or equivalent

## Canonical Params

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh cuda-qwen3-mixed-packed-prefill`

## Results

- Status: `pending-remote`
- Local verification completed:
  - `cargo fmt --check`
  - `cargo check -p infer --no-default-features --features cuda,no-cuda`

## Problems

- No CUDA runtime host is available in this macOS workspace, so the canonical `guidellm` sweep and any throughput delta are still pending.

## Learnings

- When the scheduler contract becomes slice-shaped, model-specific mixed-forward implementations should explicitly gate the shapes they actually support instead of relying on stale singular fields.

## Delta vs Baseline

- **Baseline:** [2026-04-24-bench-guidellm-cuda-mixed-packed-prefill-pending-remote.md](2026-04-24-bench-guidellm-cuda-mixed-packed-prefill-pending-remote.md)
- **Delta table:** pending remote CUDA run.

## Artefacts

- Raw: pending
- CSV: pending
- HTML: pending
- Service trace: pending

## Notes

- What changed in code since baseline: `Qwen3Model::decode_batch_with_prefill` now reads `MixedBatchRequest::prefills` / `prefill_start_positions` and returns fallback for shapes outside the currently implemented single-prefill mixed path.
- Suspected cause of any regression: mixed prefill row packing or a mismatch between scheduler-planned rows and model-supported rows.
- Follow-ups: run the CUDA `guidellm` sweep on the remote host and replace this pending result with a new dated completed entry.
