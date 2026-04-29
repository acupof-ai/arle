# Qwen3-4B c=16 FP8 Post Step-Phase Telemetry — pending remote, CUDA L4, 2026-04-29

## Goal

- Measure Qwen3-4B performance at 16-way concurrency after the step-phase
  stats and headline-table tracing patches. Goal type: regression.

## Hypothesis

- Throughput should remain within noise of the latest FP8 c=16 baseline;
  headline output should now include service trace distributions and
  step-phase fields.

## Command

```bash
CUDA_HOME=/usr/local/cuda cargo build --release --features cuda
target/release/infer \
  --model-path models/Qwen3-4B \
  --num-slots 16 \
  --max-seq-len 4608 \
  --kv-cache-dtype fp8 \
  --chunked-prefill-size 512
scripts/bench_guidellm.sh cuda-l4-qwen3-c16-post-step-phase \
  --model Qwen/Qwen3-4B \
  --processor infer/models/Qwen3-4B \
  --concurrencies 16 \
  --max-seconds 120
```

Invoked via: pending remote.

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** pending remote L4; local host has NVIDIA L4 23034 MiB, driver 580.82.07
- **Commit:** `9dd0f329`
- **Feature set:** `cargo build --release --features cuda`
- **Non-default flags / env vars:** `--num-slots 16 --max-seq-len 4608 --kv-cache-dtype fp8 --chunked-prefill-size 512`
- **Server launch:** command above

## Canonical params

- Fixed c=16 regression run: `--concurrencies 16 --max-seconds 120`
- Data remains wrapper default 4096-in / 256-out.

## Results — sweep headline table

Pending remote.

## Results — service-side KV / scheduler metrics

Pending remote.

## Results — request accounting

Pending remote.

## Problems

- Local workspace has no runnable release binary or Qwen3 weights, so this
  run is opened as `pending-remote`.

## Learnings

- Pending remote.

## Δ vs baseline

- **Baseline:** `docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-kv-quant-matrix.md`
- Delta table pending remote.

## Artefacts

- Raw: pending `bench-output/2026-04-29-cuda-l4-qwen3-c16-post-step-phase/`

## Notes

- Requested explicitly: 16-concurrency Qwen3 performance.
