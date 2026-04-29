# Scheduler Step Phase Stats — pending remote guidellm regression, CUDA L4, 2026-04-29

## Goal

- Validate that scheduler step phase telemetry does not regress c=16 CUDA
  throughput. Goal type: regression.

## Hypothesis

- The added EMA update and relaxed atomic stores should be below benchmark
  noise, while `/v1/stats` gains `step_phase_us=...` for bench tracing.

## Command

```bash
CUDA_HOME=/usr/local/cuda cargo build --release --features cuda
target/release/infer \
  --model-path models/Qwen3-4B \
  --num-slots 16 \
  --max-seq-len 4608 \
  --kv-cache-dtype fp8 \
  --chunked-prefill-size 512
scripts/bench_guidellm.sh step-phase-stats-qwen3-c16 \
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

Pending remote. Required check: `/v1/stats` includes
`step_phase_us=adm:<n>,prefill:<n>,decode:<n>,emit:<n>,total:<n>`.

## Results — request accounting

Pending remote.

## Problems

- Local cargo verification repeatedly hung in kernel I/O wait; the target
  tests were run with `timeout 180s` and did not complete.
- Local workspace has no `target/release/infer` binary and no Qwen3 weights
  under `models/` or Hugging Face cache, so the c=16 bench cannot run here.

## Learnings

- Step-phase telemetry must include outer `assign_slots()` admission time and
  must not emit fake zero phase gauges for unwired backends.

## Δ vs baseline

- **Baseline:** `docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-kv-quant-matrix.md`
- Delta table pending remote.

## Artefacts

- Raw: pending `bench-output/2026-04-29-step-phase-stats-qwen3-c16/`

## Notes

- Code change: `9dd0f329 feat(scheduler): expose step phase stats`
- Local verification completed: `cargo fmt --check`, `git diff --check`,
  `codex review --uncommitted` with P2 findings addressed.
