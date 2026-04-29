# Scheduler Step Phase Stats — guidellm regression, CUDA L4, 2026-04-29

## Goal

- Validate that scheduler step phase telemetry does not regress c=16 CUDA
  throughput. Goal type: regression.

## Hypothesis

- The added EMA update and relaxed atomic stores should be below benchmark
  noise, while `/v1/stats` gains `step_phase_us=...` for bench tracing.

## Command

```bash
CARGO_HOME=/tmp/arle-cargo-home CARGO_TARGET_DIR=/tmp/arle-target \
  ZIG=/tmp/zig-0.15.2/zig-x86_64-linux-0.15.2/zig \
  CUDA_HOME=/usr/local/cuda cargo build --release -p infer --bin infer --features cuda
/tmp/arle-target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --kv-cache-dtype fp8 \
  --chunked-prefill-size 512
scripts/bench_guidellm.sh cuda-l4-qwen3-c16-post-step-phase \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3-4B \
  --processor infer/models/Qwen3-4B \
  --concurrencies 16 \
  --max-seconds 120
```

Invoked locally on the benchmark host; same raw run as the Qwen3 c=16
post-step-phase entry.

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** NVIDIA L4, 23034 MiB, driver 580.82.07
- **Commit:** `9dd0f329` runtime change; bench docs filled after `ac6c9de3`
- **Feature set:** `cargo build --release -p infer --bin infer --features cuda`
- **Non-default flags / env vars:** `--num-slots 16 --max-seq-len 4608 --kv-cache-dtype fp8 --chunked-prefill-size 512`
- **Server launch:** command above

## Canonical params

- Fixed c=16 regression run: `--concurrencies 16 --max-seconds 120`
- Data remains wrapper default 4096-in / 256-out.

## Results — sweep headline table

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | ITL max | E2E mean | E2E p99 | conc p50 | out tok/s | total tok/s | in tok/s | total in | total out | req/s actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conc16 | 12535.3 | 488.2 | 12322.6 | 13067.3 | 121.36 | 72.48 | 0.9 | 72.39 | 73.08 | 77.58 | 77.58 | 31.02 | 31.53 | 16 | 144.37 | 2463.84 | 2770.04 | 262208 | 16320 | 0.4 |

## Results — service-side KV / scheduler metrics

- Required check passed: `/v1/stats` includes
  `step_phase_us=adm:<n>,prefill:<n>,decode:<n>,emit:<n>,total:<n>`.
- Before snapshot: `step_phase_us=adm:40,prefill:14063,decode:17432,emit:7,total:31498`.
- After snapshot: `step_phase_us=adm:6,prefill:0,decode:73422,emit:12,total:73441`.
- Service trace peaks: waiting 0, active 16, running_batch 16, kv_util 97.0%.

## Results — request accounting

- Completed input tokens: 262208; incomplete input tokens: 0; errors: 0.
- Completed output tokens: 16320; incomplete output tokens: 0; errors: 0.

## Problems

- Local default Cargo registry was on Google Drive FUSE and hung in kernel I/O
  wait; reran with `/tmp/arle-cargo-home` and `/tmp/arle-target`.
- Host lacked `zig` and `tilelang`; installed Zig 0.15.2 under `/tmp` and
  installed the repo `tilelang`/`bench` Python extras.

## Learnings

- Step-phase telemetry must include outer `assign_slots()` admission time and
  must not emit fake zero phase gauges for unwired backends.
- The c=16 Qwen3 run confirms the new phase string appears in bench traces.

## Δ vs baseline

- **Baseline:** `docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-kv-quant-matrix.md`
- No direct same-command before/after entry was available in this workspace.
  Treat this as the post-step-phase c=16 regression reference.

## Artefacts

- Raw: `bench-output/2026-04-29-cuda-l4-qwen3-c16-post-step-phase/`
- Service trace: `bench-output/2026-04-29-cuda-l4-qwen3-c16-post-step-phase/service_stats_trace_summary.md`

## Notes

- Code change: `9dd0f329 feat(scheduler): expose step phase stats`
- Local verification completed: `cargo fmt --check`, `git diff --check`,
  `codex review --uncommitted` with P2 findings addressed.
