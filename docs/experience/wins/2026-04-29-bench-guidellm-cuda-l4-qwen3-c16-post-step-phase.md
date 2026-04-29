# Qwen3-4B c=16 FP8 Post Step-Phase Telemetry — CUDA L4, 2026-04-29

## Goal

- Measure Qwen3-4B performance at 16-way concurrency after the step-phase
  stats and headline-table tracing patches. Goal type: regression.

## Hypothesis

- Throughput should remain within noise of the latest FP8 c=16 baseline;
  headline output should now include service trace distributions and
  step-phase fields.

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

Invoked locally on the benchmark host after installing the missing local
prereqs (`zig` 0.15.2 and `pip install -e '.[tilelang,bench]'`).

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

- Samples: 161 ok / 0 failed, poll interval 1000ms.
- Peaks: waiting 0, active 16, running_batch 16, prefill_queue 0, kv_util 97.0%.
- Trace distribution: waiting q99 0 / peak 0; kv_util q50 63.5%, q75 91.0%, q99 96.9%, peak 97.0%.
- Token counters: decode_tokens q75 16 / peak 16; prefill_tokens q75 7681 / peak 8192; tokens_out peak 16392.
- `/v1/stats` emitted step phase telemetry; after snapshot included `step_phase_us=adm:6,prefill:0,decode:73422,emit:12,total:73441`.

## Results — request accounting

- Completed input tokens: 262208; incomplete input tokens: 0; errors: 0.
- Completed output tokens: 16320; incomplete output tokens: 0; errors: 0.
- Completed requests: 64 inferred from output token total at 255 tokens/request.

## Problems

- Local default Cargo registry was on Google Drive FUSE and hung in kernel I/O
  wait; reran with `/tmp/arle-cargo-home` and `/tmp/arle-target`.
- Host lacked `zig` and `tilelang`; installed Zig 0.15.2 under `/tmp` and
  installed the repo `tilelang`/`bench` Python extras.

## Learnings

- Qwen3 c=16 with 4096/256 reaches 144.37 output tok/s on L4, but this run is
  KV-pressure heavy: kv_util peaked at 97.0%.
- Step-phase telemetry is visible in the trace with no request errors.

## Δ vs baseline

- **Baseline:** `docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-kv-quant-matrix.md`
- No direct same-command before/after entry was available in this workspace.
  Treat this as the post-step-phase c=16 reference point for Qwen3-4B.

## Artefacts

- Raw: `bench-output/2026-04-29-cuda-l4-qwen3-c16-post-step-phase/`
- Headline: `bench-output/2026-04-29-cuda-l4-qwen3-c16-post-step-phase/headline_table.md`
- Service trace: `bench-output/2026-04-29-cuda-l4-qwen3-c16-post-step-phase/service_stats_trace_summary.md`

## Notes

- Requested explicitly: 16-concurrency Qwen3 performance.
