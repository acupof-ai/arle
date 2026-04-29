# Qwen3.5-4B c=16 FP8 Post Step-Phase Telemetry — CUDA L4, 2026-04-29

## Goal

- Measure Qwen3.5-4B performance at 16-way concurrency after the step-phase
  stats and headline-table tracing patches. Goal type: regression.

## Hypothesis

- Qwen3.5 should retain lower ITL than Qwen3 because it has fewer full KV
  layers, but c=16 may need lower static memory fraction or chunk size to
  avoid the HD256 workspace OOM cliff observed in prior runs.

## Command

```bash
CARGO_HOME=/tmp/arle-cargo-home CARGO_TARGET_DIR=/tmp/arle-target \
  ZIG=/tmp/zig-0.15.2/zig-x86_64-linux-0.15.2/zig \
  CUDA_HOME=/usr/local/cuda cargo build --release -p infer --bin infer --features cuda
/tmp/arle-target/release/infer \
  --model-path infer/models/Qwen3.5-4B \
  --port 8000 \
  --num-slots 16 \
  --max-seq-len 4608 \
  --kv-cache-dtype fp8 \
  --mem-fraction-static 0.70 \
  --chunked-prefill-size 512
scripts/bench_guidellm.sh cuda-l4-qwen35-c16-post-step-phase \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3.5-4B \
  --processor infer/models/Qwen3.5-4B \
  --concurrencies 16 \
  --max-seconds 120
```

Invoked locally on the benchmark host after installing the missing local
prereqs (`zig` 0.15.2 and `pip install -e '.[tilelang,bench]'`).

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3.5-4B
- **Hardware:** NVIDIA L4, 23034 MiB, driver 580.82.07
- **Commit:** `9dd0f329` runtime change; bench docs filled after `ac6c9de3`
- **Feature set:** `cargo build --release -p infer --bin infer --features cuda`
- **Non-default flags / env vars:** `--num-slots 16 --max-seq-len 4608 --kv-cache-dtype fp8 --mem-fraction-static 0.70 --chunked-prefill-size 512`
- **Server launch:** command above

## Canonical params

- Fixed c=16 regression run: `--concurrencies 16 --max-seconds 120`
- Data remains wrapper default 4096-in / 256-out.

## Results — sweep headline table

| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | ITL max | E2E mean | E2E p99 | conc p50 | out tok/s | total tok/s | in tok/s | total in | total out | req/s actual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conc16 | 8974.7 | 6649.6 | 11913.9 | 17314.3 | 94.45 | 49.24 | 30.73 | 58.57 | 99.34 | 100.57 | 100.57 | 24.15 | 32.25 | 16 | 154.8 | 2639.33 | 3234.55 | 270386 | 16846 | 0.55 |

## Results — service-side KV / scheduler metrics

- Samples: 151 ok / 0 failed, poll interval 1000ms.
- Peaks: waiting 15, active 16, running_batch 16, prefill_queue 16, kv_util 80.8%.
- Trace distribution: waiting q99 14 / peak 15; kv_util q50 23.2%, q75 59.2%, q99 80.5%, peak 80.8%.
- Token counters: decode_tokens q50 2, q75 16, peak 16; prefill_tokens q75 7168 / peak 8192; tokens_out peak 12808.
- `/v1/stats` emitted step phase telemetry; before snapshot included `step_phase_us=adm:190,prefill:19851,decode:33116,emit:7,total:38628`.

## Results — request accounting

- Completed input tokens: 270386; incomplete input tokens: 65536; errors: 0.
- Completed output tokens: 16846; incomplete output tokens: 2779; errors: 0.
- The run completed without request errors, but the 120s window ended with
  active work still in flight; this is why incomplete token counters are nonzero.

## Problems

- Local default Cargo registry was on Google Drive FUSE and hung in kernel I/O
  wait; reran with `/tmp/arle-cargo-home` and `/tmp/arle-target`.
- Host lacked `zig` and `tilelang`; installed Zig 0.15.2 under `/tmp` and
  installed the repo `tilelang`/`bench` Python extras.
- GuideLLM emitted a tokenizer warning for `qwen3_5` model type, but request
  validation and the full run completed.

## Learnings

- Qwen3.5 c=16 at 4096/256 is faster than Qwen3 on output throughput in this
  run: 154.8 vs 144.37 output tok/s.
- The lower `mem_fraction_static=0.70` left enough headroom for HD256 and graph
  warmup on the L4.

## Δ vs baseline

- **Baseline:** `docs/experience/wins/2026-04-29-bench-guidellm-cuda-l4-qwen35-c8-fp8.md`
- Compared with Qwen3-4B c=16 from the same host and command shape:
  Qwen3.5 output tok/s +7.2%, ITL p50 -19.1%, E2E mean -22.1%.

## Artefacts

- Raw: `bench-output/2026-04-29-cuda-l4-qwen35-c16-post-step-phase/`
- Headline: `bench-output/2026-04-29-cuda-l4-qwen35-c16-post-step-phase/headline_table.md`
- Service trace: `bench-output/2026-04-29-cuda-l4-qwen35-c16-post-step-phase/service_stats_trace_summary.md`

## Notes

- Requested explicitly: 16-concurrency Qwen3.5 performance.
