# Stream Stop Short-Circuit — guidellm sweep, metal-stop-short-circuit, 2026-04-20

## Context

- **Status:** `pending-remote`
- **Backend:** metal
- **Model:** Qwen/Qwen3.5-4B
- **Hardware:** Apple Silicon remote run pending
- **Commit:** pending local integration
- **Feature set:** `cargo build --release --no-default-features --features metal`
- **Non-default flags / env vars:** pending remote run
- **Server launch:** pending remote run

## Canonical params (DO NOT CHANGE PER-RUN)

```bash
guidellm benchmark \
  --target http://localhost:8000 \
  --model Qwen/Qwen3.5-4B \
  --profile sweep \
  --data prompt_tokens=4096,output_tokens=256 \
  --max-seconds 60 \
  --random-seed 20260416 \
  --output-dir bench-output/2026-04-20-metal-stop-short-circuit/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh metal-stop-short-circuit`

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| pending-remote | pending-remote | pending-remote | pending-remote | pending-remote | pending-remote | pending-remote |

## Artefacts

- Raw: pending remote run
- CSV: pending remote run
- HTML: pending remote run

## Delta vs previous snapshot

- **Baseline:** none recorded for this stop-short-circuit follow-up
- **Delta table:** pending first remote run

## Notes

- What changed in the code since baseline: streamed text stops now use the shared `StopChunkProcessor` in serial runtime, server engine, and Metal runtime paths; the consumer stops seeing bytes after the marker, while final usage still comes from the backend's real completion result.
- Suspected cause of any regression: stop handling now preserves correctness without fabricating usage; any throughput delta comes from the extra backend work after a stop is matched and needs a real Metal sweep.
- Follow-ups: run `scripts/bench_guidellm.sh metal-stop-short-circuit` on the next Metal machine with model weights available; compare against the most recent Metal baseline and link the delta.
