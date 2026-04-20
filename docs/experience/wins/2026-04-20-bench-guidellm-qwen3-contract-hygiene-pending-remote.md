# Qwen3 Contract Hygiene — guidellm sweep, metal-qwen3-contract-hygiene, 2026-04-20

## Context

- **Status:** `pending-remote`
- **Backend:** metal
- **Model:** Qwen/Qwen3-4B
- **Hardware:** Apple Silicon remote run pending
- **Commit:** `596a092` + local uncommitted workspace changes
- **Feature set:** `cargo build --release --no-default-features --features metal`
- **Non-default flags / env vars:** none recorded locally
- **Server launch:** pending remote run

## Canonical params (DO NOT CHANGE PER-RUN)

```bash
guidellm benchmark \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --profile sweep \
  --data prompt_tokens=4096,output_tokens=256 \
  --max-seconds 60 \
  --random-seed 20260416 \
  --output-dir bench-output/2026-04-20-metal-qwen3-contract-hygiene/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh metal-qwen3-contract-hygiene`

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| pending-remote | pending-remote | pending-remote | pending-remote | pending-remote | pending-remote | pending-remote |

## Artefacts

- Raw: pending remote run
- CSV: pending remote run
- HTML: pending remote run

## Delta vs previous snapshot

- **Baseline:** first run for this contract-hygiene follow-up
- **Delta table:** pending first remote run

## Notes

- What changed in the code since baseline: Qwen3 tensor-name authority moved into `crates/qwen3-spec`; infer loaders now consume the shared naming contract; ChatML rendering now exposes shared byte spans for SFT supervision; `train_grpo` SFT warmup uses a Qwen3 batch forward path instead of per-row single forwards.
- Suspected cause of any regression: no material runtime regression is expected from the infer-side contract cleanup; any change should be noise-level unless the shared prompt construction alters request text in an unexpected way.
- Follow-ups: run `scripts/bench_guidellm.sh metal-qwen3-contract-hygiene` on the next Metal machine with weights available and compare against the most recent Qwen3 metal baseline.
