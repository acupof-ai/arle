# guidellm sweep qwen35-4b-infer-l4-p99 — guidellm sweep, qwen35-4b-infer-l4-p99, 2026-04-17

> Template for canonical guidellm bench wins. Copy this file when
> `scripts/bench_guidellm.sh` runs, fill the placeholders, commit. Never
> edit an existing wins entry — always create a new dated one and diff
> against the prior. Canonical params are locked in
> [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md) §3.

## Context

- **Backend:** cuda
- **Model:** Qwen/Qwen3.5-4B bf16 (hybrid 32-layer: 24 linear-attn + 8 full-attn,
  full-attn HD=256, linear SSM heads 16/32 — official HF release as of 2026-04-14)
- **Hardware:** NVIDIA L4 24GB, CUDA 12.8 (driver 580.82.07)
- **Commit:** d6cc932
- **Feature set:** `cargo build --release -p infer --features cuda`
- **Non-default flags / env vars:** `--num-slots 10 --max-seq-len 5120`
- **Server launch:** `./target/release/infer --model-path models/Qwen3.5-4B --port 8000
  --num-slots 10 --max-seq-len 5120`
- **Paired run:** `2026-04-17-bench-guidellm-qwen35-4b-sglang-l4-p99.md` and combined
  comparison `2026-04-17-sglang-p99-parity-qwen35-4b.md`.
- **Caveat — last 3 rate rows report 0**: at 0.27/0.30/0.33 r/s none of the
  requests completed 256 tokens inside the 60s window (saturation passed),
  so guidellm's per-request metrics fall through to 0. Throughput column
  is still valid (tokens per sec aggregated). Throughput column saturates
  cleanly at ~91 tok/s.

## Canonical params (DO NOT CHANGE PER-RUN)

```
guidellm benchmark \
  --target http://localhost:8000 \
  --model Qwen3.5-4B \
  --profile sweep \
  --data  prompt_tokens=4096,output_tokens=256 \
  --max-seconds 60 \
  --random-seed 20260416 \
  --output-dir /content/workspace/agent-infer/bench-output/2026-04-17-qwen35-4b-infer-l4-p99/ \
  --outputs json,csv,html
```

Invoked via: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| sync | 966.1 | 982.6 | 36.23 | 36.27 | 25.47 | 0.083 |
| throughput | 24667.5 | 44907.1 | 64.6 | 78.56 | 91.43 | 0.333 |
| 0.11458333333333333r/s | 1340.4 | 1359.1 | 41.1 | 41.13 | 28.16 | 0.1 |
| 0.14583333333333331r/s | 1356.7 | 1382.1 | 42.09 | 42.16 | 34.64 | 0.117 |
| 0.17708333333333331r/s | 1357.2 | 1373.8 | 46.6 | 46.64 | 40.08 | 0.15 |
| 0.20833333333333331r/s | 1351.9 | 1373.9 | 47.08 | 47.15 | 46.61 | 0.167 |
| 0.23958333333333331r/s | 989.8 | 1375.3 | 15.4 | 51.5 | 66.8 | 0.25 |
| 0.2708333333333333r/s | 0 | 0 | 0 | 0 | 73.62 | 0.283 |
| 0.3020833333333333r/s | 0 | 0 | 0 | 0 | 81.62 | 0.317 |
| 0.3333333333333333r/s | 0 | 0 | 0 | 0 | 89.79 | 0.333 |


## Artefacts

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-17-qwen35-4b-infer-l4-p99/benchmarks.json`
- CSV:  `/content/workspace/agent-infer/bench-output/2026-04-17-qwen35-4b-infer-l4-p99/benchmarks.csv`
- HTML: `/content/workspace/agent-infer/bench-output/2026-04-17-qwen35-4b-infer-l4-p99/benchmarks.html`

## Delta vs previous snapshot

- **Baseline:** <link to prior `2026-04-17-bench-guidellm-<label>.md`>
- **Delta table** (only when a prior snapshot exists — else "first run"):

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p50 @ synchronous | ... | ... | ... |
| out tok/s @ saturation | ... | ... | ... |

## Notes

- What changed in the code since baseline: <commits or "none">
- Suspected cause of any regression: <text or "n/a">
- Follow-ups: <issues to open or "none">
