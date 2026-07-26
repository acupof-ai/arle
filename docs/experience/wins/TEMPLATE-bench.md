# <short title> — <backend>, <date>

> Status: <Shipped | pending-remote | Killed>

## Goal

<One wall-clock metric and target workload.>

## Hypothesis

<One treatment, predicted mechanism, expected delta.>

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url <url> \
  --model <model> \
  --prompts-jsonl bench-agent-32k-64.jsonl \
  --concurrency-grid 1,4,8,16 \
  --requests-per-concurrency 16 \
  --max-tokens 256 \
  --seed 20260416 \
  --timeout-seconds 900 \
  --output bench-output/<label>/bench
```

- Baseline: `<commit, binary hash, flags>`
- Treatment: `<commit, binary hash, one changed flag>`
- Prompt tokens: `<p50 / min / max>` — target 32768, within ±10% (spec §3.3)
- Completion tokens: `<p50 / min / max>`
- Trials: `<n>`

## Environment

- Host / GPU: `<...>`
- Driver / CUDA or Metal: `<...>`
- Model / dtype: `<...>`
- TP / EP / slots / KV: `<...>`
- Server flags: `<...>`

## Results

| concurrency | arm | completed | errors | output tok/s | req/s | TTFT p50/p99 ms | ITL p50/p99 ms | delta |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline | | | | | | | — |
| 1 | treatment | | | | | | | |

Raw artifacts: `<json>`, `<csv>`, `<server log>`.

## Problems

<Failures, retries, confounders, or `None`.>

## Learnings

<PASS / KILL / pending-remote, the measured number, and the next wall.>
