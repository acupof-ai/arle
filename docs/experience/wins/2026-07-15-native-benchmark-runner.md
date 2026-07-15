# Native benchmark runner verified on DSv4 TP=4

## Goal

Replace the external benchmark dependency with one auditable repository-native client.

## Hypothesis

Fixed concurrency will complete without the unbounded request queue that exhausted KV pages.

## Command

```bash
python3 scripts/bench_throughput.py \
  --url http://127.0.0.1:18197 \
  --model DeepSeek-V4-Flash-FP8 \
  --prompts-jsonl /host/dspark_natural_32in_128out.jsonl \
  --concurrency-grid 16 --requests-per-concurrency 16 \
  --max-tokens 128 --seed 20260416 \
  --output /host/arle-megamoe-t1/bench-output/2026-07-15-native-runner-c16-smoke/benchmarks
```

## Environment

- Binary commit: `91d105f3f618`; SHA-256 `16a2dd6a30d64e333a082991e938c18e5c6558573239ed7eb963e0e38e5f98e1`
- Runner SHA-256: `33883cb58da2d9ea7b259d3903c313fb79c0ca0089a85b641e5f17f9e794b5c7`
- Hardware: 4× NVIDIA H20, TP=4, driver 535.161.08
- Client: Python 3.12.3, httpx 0.28.1
- Server: NCCL all-reduce, speculative decode off, 16 slots, 400 KV pages

## Results

| c | complete | out tok/s | total tok/s | req/s | TTFT p50 | ITL p50 | E2E p50 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4/4 | 37.66 | 75.62 | 1.18 | 174.44 ms | 21.68 ms | 868.97 ms |
| 16 | 16/16 | 175.10 | 219.22 | 1.37 | 1978.92 ms | 74.56 ms | 11692.50 ms |

The c=16 run had zero incomplete requests, zero request errors, zero correctness failures, zero prefix hits, and returned to `active=0`, `queue_depth=0`, `kv_free_pages=367`.

## Problems

- This is a runner verification, not an optimization A/B: c=1 used 32 output tokens and c=16 used 128.
- The checked-in canonical workload still needs token-calibrated long-context cases.

## Learnings

- Bound in-flight work directly; request rate is not a substitute for concurrency.
- Use server-reported usage as token truth. Missing usage makes a request incomplete.
- Persist per-request prompt and output so aggregate failures remain attributable.

## Delta vs baseline

First native-runner verification. No performance delta is claimed.

## Artefacts

- `/host/arle-megamoe-t1/bench-output/2026-07-15-native-runner-smoke/benchmarks.{json,csv}`
- `/host/arle-megamoe-t1/bench-output/2026-07-15-native-runner-c16-smoke/benchmarks.{json,csv}`
