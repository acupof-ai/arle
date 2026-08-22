# Qwen3.6-27B corrected baseline — CUDA, 2026-08-10

> Status: Shipped

## Goal

Establish the long-agent throughput baseline after correcting target/draft norm
semantics and the fixed-output runner.

## Hypothesis

The repaired DFlash norm path restores speculative acceptance, and isolated
warmup plus fixed 214-token outputs produces a complete cache-valid sweep.

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url http://127.0.0.1:8370 \
  --model ThinkingCap-Qwen3.6-27B-FP8 \
  --prompts-jsonl artifacts/bench-agent-32k-16x8.jsonl \
  --concurrency-grid 1,2,4,8,16 --requests-per-concurrency 128 \
  --max-tokens 214 --temperature 0 --seed 20260416 \
  --timeout-seconds 900 --output artifacts/c98-baseline/bench
```

- Runtime: `9b38ba6c0`, binary SHA-256 `5df97e8711ecda7787106a503b7dea432adbab5e865587c9f8a85725d0f078ca`
- Runner: `c98c4e0b2`, `ignore_eos=true`, disjoint warmup prefix
- Dataset SHA-256: `8867f63eaac2f0537bb2b17847a7d0d3c1bb8d504c1ad191e97d673e9ecc4f34`
- Prompt tokens: 32425 / 34827.5 / 37248 min/p50/max
- Completion tokens: 214 / 214 / 214
- Trials: one ascending sweep on one fresh serve

## Environment

- 1×H20 GPU2, UUID `GPU-77551814-ffe0-d267-728e-a3a20a0612de`
- Kernel bundle `e9454f1fc2320f4a62cabe87407442759f956cb4941cb632db953e19ca882cec`
- ThinkingCap-Qwen3.6-27B-FP8 target, Qwen3.6-27B-DFlash draft
- TP=1, eager, 16 slots × 195 MiB
- `--spec-type dspark --mtp-draft-model /host/Qwen3.6-27B-DFlash --dspark-block-size 6 --max-running-requests 16`

## Results

| c | completed | errors | TTFT p50/p99 ms | ITL mean/p99 ms | accept |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128/128 | 0 | 951.9 / 10276.7 | 10.56 / 49.46 | 41.57% |
| 2 | 128/128 | 0 | 564.4 / 976.2 | 17.08 / 94.75 | 27.45% |
| 4 | 128/128 | 0 | 625.2 / 1470.3 | 29.57 / 513.63 | 27.81% |
| 8 | 128/128 | 0 | 613.0 / 3668.8 | 49.85 / 542.96 | 26.90% |
| 16 | 128/128 | 0 | 939.0 / 8141.3 | 92.27 / 760.58 | 27.32% |

Prefix hits were 112/128 at c=1 and 128/128 thereafter. Concurrent needle
c=2/8/16 ×3 passed 78/78 exact with zero misses.

Raw artifacts:
`/host/agent-infer-9b38-g2/artifacts/c98-baseline/{bench.json,bench.csv,summary.json,serve.log,console.log,stats-start.json,stats-end.json,artifacts-sha256.txt}`.

## Problems

Two earlier sweeps were excluded: one allowed natural EOS and stopped at
127/128; the next warmed with dataset prompt zero and produced 113 rather than
112 c=1 prefix hits.

## Learnings

PASS. DFlash acceptance recovered from 0.334% to 26.90-27.81% at c=2-16. The
new fingerprint has one valid sweep, so candidate deltas require matched A/B
until repeat variance is measured.
