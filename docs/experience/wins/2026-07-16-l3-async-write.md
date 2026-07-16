# Non-blocking L3 writes — CUDA DSv4, 2026-07-16

> Status: pending-remote

## Goal

Remove inference-thread disk waits from DSv4 TP=4/8 prefix spill without
reducing output throughput or increasing payload write amplification above
1.01×.

## Hypothesis

A bounded single writer plus fence-polled D2H capture removes host completion
waits while preserving the existing mmap/direct formats and cache semantics.

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url http://127.0.0.1:8123 --model default \
  --prompts-jsonl /host/dsv4-kv-readme-20.jsonl \
  --concurrency-grid 1,4,8,16 --requests-per-concurrency 20 \
  --max-tokens 96 --temperature 0 --timeout-seconds 300 \
  --output /host/dsv4-l3-async/bench
```

- Baseline: [`2026-07-16-l3-direct-storage-engine.md`](2026-07-16-l3-direct-storage-engine.md)
- Treatment: bounded QD=8 writer and two pending DSv4 capture batches
- Prompt: 1,649 tokens; completion: 96 tokens
- Trials: pending TP=4/8 GPU availability

## Environment

- Target: 4×/8× NVIDIA H20, DeepSeek-V4-Flash-FP8, TP=4/8, EP=4/8
- L3: mmap default; direct only on a mounted local NVMe that passes the gate
- Automatic NUMA enabled; exact binding recorded by the target run

## Results

| concurrency | arm | completed | errors | output tok/s | TTFT p99 ms | ITL p99 ms | delta |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 20 | 0 | 40.35 | 472.0 | 42.0 | — |
| 4 | baseline | 20 | 0 | 73.05 | 1,130.2 | 99.3 | — |
| 8 | baseline | 20 | 0 | 109.81 | 1,651.0 | 95.6 | — |
| 16 | baseline | 20 | 0 | 121.82 | 3,307.7 | 122.6 | — |
| 1/4/8/16 | treatment | pending | pending | pending | pending | pending | pending |

- Baseline DSv4 warm process-to-ready: 30.506 s. This change does not touch
  weight loading; treatment remains to be measured.
- Baseline direct payload write amplification: 1.00026×. Treatment acceptance:
  ≤1.01× and zero inference-thread disk completion wait.
- Local gates: 30/30 `kv-native-sys` tests, 96/96 `infer-core` tests, CUDA/no-CUDA
  release check, and scoped Clippy passed.

Raw treatment artifacts: pending GPU availability.

## Problems

All target GPUs are occupied. No remote timing or correctness claim is made.
The compute stream still orders D2H before later GPU work; this tranche removes
host synchronization, not GPU copy-stream serialization.

## Learnings

pending-remote. Run same-binary TP=4 and TP=8 A/B when H20s are free; accept
only with coherent output, zero tier failures, non-negative throughput, and the
write-amplification/wait gates above.
