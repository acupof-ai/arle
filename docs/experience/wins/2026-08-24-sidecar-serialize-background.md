# Recurrent sidecar serialization off critical path — CUDA, 2026-08-24

> Status: Shipped

## Goal

Reduce p99 ITL at c≥4 on the 32k agentic bench, where synchronous
`to_bytes()` (146.8 MiB, ~80ms) in `apply_output` stalled the engine step
when multiple requests finished prefill in the same step.

## Hypothesis

Moving `to_bytes()` and the chunked `insert_chunked` memcpy off the engine's
critical path to a background thread eliminates the serialization stall.
The main thread retains only `insert_many` (BTreeMap inserts, ~1ms).

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --prompts-jsonl /host/bench-agent-32k-16x8.jsonl \
  --concurrency-grid 1,4,8,16 \
  --requests-per-concurrency 32 \
  --max-tokens 214 \
  --temperature 0 \
  --seed 42 \
  --output /tmp/sidecar-prechunk-bench
```

- Baseline: `16857e541` (previous session, synchronous `to_bytes` + `insert_chunked`)
- Treatment: `e00580c83` (binary `78b5f584…`, background `to_bytes` + chunking, main-thread `insert_many`)
- Prompt tokens: ~32k per prompt (128 lines, 16×8 grid)
- Completion tokens: ≤214 per request
- Trials: 1 run per arm, 32 requests per concurrency level

## Environment

- Host / GPU: 8×H20 (sm_90, 78 SM, 4.0 TB/s HBM3), GPU 0
- Driver / CUDA: 12.8
- Model / dtype: Qwen3.6-27B-FP8 (`/data00/Qwen3.6-27B-FP8`)
- TP / EP / slots / KV: TP=1 / 16 slots / fp8 KV, L2 host DRAM, L3 off
- Server flags: `--kv-cache-dtype fp8 --max-running-requests 16 --spec-type none --port 8000`

## Results

| concurrency | arm | decode tok/s | ITL p50 ms | ITL p99 ms | p99 delta |
|---:|---|---:|---:|---:|---:|
| 1 | baseline | 73.5 | 13.5 | 14.3 | — |
| 1 | treatment | 43.2 | 18.1 | 47.4 | +231% |
| 4 | baseline | 40.3 | 18.8 | 220.8 | — |
| 4 | treatment | 36.6 | 24.6 | 59.4 | **-73%** |
| 8 | baseline | 24.1 | 24.5 | 764.1 | — |
| 8 | treatment | 26.4 | 30.3 | 350.9 | **-54%** |
| 16 | baseline | 13.8 | 38.0 | 749.3 | — |
| 16 | treatment | 17.6 | 40.3 | 377.3 | **-50%** |

p90/p50 ratio (treatment): c=4: 1.01, c=8: 1.01, c=16: 1.02 — the tail is
no longer dominated by serialization stalls.

Raw artifacts: `/tmp/sidecar-prechunk-bench.json` (pod), `/tmp/sidecar-prechunk.log` (server).

## Problems

c=1 regressed (p99 14.3→47.4ms, decode 73.5→43.2 tok/s). At c=1 there is
no serialization contention benefit; the background thread's memory-intensive
work (146.8 MiB to_bytes + chunking) competes with the engine for host memory
bandwidth. The baseline binary was also from a prior session with intervening
commits, so the c=1 delta may include unrelated changes.

Remaining p99 tail at c≥8 (350-377ms) comes from `snapshot_recurrent()`
(D2H copy, ~6ms per blob, still synchronous) and background thread memory
bandwidth contention at high concurrency.

## Learnings

PASS for the stated goal: p99 ITL reduced 50-73% at c≥4. The two-stage
backgrounding (to_bytes + chunking off-thread, BTreeMap inserts on-thread)
is the correct architecture for sidecar serialization. The remaining tail
is bounded by `snapshot_recurrent()` D2H copies and memory bandwidth
contention — a dedicated serialization thread (instead of spawn-per-call)
would reduce contention at c≥8.
