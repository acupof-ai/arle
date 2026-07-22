# Centralized Metal startup MLX limits — Metal, 2026-07-22

> Status: pending-remote

## Goal

Confirm unchanged c=1 output tok/s for the canonical Qwen3.6 Metal workload after centralizing startup MLX limit application.

## Hypothesis

The refactor only shares startup setter and logging code; memory, cache, and wired limits still apply in the same order before config and weight loading, so serving metrics should remain within the existing run-to-run envelope.

## Parameters

```bash
python3 scripts/bench_throughput.py \
  --url http://127.0.0.1:8000 \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --prompts-jsonl <checked-unique-qwen36-workload.jsonl> \
  --concurrency-grid 1 \
  --seconds-per-concurrency 120 \
  --max-tokens 256 \
  --seed 20260416 \
  --output bench-output/2026-07-22-metal-startup-mlx-limits/bench
```

- Baseline: archived canonical Metal champion binary; unavailable in this workspace.
- Treatment: commit containing this entry; `--release --no-default-features --features metal,no-cuda,cli`.
- Prompt tokens: pending remote gate.
- Completion tokens: 256 target.
- Trials: pending remote gate.

## Environment

- Local inspection host / GPU: Apple M4 Pro, 20 GPU cores, 48 GiB unified memory, macOS 26.3.1, Metal 4.
- Model / dtype: `mlx-community/Qwen3.6-35B-A3B-4bit`, 19 GiB snapshot present locally; default Metal INT8 KV.
- TP / EP / slots / KV: TP=1, EP=1; runtime-planned slots and KV capacity must be recorded by the remote run.
- Server flags: canonical Metal defaults; record the exact resource-plan line.

## Results

| concurrency | arm | completed | errors | output tok/s | req/s | TTFT p50/p99 ms | ITL p50/p99 ms | delta |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline | pending | pending | pending | pending | pending | pending | — |
| 1 | treatment | pending | pending | pending | pending | pending | pending | pending |

Raw artifacts: pending `bench.json`, `bench.csv`, server log, and pre/post `/v1/stats`.

## Problems

The canonical model is cached locally, but no checked unique serving workload, running canonical server, or archived champion binary is available. Starting an ad hoc heavy run would change the workload fingerprint and cannot produce a valid comparison, so no benchmark was run and no result is inferred.

## Learnings

**pending-remote.** Run the exact c=1 gate above against the archived champion with the same checked workload and server flags. Accept when correctness passes, completed/error envelopes match, and output tok/s plus TTFT/ITL show no regression beyond measured variance.
