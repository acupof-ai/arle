# Infer CUDA profiling wrappers

> Status: Current for `arle serve`.

The repo owns two bench-anchored entrypoints:

- [`scripts/profile_nsys_bench.sh`](../../scripts/profile_nsys_bench.sh) for
  host, CUDA API, copy, kernel, and synchronization timelines.
- [`scripts/profile_ncu_bench.sh`](../../scripts/profile_ncu_bench.sh) for one
  kernel family's occupancy, memory, stalls, and roofline metrics.

Both drive [`scripts/bench_throughput.py`](../../scripts/bench_throughput.py).
The serving result remains the truth; a trace only explains it.

## Prerequisites

Start a release server. Install `httpx`, `curl`, `python3`, `lsof`, and the
relevant profiler (`nsys` or `ncu`). Pass `--server-pid` if port-based PID
resolution is unsuitable.

## Nsight Systems

```bash
scripts/profile_nsys_bench.sh cuda-qwen3 \
  --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-4B \
  --prompts-jsonl data/prompts.jsonl \
  --concurrency-grid 16 \
  --seconds-per-concurrency 20 \
  --delay-seconds 8 \
  --duration-seconds 12
```

It writes `.nsys-rep`, `.sqlite`, kernel/API summaries, environment metadata,
the replay command, and a native benchmark anchor under `bench-output/`.

Use `--bench <dir>` to replay an existing anchor. Use `--dry-run` to verify PID
resolution and commands without Nsight.

## Nsight Compute

Run it only after a timeline names a hotspot:

```bash
scripts/profile_ncu_bench.sh cuda-qwen3 \
  --family attention \
  --url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-4B \
  --prompts-jsonl data/prompts.jsonl \
  --concurrency-grid 16 \
  --seconds-per-concurrency 20 \
  --launch-skip 5 \
  --launch-count 2
```

Families: `attention`, `sampling`, `paged-kv`, `dequant`, `fused-op`. An
explicit `--kernel <regex>` may replace `--family`. The wrapper writes
`.ncu-rep`, its log, metadata, summary, and benchmark anchor.

## Evidence rule

Preserve raw profiles in `bench-output/`; do not commit them. A shipped claim
links the profile to a fixed-concurrency A/B report under `docs/experience/` and
states both profiler-window time and whole-request wall-clock share. See
[`docs/bench-and-trace-spec.md`](../bench-and-trace-spec.md).
