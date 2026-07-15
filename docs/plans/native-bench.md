# Native benchmark — canonical performance truth source

> Status: Active

## Decision

`scripts/bench_throughput.py` is the sole end-to-end throughput and latency
benchmark. It drives the OpenAI-compatible streaming API directly and writes
machine-readable JSON and CSV. No external benchmark framework is required.

Component microbenchmarks and profilers explain a result. They do not license a
serving win.

## Canonical contract

Use a checked JSONL workload for published results. Synthetic prompts are smoke
tests only because repeated prefixes can contaminate cache-sensitive results.

```bash
python3 scripts/bench_throughput.py \
  --url http://127.0.0.1:8000 \
  --model <model> \
  --prompts-jsonl <workload.jsonl> \
  --concurrency-grid 1,4,8,16 \
  --seconds-per-concurrency 120 \
  --max-tokens 256 \
  --seed 20260416 \
  --output bench-output/<label>/bench
```

For matched A/B, keep the binary, server flags, model, workload, concurrency,
duration, seed, and machine fixed. Change one treatment variable. Run baseline
and treatment side by side; use at least three trials when the delta is within
5% or variance overlaps it.

Fixed request counts are valid for deterministic short runs:

```bash
--requests-per-concurrency <n>
```

Do not combine this with `--seconds-per-concurrency`.

## Outputs

Each run atomically updates:

- `<output>.json`: parameters, per-request records, aggregates, and `/v1/stats`
  snapshots.
- `<output>.csv`: one aggregate row per concurrency.

The report must preserve the raw files and copy
[`TEMPLATE-bench.md`](../experience/wins/TEMPLATE-bench.md) to a new dated
`wins/` or `errors/` entry. Never overwrite an earlier report.

## Gates

A result counts only when:

1. the warm-up correctness gate passes;
2. every reported point has no incomplete or errored request;
3. non-empty SSE event count equals completion-token usage for valid ITL;
4. actual prompt and completion token distributions match across A/B;
5. `/v1/stats` and the server log show no OOM, retry storm, or silent empty
   output;
6. the wall-clock metric named by the hypothesis improves.

Exact protocol and stop rules live in
[`docs/bench-and-trace-spec.md`](../bench-and-trace-spec.md).
