# Benchmark and trace specification

> Status: Active

This is the reporting and evidence contract for benchmarks, profilers, and
performance claims. The canonical runner is `scripts/bench_throughput.py`; the
report skeleton is [`TEMPLATE-bench.md`](experience/wins/TEMPLATE-bench.md).

## 1. Required report

Every run records these sections:

1. **Goal** — one user-visible wall-clock metric and workload.
2. **Hypothesis** — one mechanism, one treatment, one predicted delta.
3. **Parameters** — exact command, workload, seed, concurrency, duration or
   request count, prompt/output token distribution, and server flags.
4. **Environment** — commit and binary hash, host, GPU, driver, CUDA/Metal,
   model, dtype, TP/EP, slots, and KV configuration.
5. **Results** — raw fixed-concurrency rows, completed/incomplete/error counts,
   output tok/s, req/s, TTFT p50/p99, ITL p50/p99, and A/B delta.
6. **Problems** — correctness failures, OOM, retries, cache contamination,
   profiler gaps, and other confounders.
7. **Learnings** — PASS, KILL, or pending-remote; measured number; next wall.

Missing one section means the run does not count.

## 2. Goals

Use exactly one primary goal per run:

- **Latency:** TTFT or ITL at a fixed concurrency and request shape.
- **Throughput:** output tok/s at fixed concurrency, or saturation throughput
  under an explicit SLO.
- **Memory:** peak device/host bytes for a fixed workload.
- **Correctness:** decoded outputs and request status for the failing slice.
- **Diagnosis:** wall-clock attribution of a named stage.

A component kernel speedup is diagnosis, not an end-to-end serving win.

## 3. Canonical benchmark

Published results use a checked JSONL workload:

```bash
python3 scripts/bench_throughput.py \
  --url http://127.0.0.1:8000 \
  --model <model> \
  --prompts-jsonl <workload.jsonl> \
  --concurrency-grid 1,4,8,16 \
  --seconds-per-concurrency 120 \
  --max-tokens <n> \
  --seed 20260416 \
  --output bench-output/<label>/bench
```

`--requests-per-concurrency` may replace the duration for short deterministic
runs. Synthetic prompts are smoke-only. They may repeat prefixes and therefore
cannot license cache-sensitive changes.

The JSON artifact is the source of truth. CSV is a view. Preserve both.

### 3.1 Matched A/B

Keep fixed:

- checkout, binary, machine, GPU clocks, and server lifecycle;
- model, dtype, TP/EP, slots, KV settings, and all unrelated flags;
- JSONL workload, request order, concurrency, duration, output cap, and seed.

Change one treatment variable. Run baseline and treatment from the same shell,
side by side. If the observed delta is below 5% or overlaps run variance, use
at least three trials per arm and report median plus range.

### 3.2 Scheduling envelope

For every point record:

- offered concurrency and completed request count;
- prompt and completion token min/p50/max;
- active, queued, retry, prefix-hit, and KV residency counters from `/v1/stats`;
- incomplete, errored, timed-out, or empty responses.

If these differ materially across A/B, the comparison is invalid until
explained.

## 4. Correctness gate

Before timing:

1. run one non-degenerate prompt through the same endpoint and flags;
2. require non-empty, completed, non-repeating output;
3. inspect decoded outputs on every failing slice;
4. confirm request errors and timeouts are not counted as valid samples.

For a new kernel, quantization path, rollback path, or speculative decoder, also
run the model-specific correctness gate. Token identity against another kernel
is not required when MoE non-determinism applies; coherent autoregressive output
and the model-specific gate are required.

## 5. Server and cache hygiene

- Run one benchmark process at a time.
- Start each arm from the declared cache state. For cache-independent claims,
  use unique prompts or restart and prove zero prefix hits.
- Do not reuse a server after OOM, assertion, NCCL failure, or allocator
  corruption.
- Capture server logs and `/v1/stats` before and after every point.
- A 200 response with empty output, missing usage, incomplete stream, or retry
  exhaustion is a failure, not throughput.
- ITL is valid only when each non-empty SSE event carries one completion token;
  the runner rejects event/token count mismatches instead of reporting chunk latency.
- Cold-load time is separate from request TTFT. Report it independently.

## 6. Duration and stop rules

Use 120 seconds per concurrency by default. A shorter run counts only when it
completes enough work for stable medians and the report states why. Extend or
repeat when:

- the first and second halves differ by more than 5%;
- queue depth, prefix-hit rate, or memory residency is still moving;
- the treatment delta is within normal variance;
- fewer than 20 completed requests contribute to a reported point.

Stop and fix the harness or server when any request is incomplete, errored, or
empty. Do not average through failures. Stop exploring a treatment after a
controlled wall-clock KILL unless a new measurement changes the mechanism.

## 7. Tracing

Use traces to attribute an end-to-end result:

```bash
scripts/profile_nsys_bench.sh <label> --model <model> \
  --prompts-jsonl <workload.jsonl> --concurrency-grid 4 \
  --seconds-per-concurrency 60
scripts/profile_ncu_bench.sh <label> --family <kernel-family> --model <model> \
  --prompts-jsonl <workload.jsonl> --concurrency-grid 4 \
  --seconds-per-concurrency 60
```

Every trace reports both:

- time inside the selected NVTX or kernel window;
- that time divided by per-request or whole-run wall clock.

Host/GPU overlap claims require timestamps on the same timeline: request start,
host enqueue begin/end, CUDA event or kernel begin/end, synchronization, and
response completion. A visual overlap alone is not proof. Export the timeline
or SQLite rows used for the calculation.

Nsight Compute is a kernel microbenchmark. Pair it with the fixed-concurrency
native benchmark before claiming serving impact.

## 8. Licensing

A performance change is licensed only when:

1. correctness passes;
2. the matched A/B scheduling envelope aligns;
3. the named wall-clock metric improves on the target workload;
4. no TTFT, ITL, throughput, memory, or failure-rate regression violates the
   stated SLO;
5. raw JSON/CSV, logs, and any profiler artifacts are preserved;
6. a dated `wins/` or `errors/` entry links the baseline and treatment.

Use the conservative framing when component and wall-clock percentages disagree.
The per-request result wins.

## 9. Repository lifecycle

Runtime and benchmark-parameter changes require a dated report under
`docs/experience/wins/` or `docs/experience/errors/`. If hardware is unavailable,
commit a `pending-remote` report naming the exact remote gate. Documentation and
dev-only tooling are exempt; state the exemption in the commit body.

Never overwrite a report. An after report links its baseline and reports the
delta. Update CHANGELOG on a phase exit, default flip, or license-or-kill verdict.
