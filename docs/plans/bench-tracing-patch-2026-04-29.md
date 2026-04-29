# Bench tracing instrumentation patch plan — 2026-04-29

Output of a research subagent pass on `scripts/bench_guidellm.sh`.
Identifies five patch points (A-E) that close the gap between what
guidellm + `/v1/stats` already collect and what the bench wrapper
emits in `headline_table.md`.

## Findings — what we collect but don't surface

### `benchmarks.json` (guidellm side) — keys we ignore

| metric | available | currently in headline? |
|---|---|---|
| `time_to_first_token_ms` | mean, std_dev, min, max, p25/p50/p75/p90/p95/p99/p999 | only p50/p99 |
| `inter_token_latency_ms` | full distribution | only p50/p99 — missing mean, std, p95, max |
| `time_per_output_token_ms` | full distribution | not surfaced (TPOT vs ITL distinction) |
| `request_latency` | E2E full distribution | not surfaced |
| `request_concurrency` | mean+percentiles (peak in-flight) | not surfaced |
| `output_tokens_per_second` | mean used | std/p95 unused |
| `prompt_tokens_per_second` | mean | not surfaced (input throughput) |
| `tokens_per_second` | total throughput | not surfaced |
| `prompt_token_count` / `output_token_count` / `total_token_count` | mean+sum | only used in K6 OOM detector |
| `request_streaming_iterations_count` | mean | only K6 detector |
| `request_totals` | successful/errored/incomplete | partially used |
| `duration` (per benchmark) | float seconds | not surfaced |

### `/v1/stats` — fields traced to JSONL but ignored in summary

Emitted by `infer/src/metrics/render.rs:730`, polled by
`scripts/bench_guidellm.sh:466-477` every 1 s:

- Used: `waiting`, `active`, `running_batch`, `prefill_queue`,
  `kv_util`.
- **Unused** (already in `service_stats_trace.jsonl`):
  `prefix_hit_rate`, `peak_mem`, `active_mem`, `cache_mem`,
  `queue_p50`, `ttft_p50`, `ttft_p99`, `tpot_p50`, `service_p50`,
  `step_last`, `step_p50`, `kv_fetch_q`, `kv_fetch_waiters`,
  `kv_store_q`, `prefix_skip_rate`, `tier_fetch_wait`,
  `tier_store_wait`, `decode_tokens`, `prefill_tokens`, `tokens_out`.

The summary shows only "peak" of 5 fields — no quartiles, no
evolution, no steady-state vs cold-start delta.

### Server log `step breakdown:` lines

Emitted by `infer/src/scheduler/cuda/execution.rs:457-468` when
`total_us > 100ms`. Format:

```
step breakdown: plan={label} admission= decode= emit= prefill= total= batch=
```

**No automatic aggregation today**. The bench wrapper has no path to
the server log. Server-side EMAs already exist at
`execution.rs:449-455` (`step_timing_decode_us`,
`step_timing_prefill_us`, `step_timing_emit_us`,
`step_timing_total_us`) but are NOT plumbed into `/v1/stats`.

## Patches (numbered for landing order)

### Patch E — expose step-phase EMAs in `/v1/stats`

**Server side, prerequisite for D and a cheaper alternative to C.**

`infer/src/metrics/render.rs:730` (the per-second stats line),
append:

```rust
format!(" step_phase_us=adm:{:.0},prefill:{:.0},decode:{:.0},emit:{:.0},total:{:.0}",
        self.scheduler_step_admission_us(),
        self.scheduler_step_prefill_us(),
        self.scheduler_step_decode_us(),
        self.scheduler_step_emit_us(),
        self.scheduler_step_total_us()),
```

Plumb the four existing EMAs through `MetricsInner` getters.
`admission_us` is NOT yet an EMA — add `step_timing_admission_us`
next to the others at `execution.rs:449`. After this patch, all
step-phase data flows through the existing trace JSONL — no log
parsing needed downstream.

### Patch A — extend `extract_rows` jq filter

`scripts/bench_guidellm.sh:680-714`. Replace the body:

```diff
 extract_rows() {
     jq -r '
         def pctl($m): (.metrics[$m].successful.percentiles // {});
         def avg($m):  (.metrics[$m].successful.mean        // null);
         def std($m):  (.metrics[$m].successful.std_dev     // null);
         def mx($m):   (.metrics[$m].successful.max         // null);
+        def cnt($m):  (.metrics[$m].successful.count       // 0);
+        def tot($m):  (.metrics[$m].successful.total_sum   // null);
         def rnd(d): if . == null then "n/a" else (.*pow(10;d)|round/pow(10;d)) end;
         .benchmarks
         | map({
             rate:        ( ... ),                                # unchanged
             ttft_mean:   (avg("time_to_first_token_ms")        | rnd(1)),
             ttft_std:    (std("time_to_first_token_ms")        | rnd(1)),
             ttft_p50:    (pctl("time_to_first_token_ms").p50   | rnd(1)),
             ttft_p99:    (pctl("time_to_first_token_ms").p99   | rnd(1)),
             tpot_mean:   (avg("time_per_output_token_ms")      | rnd(2)),
             itl_mean:    (avg("inter_token_latency_ms")        | rnd(2)),
             itl_std:     (std("inter_token_latency_ms")        | rnd(2)),
             itl_p50:     (pctl("inter_token_latency_ms").p50   | rnd(2)),
             itl_p95:     (pctl("inter_token_latency_ms").p95   | rnd(2)),
             itl_p99:     (pctl("inter_token_latency_ms").p99   | rnd(2)),
             itl_max:     (mx("inter_token_latency_ms")         | rnd(2)),
             e2e_mean:    (avg("request_latency")               | rnd(2)),
             e2e_p99:     (pctl("request_latency").p99          | rnd(2)),
             concurrency: (pctl("request_concurrency").p50      | rnd(1)),
             out_tok_s:   (avg("output_tokens_per_second")      | rnd(2)),
             total_tok_s: (avg("tokens_per_second")             | rnd(2)),
             in_tok_s:    (avg("prompt_tokens_per_second")      | rnd(2)),
             total_in:    (tot("prompt_token_count")),
             total_out:   (tot("output_token_count")),
             req_s:       (avg("requests_per_second")           | rnd(3))
           })
         | .[] | "| \(.rate) | \(.ttft_mean) | \(.ttft_p50) | \(.ttft_p99) | \(.tpot_mean) | \(.itl_mean) | \(.itl_p95) | \(.itl_p99) | \(.itl_max) | \(.e2e_p99) | \(.concurrency) | \(.out_tok_s) | \(.total_tok_s) | \(.total_in) | \(.total_out) | \(.req_s) |"
     ' "$JSON_FILE" 2>/dev/null || true
 }
```

Also update `emit_header` (`bench_guidellm.sh:675`) to match the
new column count.

### Patch B — trace-summary quartiles + ignored fields

`scripts/bench_guidellm.sh:466-501`
(`write_service_stats_trace_summary`). Add quartile helper, parse
the unused fields, emit them in the markdown:

- `kv_util` quartiles (q25/q50/q75/q99) — not just peak.
- `waiting` quartiles — admission queue distribution.
- `prefix_hit_rate` peak + steady-state q75.
- `peak_mem` peak + Δ vs `before`.
- `ttft_p99` (server-side, distinct from guidellm-measured).
- `kv_fetch_q` saturation (depth > 0 sample count).
- `step_last` quartiles (server's per-step wall-clock).

### Patch D — new headline-table sections

`scripts/bench_guidellm.sh:743-751` (the `{ ... } > "$TABLE_FILE"`
block). Append:

```diff
 {
   emit_header
   rows="$(extract_rows)"
   ...
+  printf '\n## Per-step phase timing (from /v1/stats EMAs after patch E)\n\n'
+  printf '| plan | n | total p50 | prefill avg | decode avg |\n|---|---|---|---|---|\n'
+  emit_step_phase_summary           # awk over service_stats_trace.jsonl
+  printf '\n## Service trace distribution\n\n'
+  printf '| metric | q25 | q50 | q75 | q99 | peak |\n|---|---|---|---|---|---|\n'
+  emit_trace_distribution_rows
+  printf '\n## Aggregate prefill/decode throughput\n\n'
+  emit_aggregate_throughput
 } > "$TABLE_FILE"
```

`emit_aggregate_throughput`: `prefill: SUM(prefill_tokens) /
SUM(prefill_us) tok/s`, `decode: SUM(decode_tokens) /
SUM(decode_us) tok/s`. Numbers come from `/v1/stats` so
`tokens_per_second_input/output` aggregates mean over the bench
window.

### Patch C — fallback server-log parser

Only if Patch E proves insufficient. Add `--server-log PATH` flag
to bench wrapper, awk-parse `step breakdown:` lines, emit per-plan
distribution. See research output for full sketch.

## Recommended landing order

1. **Patch E first** (server-side: expose step phase EMAs in
   `/v1/stats`) — unlocks D without log parsing.
2. **Patch A** (jq extract_rows expansion) — pure bench-side, low
   risk.
3. **Patch B** (trace summary quartiles + ignored fields).
4. **Patch D** (new headline-table sections, fed from A+B+E).
5. **Patch C** (server-log parser) only if E proves insufficient —
   keep as fallback.

Each patch is independently testable against
`bench-output/2026-04-28-cuda-l4-dedup-fixed-fp8-r1/` (16
successful, 8 incomplete — exercises non-trivial percentiles).

## Files

- `/content/workspace/agent-infer/scripts/bench_guidellm.sh`
  (lines 466-501, 675-714, 720-741, 743-751)
- `/content/workspace/agent-infer/infer/src/metrics/render.rs`
  (lines 625-730)
- `/content/workspace/agent-infer/infer/src/scheduler/cuda/execution.rs`
  (lines 440-468)
- `/content/workspace/agent-infer/infer/src/http_server/handlers.rs`
  (lines 785-801)
- `/content/workspace/agent-infer/.venv/lib/python3.12/site-packages/sglang/bench_serving.py`
  (lines 875-905) — reference shape
