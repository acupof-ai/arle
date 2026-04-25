# Infer Observability v1

Last updated: 2026-04-22

## Goal

Give `infer` one production-grade observability stack that is:

- fast to reach for during a regression
- opinionated about escalation order
- compatible with the repository's existing bench-and-trace workflow
- light enough to leave on in production at the metrics layer
- deep enough to explain both "where did the time go?" and "why is this kernel slow?"

This plan is intentionally biased toward **operator smoothness** over tool
maximalism. The target is not "support every tracer"; the target is:

1. one default bench path
2. one default service-metrics path
3. one default CUDA timeline path
4. one default kernel path
5. one default request-trace path

## Why this plan exists

The repository already has pieces of the right stack, but they are still
separate:

- canonical throughput/latency bench wrapper:
  [`scripts/bench_guidellm.sh`](../../scripts/bench_guidellm.sh)
- human-readable service snapshots around a bench:
  `service_stats_before.txt`, `service_stats_trace.jsonl`,
  `service_stats_after.txt`, `service_stats_trace_summary.md`
- file-backed Chrome / Perfetto-compatible trace output:
  [`infer/src/trace_reporter.rs`](../../infer/src/trace_reporter.rs)
- CUDA request/generation spans in:
  [`infer/src/server_engine.rs`](../../infer/src/server_engine.rs)
- opt-in trace file emission through:
  [`infer/src/main.rs`](../../infer/src/main.rs)
- older CUDA profiling notes in:
  [`docs/resources/profiling-guide.md`](../resources/profiling-guide.md)

What is missing is a single operator story that says:

- which layer to use first
- which command to run
- what artefacts to expect
- when to escalate
- which signals are always-on vs sampled vs one-shot

## Non-goals

This v1 plan does **not** try to:

- replace GPU profilers with OpenTelemetry
- build a custom Perfetto SDK producer before the simpler layers are stable
- make `PyTorch Profiler` a first-class default for `infer`
- record full prompts / full outputs on spans by default
- create an always-on high-cardinality trace firehose

Reasoning:

- `infer` hot path is Rust-native, not torch-native
- `nsys` / `ncu` remain the correct tools for CUDA runtime and kernel work
- prompt / output capture is expensive and often sensitive
- the team needs a stable default workflow more than another instrumentation substrate

## Design principles

### 1. One-way escalation

The happy path should be:

`bench -> metrics -> timeline -> kernel -> request trace`

No branching maze. If the previous layer already explains the issue, stop.

### 2. Bench-anchored only

Every deep profile must hang off a reproducible workload:

- a `guidellm` run
- a direct component bench
- an agent trace replay
- a minimal reproducer command

Orphan traces are not useful enough to keep.

### 3. Default-light, debug-deep

- `/metrics` stays on by default
- `/v1/stats` stays human-readable
- request spans are sampled / level-gated
- `nsys` / `ncu` stay explicit one-shot capture tools

### 4. Perfetto as the viewer, not the first collector

The repository already emits Chrome JSON traces that Perfetto can open.
v1 should standardize around that and around `ui.perfetto.dev`, not introduce
another collection stack first.

### 5. Request-first naming

Names should map to the mental model operators already use:

- enqueue
- waiting
- active
- prefix lookup
- staged readmission
- prefill
- decode
- emit
- cleanup

Not anonymous internal counters that need source-reading every time.

## Proposed stack

### Hard coverage contract

Infer observability v1 is only accepted if these five layers are all present
and explicitly covered:

1. **L0 Bench Anchor**
   Every profile hangs off one bench run, using the existing
   `guidellm + service_stats_trace` workflow as the default truth source.
2. **L1 Metrics**
   `/metrics` must cover at least:
   `running`, `waiting`, `scheduled`, `prefill_tokens`, `decode_tokens`,
   `batch_width`, `step_latency`, `prefix_hit`, `kv_util`,
   `tier_fetch_wait`, `tier_store_wait`.
   `/v1/stats` remains for humans; `/metrics` remains for Prometheus/Grafana.
3. **L2 Timeline**
   The standard CUDA `nsys` window must show:
   `request enqueue -> assign_slots -> prefix lookup/stage -> prefill launch -> decode launch/readback -> cleanup/publish`,
   plus H2D/D2H copies, streams, CUDA API, and kernel timeline.
4. **L3 Kernel**
   `ncu` is run only for hotspot kernels:
   attention, sampling, paged KV, dequant, fused ops.
   The output contract is:
   occupancy, memory throughput, stall reasons, roofline.
5. **L4 Request Trace**
   OTLP spans must cover only the request lifecycle:
   `http -> tokenize -> enqueue -> prefix -> prefill -> decode_loop -> stream_flush -> finish`,
   with sampling and dynamic trace level.

## Implementation status (2026-04-22)

### Landed in-tree

- **L1 Metrics:** `/metrics` now exposes the mandatory v1 concepts and
  `/v1/stats` includes the corresponding human-readable summaries.
- **L2/L3 Profiling entrypoints:** bench-anchored `nsys` / `ncu` wrappers land
  under `scripts/` and are documented as the default CUDA profiling path.
- **L4 Tracing/export plumbing:** file trace + OTLP export config, request
  sampling knobs, and the CUDA request lifecycle spans are wired into the
  runtime path.

### Explicitly unfinished / partial

- **Pending remote validation:** the canonical CUDA `guidellm` regression bench
  and the paired `nsys` / `ncu` captures still need to run on a real CUDA host.
- **Pending OTLP backend validation:** config/export code is present, but an
  end-to-end verification against a live Jaeger/Tempo/OTLP collector has not
  been recorded yet.
- **Partial dynamic control:** the tracing runtime supports changing level and
  sample rate in-process, but v1 does not yet expose an authenticated admin API
  or hot-reload control surface for operators.
- **Partial slow-request escalation:** `slow_request_ms` is parsed and stored in
  the tracing runtime, but v1 does not yet promote an already-running request
  to a higher trace level after observing its latency.
- **Partial non-CUDA lifecycle coverage:** deep lifecycle spans
  (`prefix/prefill/decode_loop`) are implemented on the CUDA scheduler path.
  Metal/CPU currently inherit the HTTP ingress/egress spans but do not yet emit
  equivalent scheduler-phase spans.

## L0 — Bench anchor

**Question answered:** "Is there actually a regression, under a named workload?"

Canonical tools:

- [`scripts/bench_guidellm.sh`](../../scripts/bench_guidellm.sh)
- [`scripts/bench_agent_trace.py`](../../scripts/bench_agent_trace.py)

Required outcome:

- named workload
- exact command
- raw artefacts under `bench-output/`
- wins/errors entry if this is verification-grade

This remains the SSOT for throughput, TTFT, ITL, and queue-shape comparisons.
The default path is the existing `guidellm + service_stats_trace` pair, not a
new ad-hoc profiling driver.

## L1 — Service metrics

**Question answered:** "Is the engine underfilled, queueing, KV-bound, or tier-bound?"

Default surfaces:

- `GET /v1/stats` for human quick-look
- `GET /metrics` for Prometheus / Grafana / alerting

### `/v1/stats` stays intentionally small

It should remain the "curl and read it in one line" surface:

- requests
- active
- waiting
- tokens_out
- kv_util
- TTFT / TPOT summaries

### `/metrics` becomes the operator-grade surface

It should expose stable gauges/counters/histograms for:

- request lifecycle
  - enqueued requests
  - active requests
  - waiting requests
  - cancelled / evicted / completed requests
- scheduler width and progress
  - scheduled decode rows
  - scheduled prefill rows
  - batched tokens this step
  - prefill queue depth
  - running batch width
- latency
  - queue latency
  - prefill latency
  - decode step latency
  - emit / stream flush latency
  - end-to-end request latency
- KV / prefix
  - KV utilization
  - prefix queries
  - prefix hits
  - prefix hit tokens
  - staged readmission count
  - same-slot reuse count
- tiered KV
  - fetch queued / inflight / completed
  - store queued / inflight / completed
  - fetch wait time
  - store wait time
  - readmission success / timeout / cancel

### Minimum mandatory metric set

The plan is not complete unless `/metrics` exposes at least one stable series
for each of the following concepts:

- `running`
- `waiting`
- `scheduled`
- `prefill_tokens`
- `decode_tokens`
- `batch_width`
- `step_latency`
- `prefix_hit`
- `kv_util`
- `tier_fetch_wait`
- `tier_store_wait`

### Metrics UX target

An operator looking at a dashboard should be able to answer:

- are requests piling up because `waiting >> active`?
- is prefix reuse actually happening?
- are we spending time in queue, prefill, decode, or tier fetch?
- is the scheduler failing to widen the active set?

without opening the source tree first.

## L2 — CUDA timeline

**Question answered:** "Across CPU threads, CUDA API, copies, and kernels, where does the time go?"

Default tool:

- `Nsight Systems`

Default capture shape:

- small window
- steady state only
- tied to a named bench
- `--cuda-graph-trace=node`
- NVTX / trace spans visible in timeline

### Standard capture question set

Every CUDA timeline should be able to show:

- host-side enqueue/admission gaps
- scheduler idle holes
- CUDA API sync points
- H2D / D2H copy cost
- stream underutilization
- graph launch vs kernel body time
- whether one request / one slot is monopolizing the tick

### Minimum mandatory timeline chain

Every standard CUDA timeline capture must make the following stage chain
visible in one naming scheme:

- `request enqueue`
- `assign_slots`
- `prefix lookup/stage`
- `prefill launch`
- `decode launch/readback`
- `cleanup/publish`

And it must also include:

- H2D / D2H copy tracks
- stream tracks
- CUDA API tracks
- GPU kernel tracks

### Required timeline labels

The timeline should expose the following request/scheduler slices in one
consistent vocabulary:

- `request.enqueue`
- `request.emit`
- `scheduler.assign_slots`
- `scheduler.prefix_lookup`
- `scheduler.wait_fetch`
- `scheduler.plan_step`
- `scheduler.prefill_launch`
- `scheduler.decode_launch`
- `scheduler.decode_readback`
- `scheduler.cleanup`
- `scheduler.publish_prefix`

Current `fastrace` / generation spans are a useful base, but v1 should make
the scheduler path the first-class timeline vocabulary.

## L3 — Kernel deep dive

**Question answered:** "Why is this specific kernel slow?"

Default tool:

- `Nsight Compute`

Used only after Layer 2 points at a hot kernel or kernel family.

Default targets:

- attention
- sampling
- paged KV
- dequant
- fused MLP / norm / residual kernels

Required outputs:

- occupancy
- memory workload analysis
- roofline position
- stall reasons
- achieved throughput vs expected bottleneck

### Minimum mandatory kernel families

The wrapper and docs must directly support focused `ncu` runs for:

- attention
- sampling
- paged KV
- dequant
- fused ops

This layer should remain explicit and narrow. No shotgun `ncu` runs over the
whole service.

## L4 — Request trace

**Question answered:** "Why did this one request or one cohort behave badly?"

Default tool:

- `OpenTelemetry` exported to a local collector and viewed in Jaeger/Tempo

Scope:

- request lifecycle only
- scheduler/runtime stages
- no GPU kernel replacement

Default root span:

- `infer.request`

Default child spans:

- `http.validate`
- `tokenize`
- `enqueue`
- `prefix.lookup`
- `prefix.stage_wait`
- `prefill`
- `decode_loop`
- `stream_flush`
- `finish`

### Minimum mandatory request lifecycle

Even if internal span names differ slightly, the request-trace surface must
cover this exact lifecycle:

- `http`
- `tokenize`
- `enqueue`
- `prefix`
- `prefill`
- `decode_loop`
- `stream_flush`
- `finish`

Default attributes:

- request id
- session id if present
- model id
- prompt token count
- generated token count
- finish reason
- queue latency
- prefix reuse length
- tier fetch/store involvement

### Sampling and privacy

By default:

- sample only a small fraction of requests
- do not attach full prompt/output payloads
- allow explicit debug mode to record more detail

If content capture is ever enabled, prefer external references over raw payload
on spans.

## Operator UX

The v1 plan lives or dies on this section.

## Default user stories

### A. "I think throughput regressed"

Run one command:

```bash
scripts/bench_guidellm.sh <label> --model <model> --processor <processor>
```

Read:

- `bench-output/<date>-<label>/benchmarks.json`
- `bench-output/<date>-<label>/service_stats_trace_summary.md`

Escalate only if the service trace does not already explain it.

### B. "I know the problem is on CUDA, show me the timeline"

Run one wrapper that owns the canonical `nsys` invocation:

```bash
scripts/profile_nsys_guidellm.sh <label> --bench <bench-output-dir>
```

Expected artefacts:

- `.nsys-rep`
- exported stats
- short markdown summary

The operator should not have to remember `--trace-fork-before-exec=true`,
`--cuda-graph-trace=node`, capture-range syntax, or output directory layout.

### C. "This kernel family is slow"

Run:

```bash
scripts/profile_ncu_guidellm.sh <label> --family attention
```

Expected artefacts:

- `.ncu-rep`
- roofline / occupancy / memory summary

### D. "One request is weird"

Run:

```bash
scripts/observe_local.sh up
INFER_TRACE_LEVEL=important <server command>
```

Then inspect Jaeger/Tempo for the request trace, or use a local replay input.

The important point is that request tracing should have a **single local boot path**
for the collector/viewer, not a page of manual docker commands every time.

## Proposed repo-owned commands

These are the UX surfaces v1 standardizes around. `bench_guidellm.sh`,
`profile_nsys_guidellm.sh`, and `profile_ncu_guidellm.sh` are now the
repo-owned entrypoints for L0/L2/L3.

### Existing

- `scripts/bench_guidellm.sh`
- `scripts/profile_nsys_guidellm.sh`
- `scripts/profile_ncu_guidellm.sh`
- `scripts/bench_agent_trace.py`
- `scripts/start_infer.sh`
- `scripts/start_metal_serve.sh`
- `scripts/capture_metal.sh`

### Planned next

- `scripts/observe_local.sh up|down|status`
- `scripts/export_trace_summary.py <trace>`

The wrappers should own:

- output directory naming
- sidecar metadata files (`command.txt`, `env.txt`, `sha256.txt`)
- profiler flags that are easy to forget
- conversion/export steps
- short summary generation

## Implementation strategy

## Phase 0 — Vocabulary and SSOT

Deliverables:

- lock this plan as the infer-side SSOT
- keep `guidellm` as the bench anchor
- define one stable scheduler/runtime span vocabulary
- define the stable `/metrics` metric names

Exit:

- the team no longer invents one-off names for the same stage in different docs

## Phase 1 — Metrics-first

Deliverables:

- add Prometheus `/metrics` surface to `infer`
- keep `/v1/stats` minimal and human-readable
- wire scheduler / tiered-KV counters and histograms

Exit:

- common regressions can be classified from dashboard + `service_stats_trace` alone

## Phase 2 — Timeline-first CUDA profiling

Deliverables:

- standard `nsys` wrapper for the service path
- standard output folder contract under `bench-output/`
- markdown summary that reports:
  - top kernels
  - top CUDA APIs
  - copies
  - launches per token

Exit:

- "where did the time go?" takes one wrapper invocation, not a hand-built command line

## Phase 3 — Kernel deep dive wrappers

Deliverables:

- standard `ncu` wrapper
- curated section set for attention / KV / sampling work
- summary extraction for occupancy / memory / stall / roofline

Exit:

- engineers do not start `ncu` from scratch for every kernel regression

## Phase 4 — Request tracing

Deliverables:

- local collector + viewer bootstrap
- sampled OTLP export from infer request/scheduler/runtime path
- dynamic trace level and sampling knobs

Exit:

- one bad request can be followed through enqueue, prefix, prefill, decode, and stream flush

## Phase 5 — Perfetto analysis automation

Deliverables:

- treat Perfetto as the standard viewer for Chrome JSON trace output
- add reusable trace summary queries / scripts where worthwhile

Exit:

- timeline artefacts can be explored interactively or summarized automatically

## Explicit technology choices

### Default yes

- `guidellm` for throughput / latency truth
- Prometheus `/metrics`
- `Nsight Systems`
- `Nsight Compute`
- `OpenTelemetry` for request tracing
- Perfetto UI / Trace Processor for analysis and viewing

### Default no

- `PyTorch Profiler` as infer default
- full-payload spans by default
- Perfetto native producer as phase 1
- bespoke internal-only trace viewer

## Acceptance criteria

This plan is successful when all of these become true:

1. A new engineer can answer "throughput regressed, where do I start?" with one command and one artefact directory.
2. A scheduler regression can be classified from `/metrics` and the bench-side service trace before opening `nsys`.
3. A CUDA timeline capture no longer requires remembering profiler flags from memory.
4. Kernel analysis no longer starts from an empty `ncu` command line.
5. One pathological request can be traced end-to-end through request lifecycle spans.
6. Perfetto can open the repo's trace artefacts directly, and trace summaries can be scripted without changing the collector format.

## Smoothness bar

The stack is **not** accepted as "done" if it technically works but still
requires:

- memorizing 10+ profiler flags
- manual output directory setup
- hand-written collector startup steps
- opening three unrelated docs to run one diagnosis flow
- guessing whether to use metrics, traces, or profilers first

The correct operator experience is:

- one default command
- one predictable artefact directory
- one escalation direction
- one vocabulary across metrics, spans, bench notes, and review docs

That is the actual product requirement for infer observability v1.
