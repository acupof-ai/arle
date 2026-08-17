# Observability Infrastructure — Design

Date: 2026-08-17
Status: accepted for implementation (tranches T1–T3; T4 deferred)

## 1. Architecture

### 1.1 Decisions

**Embedded store + built-in dashboard, keeping `/metrics`.** ARLE runs on
air-gapped boxes (H20 pods) where deploying Prometheus + Grafana is not free,
and the requirement is long-term retention with zero hot-path cost. The store is
therefore embedded: day-segmented append-only files written by a background
task in `infer-server`, queried by handlers in the same process, rendered by a
single self-contained HTML page served at `/dashboard`. The existing
`/metrics` Prometheus endpoint stays and is extended — external Grafana users
keep working, and both surfaces read the same `CounterSnapshot`, so every
instrument is written once.

**New crate `infer-observe`** (storage + sample types + query/bucketing, deps:
`serde`, `postcard` only) plus a new module `crates/infer-server/src/observe.rs`
(sampler task, HTTP handlers, dashboard route). The crate is server-free so a
future `arle observe export` CLI can link it without axum. No sqlite, no NVML
bindings, no metrics/tracing crates — all absent from the workspace today.

**Sampling model: cumulative counters, rates derived at query time** (the
Prometheus model). The 10 s sample stores the cumulative `CounterSnapshot`
counters as-is; the query handler derives rates (decode tok/s, hit rate, busy
%) from counter deltas between samples. This keeps the sampler trivial and
makes counter resets (restart) detectable as a negative delta.

**GPU data: backend-owned background thread, plain struct across the seam.**
CUDA spawns a sampler thread at executor construction that shells out to
`nvidia-smi` (all GPUs, one subprocess, every 2 s — the established precedent
at `crates/infer-cuda/src/numa_pin.rs:39` and `crates/cli/src/hardware.rs:14`,
zero new dependencies) and publishes into executor atomics. The engine thread
reads those atomics inside the existing `stats()` call in `publish_counters`.
The seam gains `GpuSample` (plain `Copy` struct) and
`BackendExecutor::gpu_sample()` (default `None`). Metal returns `None`; unified
memory is covered by the host RAM sampler.

**Host data: `sysinfo` in the observe task.** Already a workspace dependency
(`crates/cli`); adding it to `infer-server` is a Cargo.toml line, not a new
external dep. One `System::refresh` per 10 s on the background task.

### 1.2 Data flow

```
engine thread (hot path)                background threads              HTTP
─────────────────────────               ───────────────────             ────
ThroughputStats / KvSystemMetrics       CUDA gpu-sample thread          GET /metrics
  plain u64 fields, &mut self      ──►   nvidia-smi → atomics    ──►      (Prometheus,
  (infer-core)                           (infer-cuda, rank 0)            extended)
       │                                                               GET /v1/stats
       │ per-tick (existing)                                           (JSON, extended)
       ▼
publish_counters (execution.rs:73) ──► Arc<Mutex<CounterSnapshot>> ──► GET /v1/observe/query
       │                                     ▲                          GET /v1/observe/latest
       │                                     │ read every 10 s          GET /dashboard
       │                          observe task (observe.rs, tokio)
       │                            ├─ sysinfo refresh (CPU/RAM/disk)
       │                            ├─ HTTP counters (AtomicU64s in DpCoordinator)
       │                            └─ append Sample → day segment file + fdatasync
       ▼
day segments: $ARLE_OBSERVE_DIR/observe-YYYY-MM-DD.bin  (30-day retention)
```

Nothing on the engine thread changes cadence: `publish_counters` already runs
every tick and already calls `engine.backend_stats()`. The observe task is a
pure consumer of the published snapshot.

## 2. Instrumentation points

All new engine-side counters are plain `u64` fields mutated with
`saturating_add` on the single engine thread — the `ThroughputStats` pattern at
`crates/infer-core/src/lib.rs:157`. No atomics, locks, syscalls, or device
reads are added to the hot path.

### 2.1 Engine loop (infer-core, infer-server)

| # | Site | Metric | Implementation |
|---|------|--------|----------------|
| E1 | `crates/infer-server/src/execution.rs:315` | `step_micros_total`, `step_count` (whole-step wall incl. host phases) | Make `step_start` unconditional (drop `trace_submit.then`); after `engine.step()` returns, call a new `Engine::record_step_wall(&mut self, micros)` (lib.rs, next to `throughput_stats()` at :1032) doing two `saturating_add`s. Covers all 6 step exits because the bracket is in the caller. |
| E2 | `crates/infer-core/src/lib.rs:932` (submit site, `plan` in scope) | `decode_batch_sum`, `prefill_chunk_sum` | Two field reads on the already-built `ForwardPlan` + two `saturating_add`s. Avg decode batch = `decode_batch_sum / (decode_forward_steps + mixed_forward_steps)` (denominators already exist). |
| E3 | `crates/infer-core/src/lib.rs:1708` (`try_admit_front_waiter`, `active.insert`) | `requests_admitted` | One `saturating_add` inside the existing insert path. Closes the gap where only lookups/completions are counted. Per-rank (admission is lockstep-broadcast). |
| E4 | `crates/infer-core/src/lib.rs:1242` (Prefilling→Decoding seal) | `requests_prefill_sealed` | One `saturating_add` inside the existing transition branch. |
| E5 | `crates/infer-core/src/lib.rs:832` | `forward_busy_micros` (existing) | Reuse; the GPU-busy component of step wall. No change. |

New `ThroughputStats` fields: `step_micros_total`, `step_count`,
`decode_batch_sum`, `prefill_chunk_sum`, `requests_admitted`,
`requests_prefill_sealed`. The struct is `Copy` and flows to `/v1/stats` and
`/metrics` via the existing snapshot path.

### 2.2 Prefix / radix cache (infer-core)

| # | Site | Metric | Implementation |
|---|------|--------|----------------|
| C1 | `crates/infer-core/src/prefix.rs:570` (`evict_prefix_cache_for_pages`) | `evicted_pages_severed`, `evicted_pages_demoted` | Method is `&mut self`: accumulate internally — sever arm at :610, demoted count from the `try_demote_pages` arm at :735. Callers untouched. Closes gap (1): eviction is currently silent. |
| C2 | `crates/infer-core/src/lib.rs:1597` (admission probe) | `prefix_probes_total` | One `saturating_add` per probe. Closes gap (3): the admission-time probe is uncounted today (lookups only counts attaches). |
| C3 | `crates/infer-core/src/prefix.rs:454` (`publish_prefix_blocks`) | `prefix_deduped_pages` | `offered - newly_cached.len()` at :461, one `saturating_add`. Closes gap (4). |
| C4 | `crates/infer-core/src/prefix.rs:645` (`invalidate_prefix_cache`) | `prefix_invalidations` | One `saturating_add` on a rare path. Closes gap (5). |

New fields: `PrefixCacheStats.probes`, `PrefixCacheStats.deduped_pages`,
`PrefixCacheStats.invalidations`; `KvSystemMetrics.evicted_pages_severed`,
`KvSystemMetrics.evicted_pages_demoted`. `reuse_miss` semantics stay as-is
(documented: attach-restore with zero restored) — no conflation fix attempted,
per minimal-change.

### 2.3 Executor (infer-cuda, below the seam)

| # | Site | Metric | Implementation |
|---|------|--------|----------------|
| X1 | new `crates/infer-cuda/src/gpu_sample.rs` | per-GPU `util_pct`, `memory_used_mb`, `memory_total_mb`, `temp_c`, `power_w` | Sampler thread: `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits` every 2 s; parse; store into `AtomicU32`s held by the executor. Thread is detached, owns no engine state, never returns errors into the engine. On spawn failure or parse failure: retain last good values + set `stale_ms` timestamp. |
| X2 | `crates/infer-cuda/src/executor/qwen35.rs:604` (`from_qwen35_safetensors`) | — | Spawn the thread at construction. Rank-0 only under TP (`INFER_TP_RANK`, `stage_profile.rs:125` precedent); the single nvidia-smi call queries all 8 devices, so rank 0 sees the whole node. |
| X3 | `crates/infer-seam/src/lib.rs` (BackendStats at :258) | `gpu: Option<GpuSample>` | New plain structs: `GpuSample { devices: [GpuDeviceSample; 8], device_count: u8, sampled_at_ms: u64, stale: bool }`, `GpuDeviceSample { gpu_index: u8, util_pct: u8, memory_used_mb: u32, memory_total_mb: u32, temp_c: u8, power_w: u16 }` — both `Copy`. `BackendStats.gpu` defaults `None`. |
| X4 | executor `stats()` | fills `gpu` | Reads the atomics (8 devices × 5 loads ≈ 40 ns, no allocation — fixed array). |
| X5 | `crates/infer-server/src/execution.rs:73` (`publish_counters`) | `snap.gpu = stats.gpu` | One field copy per tick. |

Metal: `gpu_sample()` default `None`. Unified-memory pressure shows up in the
host RAM sampler (process RSS + MLX wired limit). A Metal-specific sampler is
T4.

### 2.4 Server / HTTP edge (infer-server)

Engine-side `submitted_at` starts at admission, so edge queueing is invisible
without an HTTP-arrival anchor. All of these run on tokio/frontend threads —
off the engine hot path.

| # | Site | Metric | Implementation |
|---|------|--------|----------------|
| S1 | `crates/infer-server/src/coordinator.rs:647` (`streaming_submit`) | arrival anchor | Add `arrived_at_ms: u64` (UNIX millis) to `WireRequest` (`multiproc_relay.rs:664`), set once at ingress. Rank-symmetric (broadcast with the request). |
| S2 | `crates/infer-server/src/coordinator.rs:953` (first non-empty delta) | `edge_ttft_micros_total`, `edge_ttft_count` | One `Instant`-style diff in the stream task; `Relaxed` fetch_add into `DpCoordinator` http counters. |
| S3 | stream-task end + `submit_and_collect` end | `http_requests_total`, `http_request_micros_total`, `http_errors_total` | Count at terminal delta / error conversion. Error class split: `http_errors_4xx_total`, `http_errors_5xx_total` at the `ApiError` conversion site. |
| S4 | `crates/infer-server/src/coordinator.rs:1866` (metrics scrape 2 s timeout → zero snapshot) | `scrape_timeouts_total` | One `AtomicU64` in `DpCoordinator`. |
| S5 | new `crates/infer-server/src/observe.rs` | — | The sampler task (§3.4) + handlers `GET /v1/observe/query`, `GET /v1/observe/latest`, `GET /dashboard`. Routes registered in `build_router` (`coordinator.rs:392`), which both the multiproc and in-process routers share. |

HTTP counters are a `HttpEdgeCounters` struct of `AtomicU64`s behind `Arc` in
`DpCoordinator`. They do **not** enter `CounterSnapshot`; the observe task
reads them directly when building a sample.

## 3. Hot-path overhead analysis

Reference scale: a decode step is ~35 ms at 28 tok/s (27B anchor). Budget for
new instrumentation: <1 µs/step aggregate (<3×10⁻⁵ of step wall).

| Point | Cost | Frequency | Share of 35 ms step |
|-------|------|-----------|---------------------|
| E1 step wall | 2 × `Instant::now` ≈ 50 ns + 2 saturating adds ≈ 2 ns | per step | 1.5×10⁻⁶ |
| E2 plan shape | 2 field reads + 2 adds ≈ 2 ns | per step | 6×10⁻⁸ |
| E3 admitted | 1 add ≈ 1 ns | per admission (rare vs steps) | — |
| E4 prefill sealed | 1 add ≈ 1 ns | per request once | — |
| C1 eviction | adds inside an existing O(victims) loop | only under memory pressure | 0 (off steady-state decode) |
| C2 probe | 1 add ≈ 1 ns | per admission probe | — |
| C3/C4 | 1 add each | per publish finish / rare | — |
| X4 gpu atomics read | 40 fixed-array loads ≈ 40 ns | per tick inside existing `stats()` | 1.1×10⁻⁶ |
| X5 snapshot field | 1 struct copy ≈ 32 B | per tick | — |
| S1–S4 | atomics on tokio threads | per HTTP request | 0 (off engine thread) |

**Total added to the decode step: ≈100 ns** — two orders of magnitude under
the 1 µs budget, and 3.5×10⁻⁶ of step wall.

Proof of no sync / allocation / lock in the hot loop:

- No CUDA events, no `stream.synchronize`, no `mem_get_info`, no D2H added.
  The only sanctioned sync remains the sampling D2H at
  `crates/infer-cuda/src/executor.rs:1102`. GPU time comes from a thread that
  is not the engine thread.
- `GpuSample` is a fixed `[GpuDeviceSample; 8]` array — `stats()` allocates
  nothing. `CounterSnapshot` gains one `Copy` field; the per-tick clone grows
  by 32 B.
- No new mutex on the engine thread. `publish_counters` already holds the
  snapshot mutex once per tick; new fields ride that existing lock.
- `Instant::now` appears exactly once per step (E1), matching the
  unconditional-busy-micros precedent (`lib.rs:832`), not the env-gated
  phase-timing precedent (`ARLE_STEP_PHASE`), because 50 ns/step needs no gate.
- nvidia-smi (~50–150 ms subprocess) and sysinfo refresh (~5–20 ms) run on
  background threads at 2 s / 10 s cadence. Worst case they contend for CPU
  with the engine thread on a core-constrained box; mitigated by pinning the
  sampler thread to a non-engine core where `numa_pin` already runs (T4
  nicety), and by the 10 s cadence making even a 150 ms blip a 1.5 % duty
  cycle on one core.

## 4. Storage schema

### 4.1 Sample record

One `Sample` per 10 s, serialized with `postcard` (varint, ~250–350 bytes):

```rust
// crates/infer-observe/src/sample.rs
pub struct Sample {
    pub session_id: u128,          // uuid v4 per process; separates counter epochs
    pub model: String,             // served model id
    pub ts_ms: u64,                // UNIX millis, wall clock
    // gauges (instantaneous)
    pub active_requests: u32,
    pub queue_depth: u32,
    pub kv_free_pages: u32,
    pub cached_pages: u32,
    pub gpu: Option<GpuSample>,    // reused from infer-seam (serde-derived)
    pub cpu_pct: u8,               // system-wide avg over interval
    pub ram_used_mb: u32,
    pub ram_total_mb: u32,
    pub disk_pct: u8,              // model/KV-tier mount
    // cumulative counters since engine start (rates derived at query time)
    pub steps: u64,
    pub prefill_tokens: u64,
    pub generated_tokens: u64,
    pub requests_admitted: u64,
    pub requests_prefill_sealed: u64,
    pub requests_completed: u64,
    pub requests_succeeded: u64,
    pub requests_failed: u64,
    pub step_micros_total: u64,
    pub step_count: u64,
    pub forward_busy_micros: u64,
    pub prefill_forward_steps: u64,  pub prefill_forward_busy_micros: u64,
    pub decode_forward_steps: u64,   pub decode_forward_busy_micros: u64,
    pub mixed_forward_steps: u64,    pub mixed_forward_busy_micros: u64,
    pub decode_batch_sum: u64,
    pub prefill_chunk_sum: u64,
    pub ttft_micros_total: u64,      pub ttft_count: u64,
    pub tpot_micros_total: u64,      pub tpot_count: u64,
    pub e2e_micros_total: u64,       pub e2e_count: u64,
    pub prefix_lookups: u64,  pub prefix_hits: u64,
    pub prefix_hit_tokens: u64, pub prefix_probes: u64,
    pub prefix_deduped_pages: u64, pub prefix_invalidations: u64,
    pub evicted_pages_severed: u64, pub evicted_pages_demoted: u64,
    pub spec_accept_tokens: u64,    pub spec_draft_tokens: u64,  // from SpecDecodeStats
    pub http_requests: u64,  pub http_errors_4xx: u64,  pub http_errors_5xx: u64,
    pub edge_ttft_micros_total: u64, pub edge_ttft_count: u64,
    pub http_request_micros_total: u64,
    pub scrape_timeouts: u64,
}
```

The sample is built by copying out of `CounterSnapshot` (already a clone) +
`HttpEdgeCounters` + sysinfo. Building it takes ~1 µs on the background task.

### 4.2 Segment file format

```
$ARLE_OBSERVE_DIR/observe-YYYY-MM-DD.bin   (UTC day)

file    = magic %s"ARLEOBS1\n" record*
record  = len:u32-le  payload:[len; postcard<Sample>]
```

- Append-only; one `fdatasync` per record (10 s cadence — negligible).
- Reader validates `len <= 1 MiB` and skips records that fail postcard decode
  (torn tail after a crash). No index needed: a day file is ~2.6 MB
  (8,640 samples × ~300 B); a 24 h query scans it in <10 ms.
- Retention: at startup and once per day in the observe task, delete segment
  files older than `ARLE_OBSERVE_RETENTION_DAYS` (default 30). 30-day footprint
  ≈ 78 MB.
- Config: `ARLE_OBSERVE_DIR` (default `./observe-data`),
  `ARLE_OBSERVE_INTERVAL_SECS` (default 10), `ARLE_OBSERVE_RETENTION_DAYS`
  (default 30). Follows the `ARLE_*` observability env convention
  (`ARLE_STEP_PHASE`, `ARLE_NVTX`, …).
- Disk-full / write-error policy: log once per 100 failures, keep serving,
  retry next interval. The store must never kill the server.

### 4.3 Query API and derived metrics

`GET /v1/observe/latest` → last sample (JSON).
`GET /v1/observe/query?range=1h|24h|7d|30d[&bucket=10s|5m|30m]` → bucketed
series, capped at ~1,500 points (server picks bucket from range: 1 h→10 s,
24 h→5 m, 7 d→30 m, 30 d→30 m).

Per bucket, gauges = mean; rates derived from cumulative-counter deltas:

| Derived series | Formula |
|----------------|---------|
| `decode_tok_s` | Δgenerated_tokens / Δt |
| `prefill_tok_s` | Δprefill_tokens / Δt |
| `step_latency_avg_ms` | Δstep_micros_total / Δstep_count / 1000 |
| `ttft_avg_ms` | Δttft_micros_total / Δttft_count / 1000 |
| `tpot_avg_ms` | Δtpot_micros_total / Δtpot_count / 1000 |
| `forward_busy_pct` | Δforward_busy_micros / Δwall_micros |
| `decode_batch_avg` | Δdecode_batch_sum / Δ(decode+mixed forward steps) |
| `cache_hit_rate` | Δprefix_hits / Δprefix_lookups |
| `edge_ttft_avg_ms` | Δedge_ttft_micros_total / Δedge_ttft_count / 1000 |
| `http_rps` / `http_error_rate` | Δhttp_requests / Δt ; Δerrors / Δrequests |
| `gpu_util_pct` | mean over devices of `util_pct` |

Counter reset (process restart, new `session_id` or any negative delta): the
bucket straddling the reset reports rates as `null`; gauges continue.

## 5. Dashboard

Single HTML file embedded with `include_str!` at
`crates/infer-server/src/dashboard.html`, served at `GET /dashboard` (redirect
from `/dashboard/`). Vanilla JS + inline SVG, no build step, no CDN (air-gapped
pods). Auto-refresh 10 s; range selector 1 h / 24 h / 7 d / 30 d; hover
tooltip showing exact values.

Layout:

- **Stat tiles (top, from `/latest`):** decode tok/s, prefill tok/s, active
  requests, queue depth, GPU util (avg), VRAM used/total, RAM used, TTFT avg,
  step latency avg, cache hit rate.
- **Charts (2-column grid):**
  1. GPU util % — one line per device (up to 8)
  2. VRAM used MB per device (stacked area) + total
  3. CPU % and RAM used/total
  4. Disk % of model/KV-tier mount
  5. Decode tok/s and prefill tok/s (dual axis)
  6. Active requests + queue depth
  7. TTFT avg, edge TTFT avg, TPOT avg (ms)
  8. Step latency avg + forward busy %
  9. Prefix cache hit rate + cached pages + KV free pages
  10. HTTP rps + error rate + scrape timeouts
  11. Spec decode accept rate (hidden unless `spec_draft_tokens > 0`)
  12. Eviction rate (severed + demoted pages/s)

Served by `infer-server` — no separate binary. The route lives in `build_router`
(`coordinator.rs:392`), shared by the multiproc and in-process routers.

## 6. Implementation plan

Each tranche is self-contained and shippable; each needs a dated
`docs/experience/wins/` entry with a matched A/B (step-wall p50/p99 and decode
tok/s before/after; expected delta <0.1 %, i.e. noise — state that).

### T1 — Core counters + Prometheus series (~6 files)

1. `crates/infer-core/src/lib.rs`: 6 new `ThroughputStats` fields;
   `Engine::record_step_wall`; E2 sums at :932; E3 at :1708; E4 at :1242.
2. `crates/infer-core/src/prefix.rs`: C1 internal accumulation; C3 at :454;
   C4 at :645; C2 at `lib.rs:1597`.
3. `crates/infer-server/src/execution.rs`: unconditional `step_start` at :315 +
   `record_step_wall` call.
4. `crates/infer-server/src/metrics.rs`: ~12 `push()` lines.
5. `crates/infer-server/src/schema.rs`: `StatsResponse` fields for the new
   counters (or confirm the stats structs are serialized directly).
6. Bench entry.

Shippable value: step latency, admission, batch occupancy, eviction visible in
`/metrics` and `/v1/stats`.

### T2 — GPU sampler below the seam (~7 files)

1. `crates/infer-seam/src/lib.rs`: `GpuSample`, `GpuDeviceSample`,
   `BackendStats.gpu`.
2. `crates/infer-cuda/src/gpu_sample.rs` (new): thread, nvidia-smi parse,
   atomics, stale tracking.
3. `crates/infer-cuda/src/lib.rs`: `mod gpu_sample;`
4. `crates/infer-cuda/src/executor/qwen35.rs`: spawn at :604 (rank 0), fill
   `stats().gpu`.
5. `crates/infer-server/src/execution.rs`: `snap.gpu`.
6. `crates/infer-server/src/metrics.rs` + `schema.rs`: per-device series
   (`arle_gpu_util_pct{gpu_index=...}` etc. — note: `render_prometheus` labels
   only `model_name` today; per-device series need a label extension in the
   `push` helper) and JSON fields.
7. Bench entry + Mac no-cuda typecheck
   (`cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib`).

Shippable value: GPU util/VRAM/temp/power in `/metrics` and `/v1/stats`.

### T3 — Embedded store + dashboard (~10 files)

1. `crates/infer-observe/` (new crate): `Cargo.toml`, `src/lib.rs`,
   `src/sample.rs`, `src/segment.rs` (writer, reader, retention, bucketing).
2. Root `Cargo.toml`: workspace member.
3. `crates/infer-server/Cargo.toml`: `sysinfo`, `infer-observe`.
4. `crates/infer-server/src/observe.rs` (new): sampler task, `/v1/observe/query`,
   `/v1/observe/latest`, `/dashboard` handlers.
5. `crates/infer-server/src/coordinator.rs`: routes at :392; S1 arrival stamp
   (`multiproc_relay.rs:664` `WireRequest` field); S2–S4 edge counters.
6. `crates/infer-server/src/lib.rs`: spawn observe task at serve start; wire
   `ARLE_OBSERVE_*` env.
7. `crates/infer-server/src/dashboard.html` (new).
8. Bench entry.

Shippable value: 30-day retention and the dashboard.

### T4 — Deferred (not in this plan)

Metal GPU sampler (IOKit/`host_statistics`), downsampling tiers beyond 30 days,
`arle observe export` CLI, per-rank GPU aggregation in multiproc stores,
alerting.

## 7. Risks

| Risk | Mitigation |
|------|------------|
| nvidia-smi subprocess stalls, dies, or is absent (driver break, container without the binary) | Sampler thread is detached and owns no engine state; failures set `stale` and keep last values. `gpu_sample` reads atomics only. Engine never blocks on it. |
| `publish_counters` per-tick cost creeps | New snapshot fields are `Copy` only; `GpuSample` is a fixed array, never a `Vec`. T2 bench entry measures tick time. |
| Observe store disk fills | Write errors are swallowed with rate-limited logging; retention sweep runs daily and at startup; store is best-effort, never in the request path. |
| Torn segment record after crash | Reader skips records failing length sanity or postcard decode; only the tail record is affected. |
| Multiproc: per-rank divergence | GPU sampler and observe store run on rank 0 only (`INFER_TP_RANK`); HTTP edge counters are per-process and documented as rank-0-ingress in the multiproc topology. No metric mutation inside TP collectives — C2/E3/E4 are all outside `tp_sync_min` regions. |
| Backend isolation violation | `GpuSample` is defined in `infer-seam` as plain data; nvidia-smi code lives in `infer-cuda`; no cfg-leak into core/server/api. No-cuda Mac typecheck is a T2 gate. |
| sysinfo on a shared 8×H20 box reports noisy system CPU% | Accepted: it is honest system-wide utilization; per-process RSS is also sampled. Documented in the dashboard tooltip. |
| Counter reset across restart produces garbage rates | `session_id` per process + negative-delta detection; reset buckets emit `null` rates. |
| Scope creep into per-request histograms | Explicitly rejected: sums+counts (the `ttft`/`tpot`/`e2e` precedent) are the only aggregation shape; histograms are T4 and would need a storage-format bump. |
