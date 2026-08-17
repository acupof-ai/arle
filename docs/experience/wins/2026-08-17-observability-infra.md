# Observability infrastructure: GPU/host sampling + JSONL storage + dashboard — 2026-08-17

> Status: Shipped (Mac typecheck + infer-server tests pass; V100 single-process verified; multiproc TP pending remote)

## Context

The inference server had no built-in observability. Operators relied on external
tools (nvidia-smi, top) or ad-hoc log scraping. The goal: fine-grained,
zero-hot-path-cost observation of GPU/CPU/VRAM/RAM/disk usage, token throughput,
cache hit rate, decode speed, and prefill time — with long-term retention and a
built-in dashboard.

## What Worked

**Background sampling, zero hot-path cost.** All samplers run in background
threads. The engine thread only writes the `CounterSnapshot` it already
publishes per tick — no atomics, locks, or syscalls added to the inference path.

**GPU sampler** (`crates/infer-cuda/src/gpu_sample.rs`): spawns a background
thread that runs `nvidia-smi --query-gpu=...` every 2 s, parses the CSV into a
fixed `[GpuDeviceSample; 8]` array, and stores it in a static `RwLock`. Rank 0
only (via `INFER_TP_RANK`). On query failure, marks the existing sample as
`stale` rather than discarding it.

**Host sampler** (`crates/infer-server/src/observe.rs`): sysinfo-based
sampling of CPU%, RAM, and disk% every 10 s, inline in the observe task
thread. Disk refresh every 30 ticks (5 min).

**JSONL storage**: day-segmented append-only files
(`observe-YYYY-MM-DD.jsonl`). ~5 MB/day at 10 s cadence. 30-day retention
(configurable via `ARLE_OBSERVE_RETENTION_DAYS`). Howard Hinnant's days-from-civil
algorithm for date formatting (no chrono dependency).

**Global singleton**: `flock` on `observe.lock` ensures only one process per
machine writes (TP/DP workers skip silently). The kernel releases the lock on
process exit.

**API**:
- `GET /v1/observe/query?range=1h` — time-series samples from disk
- `GET /dashboard` — self-contained HTML dashboard (vanilla JS/SVG, no CDN,
  auto-refresh 10 s, 9 charts: GPU util, VRAM, CPU, RAM, disk, token throughput,
  cache hit rate, TTFT, TPOT)

**Multiproc coordinator** (`0d5f58970`): the coordinator process is engine-less
in TP mode — workers use `CudaWorkerEngine` directly, so `spawn_observe_task`
was never called and the JSONL store stayed empty. Fixed by adding
`CoordinatorHandle::query_stats()` (shared with the `metrics()` handler) and
spawning the observe task from the router call sites that know the topology:
local relay skips (ServeHandle owns sampling), multiproc coordinator and DP
spawn exactly one. The task is a single closure-parameterized
`spawn_observe_task<F: FnMut() -> Option<CounterSnapshot>>` — one loop, one
implementation. The flock singleton handles cross-process dedup.

An earlier version (`cf028de9d`) minted a private stats-request ID counter in
the observe thread, colliding with the HTTP handlers' counter over the shared
`stats_sinks` map — collisions silently evicted awaiters, producing zeroed
samples. Fixed by routing all stats queries through `query_stats()` which
allocates from the handle's own counter.

**Existing endpoints extended**: `/v1/stats` and `/metrics` now include GPU
samples (utilization, memory, temperature, power per device).

## Files

| File | Change |
|------|--------|
| `crates/infer-seam/src/lib.rs` | `GpuSample`, `GpuDeviceSample` types; `BackendStats.gpu` |
| `crates/infer-cuda/src/gpu_sample.rs` | New: nvidia-smi background sampler |
| `crates/infer-cuda/src/lib.rs` | Wire `gpu_sample::latest()` into `stats()` |
| `crates/infer-server/src/observe.rs` | New: host sampler + JSONL storage + query |
| `crates/infer-server/src/coordinator.rs` | 2 new routes + `query_stats()` + topology-aware observe spawn |
| `crates/infer-server/src/dashboard.html` | New: 349-line dashboard |
| `crates/infer-server/src/execution.rs` | `CounterSnapshot.gpu` |
| `crates/infer-server/src/metrics.rs` | 5 per-GPU Prometheus series |
| `crates/infer-server/src/schema.rs` | `StatsResponse.gpu` |
| `crates/infer-server/src/multiproc_relay.rs` | `WireStats.gpu` |
| `crates/infer-server/src/lib.rs` | `mod observe` + closure-based `spawn_observe_task` |
| `crates/infer-api/src/serve.rs` | `observe: true` for multiproc coordinator |
| `crates/infer-server/Cargo.toml` | `+sysinfo 0.35`, `+libc 0.2` |

## Performance

No inference hot-path changes. The observe task reads `CounterSnapshot` under
the existing mutex every 10 s (contention: negligible — the mutex is already
locked per tick by `publish_counters`). GPU sampler adds one `nvidia-smi`
subprocess every 2 s on rank 0.

## Tests

- `infer-server`: 4 observe unit tests pass
- Mac typecheck + clippy `-D warnings`: clean
- V100 single-process: `/v1/observe/query` returns 19 samples (10 s cadence),
  GPU/CPU/RAM data verified, `/dashboard` serves 11.9 KB HTML
- Multiproc TP: pending remote verification (8×H20)

## Rule

Background sampling + static stores is the zero-cost pattern for runtime
observability. The engine thread publishes what it already tracks; a background
thread reads and persists. No hot-path instrumentation.
