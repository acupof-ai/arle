# Observability infrastructure: GPU/host sampling + JSONL storage + dashboard — 2026-08-17

> Status: Shipped (Mac typecheck + infer-server tests pass; CUDA tests pending remote)

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
background thread sampling CPU%, RAM, and disk% every 10 s. Same static
`RwLock` + `OnceLock` pattern.

**JSONL storage**: day-segmented append-only files
(`observe-YYYY-MM-DD.jsonl`). ~5 MB/day at 10 s cadence. 30-day retention
(configurable via `ARLE_OBSERVE_RETENTION_DAYS`). Howard Hinnant's days-from-civil
algorithm for date formatting (no chrono dependency).

**Global singleton**: `flock` on `observe.lock` ensures only one process per
machine writes (TP/DP workers skip silently). The kernel releases the lock on
process exit.

**API**:
- `GET /v1/observe/query?range=1h` — time-series samples from disk
- `GET /v1/observe/latest` — most recent sample
- `GET /dashboard` — self-contained HTML dashboard (vanilla JS/SVG, no CDN,
  auto-refresh 10 s, 9 charts: GPU util, VRAM, CPU, RAM, disk, token throughput,
  cache hit rate, TTFT, TPOT)

**Existing endpoints extended**: `/v1/stats` and `/metrics` now include GPU
samples (utilization, memory, temperature, power per device).

## Files

| File | Change |
|------|--------|
| `crates/infer-seam/src/lib.rs` | `GpuSample`, `GpuDeviceSample` types; `BackendStats.gpu` |
| `crates/infer-cuda/src/gpu_sample.rs` | New: nvidia-smi background sampler |
| `crates/infer-cuda/src/lib.rs` | Wire `gpu_sample::latest()` into `stats()` |
| `crates/infer-server/src/observe.rs` | New: host sampler + JSONL storage + query |
| `crates/infer-server/src/coordinator.rs` | 3 new routes |
| `crates/infer-server/src/dashboard.html` | New: 349-line dashboard |
| `crates/infer-server/src/execution.rs` | `CounterSnapshot.gpu` |
| `crates/infer-server/src/metrics.rs` | 5 per-GPU Prometheus series |
| `crates/infer-server/src/schema.rs` | `StatsResponse.gpu` |
| `crates/infer-server/src/multiproc_relay.rs` | `WireStats.gpu` |
| `crates/infer-server/src/lib.rs` | `mod observe` + `spawn_observe_task` |
| `crates/infer-server/Cargo.toml` | `+sysinfo 0.35`, `+libc 0.2` |

## Performance

No inference hot-path changes. The observe task reads `CounterSnapshot` under
the existing mutex every 10 s (contention: negligible — the mutex is already
locked per tick by `publish_counters`). GPU sampler adds one `nvidia-smi`
subprocess every 2 s on rank 0.

## Tests

- `infer-server`: 7 tests pass (4 observe unit tests + 3 existing)
- `infer-cuda`: 3 gpu_sample parse tests (pending remote CUDA build)
- Mac typecheck: `cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib` passes
- Clippy: `-D warnings` clean on infer-server + infer-api (Mac feature set)

## Rule

Background sampling + static stores is the zero-cost pattern for runtime
observability. The engine thread publishes what it already tracks; a background
thread reads and persists. No hot-path instrumentation.
