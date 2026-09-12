# In-process NVML sampler replaces the disabled nvidia-smi thread — 2026-09-12

> Status: Pending remote. Mac typecheck + infer-server/seam tests pass; Linux
> NVML build and the matched A/B are pending an 8-GPU pod window (all cards held
> by another container as of 2026-09-12). The sampler is default-off.

## Context

`bbd422973` disabled the GPU sampler: a background thread ran
`nvidia-smi --query-gpu=...` every 2 s and stalled the H20 driver for ~2-5 ms
per step under DSpark batched decode (c>=2). The disable returned `None` until
a non-stalling sampler replaced it. That left GPU utilization, VRAM,
temperature, and power absent from JSONL, `/metrics`, and `/v1/stats` on every
deployment. This change supplies the replacement.

## What Worked

**In-process NVML, dlopen'd, no new dependency.** `gpu_nvml.rs` resolves
`libnvidia-ml.so.1` through `dlopen`/`dlsym` (same FFI idiom as the NVTX
example) and calls `nvmlInit_v2`, `nvmlDeviceGetCount_v2`,
`nvmlDeviceGetHandleByIndex_v2`, `nvmlDeviceGetMemoryInfo`,
`nvmlDeviceGetUtilizationRates`, `nvmlDeviceGetTemperature`,
`nvmlDeviceGetPowerUsage`. A missing library, a missing symbol, or init failure
returns `None` rather than failing startup. It reads all four quantities that
the Prometheus gauges export (`util_pct`, `memory_used_mb/memory_total_mb`,
`temp_c`, `power_w`); filling only some would publish a confident zero for the
rest.

**Sampled inside the observe loop, not a new thread.** The read runs once per
observe tick in `spawn_observe_task`, alongside the existing inline host
sampler. There is no background thread, no static `RwLock`, no rank-0 guard,
and no infer-cuda involvement. The tick interval is one constant
(`TICK_INTERVAL`, 10 s) used by both the loop sleep and the documented GPU
cadence, so the sampling rate and the code cannot drift.

**One node-level writer covers both deployments.** The observe loop runs in
single-process `ServeHandle` and in the engine-less multiproc coordinator. The
GPU sample is passed into the snapshot closure, which stores it on the same
counter object the on-demand `/metrics` and `/v1/stats` readers use, so JSONL
and the live endpoints see one source. `BackendStats.gpu` and the engine
overwrite (`snap.gpu = stats.gpu`, always `None`) are deleted, along with the
dead first-wins `WireStats.gpu` merge; GPU data is no longer relayed from
workers.

**Dashboard renders all devices.** The GPU utilization and VRAM charts drew
only `devices[0]`. They now draw one line per device (up to 8, fixed palette,
legend, per-device hover) and show the across-device average in the card
header; a single-GPU host collapses to the prior single-line chart.

**Opt-in.** Enabled only with `ARLE_OBSERVE_GPU=1`, inherited across the
single/multiproc process boundary like the other `ARLE_OBSERVE_*` knobs. It
stays default-off until the A/B clears it.

## Why the prior is favorable

The disabled sampler polled every 2 s and forked an `nvidia-smi` subprocess per
poll; this one polls every 10 s in-process. That is 5x fewer polls and no
subprocess fork per poll — two independent reductions in driver-lock pressure.
This is a prior, not a measurement: it does not default-on before the A/B.

## Pending A/B (pre-registered, not run)

Matched A/B, DSpark decode c>=2 on the 8xH20 pod, sampler off vs on
(`ARLE_OBSERVE_GPU`), simultaneous or interleaved matched pairs. Primary
metric: decode p99 inter-token latency; the sampler is acceptable only if the
on/off delta is inside the matched-A/B noise envelope (~10%). If p99 moves,
the first knob is the poll interval (`TICK_INTERVAL`).

The hypothesis is frozen now, before any data. The prereg ledger row opens at
run start, because a `running` row left past 24h fails `check_repo_hygiene`
and the pod window is not scheduled:

```
scripts/prereg.py start --name nvml-observe-ab \
  --cmd "DSpark decode c>=2 matched A/B, ARLE_OBSERVE_GPU off vs on, 8xH20" \
  --hypothesis "10s in-process NVML sampling moves decode p99 ITL by less than the ~10% matched-A/B noise envelope; the prior bbd422973 stall (2s nvidia-smi fork, 2-5 ms/step) does not recur"
```


## Files

| File | Change |
|------|--------|
| `crates/infer-server/src/gpu_nvml.rs` | New: dlopen NVML node sampler |
| `crates/infer-server/src/observe.rs` | Sample GPU per tick; `TICK_INTERVAL`; closure takes the sample |
| `crates/infer-server/src/lib.rs` | `mod gpu_nvml`; both observe spawn sites overlay GPU onto counters |
| `crates/infer-server/src/coordinator.rs` | Coordinator observe closure overlays GPU onto snapshot + cached stats |
| `crates/infer-server/src/execution.rs` | Delete `snap.gpu = stats.gpu` overwrite |
| `crates/infer-seam/src/lib.rs` | Delete `BackendStats.gpu` |
| `crates/infer-cuda/src/lib.rs` | Delete `gpu: None` |
| `crates/infer-server/src/multiproc_relay.rs` | Delete `WireStats.gpu`, first-wins merge, relayed mappings |
| `crates/infer-server/src/dashboard.html` | Per-device GPU lines (up to 8) |

## Rule

Replace a stalling external-tool poll with an in-process library call sampled
inside the loop that already owns the write, at that loop's real cadence. Name
the cadence for the sleep that drives it, so a preregistered experiment never
describes a rate the code cannot run.
