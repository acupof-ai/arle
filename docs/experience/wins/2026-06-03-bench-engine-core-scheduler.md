# Bench: engine-core scheduler throughput (device-neutral, CPU)

## Goal

First bench artifact for the `infer/` rewrite (branch `arch/ideal-inference-engine`).
Measure the **device-neutral engine-core scheduler's CPU cost** — admission, radix,
chunked-prefill planning, overlap loop, `apply_output` — isolated from any GPU by a
synchronous mock backend. This is the layer that exists today (R0–R1d). Directly
targets the AI-PC concern: the scheduler tick must not be a bottleneck or steal the
user's cores (OS-citizen). Full end-to-end model throughput is a separate bench,
pending R2/R3 (real device forwards) on Metal/V100/H20.

## Params / Env

- Build: `cargo test --release -p infer-core bench_engine_core_scheduler_throughput -- --ignored --nocapture`
- Backend: `MockExecutor` (synchronous, no GPU) + `MockKvPool` (host page pool) — so
  the numbers are pure CPU scheduling cost, not model compute.
- Host: Apple Silicon dev Mac (local), release profile.
- Scenarios: (1) c=1 single long request (1024 prompt + 512 decode); (2) batched
  c=8, 64 distinct requests (384 prompt + 128 decode each), chunked_prefill_size=256.

## Results (raw)

```
[engine-core c=1 long]      gen=512   ticks=513   wall=288.583µs  us_per_tick=0.561  sched_tok_per_s=1774186
[engine-core batched c=8 n=64] gen=8192 ticks=1033 wall=7.177708ms us_per_tick=6.948  sched_tok_per_s=1141311
```

| scenario | ticks | wall | µs/tick (CPU sched) | sched tok/s |
|---|--:|--:|--:|--:|
| c=1 long (1024+512) | 513 | 289 µs | **0.56** | 1.77 M |
| batched c=8, n=64 | 1033 | 7.18 ms | **6.95** | 1.14 M |

## Learnings

- **The device-neutral scheduler is effectively free at c=1: ~0.56 µs/tick.** Real
  decode ITL on-device is tens of ms, so the scheduler adds <0.01% per token — it
  will not be tick-bound and will not steal the user's cores (the OS-citizen / AI-PC
  requirement). This retires the "scheduler-tick-bound" worry for the new core at the
  c=1 single-user shape that is the AI-PC focus.
- Batched c=8 is 6.95 µs/tick — radix + admission + chunked planning across 8 slots
  is still negligible vs GPU compute.
- These are CPU-only (mock backend) numbers: they bound the scheduler overhead, not
  end-to-end latency. The mock resolves synchronously, so the overlap loop's win
  (CPU/GPU parallelism) is not exercised here — it shows up only against a real GPU
  forward.

## Status / next

Interim artifact: the **scheduler layer** of the rewrite, benched. The headline
AI-PC bench (agent-workflow end-to-end + OS-impact, per
[`2026-06-03-rewrite-verification-targets.md`](../../projects/2026-06-03-rewrite-verification-targets.md)
G3) requires a real forward — blocked on R2/R3 (MetalExecutor MLX wiring + model
port). CUDA legs (V100/H20) run via the pod once the executors land. No GPU path
here → bench-exempt for the runtime-change rule; this is additive measurement of the
new crate.
