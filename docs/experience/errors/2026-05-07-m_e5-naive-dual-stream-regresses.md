# M_e.5 v1: naive dual-stream rotation around m->forward regresses — 2026-05-07

## TL;DR

Implemented the multi-stream encode pipelining design from
[`docs/plans/M_e5-mlx-multi-stream-pipelining.md`](../../plans/M_e5-mlx-multi-stream-pipelining.md)
in its simplest form: rotate `mlx::core::StreamContext` between `s_a` and
`s_b` per call to `qwen35_compiled_step_batch_packed`, around just the
`m->forward(inputs)` call. Path probe fires; dual-stream code path is
exercised. **But this regresses performance 5-13% across c=1/c=4/c=8 on
Qwen3.6 35B-A3B-4bit, the opposite of what the research-subagent
predicted.**

Shipped as **default-OFF env gate (`INFER_METAL_DUAL_STREAM=1`)** so this
has zero production impact. This entry documents the dead-end so future
ticks don't repeat it.

## What was implemented

`crates/mlx-sys/src/mlx_qwen35_model.cpp`:
- `cpp_dual_stream_enabled()` env probe (cached `INFER_METAL_DUAL_STREAM`)
- Thread-local `DualStreams { Stream a, Stream b, int parity }` lazily
  initialized on first use
- `std::once_flag g_dual_stream_probe` for the path probe
- In `qwen35_compiled_step_batch_packed`: when env enabled, flip parity
  → `mlx::core::StreamContext ctx(cur)` → `m->prev_outputs = m->forward(inputs)`
- StreamContext destructs at end of the if-block, restoring default stream
  before the existing FFI return path

## Bench results

Workload: `/tmp/cN_smoke.sh <N>`, max_tokens=64, temperature=0.0.
Baseline = current head with M_e.4 SwiGLU fusion + auto-wired-limit
(commit `294a1d4`).

| batch | head p50 | +M_e.5 p50 | Δ p50 |
|------:|---------:|-----------:|------:|
| c=1   | 11489 | 12928 | **+12.5%** |
| c=4   | 23143 | 24928 | **+7.7%** |
| c=8   | 40025 | 42343 | **+5.8%** |

Path probe confirmed firing:
```
metal_path_probe: M_e.5 dual-stream FIRED (initial parity=1)
```

## Root cause (hypothesis)

The research subagent's plan was based on this MLX claim from
`mlx/scheduler.h`: "Each `Stream` has its own `StreamThread` worker, so
dispatching to B does not block on A's encode." The optimistic reading:
encoder work for B runs on B's worker thread in parallel with A's GPU
exec.

**The pessimistic reading is closer to reality.** Per
`docs/experience/wins/2026-05-07-bench-qwen36-encode-bottleneck.md`,
`mx::async_eval` does graph traversal + Metal command-buffer encoding
**synchronously on the calling thread** (per `mlx/transforms.cpp`
`eval_impl(... async=true)` source). Only GPU completion is async. So
even with two streams, the encoder still runs on the same caller
thread — there's no parallel-encoder benefit from rotating streams.

What rotation DOES do: arrays produced on `s_a` (e.g. KV writes from
step N) are consumed on `s_b` by step N+1. MLX detects the
cross-stream dependency and inserts an event-fence on `s_b`'s queue
(per `mlx/transforms.cpp::eval_impl`, the `in.event().stream() != stream`
branch). **The fence costs real GPU-queue time** for every cross-stream
KV access — and Qwen3.6 MoE has 80 KV slabs to re-bind per step. That
fence cost is the regression we measured.

In short: we paid for cross-stream fences (GPU-side) without buying
parallel-encoder overlap (CPU-side, since encoding stays on caller
thread).

## What v2 would need to actually win

For dual-stream to deliver the predicted ~6-12 ms win, **the caller
thread must NOT run the encoder.** Two options:

1. **A genuine background-encode FFI**: a new C++ entry that submits
   work to a per-stream worker queue and returns IMMEDIATELY without
   running `eval_impl` synchronously. This requires modifying MLX
   itself (out of scope per project rules).

2. **Pipeline the encode work on a separate thread WITHIN ARLE**: do
   `m->forward(inputs)` + sample build + `async_eval` ALL on a
   background thread, where the caller thread merely posts work and
   collects the prev call's result. This would need a careful
   thread-safety review (per `mlx#3078`, encoders are per-thread; we'd
   need a single dedicated encode-thread) and adds a significant
   architectural footprint.

Neither is feasible this week. **The encoder-CPU overlap is blocked at
the MLX layer.**

## What we learned

1. **Don't trust an optimistic reading of MLX docs without an empirical
   test.** The "per-stream StreamThread worker" line in `scheduler.h`
   sounded like async encoders, but the actual encode lives in
   `eval_impl` on the caller thread. Subagent research is fast and
   often correct, but ambiguous claims need an empirical run.
2. **Env-gated experiments pay for themselves.** Behind a flag, this
   takes 30 LoC to rule out. If it had been the default, we'd have
   shipped a 7-12% regression.
3. **Cross-stream fences are real GPU-side cost.** When KV cache or
   any cross-step state is shared, alternating streams *adds*
   serialization rather than enabling parallelism. Future stream
   plans should isolate compute that's truly independent across steps
   (the model would need stream-local state, like per-stream KV
   shards).

## Rule

**Before adopting an MLX stream-API plan**, run the trivial
implementation behind an env flag and verify the win exists at all.
The overlap predictions from upstream API documentation can be
optimistic; cross-stream automatic fence insertion can swallow the
upside if any cross-step data dependency exists.

## What's left in tree

- `cpp_dual_stream_enabled()` — kept (cheap, useful for any future
  v2 that needs the flag).
- `DualStreams` thread-local — kept (also cheap; lazy-init only fires
  if the env is on, which it isn't by default).
- The `if (cpp_dual_stream_enabled()) { ... StreamContext ... } else
  { ... }` branch in `qwen35_compiled_step_batch_packed` — kept as a
  proven-not-helpful no-op when env is off. Future v2 work can rewire
  the body without re-adding the env-gate machinery.
- The `g_dual_stream_probe` once_flag — kept for the same reason.
- Plan
  [`docs/plans/M_e5-mlx-multi-stream-pipelining.md`](../../plans/M_e5-mlx-multi-stream-pipelining.md)
  — updated in this commit with an "Erratum" section noting v1 is
  blocked.

## Next

Per the parallel research subagent (this date), the next-tier levers
that **don't depend on MLX-side encoder threading**:

1. **DFlash with a tiny dense draft (Qwen3-0.6B) against the Qwen3.6
   35B-A3B target** — predicted 2× speedup at decode if draft sustains
   ≥150 tok/s on M-series. M effort. Synergistic with M_e.4 (smaller
   step × fewer steps).
2. **Chunked-prefill tuning** for long-context Qwen3.6 — orthogonal to
   decode. Safe down to chunk_size=512 on 128-expert MoE; below that
   gather_qmm utilization drops. S effort, ~30 min bench.

## References

- Predecessor (encoder-bottleneck localization):
  [`2026-05-07-bench-qwen36-encode-bottleneck.md`](../wins/2026-05-07-bench-qwen36-encode-bottleneck.md)
- Plan being errata'd:
  [`docs/plans/M_e5-mlx-multi-stream-pipelining.md`](../../plans/M_e5-mlx-multi-stream-pipelining.md)
- MLX upstream:
  - `mlx/transforms.cpp::eval_impl` — encode-on-caller-thread reality
  - `mlx/scheduler.h::StreamThread` — the misleading "worker" line
  - ml-explore/mlx#3078 — confirms encoder is per-thread
