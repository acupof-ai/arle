# Metal allocator Resource limit (499000) panic on concurrent burst

## Context

Discovered 2026-04-15 while integrating [vllm-project/guidellm](https://github.com/vllm-project/guidellm)
as the canonical bench tool (see [`docs/plans/guidellm-integration.md`](../../plans/guidellm-integration.md)).
Running the canonical sweep profile (`--profile sweep --data prompt_tokens=1024,output_tokens=256`)
against `metal_serve` on Qwen3-0.6B produces this panic after a few seconds:

```
thread '<unnamed>' panicked at infer/src/backend/metal/mlx.rs:38:26:
mlx_array_from_data returned a null MLX handle: [metal::malloc] Resource limit (499000) exceeded.
ERROR infer::backend::metal::runtime: Metal scheduler runtime panicked: ...
```

After the panic the scheduler thread exits and every subsequent request
returns 500 ("Scheduler unavailable or full: request submission failed").
The server does not self-heal — it has to be restarted.

## Root Cause

**`Resource limit (499000)` is NOT a byte count.** It is the MLX Metal
allocator's cap on the number of concurrent live `MTLBuffer` objects,
configured at allocator construction from the device's `resource_limit`
field. Source (MLX 0.31.1,
`mlx/backend/metal/allocator.cpp:262`):

```cpp
if (num_resources_ >= resource_limit_) {
  std::ostringstream msg;
  msg << "[metal::malloc] Resource limit (" << resource_limit_
      << ") exceeded.";
  throw std::runtime_error(msg.str());
}
```

`resource_limit_` is initialized from `device_info().at("resource_limit")`
— on our dev host that value is **499000** buffers. Unlike the wired /
memory / cache limits, this resource count is **not** adjustable from
the MLX C++ API that our bridge currently exposes.

### Why sweep hits it and synchronous does not

The `sweep` profile goes through a **throughput-burst** stage between
the synchronous baseline and the constant-rate points. During the burst
guidellm fires hundreds of concurrent requests at the server to measure
peak throughput. Each request, inside the hot loop, allocates MLX
arrays for:

- Input/output KV tensors (per layer, per head)
- Query / key / value / attention output
- RMSNorm + linear + RoPE intermediate activations
- Logits + sampling state

With `max_active_requests=4` only four requests are active at once, so
the per-step buffer count is modest. **But buffers are not being
recycled aggressively enough across steps.** The cache grows step by
step until `num_resources_` exceeds 499000 and MLX throws. That it
happens specifically on the burst (not on synchronous) narrows it down:
sustained high-frequency admission is what pushes `num_resources_` past
the cap.

Synchronous and low-rate constant profiles do NOT trigger the panic
because each request completes before the next admits — MLX's buffer
cache gets a chance to release.

### Why the panic is unrecoverable

`run_metal_scheduler_runtime` is wrapped in `catch_unwind` at its
outermost call site (`infer/src/backend/metal/runtime.rs:595`). When
any MLX call deep in the hot loop panics (via the helper
`mlx_array_from_raw_or_panic` in `infer/src/backend/metal/mlx.rs:35`
which turns a null handle into `panic!`), the catch triggers and
**logs the failure — then the scheduler task exits**. There is no
restart, so the process stays up but the scheduler is permanently dead.

## Fix (the real one — not done in this entry)

Three things need to land before Metal can ride the guidellm sweep
profile cleanly:

1. **Buffer reuse audit.** Trace `num_resources_` over a sustained
   burst using `mlx::get_peak_memory()` + whatever buffer-count hook
   MLX exposes (if none, add one to `crates/mlx-sys/src/mlx_bridge.cpp`).
   Identify the call sites that allocate and fail to recycle. Primary
   suspects, in order of likelihood:
   - `metal/forward.rs` (Qwen3 rust transformer layer) — per-step
     intermediate allocations that should be reused across layers.
   - `metal/ops.rs::extend_kv_cache` — grows a brand-new array each
     step instead of a ring/rope.
   - `metal/kv_pool.rs` — pool accounting vs. actual reuse.

2. **Error path in `mlx.rs`.** Replace the blanket
   `mlx_array_from_raw_or_panic` with an `Err(...)`-returning variant
   for hot-loop construction. Per-request failures should abort THAT
   request (and send an error to `delta_tx`) while the scheduler keeps
   serving others. 36 call sites — not a small refactor.

3. **Scheduler-level catch.** Move the `catch_unwind` boundary from
   outside `run_metal_scheduler_runtime` to inside the hot loop,
   around each `execute_prefill_chunk` / `execute_decode_batch`. On
   panic, abort the affected request(s) and **continue the loop**
   instead of exiting. This is the minimum defensive layer that
   prevents one bad request from taking down the scheduler, even if
   the buffer-reuse audit later catches the root cause. Belongs in
   `metal/runtime.rs`.

## Mitigation until fixed

- **Do NOT use `--profile sweep` or `--profile throughput` on Metal.**
  The canonical guidellm params are locked to sweep in
  [`docs/plans/guidellm-integration.md §3`](../../plans/guidellm-integration.md),
  which means **canonical bench runs cannot be produced on Metal yet**.
  Metal wins entries will have to use `--profile synchronous` or
  `--profile constant --rate <low>` until the fix lands. This is
  reflected in the plan's §9 Trip wires list.
- CUDA is unaffected — `mlx_array_from_raw_or_panic` is Metal-only,
  CUDA has its own error handling.

## Rule

When a GPU backend's allocator has a hard resource cap (MTLBuffer
count, CUDA handle count, etc.), **every hot-loop allocation must be
reuse-first**. Never assume the allocator's cache will paper over it
under sustained load. Instrument the resource count in the bench loop
before declaring a backend "production" for any profile higher than
synchronous.
