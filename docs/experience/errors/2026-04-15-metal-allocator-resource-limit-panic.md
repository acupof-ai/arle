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
peak throughput. That burst is what exposes the Metal backend's
allocation discipline problem:

- request state is rebuilt in the hot loop on every decode tick
- Qwen3.5 same-length batching repacks/slices batched KV state on every step
- temporary MLX arrays are created faster than the allocator can retire
  them under burst admission

Each request, inside the hot loop, allocates MLX arrays for:

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
the cap, and the current packed-state reuse is not strong enough to keep
that count below the allocator ceiling.

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

## Remediation plan

Metal canonical throughput/sweep remains blocked until this crash path
is fixed. The minimum remediation is three steps:

1. **Buffer reuse audit.** Trace `num_resources_` over a sustained burst
   using `mlx::get_peak_memory()` plus a buffer-count hook in
   `crates/mlx-sys/src/mlx_bridge.cpp` if MLX does not already expose one.
   The goal is to prove which allocations are actually reusable and
   which ones leak live `MTLBuffer` objects across decode steps. Start
   with the burst hot spots:
   - `infer/src/backend/metal/forward.rs` for layer-local intermediates
   - `infer/src/backend/metal/ops.rs::extend_kv_cache` for cache growth
   - `infer/src/backend/metal/kv_pool.rs` for pool accounting vs. real reuse
   - `infer/src/backend/metal/request_state.rs` for batched state packing

2. **`mlx.rs` Result-based error path.** Replace the blanket
   `mlx_array_from_raw_or_panic` pattern with an `Err(...)`-returning
   constructor path for hot-loop allocations. A request-level allocation
   failure should fail that request, emit an error to `delta_tx`, and
   leave the scheduler alive for the other requests. This is a broad
   refactor, but it is required so allocator pressure becomes a normal
   request error rather than a process-wide panic.

3. **`runtime.rs` hot-loop `catch_unwind`.** Move the panic boundary from
   outside `run_metal_scheduler_runtime` to inside the scheduler hot
   loop, around each `execute_prefill_chunk` / `execute_decode_batch`
   invocation. On panic, abort the affected request(s), record the
   failure, and continue the loop instead of exiting the scheduler
   task. This is the last line of defense while the reuse audit and
   error-path refactor are landing.

## Mitigation until fixed

- **Do NOT use `--profile sweep` or `--profile throughput` on Metal.**
  The canonical guidellm params are locked to sweep in
  [`docs/plans/guidellm-integration.md §3`](../../plans/guidellm-integration.md),
  which means **canonical throughput/sweep runs are NOT available on
  Metal until this allocator panic is fixed**. Metal wins entries must
  continue to use `--profile synchronous` or `--profile constant --rate
  <low>` until the crash path is resolved.
- CUDA is unaffected — `mlx_array_from_raw_or_panic` is Metal-only,
  CUDA has its own error handling.

## Rule

When a GPU backend's allocator has a hard resource cap (MTLBuffer
count, CUDA handle count, etc.), **every hot-loop allocation must be
reuse-first**. Never assume the allocator's cache will paper over it
under sustained load. Instrument the resource count in the bench loop
before declaring a backend "production" for any profile higher than
synchronous.
