# CPU/GPU Pipeline With Explicit Stream Synchronization

Last updated: 2026-05-15

Status: design plan. Implementation is partially present in CUDA serving
today, but the common pipeline/fence contract described here is not yet a
single source of truth in code.

## Goal

Make ARLE serving a staged CPU/GPU pipeline where CPU-owned work can overlap
GPU-owned work without weakening correctness. The target shape is:

```text
HTTP ingress
  -> CPU preprocessing: chat template, tokenization, routing hints
  -> H2D staging: prompt ids, decode metadata, prefix/KV promotions
  -> GPU compute: prefill, decode, graph replay, sampling kernels
  -> D2H readback: sampled token ids, logprobs, terminal metadata
  -> CPU postprocessing: detokenize, stop handling, SSE/JSON emit
```

The architecture must expose explicit synchronization semantics at the
necessary boundaries. In practice, this means CUDA streams and events on CUDA,
and MLX/Metal async-eval or command-buffer completion tokens on Metal.

## Overlap Modes

This plan is about overlap at three levels:

1. **Request-level CPU/GPU overlap.** While GPU worker N is running prefill or
   decode for one batch, CPU ingress can tokenize, compute routing hints, route,
   and detokenize other requests. The selected worker then performs worker-local
   prefix lookup during admission.
2. **Copy/compute overlap.** H2D and D2H transfers that are independent of the
   current compute stream work can run on a copy stream. Required dependencies
   are expressed with events.
3. **Worker-lane overlap.** On multi-GPU hosts, each CUDA ordinal owns a worker
   lane with local CPU affinity and local pinned-memory allocation. NUMA routing
   keeps requests near the right lane.

This plan is **not** claiming that one request's strictly dependent decode
steps can overlap with themselves. A sampled token from step `t` remains a hard
dependency for step `t + 1`; the pipeline only removes waits that are not
semantic dependencies.

## Non-goals

- Do not claim a throughput or latency win from this document alone. The
  acceptance gate requires nsys/bench evidence.
- Do not treat NUMA as the pipeline itself. NUMA is the locality layer that
  places CPU workers, pinned memory, NICs, and GPU workers close together.
- Do not try to overlap dependent GPU kernels inside a single model step.
  The immediate target is CPU/GPU and copy/compute overlap, not kernel DAG
  scheduling.
- Do not put H2D/D2H stream waits inside CUDA Graph capture. Graph bodies stay
  compute-stream-only unless a later trace proves a different shape is safe.

## Current ARLE State

CUDA already has the key primitives, but they are local patterns rather than
a runtime-wide pipeline contract:

- `DeviceContext` owns a compute stream and a copy stream:
  [`crates/cuda-kernels/src/tensor.rs`](../../crates/cuda-kernels/src/tensor.rs).
- `DeviceContext::copy_waits_for_compute()` records an event on the compute
  stream and makes the copy stream wait.
- `DeviceContext::compute_waits_for_copy()` records an event on the copy
  stream and makes the compute stream wait.
- Qwen3 decode already uses an async greedy-readback ring:
  `argmax/logprobs -> copy stream D2H -> event query -> CPU read`.
- CUDA prefill already has `launch_prefill_batch` and
  `complete_prefill_batch` separation in the scheduler path.
- Request preprocessing can carry `prompt_tokens` into `IncomingRequest`, so
  model workers do not need to own every tokenization path.
- CUDA prefix lookup and prefix-aware admission already run inside the selected
  worker, where worker-local radix/KV state is valid.
- CUDA multi-worker bootstrap already discovers runtime topology, binds CPU
  before CUDA init, allocates worker-local resources, and routes by NUMA cost.

The gaps:

- Most tensor H2D helpers still enqueue on the compute stream.
- Qwen3 has the strongest async readback path; Qwen3.5 and DeepSeek paths are
  less uniformly wired and can still contain full-stream sync points.
- There is no common `PipelineFence` type that scheduler code can pass between
  stages.
- Metrics report many stage timings, but do not yet form one pipeline wait
  accounting model across tokenization, H2D, compute, D2H, and detokenization.
- Metal has a scheduler/runtime pipeline, but lacks a command-buffer/fence
  abstraction equivalent to CUDA events at the Rust boundary.

## External Runtime Contract

CUDA supports the desired contract directly:

- Kernel launches are asynchronous with respect to the host.
- Streams are ordered internally; independent streams can overlap when device
  resources allow it.
- `cudaEventRecord` plus `cudaStreamWaitEvent` is the correct primitive for
  producer/consumer edges across streams.
- `cudaMemcpyAsync` only gives useful copy/compute overlap when host memory is
  page-locked and the transfer uses an async-capable stream.
- The legacy/default stream can introduce implicit synchronization; pipeline
  code should stay on explicit non-default streams.

Metal/MLX supports a coarser contract:

- MLX is lazy and can submit work asynchronously via `async_eval`.
- MLX exposes stream concepts, but ARLE's Rust side currently sees most of
  this through the C++ bridge.
- Short term, Metal fences should be coarse-grained around async eval or
  request-state operations.
- Medium term, the bridge should expose command-buffer completion tokens if
  trace evidence shows host-side waiting is material.

References:

- NVIDIA CUDA C Programming Guide, asynchronous concurrent execution.
- NVIDIA CUDA Runtime API, synchronization behavior.
- MLX lazy evaluation and `async_eval` documentation.
- MLX stream documentation.
- Apple Metal command-buffer completion and wait APIs.

## Pipeline Model

Every stage consumes a packet and returns a packet plus a fence:

```rust
pub struct PipelinePacket<T> {
    pub payload: T,
    pub fence: PipelineFence,
    pub trace_id: RequestTraceId,
    pub owner: ResourceOwner,
}

pub enum PipelineFence {
    Ready,
    Cuda(CudaPipelineFence),
    Metal(MetalPipelineFence),
}

pub struct CudaPipelineFence {
    pub device_ordinal: u32,
    pub producer: PipelineStreamKind,
    pub event: CudaEventHandle,
}

pub enum PipelineStreamKind {
    Compute,
    Copy,
}
```

Fence rules:

- `Ready` means the payload is immediately consumable by CPU code.
- A CUDA fence is ready when its event query succeeds.
- A CPU reader may poll a fence; it may only block if the next semantic action
  cannot proceed without the payload.
- A resource owner may not reuse or free a buffer until all dependent fences
  have completed.
- Cross-stream waits are encoded as edge operations, not hidden in random
  tensor helpers.

## Stage Semantics

### CPU Preprocess

Inputs:

- HTTP request, sampling params, optional `session_id`, trace context.

Work:

- Apply chat template if needed.
- Tokenize prompt.
- Compute routing hints such as ingress NUMA node and session id.
- Compute request length contract.

Output:

- `PreprocessedRequest { prompt_tokens, sampling, session_id, ingress_numa_node }`
- `PipelineFence::Ready`

This stage should run before request submission to a GPU worker whenever the
handle exposes a tokenizer. The GPU worker keeps a fallback tokenizer path for
compatibility and error isolation.

### Route

Inputs:

- Preprocessed request.
- Runtime topology and worker queue counters.

Work:

- Select worker by NUMA cost plus queue pressure.
- Preserve session stickiness while the selected worker is not overloaded.
- Record migration/rebalance metrics.
- Do not attach a prefix plan before the worker is selected. Prefix/radix state
  is worker-local.

Output:

- Worker-local packet.
- `PipelineFence::Ready`

### Worker Admission / Prefix Lookup

Inputs:

- Worker-local packet.
- Selected worker's radix/prefix cache state.

Work:

- Run worker-local prefix lookup and session-affinity lookup.
- Build `PrefixAdmissionPlan` for direct GPU reuse, staged readmission, or cold
  prefill.
- Degrade stale or non-runnable hits to cold prefill inside the same worker.

Output:

- `WorkerAdmission { prompt_tokens, prefix_plan, sampling, session_id }`
- `PipelineFence::Ready`

### H2D Stage

Inputs:

- Worker admission packet.
- Host prompt/meta buffers.
- Optional staged prefix/KV blocks from the selected worker's prefix plan.

Work:

- Allocate or borrow worker-local pinned buffers.
- Enqueue H2D on the worker copy stream.
- Record an event on the copy stream.

Output:

- Device-side prompt/meta/KV handles.
- `CudaPipelineFence { producer: Copy, event: h2d_done }`

Consumer rule:

```text
compute_stream waits h2d_done before any kernel reads uploaded data
```

### GPU Prefill/Decode Stage

Inputs:

- Device payload.
- H2D fence, if the payload was produced by copy stream.

Work:

- Make compute stream wait on H2D fence.
- Launch prefill, decode, graph replay, and sampling kernels.
- Record compute-done event if any downstream copy or host read needs results.

Output:

- Device logits/sampled-token state.
- `CudaPipelineFence { producer: Compute, event: compute_done }`

### D2H Readback Stage

Inputs:

- Device sampled token ids and logprobs.
- Compute-done fence.

Work:

- Make copy stream wait on compute-done fence.
- Copy sampled ids/logprobs into pinned host ring slots.
- Record D2H-done event on copy stream.

Output:

- Host ring slot.
- `CudaPipelineFence { producer: Copy, event: d2h_done }`

Consumer rule:

```text
CPU may only read the host ring slot after d2h_done is ready
```

### CPU Postprocess

Inputs:

- Readback host slot.
- D2H fence.

Work:

- Poll or wait on D2H fence only when token ids are required.
- Decode token ids incrementally.
- Apply stop sequence handling.
- Emit SSE/JSON delta.
- Update usage, cache, and request metrics.

Output:

- Client-visible delta or terminal response.
- `PipelineFence::Ready`

## Required Synchronization Points

These are the only required sync points in the target architecture:

| Edge | Sync primitive | Blocking policy |
| --- | --- | --- |
| H2D -> GPU compute | compute stream waits on copy-stream event | never block CPU |
| GPU compute -> D2H | copy stream waits on compute-stream event | never block CPU |
| D2H -> CPU read | CPU event query or narrow wait | block only if token ids are needed now |
| Resource reuse | owning stage waits on last consumer fence | local wait only, no device-wide sync |
| Request finish/error | drain request-local fences | local wait only |
| CUDA Graph replay | graph runs on compute stream | graph boundary handles copy waits outside capture |

Forbidden by default:

- `cudaDeviceSynchronize` on serving hot path.
- Full compute stream synchronize for sampled-token readback when an event
  query ring can be used.
- Hidden stream waits inside helper APIs that do not return or consume a fence.
- Copy from pageable host memory in a path expected to overlap with compute.

## CUDA Implementation Plan

### P0: Fence substrate

Status:

- Initial CUDA substrate is landed in `crates/cuda-kernels/src/tensor.rs`:
  `CudaPipelineFence`, `CudaPipelineStreamKind`,
  `CudaPipelineFenceStatus`, and
  `DeviceContext::{record_pipeline_fence, wait_on_pipeline_fence}`.
- Existing `copy_waits_for_compute()` and `compute_waits_for_copy()` now route
  through the explicit fence API, preserving behavior while making the fence
  boundary reusable by later stages.
- `wait_on_pipeline_fence()` uses cudarc's stream wait path and
  `CudaPipelineFence::query()` binds the event context before polling, so
  cross-thread stage handoff does not depend on the caller's current CUDA
  context.

Files likely involved:

- `crates/cuda-kernels/src/tensor.rs`
- `infer/src/model.rs`
- `infer/src/scheduler/cuda/*`
- `infer/src/metrics.rs`

Work:

- Introduce `PipelineFence` and CUDA event wrapper.
- Convert `DeviceContext::{copy_waits_for_compute, compute_waits_for_copy}`
  into lower-level helpers used by the fence wrapper.
- Add tests for event state transitions where CUDA is available; keep no-cuda
  type tests as stubs.
- Add metrics for fence poll/ready/wait counts.

Exit gate:

- No behavior change except new metrics and type surface.

GPU verification TODO for CUDA Codex:

```bash
CUDARC_CUDA_VERSION=13010 \
  cargo check -p cuda-kernels --no-default-features --features cuda,no-cuda

CUDARC_CUDA_VERSION=13010 \
  cargo check -p infer --no-default-features --features cuda,no-cuda

CUDA_HOME=/usr/local/cuda \
  cargo test -p cuda-kernels --no-default-features --features cuda \
  pipeline_fence_orders_compute_and_copy_streams -- --nocapture
```

Evidence to capture:

- the test passes on a real CUDA host.
- no `cudaDeviceSynchronize` is introduced by the fence API.
- `cuStreamWaitEvent` appears only on the intended compute/copy stream edges.

### P1: H2D as a copy-stream stage

Status:

- Initial CUDA H2D copy-stream helpers are landed in
  `crates/cuda-kernels/src/tensor.rs`:
  `DeviceContext::memcpy_pinned_htod_on_copy_stream`.
- The helper enqueues pinned-host-source uploads into an existing device
  allocation on the copy stream and returns a
  `CudaPipelineFence { producer: Copy, .. }`. Compute consumers must call
  `wait_on_pipeline_fence()` explicitly before reading the uploaded payload.
- The helper is intentionally `unsafe`: the destination allocation must stay
  alive and must already be valid on the copy stream before enqueue. Callers
  add a pre-copy wait only when the destination actually depends on prior
  work from another stream; the helper itself does not serialize behind
  unrelated compute.
- No model weight-loading path has been switched. This tranche only establishes
  the request-metadata substrate for later call-site migrations.

Files likely involved:

- `crates/cuda-kernels/src/tensor.rs`
- `infer/src/model/*`
- `infer/src/scheduler/cuda/prefill.rs`
- `infer/src/scheduler/cuda/decode.rs`

Work:

- Add async H2D helpers that enqueue on `copy_stream` and return a fence.
- Keep existing compute-stream copy helpers for load-time and graph-sensitive
  paths.
- Switch request metadata and staged-prefix promotion first; do not switch
  model weight loading in this phase.
- Add `h2d_latency_us` and `h2d_wait_us`.

Exit gate:

- nsys shows H2D copies on copy stream.
- compute stream waits only on matching H2D events.

GPU verification TODO for CUDA Codex:

```bash
CUDARC_CUDA_VERSION=13010 \
  cargo check -p cuda-kernels --no-default-features --features cuda,no-cuda

CUDA_HOME=/usr/local/cuda \
  cargo test -p cuda-kernels --no-default-features --features cuda \
  pinned_copy_stream_h2d_helper_returns_compute_waitable_fence -- --nocapture
```

### P2: Readback unification

Status:

- Qwen3 remains the reference async greedy readback implementation:
  argmax/logprob outputs are snapshotted, copied D2H on the copy stream, and
  polled through a model-owned event ring.
- Qwen3.5 now uses the same async D2H pattern for greedy batched decode:
  `sample_batch_greedy_launch()` enqueues copy-stream readback and returns the
  ring slot, while `sample_batch_greedy_readback()` polls without a full
  compute-stream sync. Qwen3.5 also stages sampled-token GPU handoff for the
  one-step-ahead scheduler path. The scheduler invalidates the per-slot handoff
  whenever distributed rank0 coordination changes the local sampled token, and
  keeps the conservative readback-before-next-launch guard for deferred
  distributed rows whose next decode input is not final yet.
- `ServerMetrics` now surfaces `d2h_latency_us`, `d2h_wait_us`, and
  `readback_poll_not_ready` through Prometheus, `/v1/stats`, and summary logs.
- DeepSeek readback remains an explicit fallback until DSV4 correctness parity
  is stable; the current worktree has unrelated DSV4 changes and this P2
  tranche intentionally does not touch them.

Files likely involved:

- `infer/src/model/qwen3/batch_decode.rs`
- `infer/src/model/qwen3/forward.rs`
- `infer/src/model/qwen35/forward.rs`
- `infer/src/model/deepseek/*`

Work:

- Keep Qwen3 async readback as the reference implementation.
- Replace Qwen3.5 full `ctx.sync()` readback with the same event/ring model
  where model semantics allow it.
- Add DeepSeek readback only after correctness parity is stable.
- Surface `d2h_latency_us`, `d2h_wait_us`, and `readback_poll_not_ready`.

Exit gate:

- No per-step full stream sync remains in the hot sampled-token path for
  Qwen3. Qwen3.5/DeepSeek exceptions must be logged as explicit fallback.

GPU verification TODO for CUDA Codex:

```bash
CUDARC_CUDA_VERSION=13010 \
  cargo check -p infer --no-default-features --features cuda,no-cuda

CUDA_HOME=/usr/local/cuda \
  cargo test --release -p infer --features cuda --test e2e_qwen35 -- --nocapture

scripts/profile_nsys_signal.sh pipeline-d2h-readback-qwen35 \
  --server-args "--model-path infer/models/Qwen3.5-4B --port 8000 --max-seq-len 8192" \
  --fast \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3.5-4B
```

### P3: Scheduler pipeline handles

Status:

- Scheduler pending GPU work now carries typed `GpuStageHandle`s for prefill
  and readback stages. Handles track stage kind, lifecycle state, stable
  stage id, and the request slots whose local fences must drain before reuse.
- Async and sync prefill launch/complete paths now create prefill stage
  handles; mixed decode+prefill carries a prefill handle alongside the
  decode readback handle.
- Decode readback pending work now carries a readback stage handle. The
  scheduler still plans CPU-side work between launch and readback, and polls
  handles at step boundaries through the existing pending readback path rather
  than synchronizing eagerly.
- `slot_has_pending_gpu_work()` now checks handle slot membership, so request
  cancellation and cleanup continue to drain only the request-local prefill or
  readback fences instead of blocking on unrelated slots.
- `ServerMetrics` exposes scheduler pipeline stage depth and transition
  counts in Prometheus, `/v1/stats`, and summary logs:
  `infer_scheduler_pipeline_stage_depth{stage,state}` for queued/in-flight
  depth and `infer_scheduler_pipeline_stage_total{stage,state}` for
  queued/ready/completed transitions.
- CUDA runtime and nsys validation remain pending on a GPU host; this tranche
  does not claim a throughput or latency result.

Files likely involved:

- `infer/src/scheduler/cuda/prefill.rs`
- `infer/src/scheduler/cuda/decode.rs`
- `infer/src/scheduler/cuda/runtime/*`

Work:

- Convert prefill launch/complete to return a typed `GpuStageHandle`.
- Keep decode planning CPU-side while GPU work from the previous stage is in
  flight.
- Poll stage handles at scheduler step boundaries instead of synchronizing
  eagerly.
- Ensure request cancellation drains only request-local fences.

Exit gate:

- Service metrics can show queued, in-flight, ready, and completed stage
  counts for prefill and readback.

GPU verification TODO for CUDA Codex:

```bash
CUDARC_CUDA_VERSION=13010 \
  cargo check -p infer --no-default-features --features cuda,no-cuda

CUDA_HOME=/usr/local/cuda \
  cargo test --release -p infer --features cuda --test e2e_qwen35 -- --nocapture

scripts/profile_nsys_signal.sh pipeline-scheduler-stage-handles \
  --server-args "--model-path infer/models/Qwen3.5-4B --port 8000 --max-seq-len 8192" \
  --fast \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3.5-4B
```

### P4: NUMA and worker-lane policy

Files likely involved:

- `infer/src/runtime_topology.rs`
- `infer/src/request_handle.rs`
- `infer/src/backend/cuda/bootstrap.rs`
- `infer/src/main.rs`

Work:

- Keep one worker-local `WorkerDeviceContext` per CUDA ordinal.
- Ensure pinned host pools are allocated after CPU binding and before hot
  request flow.
- Route requests by NUMA route cost plus queue pressure.
- Keep NIC affinity in topology logs and metrics.

Exit gate:

- Startup log prints final worker topology.
- `numastat` metrics classify local/remote pages across all active worker
  NUMA nodes.

### P5: Metal bridge

Files likely involved:

- `infer/src/backend/metal/runtime.rs`
- `infer/src/backend/metal/scheduler.rs`
- `crates/mlx-sys/src/*`

Work:

- Represent MLX async eval completion as a coarse `MetalPipelineFence`.
- Keep CPU scheduler and postprocess stages separate from MLX execution.
- Add C++ bridge command-buffer completion only if trace shows host waits are
  material and MLX-level fences are too coarse.

Exit gate:

- Metal serving keeps current correctness and CI coverage.
- Any new wait is visible in metrics.

## Metrics Contract

Add or standardize:

```text
infer_pipeline_stage_duration_microseconds{stage=preprocess|route|h2d|compute|d2h|postprocess}
infer_pipeline_stage_queue_depth{stage=...}
infer_pipeline_fence_wait_microseconds{edge=h2d_to_compute|compute_to_d2h|d2h_to_cpu}
infer_pipeline_fence_poll_total{edge=...,outcome=ready|not_ready|error}
infer_pipeline_h2d_latency_microseconds
infer_pipeline_d2h_latency_microseconds
infer_pipeline_inflight{stage=...}
infer_scheduler_pipeline_stage_depth{stage=prefill|readback,state=queued|inflight}
infer_scheduler_pipeline_stage_total{stage=prefill|readback,state=queued|ready|completed}
infer_scheduler_gpu_bubble_microseconds
```

Use existing runtime topology metrics for:

- worker GPU ordinal
- worker NUMA node
- local and remote numastat pages
- route locality and migration/rebalance counters

## Verification Plan

### Local CPU/no-cuda gates

```bash
cargo test -p infer --no-default-features --features no-cuda runtime_topology -- --nocapture
cargo test -p infer --no-default-features --features no-cuda numa_router -- --nocapture
cargo test -p infer --no-default-features --features no-cuda server_metrics --lib
```

### CUDA type gate on non-GPU hosts

```bash
CUDARC_CUDA_VERSION=13010 \
  cargo check -p infer --no-default-features --features cuda,no-cuda
```

### CUDA runtime gate

Run on a CUDA host:

Terminal A starts the server:

```bash
CUDA_HOME=/usr/local/cuda cargo build --release -p infer --features cuda
./target/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --max-seq-len 8192
```

Terminal B drives the workload against that server:

```bash
scripts/bench_guidellm.sh pipeline-fence-smoke \
  --fast \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3-4B \
  --processor infer/models/Qwen3-4B \
  --trace-interval-ms 250
```

Trace with nsys:

```bash
scripts/profile_nsys_signal.sh pipeline-fence-smoke \
  --server-args "--model-path infer/models/Qwen3-4B --port 8000 --max-seq-len 8192" \
  --fast \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3-4B
```

Acceptance evidence:

- copy stream has H2D/D2H work.
- compute stream has prefill/decode kernels.
- H2D/D2H overlap compute when workload has independent work available.
- no unexpected `cudaDeviceSynchronize`.
- `cuStreamSynchronize` count does not increase except at intentional warmup
  or shutdown boundaries.
- emitted tokens and usage match baseline.

### Metal runtime gate

Run on Apple Silicon:

```bash
cargo test -p infer --release --no-default-features --features metal --lib
cargo build -p infer --release --no-default-features --features metal --bin metal_serve
```

Terminal A starts the canonical Metal server and runs startup warmup:

```bash
./target/release/metal_serve \
  --model-path mlx-community/Qwen3.6-35B-A3B-4bit \
  --warmup 1 \
  --warmup-max-new-tokens 1 \
  --port 8000
```

Terminal B submits a request and captures stats:

```bash
curl -fsS http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen3.6-35B-A3B-4bit","prompt":"Hello","max_tokens":1}'

curl -fsS http://127.0.0.1:8000/v1/stats | jq .
```

Acceptance evidence:

- Metal scheduler still starts and warmup emits a terminal delta.
- A live `/v1/completions` request returns a non-empty completion or a valid
  terminal response.
- CPU scheduler metrics and postprocess metrics remain populated.
- Any async-eval wait is visible as a stage/fence metric.

## Rollback Flags

Each phase should have a narrow off switch:

```text
INFER_PIPELINE_FENCES=0
INFER_COPY_STREAM_H2D=0
INFER_ASYNC_D2H_READBACK=0
INFER_PIPELINE_STAGE_METRICS=0
```

Flags must disable only the new pipeline behavior. They must not disable
existing NUMA routing, topology logs, or baseline scheduler behavior.

## Main Risks

- Silent race: CPU reads a pinned host slot before D2H completes.
- Silent race: compute consumes prompt/meta buffers before H2D completes.
- Buffer reuse race: a ring slot or scratch buffer is reused before its last
  consumer fence is ready.
- CUDA Graph regression: cross-stream waits or allocations leak into graph
  capture.
- Pageable-memory regression: async copy becomes effectively synchronous.
- Hidden sync regression: a model path keeps `ctx.sync()` in the token loop.
- False attribution: a run changes pipeline, scheduler policy, and kernel
  code at the same time, making the result unexplainable.

## Review Checklist

Before landing any implementation tranche:

- Every cross-stream dependency has an explicit fence.
- Every CPU read of GPU-produced data checks or waits on a fence.
- Every reusable buffer has a last-consumer fence.
- No graph-captured path performs copy-stream waits.
- Metrics identify where time is spent and where waits occur.
- Bench entry states whether performance evidence is real, pending remote, or
  intentionally deferred.
