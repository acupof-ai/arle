# Scheduler Preprocess Pipeline

Date: 2026-05-12
Status: implemented locally; performance license pending
Scope: CUDA serving scheduler hot path. Metal/CPU may reuse the intake API
shape later, but this plan is licensed only for the CUDA continuous-batching
runtime.

## Goal

Move request CPU work out of the GPU scheduling critical path and make CPU and
GPU progress concurrently:

```text
HTTP/CLI -> Preprocess -> SchedulerCore -> CpuPlanWorker -> GpuExecutor -> EmitWorker
```

The first required step is explicit: **tokenization becomes part of the
preprocess stage**. Requests entering the scheduler should already carry
`prompt_tokens`, with scheduler-side tokenization retained only as a fallback
for tests, warmup paths, and direct internal submitters.

Success means fewer GPU bubbles caused by CPU-side request intake, admission,
planning, metadata construction, or token readback. It does not claim a
throughput win until an nsys trace proves the CPU side was blocking GPU work.

## Implementation Status 2026-05-12

Stages 1-4 are now represented in the runtime as explicit boundaries:

- **Stage 1 preprocess:** HTTP prompt tokenization runs before scheduler
  submission behind a bounded semaphore, records preprocess queue/wait/tokenize
  telemetry, and sends `IncomingRequest::prompt_tokens = Some(...)`.
- **Stage 2 snapshot planning:** CUDA `step()` now builds a cheap
  `SchedulerSnapshot`, advances a scheduler epoch, produces a `CandidatePlan`,
  validates the epoch, counts accepted/stale plans, and falls back on stale
  candidates.
- **Stage 3 metadata staging:** accepted candidate plans now carry a
  `PreparedHostMetadata` descriptor with decode rows, prefill rows, prefill
  token count, and page-table row count before launch dispatch. Model-specific
  buffer uploads still happen inside the existing decode/prefill contexts until
  an nsys gate proves this is worth pushing into a dedicated copy-stream path.
- **Stage 4 GPU command split:** CUDA launch dispatch now goes through an
  explicit `GpuCommand` boundary, and readback timing is reported as GPU
  completion wait telemetry. SchedulerCore remains the only writer of request,
  prefix-cache, and paged-KV state.

This is an architecture and correctness landing, not a performance conclusion.
The next required evidence is the CUDA GuideLLM + nsys matrix in this document.

## Current State

CUDA already has partial overlap:

- `pending_decode` and `pending_prefill` live across loop turns.
- `run_inner()` drains request/coordinator/emit events and assigns slots before
  `step()` reads back previous GPU work.
- Qwen3 has async prefill launch/complete and async greedy token readback.
- Emit/detokenization already runs in `EmitWorker`.

But the loop is still one single-writer thread:

```text
drain_request_rx
  -> normalize_waiting_request      # may tokenize today
  -> assign_slots / prefix lookup
  -> readback previous GPU work
  -> plan_step
  -> launch decode/prefill/mixed
  -> dispatch emits
  -> cleanup
```

This preserves state safety, but CPU work still shares the same scheduling
thread that must keep the GPU fed. The current `RequestHandleInferenceEngine`
also submits `prompt_tokens: None`, so scheduler intake may tokenize under
`normalize_waiting_request`.

## Non-Goals

- Do not turn scheduler state into `Arc<Mutex<_>>`.
- Do not let worker threads mutate `states`, `prefix_cache`, `block_to_pages`,
  `block_owner_slots`, or `paged_kv_pool`.
- Do not add synchronous KV swap or host readback in `assign_slots`; prior
  evidence shows synchronous T0/T1 copies destroy scheduler/GPU overlap.
- Do not claim this is production-useful without a trace showing CPU-side
  scheduler ranges are material and GPU idle time drops.

## Target Architecture

### Threads and Ownership

```text
HTTP / CLI task
  owns request parsing and response channel creation

Preprocess workers
  own tokenizer clones and request normalization
  output PreprocessedRequest

SchedulerCore thread
  only writer for scheduler state
  commits completions, slots, radix refs, paged-KV ownership, metrics
  accepts or rejects prepared plans by epoch

CpuPlanWorker
  consumes read-only SchedulerSnapshot
  prepares CandidatePlan and optional host metadata
  never mutates scheduler state

GpuExecutor
  owns model, CUDA context, streams, decode/prefill buffers
  launches accepted commands and returns CompletionEvent

EmitWorker
  unchanged: detokenization and streaming deltas
```

### Data Flow

```text
CompletionRequest
  -> PreprocessJob
  -> PreprocessedRequest { prompt, prompt_tokens, length contract, session_id, sampling, stop }
  -> SchedulerCore waiting queue
  -> SchedulerSnapshot
  -> CandidatePlan { epoch, decode rows, prefill rows, budget reservations, metadata descriptors }
  -> GpuCommand
  -> CompletionEvent
  -> SchedulerCore commit
  -> EmitWorker
```

The scheduler remains the authority. Worker outputs are proposals until
`SchedulerCore` validates their epoch and commits them.

## Stage 1: Preprocess Includes Tokenization

### Contract

Introduce a logical preprocess stage before scheduler submission:

```rust
struct PreprocessedRequest {
    prompt: String,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    sampling: SamplingParams,
    stop: Option<Vec<String>>,
    speculative: Option<RequestSpecConfig>,
    priority: RequestPriority,
    session_id: Option<SessionId>,
    delta_tx: UnboundedSender<CompletionStreamDelta>,
    trace_context: Option<SpanContext>,
}
```

The public scheduler intake can still accept `IncomingRequest`, but the
preferred path constructs it with:

```rust
IncomingRequest {
    prompt,
    prompt_tokens: Some(prompt_tokens),
    ...
}
```

### Implementation Shape

Minimal first tranche:

1. Add a private helper on `RequestHandleInferenceEngine`:

   ```rust
   fn preprocess_request(&self, req: &CompletionRequest) -> Result<Vec<u32>>;
   ```

2. `submit_request` calls `preprocess_request`, passes
   `prompt_tokens: Some(prompt_token_ids.clone())`, and returns
   `prompt_token_ids`.
3. `complete()` uses the returned `prompt_token_ids` for
   `CompletionOutput.prompt_token_ids` instead of tokenizing a second time.
4. `complete_stream()` also preprocesses before submit, but discards the token
   vector after constructing `IncomingRequest`.
5. Scheduler fallback stays:
   `normalize_waiting_request()` may tokenize only when `prompt_tokens` is
   `None`.

This is intentionally small: it moves tokenization out of the scheduler loop
for normal engine traffic without changing slot admission, prefix cache, GPU
launch, or response semantics.

### Follow-Up Worker Pool

After the minimal path is green, replace inline preprocessing with bounded
workers:

```text
request task -> preprocess_tx -> N preprocess workers -> scheduler submit
```

Use a bounded queue. If preprocess is slower than arrival rate, backpressure
should happen before scheduler enqueue, not after GPU slots are starved.

## Stage 2: Snapshot-Based CPU Planning

### Snapshot

SchedulerCore emits a read-only snapshot while GPU work from the prior tick is
in flight:

```rust
struct SchedulerSnapshot {
    epoch: SchedulerEpoch,
    waiting_head: Vec<WaitingRequestView>,
    active_slots: Vec<ActiveSlotView>,
    free_slots: Vec<usize>,
    prefix_index_view: PrefixIndexView,
    page_budget_view: PageBudgetView,
    config: SchedulerConfigView,
}
```

Snapshots must be cheap. They should contain IDs, lengths, counters, page
budget summaries, and small prefix references, not copies of model state or
large token vectors unless needed for prefix lookup.

### Candidate Plan

CpuPlanWorker returns:

```rust
struct CandidatePlan {
    epoch: SchedulerEpoch,
    plan: LogicalServePlan,
    reservations: Vec<ReservationIntent>,
    host_metadata: Option<PreparedHostMetadata>,
}
```

The plan is invalid if any epoch component changed:

- waiting queue epoch
- active slot epoch
- KV/page budget epoch
- prefix/radix epoch
- config epoch

Invalid plans are dropped and SchedulerCore falls back to current synchronous
planning for that tick.

### Why Snapshot First

This preserves the existing single-writer invariant while still overlapping
CPU planning with GPU compute. It also gives a kill switch: if snapshot
planning is stale too often, the runtime can disable it and keep the old path.

## Stage 3: Metadata Staging

Once CandidatePlan is accepted, move CPU metadata construction out of launch:

- decode row lists
- slot indices
- position offsets
- page-table descriptors
- `qo_indptr`, `kv_indptr`, `kv_last_page_len`
- prefill chunk row descriptors

Use double-buffered host metadata:

```text
host_meta[N+1] prepared while GPU runs N
copy stream uploads meta[N+1]
compute stream waits on copy event
```

Do not let metadata staging allocate unbounded per request. All buffers are
sized by scheduler config (`max_slots`, `max_num_batched_tokens`,
`max_prefill_tokens`) and reused.

## Stage 4: GpuExecutor Command/Completion Split

Only after Stages 1-3 are traced and useful, split GPU execution from
SchedulerCore:

```rust
enum GpuCommand {
    Decode(DecodeCommand),
    Prefill(PrefillCommand),
    Mixed(MixedCommand),
    SpecVerify(SpecVerifyCommand),
}

enum CompletionEvent {
    DecodeReady(DecodeCompletion),
    PrefillReady(PrefillCompletion),
    MixedReady(MixedCompletion),
    Failed { command_id: u64, error: String },
}
```

GpuExecutor owns CUDA context and model buffers. SchedulerCore owns logical
state. Completion events carry only the data needed to commit scheduler state:
sampled tokens, logprobs, completed prefill rows, and failure identity.

This is the real architecture split, but it should be the last tranche because
it changes failure handling and shutdown semantics.

## Correctness Invariants

1. SchedulerCore remains the only writer of request phases and slot lifecycle.
2. Prefix cache refs are acquired and released only during SchedulerCore commit.
3. Paged-KV page ownership changes only during SchedulerCore commit.
4. CpuPlanWorker output is never trusted without epoch validation.
5. A stale plan is a performance miss, not a correctness failure.
6. Tokenization must use the same tokenizer instance/fingerprint as the
   scheduler prefix namespace.
7. Empty tokenization is rejected in preprocess with the same finish semantics
   as current `normalize_waiting_request`.
8. Streaming cancellation must still be observed before committing a request to
   an active slot.

## Metrics and Trace Requirements

Add or extend metrics before claiming wins:

- `preprocess_queue_depth`
- `preprocess_wait_us`
- `preprocess_tokenize_us`
- `scheduler_snapshot_us`
- `cpu_plan_us`
- `cpu_plan_stale_count`
- `cpu_plan_accept_count`
- `gpu_command_queue_depth`
- `gpu_completion_wait_us`
- existing `step_admission`, `step_plan`, `step_decode_kernel_launch`,
  `step_prefill_kernel_launch`, `step_dispatch_emits`

License gates:

| Gate | Proceed | Kill / revise |
| --- | --- | --- |
| Tokenization offload | scheduler `step_admission` p95 drops or no regression with cleaner architecture | tokenization cost is noise and code adds queue latency |
| Snapshot planning | accepted plans > 90% under target workload and GPU idle drops | stale plans frequent or no GPU bubble reduction |
| Metadata staging | launch-side CPU time drops and copy stream overlaps compute | H2D staging adds syncs or allocation churn |
| GpuExecutor split | GPU idle drops > 10% relative under CPU-bound trace | throughput flat/regresses and complexity rises |

Use wall-clock and per-request framing as ground truth. NVTX sub-window wins are
not enough.

## Benchmark Plan

Read `docs/bench-and-trace-spec.md` before running performance gates.

Minimum A/B matrix:

1. Short prompt, high concurrency: agent-style W3/W4 traffic.
2. Long prompt, c=4: long-context Phase 1 workload.
3. Mixed prefill/decode: prefix-hit plus cold request blend.
4. Qwen3 and Qwen3.5 separately, because Qwen3.5 currently has weaker async
   readback behavior.

Required evidence:

- `guidellm` fixed-concurrency before/after.
- `/v1/stats` plan-label distribution.
- nsys with NVTX phase breakdown.
- GPU idle time from CUDA API/GPU summary.
- Log counters for stale/accepted CPU plans.

Bench entry required for any runtime change under `infer/src/`, with a
`pending-remote` stub if the CUDA trace cannot run locally.

## Rollout Plan

### P0: Inline Preprocess Tokenization

Files likely touched:

- `infer/src/server_engine/request_handle_engine.rs`
- targeted tests near `server_engine` or `scheduler` intake

Acceptance:

- normal `RequestHandleInferenceEngine` submit sends `prompt_tokens: Some`.
- `complete()` reuses the same token vector for trajectory output.
- scheduler fallback for `prompt_tokens: None` remains tested.

### P1: Bounded Preprocess Worker Pool

Files likely touched:

- request handle engine / HTTP intake wrapper
- metrics
- graceful shutdown path

Acceptance:

- bounded queue backpressure works.
- cancellation before preprocess completion does not enter scheduler active
  state.
- tokenization errors produce the same finish behavior as today.

### P2: Snapshot + CandidatePlan

Files likely touched:

- `infer/src/scheduler/cuda/runtime/scheduler_loop.rs`
- `infer/src/scheduler/cuda/execution.rs`
- new `scheduler/cuda/preplan.rs`
- metrics

Acceptance:

- old synchronous plan path remains fallback.
- stale plans are counted and dropped.
- no scheduler state is shared mutably across threads.

### P3: Host Metadata Staging

Files likely touched:

- decode/prefill metadata builders
- Qwen3 decode/prefill contexts
- CUDA buffer ownership wrappers

Acceptance:

- no per-step heap/GPU allocation growth.
- copy stream events correctly order H2D before compute.
- nsys shows overlap, not hidden synchronization.

### P4: GpuExecutor Split

Files likely touched:

- scheduler core loop
- model forward command wrappers
- shutdown/error handling
- tests for dropped receiver and failed GPU command

Acceptance:

- GPU executor owns CUDA context/model buffers.
- SchedulerCore can drain completions and shut down deterministically.
- failure path preserves current request finish/error semantics.

## Risks

- **Stale plans:** high churn queues can invalidate snapshots. Mitigation:
  bounded lookahead of one tick and fallback.
- **Hidden syncs:** D2H/H2D helper APIs may synchronize. Mitigation: nsys every
  stage and forbid host materialization in planning.
- **Tokenization drift:** preprocess workers must use the same tokenizer and
  namespace fingerprint as scheduler prefix cache. Mitigation: pass cloned
  `Tokenizer`, not a path reloader.
- **Queue latency:** preprocessing queues can hurt low-concurrency TTFT.
  Mitigation: inline fast path when queue is empty or concurrency is low.
- **State leaks on cancellation:** requests may cancel after preprocess but
  before admission. Mitigation: SchedulerCore checks `delta_tx.is_closed()`
  before active-slot commit, as it does today.

## Decision

Stages 1-4 have landed as a default-safe pipeline boundary while preserving the
single-writer scheduler invariant. Treat the current implementation as
correctness and observability scaffolding until GuideLLM + nsys prove that the
CPU-side ranges are wall-clock material and that GPU idle time drops.
