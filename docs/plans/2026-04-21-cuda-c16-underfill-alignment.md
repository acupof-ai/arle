# Plan: CUDA c16 active-set underfill alignment

Status: active  
Created: 2026-04-21  
Owner: ckl  
Scope: `infer/src/scheduler/cuda/*`, `infer/src/scheduler/types.rs`, bench/trace wiring

## Goal

Fix the current CUDA c16 long-context scheduler underfill so the runtime
behaves like a real continuous-batching controller under pressure:

- `waiting` requests should turn into a multi-request prefill batch
- `running_batch` should stay meaningfully populated during decode
- GuideLLM c16 numbers should be backed by same-run service trace, not by guesswork

## Current diagnosis

The current local diagnosis is already stable:

- benchmark: `bench-output/2026-04-21-cuda-l4-c16-mixed-reserve/`
- trace smoke: `bench-output/2026-04-21-cuda-l4-trace-serial-smoke/`
- error record: [`../experience/errors/2026-04-21-cuda-c16-serial-trace-active-underfill.md`](../experience/errors/2026-04-21-cuda-c16-serial-trace-active-underfill.md)

Observed shape:

- c16 fast run produced low throughput despite non-zero KV pressure
- same-run service trace showed `peak waiting=15`, `peak active=1`, `peak kv_util=93.0%`

That means the primary problem is not GuideLLM output accounting. The scheduler
is underfilling the active set.

## SGLang findings to align with

Source of truth inspected locally from `/tmp/sglang`:

- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/schedule_policy.py`
- `python/sglang/srt/managers/schedule_batch.py`
- `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`
- `python/sglang/srt/model_executor/pool_configurator.py`

Relevant behavior:

1. `scheduler.py` is a single-thread control loop over `waiting_queue` and `running_batch`.
2. Each loop tries `get_new_batch_prefill()` first, then continues decode on `running_batch`.
3. `update_running_batch()` handles decode-memory checks and retracts requests back to queue.
4. `PrefillAdder` carries two independent controls:
   - `max_prefill_tokens`: batch-wide prefill budget
   - `chunked_prefill_size`: per-request truncation limit
5. Chunked requests are explicitly re-added and remain schedulable until finished.
6. KV capacity is not sized in the scheduler. SGLang sizes `max_total_num_tokens`
   in the model-runner memory-pool path from:
   - post-weight available GPU memory
   - `mem_fraction_static`
   - model KV bytes-per-token
7. SGLang docs explicitly treat activations and CUDA-graph buffers as
   headroom outside the KV pool. The documented tuning rule is to leave
   roughly `5-8 GB` free for those transient/runtime allocations.

## Gap in our runtime

The current Rust scheduler already has the right top-level shape:

- `waiting`
- `prefill_queue`
- `running_batch`
- decode retract → requeue

But one key semantic is wrong:

1. `PrefillBudget::from_scheduler()` collapses batch budget to
   `min(max_prefill_tokens, chunked_prefill_size)`.
2. On the c16 config (`chunked_prefill_size=4096`, `max_prefill_tokens=16384`),
   this reduces the total prefill budget to `4096`.
3. The planner therefore admits at most one 4096-token request per tick.
4. Under long-prompt pressure, `waiting -> active` becomes structurally narrow,
   which matches the observed `peak active=1`.

That admission bug was real and is now fixed, but deeper tracing showed a
second bottleneck that now dominates the c16 long-context shape:

5. We size the long-lived CUDA mixed/decode workspace directly from
   `max_prefill_tokens`, not from observed mixed-step demand.
6. On the current c16 config, that means persistent workspace is reserved for a
   `16384`-token mixed prefill worst case even though the common steady-state
   path is bounded by `chunked_prefill_size=4096`.
7. For Qwen3-4B on L4 this is large enough to materially shrink the T0 KV pool:
   - `max_prefill_tokens=16384` pushes the mixed/prefill runtime reservation to
     roughly `3.3 GiB`
   - `chunked_prefill_size=4096` would be closer to roughly `1.6 GiB`
8. That delta alone is enough to consume more than ten thousand KV tokens, so
   the active-set ceiling looks like a scheduler issue even when the real cause
   is capacity lost to persistent workspace.

This is the current leading architecture gap versus SGLang: our hot path keeps
worst-case mixed buffers resident; SGLang primarily sizes KV statically and
leaves activation/runtime memory in a looser headroom bucket.

## How this can fail

This refactor is safe only if all of these are checked:

1. Workspace sizing must use the real batch-wide prefill budget, or mixed
   prefill workspace can become too small and fail mid-request.
2. Per-request prefill truncation must still honor `chunked_prefill_size`, or
   one request can monopolize a whole tick and recreate the old serial shape.
3. If `max_prefill_tokens < chunked_prefill_size`, a request must still be able
   to run with a smaller truncated chunk; otherwise the planner can deadlock by
   rejecting every long request.
4. The bench wrapper must continue to produce service trace for every run, or
   we lose the ability to distinguish scheduler underfill from measurement bugs.
5. If we shrink persistent mixed buffers without changing the mixed planner, a
   valid multi-request mixed step can start failing at runtime when total
   admitted prefill tokens exceed the new buffer bound.
6. If we keep persistent buffers sized for worst-case `max_prefill_tokens`, c16
   long-context throughput can stay pinned by KV-capacity loss even after the
   scheduler itself is logically correct.

## Implementation plan

### 1. Unify prefill budget semantics

Make one semantic path only:

- `chunked_prefill_size` = per-request chunk cap
- `max_prefill_tokens` = total prefill tokens available in one scheduler tick

Code changes:

- remove the `min(max_prefill_tokens, chunked_prefill_size)` collapse
- clamp each planned request by:
  - remaining request tokens
  - `chunked_prefill_size`
  - current remaining batch budget
- use the same helper for queued-prefill and waiting-admission paths

### 2. Align mixed workspace sizing

Mixed prefill workspace should be sized by the true batch-wide budget, not the
single-request chunk cap.

Code changes:

- size `prepare_mixed_decode_context(...)` from `max_prefill_tokens`
- keep one planner path for decode-only / mixed / prefill-only, no secondary
  serial-prefill escape hatch

Status update:

- this step fixed the previous serial-admission bug, but it also exposed the
  larger capacity problem: keeping a `max_prefill_tokens`-sized mixed context
  resident all the time is too expensive on L4 for the c16 / 4096-in workload.

### 2b. Split persistent workspace from burst headroom

This is now the main remaining alignment task.

Goal:

- keep one mixed path
- stop paying the full `max_prefill_tokens` memory cost as a long-lived buffer
- preserve correctness for the occasional wider mixed step

Code direction:

- trace the actual peak mixed prefill tokens seen in steady-state decode-active
  runs
- separate "must stay allocated every tick" buffers from "can be grown or paid
  via headroom only when needed"
- reduce persistent buffer sizing to the common mixed bound, and treat larger
  one-off mixed steps as explicit burst cases instead of the default resident
  footprint

### 3. Extend service trace with scheduler-owned occupancy

The current `/v1/stats` trace catches `waiting` and `active`, but we also need
the scheduler's own internal occupancy to diagnose underfill.

Code changes:

- add scheduler-facing gauges / summary fields for:
  - `prefill_queue`
  - `running_batch`
- keep them in `/v1/stats` so `scripts/bench_guidellm.sh` captures them in the
  same trace loop as bench output

### 4. Validate in the order that reduces ambiguity

1. `cargo check -p infer --no-default-features --features cuda,no-cuda`
2. focused scheduler tests
3. serial c16 fast smoke with trace
4. inspect:
   - `peak active`
   - `peak running_batch`
   - `peak prefill_queue`
   - headline tok/s / req/s
5. if the shape improves and stays stable, run the clean c16 benchmark and
   land a wins entry

## Acceptance criteria

This plan counts as closed only if all hold:

1. Budget semantics are single-source and documented consistently.
2. A c16 fast trace smoke no longer shows the pathological
   `waiting >> active` shape with `peak active=1`.
3. Bench and trace artefacts are emitted in the same output directory.
4. The change lands with an experience entry under `wins/` or `errors/`.

## Immediate next move

The admission fix is done. The current next move is step 2b:

1. measure the real mixed-step token width under c16 pressure
2. shrink or make dynamic the persistent mixed workspace
3. rerun clean c16 trace + bench to see how much KV capacity returns
