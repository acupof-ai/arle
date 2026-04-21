# Metal Continuous Batching Unification

**Status:** active  
**Owner:** infer / Metal runtime  
**Scope:** `infer/src/backend/metal/*`, `crates/mlx-sys/src/mlx_qwen35_model.cpp`  
**Related:** `docs/plans/p99-unified-mixed-batch.md`, `projects/mlx-backend-roadmap.md`

## Why

Metal serving has already converged on the right top-level scheduling shape:

- one scheduler
- decode-first ticks
- a shared per-tick token budget
- chunked prefill

But the execution path is still split in the wrong place:

- the scheduler emits one logical mixed tick (`decode` + optional `prefill`)
- runtime still contains a **Qwen3-only** fused mixed fast path
- Qwen3.5 has **varlen packed decode**, but not true mixed prefill+decode
- DFlash is still a separate special path layered on top

That leaves us with scheduler-level unification but execution-level half-state.
The result is complexity without a clean model contract.

## External guidance we should copy

This plan follows the consistent cross-project pattern from:

- **vLLM**: decode-first scheduling with one shared token budget and chunked prefill
- **SARATHI**: treat mixed batching as "one prefill chunk + as many decode rows as possible", not as a separate scheduler mode
- **SGLang**: unified scheduling first; only split prefill/decode later if tail latency forces it
- **vllm-metal**: real continuous batching on Metal requires paged/block-managed KV plus varlen kernels, not just a scheduler knob

What that means for this repo:

1. keep one unified scheduler
2. keep decode-first priority explicit
3. treat mixed prefill+decode as the normal execution shape of that scheduler
4. stop keeping model-specific mixed policy in the runtime
5. only then extend backend/model hot paths so more models can actually execute the mixed plan

## Current local state

### Already correct

- `scheduler.rs` already models a tick as `decode` then optional `prefill`
- `max_batch_tokens` is already the shared token budget
- `request_state.rs` already has:
  - Qwen3 packed mixed execution
  - Qwen3.5 varlen packed decode
  - Qwen3.5 session-owned chunked prefill

### Still wrong

- `runtime.rs` decides "mixed batching" through a `guard_qwen3_mixed_batch(...)` branch
- mixed execution is not a generic request-state contract
- Qwen3.5 packed decode still samples in the same call that launches the batch
- Qwen3.5 still lacks a fused mixed prefill+decode path
- DFlash still overrides prefill behavior with a side-path instead of integrating through one mixed execution contract

## Architecture target

## Can Qwen3 and Qwen3.5 be fully unified?

**No.**

What can be unified:

- scheduler policy
- request lifecycle
- decode-first mixed-tick planning
- mixed-batch execution contract
- prefix/prefill/decode runtime orchestration
- metrics, teardown, and backpressure behavior

What should **not** be force-unified:

- the actual forward implementation
- cache/state layout details
- Qwen3.5 recurrent/GDR handling
- Qwen3.5 compiled C++ step model internals

Why:

- **Qwen3** currently uses the Rust/MLX path for its packed mixed execution.
- **Qwen3.5** uses the compiled C++ step/session model and carries both full-attention KV state and recurrent GDR state.
- Qwen3.5 varlen packed decode is already shaped around left-padding + additive mask + per-row RoPE offsets. Qwen3 mixed execution has a different implementation and state contract today.

So the right target is:

- **one scheduler**
- **one runtime tick contract**
- **one mixed execution interface**
- **two model-specific implementations behind that interface**

This plan treats that as a hard constraint, not a later cleanup.

## Architecture target

### Scheduler

Keep the scheduler simple:

- one waiting queue
- one running set
- one decode-first `MetalScheduleStep`
- one shared token budget
- at most one prefill chunk per tick

Do **not** add more scheduler modes.

### Runtime

The runtime should expose one execution entrypoint for a mixed tick:

- decode rows
- optional prefill row
- one generic "attempt mixed execution" contract
- fallback to `decode then prefill` only if the model path cannot fuse the tick

This keeps policy in the scheduler and capability in request-state/model code.

### Model contracts

- **Qwen3**
  - keep packed mixed batch support
  - route it through the generic mixed contract, not a Qwen3-specific runtime branch
- **Qwen3.5**
  - keep session prefill and varlen packed decode
  - extend the C++ bridge so packed execution can eventually consume one prefill row plus decode rows under one contract
- **DFlash**
  - remain optional
  - stay behind the same runtime tick planner instead of creating a second scheduling model

## Phases

### Phase 1 — Delete the runtime special case

**Goal:** runtime no longer knows about a Qwen3-specific mixed path.

Changes:

- introduce one generic `try_mixed_batch(...)` request-state API
- move model dispatch into `request_state.rs`
- rename runtime helpers so they talk about "mixed batch" generically
- keep current behavior: Qwen3 returns `Some(...)`, other model mixes return `None`

Acceptance:

- no `Qwen3`-named mixed-batch branch remains in `runtime.rs`
- behavior stays unchanged for Qwen3
- non-Qwen3 requests fall back cleanly to `decode + prefill`

### Phase 2 — Clean mixed-step boundaries

**Goal:** runtime executes one mixed tick shape with one teardown/error path.

Changes:

- make mixed-tick activation, client-drop handling, and finalize/requeue logic uniform
- reduce duplicate "detach request / process token / finalize" code between fused and fallback paths
- keep prefix publish and metrics on one path

Acceptance:

- one mixed executor path in runtime
- one fallback path
- no duplicated request detach/finalize logic for Qwen3-only mixed handling

### Phase 3 — Qwen3.5 true mixed execution

**Goal:** Qwen3.5 can fuse decode rows with one prefill chunk instead of only doing separate packed decode and chunked prefill.

Required backend work:

- extend the C++ bridge with a packed mixed API
- support per-row query length, per-row cache write position, additive mask, and per-row RoPE offsets
- preserve the session-owned prefill rule so prefill does not round-trip full KV state to Rust every chunk

Acceptance:

- Qwen3.5 mixed ticks no longer always fall back
- Qwen3.5 mixed path keeps varlen correctness

### Phase 4 — Revisit DFlash and budget defaults

**Goal:** after the execution contract is unified, decide whether DFlash and batch-budget defaults belong in the main line as-is.

Only after Phases 1-3:

- bench long-prompt sync and concurrent profiles
- decide whether `max_batch_tokens` default should stay conservative or scale up
- decide whether DFlash should become the default decode path for eligible Qwen3.5 traffic

## Non-goals

- adding a second Metal scheduler
- prefill/decode disaggregation
- copying `mlx_lm.server` design directly
- introducing more runtime-only mixed caps

## Immediate next step

Implement **Phase 1** now:

- write the generic mixed execution contract in `request_state.rs`
- delete the Qwen3-named mixed branch from `runtime.rs`
- run targeted Metal verification and a local regression bench
