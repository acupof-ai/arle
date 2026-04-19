# DFlash long-prompt prefill desyncs scheduler phase → `WrongPhase`

**Date**: 2026-04-19
**Status**: documented, not yet fixed. Not blocking — works for prompts ≤
`prefill_chunk_size` (512 default) and for the wins-doc canonical bench
params (prompt_tokens=1024 → ~1500 tokens, still >512 → latent but surfaced
only above ~43 chunks where it becomes frequent enough to always error).

## Context

Running `scripts/bench_guidellm.sh metal-m4max --profile concurrent ...` with
the canonical params from `docs/plans/guidellm-integration.md` §3
(`prompt_tokens=4096`, which Qwen3.5's tokenizer expands to ~21,856 actual
tokens) immediately errors with:

```
Metal complete_prefill failed for RequestId(N): request is in phase
Prefilling, expected Decoding
```

This blocks every request in the bench. Surfaces because DFlash is now
always-on by default (commit `47f958f`, 2026-04-19). Pre-existing bug — not
a regression from the flip — but the flip removed the gate (`active.is_empty()`)
that used to hide it.

## Root Cause

**Two independent `prefill_progress` trackers that can desync under DFlash.**

The Metal scheduler tracks prompt progress in two places:

1. **Scheduler-side** (`infer/src/backend/metal/scheduler.rs`):
   `MetalRequestState.prefill_progress`, advanced by `build_prefill_chunk`
   by `chunk_cap = min(prefill_chunk_budget(), 512)` per call. Phase
   transitions to `Decoding` only when this reaches `prompt_len`
   (`scheduler.rs:542-547`).

2. **Runtime-side** (`infer/src/backend/metal/request_state.rs`):
   `RequestState.prompt_progress()`, advanced by
   `ActiveMetalRequest.prefill_chunk(budget)`. Returns
   `emitted_token: Some(...)` when its own progress reaches `prompt_len`
   (`request_state.rs:215-248`).

The DFlash path in `runtime.rs:957-968` overrides the `budget` for the
runtime-side consumer:

```rust
if request.request_state.is_dflash_enabled() {
    let remaining = request.request_state.prompt_len()
        .saturating_sub(request.request_state.prompt_progress());
    budget = budget.max(remaining);
}
```

Reason for the override is legit: `qwen3_forward_with_hidden_states` captures
hidden states for all prompt positions, which chunked FFI prefill can't
produce. DFlash needs the whole prompt in one shot.

The problem: the budget override only widens the **runtime-side** consumer.
The scheduler-side `build_prefill_chunk` still uses `chunk_cap = 512` and
still only advances scheduler-side `prefill_progress` by 512 per call. So
on call #1:

- Scheduler gives `input_tokens = prompt[0..512]`, `prefill_progress = 512`,
  phase stays `Prefilling` (21856 > 512).
- Runtime overrides budget to 21344 (remaining), chews the entire prompt
  through the FFI, runtime-side says `emitted_token = Some(first_decode_tok)`.
- `runtime.rs:1018-1033` calls `scheduler.complete_prefill(req_id, token)`.
- `complete_prefill` (`scheduler.rs:370-391`) rejects: scheduler phase is
  still `Prefilling` because scheduler-side `prefill_progress = 512`, not
  21856.
- Request errors out; HTTP client sees 500.

For DFlash-disabled prompts, the budget is `chunk_cap = 512`, so both sides
advance in lockstep. For DFlash prompts ≤ 512 tokens, `remaining ≤ 512` so
the override is a no-op. Only DFlash prompts > 512 tokens are broken — which
is exactly what the canonical bench params hit.

## Fix (sketch — not yet implemented)

Two options, listed in order of least-risk:

### Option A (preferred) — scheduler-side fast-forward under DFlash

When `execute_prefill_chunk` takes the DFlash branch and actually consumes
more than `chunk_cap` tokens, fast-forward the scheduler-side
`prefill_progress` to match. New scheduler API:

```rust
// in MetalScheduler
pub fn fast_forward_prefill(&mut self, req_id: RequestId, new_progress: usize) {
    if let Some(state) = self.requests.get_mut(&req_id) {
        state.prefill_progress = new_progress;
        if state.prefill_progress >= state.prompt_len() {
            state.phase = MetalRequestPhase::Decoding;
            state.last_token = state.prompt_tokens.last().copied();
        }
    }
}
```

Call site in `runtime.rs::execute_prefill_chunk` after DFlash override
successfully produced `emitted_token`:

```rust
if request.request_state.is_dflash_enabled() {
    scheduler.fast_forward_prefill(req_id, request.request_state.prompt_len());
}
```

Minimal blast radius; keeps scheduler-side chunking semantics for every
non-DFlash path.

### Option B — bump default `prefill_chunk_size`

Set `prefill_chunk_size` to something huge (e.g. 65536) when DFlash is
enabled, so `build_prefill_chunk` and runtime consume the whole prompt in
one call. Cleaner conceptually — one tracker never lags — but changes the
bursty-dispatch behavior for all requests (including non-DFlash), and a
65 k prompt now generates a single giant FFI call which may blow latency
budgets for streaming. Rejected.

## Diagnostic evidence

Bench run showing the error (prompts of ~21856 tokens from the canonical
`prompt_tokens=4096` shorthand, which the Qwen3.5 tokenizer expands):

```
2026-04-19T22:34:12Z  ERROR metal_serve Metal complete_prefill failed
  for RequestId(3): request is in phase Prefilling, expected Decoding
```

All 16 warmup + 512 main requests errored identically. Killed the run and
re-ran with `prompt_tokens=1024` (wins-doc baseline params, ~1500 tokens)
— same code path but only ~3 chunks, so the failure is softer / less
reliable to reproduce but still latent. The 1024-token bench succeeded,
suggesting whatever short-prompt prefill path request_state takes happens
to not trigger the terminal-emit branch until scheduler catches up.

## Why not fix now

1. The wins-doc baseline (`prompt_tokens=1024`) doesn't reliably hit it.
2. No /loop-remaining optimization lever this work serves — it's a bug
   cleanup, not part of the Metal throughput arc.
3. Fix is a scheduler-API addition; needs a matched regression bench
   (long-prompt + short-prompt) to confirm Option A doesn't leak progress
   under non-DFlash paths.

Flagged for the next session that touches scheduler or runtime prefill
paths. Reference this entry when implementing Option A.

## Rule

**Two independent progress counters that both advance based on the same
source-of-truth (`prompt_len`) will desync the moment one of them takes a
fast-path that bypasses the other's increment logic.** When a codepath
widens one counter's stride (DFlash budget override), it must also
fast-forward any sibling counter, or the siblings must share a single
authority.

Applies generally to any two-tracker chunking setup (scheduler slot
lifecycle + FFI-side request state, CUDA scheduler + KV manager, etc.).

## Cross-refs

- `infer/src/backend/metal/runtime.rs:949-1033` — `execute_prefill_chunk`
  with the DFlash budget override (lines 957-968).
- `infer/src/backend/metal/scheduler.rs:519-572` — `build_prefill_chunk`
  with fixed `chunk_cap`.
- `infer/src/backend/metal/scheduler.rs:370-417` — `complete_prefill`
  phase check that rejects on desync.
- `infer/src/backend/metal/request_state.rs:215-248` — `prefill_chunk`
  that returns terminal `emitted_token` based on its own progress.
- `docs/experience/wins/2026-04-19-metal-qwen35-concurrent-dflash-default-on.md`
  — commit `47f958f` that surfaced this pre-existing bug.
