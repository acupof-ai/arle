# DFlash long-prompt prefill: scheduler-side fast-forward

**Date**: 2026-04-20
**Status**: fix landed; bench gated `pending-remote` (Metal toolchain
missing on this Mac, see Problems below).

## Context

Running `scripts/bench_guidellm.sh metal-m4max` at the canonical params
(`prompt_tokens=4096` → Qwen3.5 tokenizer expands to ~21,856 tokens)
erroring every request with:

```
Metal complete_prefill failed for RequestId(N): request is in phase
Prefilling, expected Decoding
```

Full root cause documented in
`docs/experience/errors/2026-04-19-dflash-long-prompt-prefill-chunking-desync.md`.
Short form: the Metal scheduler keeps a `prefill_progress` counter
independent of the runtime-side `RequestState::prompt_progress()`. DFlash
widens the **runtime-side** budget to consume the whole prompt in one
FFI call (because `qwen3_forward_with_hidden_states` captures hidden
states for every position), but the **scheduler-side**
`build_prefill_chunk` still advances by `chunk_cap = min(budget, 512)`
per call. For prompts > 512, the scheduler stays in `Prefilling` while
the runtime already emitted a terminal decode token — `complete_prefill`
then rejects with `WrongPhase`.

## Root Cause

Two sibling counters that both claim ownership of the same source of
truth (`prompt_len`) desync the moment one takes a fast path that
bypasses the other. See the error doc's §Rule for the general principle.

## Fix

Option A from the error doc — the minimum-blast-radius variant that
keeps scheduler-side chunking semantics intact for every non-DFlash
path.

**`infer/src/backend/metal/scheduler.rs`** — new public method
`MetalScheduler::fast_forward_prefill(req_id, new_progress)`:

- Never rewinds progress (stale fast-forward is a no-op).
- Transitions phase to `Decoding` when progress reaches `prompt_len`.
- Sets `last_token` to the terminal prompt token if unset, matching
  `build_prefill_chunk`'s completion branch.
- Silently ignores unknown `req_id` (symmetric with how the runtime
  treats missing requests elsewhere).

**`infer/src/backend/metal/runtime.rs::execute_prefill_chunk`** — call
site gated on `is_dflash_enabled()`, placed after the runtime-side
prefill succeeds (`emitted_token.is_some()`) and before
`scheduler.complete_prefill(...)`:

```rust
if let Some(request) = active.get(&req_id)
    && request.request_state.is_dflash_enabled()
{
    let prompt_len = request.request_state.prompt_len();
    scheduler.fast_forward_prefill(req_id, prompt_len);
}
```

Non-DFlash prefill keeps the old single-counter chunked path unchanged.

## Acceptance

1. Scheduler unit tests: **11/11 passing** under
   `cargo test --release --no-default-features --features no-cuda -p infer --lib backend::metal::scheduler`
   (8 pre-existing + 3 new):
   - `fast_forward_prefill_transitions_to_decoding_when_prompt_complete`
     — exercises the exact scenario from the error doc: a chunk_cap < prompt_len
     prefill followed by a one-shot fast-forward + `complete_prefill`.
   - `fast_forward_prefill_is_a_noop_for_unknown_request` — ignore-missing
     symmetric with runtime-side behavior.
   - `fast_forward_prefill_does_not_regress_progress` — guards against a
     stale smaller value accidentally rewinding the scheduler.
2. Full `infer` lib tests under `no-cuda`: **302/302 passing** — no
   regression on non-DFlash paths, since the fast-forward call is gated
   on `is_dflash_enabled()` and the scheduler method is unused outside
   the DFlash branch.
3. Non-DFlash behavior preserved by construction: `build_prefill_chunk`
   is untouched, `fast_forward_prefill` is only called from the DFlash
   `is_dflash_enabled()` branch in `execute_prefill_chunk`.

## Problems

- **Metal build blocked on this host**: Apple Xcode 26.4.1 on this M4
  Max ships without the Metal Toolchain; `xcrun metal` errors with
  `cannot execute tool 'metal' due to missing Metal Toolchain; use:
  xcodebuild -downloadComponent MetalToolchain`. `cargo build --features
  metal` cannot compile `mlx-sys`'s `.metal` shaders locally, so the
  end-to-end bench verifying the error no longer surfaces has to run on
  a machine with the toolchain installed. Scheduler + runtime Rust code
  paths type-check cleanly under `--features no-cuda` (scheduler is
  always-on per `infer/src/backend/metal/AGENTS.md` §Feature gating) and
  all unit tests pass. The `runtime.rs` diff follows the same
  `is_dflash_enabled()` / `prompt_len()` pattern as the pre-existing
  DFlash budget override 20 lines above; no new API surface, no new
  lifetime concerns.
- **Bench status: `pending-remote`** — next Mac run with Metal Toolchain
  should execute `scripts/bench_guidellm.sh metal-m4max` at the
  canonical `prompt_tokens=4096` params and confirm the 500s are gone.
  A regression bench against the c=1/2/4 baselines under
  `prompt_tokens=1024` (wins-doc canonical) should show 0% delta — the
  fast-forward path is a pure scheduler-state sync, no MLX ops.

## Rule

**When a codepath widens one progress counter via a fast-path, it must
fast-forward every sibling counter that shares `prompt_len` as its
source of truth** — or the siblings must share a single authority.
Preferred: add a single-purpose fast-forward API that makes the sync
explicit at the call site. Avoid invisible state sync via a second
"one-big-budget" path because it mutates behavior for every non-fast
caller.

## Cross-refs

- `docs/experience/errors/2026-04-19-dflash-long-prompt-prefill-chunking-desync.md`
  — full root cause and Option A/B comparison.
- `infer/src/backend/metal/scheduler.rs` — `fast_forward_prefill`
  definition and three unit tests.
- `infer/src/backend/metal/runtime.rs::execute_prefill_chunk` —
  DFlash-gated call site directly before `complete_prefill`.
- `docs/experience/wins/2026-04-19-metal-qwen35-concurrent-dflash-default-on.md`
  — flip that made DFlash-always-on the default and surfaced this
  pre-existing bug at canonical bench params.
