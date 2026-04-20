# DFlash long-prompt prefill: scheduler-side fast-forward

**Date**: 2026-04-20
**Status**: fix landed; end-to-end bench ✅ (Metal Toolchain was in fact
installed — rebuild was incremental since mlx-sys cache intact; see
§End-to-end validation below).

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

## End-to-end validation

Ran `metal_serve` with `--dflash-draft-model
z-lab/Qwen3.5-4B-DFlash` on M4 Max 40-core, macOS 26.3.1, Qwen3.5-4B-MLX-4bit
and drove it with `guidellm benchmark run`.

**Reproduce-the-bug smoke**: one synchronous `POST /v1/chat/completions`
with a ~4100-token prompt (same shape that hit `WrongPhase` pre-fix)
returned HTTP 200 + 8 tokens cleanly. Server log:
`chat/completions done: prompt_tokens=4104, completion_tokens=8`.
Pre-fix this request set errored every time with `Metal complete_prefill
failed for RequestId(N): request is in phase Prefilling, expected
Decoding`.

**Sweep**: ran guidellm sweep (10 strategies, 60s each) at
`prompt_tokens=1024,output_tokens=128` → Qwen3.5 tokenizer expands to
~5400 actual prompt tokens. All ten strategies completed; no
`WrongPhase` errors, no 500s. Artefacts at
`bench-output/2026-04-20-dflash-fixes-validation/{benchmarks.json,csv}`
(10.6 MB JSON).

| Strategy    | Req lat mdn (s) | TTFT mdn (ms) | TPOT mdn (ms) | ITL mdn (ms) | Thru req/s |
|-------------|-----------------|---------------|---------------|--------------|------------|
| synchronous | 8.4             | 1364          | 65.3          | 54.6         | 0.12       |
| throughput  | 42.6            | 12549         | 333.0         | 236.9        | 27.5       |
| constant ×8 | 8.0–8.8         | 1387–1537     | 62.9–68.7     | 52.1–58.0    | ~0.1       |

The canonical `prompt_tokens=4096` bench was tried but all requests
exceeded the 60s max-seconds window per strategy (synthetic expansion
to ~21k actual tokens, prefill alone ≈ 40 s each), so guidellm aborted
with zero completed requests — an orthogonal guidellm-side tuning
issue, not a regression. The 5400-token sweep above covers the
fix's functional gate: long-prompt DFlash prefill no longer desyncs
the scheduler.

**Intentionally not measured here**: the numeric delta of the paired
`perf(metal): defer DFlash batched terminal eval via async_eval` change
(commit `d8cb2f4`). Agent B's expected effect is 2–5% at c=2 — well
below the matched-A/B threshold in `feedback_matched_ab_for_small_bench_effects.md`.
Same-binary same-session before/after runs are needed to resolve it
above thermal noise; punted to a dedicated bench session.

## Problems

- **Canonical `prompt_tokens=4096` guidellm bench can't complete inside
  the 60s max-seconds window on M4 Max** — synthetic expansion pushes
  prompts to ~21k actual tokens, so one request takes ≈ 40 s of
  prefill + ≈ 5 s of 256-token decode ≈ 45+ s, which guidellm's
  warmup-calibration phase rejects as "no successful requests" when it
  can't finish enough of them inside the window. Not a regression of
  this fix — the canonical params simply outgrew the locked budget.
  Follow-up: either raise max-seconds for long-prompt sweeps, or split
  the canonical bench into a short-prompt throughput sweep + a
  long-prompt correctness smoke.

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
