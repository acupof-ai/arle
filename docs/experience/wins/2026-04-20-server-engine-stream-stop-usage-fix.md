# `BackendInferenceEngine::complete_stream` — share `StopChunkProcessor`, propagate real usage

**Commit (pending):** follow-up to 70e2776 — addresses the two codex review
findings on the Ctrl-C / Metal streaming rewrite.

## Context

`70e2776` rewrote `BackendInferenceEngine::complete_stream` (Metal + CPU
path) to route streaming through `generate_stream(…, on_chunk)` so
`tx.send` failure propagates back to the backend and Ctrl-C actually
cancels mid-generation. That landed Ctrl-C correctness, but the new
stop-handling was not equivalent to the serial-runtime path's
`StopChunkProcessor`, and the stop branch hard-coded usage to zero.

`codex review --commit 70e2776` flagged (manually transcribed here
because `~/.codex/sessions` was permission-blocked in the review
sandbox):

> **High** — streamed stop handling is no longer equivalent to
> `complete()`. The new path only recognizes stop strings when the
> *entire buffered output ends with one* (`server_engine.rs:111, 999`)
> and otherwise eagerly forwards every unsent byte
> (`server_engine.rs:1022`). Breaks: (a) stop appearing mid-chunk — the
> default `StreamingInferenceBackend::generate_stream` impl emits the
> whole completion as one chunk (`backend.rs:62`); (b) stop spanning
> chunk boundaries, where the prefix is already sent before the final
> bytes complete the marker. The serial runtime already has the correct
> pattern in `StopChunkProcessor::push_chunk` (`backend/runtime.rs:346`,
> `backend/metal/runtime.rs:1808`): scan for earliest stop anywhere,
> withhold `max_stop_len - 1` bytes.

> **Medium** — the stop path hard-codes terminal usage to zero
> (`server_engine.rs:1059`). The REPL always sets `stop = "<|im_end|>"`
> (`crates/cli/src/repl.rs:719`) and trusts the final delta's usage for
> token counts and session stats (`repl.rs:787, 855`). Every normal
> chat turn now records `0` prompt/completion tokens and skews the TPS
> stats.

## What Worked

**One shared helper, two behavioral corrections, two regression tests.**

1. **Promoted `StopChunkProcessor` to `pub(crate)`** in
   `infer/src/backend/runtime.rs` (with `pub(crate)` methods) and imported
   it into `infer/src/server_engine.rs`. No new duplicate — the
   Metal/CPU streaming path now uses *the same* helper the serial
   runtime has used since its introduction. The second
   `StopChunkProcessor` copy in `infer/src/backend/metal/runtime.rs:1788`
   is pre-existing and out of scope for this fix (would be nice to
   dedupe in a follow-up).

2. **Replaced the end-of-buffer `truncate_at_stop` check** inside
   `complete_stream`'s `on_chunk` with `processor.push_chunk(chunk)`.
   The helper:
   - Scans the **unsent suffix** for the earliest stop match (handles
     mid-chunk stops).
   - Withholds `max_stop_len - 1` bytes per chunk (handles stops
     spanning chunk boundaries).
   - After `hit_stop` flips, silently absorbs further chunks — no raw
     marker bytes can leak to the consumer.

3. **Let the backend run to natural completion** even after a stop is
   matched. This is the pattern `backend/runtime.rs:execute_request`
   already uses: the callback returns `Ok(())` post-stop; the backend
   hits its own EOS or `max_tokens`; we get *real*
   `prompt_tokens`/`completion_tokens` in `GenerateResult`. The final
   delta's usage comes from `generated.prompt_tokens` +
   `generated.completion_tokens`, not a hard-coded zero.

   Tradeoff: this means the backend generates a few extra tokens after
   a stop is detected (until EOS or `max_tokens`). That's a
   compute-waste cost that the CUDA continuous-batching path avoids by
   propagating stop into the scheduler; the Metal/CPU serial path
   doesn't have that wiring yet. Accepted for now — the Medium
   complaint was about *correctness* (usage numbers), not waste. The
   waste existed in the pre-70e2776 non-streaming path too.

4. **Kept the `consumer_dropped` path intact.** Ctrl-C still works:
   dropping `rx` makes `tx.send` fail → callback returns `Err` → backend
   exits `generate_stream` early → `consumer_dropped` flag set →
   `complete_stream` returns `Ok(())` without emitting a final delta.

5. **Added two regression tests** in
   `infer/src/server_engine.rs::tests`:
   - `backend_complete_stream_stop_inside_single_chunk` — mock backend
     emits `"hello<|im_end|>trailing"` as a single chunk; asserts text
     is `"hello"`, no `<|im_end|>` leaks, `finish_reason == Stop`, and
     `usage.completion_tokens == 7` (not 0).
   - `backend_complete_stream_stop_spanning_chunks` — mock backend
     emits `"hello<|im_"` then `"end|>trail"` across two chunks;
     asserts text is `"hello"`, no partial marker bytes
     (`<|im_`/`im_end`/`|>`) leak, and `usage.completion_tokens == 5`
     (not 0).

   Together they pin the exact High/Medium failure modes from the
   review. The existing
   `backend_complete_stream_short_circuits_when_rx_dropped` and
   `backend_complete_stream_emits_all_chunks_and_finish_marker` still
   pass — 4/4 green.

## Rule

**A streaming runtime's stop handling must scan the *unsent suffix*
and withhold the chunk tail.** End-of-buffer `.ends_with(stop)` checks
look fine in a per-token loop where each chunk is a short token-text
(the CUDA single-request path) but break as soon as the chunk size can
be larger than the stop marker — which is the default
`StreamingInferenceBackend` impl's one-shot emission pattern. The fix
is already in the repo (`StopChunkProcessor`); any new streaming-stop
call site must reuse it, never re-derive an end-anchored check.

**Do not hard-code terminal usage.** If a streaming path decides to
exit early and fabricate a final delta, the REPL (and any aggregator
that reads the last delta's `usage`) will report zero tokens. Either
let the backend run to completion and surface its real numbers, or
thread real token counts through the early-exit path. "I don't know
how many tokens this was" is a silent data corruption, not a clean
`None`.

## Bench Policy

Bench-exempt? **Not strictly** — this touches `infer/src/` hot path.
But the practical impact is:
- Correctness fix on stop detection (no throughput delta on the happy
  path; backend generates the same number of tokens before reaching
  EOS/stop).
- Usage-plumbing fix (metadata only, no token-generation impact).
- Small CPU overhead of `StopChunkProcessor.push_chunk` buffering — a
  `String` append + a `.find` per chunk. Negligible compared to a
  single decode step.

Committing a `pending-remote` stub under the 70e2776 bench entry's
cross-reference; regression-check `bench_guidellm.sh dx-stop-fix` on
the next Metal run to confirm no TPS drop from the extra
`StopChunkProcessor` state. **Status: pending-remote.**

## Cross-refs

- [`infer/src/backend/runtime.rs:326`](../../../infer/src/backend/runtime.rs) —
  now-shared `StopChunkProcessor` source of truth.
- [`infer/src/server_engine.rs`](../../../infer/src/server_engine.rs) —
  `BackendInferenceEngine::complete_stream` rewrite (this fix).
- `/tmp/codex-review/70e2776.log` lines 5844–5852 — the review text
  this entry closes.
- 97c1a95 codex review (prior Ctrl-C High) closed by 70e2776 itself.
