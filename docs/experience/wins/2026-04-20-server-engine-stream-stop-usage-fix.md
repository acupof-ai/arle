# `BackendInferenceEngine::complete_stream` — share `StopChunkProcessor`, short-circuit on text stops, keep real usage

**Commit (pending):** follow-up to 70e2776 — addresses the two codex review
findings on the Ctrl-C / Metal streaming rewrite.

## Context

`70e2776` rewrote `BackendInferenceEngine::complete_stream` (Metal + CPU
path) to route streaming through `generate_stream(…, on_chunk)` so
`tx.send` failure propagates back to the backend and Ctrl-C actually
cancels mid-generation. That landed Ctrl-C correctness, but the new
stop-handling was not equivalent to the serial-runtime path's
`StopChunkProcessor`. The first follow-up fixed truncation but still
left incremental backends decoding past matched text stops; the second
follow-up then fixed prompt termination by fabricating zero usage. This
landing closes the loop: prompt text stops now short-circuit decode
*and* preserve real backend usage.

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

> **P1** — once a streamed text stop is matched, `complete_stream`
> still returns `Ok(())` from the callback and lets incremental
> backends keep sampling until EOS / `max_tokens`. On Metal that keeps
> the REPL / HTTP stream open while burning decode on tokens that are
> immediately discarded.

## What Worked

**One shared helper, one explicit stop sentinel, and backends that treat
callback abort as graceful termination instead of an error.**

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

3. **Introduced `backend::StreamStopMatched` as the shared graceful-stop
   sentinel.** Once `StopChunkProcessor::hit_stop()` flips, both
   `backend/runtime.rs::execute_request` and
   `server_engine.rs::complete_stream` return
   `Err(StreamStopMatched)` from the callback. That means:
   - the consumer stops seeing text immediately after the matched stop;
   - incremental backends can stop sampling promptly instead of burning
     decode to EOS / `max_tokens`;
   - the callback error still has a distinct type from real failures.

4. **Updated backend implementations to treat `StreamStopMatched` as a
   graceful finish and still return `GenerateResult`.**
   - The default `StreamingInferenceBackend::generate_stream` catches a
     callback `StreamStopMatched` and returns the already-computed
     `generate()` result.
   - The Metal Rust loops (`generate.rs`, `dflash.rs`, `qwen35.rs`)
     break immediately on `StreamStopMatched` after recording the token
     that completed the stop and then return their partial
     `MetalGenerateOutput`.
   - The Qwen3/Qwen3.5 compiled C++ callback wrapper records partial
     `out_tokens[..out_count]` and returns those as a normal result when
     the callback asked to stop.

   Result: the final delta's `usage` still comes from the backend's
   real counters; no more fabricated zeroes, and no more wasted decode
   on the Metal incremental path.

5. **Kept the `consumer_dropped` path intact.** Ctrl-C still works:
   dropping `rx` makes `tx.send` fail → callback returns `Err` → backend
   exits `generate_stream` early → `consumer_dropped` flag set →
   `complete_stream` returns `Ok(())` without emitting a final delta.

6. **Added / updated regression tests** in
   `infer/src/server_engine.rs::tests` and
   `infer/src/backend/runtime.rs::tests`:
   - `backend_complete_stream_stop_inside_single_chunk` — mock backend
     emits `"hello<|im_end|>trailing"` as a single chunk; asserts text
     is `"hello"`, no `<|im_end|>` leaks, `finish_reason == Stop`, and
     `usage == { prompt_tokens: 3, completion_tokens: 7, total_tokens: 10 }`.
   - `backend_complete_stream_stop_spanning_chunks` — mock backend
     emits `"hello<|im_"` then `"end|>trail"` across two chunks;
     asserts text is `"hello"`, no partial marker bytes
     (`<|im_`/`im_end`/`|>`) leak, and
     `usage == { prompt_tokens: 2, completion_tokens: 5, total_tokens: 7 }`.
   - `backend_runtime_short_circuits_after_text_stop_match` — serial runtime
     mock backend emits a stop-bearing chunk plus one more chunk, but now
     catches `StreamStopMatched` and returns a partial result immediately;
     asserts stop-truncated text, real usage
     `{ prompt_tokens: 11, completion_tokens: 7, total_tokens: 18 }`, and
     that the backend only attempted one chunk.
   - `backend_complete_stream_short_circuits_on_text_stop_match` — same
     contract at `BackendInferenceEngine`: prompt stop terminates after
     the first stop-bearing chunk and still reports
     `{ prompt_tokens: 4, completion_tokens: 5, total_tokens: 9 }`.

   Together they pin the exact High/Medium failure modes from the
   review and the final contract: streamed text stops are truncated for
   consumers, decode short-circuits promptly, and usage is always
   sourced from the backend's final completion result. The existing
   `backend_complete_stream_short_circuits_when_rx_dropped` and
   `backend_complete_stream_emits_all_chunks_and_finish_marker` still
   pass.

## Rule

**A streaming runtime's stop handling must scan the *unsent suffix*
and withhold the chunk tail.** End-of-buffer `.ends_with(stop)` checks
look fine in a per-token loop where each chunk is a short token-text
(the CUDA single-request path) but break as soon as the chunk size can
be larger than the stop marker — which is the default
`StreamingInferenceBackend` impl's one-shot emission pattern. The fix
is already in the repo (`StopChunkProcessor`); any new streaming-stop
call site must reuse it, never re-derive an end-anchored check.

**If a callback needs to stop decode early, encode that as a typed,
graceful control signal and make the backend return a real result.**
Letting the backend ignore the stop wastes compute; aborting without a
result tempts the caller to fabricate `usage=0`. The correct contract is
"typed sentinel in the callback, real counters in the returned result."

## Bench Policy

Bench-exempt? **Not strictly** — this touches `infer/src/` hot path.
But the practical impact is:
- Correctness fix on stop detection and prompt stop termination.
- Small decode win on incremental backends because matched text stops no
  longer burn tokens until EOS / `max_tokens`.
- Usage-plumbing fix: metadata stays real instead of falling back to
  zeroes.
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
