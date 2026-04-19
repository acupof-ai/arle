# server_engine: dedup three `generate*` loops into one driver

## Context

`infer/src/server_engine.rs` had three near-identical generation loops under
`#[cfg(feature = "cuda")]`:

- `generate` (lines 290–369 pre-diff): plain token generation, full fastrace
  telemetry (outer "generate" span, ttft_ms/tpot_avg_ms/generated_tokens/
  tok_per_sec span properties, TTFT/TPOT `debug!`).
- `generate_tokens_with_logprobs_inner` (lines 372–417): same as above but
  uses `select_token_with_logprob`, pushes each logprob. No fastrace span,
  no TTFT/TPOT debug logs.
- `generate_streaming_with_callback` (lines 420–517): callback-driven
  streaming, returns `StreamingStats { emitted_tokens, hit_eos,
  consumer_dropped }`; outer "generate_streaming" span (no TTFT/TPOT span
  properties); TTFT/TPOT via `debug!` only; aborts when the callback
  returns false.

All three shared the same prefill → save_prefix_snapshot → first-token
sample → stop-token early-return → decode-loop → push → stop-check → TPOT
report flow, only diverging at three extension points.

The architecture review on 2026-04-19 flagged this as the #1 half-state
in `infer/`.

## What Worked

Introduced a single `generate_inner<M, Observe, Emit>` with two orthogonal
extension points plus a small `TraceMode` enum:

- **`want_logprobs: bool`** switches between `select_token` and
  `select_token_with_logprob`. The trait's default impl of
  `select_token_with_logprob` is `select_token` + `None`, so
  `want_logprobs = false` still dispatches through the non-greedy
  fast path on model overrides.
- **`on_sampled(token, lp)`** — fires IMMEDIATELY after sampling,
  BEFORE the stop-token check. Used by the logprobs path so that a
  stop token's logprob is still recorded (preserves original timing).
- **`on_emit(token) -> SinkControl`** — fires after the token has been
  pushed onto `tokens`. Returns `ConsumerDropped` to abort; used by the
  streaming path when the HTTP client disconnects. Returns `Continue`
  for non-streaming callers.
- **`trace: TraceMode`** — one of `Full` (outer "generate" span +
  TTFT/TPOT span properties + debug logs), `Streaming` (outer
  "generate_streaming" span + debug logs only, no span properties for
  TTFT/TPOT per the original), or `Silent` (no span, no debug).

The unified driver returns `(Vec<u32>, StreamingStats)`. Non-streaming
callers in `ModelInferenceEngine::complete` just discard `_stats`;
`ModelInferenceEngine::complete_stream` uses both.

The three thin wrapper functions (`generate`, `generate_tokens_with_logprobs_inner`,
`generate_streaming_with_callback`) were removed — the two callers
(`InferenceEngine::complete` and `complete_stream`) invoke `generate_inner`
directly, which keeps the change localized to `server_engine.rs` per
the "no half-states" rule.

## What the three loops actually differed in

| Aspect | `generate` | `generate_tokens_with_logprobs_inner` | `generate_streaming_with_callback` |
|---|---|---|---|
| Sampler | `select_token` | `select_token_with_logprob` | `select_token` |
| Per-token side effect | none | push logprob (even for stop token) | call user callback, abort on false |
| Return type | `Vec<u32>` | `(Vec<u32>, Vec<f32>)` | `StreamingStats` |
| Outer fastrace span | "generate" + full props | none | "generate_streaming" (no TTFT/TPOT props) |
| Per-decode-step span | yes | no | yes |
| TTFT/TPOT `debug!` | yes | no | yes |
| Consumer-drop handling | N/A | N/A | return `consumer_dropped = true`, preserve emitted count |

All three shared `anyhow::ensure!(!prompt_tokens.is_empty())`, the
prefill + `save_prefix_snapshot` warn-on-error, the stop-token
early-return, the decode-loop structure, and the TPOT reporting
conditional on `generated_count > 0`.

## LOC

- Before: 1338 lines total. Three generation-loop helper functions
  (lines 278–517, 240 lines) held ~240 lines of duplicated
  prefill+decode logic — three near-identical 80-line bodies.
- After: 1346 lines total (+8). The helper region (lines 278–513,
  236 lines) holds a SINGLE copy of the prefill+decode logic in one
  generic `generate_inner` driver (~170 lines body + 3 tiny closures
  at the two call-sites).

Total file LOC is roughly unchanged (+8) because each of the two
call-sites in `ModelInferenceEngine::{complete, complete_stream}` now
spells out `generate_inner`'s 10 parameters + 2 closures inline. But
the **semantic duplication is gone**: the prefill → sample → stop-check
→ decode-loop → TPOT sequence exists in exactly one place. Future
changes to the loop (telemetry, new stop conditions, kernel
integration) happen once instead of in three near-identical copies —
which is the actual half-state that the architecture review flagged.

## Tests

- `cargo build --release --no-default-features --features metal` — green
- `cargo check -p infer --no-default-features --features cuda,no-cuda` —
  green (CUDA-Rust typecheck on Mac)
- `cargo test --release --no-default-features --features metal -p infer` —
  all 21 binaries green, 329 unit tests + 4 lora loader tests pass, no
  failures.
- `cargo clippy -p infer --no-default-features --features cuda,no-cuda
  -- -D warnings` — my diff added zero new clippy issues in
  `server_engine.rs` (the two remaining issues at lines 723 and 1021
  are pre-existing: `clone_from` suggestion and `large_enum_variant`).
- `scripts/bench_guidellm.sh edge-dedupe-smoke` — NOT RUN locally.
  This is a pure refactor with no numerical or hot-path changes:
  the unified driver is generic + monomorphized identically to the
  prior three functions, the same ops sequence runs per token, no
  new allocations on the hot path (closures are `FnMut` and captured
  by value into stack locals). Regression-check gate for runtime
  changes per `CLAUDE.md::Benchmarks` should still be run on the CUDA
  remote host before a full sign-off — opening this entry with
  status `pending-remote-cuda-bench` for that reason.

**Status:** `pending-remote` — Mac build + test suite green; the CUDA
remote regression bench (guidellm smoke) remains to be run to close
out the Verify gate.

## Rule

When three functions share 80%+ of their body and differ only in
per-step side effects + per-variant telemetry flavor, a single driver
with (a) a `Copy` enum for telemetry mode and (b) two `FnMut` callbacks
— one for observe-before-stop-check, one for emit-after-push — captures
the variance without forcing a trait. Preserve the original
stop-token/logprob timing carefully: a naive single-sink that fires
after the stop-check silently drops the stop-token logprob, changing
test-baseline behavior.
