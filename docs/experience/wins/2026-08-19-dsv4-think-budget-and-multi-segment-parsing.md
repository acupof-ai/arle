# DSv4 multi-segment think parsing + forced think-end budget — CUDA, 2026-08-19

> Status: pending-remote (Mac can't build CUDA; verify on pod)

## Context

DeepSeek-V4-Flash-0731's reasoning sometimes degenerated into repetition
loops without generating `</think>`, so the server returned empty content.
The parser also only found the FIRST `</think>`, breaking multi-segment
thinking (`r1</think>c1<think>r2</think>c2` — the second block leaked into
content).

## What changed

Three fixes across `infer-plan`, `infer-core`, `infer-server`:

### Fix C — State machine parsing (schema.rs, sse_util.rs)

`split_reasoning` and `StreamingReasoningSplitter` rewritten from
first-`</think>`-split to a state machine over `<think>` / `</think>`
markers. Handles multi-segment thinking: reasoning segments concatenate,
content segments concatenate.

### Fix A — Fallback when think block never closes

When the model hits max_tokens or a repetition loop without generating
`</think>`, the entire output is returned as `content` (was: empty content
with all output trapped in `reasoning_content`).

### Fix B — Forced think-end budget (infer-plan, infer-core, coordinator)

- `SamplingParams.force_next_token: Option<u32>` — when set, the sampler
  returns this token directly, bypassing all sampling.
- `RequestState` tracks `in_thinking` + `reasoning_token_count`. After
  `max_thinking_tokens` reasoning tokens, sets `force_next_token =
  Some(think_end_token_id)`. The next decode step forces `</think>`, then
  the model continues with content.
- Think token IDs (128821/128822) cached in `CoordinatorHandle` at load.
- `max_thinking_tokens` semantics changed: was a TOTAL generation cap,
  now a REASONING-ONLY budget. Total generation still capped by
  `max_tokens`.

### Default thinking budget

- DSv4 default: 32768 (high tier) when `--max-thinking-tokens 0`
- `reasoning_effort` parameter maps: low=2048, medium=8192, high=32768
- Default `max_tokens` for thinking: budget + 4096 (content headroom)

Commit: `c8c344051`

## Expected result on pod

- Needle retrieval in thinking mode: model finds needle, generates
  `</think>` (naturally or forced), returns answer as content.
- Repetition loops: forced `</think>` after 32768 reasoning tokens,
  model transitions to content instead of looping forever.
- Multi-segment thinking: second `<think>...</think>` block correctly
  parsed as reasoning, not leaked into content.

## Verify on pod

1. Build: `pod.sh build think-budget`
2. Serve TP=2 with `--max-thinking-tokens 4096`
3. Needle gate thinking mode: model should return answer (not empty)
4. Check logs for forced think-end: `force_next_token` should trigger
   after 4096 reasoning tokens if the model loops
5. Multi-segment: prompt that triggers re-thinking should have both
   segments in `reasoning_content`

## Rule

A reasoning model's think-end token is a lifecycle boundary, not just a
string to search for. Track it at the token level with a budget, and
parse the output as a state machine — not a first-match split.
