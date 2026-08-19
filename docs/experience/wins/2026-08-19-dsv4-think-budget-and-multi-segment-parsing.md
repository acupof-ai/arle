# DSv4 multi-segment think parsing + forced think-end budget — CUDA, 2026-08-19

> Status: verified on pod (H20 ×4, TP=4, 2026-08-19)

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

## Verified on pod (H20 ×4, TP=4, 2026-08-19)

Build: `think-budget-v6` (head `ad8189dc2`), serve `--max-thinking-tokens 128`.

### Forced think-end fires at exact budget

Prompt: "Explain in detail how transformer attention works…" (streaming).
Result: **128 reasoning deltas exactly**, then engine forced `</think>`,
model continued with 10 311 chars of coherent content (2 608 deltas).
The last reasoning delta ended mid-word — the cut is unambiguous.

### Natural think-end still works

Prompt: "What is 17×23?" — model emitted `</think>` naturally at ~85
reasoning tokens, well under the 128 budget. Budget is a ceiling, not a
target.

### DSv4 eos_token_id ≠ 128822

The model continued generating content after the forced `</think>`,
confirming `eos_token_id` is not the think-end token. If it were, the
forced token would terminate the request instead of transitioning to
content.

### reasoning_tokens usage reporting (fixed in `b3a1f675a`)

`usage.completion_tokens_details.reasoning_tokens` was hardcoded to 0
in three places (`from_parts`, streaming chat, n>1 path). The n>1 path
also used content char length instead of token count for
`completion_tokens`. Fixed by counting reasoning tokens at the token
level in the coordinator, using the same think-state logic as the
engine's `update_think_state`.

## Rule

A reasoning model's think-end token is a lifecycle boundary, not just a
string to search for. Track it at the token level with a budget, and
parse the output as a state machine — not a first-match split.
