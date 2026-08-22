# Unsolicited thinking blocks aborted every Claude Code rollout

## Context

`agent-opd` drives rollouts through cc-harness: Claude Code is the agent
scaffold and the model behind `ANTHROPIC_BASE_URL` is ARLE's own in-process
serve. A run over 218 tasks returned `passed=0` on 968 rollouts, every one
`edited=false` with `turns=Some(1..2)` and ~40 s of wall time.

The zero was read as a task-difficulty result at first. It was not: the model
never completed a single turn.

## Root Cause

Both `/v1/messages` paths emitted Anthropic `thinking` content blocks whenever
the model produced reasoning, with no check on whether the client had enabled
extended thinking. `ThinkingCap-Qwen3.6-27B` reasons by default through its
chat template, so `thinking` was true for every request and every client got
thinking blocks it never asked for.

The non-streaming path documented the correct rule and did not implement it:

```rust
// Thinking block first (Anthropic convention), when the model produced
// reasoning content and the request enabled thinking.
if let Some(reasoning) = choice.message.reasoning_content.as_ref()
```

The comment names a gate the condition does not contain. `thinking` in the
handler comes from `thinking || prompt_prefills_think(&prompt)` -- a property
of the model, not of the request.

Claude Code refuses the stream: `terminal_reason=aborted_streaming`,
`is_error=true`, `output_tokens=0`, `duration_api_ms=0`.

## Fix

Thread whether the CLIENT enabled extended thinking
(`ThinkingConfig::client_enabled`) into both paths, and drop reasoning rather
than emit blocks the client did not enable. A model that reasons by default
still reasons — that is a generation property, not a wire one.

## Verified

One Claude Code turn against the same NVFP4 serve, before and after:

| field | before | after |
|---|---|---|
| `is_error` | true | false |
| `terminal_reason` | aborted_streaming | completed |
| `subtype` | error_during_execution | success |
| `output_tokens` | 0 | 216 |
| `stop_reason` | null | end_turn |

## Rule

A rollout harness returning all-zero reward is a harness claim before it is a
model claim. Check that a single turn completes — `turns`, `output_tokens`,
`terminal_reason` — before reading the score as capability.
