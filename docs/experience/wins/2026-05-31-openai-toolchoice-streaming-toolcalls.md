# OpenAI-compatible tool_choice enforcement + streaming tool_calls + CLI thinking spinner

## Context

The runtime accepted `tools` and returned non-streaming `tool_calls`, but two
gaps blocked dropping ARLE into the OpenAI agent ecosystem:

- `tool_choice` was a **complete no-op** (`none` / `required` / forced-function
  all ignored — the model just picked).
- **`stream=true` + `tools` was hard-rejected (HTTP 400)** — so no streamed
  `tool_calls` deltas, which most OpenAI tool clients expect.

Plus the CLI agent REPL showed nothing during the prefill wait before the first
token (silent, felt hung).

## What worked

`infer/src/http_server/{openai_v1,handlers}.rs` + `crates/chat/src/{protocol,lib}.rs`:

- `chat::ToolChoiceMode { Auto, None, Required, Function(name) }` +
  `build_tool_block_with_choice` / `messages_to_prompt_with_tool_choice`.
  `None` withholds the tools block (model answers as prose, tool extraction
  suppressed); `Required`/`Function` append a directive to the tool block.
- Validation: `Required`/`Function` with empty `tools` → 400; `Function` naming
  a tool not in `tools` → 400.
- Removed the stream+tools rejection. New `chat::StreamingToolCalls` (sibling to
  `VisibleTextStream`) hides `<think>`/`<tool_call>` from streamed content and
  surfaces completed `ToolCall`s as they close. The chat streaming handler routes
  through it **only when tools are present** — the no-tools path stays byte-for-
  byte identical (no `<think>` hiding on plain chat). Emits OpenAI
  `delta.tool_calls[]` chunks (`index`/`id`/`type`/`function.name`/`arguments`)
  then a `finish_reason:"tool_calls"` chunk.
- CLI: TTY-gated `indicatif` "thinking…" spinner in `run_agent_turn`, cleared the
  instant output streams; no-op in non-TTY / `--json` / piped (`run_one_shot`
  never routes through it).

Tests: 620 `infer` lib (48 `openai_v1` + 106 `http_server`) + 34 `chat` green.

## Verification (live, Qwen3.6-35B-A3B-4bit Metal, M4 Pro, warm c=1)

Smoke against `metal_serve` `/v1/chat/completions`:

| Check | Result |
|---|---|
| `tool_choice:required` + empty tools | **400** `requires a non-empty 'tools' array` ✓ |
| `tool_choice:function` unknown name | **400** `tool 'does_not_exist' is not in 'tools'` ✓ |
| non-streaming tool call | `finish:"tool_calls"`, `tool_calls:[get_weather {"city":"Paris"}]`, `content:null` ✓ |
| **streaming + tools** (was 400) | role → content → `delta.tool_calls[0]={id:call_…,type:function,function:{name:get_weather,arguments:"{\"city\":\"Tokyo\"}"}}` → `finish_reason:tool_calls` ✓ |
| `tool_choice:none` + tools | plain content, no tool_calls ✓ |

Perf (no regression — tool path is control-plane, gated off for no-tools requests):
warm c=1 greedy via `/v1/completions`: **512/128 → TTFT 2434 ms, decode 73.5 tok/s**;
**2048/16 → TTFT 9229 ms, decode 80.3 tok/s**.

> Note flagged for follow-up: prefill ~220 tok/s (2048-prompt TTFT = 9.2 s) is
> the user-reported "prefill 很慢"; quantified vs mlx-lm in the A/B (separate entry).

Build env: required the Metal JIT preamble rebuild first — see
[`2026-05-31-metal-jit-preamble-macos2631-rebuild-fix.md`](2026-05-31-metal-jit-preamble-macos2631-rebuild-fix.md).

## Rule

OpenAI tool-calling parity = (1) `tool_choice` actually shapes the prompt +
validates, (2) streaming emits `delta.tool_calls` deltas, not just non-streaming.
Gate the streaming tool-extraction transform on `!tools.is_empty()` so plain chat
streams keep raw passthrough (no surprise `<think>` stripping).
