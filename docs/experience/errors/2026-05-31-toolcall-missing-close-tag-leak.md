# Agent leaks raw tool-call JSON + gives up when the model drops `</tool_call>`

## Context

`arle` agent REPL on Qwen3.6: during multi-step exploration ("看看本项目") the
agent ran several shell tools fine, then leaked a raw fragment like
`{"arguments":{"command":"ls crates/cli/src/"}` into the visible reply and ended
the turn — looking like it "直接退出了 / 过程有点奇怪". No crash (verified: REPL
stayed alive).

## Root Cause (traced — captured raw `completion_text` via `--trace`)

Qwen3.6's **native** tool-call format (its `chat_template.jinja`) is XML —
`<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`
— but ARLE's agent overrides the model's template and instructs the **JSON**
form `<tool_call>{"name":…,"arguments":…}</tool_call>`. Caught between the two,
the model mostly emits valid JSON but intermittently **drops the `</tool_call>`
close tag** (or gets truncated mid-JSON). `chat::parse_tool_calls` used
`TaggedBlock::strip_and_collect`, whose no-close branch **pushes the raw
remainder into the visible content** and stops — so the un-parsed `<tool_call>`
JSON became the turn's "final answer" (leak) and no tool ran.

## Fix

Replaced the strict pass with `extract_tool_calls` — a **brace-matching**
extractor (`json_object_len` counts `{}` depth, respecting strings/escapes) that
parses `<tool_call>{json}` with **or without** the close tag, and **drops** a
truncated/unbalanced fragment to end-of-input instead of leaking it. Shared by
the agent and the HTTP `/v1/chat/completions` parser. +3 chat unit tests
(missing-close, truncated-dropped, text-after-close-preserved). Verified e2e:
re-running the exploration, **0 raw tool-call JSON/tags leak**, 11 tools execute,
no crash.

## Rule

A tool-call parser must not require the `</tool_call>` close tag — locate the
JSON by brace-matching the `{}` after `<tool_call>`, and **hide** (never leak)
fragments you can't parse. When a model misbehaves on tool formatting, check its
**native chat-template format first** (`chat_template.jinja`) — imposing a
different tool format than the one the model was post-trained on is the deeper
cause of malformed/looping tool calls (see follow-up: align ARLE to Qwen3.6's
native XML tool format).
