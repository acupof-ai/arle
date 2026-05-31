# Agent tool-use convergence: nudge + dual-format (JSON + native XML) parsing

## Context

After fixing the tool-call leak ([errors entry](../errors/2026-05-31-toolcall-missing-close-tag-leak.md)),
the deeper symptom remained: on "看看本项目" the agent ran **11** redundant
`ls`/`cat` calls (context ballooning to 49 K tokens) and ended with
**"(agent finished without a visible reply)"** — it explored but never answered.
Suspected cause: the JSON tool format ARLE imposes conflicts with Qwen3.6's
native XML (`<function=…><parameter=…>`) training, plus no instruction to stop.

## What Worked

Two small, low-risk changes in `crates/chat/src/protocol.rs` (shared by agent +
HTTP) — chose this over a full prompt-format switch, which would have broken the
JSON-only HTTP streaming tool parser (`StreamingToolCalls`):

1. **Convergence nudge** in `build_tool_block`: "Call a tool only when you need
   info you don't have, never repeat a call, and as soon as you have enough info
   STOP calling tools and reply with a direct final answer."
2. **Dual-format parser** (`extract_tool_calls` + `parse_native_function_block`):
   parses Qwen3.6's native XML `<tool_call><function=NAME><parameter=K>V</parameter>
   …</function></tool_call>` **and** the JSON form, so the agent executes
   whichever the model emits (close tag optional; truncated fragments dropped,
   never leaked). +4 chat tests.

## Results (e2e, Qwen3.6 Metal, "看看本项目，简单介绍一下")

| | before | after |
|---|---|---|
| tool calls | 11 (redundant loop) | **3** |
| final answer | "(no visible reply)" | **clean structured summary** |
| context peak | 49 K tokens | 11 K |
| leaked JSON | yes | **none** |

The agent produced an accurate multi-point summary of the project and stopped.

## Rule

Two distinct levers for flaky agent tool-use: (1) a **convergence instruction**
("stop when you have enough, then answer") cures explore-forever loops far more
cheaply than format surgery; (2) make the parser **dual-format** (model's native
XML + JSON) so it executes whatever the model emits — but don't flip the *prompt*
format universally when a shared streaming parser only understands one form.
