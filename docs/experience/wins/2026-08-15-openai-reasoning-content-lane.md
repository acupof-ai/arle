# Reasoning always reaches the wire (OpenAI `reasoning_content`, tools included)

## Context

A SwiftUI client driving `arle serve --backend metal` (Qwen3.5-9B-MLX-4bit,
port 18080) showed a long dead window: the thinking indicator cleared and
nothing arrived for seconds. A raw SSE trace found the cause on the server, not
the client — the model generated to `finish_reason: length` (600 tokens) while
the stream carried **zero** deltas.

Three places dropped reasoning that the model had actually produced:

| Site | Behaviour before |
|------|------------------|
| `sse_util.rs` `StreamPipeline::route` | tools active → `Reasoning` deltas discarded ("no wire lane in OpenAI-compat") |
| `sse_util.rs` `emit_reasoning` | request did not ask for thinking → auto-detected `<think>` block stripped, text discarded |
| `schema.rs` `split_reasoning` | non-streaming: model-emitted `<think>` block returned `None` unless `enable_thinking` |

The second case is the common one for checkpoints that ship no chat template
(the 9B MLX build): serve falls back to generic ChatML, `enable_thinking` is
false, and a reasoning-trained model thinks anyway.

## What Worked

`reasoning_content` is the de-facto field across DeepSeek / vLLM / SGLang, and
clients ignore unknown fields, so there was no reason to gate it. Reasoning the
model produced now always reaches the client: `reasoning_content` deltas on
OpenAI, `thinking` blocks on Anthropic, tools mode included. Both gate flags
(`emit_reasoning`, `emit_reasoning_in_tools`) and the second constructor
(`new_anthropic`) were deleted rather than rewired — the two lanes now share one
path.

Measured, same prompt (`用一句话说明 Rust 的所有权`), `max_tokens` 400, one
model load, matched A/B:

| Arm | reasoning deltas | first event | content deltas |
|-----|------------------|-------------|----------------|
| before | 0 | none (silent to `length`) | 0 |
| after | 398 | **0.23 s** | 0 |

Tools-active arm (`列出当前目录文件`) returns a tool call with no thinking:
0 reasoning deltas, first content at 1.36 s — unchanged, so the tools path did
not regress.

Content bytes are untouched in both arms; the change only adds a field that was
previously discarded.

## Rule

A field the model actually produced is not the transport's to drop. When a wire
format has a standard optional field for it (`reasoning_content`), emit it and
let the client choose; gating it on the request flag hides work the user already
paid for and reads to the client as a hang.
