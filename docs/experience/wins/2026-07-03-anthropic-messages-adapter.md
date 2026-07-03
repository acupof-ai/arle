# Anthropic Messages API adapter — Claude Code drives the local serve

## Context

Claude Code speaks the Anthropic Messages API (`ANTHROPIC_BASE_URL`); ARLE
served only OpenAI v1. Goal: a pure translation layer in `infer-server` — no
new inference plumbing — so `POST /v1/messages` (+ `count_tokens`) rides the
existing chat machinery (`render_chat_full`, thinking gate, tools gating,
`streaming_submit` deltas).

## What Worked

- **One router covers all backends.** Both TP=1 (`coordinator_local_router`)
  and multiproc serve share `coordinator_router`, so the two new routes land
  once in `coordinator.rs`.
- **Translate at the seams, reuse the middle.** Requests map into
  `ChatCompletionRequest` (system concat; tool_result fans out into
  `tool`-role messages preserving block order; `tool_use` → `tool_calls`;
  tool_choice auto/any/none/tool matrix). Non-streaming reuses
  `ChatCompletionResponse::from_parts` (reasoning split included) then maps
  the envelope. Streaming taps the internal `RelayCompletionDelta` receiver
  (pre-OpenAI-encoding) and re-encodes as Anthropic events with a 5s
  keep-alive ping; each tool call streams as one `tool_use` block
  (whole-arguments `input_json_delta` — the tool stream only surfaces closed
  `<tool_call>` blocks).
- **Live smoke on Metal (Qwen3.5-0.8B-MLX-4bit, port 18123)**: text
  message, exact SSE sequence (`message_start` … `message_stop`), forced tool
  call → `stop_reason:"tool_use"` + parsed `input`, tool_result follow-up →
  correct text answer, `count_tokens` → 26.
- **The smoke exposed two pre-existing OpenAI-path render bugs** (A/B
  confirmed identical on `/v1/chat/completions`, same serve): ① minijinja
  lacked the `json` feature, so Qwen templates' `{{ tool | tojson }}` 400'd
  every tools request; ② templates iterate history arguments
  (`arguments|items`) as an HF-convention mapping, but the OpenAI wire is a
  JSON string — now parsed for the render context (vLLM/HF behavior).

## Known gaps (wire details not supported)

- `stop_sequences` accepted but unenforced (engine has no stop strings; the
  OpenAI `stop` field has the same gap) — `stop_reason:"stop_sequence"` never
  occurs.
- `tool_choice` `any`/`tool` degrade to advertise-only (the render layer has
  no forced-function decoding); `none` correctly disables tools.
- `thinking`/`metadata` request fields accepted and ignored; reasoning
  (`<think>`) output is dropped, never surfaced as Anthropic thinking blocks.
- Images degrade to a `[image omitted]` text block (no Anthropic-side VLM
  routing); malformed (non-JSON) bodies get axum's plain 400, not the
  Anthropic error envelope.

## Rule

Smoke a new API facade against the *production path on the same config*
before blaming the new code — both render bugs looked like adapter bugs and
were OpenAI-path bugs; the adapter surfaced them because Claude-Code-shaped
requests exercise tools + multi-turn tool history that nothing else had.

Bench: exempt (additive endpoint + COLD template-render repair; inference hot
path untouched — OpenAI tool-less renders byte-identical by construction).

## Addendum — live Claude Code end-to-end (same day)

Real `claude` CLI pointed at the local Metal serve
(`ANTHROPIC_BASE_URL=http://127.0.0.1:18123`, Qwen3.5-0.8B): **wire-level
PASS** — `is_error=false`, `terminal_reason=completed`, 2 turns, 165,121
input / 31,604 output tokens through `/v1/messages` with zero API errors.
Two integration findings, both fixed en route:

1. **Claude Code sends mid-conversation `role:"system"` messages** (Opus 4.8
   feature); the Qwen template raises on non-first system → degraded to
   `<system-reminder>` user turns (`29df035c`).
2. **Claude Code fires concurrent requests** (main + side calls) — a 1-slot
   serve 500s (`server is busy: backend allows at most 1 live request`).
   CC-as-harness serves need `--max-running-requests ≥ 2` (used 4).

The empty `result` text is 0.8B model quality (31K tokens of rambling CC
couldn't reduce to a final answer), not a protocol gap — the harness target
is the 27B on CUDA.
