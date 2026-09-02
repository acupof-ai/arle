# HTTP API

`arle serve` exposes an Anthropic Messages surface and an OpenAI-compatible
surface for text generation, plus model discovery, a health probe, metrics, and
runtime stats.

This document is the reference map for the current HTTP boundary. Stability
tiers still live in [docs/support-matrix.md](support-matrix.md) and
[docs/stability-policy.md](stability-policy.md).

## Route Map

The router lives in `crates/infer-server/src/coordinator.rs` (`build_router`).

| Category | Route | Notes |
| --- | --- | --- |
| Generation | `POST /v1/chat/completions` | OpenAI chat surface. SSE streaming, tools, `chat_template_kwargs`. |
| Generation | `POST /v1/completions` | Raw prompt surface. SSE streaming. |
| Generation | `POST /v1/messages` | Anthropic Messages surface: streaming events, `tools` / `tool_use` / `tool_result`, `thinking`. Any `model` string is accepted and routed to the served model, so Claude Code works with `ANTHROPIC_BASE_URL` alone. |
| Generation | `POST /v1/messages/count_tokens` | Anthropic token count for a Messages request. |
| Generation | `POST /v1/raw_logits` | Teacher logits for OPD. |
| Discovery | `GET /v1/models` | The served model id. |
| Probes | `GET /health` | Liveness. |
| Operations | `GET /metrics` | Prometheus metrics. |
| Operations | `GET /v1/stats` | Runtime stats: scheduler, prefix cache, KV tiers. |
| Operations | `GET /v1/observe/query` · `GET /dashboard` | Observability query surface and its dashboard page. |
| Not implemented | `POST /v1/embeddings` | Returns a structured not-implemented error. |

## Streaming Behavior

- `POST /v1/chat/completions` and `POST /v1/completions` stream as OpenAI SSE
  chunks and end with `data: [DONE]`. `stream_options.include_usage` puts usage
  on the terminal chunk.
- `POST /v1/messages` streams the Anthropic event sequence (`message_start`,
  `content_block_start`, `content_block_delta`, …, `message_stop`). `thinking`
  content blocks are emitted only when the request enabled extended thinking.

## HTTP Boundary Guarantees

- JSON routes require `Content-Type: application/json`; malformed JSON, missing
  content type, and oversized bodies return structured JSON errors instead of
  framework default text.
- Unsupported top-level parameters on `/v1/completions` and
  `/v1/chat/completions` return structured `invalid_parameter` errors instead
  of being silently ignored.
- Structured `invalid_parameter` responses include a machine-readable
  `error.param` field.
- Blank `prompt`, empty `messages`, and blank `input` are validated through the
  same structured `invalid_parameter` path.
- `model` is optional on request bodies and is treated as a label: every request
  is routed to the single served model reported by `GET /v1/models`.
- Chat validation is explicit: supported roles are `system`,
  `user`, `assistant`, and `tool`; `content` part arrays must be text-only.
- Tool definitions must use `type=function`; malformed assistant `tool_calls`
  and tool messages without `tool_call_id` are rejected with structured
  `invalid_parameter` errors.
- JSON request bodies are capped at `16 MiB`.
- Optional auth uses `Authorization: Bearer <token>`; `401` responses include
  `WWW-Authenticate`.
- Every HTTP response includes `X-Request-Id`; a client-supplied value is
  preserved when valid, otherwise the server generates one.
- `GET /health` stays lightweight and unauthenticated.
- `405 Method Not Allowed` responses keep structured JSON bodies and include
  `Allow`.

## Current Gaps

- No OpenAI `/v1/responses` surface.
- Structured outputs need an xgrammar compiler built from the model vocabulary;
  the server logs `structured output unavailable` when it is not.
