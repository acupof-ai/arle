# HTTP API

`ARLE` exposes an OpenAI-compatible serving surface for text generation, model
discovery, health/readiness probes, runtime stats, and session persistence
through the dedicated `infer` binary.

This document is the reference map for the current HTTP boundary. Stability
tiers still live in [docs/support-matrix.md](support-matrix.md) and
[docs/stability-policy.md](stability-policy.md).

## Route Map

This document covers the serving surface. The optional train-control proxy
under `/v1/train/*` is documented separately in the train/runtime docs.

| Category | Route | Notes |
| --- | --- | --- |
| Generation | `POST /v1/completions` | Raw prompt surface. SSE supported. |
| Generation | `POST /v1/chat/completions` | Chat message surface. SSE supported. |
| Generation | `POST /v1/responses` | Newer text/tool-call subset. Non-streaming and SSE supported. |
| Discovery | `GET /v1/models` | Returns the boot-time serving identity snapshot. |
| Probes | `GET /healthz` | Lightweight unauthenticated liveness probe. |
| Probes | `GET /readyz` | Lightweight unauthenticated readiness probe; includes the boot-time identity snapshot. |
| Session persistence | `POST /v1/sessions/{session_id}/save` | Persist a session snapshot. |
| Session persistence | `POST /v1/sessions/{session_id}/load` | Load a previously saved session snapshot. |
| Session persistence | `GET /v1/sessions/{session_id}/manifest` | Read the last saved session manifest. |
| Session persistence | `DELETE /v1/sessions/{session_id}` | Delete persisted session state. |
| Operations | `GET /metrics` | Prometheus metrics surface. |
| Operations | `GET /v1/stats` | Human-readable runtime stats surface. |

## Streaming Behavior

- `POST /v1/completions` supports SSE streaming. `stream_options.include_usage`
  is supported, and `stream_options.continuous_usage_stats` is accepted as a
  compatibility hint when `include_usage=true`.
- `POST /v1/chat/completions` supports SSE streaming for plain assistant text.
  Requests that combine `stream=true` with `tools` are rejected until the
  server can emit structured `delta.tool_calls` chunks.
- `POST /v1/responses` supports both non-streaming and SSE forms for the
  current text/tool-call subset. SSE emits `response.created`,
  `response.output_text.delta`, terminal `response.completed`, then `[DONE]`.
  Requests that combine `stream=true` with `tools` are rejected until the
  server can emit structured function-call deltas.

## HTTP Boundary Guarantees

- JSON routes require `Content-Type: application/json`; malformed JSON, missing
  content type, and oversized bodies return structured JSON errors instead of
  framework default text.
- Unsupported top-level parameters on `/v1/completions`,
  `/v1/chat/completions`, and `/v1/responses` return structured
  `invalid_parameter` errors instead of being silently ignored.
- Structured `invalid_parameter` responses include a machine-readable
  `error.param` field.
- Blank `prompt`, empty `messages`, and blank `input` are validated through the
  same structured `invalid_parameter` path.
- `model` is optional on request bodies, but when present it must match the
  currently served model reported by `GET /v1/models` (case-insensitive; final
  path segment match allowed). Mismatches return `404 model_not_found`.
- Streaming completions accept `stream_options.include_usage`;
  `/v1/completions` also accepts `stream_options.continuous_usage_stats` as a
  compatibility hint, and it requires `stream_options.include_usage=true`.
- Chat and responses validation is explicit: supported roles are `system`,
  `user`, `assistant`, and `tool`; `content` part arrays must be text-only.
- Tool definitions must use `type=function`; malformed assistant `tool_calls`
  and tool messages without `tool_call_id` are rejected with structured
  `invalid_parameter` errors.
- `/v1/chat/completions` and `/v1/responses` reject `stream=true` when `tools`
  are present instead of pretending to support streamed tool-call deltas.
- JSON request bodies are capped at `16 MiB`.
- Optional auth uses `Authorization: Bearer <token>`; `401` responses include
  `WWW-Authenticate`.
- Every HTTP response includes `X-Request-Id`; a client-supplied value is
  preserved when valid, otherwise the server generates one.
- `GET /healthz` and `GET /readyz` stay lightweight and unauthenticated;
  `readyz` reports the boot-time identity snapshot without probing the backend
  again.
- `405 Method Not Allowed` responses keep structured JSON bodies and include
  `Allow` on both top-level and session routes.

## Current Gaps

- `/v1/responses` is still the current text/tool-call subset, not the full
  OpenAI Responses API.
- Structured streamed tool-call deltas are not implemented yet, so
  `stream=true` with `tools` is rejected on both chat completions and
  responses.
- Structured outputs are still pending on the `/v1/responses` surface.
