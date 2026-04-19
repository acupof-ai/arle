# `infer::http_server` — Agent Guide

OpenAI-compatible HTTP API built on `axum`. Load before touching any
HTTP-facing code — the wire format is a product contract, not an
implementation detail.

## Endpoints (what wire-format change cost looks like)

| Route | Handler | Notes |
|-------|---------|-------|
| `POST /v1/completions` | `http_server.rs` via `openai_v1::CompletionRequest` | Raw prompt. Streaming via SSE, `stream_options.include_usage` adds a final usage chunk. |
| `POST /v1/chat/completions` | via `openai_v1::ChatCompletionRequest` | Uses `infer_chat::openai_messages_to_prompt` to render ChatML. |
| `POST /v1/responses` | via `openai_v1::ResponsesRequest` | Newer API surface; uses `max_output_tokens`, not `max_tokens`. |
| `GET /v1/models` | `ModelsListResponse::single(model_id, ...)` | Always returns the one configured model. `owned_by = "agent-infer"`. |
| `GET /v1/stats` | scheduler metrics readout | Defined on the request handle, not here directly. |
| Auth | optional `HttpServerConfig.api_key` | Bearer check in `http_server.rs`. |

## Invariants

1. **`RESPONSE_TIMEOUT = 300s` caps non-streaming requests only.** Streaming
   SSE has natural per-chunk flow control. Do not add a blanket timeout to
   the SSE path — long multi-turn agent runs rely on that.
2. **`session_id` is the agent-routing knob.** Accepted on every request
   type via `session_id` (primary) or `user` (alias, matches OpenAI). Empty
   string and whitespace normalize to `None` (see
   `openai_v1::normalize_session_id`). When present, the scheduler uses it
   for sticky slot routing (`docs/projects/agent-first-architecture.md::A2`).
   Never strip it silently.
3. **All three request types converge on `RequestExecutionOptions`.** Add
   new sampling / stop / session fields there once, then plumb through
   `from_completion` / `from_chat` / `from_responses`. Don't re-parse at
   the handler level.
4. **`IncomingRequest` is the scheduler's input contract** — it's built via
   `RequestExecutionOptions::into_incoming_request(prompt, delta_tx)`. The
   `delta_tx` is the backchannel the scheduler writes `CompletionStreamDelta`
   into.
5. **`CompletionStreamDelta` accumulation** — `BufferedResponse::apply_delta`
   is the single place that collects streaming chunks into a non-streaming
   response. The order matters: text_delta first, then finish_reason, then
   usage, then logprob-per-token. If you reorder, the non-streaming path
   drops data.
6. **The handle is `dyn RequestHandle`, not a concrete type.** The HTTP
   layer must never know whether it's talking to the CUDA scheduler
   (`SchedulerHandle`) or `BackendRuntimeHandle` (Metal/CPU). Adding a
   backend-specific path here re-creates the cfg-leak problem.
7. **`stop`, `stop_token_ids`, `ignore_eos`, `seed`** are all first-class
   sampling inputs. The match between these and `SamplingParams` is
   one-to-one via `sampling_params_from_request` — don't branch.

## Common pitfalls

- Adding a third place where stream chunks get built. There are two:
  live SSE emission in the handler, buffered accumulation in
  `BufferedResponse`. That's it.
- Using `tokio::time::timeout` around the streaming path. Streaming is
  naturally flow-controlled; a wrapping timeout causes silent cancellation.
- Emitting `logprobs` as a field on every chunk. OpenAI's protocol puts
  them in the final chunk or in non-streaming responses only; matches the
  `CompletionStreamDelta.logprob` Option semantics.

## Pointers

- `infer/src/server_engine.rs` — `InferenceEngine`, `CompletionRequest`,
  `CompletionOutput`, `CompletionStreamDelta`, `TokenUsage`, `FinishReason`.
- `infer/src/request_handle.rs` — `RequestHandle` trait (backend-agnostic).
- `crates/chat/src/lib.rs` — chat → prompt rendering.
- `docs/projects/agent-first-architecture.md` — session routing design.
- `docs/plans/2026-04-15-metal-backend-acceptance-plan.md` — HTTP API
  acceptance gates for the Metal backend.

## Performance verification

External perf measurement of this HTTP surface is done via
[`vllm-project/guidellm`](https://github.com/vllm-project/guidellm), wrapped
by [`scripts/bench_guidellm.sh`](../../../scripts/bench_guidellm.sh). That
wrapper is the **canonical** throughput / TTFT / ITL truth source — do not
hand-roll alternative load generators when changing anything in this module,
run the wrapper and snapshot to `docs/experience/wins/`. Canonical params
and plumbing live in
[`docs/plans/guidellm-integration.md`](../../../docs/plans/guidellm-integration.md).
