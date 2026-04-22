# HTTP chat streaming tools boundary pending remote verification

## Goal

Stop advertising a false OpenAI-compatible path by rejecting
`POST /v1/chat/completions` requests that combine `stream=true` with tool
definitions until the server can emit structured `delta.tool_calls` chunks.

## Hypothesis

Failing this unsupported request shape up front should improve API
correctness and client debuggability without affecting throughput, because
the change only touches request validation before prompt submission.

## Params

- Label: `http-chat-stream-tools-boundary`
- Planned command: `scripts/bench_guidellm.sh http-chat-stream-tools-boundary`
- Backend/model: pending remote host selection
- Feature set: pending remote host selection

## Env

- Local code change only on 2026-04-23
- Remote benchmark pending because this workspace is not the canonical bench
  host

## Results

- Status: `pending-remote`
- Local verification covers:
  - `cargo test -p infer --release --lib --no-default-features --features no-cuda http_server::openai_v1::tests -- --nocapture`
  - `cargo test -p infer --release --lib --no-default-features --features no-cuda http_server:: -- --nocapture`
  - `cargo clippy -p infer --release --lib --no-default-features --features no-cuda -- -D warnings`

## Problems

- No remote `guidellm` run has been executed yet for this request-boundary
  runtime change.

## Learnings

- A chat streaming endpoint should not accept tool-calling requests unless it
  can emit structured `delta.tool_calls` events instead of plain text deltas.
