# HTTP message-shape validation pending remote verification

## Goal

Tighten the OpenAI-compatible chat and responses request boundary so unsupported
message roles, malformed assistant tool calls, and non-function tool
definitions are rejected explicitly instead of being flattened into ambiguous
prompts.

## Hypothesis

Rejecting unsupported message and tool shapes should improve API correctness
and client debuggability without affecting serving throughput, because the
change only runs in request validation before the scheduler sees the prompt.

## Params

- Label: `http-message-shape-validation`
- Planned command: `scripts/bench_guidellm.sh http-message-shape-validation`
- Backend/model: pending remote host selection
- Feature set: pending remote host selection

## Env

- Local code change only on 2026-04-22
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

- Text-only OpenAI-compatible serving still needs strict message-shape
  validation; otherwise invalid roles and malformed tool payloads degrade into
  hard-to-debug prompt corruption.
