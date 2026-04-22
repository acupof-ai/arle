# HTTP text-only content rejection pending remote verification

## Goal

Tighten the OpenAI-compatible chat and responses surfaces so non-text content
parts are rejected explicitly instead of being silently dropped during prompt
flattening.

## Hypothesis

Rejecting unsupported multimodal content parts should improve API correctness
and client debuggability without affecting serving throughput, because the
change only runs in request validation before the scheduler sees the prompt.

## Params

- Label: `http-text-only-content-rejection`
- Planned command: `scripts/bench_guidellm.sh http-text-only-content-rejection`
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

- A text-only server should reject multimodal inputs explicitly; flattening them
  into partial text is a worse compatibility story than a fast structured
  `400`.
