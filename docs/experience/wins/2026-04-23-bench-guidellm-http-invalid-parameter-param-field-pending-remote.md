# HTTP invalid-parameter param field pending remote verification

## Goal

Align structured request-boundary errors more closely with OpenAI-style client
expectations by adding a machine-readable `error.param` field to
`invalid_parameter` responses.

## Hypothesis

Adding `error.param` should improve SDK and CI debuggability without affecting
throughput, because the change only touches structured error serialization on
failed requests before prompt submission.

## Params

- Label: `http-invalid-parameter-param-field`
- Planned command: `scripts/bench_guidellm.sh http-invalid-parameter-param-field`
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

- String-only `invalid_parameter` errors are weaker than they look: once the
  field name also ships as `error.param`, proxies and SDKs can make stable
  assertions without parsing human text.
