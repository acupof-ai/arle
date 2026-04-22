# HTTP empty-input validation unification pending remote verification

## Goal

Unify empty-request validation across `/v1/completions`,
`/v1/chat/completions`, and `/v1/responses` so blank `prompt`, empty
`messages`, and blank `input` fail through the same structured
`invalid_parameter` path instead of endpoint-specific error codes.

## Hypothesis

Collapsing empty-input checks into the shared request validators should improve
API consistency and SDK debuggability without affecting serving throughput,
because the change only runs before request submission.

## Params

- Label: `http-empty-input-validation`
- Planned command: `scripts/bench_guidellm.sh http-empty-input-validation`
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

- Empty-input handling is part of the public API contract; letting each route
  invent its own error code is worse DX than one explicit validation path.
