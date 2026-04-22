# HTTP model-id validation pending remote verification

## Goal

Tighten the OpenAI-compatible request boundary so `/v1/completions`,
`/v1/chat/completions`, and `/v1/responses` reject requests that target a
different model than the one currently served, instead of silently running on
the loaded model.

## Hypothesis

Rejecting unavailable request-side `model` values should improve API
correctness and client debuggability without affecting throughput, because the
change only runs in request validation before prompt submission.

## Params

- Label: `http-model-id-validation`
- Planned command: `scripts/bench_guidellm.sh http-model-id-validation`
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

- Single-model serving should not pretend request-side `model` selection works;
  if the server exposes one loaded model, mismatched `model` ids must fail
  explicitly.
