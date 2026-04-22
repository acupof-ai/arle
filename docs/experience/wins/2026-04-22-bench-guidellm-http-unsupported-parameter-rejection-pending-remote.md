# HTTP unsupported parameter rejection pending remote verification

## Goal

Tighten the OpenAI-compatible request boundary so `/v1/completions`,
`/v1/chat/completions`, and `/v1/responses` reject unsupported top-level
parameters instead of silently ignoring them.

## Hypothesis

Rejecting unsupported parameters should improve API correctness and DX without
affecting hot-path throughput, because the change only alters request decoding
and structured error classification before scheduler submission.

## Params

- Label: `http-unsupported-parameter-rejection`
- Planned command: `scripts/bench_guidellm.sh http-unsupported-parameter-rejection`
- Backend/model: pending remote host selection
- Feature set: pending remote host selection

## Env

- Local code change only on 2026-04-22
- Remote benchmark pending because this workspace is not the canonical bench
  host

## Results

- Status: `pending-remote`
- Local verification covers:
  - `cargo test -p infer --release --lib --no-default-features --features no-cuda http_server:: -- --nocapture`
  - `cargo clippy -p infer --release --lib --no-default-features --features no-cuda -- -D warnings`

## Problems

- No remote `guidellm` run has been executed yet for this boundary-only runtime
  change.

## Learnings

- OpenAI-compatible surfaces should either support a parameter or reject it
  explicitly; silent drops create the worst debugging path for clients.
