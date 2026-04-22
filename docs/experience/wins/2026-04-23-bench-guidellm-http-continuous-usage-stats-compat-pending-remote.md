# HTTP continuous-usage-stats compatibility pending remote verification

## Goal

Unblock the canonical GuideLLM streaming probe on `/v1/completions` by
accepting `stream_options.continuous_usage_stats` instead of rejecting the
request body as an unsupported nested field.

## Hypothesis

Treating `continuous_usage_stats` as an explicit streaming usage option should
restore benchmark compatibility without affecting throughput, because the
change only touches request validation and SSE usage-chunk emission.

## Params

- Label: `http-continuous-usage-stats-compat`
- Planned command: `scripts/bench_guidellm.sh http-continuous-usage-stats-compat`
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

- The repo's own benchmark wrapper is part of the effective HTTP contract:
  nested stream options that it emits must be explicitly accepted or the
  canonical perf probe becomes unusable.
