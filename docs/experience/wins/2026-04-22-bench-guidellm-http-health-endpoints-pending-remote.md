# HTTP health endpoints pending remote verification

## Goal

Add explicit `GET /healthz` and `GET /readyz` probes so orchestrators and
proxies do not need to scrape `/metrics` or hit authenticated OpenAI routes to
determine whether the process is live and ready.

## Hypothesis

Process-level health endpoints should improve operational DX without affecting
serving throughput or latency on the hot path, because they only add two
lightweight GET handlers at the HTTP boundary.

## Params

- Label: `http-health-endpoints`
- Planned command: `scripts/bench_guidellm.sh http-health-endpoints`
- Backend/model: pending remote host selection
- Feature set: pending remote host selection

## Env

- Local code change only on 2026-04-22
- Remote benchmark pending because this workspace is not the target benchmark
  machine

## Results

- Status: `pending-remote`
- Local verification covers:
  - `cargo test -p infer --release --lib --no-default-features --features no-cuda http_server:: -- --nocapture`
  - `cargo clippy -p infer --release --lib --no-default-features --features no-cuda -- -D warnings`

## Problems

- No remote `guidellm` run has been executed yet for this HTTP-only runtime
  change.

## Learnings

- Health and readiness probes belong on the documented HTTP surface rather than
  being inferred indirectly from metrics or model routes.
