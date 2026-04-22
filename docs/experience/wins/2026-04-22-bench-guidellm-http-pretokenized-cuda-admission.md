# HTTP pretokenized CUDA admission

## Context

The CUDA scheduler overlap work had already removed decode readback polling and
cut per-tick waiting-queue sorting, but normal HTTP requests still entered the
scheduler with `prompt_tokens: None`. That left `assign_slots()` responsible
for tokenizer work on the hot path even though the HTTP layer already owns the
raw prompt string before submission.

## What Worked

- `SchedulerHandle` now carries an optional cloned tokenizer snapshot for the
  CUDA runtime.
- `RequestHandle` exposes a backend-agnostic `tokenizer_clone()` hook with a
  default `None`, so HTTP stays backend-neutral and only pretokenizes when the
  active handle actually has a tokenizer.
- `http_server.rs` now snapshots that optional tokenizer into `AppState` at
  router construction time and uses it to fill `IncomingRequest.prompt_tokens`
  before submission.
- The scheduler still keeps `prompt_tokens: None` as a compatibility fallback,
  but the normal CUDA HTTP path no longer needs to tokenize inside
  `assign_slots()`.
- Local validation passed:
  - `cargo check -p infer --release --no-default-features --features no-cuda`
  - `cargo test -p infer --release --no-default-features --features no-cuda http_server -- --nocapture`
  - `cargo test -p infer --release --no-default-features --features no-cuda scheduler -- --nocapture`

## Rule

Status: `pending-remote`

- HTTP-originated CUDA requests should arrive at the scheduler already
  tokenized whenever the active request handle can provide a tokenizer.
- Backend neutrality belongs in the `RequestHandle` contract; the HTTP layer
  may snapshot optional capabilities, but it must not branch on concrete
  backend types.
- CUDA before/after GuideLLM data is still required before claiming TTFT or
  throughput impact from this overlap cleanup.
