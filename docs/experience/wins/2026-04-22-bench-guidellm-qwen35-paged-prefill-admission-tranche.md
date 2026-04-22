# Qwen3.5 Paged Prefill + Admission Tranche

## Context

This tranche lands three runtime-facing changes on the CUDA scheduler/model
path:

- `Qwen3.5` now advertises paged prefill on the shipped path
- decode launch no longer has a separate public mixed entrypoint
- prefill planning now reserves full remaining decode headroom and can advance
  multiple prefills per prefill-only tick
- admission now defers requests that cannot reserve their full ISL envelope

The commissioning benchmark gap is the current `c4/c8/c16` collapse against
SGLang, where TTFT and throughput both regress under concurrency.

## What Worked

Local validation passed:

- `cargo check -p infer --release --no-default-features --features cuda,no-cuda`
- `cargo check -p infer --release --no-default-features --features no-cuda`
- `cargo test -p infer --release --no-default-features --features no-cuda scheduler -- --nocapture`
- `cargo clippy -p infer --release --no-default-features --features no-cuda -- -D warnings`

Structural results:

- `Qwen3.5` no longer keeps paged prefill globally disabled in
  `model/qwen35/forward.rs`
- `step_decode_launch_mixed` is no longer part of the scheduler surface; mixed
  decode routes through the same decode launch entry
- `execution.rs` now reserves page headroom against full remaining decode
  budget instead of a single token
- `assign_slots()` now performs a best-effort full-ISL reservation check and
  defers requests that cannot fit, rather than over-admitting blindly

## Rule

Status: `pending-remote`

This is a runtime change and needs a CUDA guidellm before/after snapshot on the
real benchmark host. Local compile/test closure is necessary but not sufficient
for shipping any throughput claim.
