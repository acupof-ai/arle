# 2026-04-15 · Metal Qwen3 Live Prefix Reuse

## Context

`metal_serve` already had a live Metal scheduler runtime, but repeated-prefix
requests still re-prefilled from scratch. `MetalPrefixCache` and `MetalKVPool`
existed, yet they were not wired into the live serving path, so
`infer_prefix_hit_rate` stayed at `0`.

This tranche narrowed the problem to `Qwen3` only:

- keep `Qwen3.5` out of scope for now
- add a runtime-owned shared prefix KV pool
- import matched prefixes into fresh request state before scheduler admission
- publish aligned prompt prefixes back into the shared cache when terminal
  prefill completes

## What Worked

- `MetalKVPool` gained detached-slot APIs so the live runtime can own cached
  prefix slots outside any request-local sequence.
- `MetalRequestState` gained `Qwen3`-only prefix import/export hooks, letting
  the runtime move cached prefix rows between the shared pool and a request's
  local caches without changing decode semantics.
- `metal_serve` now performs:
  - prefix lookup before scheduler submit
  - request-state import for the matched prefix
  - suffix-only scheduler admission
  - prompt-prefix publish back into the live cache after terminal prefill

Validation run on 2026-04-15 (`M4 Pro`, `mlx-community/Qwen3-0.6B-4bit`):

- `cargo check -p infer --release --no-default-features --features metal,no-cuda --bin metal_serve`
- `cargo test -p infer --release --no-default-features --features metal,no-cuda kv_pool -- --nocapture`
- `cargo test -p infer --release --no-default-features --features metal,no-cuda request_state -- --nocapture`
- `cargo test --workspace --release --no-default-features --features metal,no-cuda`
- `git diff --check`

Focused live smoke:

- server:
  `./target/release/metal_serve --model-path mlx-community/Qwen3-0.6B-4bit --port 8013 --kv-pool`
- runtime log:
  `Metal live prefix cache enabled for Qwen3: block_size=16, max_cached_tokens=8192`
- identical sequential `max_tokens=1` HTTP requests:
  - run 1: `186.7 ms`
  - run 2: `65.1 ms`
- `/metrics` after warmup + the two requests:
  - `infer_prefix_lookups_total = 3`
  - `infer_prefix_hits_total = 1`
  - `infer_prefix_hit_rate = 0.3333`

The key signal is not the exact milliseconds; it is that the live runtime now
shows both a measurable reuse effect and a non-zero prefix-hit metric on real
HTTP traffic.

## Rule

On Metal, prefix reuse only counts as serving progress when the live runtime
does all three: lookup before scheduler admission, suffix-only prefill, and
publish back into a shared cache. Internal cache counters alone are not enough.
