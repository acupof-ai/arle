# 2026-04-15 · Metal Backend Acceptance Plan

This document turns the Metal execution checklist into strict release gates.

References:

- [2026-04-15-metal-backend-execution-checklist.md](2026-04-15-metal-backend-execution-checklist.md)
- [../projects/mlx-backend-roadmap.md](../projects/mlx-backend-roadmap.md)
- [../reviews/2026-04-15-metal-ecosystem-route-correction.md](../reviews/2026-04-15-metal-ecosystem-route-correction.md)

Completion rule for every milestone:

1. the live code path is implemented
2. user-facing docs are updated in the same change
3. the listed verification commands pass
4. any remaining non-goals are stated explicitly

Benchmark rule:

- direct `metal_bench` is a required sanity check, not a serving milestone exit
- `M0.2` and later serving milestones require an HTTP sweep against
  `metal_serve`, not just direct benchmark output

## Status Snapshot

| Milestone | Status | Notes |
| --- | --- | --- |
| `M0.1` local-only bind + auth | Shipped | `metal_serve` defaults to `127.0.0.1`; optional Bearer auth protects `/v1/*` |
| `M0.2` live Metal scheduler | Blocked / not shipped | `M0.2a` local request-state layer landed; serving/runtime rewiring is still pending |
| `M0.3` live prefix cache + KV pool | Not shipped | KV pool still only affects the Qwen3 single-request fallback path |
| `M0.4` memory + reuse observability | Not shipped | current stats still stop at queue / KV utilization / TTFT histograms |
| `M1.1` Metal env toggles to CLI flags | Shipped | `--kv-pool` / `--no-kv-pool` added to all user-facing Metal entry points |
| `M1.2` models + responses API | Partial | `/v1/models` shipped; `/v1/responses` non-streaming subset shipped; streaming parity still pending |
| `M1.3` structured outputs | Not shipped | no JSON-schema constrained decoding yet |
| `M1.4` one-command Apple path | Not shipped | still assumes Cargo for first-time local bring-up |

## P0 · Serving Floor

### `M0.1` Secure local-by-default serving

Status: shipped.

Acceptance:

- `metal_serve` binds `127.0.0.1` by default.
- `--api-key` and `AGENT_INFER_API_KEY` protect `/v1/completions`,
  `/v1/chat/completions`, `/v1/responses`, `/v1/models`, and `/v1/stats`.
- `/metrics` stays unauthenticated.

Verification:

```bash
cargo test -p infer --no-default-features --features metal,no-cuda http_server::tests -- --nocapture
cargo check -p infer --no-default-features --features metal,no-cuda --bin metal_serve
./target/debug/metal_serve --model-path <MODEL>
curl http://127.0.0.1:8000/metrics
curl -i http://127.0.0.1:8000/v1/models
curl -i -H 'Authorization: Bearer <TOKEN>' http://127.0.0.1:8000/v1/models
```

Exit signal:

- unauthenticated `/v1/*` returns `401` when auth is enabled
- `/metrics` still returns `200`

### `M0.2` Replace serial runtime with a live `MetalScheduler`

Status: not shipped. This is the current hard blocker.

Why blocked:

- `infer/src/backend/metal/scheduler.rs` is still an accounting scheduler.
- `metal_serve` still depends on `BackendRuntimeHandle`.
- the Metal backend still exposes whole-request generation, not resumable
  request state that a live scheduler can step.

Required sub-gates:

#### `M0.2a` Resumable Metal request state

Status: local structural tranche landed on 2026-04-15; scheduler-backed serving
is still pending.

Acceptance:

- Qwen3 and Qwen3.5 each expose a request state object that supports:
  - prefill in chunks
  - one-step decode
  - deterministic cleanup on completion / cancellation
- request state owns KV / recurrent state instead of rebuilding the full
  request inside `generate_stream`.

Verification:

```bash
cargo test -p infer --no-default-features --features metal,no-cuda request_state -- --nocapture
cargo test -p infer --no-default-features --features metal,no-cuda --lib metal::scheduler -- --nocapture
cargo check -p infer --no-default-features --features metal,no-cuda --bin metal_serve
```

#### `M0.2b` Scheduler-backed serving path

Acceptance:

- live HTTP traffic goes through a Metal scheduler runtime, not
  `BackendRuntimeHandle`
- the scheduler owns admission, prefill/decode selection, and cleanup
- request cancellation or disconnect cannot leak request state
- aggregate throughput under concurrency rises materially beyond the current
  serial-server shape
- TTFT no longer scales roughly linearly with queue depth the way the current
  serial runtime does

Verification:

```bash
rg -n "BackendRuntimeHandle" infer/src/bin/metal_serve.rs infer/src/http_server.rs
cargo test -p infer --no-default-features --features metal,no-cuda metal_scheduler -- --nocapture
python3 scripts/bench_throughput_sweep.py --url http://127.0.0.1:8000 --quick --label metal-m0.2
```

Exit signal:

- on the same machine / model / build, `C>=4` rows no longer look throughput-flat
  relative to `C=1`
- TTFT at `C>=4` is not dominated by request-level FIFO queueing

#### `M0.2c` Runtime retirement proof

Acceptance:

- `metal_serve` no longer imports or constructs the serial runtime
- concurrent requests can make forward progress without request-level FIFO
  serialization
- a detached-worktree or equivalent isolated build can reproduce the serving
  benchmark used for sign-off

Verification:

```bash
cargo check -p infer --no-default-features --features metal,no-cuda --bin metal_serve
```

### `M0.3` Live prefix cache + KV pool

Status: not shipped.

Acceptance:

- shared-prefix requests skip matched prefill in the live Metal serving path
- scheduler-owned request state uses prefix cache lookups before prefill
- KV pool lifecycle is tied to request admission / completion, not the old
  single-request fallback only
- the serving benchmark can demonstrate a measurable reuse effect on repeated
  prefixes, not just internal cache counters

Verification:

```bash
cargo test -p infer --no-default-features --features metal,no-cuda --lib prefix_cache -- --nocapture
cargo test -p infer --no-default-features --features metal,no-cuda --lib backend::metal::kv_pool -- --nocapture
python3 scripts/bench_throughput_sweep.py --url http://127.0.0.1:8000 --quick --label metal-m0.3
```

### `M0.4` Memory and reuse observability

Status: not shipped.

Acceptance:

- `/metrics` and `/v1/stats` expose at least:
  - `prefix_hit_rate`
  - `kv_util`
  - `active_memory`
  - `peak_memory`
  - queue depth
- scheduler updates these from the live Metal serving path
- those numbers are sufficient to explain the result of the `M0.2/M0.3` HTTP
  sweep without having to attach a profiler trace first

Verification:

```bash
curl http://127.0.0.1:8000/metrics | rg 'prefix_hit_rate|active_memory|peak_memory|infer_requests_waiting|infer_kv_gpu_utilization'
curl http://127.0.0.1:8000/v1/stats
```

## P1 · API And DX

### `M1.1` Promote user-facing Metal toggles to CLI flags

Status: shipped.

Acceptance:

- `metal_request`, `metal_bench`, and `metal_serve` each expose
  `--kv-pool` and `--no-kv-pool`
- `AGENT_INFER_METAL_KV_POOL` remains only as a compatibility fallback when
  neither flag is passed

Verification:

```bash
cargo check -p infer --no-default-features --features metal,no-cuda --bin metal_request
cargo check -p infer --no-default-features --features metal,no-cuda --bin metal_bench
cargo check -p infer --no-default-features --features metal,no-cuda --bin metal_serve
./target/debug/metal_request --help | rg "kv-pool"
./target/debug/metal_bench --help | rg "kv-pool"
./target/debug/metal_serve --help | rg "kv-pool"
```

### `M1.2` Add `/v1/models` and `/v1/responses`

Status: partial.

#### `M1.2a` `/v1/models`

Status: shipped.

Acceptance:

- `GET /v1/models` returns the loaded model id in OpenAI list format
- endpoint is covered by the same optional Bearer auth as other `/v1/*` routes

Verification:

```bash
cargo test -p infer --no-default-features --features metal,no-cuda http_server::tests -- --nocapture
curl http://127.0.0.1:8000/v1/models
```

#### `M1.2b` `/v1/responses` non-streaming subset

Status: shipped.

Acceptance:

- `POST /v1/responses` accepts `input` as a string or OpenAI-style messages
- `instructions` is mapped into a system message
- response object includes `object=response`, `output`, `output_text`, and
  `usage`

Verification:

```bash
cargo test -p infer --no-default-features --features metal,no-cuda --lib -- --nocapture
curl -X POST http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"ignored-by-server","input":"hello","max_output_tokens":16}'
```

#### `M1.2c` `/v1/responses` streaming parity

Status: not shipped.

Acceptance:

- `stream=true` returns SSE events in a stable Responses-API-compatible shape
- final stream event carries terminal status / usage
- tool-call outputs stream cleanly or are explicitly deferred with a terminal
  structured item

### `M1.3` Structured outputs

Status: not shipped.

Acceptance:

- `response_format` is accepted on chat and responses requests
- JSON-schema constrained decoding enforces syntax during sampling, not after
  the fact
- tool-call syntax error rate drops materially on the agent bench

### `M1.4` One-command Apple Silicon path

Status: not shipped.

Acceptance:

- a first-time Apple user can install and start a local Metal server without
  needing to reason about Cargo feature flags
- docs point to a single canonical install/run command or script

## P2 · Product Breadth

### `M2.1` DFlash backend-level generalization

Acceptance:

- speculative decode lives behind a Metal backend contract, not a Qwen3-only
  special case

### `M2.2` Metal model coverage

Acceptance:

- GLM4 moves from limited to supported or is explicitly deferred with a clear
  rationale

### `M2.3` Capture ergonomics

Acceptance:

- capture / profiling can be enabled from the CLI and is documented as part of
  regression workflow

## Current Exit For This Wave

The current wave is acceptable to merge when all of the following are true:

- `M0.1` stays green
- `M1.1` is green
- `M1.2a` is green
- `M1.2b` is green
- `M0.2` remains explicitly marked blocked instead of being implied complete
