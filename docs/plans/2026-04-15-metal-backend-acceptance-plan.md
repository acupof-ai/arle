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
| `M0.2` live Metal scheduler | Partial / not shipped | `M0.2a` request state, `M0.2b` live scheduler runtime, `M0.2c` Qwen3 same-length decode batching, and `M0.2d` Qwen3.5 same-length decode batching landed locally; throughput exit is still blocked on variable-length decode and per-step batch-state rebuild cost |
| `M0.3` live prefix cache + KV pool | Shipped | Qwen3 uses runtime-owned shared KV pool + prefix cache; Qwen3.5 now ships snapshot-replay prefix reuse on the compiled path |
| `M0.4` memory + reuse observability | Shipped | live runtime metrics and MLX allocator knobs are wired; repeated-prefix Qwen3 smoke now drives non-zero `prefix_hit_rate` |
| `M1.1` Metal env toggles to CLI flags | Shipped | `--kv-pool` / `--no-kv-pool` added to all user-facing Metal entry points |
| `M1.2` models + responses API | Shipped | `/v1/models` and `/v1/responses` now ship in both non-streaming and SSE forms |
| `M1.3` structured outputs | Not shipped | no JSON-schema constrained decoding yet |
| `M1.4` one-command Apple path | Shipped | `scripts/start_metal_serve.sh` is the canonical first-time Apple bring-up path |

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

Status: partial / not shipped. The live scheduler path exists locally, but the
milestone exit is still blocked by throughput.

Current state:

- `M0.2a` request state landed for Qwen3 / Qwen3.5.
- `M0.2b` rewired standard `metal_serve` traffic onto a live Metal scheduler
  runtime with chunked prefill, decode-priority interleave, and cancellation
  cleanup.
- `metal_serve` still falls back to the legacy serial runtime when Metal
  DFlash is enabled.
- aggregate throughput is still below the full milestone exit because the live
  runtime only batches narrow same-length decode cases today, and Qwen3.5 still
  rebuilds batched state on every decode step.

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

Status: local runtime landed on 2026-04-15; `M0.2c` added same-length Qwen3
decode batching and `M0.2d` added same-length Qwen3.5 decode batching. TTFT
exit passed, but the general throughput exit is still pending.

Verification:

```bash
rg -n "BackendRuntimeHandle" infer/src/bin/metal_serve.rs infer/src/http_server.rs
cargo test -p infer --no-default-features --features metal,no-cuda metal::scheduler -- --nocapture
scripts/bench_guidellm.sh metal-m0.2
```

Exit signal:

- `metal_serve` no longer imports `BackendRuntimeHandle` on the standard path
- on the same machine / model / build, TTFT at `C>=4` is no longer dominated by
  request-level FIFO queueing
- aggregate throughput at `C>=4` materially exceeds the old serial-server shape
  instead of merely trading throughput for latency

Current local evidence (2026-04-15, M4 Pro, `Qwen3.5-4B-MLX-4bit`, quick HTTP sweep):

- old serial reference (`M0.2a`): `512/256 C=4` = `65.8 tok/s`, `TTFT p50 7994 ms`
- current live runtime (`M0.2b`): `512/256 C=4` = `58.7 tok/s`, `TTFT p50 1826 ms`

Interpretation:

- latency exit is materially better (`-77%` TTFT p50)
- throughput exit is still blocked (`-11%` aggregate throughput vs the serial reference)
- post-`M0.2c`, Qwen3 same-length decode batches now show a real local win, but
  still not the full exit:
  - focused Qwen3 server check: `C=4` throughput `23.30 -> 25.39 tok/s`
    (`+9.0%`), `TTFT p50 3559 -> 2716 ms` (`-23.7%`)
  - `M0.2d` added a real same-length Qwen3.5 batched compiled-step path:
    direct `128/128` improved from `82.0 -> 84.2 tok/s` generation TPS
    (`+2.7%`)
  - quick Qwen3.5 sweep still stayed effectively flat: `512/256 C=4`
    `66.4 -> 66.2 tok/s`, `TTFT p50 1737 -> 1757 ms`
  - interpretation: the runtime shape improved for Qwen3, and Qwen3.5 now has
    a real batched decode path, but the serving-wide throughput exit remains
    blocked on variable-length decode and the cost of rebuilding batched
    request state each step

#### `M0.2c` Runtime retirement proof

Acceptance:

- `metal_serve` no longer imports or constructs the serial runtime
- concurrent requests can make forward progress without pure request-level FIFO
  serialization on the standard path
- a detached-worktree or equivalent isolated build can reproduce the serving
  benchmark used for sign-off

Verification:

```bash
cargo check -p infer --no-default-features --features metal,no-cuda --bin metal_serve
```

### `M0.3` Live prefix cache + KV pool

Status: shipped. `Qwen3` ships a live runtime-owned prefix cache + shared KV
pool path, and `Qwen3.5` now ships live prefix reuse via replayed compiled-path
snapshots on the scheduler runtime.

Acceptance:

- shared-prefix requests skip matched prefill in the live Metal serving path
- scheduler-owned request state uses prefix cache lookups before prefill
- KV pool lifecycle is tied to request admission / completion, not the old
  single-request fallback only
- the serving benchmark can demonstrate a measurable reuse effect on repeated
  prefixes, not just internal cache counters
- non-goal for this tranche: zero-copy shared recurrent-state ownership for Qwen3.5

Verification:

```bash
cargo test -p infer --no-default-features --features metal,no-cuda --lib prefix_cache -- --nocapture
cargo test -p infer --no-default-features --features metal,no-cuda --lib backend::metal::kv_pool -- --nocapture
cargo test -p infer --no-default-features --features metal,no-cuda request_state -- --nocapture
./target/release/metal_serve --model-path mlx-community/Qwen3-0.6B-4bit --port 8013 --kv-pool
python3 - <<'PY'
import json, time, urllib.request
url='http://127.0.0.1:8013/v1/completions'
prompt=('System: You are a precise benchmark assistant. '
        'Summarize the same deployment checklist in one short sentence. '
        'Checklist: verify auth, verify warmup, verify metrics, verify prefix cache, verify latency.')
body=json.dumps({'model':'mlx-community/Qwen3-0.6B-4bit','prompt':prompt,'max_tokens':1,'temperature':0.0}).encode()
for i in range(2):
    req=urllib.request.Request(url, data=body, headers={'Content-Type':'application/json'}, method='POST')
    t0=time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        resp.read()
    print({'run': i + 1, 'elapsed_ms': round((time.perf_counter() - t0) * 1000, 1)})
PY
curl -s http://127.0.0.1:8013/metrics | rg 'infer_prefix_(lookups|hits|hit_rate)'
```

Current local evidence (2026-04-15, M4 Pro):

- server startup now logs:
  `Metal live prefix cache enabled for Qwen3: block_size=16, max_cached_tokens=8192`
- sequential identical `max_tokens=1` requests over HTTP:
  - run 1: `186.7 ms`
  - run 2: `65.1 ms`
- `/metrics` after warmup + the two requests:
  - `infer_prefix_lookups_total = 3`
  - `infer_prefix_hits_total = 1`
  - `infer_prefix_hit_rate = 0.3333`
- `Qwen3.5-4B-MLX-4bit` now logs:
  `Metal live prefix cache enabled for Qwen3.5 snapshot replay: block_size=16, max_cached_tokens=8192`
- sequential identical `max_tokens=1` requests on `Qwen3.5-4B-MLX-4bit`:
  - run 1: `533.9 ms`
  - run 2: `145.6 ms`
  - run 3: `186.2 ms`
- `/metrics` and `/v1/stats` after the Qwen3.5 smoke:
  - `infer_prefix_lookups_total = 3`
  - `infer_prefix_hits_total = 1`
  - `infer_prefix_hit_rate = 0.3333`
  - `/v1/stats` reports `prefix_hit_rate=33.3%`

### `M0.4` Memory and reuse observability

Status: shipped.

Acceptance:

- `/metrics` and `/v1/stats` expose at least:
  - `prefix_hit_rate`
  - `kv_util`
  - `active_memory`
  - `peak_memory`
  - queue depth
- scheduler updates these from the live Metal serving path
- user-facing Metal entry points expose the same MLX allocator controls:
  - `--memory-limit-bytes`
  - `--cache-limit-bytes`
  - `--wired-limit-bytes`
- those numbers are sufficient to explain the result of the `M0.2/M0.3` HTTP
  sweep without having to attach a profiler trace first

Verification:

```bash
curl http://127.0.0.1:8000/metrics | rg 'prefix_hit_rate|active_memory|peak_memory|infer_requests_waiting|infer_kv_gpu_utilization'
curl http://127.0.0.1:8000/v1/stats
./target/debug/metal_serve --help | rg 'memory-limit-bytes|cache-limit-bytes|wired-limit-bytes'
./target/debug/metal_bench --help | rg 'memory-limit-bytes|cache-limit-bytes|wired-limit-bytes'
./target/debug/metal_request --help | rg 'memory-limit-bytes|cache-limit-bytes|wired-limit-bytes'
```

Current local evidence (2026-04-15, M4 Pro, `Qwen3-0.6B-4bit`):

- idle `/v1/stats` now reports live MLX memory:
  `active_mem=220.5MB peak_mem=315.0MB cache_mem=94.5MB`
- after one completion request:
  `requests=1 active=0 waiting=0 tokens_out=8 ... ttft_p50=0.1ms ... tpot_p50=5.0ms`
- `/metrics` now exports:
  - `infer_prefix_hit_rate`
  - `infer_memory_active_bytes`
  - `infer_memory_peak_bytes`
  - `infer_memory_cache_bytes`
- user-facing Metal binaries now expose allocator controls:
  - `--memory-limit-bytes`
  - `--cache-limit-bytes`
  - `--wired-limit-bytes`

Notes:

- `prefix_hit_rate` is now a real, non-zero live metric on the shipped Qwen3
  repeated-prefix path.
- Qwen3.5 prefix reuse is copy-based snapshot replay today, not zero-copy shared
  recurrent-state ownership.

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

Status: shipped.

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

Status: shipped.

Acceptance:

- `stream=true` returns SSE events in a stable Responses-API-compatible shape
- final stream event carries terminal status / usage
- tool-call outputs stream cleanly or are explicitly deferred with a terminal
  structured item

Verification:

```bash
cargo test -p infer --no-default-features --features metal,no-cuda http_server::tests -- --nocapture
curl -N -X POST http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"ignored-by-server","input":"hello","max_output_tokens":16,"stream":true}'
```

### `M1.3` Structured outputs

Status: not shipped.

Acceptance:

- `response_format` is accepted on chat and responses requests
- JSON-schema constrained decoding enforces syntax during sampling, not after
  the fact
- tool-call syntax error rate drops materially on the agent bench

### `M1.4` One-command Apple Silicon path

Status: shipped.

Acceptance:

- a first-time Apple user can install and start a local Metal server without
  needing to reason about Cargo feature flags
- docs point to `scripts/start_metal_serve.sh` as the canonical install/run
  command
- the wrapper defaults to a small local model and forwards extra `metal_serve`
  flags after `--`

Verification:

```bash
./scripts/start_metal_serve.sh --help
rg -n "start_metal_serve.sh" README.md docs/environment.md \
  docs/plans/2026-04-15-metal-backend-acceptance-plan.md \
  docs/plans/2026-04-15-metal-backend-execution-checklist.md \
  docs/support-matrix.md
```

Exit signal:

- first-time Apple users are directed to the same script in the README,
  environment doc, and acceptance plan
- the wrapper hides feature flags and makes the local Metal server start path
  repeatable without cargo-flag knowledge

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
