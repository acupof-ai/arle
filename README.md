<p align="center">
  <strong>agent-infer</strong><br>
  <em>KV-cache-first inference engine for LLM agents. Pure Rust, with CUDA as the primary serving path.</em>
</p>

<p align="center">
  <a href="https://cklxx.github.io/agent-infer/"><img src="https://img.shields.io/badge/website-cklxx.github.io%2Fagent--infer-D97757?style=flat-square" alt="Website"></a>
  <a href="https://github.com/cklxx/agent-infer/actions"><img src="https://github.com/cklxx/agent-infer/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://github.com/cklxx/agent-infer/releases"><img src="https://img.shields.io/github/v/release/cklxx/agent-infer?include_prereleases" alt="Release"></a>
</p>

<p align="center">
  <a href="https://cklxx.github.io/agent-infer/">Website</a> ·
  <a href="#-latest-updates">News</a> ·
  <a href="#-status-at-a-glance">Status</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#api">API</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="ROADMAP.md">Roadmap</a> ·
  <a href="CHANGELOG.md">Changelog</a>
</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

---

## 📰 Latest Updates

<!-- Keep this list to the last 3 entries. Older history lives in CHANGELOG.md. -->

- **2026-04-22** — CUDA `Qwen3.5` now ships through a true packed multi-request paged-prefill path. Full-attention layers write directly into the paged pool, hybrid linear-attention layers use packed recurrent-state launches, and paged-prefill logits now land on the canonical sampling surface. See [`docs/plans/2026-04-22-sglang-gap-closure-execution.md`](docs/plans/2026-04-22-sglang-gap-closure-execution.md).
- **2026-04-22** — CUDA scheduler overlap was tightened again: decode launch/readback is split across iterations, fetch waits are event-driven instead of hot polling, and streaming emit now goes through a dedicated emit worker with gate results fed back into the scheduler loop. Runtime ownership graph: [`docs/projects/tiered-kv-runtime-flow.md`](docs/projects/tiered-kv-runtime-flow.md).
- **2026-04-20** — Metal DFlash long-prompt prefill fixed (`fast_forward_prefill`, commit `3bc8802`) and batched terminal `eval` deferred via `async_eval` (commit `d8cb2f4`). DFlash is now default-on for Qwen3.5 on Metal, validated across guidellm's 10-strategy sweep with 5400-token prompts — zero `WrongPhase` errors, 100% request success. Canonical usage: [`docs/resources/metal-dflash.md`](docs/resources/metal-dflash.md).

Full history: [CHANGELOG.md](CHANGELOG.md) · Next up: [ROADMAP.md](ROADMAP.md)

## 🚦 Status at a glance

Four axes, each answering one question. Authoritative matrix lives in
[docs/support-matrix.md](docs/support-matrix.md); stability tiers
(**Stable** → **Beta** → **Dev**) are defined in
[docs/stability-policy.md](docs/stability-policy.md).

### Backends — *where does it run?*

| Backend | Platform | Status | What ships |
|---------|----------|:------:|------------|
| **CUDA** | Linux + NVIDIA | **Stable** | Primary serving path. Continuous batching, paged KV, radix-backed reuse, tiered-KV readmission, FlashInfer, CUDA Graph decode, packed paged-prefill for Qwen3 and Qwen3.5. |
| **Metal** | Apple Silicon | **Beta** | Live scheduler-backed serving, chunked prefill, replay-backed prefix reuse. Still behind CUDA on serving-grade batched decode and long-context parity. |
| **Metal DFlash** | Apple Silicon | **Beta — default-on** | Speculative decode for Qwen3 / Qwen3.5. Qwen3-4B bf16 5.9× decode, Qwen3.5-4B-4bit bit-identical parity, c=1..8 validated (2026-04-20). |
| **CPU** | Portable | **Dev only** | Smoke tests and request-path validation. Not a serving target. |

### Models — *what does it load?*

| Model | Attention | CUDA | Metal |
|-------|-----------|:----:|:-----:|
| Qwen3 (0.6B – 72B) | GQA | ✅ | ✅ |
| Qwen3.5-4B | Hybrid (linear + full) | ✅ | ✅ |
| GLM4 | GQA | ✅ | — |
| Llama 3 / 4 | GQA | *planned* | *planned* |
| DeepSeek V3 / R1 | MLA | *planned* | *planned* |

### HTTP API — *what can clients call?*

| Endpoint | Status | Notes |
|----------|:------:|-------|
| `POST /v1/completions` · `POST /v1/chat/completions` · `GET /v1/models` | **Stable** | OpenAI-compatible. Chat supports SSE streaming. |
| `POST /v1/responses` | **Beta** | Text/tool-call subset with both non-streaming and SSE forms. |
| `GET /metrics` · `GET /v1/stats` | **Stable** | Prometheus + human-readable ops surface. |

### Quantization — *how small does it get?*

| Format | Status | Available on |
|--------|:------:|--------------|
| FP8 / INT8 / TurboQuant KV | **Beta** | CUDA |
| GPTQ W4 · AWQ W4 | **Beta** | CUDA |
| Q4_K GGUF | **Beta** | CUDA |
| MLX 4-bit | **Beta** | Metal (default for the canonical `start_metal_serve.sh` path) |

---

<!--
  Everything below is stable reference material: features, install, API,
  architecture, development workflow. It changes only on architectural or
  API-level shifts. Fresh project state lives in the two sections above.
-->

## Why agent-infer?

In agent workloads every turn pays a prefill tax: system prompt + conversation history + tool results must be re-processed. As context grows, **prefill dominates latency**.

agent-infer treats this as the core problem:

| Capability | What it does | Impact |
|---|---|---|
| **Multi-turn KV reuse** | Slot-sticky reuse keeps prior-turn KV hot for the next turn. CUDA also carries a radix-backed tiered-KV path (`T0 GPU -> T1 host pinned -> T2 local disk -> T3 cluster-shared backend surface`) for full-block reuse and staged readmission. | Only the new user message prefills each turn when the prefix stays reusable |
| **Paged KV pool** | Main CUDA KV formats use `page_size=16`, with direct GPU page attach and tail-page CoW on shared prefixes. | Predictable KV accounting, reusable full blocks, cheaper prefix sharing |
| **Transparent slower-tier spill / promote** | Cold blocks can spill from GPU to host pinned memory and local disk, then promote back before use. The in-tree cluster-shared path is currently a minimal shared-fs backend. | Longer contexts and cached-prefix reuse beyond pure GPU residency |
| **Shared-prefix CoW** | Shared full blocks stay immutable on the radix path; writes split only the active tail page. | Shared prefixes across concurrent requests do not multiply base KV memory |
| **Scheduler overlap** | CUDA scheduler overlaps decode launch/readback across iterations, sleeps on fetch waits instead of spinning, and uses an emit worker for streaming text decode and stop scanning. | Better CPU/GPU overlap and less scheduler-side overhead at concurrency |

Current benchmark closure work is focused on high-concurrency CUDA parity
(`c4/c8/c16`) against SGLang; treat the dated headline snapshots below as
historical records, not as the current public claim.

---

## Performance

Canonical benchmark source of truth is the dated snapshot log under
[`docs/experience/wins/`](docs/experience/wins/), produced via
[`scripts/bench_guidellm.sh`](scripts/bench_guidellm.sh).

The current CUDA benchmark program is not "publish one timeless headline
number"; it is an active closure effort against SGLang focused on the
remaining high-concurrency gap:

- `c1`: near parity
- `c2`: small throughput deficit
- `c4/c8/c16`: still the main open optimization target

Active execution plan:
[`docs/plans/2026-04-22-sglang-gap-closure-execution.md`](docs/plans/2026-04-22-sglang-gap-closure-execution.md)

<details>
<summary>Agent benchmark (multi-turn tool calling)</summary>

| Model | Turns | Tool Calls | Re-prefill across turns | Avg Latency |
|-------|:-----:|:----------:|:-----------------------:|:-----------:|
| Qwen3-4B | 10 | 8 | **none** (only new user tokens per turn) | 31.9s |
| Qwen3-8B | 10 | 11 | **none** (only new user tokens per turn) | 88.5s |

_"Re-prefill across turns = none" means no prior-turn token is re-processed by the
model: the full conversation KV is reused in place between turns, so prefill cost
scales with the new user message length, not total context length._

</details>

<details>
<summary>Optimization journey (C=8)</summary>

| Phase | What changed | Throughput | Delta |
|:-----:|---|:----------:|:-----:|
| 0 | Per-request decode loop | 128 tok/s | — |
| 1 | Token-level KV pool + FlashInfer paged decode | 434 tok/s | +239% |
| 2 | Buffer pre-allocation | 681 tok/s | +57% |
| 3 | FlashInfer plan-once-per-step | 690 tok/s | +1% |
| 4 | Embedding + logits buffer pre-alloc | 700 tok/s | +1% |
| 5 | CUDA Graph batched decode | 756 tok/s | +8% |
| 6 | Batched argmax + skip D2D scatter | **811 tok/s** | +7% |

</details>

Latest bench snapshots: [docs/experience/wins/](docs/experience/wins/) ·
Run your own: [docs/plans/guidellm-integration.md](docs/plans/guidellm-integration.md)

---

## Quick Start

```bash
# Docker (recommended)
docker run --gpus all -v /path/to/Qwen3-4B:/model \
  ghcr.io/cklxx/agent-infer:latest --model-path /model --port 8000

# Or build from source
git clone https://github.com/cklxx/agent-infer && cd agent-infer
cargo build -p infer --release
./target/release/infer --model-path /path/to/Qwen3-4B --port 8000
```

```bash
# Test it
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

**Prerequisites**: CUDA 12.x, Rust 1.85+, Python 3.10+ with `flashinfer-python` (build-time only). Zig `0.16.0` for `crates/kv-native-sys` is bootstrapped by [`scripts/setup_zig_toolchain.sh`](scripts/setup_zig_toolchain.sh) and `./setup.sh`.

## Metal on Apple Silicon

```bash
# Default: builds + runs with Metal features, binds 127.0.0.1:8000,
#          loads mlx-community/Qwen3-0.6B-4bit
./scripts/start_metal_serve.sh

# Override model + port; extra flags after `--`
./scripts/start_metal_serve.sh mlx-community/Qwen3-4B-bf16 8012 -- --warmup 0

# Speculative decode (DFlash, default-on for Qwen3.5 target + draft)
./scripts/run_dflash.sh           # serve
./scripts/run_dflash.sh bench     # baseline vs DFlash throughput
```

`metal_serve`, `metal_bench`, `metal_request` also expose
`--memory-limit-bytes`, `--cache-limit-bytes`, `--wired-limit-bytes` for MLX
allocator caps. Full DFlash reference and supported model pairs:
[docs/resources/metal-dflash.md](docs/resources/metal-dflash.md).

---

## Supported Models

| Model | Attention | Status |
|-------|-----------|:------:|
| Qwen3 (0.5B-72B) | GQA | :white_check_mark: |
| Qwen3.5-4B | Hybrid (linear + full attention) | :white_check_mark: |
| GLM4 | GQA | :white_check_mark: |
| Llama 3 / 4 | GQA | Planned |
| DeepSeek-V3 / R1 | MLA | Planned |

See [ROADMAP.md](ROADMAP.md) for the full plan.

---

## API

OpenAI-compatible. Current HTTP surface:

- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /healthz`
- `GET /readyz`
- `POST /v1/responses` for the current text/tool-call subset
- `POST /v1/sessions/{session_id}/save`
- `POST /v1/sessions/{session_id}/load`
- `GET /v1/sessions/{session_id}/manifest`
- `DELETE /v1/sessions/{session_id}`

Streaming today is available on both `/v1/chat/completions` and
`/v1/responses`. On `/v1/chat/completions`, tool definitions are currently
non-streaming only: `stream=true` requests with `tools` are rejected until
the server can emit structured `delta.tool_calls` chunks.

HTTP boundary guarantees:

- JSON routes require `Content-Type: application/json`; malformed JSON, missing content type, and oversized bodies return structured JSON errors instead of framework default text.
- Unsupported top-level parameters on `/v1/completions`, `/v1/chat/completions`, and `/v1/responses` return structured `invalid_parameter` errors instead of being silently ignored.
- Blank `prompt`, empty `messages`, and blank `input` are validated through the same structured `invalid_parameter` path instead of route-specific error codes.
- `model` is optional on request bodies, but when present it must match the currently served model reported by `GET /v1/models` (case-insensitive, final path segment match allowed); mismatches return `404 model_not_found`.
- Streaming completions/chat accept `stream_options.include_usage`; `/v1/completions` also accepts `stream_options.continuous_usage_stats` as a compatibility hint, and it requires `stream_options.include_usage=true`.
- Chat / responses message validation is explicit: supported roles are `system`, `user`, `assistant`, and `tool`; `content` part arrays must be text-only; tool definitions must use `type=function`; malformed assistant `tool_calls` and tool messages without `tool_call_id` are rejected with structured `invalid_parameter` errors.
- `/v1/chat/completions` rejects `stream=true` when `tools` are present instead of pretending to support streamed tool-call deltas it cannot emit yet.
- Request body limit for JSON routes is an explicit `16 MiB`.
- Optional auth uses `Authorization: Bearer <token>`; `401` responses include `WWW-Authenticate: Bearer realm="agent-infer"`.
- Every HTTP response includes `X-Request-Id`; a client-supplied value is preserved when valid, otherwise the server generates one.
- `GET /healthz` and `GET /readyz` are lightweight unauthenticated JSON probes; `readyz` includes the boot-time model identity snapshot without probing the backend again.
- `405 Method Not Allowed` responses keep structured JSON bodies and now also include the standard `Allow` header on both top-level and session routes.

```bash
# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain KV caching"}],"stream":true}'

# Completions
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The quick brown fox","max_tokens":64,"temperature":0.7}'

# Model discovery
curl http://localhost:8000/v1/models

# Responses API
curl http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"input":"Summarize radix prefix caching in one sentence.","max_output_tokens":32}'

# Responses API streaming
curl -N http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"input":"Summarize radix prefix caching in one sentence.","max_output_tokens":32,"stream":true}'
```

<details>
<summary>Full parameter reference</summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | int | 16 | Maximum output tokens |
| `temperature` | float | 0.0 | Sampling temperature (0 = greedy) |
| `top_p` | float | 1.0 | Nucleus sampling |
| `top_k` | int | -1 | Top-K (-1 = off) |
| `min_p` | float | 0.0 | Min-P filter |
| `repetition_penalty` | float | 1.0 | Repetition penalty |
| `frequency_penalty` | float | 0.0 | Frequency penalty |
| `presence_penalty` | float | 0.0 | Presence penalty |
| `stop` | list | null | Stop strings |
| `seed` | int | null | RNG seed |
| `stream` | bool | false | SSE streaming |

</details>

Additional endpoints: `GET /healthz`, `GET /readyz`, `GET /metrics`
(Prometheus), `GET /v1/stats` (human-readable). On Metal, `/metrics` and
`/v1/stats` expose live queue / latency / MLX memory stats from the running
runtime.

---

## Agent Mode

Built-in agent runtime with tool calling:

```bash
./target/release/agent-infer \
  --max-turns 10 --temperature 0
```

The root CLI binary is behind the `cli` feature. Without `--features cli`, `agent-infer` is not built.

Current package boundary for agent mode:

- `agent-infer` -> thin binary wrapper
- `cli` -> REPL and slash commands
- `infer` -> `server_engine::LoadedInferenceEngine` backend loading and `hf_hub::resolve_model_source` for model auto-discovery
- `agent` -> conversation loop and tool-call recovery
- `tools` / `chat` -> shared tool definitions, execution helpers, and protocol types

If `--model-path` is omitted, the CLI first checks `AGENT_INFER_MODEL`, then auto-detects a local model from common directories and the local HuggingFace cache.

Use `--doctor` to print a self-check report for the current CLI build without
loading a model. It shows the compiled backend, detected hardware, TTY state,
HuggingFace cache root, model-resolution source, and curated model
recommendations. Add `--json` for machine-readable output in scripts and CI.

```bash
cargo run -p agent-infer --release --no-default-features --features cpu,no-cuda,cli -- \
  --doctor

cargo run -p agent-infer --release --no-default-features --features cli,no-cuda -- \
  --doctor --json
```

Use `--list-models` for the lighter-weight discovery view when you only want
the resolved model source, supported local hub snapshots, and curated
recommendations without the full environment report. It also supports
`--json`.

```bash
cargo run -p agent-infer --release --no-default-features --features cli,no-cuda -- \
  --list-models

cargo run -p agent-infer --release --no-default-features --features cli,no-cuda -- \
  --list-models --json
```

Invalid `--max-turns`, `--max-tokens`, and `--temperature` values fail during
argument parsing instead of surfacing later at runtime.

Tools: `python` (execute Python snippets), `shell` (execute bash commands). The
KV prefix cache reuses the full prior-turn KV in place for every subsequent
turn of a session — only the new user message (and any tool-result content)
runs prefill. CUDA runtime reuse now sits on top of the same radix / tiered-KV
surface used by the HTTP scheduler; the main remaining model-specific limit is
that hybrid `Qwen3.5` does not yet support cross-slot partial-prefix restore.
On macOS, tool execution uses `sandbox-exec` automatically when `nsjail` is unavailable; Linux keeps using `nsjail` when installed.

On Apple Silicon, build the same CLI against the Metal backend:

```bash
cargo run --release --no-default-features --features metal,no-cuda,cli -- \
  --model-path mlx-community/Qwen3-0.6B-4bit
```

The CLI keeps conversation history across turns, stores line history in `~/.agent-infer-history`, and supports slash commands:

- `/help` for command help
- `/reset` or `/clear` to clear the current conversation
- `/tools` to inspect built-in tools
- `/model` and `/stats` to inspect the loaded runtime
- `/save <path>` and `/load <path>` to persist or resume a session as JSON

---

## Architecture

Workspace split:

- `agent-infer` — thin binary wrapper
- `cli` — REPL / CLI flow
- `agent` — conversation state, tool-call recovery, agent turn loop
- `tools` / `chat` — tool execution helpers and protocol types
- `infer` — HTTP server, scheduler, runtime, backend implementations; owns the single `InferenceEngine` contract
- `cuda-kernels` — extracted CUDA kernel layer (csrc/, Triton AOT, Rust FFI). One-way dep: `infer → cuda-kernels`.
- `mlx-sys` — MLX C++ bridge for the Metal backend

See [docs/architecture.md](docs/architecture.md), [docs/codebase-map.md](docs/codebase-map.md), and [crates/README.md](crates/README.md)
for the current package boundaries.

```
┌───────────────────────────────────────────────────────────────────────┐
│ HTTP / agent request ingress                                         │
└───────────────────────────────┬───────────────────────────────────────┘
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│ CUDA scheduler runtime                                               │
│ runtime.rs / execution.rs / core.rs                                  │
│ - assign_slots()  - plan_step()  - step()  - cleanup()               │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ prefix lookup / publish
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│ RadixCache / CacheIndex                                              │
│ prefix_cache.rs                                                      │
└───────────────┬───────────────────────────────┬───────────────────────┘
                │ ready on T0                   │ staged below T0
                ▼                               ▼
      ┌──────────────────────┐        ┌──────────────────────────────┐
      │ paged_kv / T0 GPU    │        │ ReadmissionPlan + Coordinator│
      │ direct attach + CoW  │        │ PlanQueue / Fetch / Store    │
      └──────────┬───────────┘        └──────────────┬───────────────┘
                 │                                   │
                 ▼                                   ▼
      ┌──────────────────────┐      ┌─────────────────────────────────┐
      │ ModelForward         │      │ T1 host pinned / T2 disk / T3   │
      │ Qwen3 · Qwen3.5      │      │ cluster-shared backend surface  │
      └──────────┬───────────┘      └────────────────┬────────────────┘
                 │                                    │ promote / restore
                 └──────────────────┬─────────────────┘
                                    ▼
                     FlashInfer · RMSNorm · cuBLAS GEMM · CUDA Graph
                        (CUDA C + Triton AOT, crates/cuda-kernels/)

Separate from the scheduler hot path:
- emit worker: UTF-8 decode, delta emission, stop-sequence scan
- kv-native-sys: Zig substrate for host/disk KV plumbing
```

Detailed runtime ownership graph:
[docs/projects/tiered-kv-runtime-flow.md](docs/projects/tiered-kv-runtime-flow.md).

---

## Stability and Support

`agent-infer` uses explicit stability tiers:

- **Stable**: documented HTTP endpoints (`/v1/completions`, `/v1/chat/completions`,
  `GET /v1/models`, `GET /healthz`, `GET /readyz`), `GET /metrics`,
  `GET /v1/stats`, and the main documented build/test workflows.
- **Beta**: `POST /v1/responses` (current text/tool-call subset with
  non-streaming and SSE forms), CLI agent behavior, Metal serving path, GGUF
  loading, benchmark tooling.
- **Experimental**: fast-moving quantization paths, tensor-parallel
  scaffolding, and undocumented flags or environment variables.

Current project state lives in [§Status at a glance](#-status-at-a-glance) above
and in the authoritative [docs/support-matrix.md](docs/support-matrix.md).

Governance references:

- [docs/stability-policy.md](docs/stability-policy.md)
- [docs/compatibility.md](docs/compatibility.md)
- [docs/perf-and-correctness-gates.md](docs/perf-and-correctness-gates.md)
- [docs/release-checklist.md](docs/release-checklist.md)
- [docs/environment.md](docs/environment.md)

---

## Development

```bash
make install-hooks                                      # Install repo-managed Git hooks (.githooks/pre-push)
make pre-push                                           # Run the CI-aligned local pre-push checks
./scripts/setup_zig_toolchain.sh                          # Validate/install Zig 0.16.0 for kv-native-sys
./scripts/check_kv_zig.sh                                 # Zig substrate local validation
cargo test --no-default-features --features no-cuda   # Unit tests (no GPU)
cargo clippy --workspace -- -D warnings                # Lint
cargo fmt --all -- --check                             # Format

# CPU backend smoke path (downloads runtime assets like config/tokenizer, not full weights)
cargo run -p agent-infer --no-default-features --features cpu,no-cuda,cli -- \
  --model-path Qwen/Qwen3-0.6B --max-turns 1 --max-tokens 64

# E2E (requires GPU + model weights)
INFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e

# Agent CLI live-model E2E on Apple Silicon (auto-detects a local model when available)
cargo test --release --no-default-features --features metal,no-cuda,cli -- --ignored --nocapture
```

Before opening a PR: [CONTRIBUTING.md](CONTRIBUTING.md),
[support-matrix](docs/support-matrix.md),
[perf-and-correctness-gates](docs/perf-and-correctness-gates.md),
[compatibility](docs/compatibility.md),
[environment](docs/environment.md). Release work:
[release-checklist](docs/release-checklist.md).

After `make install-hooks`, every `git push` runs `.githooks/pre-push`, which
delegates to `scripts/pre_push_checks.sh`. Set
`AGENT_INFER_SKIP_PRE_PUSH=1` only when you explicitly need to bypass the hook.

---

## License

[MIT](LICENSE)
