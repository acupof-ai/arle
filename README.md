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

---

## 📰 Latest Updates

<!-- Keep this list to the last 3 entries. Older history lives in CHANGELOG.md. -->

- **2026-04-20** — Metal DFlash long-prompt prefill fixed (`fast_forward_prefill`, commit `3bc8802`) and batched terminal `eval` deferred via `async_eval` (commit `d8cb2f4`). DFlash is now default-on for Qwen3.5 on Metal, validated across guidellm's 10-strategy sweep with 5400-token prompts — zero `WrongPhase` errors, 100% request success. See [`docs/resources/metal-dflash.md`](docs/resources/metal-dflash.md) for the canonical usage guide.
- **2026-04-19** — DFlash ships default-on for Metal (commit `47f958f`). Qwen3.5-4B-4bit bit-identical parity against scalar path for B≤2 batched verify, concurrent c=1..8 stable.
- **2026-04-16** — Metal packed-batch concurrent decode fix: `extend_kv_cache` batch-dim bug repaired, varlen additive mask now emitted in bf16 for MLX ≥ 0.32 SDPA. Packed decode stable under 4× / 8× concurrency.

Full history: [CHANGELOG.md](CHANGELOG.md) · Next up: [ROADMAP.md](ROADMAP.md)

## 🚦 Status at a glance

| Area | Status | Notes |
|------|--------|-------|
| CUDA / Linux — Qwen3 / Qwen3.5 / GLM4 | **Supported** | Primary serving path. |
| Metal / Apple Silicon — Qwen3 / Qwen3.5 | **Beta** | Live scheduler, chunked prefill, narrow same-length packed decode. Variable-length decode not yet batched. |
| Metal DFlash (Qwen3 / Qwen3.5) | **Beta — default-on** | Shipped default-on 2026-04-19; Qwen3-4B bf16 5.9× decode, Qwen3.5-4B-4bit bit-ident parity + long-prompt + c=1..8 validated 2026-04-20. |
| CPU-only / `no-cuda` | **Development only** | Smoke tests, request-path validation. Not a production target. |
| `/v1/completions`, `/v1/chat/completions`, `/v1/models` | **Stable** | OpenAI-compatible. |
| `/v1/responses` | **Beta** | Non-streaming + SSE `output_text.delta`. |
| FP8 / INT8 / TurboQuant KV, GPTQ/AWQ W4, Q4_K GGUF | **Beta** (CUDA) | Quantized KV is CUDA-only today. |

Authoritative matrix: [docs/support-matrix.md](docs/support-matrix.md) ·
Stability policy: [docs/stability-policy.md](docs/stability-policy.md)

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
| **Multi-turn KV reuse** | Slot-sticky prefix matching reuses the previous turn's KV in place. Shared system prompts and conversation prefixes skip prefill entirely. Cross-session radix-tree reuse wiring lands in the tiered-kv-cache M1 milestone; today's fast path is a per-slot linear compare. | Only the new user message prefills each turn — prior conversation KV is never re-computed |
| **Token-level KV pool** | page_size=1 pooling (SGLang-style). Zero fragmentation, instant alloc/free. | Eliminates memory waste from fixed-page padding |
| **Transparent GPU-CPU offload** | Oldest KV blocks migrate to host RAM; prefetch back before attention. | Contexts beyond GPU VRAM capacity |
| **Copy-on-write block sharing** | Paged blocks with ref-counting. Shared prefixes across concurrent requests use one copy. | N requests, 1x prefix memory |
| **CUDA Graph batched decode** | One captured graph per batch size (1-32). 504 kernel launches → 1 replay. | Eliminates CPU-GPU dispatch overhead |

**The result**: 4.6x faster time-to-first-token than SGLang, with matching throughput — because the cache does the heavy lifting.

---

## Performance

Qwen3-4B on A100-SXM4-40GB vs SGLang v0.5.9:

| Concurrency | Throughput | vs SGLang | TTFT | vs SGLang |
|:-----------:|:----------:|:---------:|:----:|:---------:|
| 1 | 119.5 tok/s | 0.99x | **8.6ms** | **4.6x faster** |
| 4 | 414.8 tok/s | 0.99x | **53.1ms** | **2.6x faster** |
| 8 | 811 tok/s | 0.92x | **68.7ms** | **1.1x faster** |

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

**Prerequisites**: CUDA 12.x, Rust 1.85+, Python 3.10+ with `flashinfer-python` (build-time only).

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
- `POST /v1/responses` for the current non-streaming subset

Streaming today remains on `/v1/chat/completions`; `/v1/responses` returns a
clear `400` when `stream=true`.

```bash
# Streaming
curl http://localhost:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Explain KV caching"}],"stream":true}'

# Completions
curl http://localhost:8000/v1/completions \
  -d '{"prompt":"The quick brown fox","max_tokens":64,"temperature":0.7}'

# Model discovery
curl http://localhost:8000/v1/models

# Responses API (non-streaming subset)
curl http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"input":"Summarize radix prefix caching in one sentence.","max_output_tokens":32}'
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

Additional endpoints: `GET /metrics` (Prometheus), `GET /v1/stats`
(human-readable). On Metal, these expose live queue / latency / MLX memory
stats from the running runtime.

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

Tools: `python` (execute Python snippets), `shell` (execute bash commands). The
KV prefix cache reuses the full prior-turn KV in place for every subsequent
turn of a session — only the new user message (and any tool-result content)
runs prefill (slot-sticky match; cross-session reuse via radix tree lands in
the tiered-kv-cache M1 milestone).
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
┌──────────────────────────────────────────────────────────┐
│  HTTP API  (/v1/completions, /v1/chat/completions, /v1/models, /v1/responses)  │
└────────────────────────┬─────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Scheduler  (decode-priority, chunked prefill)           │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────────┐  │
│  │ Prefix Cache │  │ Token KV    │  │ Block Manager  │  │
│  │ (radix tree) │  │ Pool (p=1)  │  │ (CoW paging)   │  │
│  └──────────────┘  └─────────────┘  └────────────────┘  │
└────────────────────────┬─────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│  ModelForward trait                                       │
│  Qwen3 (GQA) · Qwen3.5 (Hybrid recurrent + attention)   │
└────────────────────────┬─────────────────────────────────┘
                         ▼
      FlashInfer · RMSNorm · cuBLAS GEMM · CUDA Graph
         (CUDA C + Triton AOT, crates/cuda-kernels/)
```

---

## Stability and Support

`agent-infer` uses explicit stability tiers:

- **Stable**: documented HTTP endpoints (`/v1/completions`, `/v1/chat/completions`,
  `GET /v1/models`), `GET /metrics`, `GET /v1/stats`, and the main documented
  build/test workflows.
- **Beta**: `POST /v1/responses` (current non-streaming subset), CLI agent
  behavior, Metal serving path, GGUF loading, benchmark tooling.
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

---

## License

[MIT](LICENSE)
