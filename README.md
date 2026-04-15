<p align="center">
  <strong>agent-infer</strong><br>
  <em>KV-cache-first inference engine for LLM agents. Pure Rust, with CUDA as the primary serving path.</em>
</p>

<p align="center">
  <a href="https://github.com/cklxx/agent-infer/actions"><img src="https://github.com/cklxx/agent-infer/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://github.com/cklxx/agent-infer/releases"><img src="https://img.shields.io/github/v/release/cklxx/agent-infer?include_prereleases" alt="Release"></a>
</p>

---

## Why agent-infer?

In agent workloads every turn pays a prefill tax: system prompt + conversation history + tool results must be re-processed. As context grows, **prefill dominates latency**.

agent-infer treats this as the core problem:

| Capability | What it does | Impact |
|---|---|---|
| **Multi-turn KV reuse** | Slot-sticky prefix matching reuses the previous turn's KV in place. Shared system prompts and conversation prefixes skip prefill entirely. Cross-session radix-tree reuse wiring lands in the tiered-kv-cache M1 milestone; today's fast path is a per-slot linear compare. | 100% cache hit on single-session multi-turn agent benchmarks |
| **Token-level KV pool** | page_size=1 pooling (SGLang-style). Zero fragmentation, instant alloc/free. | Eliminates memory waste from fixed-page padding |
| **Transparent GPU-CPU offload** | Oldest KV blocks migrate to host RAM; prefetch back before attention. | Contexts beyond GPU VRAM capacity |
| **Copy-on-write block sharing** | Paged blocks with ref-counting. Shared prefixes across concurrent requests use one copy. | N requests, 1x prefix memory |
| **CUDA Graph batched decode** | One captured graph per batch size (1-32). 504 kernel launches → 1 replay. | Eliminates CPU-GPU dispatch overhead |

**The result**: 4.6x faster time-to-first-token than SGLang, with matching throughput — because the cache does the heavy lifting.

---

## Performance (Qwen3-4B, A100-SXM4-40GB)

| Concurrency | Throughput | vs SGLang v0.5.9 | TTFT | vs SGLang |
|:-----------:|:----------:|:-----------------:|:----:|:---------:|
| 1 | 119.5 tok/s | 0.99x | **8.6ms** | **4.6x faster** |
| 4 | 414.8 tok/s | 0.99x | **53.1ms** | **2.6x faster** |
| 8 | 811 tok/s | 0.92x | **68.7ms** | **1.1x faster** |

<details>
<summary>Agent benchmark (multi-turn tool calling)</summary>

| Model | Turns | Tool Calls | KV Hit Rate | Avg Latency |
|-------|:-----:|:----------:|:-----------:|:-----------:|
| Qwen3-4B | 10 | 8 | **100%** | 31.9s |
| Qwen3-8B | 10 | 11 | **100%** | 88.5s |

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

## Metal DFlash

Metal DFlash is available as an experimental Apple Silicon decode path for
`Qwen3`.

Quick example:

```bash
cargo run -p infer --bin metal_request --release --no-default-features --features metal,no-cuda -- \
  --model mlx-community/Qwen3-4B-bf16 \
  --dflash-draft-model z-lab/Qwen3-4B-DFlash-b16 \
  --prompt "write a quicksort in python" \
  --raw-prompt
```

For full usage, limits, and benchmark workflow, see
[docs/resources/metal-dflash.md](docs/resources/metal-dflash.md).

---

## Stability and Support

`agent-infer` uses explicit stability and support rules.

- **Stable**: documented HTTP endpoints (`/v1/completions`, `/v1/chat/completions`,
  `GET /v1/models`), `GET /metrics`, `GET /v1/stats`, and the main documented
  build/test workflows.
- **Beta**: `POST /v1/responses` (current non-streaming subset), CLI agent
  behavior, Metal serving path, GGUF loading, benchmark tooling.
- **Experimental**: fast-moving quantization paths, speculative decoding,
  tensor-parallel scaffolding, Metal DFlash, and undocumented flags or
  environment variables.

Current support should be read conservatively:

- **CUDA on Linux** is the primary supported serving path.
- **Metal on Apple Silicon** is usable, but not yet equivalent to the CUDA
  scheduler runtime. Standard `metal_serve` now uses a live Metal scheduler
  runtime, but batched decode / prefix-reuse parity is still pending.
- **CPU-only / `no-cuda`** now includes a development-oriented CPU backend for
  local smoke tests and request-path validation, but it is still not a
  production inference target.

Governance references:

- [docs/stability-policy.md](docs/stability-policy.md)
- [docs/support-matrix.md](docs/support-matrix.md)
- [docs/compatibility.md](docs/compatibility.md)
- [docs/perf-and-correctness-gates.md](docs/perf-and-correctness-gates.md)
- [docs/release-checklist.md](docs/release-checklist.md)
- [docs/environment.md](docs/environment.md)

---

## Architecture

Workspace split summary:

- `agent-infer` is now a thin binary wrapper.
- `infer-cli` owns the REPL/CLI flow.
- `infer-agent` owns conversation state, tool-call recovery, and the agent turn loop.
- `infer-tools` and `infer-chat` are reusable tool execution helpers and protocol types.
- `infer` continues to own the HTTP server, scheduler, runtime, and backend
  implementations. `infer::server_engine::{InferenceEngine, LoadedInferenceEngine,
  CompletionRequest, CompletionOutput}` is the single engine contract used by
  both the HTTP server and the agent CLI.

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
         (CUDA C + Triton AOT, crates/infer-cuda-kernels/)
```

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
(human-readable). On Metal, these now expose live queue / latency / MLX memory
stats from the running runtime, not just a detached placeholder metrics object.

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
- `infer-cli` -> REPL and slash commands
- `infer` -> `server_engine::LoadedInferenceEngine` backend loading and `hf_hub::resolve_model_source` for model auto-discovery
- `infer-agent` -> conversation loop and tool-call recovery
- `infer-tools` / `infer-chat` -> shared tool definitions, execution helpers, and protocol types

If `--model-path` is omitted, the CLI first checks `AGENT_INFER_MODEL`, then auto-detects a local model from common directories and the local HuggingFace cache.

Tools: `python` (execute Python snippets), `shell` (execute bash commands). KV prefix cache ensures each turn within a single session reuses prior context at 100% hit rate (slot-sticky match; cross-session reuse via radix tree lands in the tiered-kv-cache M1 milestone).
On macOS, tool execution now uses `sandbox-exec` automatically when `nsjail` is unavailable; Linux keeps using `nsjail` when installed.

On Apple Silicon, build the same CLI against the Metal backend:

```bash
cargo run --release --no-default-features --features metal,no-cuda,cli -- \
  --model-path mlx-community/Qwen3-0.6B-4bit
```

For OpenAI-compatible serving on Apple Silicon:

```bash
cargo run --release -p infer --no-default-features --features metal,no-cuda --bin metal_serve -- \
  --model-path mlx-community/Qwen3-0.6B-4bit --port 8000
```

Current status: standard `metal_serve` on Qwen3/Qwen3.5 now runs through a live
Metal scheduler runtime with chunked prefill and decode-priority interleave.
It now has narrow same-length cross-request decode batching for Qwen3 and
Qwen3.5, but variable-length decode is still not batched and Metal DFlash still
uses the legacy serial runtime path.

The CLI keeps conversation history across turns, stores line history in `~/.agent-infer-history`, and supports slash commands:

- `/help` for command help
- `/reset` or `/clear` to clear the current conversation
- `/tools` to inspect built-in tools
- `/model` and `/stats` to inspect the loaded runtime
- `/save <path>` and `/load <path>` to persist or resume a session as JSON

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
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e

# Agent CLI live-model E2E on Apple Silicon (auto-detects a local model when available)
cargo test --release --no-default-features --features metal,no-cuda,cli -- --ignored --nocapture
```

Before opening a PR, read:

- [CONTRIBUTING.md](CONTRIBUTING.md) for workflow and contribution rules
- [docs/support-matrix.md](docs/support-matrix.md) for what is currently supported
- [docs/perf-and-correctness-gates.md](docs/perf-and-correctness-gates.md) for
  minimum verification expectations
- [docs/compatibility.md](docs/compatibility.md) if your change affects CLI,
  API, documented env vars, or migration-sensitive behavior
- [docs/environment.md](docs/environment.md) for the current environment-variable
  reference

For release work, also use [docs/release-checklist.md](docs/release-checklist.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow.

---

## License

[MIT](LICENSE)
