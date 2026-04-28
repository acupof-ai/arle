<p align="center">
  <strong>ARLE</strong><br>
  <em>Pure-Rust runtime for serving, local agents, training, and evaluation. <code>infer</code> is the OpenAI-compatible serving binary; <code>arle</code> is the unified front door.</em>
</p>

<p align="center">
  <a href="https://cklxx.github.io/arle/"><img src="https://img.shields.io/badge/website-cklxx.github.io%2Farle-D97757?style=flat-square" alt="Website"></a>
  <a href="https://github.com/cklxx/arle/actions/workflows/ci.yml"><img src="https://github.com/cklxx/arle/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/cklxx/arle/actions/workflows/cuda-ci.yml"><img src="https://github.com/cklxx/arle/actions/workflows/cuda-ci.yml/badge.svg" alt="CUDA CI"></a>
  <a href="https://github.com/cklxx/arle/actions/workflows/metal-ci.yml"><img src="https://github.com/cklxx/arle/actions/workflows/metal-ci.yml/badge.svg" alt="Metal CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://github.com/cklxx/arle/releases"><img src="https://img.shields.io/github/v/release/cklxx/arle?include_prereleases" alt="Release"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="docs/http-api.md">HTTP API</a> ·
  <a href="docs/support-matrix.md">Support Matrix</a> ·
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="ROADMAP.md">Roadmap</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

---

## Quick Start

### 1. Install

**Apple Silicon — Homebrew (recommended):**

```bash
brew install cklxx/tap/arle
arle --doctor
```

**Apple Silicon or Linux x86_64 — one-line installer:**

```bash
curl -fsSL https://github.com/cklxx/arle/releases/latest/download/install.sh | sh
```

The script grabs the matching tarball from the latest GitHub Release,
SHA256-verifies it, and drops the binaries into `~/.local/bin` (override
with `INSTALL_DIR=...`). See [docs/install.md](docs/install.md) for the full
matrix, env-var overrides, and uninstall steps.

**Linux + NVIDIA — pull the published Docker image, no compile:**

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v /path/to/Qwen3-4B:/model:ro \
  ghcr.io/cklxx/arle:latest \
  serve --backend cuda --model-path /model --port 8000
```

The `:latest` tag tracks `main`; tagged releases are published as
`ghcr.io/cklxx/arle:X.Y.Z` (note: no `v` prefix — docker metadata-action
strips it). For v0.1.0 today: `ghcr.io/cklxx/arle:0.1.0`.

**From source** (any backend; needed for `cpu`, `tilelang-attn`, or local hacking):

```bash
git clone https://github.com/cklxx/arle && cd arle
# Apple Silicon:
cargo build --release --no-default-features --features metal,no-cuda,cli --bin arle
# Linux + NVIDIA:
cargo build --release --features cli --bin arle
```

### 2. Serve a model

```bash
arle serve --backend metal \
  --model-path mlx-community/Qwen3-0.6B-4bit --port 8000   # Apple Silicon
arle serve --backend cuda \
  --model-path /path/to/Qwen3-4B --port 8000               # Linux + NVIDIA
```

### 3. Talk to it

```python
# pip install openai
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "Hello from ARLE"}],
).choices[0].message.content)
```

Or with curl: see [`examples/curl_chat.sh`](examples/curl_chat.sh).
More copy-paste paths: [`examples/`](examples/).

### 4. Run the local agent

```bash
arle                                                       # interactive REPL with built-in tools
arle --model-path /path/to/Qwen3-4B run --prompt "Summarize this repo"   # one-shot
arle --doctor --json                                       # self-check, machine-readable
```

CPU-only smoke build (no GPU required, source build):

```bash
cargo build --release --no-default-features --features cpu,no-cuda,cli --bin arle
./target/release/arle --doctor
```

---

## Status at a glance

| Backend | Platform | Status | Notes |
|---|---|:---:|---|
| **CUDA** | Linux + NVIDIA | **Stable** | Continuous batching, paged KV, radix-backed reuse, FlashInfer, CUDA Graph decode, packed paged-prefill for Qwen3 / Qwen3.5. **L4 / Qwen3-4B BF16 + FP8 paged KV (auto): 197 tok/s @ c=16 / 4096-in, peak_active=16 saturated.** |
| **Metal** | Apple Silicon | **Beta** | Live scheduler-backed serving, chunked prefill, replay-backed prefix reuse, and Qwen3.5-0.8B GGUF Q4_K_M decode at 211.7 tok/s on M4 Pro. |
| **Metal DFlash** | Apple Silicon | **Beta — default-on** | Speculative decode for Qwen3 / Qwen3.5. Qwen3-4B bf16 5.9× decode, Qwen3.5-4B-4bit bit-identical parity, c=1..8 validated. |
| **CPU** | Portable | **Dev-only** | Smoke tests and request-path validation; not a serving target. |

Models: **Qwen3 (0.6B – 72B)** and the **Qwen3.5 family** (including 0.8B GGUF Q4_K_M and 4B hybrid linear + full attention) are supported on CUDA and Metal according to the current matrix. **Qwen3.6 / Qwen3.5-MoE** has a narrow Metal Beta path; CUDA remains stubbed. Llama 3 / 4 and DeepSeek V3 / R1 are planned — see [ROADMAP.md](ROADMAP.md).

Authoritative matrix (HTTP API tiers, quantization, agent / train / eval surfaces): [docs/support-matrix.md](docs/support-matrix.md).
Stability tiers: [docs/stability-policy.md](docs/stability-policy.md).

---

## Why ARLE

In agent and RL workloads every turn pays a prefill tax: system prompt + history + tool results must be re-processed. As context grows, **prefill dominates latency**. ARLE treats this as the core problem in both serving and agent / RL loops:

- **Multi-turn KV reuse.** Slot-sticky reuse keeps prior-turn KV hot for the next turn. CUDA also carries a radix-backed tiered-KV path (`T0 GPU → T1 host pinned → T2 local disk → T3 cluster-shared`) for full-block reuse and staged readmission, so only the new user message prefills each turn when the prefix stays reusable.
- **Paged KV pool.** Main CUDA KV formats use `page_size=16` with direct GPU page attach and tail-page CoW on shared prefixes — predictable accounting, reusable full blocks, cheaper prefix sharing.
- **Shared runtime authority.** `infer`, `arle`, and the in-tree train / eval jobs resolve models and reuse the same Rust runtime / model contracts. Serving, local agent work, and RL tooling stay on one code path instead of drifting across separate stacks.

Architecture deep-dive: [docs/architecture.md](docs/architecture.md) · [docs/codebase-map.md](docs/codebase-map.md).
Latest benchmark snapshots (per change, dated): [docs/experience/wins/](docs/experience/wins/) · run your own with [`scripts/bench_guidellm.sh`](scripts/bench_guidellm.sh).

---

## Entry surfaces

`arle` is the single binary users interact with:

| Command | What it does |
|---|---|
| `arle` (no args) | Interactive agent REPL with built-in `python` and `shell` tools (sandboxed). |
| `arle run --prompt "…"` / `--stdin --json` | Script-friendly one-shot agent prompt. Use `--no-tools` to disable tool execution. |
| `arle serve --backend {cuda,metal,cpu} --model-path …` | Launch the OpenAI-compatible HTTP server. |
| `arle train {pretrain,sft,grpo,multi-turn,eval}` | In-tree training and RL workflows on the same runtime. |
| `arle data {download,convert}` | Dataset utilities. |
| `arle --doctor [--json] [--strict]` | Self-check: backend, hardware, HF cache, model resolution. CI-friendly. |

The REPL persists line history at `~/.arle-history` and exposes slash commands: `/help`, `/reset`, `/clear`, `/tools`, `/model`, `/stats`, `/models`, `/save`, `/load`, `/export`.

Operators who want only the serving binary can use `infer` directly (`cargo build -p infer --release --features cuda` on Linux, `--features metal,no-cuda` on Apple Silicon) — same HTTP contract, no agent / train / data surface.

---

## 📰 Latest Updates

<!-- Keep this list to the last 2 entries. Older history lives in CHANGELOG.md. -->

- **2026-04-28** — CUDA L4 `Qwen3-4B` BF16, c=16 / 4096-in jumped from **120 → 197 tok/s (+64%)** after auto HBM-tier `chunked_prefill_size` and FP8 paged KV defaulting on L4-class GPUs. `peak_active` saturates at 16/16; +42% vs SGLang reference on the same workload. Evidence: [`docs/experience/wins/2026-04-28-bench-guidellm-cuda-l4-kv-fp8-auto.md`](docs/experience/wins/2026-04-28-bench-guidellm-cuda-l4-kv-fp8-auto.md).
- **2026-04-27** — Metal `Qwen3.5-0.8B` GGUF `Q4_K_M` decode crossed 200 tok/s on M4 Pro after Q5_K/Q8_0 affine repack and Q6/group16 qmv tile tuning. Evidence: [`docs/experience/wins/2026-04-27-bench-metal-qwen35-0p8b-gguf-q5-q8-q6qmv.md`](docs/experience/wins/2026-04-27-bench-metal-qwen35-0p8b-gguf-q5-q8-q6qmv.md).

Full history: [CHANGELOG.md](CHANGELOG.md). Next up: [ROADMAP.md](ROADMAP.md).

---

## Documentation map

- [docs/http-api.md](docs/http-api.md) — HTTP route contract, streaming behavior, boundary guarantees
- [docs/support-matrix.md](docs/support-matrix.md) — backend / model / quant / API support tiers
- [docs/stability-policy.md](docs/stability-policy.md) — stability levels and compatibility posture
- [docs/architecture.md](docs/architecture.md) — package boundaries and dependency direction
- [docs/codebase-map.md](docs/codebase-map.md) — workspace layout and main execution paths
- [docs/environment.md](docs/environment.md) — environment variables and runtime knobs
- [docs/troubleshooting.md](docs/troubleshooting.md) — common build / runtime errors and fixes
- [docs/comparison.md](docs/comparison.md) — how ARLE compares to vLLM / SGLang / mistral.rs / llama.cpp
- [docs/release-checklist.md](docs/release-checklist.md) · [docs/perf-and-correctness-gates.md](docs/perf-and-correctness-gates.md)
- [CONTRIBUTING.md](CONTRIBUTING.md) — contributor setup, validation, release expectations
- [SECURITY.md](SECURITY.md) — vulnerability reporting policy
- [examples/](examples/) — copy-paste smoke paths (curl, OpenAI SDK, Docker, Metal, train fixtures)
- [docs/index.md](docs/index.md) — maintainer-facing PARA index, plans, and experience logs

---

## License

[MIT](LICENSE)
