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

The `:latest` tag tracks the newest non-prerelease release image. Tagged
releases are published as `ghcr.io/cklxx/arle:X.Y.Z` (note: no `v` prefix -
the docker metadata-action strips it). For the current release:
`ghcr.io/cklxx/arle:0.1.5`.

**From source** (any backend; needed for `cpu`, CUDA/TileLang, or local hacking):

```bash
git clone https://github.com/cklxx/arle && cd arle
# Apple Silicon:
cargo build --release --no-default-features --features metal,no-cuda,cli --bin arle
# Linux + NVIDIA:
cargo build --release --features cuda --bin arle
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
| **CUDA** | Linux + NVIDIA | **Stable** | Continuous batching, paged KV, radix-backed reuse, TileLang BF16 attention, CUDA Graph decode. L4 / Qwen3-4B BF16 + FP8 KV: **197 tok/s @ c=16 / 4k-in**. |
| **Metal** | Apple Silicon | **Beta** | Scheduler-backed serving, chunked prefill, replay prefix reuse. Qwen3.5-0.8B MLX-4bit step-driver: **305.5 tok/s** on M4 Pro 20c. |
| **Metal DFlash** | Apple Silicon | **Beta — default-on** | Speculative decode for Qwen3 / Qwen3.5. Qwen3-4B bf16: **5.9× decode**; Qwen3.5-4B-4bit bit-identical, c=1..8. |
| **CPU** | Portable | **Dev-only** | Smoke tests and request-path validation; not a perf target. |

Models: **Qwen3 (0.6B – 72B)** and **Qwen3.5 family** (incl. 0.8B GGUF Q4_K_M and 4B hybrid attention) on CUDA + Metal. **Qwen3.6 / Qwen3.5-MoE** has a narrow Metal Beta path; CUDA stubbed. Next-model queue: **DeepSeek V4 (#1)** → **Qwen 3.6 (#2)**, see [ROADMAP.md](ROADMAP.md#next-model-priority-order). DeepSeek V2/V3/R1 intentionally out of scope.

Authoritative matrix (HTTP API tiers, quantization, agent / train / eval surfaces): [docs/support-matrix.md](docs/support-matrix.md).
Stability tiers: [docs/stability-policy.md](docs/stability-policy.md).

---

## Why ARLE

In agent and RL workloads every turn pays a prefill tax: system prompt + history + tool results must be re-processed. As context grows, **prefill dominates latency**. ARLE treats this as the core problem in both serving and agent / RL loops:

- **Multi-turn KV reuse.** Slot-sticky reuse keeps prior-turn KV hot for the next turn. CUDA also includes a radix-backed tiered-KV path (`T0 GPU → T1 host pinned → T2 local disk → T3 cluster-shared`) for full-block reuse and staged readmission, so only the new user message requires prefill each turn when the prefix stays reusable.
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
| `arle serve --backend {cuda,metal,cpu} --model-path …` | Launch the OpenAI-compatible HTTP server through an ARLE-native backend. |
| `arle train {pretrain,sft,grpo,multi-turn,eval}` | In-tree training and RL workflows on the same runtime. |
| `arle data {download,convert}` | Dataset utilities. |
| `arle --doctor [--json] [--strict]` | Self-check: backend, hardware, HF cache, model resolution. CI-friendly. |

The REPL persists line history at `~/.arle-history` and exposes slash commands: `/help`, `/reset`, `/clear`, `/tools`, `/model`, `/stats`, `/models`, `/save`, `/load`, `/export`.

Operators who want only the native serving binary can use `infer` directly (`cargo build -p infer --release --features cuda` on Linux, `--features metal,no-cuda` on Apple Silicon) — same HTTP contract, without the agent / train / data surface.

---

## 📰 Latest Updates

<!-- Keep this list to the last 2 entries. Older history lives in CHANGELOG.md. -->

- **2026-05-15** — DSv4 DeepEP decode lands default B=1 padded BF16
  reduce-scatter combine, fused local-expert prepare kernel, and broad
  scratch-reuse cleanup. Real 8xH20 on `DeepSeek-V4-Flash`: `decode64` holds
  **12.05 post-first tok/s**; isolated single-token nsys wave **105.2 → 87.7 ms**,
  `cuMemsetD8Async` calls **3,640 → 544**, arithmetic exact (`410`/`406`).
  Remaining stack: NCCL SendRecv/AllReduce, FP8/FP4 expert GEMV (awaits true
  grouped GEMM/DeepGEMM), launch churn, D2H route-count readback.
  Evidence: [`docs/trace-artifacts/2026-05-15-dsv4-deepep/`](docs/trace-artifacts/2026-05-15-dsv4-deepep/),
  [`docs/trace-artifacts/2026-05-14-dsv4-deepep/`](docs/trace-artifacts/2026-05-14-dsv4-deepep/),
  [`docs/experience/errors/2026-05-14-dsv4-decode-nccl-bottleneck.md`](docs/experience/errors/2026-05-14-dsv4-decode-nccl-bottleneck.md).
- **2026-05-10** — W4-hybrid prefill graph capture closes the 4k/c=4 SGLang
  +76.6% gap via Path B.2 bucketed allocation key. Engine-side TTFT p50
  **2000 → 150 ms (-92.5%)**, throughput **+632%** in 60s on RTX 4070 Ti
  SUPER 16GB. Capture-key churn **388 → 7 unique** (98.5% LRU reuse). Opt-in
  via `INFER_PREFILL_GRAPH=1` + `INFER_HYBRID_W4A8_PREFILL=1`. Also lands
  RoPE YARN/Linear/NtkAware scaling.
  Evidence: [`docs/experience/wins/2026-05-10-bench-40-pathB2-tier1-strong-proceed.md`](docs/experience/wins/2026-05-10-bench-40-pathB2-tier1-strong-proceed.md),
  [`docs/experience/wins/2026-05-10-m-rope-yarn-scaling-phase1-phase2-landed.md`](docs/experience/wins/2026-05-10-m-rope-yarn-scaling-phase1-phase2-landed.md).

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
