<p align="center">
  <strong>ARLE</strong><br>
  <em>Agent reinforcement learning engine for long-context LLM agents. Pure Rust, with CUDA as the primary serving path and train/eval/agent workflows in-tree.</em>
</p>

<p align="center">
  <a href="https://cklxx.github.io/arle/"><img src="https://img.shields.io/badge/website-cklxx.github.io%2Farle-D97757?style=flat-square" alt="Website"></a>
  <a href="https://github.com/cklxx/arle/actions"><img src="https://github.com/cklxx/arle/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://github.com/cklxx/arle/releases"><img src="https://img.shields.io/github/v/release/cklxx/arle?include_prereleases" alt="Release"></a>
</p>

<p align="center">
  <a href="https://cklxx.github.io/arle/">Website</a> ·
  <a href="#-latest-updates">News</a> ·
  <a href="#-status-at-a-glance">Status</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="docs/http-api.md">API</a> ·
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

- **2026-04-23** — The `arle` front door now unifies `train pretrain|sft|grpo|multi-turn|eval` and `data download|convert` under one top-level Rust CLI, instead of requiring users to memorize separate train binaries. Entry-point consolidation notes: [`docs/experience/wins/2026-04-23-train-cli-unified-entrypoints.md`](docs/experience/wins/2026-04-23-train-cli-unified-entrypoints.md).
- **2026-04-22** — CUDA `Qwen3.5` now ships through a true packed multi-request paged-prefill path. Full-attention layers write directly into the paged pool, hybrid linear-attention layers use packed recurrent-state launches, and paged-prefill logits now land on the canonical sampling surface. See [`docs/plans/2026-04-22-sglang-gap-closure-execution.md`](docs/plans/2026-04-22-sglang-gap-closure-execution.md).
- **2026-04-20** — Metal DFlash long-prompt prefill fixed (`fast_forward_prefill`, commit `3bc8802`) and batched terminal `eval` deferred via `async_eval` (commit `d8cb2f4`). DFlash is now default-on for Qwen3.5 on Metal, validated across guidellm's 10-strategy sweep with 5400-token prompts — zero `WrongPhase` errors, 100% request success. Canonical usage: [`docs/resources/metal-dflash.md`](docs/resources/metal-dflash.md).

Full history: [CHANGELOG.md](CHANGELOG.md) · Next up: [ROADMAP.md](ROADMAP.md)

ARLE stands for **agent reinforcement learning engine**: one Rust workspace for
serving, agent execution, training, evaluation, and the toolchain around them.
The serving/runtime path is still CUDA-first, but the project identity is now
broader than a standalone inference binary.

In practice that shows up as three top-level surfaces:

- `infer` for OpenAI-compatible HTTP serving
- `arle` for the local agent runtime plus `train/*` and `data/*` workflows
- shared in-tree Rust runtime / model code underneath both, so serving and RL tooling do not drift apart

## 🚦 Status at a glance

Five axes, each answering one question. Authoritative matrix lives in
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
| Llama 3 / 4 | GQA | *planned* | *planned* |
| DeepSeek V3 / R1 | MLA | *planned* | *planned* |

### HTTP API — *what can clients call?*

| Endpoint | Status | Notes |
|----------|:------:|-------|
| `POST /v1/completions` · `POST /v1/chat/completions` · `GET /v1/models` | **Stable** | OpenAI-compatible core serving surface. Full contract: [`docs/http-api.md`](docs/http-api.md). |
| `POST /v1/responses` | **Beta** | Text/tool-call subset with both non-streaming and SSE forms. |
| `GET /metrics` · `GET /v1/stats` | **Stable** | Prometheus + human-readable ops surface. |

### Agent / Train / Eval — *what does ARLE itself run?*

| Surface | Status | What ships |
|---------|:------:|------------|
| `arle` local agent runtime | **Beta** | Tool calling by default, session save/load/export, `--doctor`, model auto-discovery, and KV-backed multi-turn reuse. |
| `train pretrain|sft|grpo|multi-turn|eval` | **Beta** | Unified CLI front door into the in-tree Rust train stack, exact resume, HF-style checkpoint directories, and the active Qwen3.5-family train/RL path. |
| `data download|convert` + operator DX | **Beta** | Dataset utilities, `train env`, standalone eval, and one coherent Rust front door instead of separate ad-hoc binaries. |

### Quantization — *how small does it get?*

| Format | Status | Available on |
|--------|:------:|--------------|
| FP8 / INT8 / TurboQuant KV | **Beta** | CUDA |
| GPTQ W4 · AWQ W4 | **Beta** | CUDA |
| Q4_K GGUF | **Beta** | CUDA; Metal (dense Qwen3.5 GGUF via load-time dequant) |
| MLX 4-bit | **Beta** | Metal (default for the canonical `start_metal_serve.sh` path) |

---

<!--
  Everything below is stable reference material: features, install, API,
  architecture, development workflow. It changes only on architectural or
  API-level shifts. Fresh project state lives in the two sections above.
-->

## Why ARLE (agent reinforcement learning engine)?

In agent workloads every turn pays a prefill tax: system prompt + conversation history + tool results must be re-processed. As context grows, **prefill dominates latency**.

ARLE (agent reinforcement learning engine) treats this as the core problem in both serving and agent/RL loops:

| Capability | What it does | Impact |
|---|---|---|
| **Multi-turn KV reuse** | Slot-sticky reuse keeps prior-turn KV hot for the next turn. CUDA also carries a radix-backed tiered-KV path (`T0 GPU -> T1 host pinned -> T2 local disk -> T3 cluster-shared backend surface`) for full-block reuse and staged readmission. | Only the new user message prefills each turn when the prefix stays reusable |
| **Paged KV pool** | Main CUDA KV formats use `page_size=16`, with direct GPU page attach and tail-page CoW on shared prefixes. | Predictable KV accounting, reusable full blocks, cheaper prefix sharing |
| **Transparent slower-tier spill / promote** | Cold blocks can spill from GPU to host pinned memory and local disk, then promote back before use. The in-tree cluster-shared path is currently a minimal shared-fs backend. | Longer contexts and cached-prefix reuse beyond pure GPU residency |
| **Shared-prefix CoW** | Shared full blocks stay immutable on the radix path; writes split only the active tail page. | Shared prefixes across concurrent requests do not multiply base KV memory |
| **Scheduler overlap** | CUDA scheduler overlaps decode launch/readback across iterations, sleeps on fetch waits instead of spinning, and uses an emit worker for streaming text decode and stop scanning. | Better CPU/GPU overlap and less scheduler-side overhead at concurrency |
| **Shared runtime authority** | `infer`, `arle`, and the in-tree train/eval jobs resolve models and reuse the same Rust runtime/model contracts. | Serving, local agent work, and RL tooling stay on one code path instead of drifting across separate stacks |

ARLE is therefore not "an inference engine plus some scripts". The inference
spine is what the wider agent RL loop builds on: shared Rust model/runtime
authority, train-side binaries in the same workspace, and a top-level CLI that
can act as local agent, training front-end, or evaluation entrypoint without
bouncing through a separate Python control plane.

Current benchmark closure work is focused on high-concurrency CUDA parity
(`c4/c8/c16`) against SGLang; treat the dated headline snapshots below as
historical records, not as the current public claim.

---

## Performance

Canonical benchmark source of truth is the dated snapshot log under
[`docs/experience/wins/`](docs/experience/wins/), produced via
[`scripts/bench_guidellm.sh`](scripts/bench_guidellm.sh).

The published numbers below are still mostly serving-side because the active
benchmark closure program is concentrated on CUDA parity. The project surface is
broader than that chart: the same runtime authority also backs the local
`arle` agent front door and the in-tree train/eval stack.

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

Two entrypoints are first-class:

### `arle` — local agent / train / eval front door

```bash
git clone https://github.com/cklxx/arle && cd arle
cargo build --release --features cli -p agent-infer --bin arle
./target/release/arle --model-path /path/to/Qwen3-4B --max-turns 10
./target/release/arle train env
./target/release/arle train eval --help
```

### `infer` — OpenAI-compatible serving

```bash
# Current published container image path
docker run --gpus all -v /path/to/Qwen3-4B:/model \
  ghcr.io/cklxx/agent-infer:latest --model-path /model --port 8000

# Or build the serving binary from source
cargo build -p infer --release
./target/release/infer --model-path /path/to/Qwen3-4B --port 8000
```

```bash
# Smoke test the HTTP surface
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

`arle` is the ARLE workspace front-end for agent execution, train/eval jobs,
and dataset utilities. `infer` is the dedicated OpenAI-compatible serving
binary.

**Prerequisites**: CUDA 12.x, the repo-pinned Rust toolchain from
[`rust-toolchain.toml`](rust-toolchain.toml) (currently `1.95.0`), and Python
3.10+ with `flashinfer-python` (build-time only). Zig `0.16.0` for
`crates/kv-native-sys` is bootstrapped by
[`scripts/setup_zig_toolchain.sh`](scripts/setup_zig_toolchain.sh) and
`./setup.sh`.

For a repo-managed workstation bootstrap, run [`./setup.sh`](setup.sh). For
contributor workflow and validation expectations, use
[CONTRIBUTING.md](CONTRIBUTING.md) as the source of truth.

Common repo hygiene commands:

```bash
make hygiene      # public docs / templates / local-link guardrails
make pre-push     # CI-aligned snapshot validation before git push
make check-metal  # Apple Silicon quick check
./setup.sh --check  # Linux/CUDA workstation check
```

## Documentation Map

- [README.md](README.md) — public project overview, install, CLI, architecture
- [docs/http-api.md](docs/http-api.md) — HTTP route contract and streaming behavior
- [docs/support-matrix.md](docs/support-matrix.md) — backend/model/quant support levels
- [docs/stability-policy.md](docs/stability-policy.md) — stability tiers and compatibility posture
- [CONTRIBUTING.md](CONTRIBUTING.md) — contributor setup, validation, release expectations
- [docs/index.md](docs/index.md) — maintainer-facing PARA index, plans, and experience logs

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
| Llama 3 / 4 | GQA | Planned |
| DeepSeek-V3 / R1 | MLA | Planned |

See [ROADMAP.md](ROADMAP.md) for the full plan.

---

## API

Serving is one surface of ARLE. The full HTTP API reference now lives in
[docs/http-api.md](docs/http-api.md). That document is the single place for
the route map, streaming behavior, boundary guarantees, auth and request-id
behavior, and current gaps.

The core generation surface is `POST /v1/completions`,
`POST /v1/chat/completions`, `POST /v1/responses`, and `GET /v1/models`.
Operational probes live at `GET /healthz`, `GET /readyz`, `GET /metrics`, and
`GET /v1/stats`. Session persistence lives under
`/v1/sessions/{session_id}/*`.

The same runtime authority also sits behind the local `arle` CLI and the
training/eval flows, so the HTTP surface is not a separate Python control
plane layered on top of a different engine.

SSE streaming ships on `/v1/completions`, `/v1/chat/completions`, and
`/v1/responses`. Requests that combine `stream=true` with `tools` are rejected
on chat completions and responses until the server can emit structured
tool-call deltas.

---

## ARLE CLI

Built-in ARLE runtime with tool calling:

```bash
./target/release/arle \
  --max-turns 10 --temperature 0
```

```bash
./target/release/arle train env
./target/release/arle train sft --help
./target/release/arle data convert --help
```

The root CLI binary is behind the `cli` feature. Without `--features cli`, `arle` is not built.

The CLI is agent-first: there is no separate chat mode and no `--tools`
switch. Tool calling is the default runtime, and the same top-level entrypoint
also fronts the train/eval/data subcommands that make the "agent reinforcement
learning engine" identity concrete in day-to-day DX.

Current package boundary behind the ARLE front door:

- `arle` -> thin binary wrapper
- `cli` -> REPL and slash commands
- `infer` -> `server_engine::LoadedInferenceEngine` backend loading and `hf_hub::resolve_model_source` for model auto-discovery
- `agent` -> conversation loop and tool-call recovery
- `tools` / `chat` -> shared tool definitions, execution helpers, and protocol types

If `--model-path` is omitted, the CLI first checks `ARLE_MODEL`, then falls
back to legacy `AGENT_INFER_MODEL`, then auto-detects a local model from
common directories and the local HuggingFace cache.

Use `--doctor` to print a self-check report for the current CLI build without
loading a model. It shows the compiled backend, detected hardware, TTY state,
HuggingFace cache root, model-resolution source, and curated model
recommendations. Add `--json` for machine-readable output in scripts and CI;
inspection JSON now includes a `schema_version`, top-level `status`, and
stable per-check / resolution codes so callers do not need to infer health
from prose. Add `--strict` when you want `--doctor` to exit non-zero on
warnings.

```bash
cargo run -p agent-infer --bin arle --release --no-default-features --features cpu,no-cuda,cli -- \
  --doctor

cargo run -p agent-infer --bin arle --release --no-default-features --features cli,no-cuda -- \
  --doctor --json

cargo run -p agent-infer --bin arle --release --no-default-features --features cli,no-cuda -- \
  --doctor --json --strict
```

Use `--list-models` for the lighter-weight discovery view when you only want
the resolved model source, supported local hub snapshots, and curated
recommendations without the full environment report. It also supports
`--json`.

```bash
cargo run -p agent-infer --bin arle --release --no-default-features --features cli,no-cuda -- \
  --list-models

cargo run -p agent-infer --bin arle --release --no-default-features --features cli,no-cuda -- \
  --list-models --json
```

Invalid `--max-turns`, `--max-tokens`, and `--temperature` values fail during
argument parsing instead of surfacing later at runtime.

Tools: `python` (execute Python snippets), `shell` (execute bash commands). The
CLI keeps tool protocol internal by default: raw tool-call / tool-result trace
output is hidden unless you opt into verbose logging with `RUST_LOG`, and
`/tools` remains the explicit way to inspect the built-in tool inventory. File
listing requests (`本地有哪些文件`, `list files`, etc.) return the shell output
directly; broader repo-inspection requests (`你看看本地仓库`, `look at the repo`)
return a deterministic shell overview of the repo root, top-level entries, and
`git status` instead of bouncing that tool output back through the model.
Completed turns are compacted back down to `user -> assistant`, so old raw tool
results do not keep polluting later prompts. The KV prefix cache therefore
reuses the compact prior-turn KV in place for every subsequent turn of a
session — only the new user message plus the current turn's temporary tool
context runs prefill. CUDA runtime reuse now sits on top of the same radix /
tiered-KV surface used by the HTTP scheduler; the main remaining
model-specific limit is that hybrid `Qwen3.5` does not yet support cross-slot
partial-prefix restore.
On macOS, tool execution uses `sandbox-exec` automatically when `nsjail` is unavailable; Linux keeps using `nsjail` when installed.

On Apple Silicon, build the same CLI against the Metal backend:

```bash
cargo run --release --no-default-features --features metal,no-cuda,cli -- \
  --model-path mlx-community/Qwen3-0.6B-4bit
```

The CLI keeps conversation history across turns, stores line history in
`~/.arle-history` (migrating legacy `~/.agent-infer-history` on first run),
and supports slash commands:

- `/help` for command help
- `/reset` or `/clear` to clear the current conversation
- `/tools` to inspect built-in tools
- `/model` and `/stats` to inspect the loaded runtime
- `/models [N]` to list local models and restart hints
- `/save <path>` and `/load <path>` to persist or resume a session as JSON
- `/export [path]` to dump the conversation as markdown

---

## Architecture

ARLE is one workspace, not just one binary. Workspace split:

- `arle` — thin binary wrapper
- `cli` — REPL / CLI flow
- `agent` — conversation state, tool-call recovery, agent turn loop
- `tools` / `chat` — tool execution helpers and protocol types
- `autograd` — from-scratch autograd and optimizer core for the train stack
- `train` — pretrain / SFT / GRPO / multi-turn / eval runtime and control plane
- `infer` — HTTP server, scheduler, runtime, backend implementations; owns the single `InferenceEngine` contract
- `cuda-kernels` — extracted CUDA kernel layer (csrc/, Triton AOT, Rust FFI). One-way dep: `infer → cuda-kernels`.
- `mlx-sys` — MLX C++ bridge for the Metal backend

See [docs/architecture.md](docs/architecture.md), [docs/codebase-map.md](docs/codebase-map.md), and [crates/README.md](crates/README.md)
for the current package boundaries.

The diagram below is the serving hot path. The agent/runtime and train/eval
surfaces sit beside it on the same shared Rust model/runtime authority; they
are not a separate Python layer wrapped around a different engine.

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

`arle` uses explicit stability tiers:

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

The day-to-day developer loop spans both sides of ARLE: the `infer` serving
surface and the `arle` agent/train/data front door.

```bash
make install-hooks                                      # Install repo-managed Git hooks (.githooks/pre-push)
make pre-push                                           # Run the CI-aligned local pre-push checks
./scripts/setup_zig_toolchain.sh                          # Validate/install Zig 0.16.0 for kv-native-sys
./scripts/check_kv_zig.sh                                 # Zig substrate local validation
cargo test --no-default-features --features no-cuda   # Unit tests (no GPU)
cargo clippy --workspace -- -D warnings                # Lint
cargo fmt --all -- --check                             # Format

# CPU backend smoke path (downloads runtime assets like config/tokenizer, not full weights)
cargo run -p agent-infer --bin arle --no-default-features --features cpu,no-cuda,cli -- \
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
`ARLE_SKIP_PRE_PUSH=1` only when you explicitly need to bypass the hook
(`AGENT_INFER_SKIP_PRE_PUSH=1` still works as a legacy alias).

---

## License

[MIT](LICENSE)
