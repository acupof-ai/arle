# agent-infer — Claude Code Config

Assisting **ckl**. Agent contract below.

---

## Project Overview

Pure Rust + CUDA LLM inference engine. No PyTorch, no frameworks. Supports Qwen3 (4B/8B) and Qwen3.5-4B (hybrid linear + full attention) with continuous batching scheduler. OpenAI-compatible `/v1/completions` + `/v1/chat/completions` API. FlashInfer for both prefill (HD128) and batched decode.

Primary language: **Rust** (inference engine + agent runtime). Secondary: **Python** (tooling, benchmarks, test scripts) + **CUDA C / Triton** (kernels).

---

## Rules

### Execution Workflow

Non-trivial tasks follow phases. **Each phase has a clear exit condition.**

| Phase | Posture | Key rules |
|-------|---------|-----------|
| **Explore** | Cartographer | Trace callers/dependents before reading. Grep for existing implementations before writing new code. Trait change → list all implementors, mark blast radius. Uncertain → surface as open question, don't guess. |
| **Plan** | Architect (decisions, not options) | "How would this fail?" before "how should this work?". >5 files → question if simpler path exists. Irreversible decision → stop, flag, wait. |
| **Implement** | Contractor (build to spec) | Check prior art in `infer/src/` and `docs/` first. Outside plan → stop and update plan. Match adjacent code style. |
| **Verify** | Adversary | `cargo test --workspace`. Each diff line must serve the goal. New `unwrap()` → can this panic? GPU code → CUDA errors, OOM, stream sync? Async → cancel-safe? |
| **Reflect** | Retrospector | Bug >1 attempt → `docs/experience/errors/YYYY-MM-DD-slug.md`. Win → `docs/experience/wins/`. User corrected → write feedback memory. |

**Skip rules**: Trivial → Implement + Verify only. Exploration → Phase 1 only. "Just do it" → Implement + Verify.

### Editing Rules

- **Preserve by default**: NEVER delete content not explicitly asked to change.
- **Diff before delete**: Show what's being removed, get confirmation.
- **Approach-first**: Changes >3 files or architectural decisions → outline approach and wait for approval.

### Behavior Rules

- **Self-correction**: On ANY user correction → codify a preventive feedback memory before resuming.
- **GPU/CPU separation**: Mark GPU-only code with `// GPU required` comments. CUDA/Triton kernel stubs use `todo!("GPU required: ...")`.
- **Opportunistic cleanup**: Spot something inelegant → fix in separate commit.

### Benchmark Rules

- **Snapshot before & after** in `docs/experience/wins/YYYY-MM-DD-bench-<label>.md`.
- **Never overwrite** old snapshots — immutable history.
- **Standard tool**: `scripts/bench_throughput_sweep.py` with `--label`.
- **Include environment**: GPU model, CUDA version, model name, num_slots, non-default flags.
- **Raw data**: Full output table, not summaries. After-snapshot references before-snapshot with delta on key metrics.

### Git Conventions

Commitizen format: `<type>(<scope>): <subject>`. Never commit directly to `main`.

### Code Conventions

Module files use the flat layout (`src/ops.rs` + `src/ops/`) — no `mod.rs`.

---

## Memory

**Always-load**: auto memory + latest 3 from `docs/experience/errors/` and `docs/experience/wins/`.

**On-demand**: `docs/plans/`, full experience entries, `ROADMAP.md`.

### Experience Entry Templates

**Error** (`docs/experience/errors/YYYY-MM-DD-slug.md`):
```
# YYYY-MM-DD · Title
## Context
## Root Cause
## Fix
## Rule
```

**Win** (`docs/experience/wins/YYYY-MM-DD-slug.md`):
```
# YYYY-MM-DD · Title
## Context
## What Worked
## Rule
```

---

## Build & Run

**Always use `--release`** — debug builds are extremely slow for GPU/CUDA.

```bash
# Build (CPU-only)
cargo build --release

# Build with CUDA
CUDA_HOME=/usr/local/cuda cargo build --release

# Run inference server
cargo run -p infer --release -- --model-path models/Qwen3.5-4B

# Lint + format
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

**Key env vars:**
- `PEGAINFER_CUDA_SM` — GPU SM target override (e.g. `120` or `120,80`)
- `PEGAINFER_TRITON_PYTHON` — Python with Triton for AOT kernel generation
- `PEGAINFER_TEST_MODEL_PATH` — override test model path (default: `models/Qwen3-4B`)

### Tests

```bash
# Unit tests (~9s)
cargo test --release

# E2E greedy regression (requires GPU + model weights)
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e
cargo test --release --test e2e_qwen35

# Single test
cargo test -p infer --release -- <test_name>

# Python tests
pip install -e ".[dev]"
python -m pytest tests/ -v
```

E2E tests compare against JSON baselines in `infer/test_data/`. Regenerate baselines after any change that affects numerical output.

---

## Architecture

### Workspace layout

```
agent-infer/          ← top-level Cargo workspace
├── src/              ← Rust agent binary (ChatML, tool calling, REPL)
├── agent_infer/      ← Python agent package (async HTTP client mode)
├── infer/            ← Inference engine (Rust library + CUDA kernels)
│   ├── src/
│   │   ├── model/       ← Model implementations (Qwen3, Qwen35, ...)
│   │   ├── ops/         ← CUDA-backed tensor ops (attention, linear, norm...)
│   │   ├── scheduler/   ← Multi-request continuous batching scheduler
│   │   ├── sampler.rs   ← Sampling strategies
│   │   ├── http_server/ ← OpenAI-compatible HTTP API
│   │   └── server_engine.rs ← Single-request inference façade
│   ├── csrc/            ← CUDA C kernels
│   └── tools/triton/    ← Triton Python kernels (AOT compiled)
└── scripts/          ← Benchmark + utility scripts
```

### Key abstractions

- **`ModelForward` trait** (`infer/src/model.rs`) — `forward(&self, tokens, state)`; `tokens.len() > 1` → prefill, `== 1` → decode. Weights are `&self` (immutable), per-request state in associated `State` type.
- **`Scheduler`** (`infer/src/scheduler/`) — multi-request continuous batching. Decode-priority, chunked prefill (4096 tok, 64 when decode active), prefix-aware slot assignment. `--num-slots N` controls concurrency (default 4). CUDA Graph warmup for batch sizes 1–32.
- **`SchedulerHandle`** — `Clone + Send` handle for submitting requests from HTTP handlers.
- **`GenericServerEngine`** (`server_engine.rs`) — single-request engine for REPL/agent CLI.

### Model implementation pattern

Each model in `infer/src/model/<name>/`: `config.rs`, `weights.rs`, `decode_buffers.rs`, `prefill_buffers.rs`, `forward.rs`.

### CUDA kernel integration

Kernels: `infer/csrc/` (CUDA C) + `infer/tools/triton/` (Triton). FFI in `infer/src/ffi.rs`. `build.rs` compiles CUDA C (auto-detect SM), runs Triton AOT, links FlashInfer.

### Key references

[ModelForward trait](infer/src/model.rs) · [Scheduler](infer/src/scheduler/) · [KV cache](infer/src/model/kv_cache.rs) · [Attention ops](infer/src/ops/attention.rs) · [HTTP server](infer/src/http_server.rs) · [Roadmap](ROADMAP.md)

---

## Documentation Workflow (PARA)

```
docs/
├── index.md           # Document index
├── projects/          # Time-bound efforts with clear deliverables
├── areas/             # Ongoing responsibilities
├── resources/         # References with potential future value
└── archives/          # Inactive items
```

- Docs cover what `--help` and code can't: pitfalls, diagnostic paths, decision context.
- Refactor over append. Every document points to a next step.
- At session start, read `index.md` and load relevant docs. At session end, update modified docs and `index.md`.
