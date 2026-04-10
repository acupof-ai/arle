# agent-infer — Claude Code Config

Assisting **ckl**. Agent contract below.

---

## Project Overview

Pure Rust LLM inference engine with dual GPU backends: **CUDA** (Linux/NVIDIA, via `cudarc` + FlashInfer + Triton AOT) and **Metal** (Apple Silicon, via `mlx-sys` bindgen bridge). No PyTorch, no frameworks. Models: Qwen3 (4B/8B), Qwen3.5-4B (hybrid linear + full attention), GLM4. Continuous batching scheduler, OpenAI-compatible `/v1/completions` + `/v1/chat/completions` API. FlashInfer drives CUDA prefill (HD128) and batched decode.

Primary language: **Rust** (inference engine + agent runtime). Secondary: **Python** (tooling, benchmarks, test scripts) + **CUDA C / Triton / Metal / C++** (kernels and bridges).

---

## Rules

### Execution Workflow

Non-trivial tasks follow phases. **Each phase has a clear exit condition.**

| Phase | Posture | Key rules |
|-------|---------|-----------|
| **Explore** | Cartographer | Trace callers/dependents before reading. Grep for existing implementations before writing new code. Trait change → list all implementors, mark blast radius. Uncertain → surface as open question, don't guess. |
| **Plan** | Architect (decisions, not options) | "How would this fail?" before "how should this work?". >5 files → question if simpler path exists. Irreversible decision → stop, flag, wait. |
| **Implement** | Contractor (build to spec) | Check prior art in `infer/src/` and `docs/` first. Outside plan → stop and update plan. Match adjacent code style. |
| **Verify** | Adversary | `cargo test --workspace`. Each diff line must serve the goal. New `unwrap()` → can this panic? GPU code → CUDA/Metal errors, OOM, stream/command-buffer sync? Async → cancel-safe? |
| **Reflect** | Retrospector | Bug >1 attempt → `docs/experience/errors/YYYY-MM-DD-slug.md`. Win → `docs/experience/wins/`. User corrected → write feedback memory. |

**Skip rules**: Trivial → Implement + Verify only. Exploration → Phase 1 only. "Just do it" → Implement + Verify.

### Editing Rules

- **Preserve by default**: NEVER delete content not explicitly asked to change.
- **Diff before delete**: Show what's being removed, get confirmation.
- **Approach-first**: Changes >3 files or architectural decisions → outline approach and wait for approval.

### Behavior Rules

- **Self-correction**: On ANY user correction → codify a preventive feedback memory before resuming.
- **GPU/CPU separation**: Mark GPU-only code with `// GPU required` comments. CUDA/Triton/Metal kernel stubs use `todo!("GPU required: ...")`.
- **Backend isolation**: CUDA-only code gated behind `#[cfg(feature = "cuda")]`; Metal-only behind `#[cfg(feature = "metal")]`. Never `cfg`-leak backend types into cross-backend modules — route through `backend.rs`.
- **Opportunistic cleanup**: Spot something inelegant → fix in separate commit.

### Benchmark Rules

- **Snapshot before & after** in `docs/experience/wins/YYYY-MM-DD-bench-<label>.md`.
- **Never overwrite** old snapshots — immutable history.
- **Standard tool**: `scripts/bench_throughput_sweep.py` with `--label`.
- **Include environment**: GPU model (e.g. A100 / M3 Max), CUDA or Metal version, model name, num_slots, non-default flags, feature set.
- **Raw data**: Full output table, not summaries. After-snapshot references before-snapshot with delta on key metrics.

### Git Conventions

Commitizen format: `<type>(<scope>): <subject>`. Scope examples: `metal`, `cuda`, `scheduler`, `qwen3`, `glm4`, `http`.

### Code Conventions

Module files use the flat layout (`src/ops.rs` + `src/ops/`) — no `mod.rs`. Models follow the same pattern: `model/qwen3.rs` + `model/qwen3/`.

### GPU Kernel Optimization — Six Principles

Every kernel in `csrc/cuda/` (CUDA) and `csrc/metal/` (Metal) must be evaluated against these. Use `ncu` for CUDA and Xcode Metal GPU capture / MLX instruments for Metal. The principles are universal; terminology translates (CUDA warp ↔ Metal SIMD-group, shared memory ↔ threadgroup memory, SM ↔ GPU core).

**1. Global Memory Coalescing**
- Warp of 32 threads → hardware groups addresses by 128B cache line → 1 transaction per line.
- Optimal: consecutive threads access consecutive addresses (stride = `sizeof(elem)`, aligned).
- Anti-pattern: `A[tid * N]` → stride N → 32 transactions instead of 1. Fix: transpose layout or tile.
- Verify: substitute `tid = 0..31` into address expression; adjacent difference should = element size.
- ncu metric: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg` / requests. Ideal = 4 (128B/32B).

**2. Shared Memory Bank Conflicts**
- 32 banks, 4B each. Bank = `(byte_addr / 4) % 32`.
- Same address → broadcast (free). Different addresses, same bank → serial (N-way conflict = N cycles).
- Classic trigger: `smem[32][32]` column access → all hit bank 0. Fix: pad to `smem[32][33]`.
- Distinct from coalescing: coalescing saves HBM bandwidth; bank-conflict-free saves smem latency.

**3. Occupancy**
- Active warps / max warps per SM. Limited by: threads, registers, shared memory, block count.
- Sweet spot: ≥50%. Below 25% → scheduler starves → compute units idle.
- Register pressure: >64 regs/thread → spill to local memory (HBM, ~400 cycles). Target 32-64.
- Trade-off: more smem per block → fewer blocks → lower occupancy. Balance tile size vs. occupancy.

**4. Tiling & Data Reuse**
- HBM load costs ~400 cycles. Shared memory costs ~5 cycles. Load once, reuse N times.
- GEMM: tile A and B into smem, each element reused by TILE threads → N/TILE HBM loads.
- Applies to attention (Q×K reuse), convolution, any overlapping computation.
- Larger tile = better reuse but more smem → occupancy drops. Profile to find sweet spot.

**5. Warp Divergence**
- Warp executes one instruction at a time. Branch → serialize both paths (masked execution).
- `if (tid % 2)` → 2× slowdown. `if (tid / 32 % 2)` → zero cost (different warps, not divergent).
- Fix: align branches to warp boundaries. Move divergent code outside inner loops.
- Boundary checks `if (idx < N)` → only last warp affected → usually negligible.

**6. Launch Config & Tail Effect**
- Grid must be >> SM_count × blocks_per_SM, otherwise last wave underutilizes.
- 900 blocks on 108 SMs (8 blocks/SM = 864 capacity): last wave = 36 blocks → 5% utilization.
- Fix: align grid to SM capacity, or use persistent kernels with `atomicAdd` work-stealing.
- Block size: 256 is default sweet spot. 128 for register-heavy kernels. 512+ only with profiling data.

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

**Always use `--release`** — debug builds are extremely slow for GPU.

```bash
# CUDA build (Linux, default feature set = ["cuda"])
CUDA_HOME=/usr/local/cuda cargo build --release

# Metal build (Apple Silicon) — disables CUDA
cargo build --release --no-default-features --features metal

# No-GPU build — compiles on CI/macOS without any GPU backend (ops panic at runtime)
cargo build --release --no-default-features --features no-cuda

# Run inference server (CUDA)
cargo run -p infer --release -- --model-path models/Qwen3.5-4B

# Lint + format
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

**Key env vars:**
- `PEGAINFER_CUDA_SM` — CUDA SM target override (e.g. `120` or `120,80`)
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

# Metal test suite (Apple Silicon)
cargo test --release --no-default-features --features metal

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
├── infer/            ← Inference engine (Rust library + GPU kernels)
│   ├── src/
│   │   ├── model/              ← Model impls: qwen3/, qwen35/, glm4/
│   │   ├── ops/                ← Tensor ops (attention, linear, norm, …)
│   │   ├── scheduler/          ← CUDA multi-request continuous batching
│   │   ├── metal_backend/      ← Metal backend module (Apple Silicon)
│   │   ├── metal_*.rs          ← Metal scheduler, KV pool, prefix cache, GDR
│   │   ├── backend.rs          ← Backend abstraction (CUDA vs. Metal)
│   │   ├── backend_runtime.rs  ← Runtime dispatch over backends
│   │   ├── model_registry.rs   ← Model ID → builder map
│   │   ├── sampler.rs          ← Sampling strategies
│   │   ├── http_server/        ← OpenAI-compatible HTTP API
│   │   └── server_engine.rs    ← Single-request inference façade
│   ├── csrc/cuda/              ← CUDA C kernels
│   ├── csrc/metal/             ← Metal .metal shaders
│   ├── mlx-sys/                ← Bindgen bridge to MLX C++ API
│   └── tools/triton/           ← Triton Python kernels (AOT compiled)
└── scripts/          ← Benchmark + utility scripts
```

### Key abstractions

- **`ModelForward` trait** (`infer/src/model.rs`) — `forward(&self, tokens, state)`; `tokens.len() > 1` → prefill, `== 1` → decode. Weights are `&self` (immutable), per-request state in associated `State` type.
- **`Backend` trait** (`infer/src/backend.rs`) — abstracts CUDA vs. Metal execution; selected at compile time via `--features cuda|metal`. `backend_runtime.rs` dispatches at runtime.
- **`Scheduler`** (`infer/src/scheduler/`) — CUDA multi-request continuous batching. Decode-priority, chunked prefill (4096 tok, 64 when decode active), prefix-aware slot assignment. `--num-slots N` controls concurrency (default 4). CUDA Graph warmup for batch sizes 1–32.
- **`MetalScheduler`** (`infer/src/metal_scheduler.rs`) — Metal counterpart with its own KV pool (`metal_kv_pool.rs`) and prefix cache (`metal_prefix_cache.rs`).
- **`SchedulerHandle`** — `Clone + Send` handle for submitting requests from HTTP handlers.
- **`GenericServerEngine`** (`server_engine.rs`) — single-request engine for REPL/agent CLI.

### Model implementation pattern

Each model is a flat module: `infer/src/model/<name>.rs` + `infer/src/model/<name>/`. Typical files: `config.rs`, `weights.rs`, `forward.rs`, `prefill.rs`, `decode.rs`, `batch_decode.rs`, `decode_buffers.rs`. Hybrid models (qwen35) add `recurrent_state.rs`, `prefill_buffers.rs`, `single_token_buffers.rs`.

### GPU kernel integration

- **CUDA**: `infer/csrc/cuda/` (CUDA C) + `infer/tools/triton/` (Triton). FFI in `infer/src/ffi.rs`. `build.rs` compiles CUDA C (auto-detects SM), runs Triton AOT, links FlashInfer.
- **Metal**: `infer/csrc/metal/` (.metal shaders) + `infer/mlx-sys/` (bindgen bridge to MLX C++ API). Enabled via `--features metal`; `libc` pulled in for `getrusage`-based RSS reporting.

### Key references

[ModelForward trait](infer/src/model.rs) · [Scheduler](infer/src/scheduler/) · [KV cache](infer/src/model/kv_cache.rs) · [Attention ops](infer/src/ops/attention.rs) · [HTTP server](infer/src/http_server.rs) · [Backend trait](infer/src/backend.rs) · [Metal backend](infer/src/metal_backend.rs) · [Roadmap](ROADMAP.md)

---

## Documentation Workflow (PARA)

```
docs/
├── index.md           # Document index
├── projects/          # Time-bound efforts with clear deliverables
├── plans/             # Design plans for in-flight work
├── experience/        # errors/ + wins/ + reviews/ — postmortems & playbooks
├── reviews/           # Standalone code/architecture reviews
├── resources/         # References with potential future value
└── archives/          # Inactive items
```

- Docs cover what `--help` and code can't: pitfalls, diagnostic paths, decision context.
- Refactor over append. Every document points to a next step.
- At session start, read `index.md` and load relevant docs. At session end, update modified docs and `index.md`.
