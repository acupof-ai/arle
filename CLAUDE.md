# agent-infer — Agent Contract

Assisting **ckl**. **Project-specific** rules only; generic Rust/CUDA/Metal/git
knowledge is intentionally absent. Load the relevant module `AGENTS.md`
(§Module Guides) before editing inside that module.

---

## Project shape

Pure-Rust LLM inference engine; no PyTorch, no Python on the hot path. Two
backends plug into one contract (`server_engine::InferenceEngine`): the CUDA
continuous-batching scheduler (Linux/NVIDIA, `cudarc` + FlashInfer + Triton AOT)
and the Metal scheduler runtime (Apple Silicon, `crates/mlx-sys` C++ bridge —
continuous batching with same-length packed decode; variable-length is
scaffolded but gated on a pending RoPE fix, see
[`infer/src/backend/metal/AGENTS.md`](infer/src/backend/metal/AGENTS.md) §7 and
[`docs/experience/errors/2026-04-16-metal-varlen-rope-blocker.md`](docs/experience/errors/2026-04-16-metal-varlen-rope-blocker.md)).
Models: Qwen3 (4B/8B), Qwen3.5-4B (hybrid linear + full attention), GLM4.
FlashInfer drives CUDA prefill HD128 and batched decode HD128+HD256.
Tests compare against JSON baselines in `infer/test_data/` — regenerate
after any change affecting numerical output.

**Workspace (post 2026-04-15 Route-A):**

```
agent-infer/
├── src/                       ← thin infer-cli::run() binary
├── infer/                     ← runtime crate (scheduler/model/ops/backends/HTTP)
├── crates/
│   ├── infer-cuda-kernels/    ← csrc/{attention,gemm,kv,quant,misc}/, tools/triton/, ffi/
│   ├── mlx-sys/               ← MLX + C++ bridge (cmake + cc)
│   ├── infer-agent/chat/cli/tools
└── docs/                      ← projects/ plans/ experience/ reviews/ resources/
```

CUDA kernels live at `crates/infer-cuda-kernels/csrc/`, **not** `infer/csrc/`
(common mistake — extracted 2026-04-15).

---

## Rules

### Execution phases (non-trivial tasks)

| Phase | Exit condition |
|-------|----------------|
| **Explore** (trace callers, grep prior art, list trait implementors) | You can name every file you will touch. |
| **Plan** (ask "how would this fail?" first; >5 files or irreversible → stop + flag) | Written approach the user accepted. |
| **Implement** (check prior art in `infer/src/` + `docs/`; outside plan → update plan) | Diff compiles under the relevant feature set. |
| **Verify** (`cargo test --workspace`; justify every new `unwrap()`/alloc/async path) | Tests green, `cargo clippy -- -D warnings` clean. |
| **Reflect** (bug >1 attempt → `docs/experience/errors/`; correction → feedback memory) | Experience entry committed. |

Skip rules: trivial → Implement + Verify; exploration questions → Explore only.

### Editing

- **Preserve by default.** Never delete content not explicitly in scope.
- **Approach-first for >3 files or architectural decisions** — outline and wait.
- **No half-states** (`feedback_no_half_states.md`): finish a refactor unit or
  revert it, never leave parallel old+new paths in the tree.

### Backend isolation (CRITICAL)

- `#[cfg(feature = "cuda")]` / `#[cfg(feature = "metal")]` gating; **never
  `cfg`-leak backend types into cross-backend modules** — route through
  `backend.rs` / `server_engine.rs`.
- CUDA stubs on non-CUDA targets: `todo!("GPU required: ...")`.
- Pre-push type check on Mac without nvcc:
  `cargo check -p infer --no-default-features --features cuda,no-cuda`.

### Delegation to Codex

Claude = **direction**; Codex = **execution**. Reach via `codex:codex-rescue`
(Agent `subagent_type: "codex:codex-rescue"`) or `mcp__openmax__execute_with_codex`.

| Area | Owner |
|------|-------|
| Docs, planning, architecture, roadmaps | Claude |
| Code execution (implement/refactor/tests) | **Codex** |
| Code review of non-trivial diffs | **Codex** |
| Stuck-problem rescue | **Codex** (after 2 failed attempts) |

- **Task-execution bias:** when a task is "write/change code", draft a brief
  (files, constraints, acceptance criteria) and delegate. Claude integrates
  and verifies — Claude does not hand-write substantial diffs.
- **2-strike rule:** two good-faith failed attempts → hand off. Brief must
  list what was tried, what was observed, why each attempt failed, so Codex
  picks a different angle.
- **Claude always owns:** planning docs, experience entries, roadmap edits,
  user-facing explanations, final integration after Codex reports back.

### Benchmarks

- Snapshot to `docs/experience/wins/YYYY-MM-DD-bench-guidellm-<label>.md`
  using the [`TEMPLATE-bench-guidellm.md`](docs/experience/wins/TEMPLATE-bench-guidellm.md)
  skeleton. **Never overwrite**; after-snapshots cite before-snapshots with deltas.
- **Canonical tool: `scripts/bench_guidellm.sh <label>`** — thin wrapper around
  [`vllm-project/guidellm`](https://github.com/vllm-project/guidellm) (vLLM
  official, LLM-native TTFT/ITL/tok-s metrics, sweep profile, HTML report).
  Canonical params are locked in
  [`docs/plans/guidellm-integration.md`](docs/plans/guidellm-integration.md) §3;
  changing them is a deliberate commit, not a flag flip.
- `scripts/bench_throughput_sweep.py` is **deprecated** — kept only so
  historical wins remain reproducible. New wins MUST use the guidellm wrapper.
- Include: GPU model, CUDA/Metal version, model, num_slots, non-default flags,
  feature set. Raw output table, not summaries.
- Install the Python dep once: `pip install -e .[bench]` (guidellm ships in
  the `bench` extra).

### Git

- Commitizen: `<type>(<scope>): <subject>`. Scopes: `metal`, `cuda`,
  `scheduler`, `qwen3`, `qwen35`, `glm4`, `http`, `kv-tier`, `docs`.
- Commit directly to `main` (no feature branches — `feedback_commit_to_main.md`).
- After `git mv` + batch Edits, re-check `git status` and re-stage by path —
  the fmt hook de-stages renames (`feedback_git_mv_with_fmt_hook.md`).

### Code conventions

- **Flat module layout, no `mod.rs`.** `src/ops.rs` declares `#[path = "ops/attention.rs"] mod attention;`
  siblings; models follow `model/qwen3.rs` + `model/qwen3/`.
- Weights `&self` (immutable, pool-shared); per-request mutable state in `State`
  associated types.

### GPU kernel work

Touching `crates/infer-cuda-kernels/csrc/` or `crates/mlx-sys/src/` hot paths?
Evaluate against the project-specific heat map in
[`docs/reviews/2026-04-14-cuda-kernel-six-principles-review.md`](docs/reviews/2026-04-14-cuda-kernel-six-principles-review.md)
— that's where the audited priorities live. Measure with `ncu` (CUDA) or
Xcode Metal capture / MLX instruments (Metal).

---

## Memory

- **Always-load:** auto-memory index + latest 3 of `docs/experience/errors/`
  and `docs/experience/wins/`.
- **On-demand:** `docs/plans/`, `docs/projects/`, `docs/research/`, full
  experience entries, `ROADMAP.md`.
- **User correction → write preventive feedback memory before resuming.**

Experience entry skeletons:
```
errors/YYYY-MM-DD-slug.md: # Title  ## Context  ## Root Cause  ## Fix  ## Rule
wins/YYYY-MM-DD-slug.md  : # Title  ## Context  ## What Worked  ## Rule
```

---

## Build & run

Always `--release` — debug GPU builds are unusably slow.

```bash
CUDA_HOME=/usr/local/cuda cargo build --release                              # CUDA
cargo build --release --no-default-features --features metal                 # Metal
cargo build --release --no-default-features --features no-cuda               # no-GPU
cargo check -p infer --no-default-features --features cuda,no-cuda           # Mac CUDA-Rust typecheck

cargo test --release                                   # ~9s, CPU-only
cargo test --release --test e2e                        # GPU + weights
cargo test --release --test e2e_qwen35
cargo test --release --no-default-features --features metal
```

Env vars: `PEGAINFER_CUDA_SM` (SM override), `PEGAINFER_TRITON_PYTHON`
(Triton AOT Python), `PEGAINFER_TEST_MODEL_PATH` (default `models/Qwen3-4B`).
Full list: [`docs/environment.md`](docs/environment.md).

---

## Module Guides

Load the relevant `AGENTS.md` **before** editing inside a module.

| Path | Guide |
|------|-------|
| `infer/src/backend/` | [AGENTS.md](infer/src/backend/AGENTS.md) — backend trait, dispatch, cfg discipline |
| `infer/src/backend/metal/` | [AGENTS.md](infer/src/backend/metal/AGENTS.md) — MLX bridge, unified memory, scheduler runtime + varlen scaffolding |
| `infer/src/scheduler/` | [AGENTS.md](infer/src/scheduler/AGENTS.md) — continuous batching, prefix cache, slot lifecycle |
| `infer/src/model/` | [AGENTS.md](infer/src/model/AGENTS.md) — ModelForward, state/weights split, hybrid models |
| `infer/src/ops/` | [AGENTS.md](infer/src/ops/AGENTS.md) — visibility policy, `_into` variants, batched conventions |
| `infer/src/kv_tier/` | [AGENTS.md](infer/src/kv_tier/AGENTS.md) — tier model, RadixCache invariant, MR stability |
| `infer/src/http_server/` | [AGENTS.md](infer/src/http_server/AGENTS.md) — OpenAI v1 compat, `session_id`, streaming |
| `crates/infer-cuda-kernels/` | [AGENTS.md](crates/infer-cuda-kernels/AGENTS.md) — prelude discipline, csrc layout, Triton AOT |
| `crates/mlx-sys/` | [AGENTS.md](crates/mlx-sys/AGENTS.md) — single Metal bridge, cmake+cc build, no repo `.metal` |

---

## Core docs (on-demand)

- [`docs/index.md`](docs/index.md) — PARA index; always start a session here.
- [`docs/codebase-map.md`](docs/codebase-map.md) — execution paths + where to start reading.
- [`docs/architecture.md`](docs/architecture.md) — workspace topology + Option-A→B kernel-crate extraction story.
- [`docs/plans/cuda-kernel-crate-extraction.md`](docs/plans/cuda-kernel-crate-extraction.md) — final `infer-cuda-kernels` extraction blueprint (trip wires + acceptance).
- [`docs/support-matrix.md`](docs/support-matrix.md) — backend / model / quant support levels.
