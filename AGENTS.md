# ARLE — Agent Contract

Assisting **ckl**. **Project-specific** rules only; generic Rust/CUDA/Metal/git
knowledge is intentionally absent. Load the relevant module `AGENTS.md`
(§Module Guides) before editing inside that module.

---

## Project shape

`ARLE` is a Rust-native inference runtime with integrated local
agent/train/self-evolution workflows. The runtime remains primary:

- `infer` owns serving/runtime truth.
- `arle` is the runtime-led CLI front door for local agent, train, eval, and
  data workflows.
- `train` extends the same runtime/model authority; it is not a second equal
  product line with its own independent truth surface.

No PyTorch and no Python on the hot path. Two backends plug into one contract
(`server_engine::InferenceEngine`): the CUDA continuous-batching scheduler
(Linux/NVIDIA, `cudarc` + FlashInfer + Triton AOT) and the Metal scheduler
runtime (Apple Silicon, `crates/mlx-sys` C++ bridge — continuous batching with
variable-length packed decode via mlx-lm `BatchKVCache` pattern: left-padding +
additive mask + per-row RoPE offsets, see
[`infer/src/backend/metal/AGENTS.md`](infer/src/backend/metal/AGENTS.md) §7).
Models: Qwen3 and Qwen3.5-family. FlashInfer drives CUDA prefill HD128 and
batched decode HD128+HD256. Tests compare against JSON baselines in
`infer/test_data/` — regenerate after any change affecting numerical output.

**Workspace (current):**

```
ARLE/
├── src/                       ← thin `arle` binary
├── infer/                     ← primary runtime crate (scheduler/model/ops/backends/HTTP)
├── crates/
│   ├── agent/chat/cli/tools   ← runtime-facing control-plane crates
│   ├── autograd/              ← from-scratch autograd + optimizer core
│   ├── cuda-kernels/          ← csrc/{attention,gemm,kv,quant,misc}/, tools/triton/, ffi/
│   ├── kv-native-sys/         ← local persistence substrate for KV tier transports
│   ├── mlx-sys/               ← MLX + C++ bridge (cmake + cc)
│   ├── qwen3-spec/            ← shared Qwen3 config + tensor-name contract
│   ├── qwen35-spec/           ← shared Qwen3.5 config + tensor-name contract
│   └── train/                 ← train-side control plane + runtime-integrated RL stack
└── docs/                      ← projects/ plans/ experience/ reviews/ resources/
```

CUDA kernels live at `crates/cuda-kernels/csrc/`, **not** `infer/csrc/`
(common mistake — extracted 2026-04-15).

Workspace topology source of truth: [`docs/codebase-map.md`](docs/codebase-map.md).

---

## Rules

### Execution phases (non-trivial tasks)

| Phase | Exit condition |
|-------|----------------|
| **Explore** (trace callers, grep prior art, list trait implementors) | You can name every file you will touch. |
| **Plan** (ask "how would this fail?" first; >5 files or irreversible → stop + flag) | Written approach the user accepted. |
| **Implement** (check prior art in `infer/src/` + `docs/`; outside plan → update plan) | Diff compiles under the relevant feature set. |
| **Verify** (`cargo test --workspace`; justify every new `unwrap()`/alloc/async path; **bench entry per §Benchmarks** if diff is in-scope) | Tests green, `cargo clippy -- -D warnings` clean, **wins/ entry committed (or stub with `pending-remote`)**. |
| **Reflect** (bug >1 attempt → `docs/experience/errors/`; correction → feedback memory) | Experience entry committed. |

Skip rules: trivial → Implement + Verify; exploration questions → Explore only.

### Editing

- **Preserve by default.** Never delete content not explicitly in scope.
- **Keep code simple and uniform.** Prefer deletion-style refactors:
  remove obsolete paths, collapse duplicate helpers/branches, and converge on
  one canonical flow instead of layering adapters.
- **`AGENTS.md` is canonical.** If a sibling `CLAUDE.md` exists, keep both
  files as full rule documents and keep their contents aligned; do not
  collapse one into a thin pointer.
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

### Delegation (general-purpose subagents execute, Codex reviews, parallel by default)

Claude = **direction + integration**. Execution runs through **`general-purpose`
subagents** (Agent tool). Research/mapping runs through **`Explore`**; large
cross-cutting plans through **`Plan`**. Review runs through **`codex review`
at the Bash tool** — a shell command, not a subagent.

**DO NOT use `codex:codex-rescue` or `mcp__openmax__execute_with_codex` for
execution** — both hang ("codex 会卡死", observed 2026-04-19). See
`memory/feedback_codex_subagent_hangs.md`. The review-via-Bash path is
unaffected.

Reserve direct hand-written diffs for edits ≤ ~3 files / trivial mechanical
changes.

| Area | Owner |
|------|-------|
| Docs, planning, architecture, roadmaps | Claude |
| Code execution (implement/refactor/tests) | **`general-purpose` subagent** (delegate via Agent tool) |
| Broad codebase exploration / scope mapping | **`Explore` subagent** |
| Implementation planning spanning >5 files | **`Plan` subagent** |
| Code review of non-trivial diffs | **Claude runs `codex review` at Bash** |
| Stuck-problem rescue (2-strike hand-off) | **`general-purpose` with full context** |

- **Parallelize by default.** When multiple delegated tasks are independent
  (different files, different layers, research + execution), fire them in a
  **single message with multiple Agent tool uses** so they run concurrently.
  Serial delegation is reserved for genuinely data-dependent steps.
- **Execution bias:** when a task is "write/change code", draft a brief
  (files, constraints, acceptance criteria) and delegate to a `general-purpose`
  subagent. Claude integrates and verifies — Claude does not hand-write
  substantial diffs.
- **Code review is Claude-driven via Codex CLI at Bash:** invoke
  `codex review --uncommitted` (or `--commit <sha>` / `--base <branch>`)
  directly from the Bash tool and relay the findings. This is a shell
  command — unlike the Codex subagent path, it does not hang.
- **2-strike rule:** two good-faith failed subagent attempts → either
  hand-write the diff yourself (if small) or re-brief a fresh `general-purpose`
  agent with explicit notes on what the prior attempts tried and why they
  failed.
- **Claude always owns:** planning docs, experience entries, roadmap edits,
  user-facing explanations, final integration after subagents report back,
  and the `codex review` pass before commit on non-trivial diffs.

### Benchmarks

- **Spec — always read first:**
  [`docs/bench-and-trace-spec.md`](docs/bench-and-trace-spec.md) — mandatory
  report sections (Goal · Hypothesis · Params · Env · Results · Problems ·
  Learnings), goal taxonomy, watch-list during runs, and **auto-iteration
  rules** (§7: when to loop, when to stop, information-volume triggers).
  Applies to both benchmarks and traces.
- **MANDATORY — every runtime change produces a bench entry.** A change is
  not "done" until a dated entry lands under `docs/experience/wins/` (or
  `errors/` if a regression was found). This is the Verify phase exit
  condition for any diff that could move numbers. No bench entry → not
  shipped.
  - **In scope** (bench required): anything under `infer/src/`,
    `crates/cuda-kernels/csrc/`, `crates/mlx-sys/src/`, `src/`, any
    `scripts/bench_*.{sh,py}` parameter change, feature-flag default flips,
    scheduler/kernel/ops/model/backend edits, dependency bumps that touch
    the hot path.
  - **Exempt** (no bench): pure docs (`docs/`, `*.md`, `AGENTS.md`),
    comment-only diffs, `CLAUDE.md` / memory, dev-only tooling,
    gitignored-output paths. When exempt, state so in the commit body.
  - **Regression-check minimum.** Even a "small" change: one
    `scripts/bench_guidellm.sh` run against the most recent baseline for
    the affected backend+model, with a Δ% row. Full sweep only when the
    change is an optimization or architectural.
  - **If the bench can't run locally** (e.g. CUDA change on a Mac),
    the commit body MUST cite the remote-machine ticket or plan entry
    that will execute it, and the entry is opened as a stub under
    `wins/` with status `pending-remote`. No silent skips.
  - **Auto-iterate** per spec §7 until stopping rules hold; then cross-link
    the wins entry from the project/plan that commissioned the change.
- Snapshot to `docs/experience/wins/YYYY-MM-DD-bench-guidellm-<label>.md`
  using the [`TEMPLATE-bench-guidellm.md`](docs/experience/wins/TEMPLATE-bench-guidellm.md)
  skeleton. **Never overwrite**; after-snapshots cite before-snapshots with deltas.
- **Canonical tool: `scripts/bench_guidellm.sh <label>`** — thin wrapper around
  [`vllm-project/guidellm`](https://github.com/vllm-project/guidellm) (vLLM
  official, LLM-native TTFT/ITL/tok-s metrics, sweep profile, HTML report).
  Canonical params are locked in
  [`docs/plans/guidellm-integration.md`](docs/plans/guidellm-integration.md) §3;
  changing them is a deliberate commit, not a flag flip.
- Include: GPU model, CUDA/Metal version, model, num_slots, non-default flags,
  feature set. Raw output table, not summaries.
- Install the Python dep once: `pip install -e .[bench]` (guidellm ships in
  the `bench` extra).

### Git

- Commitizen: `<type>(<scope>): <subject>`. Scopes: `metal`, `cuda`,
  `scheduler`, `qwen3`, `qwen35`, `http`, `kv-tier`, `docs`.
- Commit directly to `main` (no feature branches — `feedback_commit_to_main.md`).
- **Always commit and push from the current branch in the current workspace.**
  Do not create a separate worktree or alternate checkout to prepare or ship
  code changes.
- **Commit small tranches immediately.** Each small, self-contained change
  should land as its own commit. Run the relevant verification after that
  commit; if verification finds issues, fix them in a follow-up commit instead
  of folding multiple micro-changes into one opaque diff.
- **Never use `git stash` to move unrelated user changes out of the way.**
  Leave other people's dirty paths in place, work around them, and commit only
  your own files by explicit path.
- After `git mv` + batch Edits, re-check `git status` and re-stage by path —
  the fmt hook de-stages renames (`feedback_git_mv_with_fmt_hook.md`).

### Code conventions

- **Flat module layout, no `mod.rs`.** `src/ops.rs` declares `#[path = "ops/attention.rs"] mod attention;`
  siblings; models follow `model/qwen3.rs` + `model/qwen3/`.
- Weights `&self` (immutable, pool-shared); per-request mutable state in `State`
  associated types.

### GPU kernel work

Touching `crates/cuda-kernels/csrc/` or `crates/mlx-sys/src/` hot paths?
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

Env vars: `INFER_CUDA_SM` (SM override), `INFER_TRITON_PYTHON`
(Triton AOT Python), `INFER_TEST_MODEL_PATH` (default `models/Qwen3-4B`).
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
| `crates/cuda-kernels/` | [AGENTS.md](crates/cuda-kernels/AGENTS.md) — prelude discipline, csrc layout, Triton AOT |
| `crates/mlx-sys/` | [AGENTS.md](crates/mlx-sys/AGENTS.md) — single Metal bridge, cmake+cc build, no repo `.metal` |

---

## Core docs (on-demand)

- [`docs/index.md`](docs/index.md) — PARA index; always start a session here.
- [`docs/codebase-map.md`](docs/codebase-map.md) — execution paths + where to start reading.
- [`docs/architecture.md`](docs/architecture.md) — package boundaries, dependency direction, crate-split governance.
- [`docs/plans/cuda-kernel-crate-extraction.md`](docs/plans/cuda-kernel-crate-extraction.md) — final `cuda-kernels` extraction blueprint (trip wires + acceptance).
- [`docs/support-matrix.md`](docs/support-matrix.md) — backend / model / quant support levels.
