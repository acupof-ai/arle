# ARLE — Agent Contract

Assisting **ckl**. **Project-specific** rules only; generic Rust/CUDA/Metal/git
knowledge is intentionally absent. Load the relevant module `AGENTS.md`
(§Module Guides) before editing inside that module.

---

## §0 第一原则 — SOLID(求真务实,追求极致)

**所有事必须 SOLID。不够 SOLID 就不断深入,不断突破。** 不是建议,是 quality bar。

- **推断 ≠ SOLID**:source survey、code grep、文档分析、callgraph 推断 都是 *hypothesis*,
  不是 evidence。Evidence = 实测 nsys trace / bench 数字 / runtime log counter / 控制变量
  对照实验。没有 evidence 不下结论,只标 hypothesis。
- **混淆变量必须隔离**:一个实验同时改 N 个变量(buffer pool + scheduler clamp + KV format
  + graph capture)→ 任何结果都 **不能归因**。每次只改一个变量,或显式跑控制实验隔离 confounder。
- **Root cause 假设也要 license-or-kill**:license-or-kill 不只用在 fix 上,**root cause 推断**
  本身也要 cheap experiment 验证(nsys 占比 / log 计数器 / source 二次读 / 实验对照)。
  Root cause 错 → 所有 sub-experiment 全废。
- **80% SOLID 不够**:发现 gap 必须深入到 95%+,或显式声明 "deferred,接受不确定性",**禁止
  silent 放过**。
- **写完先自检**:每份 plan / wins / errors / brief / 推荐落地前,先问"SOLID 吗?gap 在哪?
  深入还是显式 deferred?"。不达标自我反思,继续深入。
- **Framing 多角度交叉**:同一数据用不同 framing(per-NVTX-window vs per-wall-clock,
  per-launch vs per-token,per-layer vs per-request)给出不同结论时,**wall-clock /
  per-request framing 是 ground truth**。Narrow window 占比 X% 不等于实际 wall-clock 影响 X%。
  License-or-kill 决策必须用 wall-clock framing,不用 narrow window framing 自欺。

实证 anchors:
- **M_pf-graph Phase 0 KILL** (2026-05-08):errors entry 80% SOLID 仍栽,3 个 gap —
  launch overhead 占比未 nsys 验证 / SGLang graph trigger 实计数未对照 / 4 变量同改未隔离
  → strategic conclusion 全废。
- **M_pf-graph v2 framing trap** (2026-05-08 EOD+19):nsys "55.7% of prefill window"
  看似 PASS,但 191ms / 60s trace = 6.4ms per prefill / 1995ms TTFT = **0.32% wall-clock**,
  远低于 10% kill 阈。**Lesson**:nsys "X% of NVTX window" 必须 cross-check "Y ms /
  per-request total" framing,取保守者作 license-or-kill 基准。

---

## Project shape

`ARLE` is a Rust-native inference runtime with integrated local
agent and **On-Policy Distillation (OPD)** workflows. The runtime
remains primary:

- `infer` owns serving/runtime truth.
- `arle` is the runtime-led CLI front door for local agent, OPD train,
  and eval workflows.
- `train` extends the same runtime/model authority via **OPD only**;
  it is not a second equal product line with its own independent
  truth surface. Scratch pretrain, SFT, GRPO, and multi-turn RL
  surfaces have been deleted (2026-05-18 pivot — see
  [`docs/projects/2026-05-18-opd-only-pivot.md`](docs/projects/2026-05-18-opd-only-pivot.md))
  because the industry baseline made pretrain unwinnable (322× gap)
  and SFT/GRPO/multi-turn duplicate mature OSS (vLLM+verl, TRL,
  axolotl). OPD is the one training axis where ARLE's runtime
  authority is structurally differentiating: it needs a strong
  inference path for the teacher and tight latency to score student
  rollouts — both already in `infer`.

No PyTorch and no Python on the hot path. Two backends plug into one contract
(`server_engine::InferenceEngine`): the CUDA continuous-batching scheduler
(Linux/NVIDIA, `cudarc` + TileLang AOT + native CUDA C) and the Metal scheduler
runtime (Apple Silicon, `crates/mlx-sys` C++ bridge — continuous batching with
variable-length packed decode via mlx-lm `BatchKVCache` pattern: left-padding +
additive mask + per-row RoPE offsets, see
[`infer/src/backend/metal/AGENTS.md`](infer/src/backend/metal/AGENTS.md) §7).
Models: Qwen3.5-family. TileLang drives CUDA paged prefill/decode
for BF16 attention; custom CUDA C handles quantized decode and supporting ops.
Tests compare against JSON baselines in
`infer/test_data/` — regenerate after any change affecting numerical output.

**Metal canonical model — globally unified (2026-05-07).** All Metal
backend development, benchmarking, and testing uses
`mlx-community/Qwen3.6-35B-A3B-4bit` (MoE, ~19 GB, cached at
`~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit`).

- **Why**: Qwen3.6 is the canonical Metal production target per
  [`README.md`](README.md) backend matrix and the
  [`ROADMAP.md`](ROADMAP.md) Next-Model priority queue. Benching against
  the production shape catches MoE-specific perf and correctness
  regressions that Qwen3.5-0.8B (dense) cannot surface.
- **Scope**: every Metal `metal_serve` invocation, `scripts/bench_*.sh`
  default, smoke test, and `docs/experience/wins`/`errors` entry on the
  Metal track must use Qwen3.6. CUDA-side benches keep their existing
  defaults.
- **Opt-out**: Qwen3.5-0.8B-MLX-4bit and friends remain in
  `models/` for unit tests that explicitly need a small model;
  set `INFER_TEST_MODEL_PATH=models/Qwen3.5-0.8B-MLX-4bit` and document
  the reason in the test/wins entry.
- **Bench-script invocation**: `./scripts/bench_*.sh <label> --model
  mlx-community/Qwen3.6-35B-A3B-4bit` (HF id; `metal_serve` resolves to
  the cached snapshot). For `metal_serve` directly: `--model-path
  mlx-community/Qwen3.6-35B-A3B-4bit`.
- **Auto-wired-limit** (default since
  [`2026-05-07-bench-qwen36-mle-perf.md`](docs/experience/wins/2026-05-07-bench-qwen36-mle-perf.md)):
  `metal_serve` auto-pins model weights via `mlx::set_wired_limit`
  when `--wired-limit-bytes` isn't passed. Computes
  (model dir size + 1 GiB headroom) and follows HF cache symlinks.
  Drops c=1 p99 from 86 ms → 15 ms on Qwen3.6 (−82%). Opt-out via
  `--wired-limit-bytes 0`.
- **MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER — not a default.**
  Qwen3.5-dense-only tune; on Qwen3.6 MoE benched wash-or-loss because 95% of
  step is `mx::async_eval` encoding ~600-1000 primitives — buffer cap doesn't
  help. Per-workload matched-A/B only.
  Refs: [baseline](docs/experience/wins/2026-05-07-bench-qwen36-baseline.md),
  [encode-bottleneck](docs/experience/wins/2026-05-07-bench-qwen36-encode-bottleneck.md).

**Workspace (current):**

```
ARLE/
├── src/                       ← thin `arle` binary
├── infer/                     ← primary runtime crate (scheduler/model/ops/backends/HTTP/distributed)
├── crates/
│   ├── agent/chat/cli/tools   ← runtime-facing control-plane crates
│   ├── autograd/              ← from-scratch autograd + optimizer + lr-schedule + AdamW codec
│   ├── cuda-kernels/          ← csrc/{attention,gemm,kv,quant,misc}/, tools/tilelang/, ffi/, collective.rs (NCCL)
│   ├── deepseek-spec/         ← DeepSeek V4 readiness scaffold (DS0 config + tensor names + Shard)
│   ├── kv-native-sys/         ← local persistence substrate for KV tier transports
│   ├── mlx-sys/               ← MLX + C++ bridge (cmake + cc), Qwen3.5 step / MoE / DFlash draft / Metal capture hook
│   ├── qwen3-spec/            ← Qwen3 config + tensor-parallel Shard enum (TP layout authority)
│   ├── qwen35-spec/           ← shared Qwen3.5 config + tensor-name contract
│   ├── train/                 ← train-side control plane + runtime-integrated RL stack
│   └── xgrammar-sys/          ← Rust wrapper over upstream mlc-ai/xgrammar matcher (grammar-constrained decode)
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

- **Parallel by default.** Independent delegated tasks → single message,
  multiple Agent calls. Serial only when data-dependent.
- **Code review invocation:** `codex review --uncommitted` (or `--commit <sha>` /
  `--base <branch>`) at Bash, run in background + tee to tmp file —
  non-blocking (`feedback_codex_review_async.md`).
- **2-strike rule:** two failed subagent attempts → hand-write the diff (if
  small) or re-brief a fresh `general-purpose` with notes on what prior
  attempts tried and why they failed.

### Execution hygiene (Claude and delegated agents alike)

- Surface known failure logs upfront so the same blocker isn't re-discovered.
- Pin SKU / shape / scope at exact granularity, not by fuzzy name — otherwise everything gets enabled then narrowed down.
- Before patching an upstream component, grep the raise point and lock the root cause first.
- When probing install / directory / env layout, enumerate candidate paths upfront, not fail-then-retry.
- PR branches start from `upstream/main`, never from a local WIP branch — defaults pick current HEAD, so state it explicitly.
- Verify a patched upstream lib in an isolated dir, never the existing dev install, to dodge editable / `.pth` finder hijacks.
- When an upstream patch crosses a size or cross-cutting-policy threshold, pause and ack before landing.
- Regression tests should mirror the failure mode with a minimal in-component kernel, not by importing caller code.

### Benchmarks

- **Spec — always read first:**
  [`docs/bench-and-trace-spec.md`](docs/bench-and-trace-spec.md) — mandatory
  report sections (Goal · Hypothesis · Params · Env · Results · Problems ·
  Learnings), goal taxonomy, watch-list during runs, and **auto-iteration
  rules** (§6: when to loop, when to stop, information-volume triggers),
  and **§7 hard-won protocol rules** (correctness gate, sweep≠fixed-c,
  duration adequacy, param-alignment via the §3.2 envelope log, server
  lifecycle hygiene). Internal info sources (§3: `/v1/stats` service trace,
  scheduling envelope, K6 OOM detector) are first-class report content.
  Applies to both benchmarks and traces.
- **MANDATORY — every runtime change produces a bench entry.** A diff isn't
  "done" until a dated entry lands under `docs/experience/wins/` (or
  `errors/` on regression). Verify-phase exit condition. No entry → not shipped.
  - **In scope:** `infer/src/`, `crates/cuda-kernels/csrc/`,
    `crates/mlx-sys/src/`, `src/`, `scripts/bench_*.{sh,py}` param changes,
    feature-flag default flips, hot-path dep bumps.
  - **Exempt:** docs / `AGENTS.md` / `CLAUDE.md` / memory / dev-only tooling
    / gitignored output. State so in the commit body.
  - **Minimum:** one `scripts/bench_guidellm.sh` run vs latest baseline for
    affected backend+model, with Δ% row. Full sweep only for optimization /
    architectural changes.
  - **Can't run locally** (e.g. CUDA on a Mac): commit body cites the remote
    ticket; stub the entry under `wins/` with `pending-remote`. No silent skips.
  - **Auto-iterate** per spec §7; cross-link wins back to the commissioning
    project/plan.
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

### Distilled lessons (cross-module, recurring ≥3 entries)

- **SLO verdict must come from the SLO workload, not a smoke shape.** A c=1 short-prompt
  nsys breakdown predicting "2× win" routinely flips on the production prompt length
  because the path's scaling curve is shape-specific
  (`errors/2026-05-27-dsv4-tp-allreduce-slo-prefill-kill.md`).
- **`plan_label=mixed` / "executes new path" is reachability evidence, not a license to land.**
  c-sweep must clear TTFT *and* ITL *and* output throughput before any default flip
  (`errors/2026-05-25-axis2-mixed-default-kill.md`, `errors/2026-05-26-qwen35-hybrid-mixed-kill.md`,
  `errors/2026-05-25-axis3-chunked-prefill-size-kill.md`).
- **Backend / quant / decoding default flips need multi-shape verification.** Single-shape ROI
  shows "what's possible"; ≥2 binding production shapes show "what's safe to default"
  (`wins/2026-05-08-prefill-cap-8-multi-shape-safe-default-flip.md`,
  `errors/2026-05-08-w4-c8-deadlock-confirms-workload-dependent.md`).
- **A/B must be same-binary, same-shell, same-prompt, two env flips, side-by-side.** Cross-day
  baseline-vs-treatment claims don't survive — intermediate commits drift backend / KV dtype
  / scheduler tuning (`wins/2026-05-27-dsv4-native-deepep-perf-ab.md`).
- **Smoke-output garbage is config-suspect first, code-suspect second.** When a new GPU forward
  path produces nonsense, A/B against the prod backend on the *same* config before staring at
  the new code; if prod is also broken, the serving config is the bug
  (`wins/2026-05-27-dsv4-native-deepep-pod-e2e.md`).
- **Launch-count source-survey is hypothesis, not evidence.** For tiny CUDA operators, a fused-kernel
  rewrite is only licensed by a paired component A/B (or nsys/CUDA-event profile) under the
  same sync framing the runtime uses (`errors/2026-05-12-fp8-kv-pair-quantize-fusion-no-license.md`,
  `errors/2026-05-21-arle-cuda-opd-swiglu-fused-kill.md`).
- **Capability/quality claims with magnitude < 5pp on small-n evals (≤200 samples) MUST run
  multi-seed (≥5) and report mean ± σ + Wilson 95% CI before the wins entry ships.** Picking
  "best ckpt across save-every-10" is a positively-biased estimator
  (`errors/2026-05-28-mmlu-cross-base-was-noise.md`).
- **Pod-side probe trust is conditional on git+symbol checks.** Before flipping a default based on
  pod output, verify the pod tree is a git repo at HEAD and `strings target/release/infer | grep <symbol>`
  shows the change actually landed — `target/release/infer` proves *some* tree was current
  *whenever it was last built*, not that the current source built it
  (`errors/2026-05-28-dsv4-flashmla-decode-parity-precond-fail.md`).
- **Decode greedy-token decode the actual generation when a metric looks catastrophic.** Three weeks
  of "FP8 KV is broken" investigation collapsed when one `eprintln!` of decoded tokens showed the
  metric was a test-framework artifact (`errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md`).
- **`scripts/dsv4_toolchain.sh` validates DSv4 build-flow before launch.** Native DeepEP / DeepGEMM
  consumers need env-checked source + compile-time prereqs; without the toolchain helper users
  get a stub binary that errors at runtime
  (`wins/2026-05-27-dsv4-native-deepep-run-guide.md`).

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

# KV precision parity audit (CUDA only) — BF16 reference vs INT8/FP8/TQ4
# trajectory match. Required after any change touching KV quant kernels,
# `infer/src/model/qwen3/{prefill,batch_decode}.rs`, the paged-pool gating,
# or `auto`-default selection.
cargo test --release -p infer --features cuda \
  --test kv_precision_parity -- --nocapture --test-threads=1
# Knobs: KV_PARITY_PROMPTS / KV_PARITY_MAX_TOKENS / KV_PARITY_MAX_SEQ_LEN /
# KV_PARITY_INCLUDE_TQ23=1. Report lands at target/kv-parity-<model>-<unix>.json.
```

Env vars: `TORCH_CUDA_ARCH_LIST` (SM override, PyTorch convention; alt `CMAKE_CUDA_ARCHITECTURES`),
`INFER_TILELANG_PYTHON` (TileLang AOT Python), `INFER_TEST_MODEL_PATH`
(default `models/Qwen3.5-4B`). Full list: [`docs/environment.md`](docs/environment.md).
SM tier policy: [`docs/plans/sm-coverage.md`](docs/plans/sm-coverage.md).

Disk hygiene: `cargo sweep --time 30` (weekly) prunes target/ artifacts
older than 30 days. Dev profile already keeps deps DWARF-free (see root
`Cargo.toml` `[profile.dev.package."*"] debug = false`).

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
| `crates/autograd/` | [AGENTS.md](crates/autograd/AGENTS.md) — training-tape engine, CPU + Metal backends, host-authoritative `Vec<f32>` |
| `crates/cuda-kernels/` | [AGENTS.md](crates/cuda-kernels/AGENTS.md) — prelude discipline, csrc layout, TileLang AOT |
| `crates/mlx-sys/` | [AGENTS.md](crates/mlx-sys/AGENTS.md) — single Metal bridge, cmake+cc build, no repo `.metal` |

---

## Core docs (on-demand)

- [`docs/index.md`](docs/index.md) — PARA index; always start a session here.
- [`docs/codebase-map.md`](docs/codebase-map.md) — execution paths + where to start reading.
- [`docs/architecture.md`](docs/architecture.md) — package boundaries, dependency direction, crate-split governance.
- [`docs/plans/cuda-kernel-crate-extraction.md`](docs/plans/cuda-kernel-crate-extraction.md) — final `cuda-kernels` extraction blueprint (trip wires + acceptance).
- [`docs/support-matrix.md`](docs/support-matrix.md) — backend / model / quant support levels.
