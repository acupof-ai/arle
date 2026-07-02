# ARLE — Agent Contract

Assisting **ckl**. **Project-specific** rules only; generic Rust/CUDA/Metal/git
knowledge is intentionally absent. Load the relevant module `AGENTS.md`
(§Module Guides) before editing inside that module.

---

## §0 First principle — SOLID (truth-seeking, pushed to the extreme)

**Everything must be SOLID. Not SOLID enough → keep digging.** The quality bar,
not a suggestion.

- **Inference ≠ evidence.** Source survey / grep / doc / callgraph are
  *hypothesis*; evidence = measured nsys / bench numbers / runtime log counter /
  controlled-variable A/B. No evidence → label it hypothesis, no conclusion.
- **Isolate confounders.** One experiment changing N variables at once (buffer
  pool + scheduler clamp + KV format + graph capture) is **unattributable** —
  change one variable at a time, or run an explicit control.
- **Root-cause hypotheses get license-or-kill too** — not just fixes. The
  inference itself needs a cheap verify (nsys fraction / log counter / second
  source read / A/B). Wrong root cause → every sub-experiment wasted.
- **80% SOLID is not enough.** Dig to 95%+, or explicitly declare "deferred,
  accepting the uncertainty". **No silent pass.**
- **Self-check before shipping** any plan / wins / errors / brief /
  recommendation: "Is this SOLID? Where's the gap? Dig, or explicitly defer?"
- **Wall-clock / per-request framing is ground truth.** When framings disagree
  (per-NVTX-window vs wall-clock, per-launch vs per-token, per-layer vs
  per-request), a narrow-window X% share ≠ the actual wall-clock impact.
  License-or-kill uses the wall-clock framing — never the narrow-window one.
- **Case-as-fact — attribute decoded cases before overturning a hypothesis (做算法
  以 case 为事实).** A negative/bug result (a regression, a failed metric, a
  subagent's "it's structural") is a **case to debug at the token level**, NOT a
  license to generalize into a structural KILL. Before trusting any aggregate:
  ① **decode the actual model outputs** per-case/per-step on the *failing* slice;
  ② **audit the eval harness for artifacts** (timeouts/errors bucketed as a
  class, request-errors counted as wrong, a metric rewarding the wrong thing).
  Aggregate **and** mechanism can both lie; decoded cases are ground truth. The
  fix usually falls out once attributed at case level. **先归因清楚再推翻。**

Empirical anchors:
- **Agentic-OPD "structural" false-KILL** (2026-06-20): a −14pp regression +
  plausible mechanism ("on-policy can't teach abstention") written up as a
  structural KILL — **wrong**. The gate's "+42pp teacher abstention" was 14/17
  teacher TIMEOUTS counted as abstention (fake gate); the think-on teacher
  actually over-calls (33% abstain < base 46%). Aggregate **and** subagent
  summary both misled; only decoded cases were true. Fixable, hypothesis intact.
- **M_pf-graph Phase 0 KILL** (2026-05-08): the errors entry was 80% SOLID and
  still void — launch-overhead share not nsys-verified / graph-trigger count not
  measured against a control / 4 variables changed at once.
- **M_pf-graph v2 framing trap** (2026-05-08): nsys "55.7% of prefill window"
  looked like a PASS, but 191ms / 60s trace = 6.4ms per prefill / 1995ms TTFT =
  **0.32% wall-clock**, far below the 10% kill threshold. nsys "X% of NVTX
  window" must cross-check "Y ms / per-request total"; take the conservative one.

---

## §0.1 Decompose to the implementation level

For a hard/complex task, don't be intimidated: unravel into atomic tasks →
dependency DAG (who blocks whom) → critical path + budget. Fine-grained enough,
the plan falls out on its own.

- **To the implementation level, not the principle level.** "Pre-allocate, don't
  copy big buffers" is a principle, not a spec — go to the exact buffer / size /
  call site / precondition. **Claude produces the line-level spec, the executor
  copies it verbatim**; free improvisation drops fields (the DSv4 rollback
  snapshot's first version missed the `sw_window` + `fp8_kv_pool` buffers).
- **Every state change enumerates each mutated buffer and proves each.** rollback
  / cache / scratch / fusion / quant: list **every** device buffer written, give
  each a disposition + **exact precondition** — never "should self-heal":
  ① rolled back by an existing path (name it); ② self-heals (write the
  precondition — a ring's speculative write self-heals **only** for seq_len <
  ring_size; beyond that it aliases a live slot); ③ snapshot/restore. Full
  enumeration exposes the gap a partial fix missed.
- **Inline speed into correctness.** Pre-allocate once and reuse (no per-step
  alloc — churn + the disabled-event-tracking premature-free); copy at the
  smallest grain (ring moves one slot → store that one slot, not the whole ring);
  fold into the opt-in path, default baseline byte-for-byte unchanged,
  A/B-verify the baseline tok/s doesn't regress.
- **Correct inference ≠ baseline identity.** The gate for spec-decode / quant /
  kernel-swap is **correct inference** (needle retrieval + same-config-twice
  non-determinism floor + self-consistency: the new kernel's own autoregressive
  output is the reference), NOT token-exact-vs-baseline (confounded by MoE
  non-determinism). A degenerate (looping) prompt is not a valid test case.
- **Root-cause on a clean baseline.** Land + isolate the precondition first;
  refining a fix on a baseline that still has the bug confounds the garbage.

Empirical anchor:
- **DSv4 EAGLE rollback** (2026-06-06): `truncate_decode_len` restored only
  `compressed.seq_len`, missing `pending_kv`/`prev_overlap` → the draft corrupted
  at the compression boundary; full enumeration then exposed the missing
  `sw_window` + `fp8_kv_pool` ring slots (self-heal only for seq_len <
  sliding_window). Byte-identity had been (wrongly) used as the EAGLE gate.

**Implementation gate — simple before you start** (skill
[`understand-until-simple`](.claude/skills/understand-until-simple/SKILL.md)).
Before ANY implementation code: ① decompose to the atomic level (a concrete
**file:line / kernel / loop**, not a concept) ② get measured evidence
(controlled-variable A/B, not inference) ③ let measurement **correct** your
hypotheses (the load-bearing assumption is exactly the one to measure) ④ until
you can state the fix in **one sentence + one measured number** and name the next
wall. **Still saying "hard / tough / multi-day" means not yet decomposed — "hard"
is a confession, not a property; keep decomposing.** Can't compress to one
sentence = no code.

**Cost comes AFTER the concrete work, never before** (ckl 2026-06-20). ①
investigate to the file:line decomposition first → ② only THEN evaluate cost.
**Don't label difficulty / risk / "infeasible" / "multi-day" before the steps are
concrete** — a cost guessed ahead of decomposition raises perplexity, biases the
work down, pre-commits a wrong size. Catch yourself ranking risk before naming
file:line steps → stop and decompose; the estimate is noise. Empirical: DSv4
prefix-reuse (2026-06-20) decomposed into 6 file:line steps reusing the existing
`Dsv4LayerImage` capture/restore, yet got wrapped in "hard / infeasible / HIGH
risk" — pure noise. DSv4 batched decode (2026-06-14) hand-waved "very hard,
multi-day new infra"; reading the code collapsed it to "one per-row `for` loop at
`dsv4.rs:1872` → batch it → ~2× aggregate, MoE 3.70× sub-linear doesn't cap".

---

## Project shape

`ARLE` is a Rust-native, device-neutral inference runtime with integrated local
agent and **On-Policy Distillation (OPD)** workflows. The runtime is primary:

- **`infer-*` rewrite stack owns serving/runtime truth**: `infer-plan` (IR) →
  `infer-seam` (host-only traits) → `infer-core` (Engine/scheduler/RadixCache) →
  `infer-cuda`/`infer-metal` (executors) → `infer-server`/`infer-api`;
  `infer-topo`/`infer-moe`/`infer-util` are shared leaves. The monolithic
  `infer/` crate was **deleted 2026-06-04** (`e81b98fb`, ~167k LOC) — any
  doc/command referencing `infer/` or `-p infer` is stale.
- `arle` is the runtime-led CLI front door (local agent, OPD train, eval).
  `infer-api` (`LoadedInferenceEngine`) is the single programmatic front door.
- `train` extends the same runtime/model authority via **OPD only** — not a
  second equal product line. Scratch pretrain / SFT / GRPO / multi-turn RL were
  deleted (2026-05-18 pivot —
  [`docs/projects/2026-05-18-opd-only-pivot.md`](docs/projects/2026-05-18-opd-only-pivot.md)):
  pretrain unwinnable (322× gap), SFT/GRPO/multi-turn duplicate mature OSS
  (vLLM+verl, TRL, axolotl). OPD is the one axis where ARLE's runtime authority
  structurally differentiates — strong teacher inference + tight student-scoring
  latency, both already in the `infer-*` runtime (teacher surface on `infer-api`).

No PyTorch and no Python on the hot path. Two backends plug into one seam
(`infer_seam::{BackendExecutor, KvPool}` — two host-only traits): the CUDA
continuous-batching executor (Linux/NVIDIA, `cudarc` + vendored FlashMLA/DeepGEMM/
DeepEP + TileLang AOT + native CUDA C) and the Metal executor (Apple Silicon,
`crates/mlx-sys` C++ bridge — continuous batching, variable-length packed decode
via mlx-lm `BatchKVCache`). The same `infer_core::Engine<E, K>` drives both; a new
backend = implementing the two seam traits, not touching scheduler/cache/server.
Models: Qwen3-dense + Qwen3.5/3.6 (hybrid·MoE) on CUDA + Metal; DeepSeek-V4-Flash
+ GLM-5.2 (CUDA 8×H20 TP=8/EP=8; GLM-5.2 verify pending); Qwen3.6 + Gemma4 ·
DeepSeek-OCR VLMs + DiffusionGemma (Metal). Full tiers: docs/support-matrix.md.

**Metal canonical model — globally unified (2026-05-07):
`mlx-community/Qwen3.6-35B-A3B-4bit`** (MoE, ~19 GB, cached at
`~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit`).
Production target per [`README.md`](README.md) backend matrix +
[`ROADMAP.md`](ROADMAP.md) Next-Model queue; catches MoE perf/correctness
regressions that Qwen3.5-0.8B (dense) can't.
- **Scope**: every Metal serve (`arle serve --backend metal`; legacy
  `metal_serve` bin deleted), `scripts/bench_*.sh` default, smoke test, and
  Metal-track `wins`/`errors` entry. CUDA benches keep existing defaults.
- **Opt-out**: small models stay in `models/` for unit tests that need one — set
  `INFER_TEST_MODEL_PATH=models/Qwen3.5-0.8B-MLX-4bit` and document the reason.
- **Bench invocation**: `./scripts/bench_*.sh <label> --model
  mlx-community/Qwen3.6-35B-A3B-4bit` (HF id resolves to the cached snapshot).
  Direct: `arle serve --backend metal --model-path mlx-community/Qwen3.6-35B-A3B-4bit`.
- **Auto-wired-limit** (always-on,
  [2026-05-07-bench-qwen36-mle-perf](docs/experience/wins/2026-05-07-bench-qwen36-mle-perf.md)):
  the Metal executor auto-pins weights via `mlx::set_wired_limit` at construction
  (`infer-metal/src/wired_limit.rs`; model dir size + 1 GiB headroom, follows HF
  symlinks). c=1 p99 86 → 15 ms on Qwen3.6 (−82%). Monolith-era
  `--wired-limit-bytes` flag gone.
- **MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER — not a default.**
  Qwen3.5-dense-only tune; on Qwen3.6 MoE benched wash-or-loss (95% of step is
  `mx::async_eval` encoding ~600-1000 primitives). Per-workload matched-A/B only.
  Refs: [baseline](docs/experience/wins/2026-05-07-bench-qwen36-baseline.md),
  [encode-bottleneck](docs/experience/wins/2026-05-07-bench-qwen36-encode-bottleneck.md).

**Workspace (current, post-rewrite 2026-06-04):**

```
ARLE/
├── src/                       ← thin `arle` binary (root package `arle`)
├── crates/
│   ├── infer-plan/            ← backend-independent forward IR (ForwardPlan)
│   ├── infer-seam/            ← host-only traits: BackendExecutor, KvPool, KvBatchDescriptor
│   ├── infer-core/            ← Engine<E,K>: scheduler, RadixCache, chunked prefill, sampling
│   ├── infer-cuda/            ← CUDA executor (Qwen + DSv4, TP/EP, FlashMLA/DeepGEMM/DeepEP)
│   ├── infer-metal/           ← Metal/MLX executor (packed varlen decode)
│   ├── infer-hip/ infer-vulkan/  ← experimental AIPC backends (HIP DSv4 GGUF lane; Vulkan skeleton)
│   ├── infer-server/          ← HTTP serving (OpenAI v1 compat)
│   ├── infer-api/             ← single front door: LoadedInferenceEngine + OPD-teacher surface
│   ├── infer-topo/ infer-moe/ infer-util/  ← shared leaves (TP/EP topology, MoE, hf_hub/logging)
│   ├── agent/ agent-bench/ chat/ cli/ tools/  ← control-plane crates (`cli` hosts serve/REPL)
│   ├── autograd/              ← from-scratch autograd + optimizer (OPD substrate)
│   ├── cuda-kernels/          ← csrc/{attention,gemm,kv,quant,misc}/, tools/tilelang/, ffi/, NCCL
│   ├── deepep-sys/            ← DeepEP/NVSHMEM FFI (internode_ll dispatch/combine)
│   ├── hip-sys/ hip-kernels/ vulkan-sys/ vulkan-kernels/  ← AIPC FFI + kernel layers (stub off-box)
│   ├── deepseek-spec/ qwen3-spec/ qwen35-spec/ gemma-spec/  ← model config + tensor-name + Shard contracts
│   ├── kv-native-sys/         ← local persistence substrate for KV tier transports
│   ├── mlx-sys/               ← MLX + C++ bridge (cmake + cc)
│   ├── train/                 ← OPD-only training (teacher via infer-api)
│   └── xgrammar-sys/          ← grammar-constrained decode FFI
├── vendor/                    ← vendored official kernels (FlashMLA, DeepGEMM, DeepEP, …)
└── docs/                      ← projects/ plans/ experience/ reviews/ resources/
```

CUDA kernels live at `crates/cuda-kernels/csrc/` + `vendor/`
(adopt-official-first; hand-rolled only for the genuine gap). Workspace topology
source of truth: [`docs/codebase-map.md`](docs/codebase-map.md).

---

## Rules

### Execution phases (non-trivial tasks)

| Phase | Exit condition |
|-------|----------------|
| **Explore** (trace callers, grep prior art, list trait implementors) | You can name every file you will touch. |
| **Plan** (ask "how would this fail?" first; >5 files or irreversible → stop + flag) | Written approach the user accepted. |
| **Implement** (check prior art in `crates/infer-*/src/` + `docs/`; outside plan → update plan) | Diff compiles under the relevant feature set. |
| **Verify** (`cargo test --workspace`; justify every new `unwrap()`/alloc/async path; **bench entry per §Benchmarks** if in-scope) | Tests green, `cargo clippy -- -D warnings` clean, **wins/ entry committed (or stub `pending-remote`)**. |
| **Reflect** (bug >1 attempt → `docs/experience/errors/`; correction → feedback memory) | Experience entry committed. |

Skip rules: trivial → Implement + Verify; exploration questions → Explore only.

### Editing

- **Preserve by default.** Never delete content not explicitly in scope.
- **Keep code simple and uniform.** Prefer deletion-style refactors: remove
  obsolete paths, collapse duplicate helpers/branches, converge on one canonical
  flow instead of layering adapters.
- **Module reorganisation is in scope for deletion passes.** Merging, splitting,
  reordering, renaming — the target is the minimum correct arrangement, not
  preserving the original structure. Large-scale restructuring (>5 files) follows
  the Approach-first rule below.
- **`AGENTS.md` is canonical.** A sibling `CLAUDE.md` stays a full rule document
  aligned with it, not a thin pointer.
- **Approach-first for >3 files or architectural decisions** — outline and wait.
- **No half-states** (`feedback_no_half_states.md`): finish a refactor unit or
  revert it, never leave parallel old+new paths in the tree.

### Backend isolation (CRITICAL)

- `#[cfg(feature = "cuda")]` / `#[cfg(feature = "metal")]` gating; **never
  `cfg`-leak backend types into cross-backend modules** — everything above the
  seam (`infer-core`/`-server`/`-api`) stays device-neutral; backend types live
  only in `infer-cuda` / `infer-metal`.
- CUDA stubs on non-CUDA targets: `todo!("GPU required: ...")`.
- Pre-push type check on Mac without nvcc:
  `cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib`.

### Delegation (subagents execute, Codex reviews, parallel by default)

Claude = **direction + integration**. Execution → **`general-purpose`** subagents
(Agent tool); research/mapping → **`Explore`**; large cross-cutting plans →
**`Plan`**; review → **`codex review` at the Bash tool** (a shell command, not a
subagent). **DO NOT use `codex:codex-rescue` / `mcp__openmax__execute_with_codex`
for execution** — both hang ("codex hangs", observed 2026-04-19;
`feedback_codex_subagent_hangs.md`); review-via-Bash is unaffected. Reserve direct hand-written diffs for ≤ ~3 files / trivial
mechanical changes.

| Area | Owner |
|------|-------|
| Docs, planning, architecture, roadmaps | Claude |
| Code execution (implement/refactor/tests) | **`general-purpose` subagent** |
| Broad codebase exploration / scope mapping | **`Explore` subagent** |
| Implementation planning spanning >5 files | **`Plan` subagent** |
| Code review of non-trivial diffs | **Claude runs `codex review` at Bash** |
| Stuck-problem rescue (2-strike hand-off) | **`general-purpose` with full context** |

- **Parallel by default.** Independent delegated tasks → single message, multiple
  Agent calls. Serial only when data-dependent.
- **Code review:** `codex review --uncommitted` (or `--commit <sha>` / `--base
  <branch>`) at Bash, background + tee to tmp — non-blocking
  (`feedback_codex_review_async.md`).
- **2-strike rule:** two failed subagent attempts → hand-write the diff (if
  small) or re-brief a fresh `general-purpose` with what prior attempts tried.

### Execution hygiene (Claude and delegated agents alike)

- Surface known failure logs upfront so the same blocker isn't re-discovered.
- Pin SKU / shape / scope at exact granularity, not by fuzzy name — else
  everything gets enabled then narrowed down.
- Before patching an upstream component, grep the raise point and lock the root
  cause first.
- Probing install / directory / env layout: enumerate candidate paths upfront,
  not fail-then-retry.
- PR branches start from `upstream/main`, never a local WIP branch — defaults
  pick current HEAD, so state it explicitly.
- Verify a patched upstream lib in an isolated dir, never the existing dev
  install (dodges editable / `.pth` finder hijacks).
- Upstream patch crossing a size or cross-cutting-policy threshold → pause + ack
  before landing.
- Regression tests mirror the failure mode with a minimal in-component kernel,
  not by importing caller code.

### Benchmarks

- **Spec — always read first:**
  [`docs/bench-and-trace-spec.md`](docs/bench-and-trace-spec.md) — mandatory
  report sections (Goal · Hypothesis · Params · Env · Results · Problems ·
  Learnings), goal taxonomy, watch-list, **auto-iteration rules** (§6: when to
  loop/stop, information-volume triggers), and **§7 hard-won protocol rules**
  (correctness gate, sweep≠fixed-c, duration adequacy, param-alignment via §3.2
  envelope log, server lifecycle hygiene). Internal info sources (§3: `/v1/stats`
  service trace, scheduling envelope, K6 OOM detector) are first-class report
  content. Applies to benchmarks and traces.
- **MANDATORY — every runtime change produces a bench entry.** A diff isn't
  "done" until a dated entry lands under `docs/experience/wins/` (or `errors/` on
  regression). Verify-phase exit condition. No entry → not shipped.
  - **In scope:** `crates/infer-*/src/`, `crates/cuda-kernels/csrc/`,
    `crates/mlx-sys/src/`, `src/`, `scripts/bench_*.{sh,py}` param changes,
    feature-flag default flips, hot-path dep bumps.
  - **Exempt:** docs / `AGENTS.md` / `CLAUDE.md` / memory / dev-only tooling /
    gitignored output. State so in the commit body.
  - **Minimum:** one `scripts/bench_guidellm.sh` run vs latest baseline for the
    affected backend+model, with Δ% row. Full sweep only for optimization /
    architectural changes.
  - **Can't run locally** (e.g. CUDA on a Mac): commit body cites the remote
    ticket; stub the `wins/` entry with `pending-remote`. No silent skips.
  - **Auto-iterate** per spec §7; cross-link wins back to the commissioning
    project/plan.
- Snapshot to `docs/experience/wins/YYYY-MM-DD-bench-guidellm-<label>.md` using
  the [`TEMPLATE-bench-guidellm.md`](docs/experience/wins/TEMPLATE-bench-guidellm.md)
  skeleton. **Never overwrite**; after-snapshots cite before-snapshots with deltas.
- **Canonical tool: `scripts/bench_guidellm.sh <label>`** — thin wrapper around
  [`vllm-project/guidellm`](https://github.com/vllm-project/guidellm) (vLLM
  official, LLM-native TTFT/ITL/tok-s, sweep profile, HTML report). Canonical
  params locked in
  [`docs/plans/guidellm-integration.md`](docs/plans/guidellm-integration.md) §3 —
  changing them is a deliberate commit, not a flag flip.
- Include: GPU model, CUDA/Metal version, model, num_slots, non-default flags,
  feature set. Raw output table, not summaries. Install once:
  `pip install -e .[bench]` (guidellm ships in the `bench` extra).

### Docs lifecycle & progress spine

- **Per-file status header is the truth.** Every `docs/plans/` +
  `docs/projects/` doc carries `> Status: Active | Shipped | Superseded by
  <link> | Killed — <date>` directly under its title; `docs/index.md` tables
  are pointers only and carry no narrative state snapshots (they rot against
  ROADMAP — the 2026-06-10 index snapshot contradicted it by 06-21). Migrate
  legacy docs on touch, not in bulk.
- **CHANGELOG is the progress spine.** Three event classes land a CHANGELOG
  line the same day, linking the wins/errors entry: **phase exit · default
  flip · license-or-kill verdict**. Phase exits also cut a release tag; a tag
  without its CHANGELOG section is a regression (v0.1.5→v0.2.1 backfilled
  2026-07-02).
- **Weekly resync (~30 min):** ① ROADMAP phase table ↔ GitHub issues (issues
  win); ② index "Active" sweep — Active docs untouched for 30 days get
  confirmed or moved to Archived; ③ CHANGELOG catch-up for the week's three
  event classes; ④ promote errors/wins patterns recurring ≥3× into
  §Distilled lessons; ⑤ wins-cap headroom — when the top-level count is
  within ~20 of the `check_repo_hygiene` cap, archive the oldest
  zero-inbound-reference entries (verify per basename via `git grep`) so
  the ratchet batches here instead of blocking mid-week pushes (it tripped
  twice on 2026-07-02). Bench-entry drift probe: diff
  `git log --since='7 days ago' --oneline -- 'crates/infer-*/src' crates/cuda-kernels/csrc`
  against the same for `docs/experience/`.

### Git

- Commitizen: `<type>(<scope>): <subject>`. Scopes: `metal`, `cuda`,
  `scheduler`, `qwen3`, `qwen35`, `http`, `kv-tier`, `docs`.
- Commit directly to `main`, from the current branch in the current workspace —
  no feature branches, no separate worktree/alternate checkout
  (`feedback_commit_to_main.md`).
- **Commit small tranches immediately.** Each self-contained change lands as its
  own commit; run verification after, fix issues in a follow-up commit, don't
  fold micro-changes into one opaque diff.
- **Never `git stash`** unrelated user changes — leave others' dirty paths in
  place, commit only your own files by explicit path.
- After `git mv` + batch Edits, re-check `git status` and re-stage by path — the
  fmt hook de-stages renames (`feedback_git_mv_with_fmt_hook.md`).

### Code conventions

- **Flat module layout, no `mod.rs`.** `src/ops.rs` declares `#[path =
  "ops/attention.rs"] mod attention;` siblings; models follow `model/qwen3.rs` +
  `model/qwen3/`.
- Weights `&self` (immutable, pool-shared); per-request mutable state in `State`
  associated types.
- **Comments concise** — ≤1-2 lines on the non-obvious *why*, not what the code
  does; no essay blocks. Load-bearing invariant/ordering notes stay, compressed.
- **Code as poetry — every expression earns its place.** Use the stdlib's
  vocabulary when it names the operation exactly: `.unzip()` over a 4-line match,
  `ensure!` over `if { return Err }`, `.is_some_and()` over `.map().unwrap_or()`,
  iterator chains over for-push when the shape is a direct formula. The test:
  *can a reader parse the intent in one pass without re-reading?* Named
  temporaries that just alias the previous line add noise; the range
  `i * tp_size..(i+1) * tp_size` is already named.
- **Module ordering is part of the design.** Items should appear in dependency
  order — helpers before callers, types before impls that use them, public API
  before internals. Arbitrary ordering forces readers to scroll; ordered code
  reads like a proof.

### GPU kernel work

Touching `crates/cuda-kernels/csrc/` or `crates/mlx-sys/src/` hot paths? Evaluate
against the project-specific heat map in
[`docs/reviews/2026-04-14-cuda-kernel-six-principles-review.md`](docs/reviews/2026-04-14-cuda-kernel-six-principles-review.md).
Measure with `ncu` (CUDA) or Xcode Metal capture / MLX instruments (Metal).

### Distilled lessons (cross-module, recurring ≥3 entries)

- **SLO verdict from the SLO workload, not a smoke shape** — a c=1 short-prompt
  nsys "2× win" routinely flips on production prompt length (path scaling is
  shape-specific) (`errors/2026-05-27-dsv4-tp-allreduce-slo-prefill-kill.md`).
- **`plan_label=mixed` / "executes new path" is reachability, not a license** —
  c-sweep must clear TTFT *and* ITL *and* output throughput before a default flip
  (`errors/2026-05-25-axis2-mixed-default-kill.md`,
  `errors/2026-05-26-qwen35-hybrid-mixed-kill.md`,
  `errors/2026-05-25-axis3-chunked-prefill-size-kill.md`).
- **Default flips need multi-shape verification** — single-shape ROI shows what's
  possible; ≥2 binding production shapes show what's safe
  (`wins/2026-05-08-prefill-cap-8-multi-shape-safe-default-flip.md`,
  `errors/2026-05-08-w4-c8-deadlock-confirms-workload-dependent.md`).
- **A/B must be same-binary, same-shell, same-prompt, two env flips,
  side-by-side** — cross-day claims drift backend / KV dtype / scheduler tuning
  (`wins/2026-05-27-dsv4-native-deepep-perf-ab.md`).
- **Smoke-output garbage is config-suspect first** — A/B against the prod backend
  on the *same* config before staring at new code; if prod also breaks, the
  serving config is the bug (`wins/2026-05-27-dsv4-native-deepep-pod-e2e.md`).
- **Launch-count source-survey is hypothesis** — a fused-kernel rewrite for tiny
  CUDA ops is licensed only by a paired component A/B (or nsys/CUDA-event) under
  the runtime's sync framing
  (`errors/2026-05-12-fp8-kv-pair-quantize-fusion-no-license.md`,
  `errors/2026-05-21-arle-cuda-opd-swiglu-fused-kill.md`).
- **Capability claims <5pp on small-n (≤200) evals need multi-seed (≥5) +
  mean±σ + Wilson 95% CI** before the wins entry — "best ckpt across
  save-every-10" is positively biased
  (`errors/2026-05-28-mmlu-cross-base-was-noise.md`).
- **Pod-side probe trust is conditional on git+symbol checks** — verify the pod
  tree is a git repo at HEAD and `strings target/release/arle | grep <symbol>`
  shows the change landed; the binary proves some tree was current *whenever last
  built*, not that current source built it
  (`errors/2026-05-28-dsv4-flashmla-decode-parity-precond-fail.md`).
- **Greedy-decode the actual generation when a metric looks catastrophic** — 3
  weeks of "FP8 KV broken" collapsed when one `eprintln!` showed a test-framework
  artifact (`errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md`).
- **`scripts/dsv4_toolchain.sh` validates DSv4 build-flow before launch** —
  native DeepEP/DeepGEMM need env-checked source + compile prereqs, else a stub
  binary errors at runtime (`wins/2026-05-27-dsv4-native-deepep-run-guide.md`).

---

## Memory

- **Always-load:** auto-memory index + latest 3 of `docs/experience/errors/` and
  `docs/experience/wins/`.
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
CUDA_HOME=/usr/local/cuda cargo build --release --features cuda              # CUDA (Linux+NVIDIA)
cargo build --release --no-default-features --features metal,no-cuda         # Metal (Apple Silicon)
cargo build --release --no-default-features --features cpu,no-cuda           # portable / CI smoke
# Multi-GPU features stack: nccl (implies cuda) → deepep (implies nccl)

# Mac CUDA-Rust typecheck without nvcc (CI-mirrored):
cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib

# Tests (CI-mirrored; see .github/workflows/*.yml for the full matrix):
cargo test -p arle --release --no-default-features --features cpu,no-cuda,cli
cargo test -p cli --release --no-default-features --features metal,no-cuda   # Metal
cargo test -p kv-native-sys --release
```

**KV precision parity gate — re-ported 2026-06-10 (#58).** The monolith's
trajectory-match audit is superseded by the correct-inference gate
(`scripts/needle_gate.py` + `scripts/lever_gate.sh`): needle ladder x3
same-config repeats vs the baseline envelope, NOT byte-identity (MoE
non-determinism). DSv4 lever verdicts
([wins](docs/experience/wins/2026-06-10-dsv4-lever-gate-license-or-kill.md)):
FlashMLA decode + fused-wqkv correctness LICENSED (default flips still need a
wall-clock perf license); pooled/contig-MoE flip KILLED (−24%). Qwen dense
KV-dtype matrix **resolved 2026-06-12 (#68)**: seam-level kv-dtype dispatch
landed (`--kv-cache-dtype`, default bf16 unchanged); INT8/FP8 correctness
LICENSED (needle exact 15/15 DET = BF16 envelope); the initial decode −77% at B=1
was an uncached per-layer-per-step `cudaGetDeviceProperties` in the quant decode
shim — fixed same day (static SM-count cache), post-fix −27% vs bf16+graph / −7%
vs eager bf16 — opt-in only, no default flip without a perf license; TQ4 DEFERRED
(TurboQuant page_size=1 vs TileLang PAGE_SIZE=16). Verdicts:
[wins](docs/experience/wins/2026-06-12-cuda-quant-kv-dispatch-int8-fp8.md).

Env vars: `TORCH_CUDA_ARCH_LIST` (SM override, PyTorch convention; alt
`CMAKE_CUDA_ARCHITECTURES`), `INFER_TILELANG_PYTHON` (TileLang AOT Python),
`INFER_TEST_MODEL_PATH` (default `models/Qwen3.5-4B`). Full list:
[`docs/environment.md`](docs/environment.md). SM tier policy:
[`docs/plans/sm-coverage.md`](docs/plans/sm-coverage.md).

Disk hygiene: `cargo sweep --time 30` (weekly) prunes target/ artifacts older
than 30 days. Dev profile keeps deps DWARF-free (root `Cargo.toml`
`[profile.dev.package."*"] debug = false`).

---

## Module Guides

Load the relevant `AGENTS.md` **before** editing inside a module. The per-module
guides under the old `infer/src/**` were deleted with the monolith; for the
`infer-*` rewrite crates the module truth is
[`docs/architecture.md`](docs/architecture.md) +
[`docs/codebase-map.md`](docs/codebase-map.md).

| Path | Guide |
|------|-------|
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
