# ARLE — Agent Contract

Assisting **ckl**. **Project-specific** rules only; generic Rust/CUDA/Metal/git
knowledge is intentionally absent. Load the relevant module `AGENTS.md`
(§Module Guides) before editing inside that module.

---

## §0 First principle — SOLID (truth-seeking, push to the extreme)

**Everything must be SOLID. Not SOLID enough → keep digging, keep breaking through.**
Not a suggestion — the quality bar.

- **Inference ≠ SOLID**: source survey, code grep, doc analysis, callgraph
  inference are all *hypothesis*, not evidence. Evidence = measured nsys trace /
  bench numbers / runtime log counter / controlled-variable A/B. No evidence → no
  conclusion; label it hypothesis.
- **Confounders must be isolated**: one experiment changing N variables at once
  (buffer pool + scheduler clamp + KV format + graph capture) → any result is
  **unattributable**. Change one variable at a time, or run an explicit control to
  isolate the confounder.
- **Root-cause hypotheses also get license-or-kill**: license-or-kill is not just
  for fixes — the **root-cause inference** itself needs a cheap experiment to
  verify (nsys fraction / log counter / a second source read / A/B). Wrong root
  cause → every sub-experiment is wasted.
- **80% SOLID is not enough**: when you find a gap, dig to 95%+, or explicitly
  declare "deferred, accepting the uncertainty". **No silent pass.**
- **Self-check before shipping**: before any plan / wins / errors / brief /
  recommendation lands, ask "Is this SOLID? Where's the gap? Dig, or explicitly
  defer?". Below bar → reflect, keep digging.
- **Cross-check framing from multiple angles**: when the same data yields
  different conclusions under different framings (per-NVTX-window vs
  per-wall-clock, per-launch vs per-token, per-layer vs per-request),
  **wall-clock / per-request framing is ground truth**. A narrow-window X% share
  ≠ the actual X% wall-clock impact. License-or-kill must use the wall-clock
  framing — never deceive yourself with the narrow-window framing.
- **Case-as-fact — attribute from decoded cases before overturning a hypothesis
  (做算法以 case 为事实)**: a negative / bug result (a regression, a failed metric,
  a subagent's "it's structural" summary) is a **case to debug at the token
  level**, NOT a license to generalize into a structural conclusion that kills the
  hypothesis. Before trusting any aggregate: ① **decode the actual model outputs**
  per-case, per-step (base vs each checkpoint, on the *failing* slice); ② **audit
  the eval harness for artifacts** — timeouts/errors silently bucketed as a class,
  request-errors counted as wrong, a metric that rewards the wrong thing. The
  aggregate **and** a plausible mechanism can *both* lie; the decoded cases are
  ground truth. Once attributed at case level, the fix usually falls out — never
  kill a hypothesis on a confounded experiment. **先归因清楚再推翻。**

Empirical anchors:
- **Agentic-OPD "structural" false-KILL** (2026-06-20): a −14pp agentic-OPD
  regression + a plausible mechanism ("on-policy can't teach abstention") was
  written up as a structural KILL — **wrong**. Decoding the cases: the gate's
  "+42pp teacher abstention" was **14/17 teacher TIMEOUTS counted as abstention**
  (a fake gate); the no-think teacher actually *over-calls* (33% abstain < base
  46%); OPD faithfully distilled that bad target. Fixable (timeout-clean gate +
  think-on teacher), hypothesis intact. The aggregate metric **and** the subagent
  summary both misled; only the decoded cases were true.
- **M_pf-graph Phase 0 KILL** (2026-05-08): the errors entry was 80% SOLID and
  still fell — 3 gaps: launch-overhead share not nsys-verified / SGLang
  graph-trigger count not measured against a control / 4 variables changed at once
  → the strategic conclusion was wholly void.
- **M_pf-graph v2 framing trap** (2026-05-08 EOD+19): nsys "55.7% of prefill
  window" looked like a PASS, but 191ms / 60s trace = 6.4ms per prefill / 1995ms
  TTFT = **0.32% wall-clock**, far below the 10% kill threshold. **Lesson**: nsys
  "X% of NVTX window" must cross-check the "Y ms / per-request total" framing; take
  the more conservative as the license-or-kill baseline.

---

## §0.1 Decompose to the implementation level — unravel it, leave nothing out

For a hard/complex task, **don't be intimidated by the difficulty**: unravel it
into atomic tasks → draw the dependency DAG (who blocks whom) → compute the
critical path + budget. Fine-grained enough, and the plan falls out on its own.

- **Down to the implementation level, not the principle level.** "Pre-allocate,
  don't copy big buffers" is a principle, not a spec; go to the exact buffer /
  size / call site / precondition. **Claude produces the line-level spec, the
  executor copies it verbatim** — free improvisation drops fields (empirical: the
  DSv4 rollback snapshot's first version missed the `sw_window` + `fp8_kv_pool`
  buffers).
- **Every state change must enumerate each mutated buffer and prove each one.**
  rollback / cache / scratch / fusion / quant all count: list **every** device
  buffer the operation writes, and for each give the disposition + **exact
  precondition** — never rely on "should self-heal": ① rolled back by an existing
  path (name it); ② self-heals (write the precondition — e.g. a ring buffer's
  speculative write self-heals **only** for seq_len < ring_size; beyond that it
  aliases a live slot); ③ must snapshot/restore. Full enumeration is what exposes
  the gap a partial fix missed.
- **Inline speed into correctness.** Pre-allocate once and reuse (no per-step
  alloc — churn + the disabled-event-tracking premature-free); copy at the
  smallest grain (the ring moves one slot → store only that one slot, not the
  whole ring); fold the whole thing into the opt-in path, the default baseline
  stays byte-for-byte unchanged, A/B-verify the baseline tok/s doesn't regress.
- **Correct inference ≠ baseline identity.** The gate for spec-decode / quant /
  kernel-swap is **correct inference** (needle retrieval + the same-config-twice
  non-determinism floor + self-consistency: the new kernel's own autoregressive
  output is the reference), **not** token-exact-vs-baseline (confounded by MoE
  run-to-run non-determinism). A degenerate (looping) prompt is not a valid test
  case.
- **Root-cause on a clean baseline.** Don't design the next fix's detail on a
  confounded baseline; land + isolate the precondition first (a broken s_q=K
  running on a baseline that still has the rollback bug → the garbage is
  confounded; refining on a dirty baseline is self-deception).

Empirical anchor:
- **DSv4 EAGLE rollback** (2026-06-06): `truncate_decode_len` only restored
  `compressed.seq_len`, missing `pending_kv`/`prev_overlap` → the draft corrupted
  at the compression boundary; full enumeration then revealed it also missed the
  `sw_window` + `fp8_kv_pool` ring slots (self-heals only for seq_len <
  sliding_window). Byte-identity had been used as the EAGLE gate (violating my own
  MoE-non-determinism memory), misjudging possible non-determinism as a bug.

**Implementation gate — simple before you start (skill
[`understand-until-simple`](.claude/skills/understand-until-simple/SKILL.md)).**
Before writing ANY implementation code, clear this gate: ① decompose to the atomic
level (a concrete **file:line / kernel / loop**, not a concept) ② get measured
evidence (controlled-variable profile / A-B, not inference) ③ let the measurement
**correct** your hypotheses (the load-bearing assumption is exactly the one you
must measure, not guess) ④ until you can state the fix in **one sentence + one
measured number** and name the next wall. **Still saying "hard / tough / complex /
multi-day" means you haven't decomposed enough — "hard" is a confession of
not-yet-decomposed, not a property of the problem; keep decomposing.** Can't
compress it to one sentence = you don't understand it yet, no code.

**Cost comes AFTER the concrete work, never before (ckl 2026-06-20).** Sequence
is mandatory: ① investigate to the file:line decomposition first → ② only THEN
evaluate cost. **Do not label difficulty / risk / "infeasible" / "multi-day"
before the steps are concrete** — a cost guessed ahead of the decomposition adds
confusion, not information (it raises perplexity, biases the work down, and
pre-commits you to a wrong size). When the work is clear the plan is usually
*simple*; state the steps and proceed. If you catch yourself ranking risk or
estimating effort before you can name the file:line steps, stop and go decompose
— the estimate is noise. Empirical: the DSv4 prefix-reuse investigation
(2026-06-20) decomposed cleanly into 6 file:line steps reusing the existing
`Dsv4LayerImage` capture/restore, yet still got wrapped in "hard / likely
infeasible cheaply / HIGH risk" — the hedging was pure noise on top of a clear plan.

Empirical: DSv4 batched decode (2026-06-14) was hand-waved as "very hard,
multi-day new infra"; reading the code + measuring the step decomposition
collapsed it to "one per-row `for` loop at `dsv4.rs:1872` → batch it → ~2×
aggregate, MoE 3.70× sub-linear doesn't cap" — the "hard" was just un-decomposed.

---

## Project shape

`ARLE` is a Rust-native, device-neutral inference runtime with integrated
local agent and **On-Policy Distillation (OPD)** workflows. The runtime
remains primary:

- The **`infer-*` rewrite stack owns serving/runtime truth**: `infer-plan`
  (IR) → `infer-seam` (host-only traits) → `infer-core`
  (Engine/scheduler/RadixCache) → `infer-cuda`/`infer-metal` (executors) →
  `infer-server`/`infer-api`, with `infer-topo`/`infer-moe`/`infer-util` as
  shared leaves. The monolithic `infer/` crate was **deleted 2026-06-04**
  (`e81b98fb`, ~167k LOC) — any doc/command referencing `infer/` or
  `-p infer` is stale.
- `arle` is the runtime-led CLI front door for local agent, OPD train,
  and eval workflows. `infer-api` (`LoadedInferenceEngine`) is the single
  programmatic front door.
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
  rollouts — both already in the `infer-*` runtime (teacher surface on
  `infer-api`).

No PyTorch and no Python on the hot path. Two backends plug into one seam
(`infer_seam::{BackendExecutor, KvPool}` — two host-only traits): the CUDA
continuous-batching executor (Linux/NVIDIA, `cudarc` + vendored official
kernels FlashMLA/DeepGEMM/DeepEP + TileLang AOT + native CUDA C) and the
Metal executor (Apple Silicon, `crates/mlx-sys` C++ bridge — continuous
batching with variable-length packed decode via mlx-lm `BatchKVCache`
pattern). The same `infer_core::Engine<E, K>` drives both; adding a backend
means implementing the two seam traits, not touching scheduler/cache/server.
Models: Qwen3.5-family (CUDA + Metal), Qwen3.6 (Metal), DSv4-Flash
(CUDA 8×H20 TP=8/EP=8).

**Metal canonical model — globally unified (2026-05-07).** All Metal
backend development, benchmarking, and testing uses
`mlx-community/Qwen3.6-35B-A3B-4bit` (MoE, ~19 GB, cached at
`~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit`).

- **Why**: Qwen3.6 is the canonical Metal production target per
  [`README.md`](README.md) backend matrix and the
  [`ROADMAP.md`](ROADMAP.md) Next-Model priority queue. Benching against
  the production shape catches MoE-specific perf and correctness
  regressions that Qwen3.5-0.8B (dense) cannot surface.
- **Scope**: every Metal serve invocation (`arle serve --backend metal`;
  the legacy `metal_serve` bin is deleted), `scripts/bench_*.sh`
  default, smoke test, and `docs/experience/wins`/`errors` entry on the
  Metal track must use Qwen3.6. CUDA-side benches keep their existing
  defaults.
- **Opt-out**: Qwen3.5-0.8B-MLX-4bit and friends remain in
  `models/` for unit tests that explicitly need a small model;
  set `INFER_TEST_MODEL_PATH=models/Qwen3.5-0.8B-MLX-4bit` and document
  the reason in the test/wins entry.
- **Bench-script invocation**: `./scripts/bench_*.sh <label> --model
  mlx-community/Qwen3.6-35B-A3B-4bit` (HF id; the serve path resolves to
  the cached snapshot). Direct: `arle serve --backend metal --model-path
  mlx-community/Qwen3.6-35B-A3B-4bit`.
- **Auto-wired-limit** (always-on since
  [`2026-05-07-bench-qwen36-mle-perf.md`](docs/experience/wins/2026-05-07-bench-qwen36-mle-perf.md)):
  the rewrite Metal executor auto-pins model weights via
  `mlx::set_wired_limit` at construction
  (`infer-metal/src/wired_limit.rs`). Computes
  (model dir size + 1 GiB headroom) and follows HF cache symlinks.
  Drops c=1 p99 from 86 ms → 15 ms on Qwen3.6 (−82%). The monolith-era
  `--wired-limit-bytes` flag (and its `0` opt-out) no longer exists.
- **MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER — not a default.**
  Qwen3.5-dense-only tune; on Qwen3.6 MoE benched wash-or-loss because 95% of
  step is `mx::async_eval` encoding ~600-1000 primitives — buffer cap doesn't
  help. Per-workload matched-A/B only.
  Refs: [baseline](docs/experience/wins/2026-05-07-bench-qwen36-baseline.md),
  [encode-bottleneck](docs/experience/wins/2026-05-07-bench-qwen36-encode-bottleneck.md).

**Workspace (current, post-rewrite 2026-06-04):**

```
ARLE/
├── src/                       ← thin `arle` binary (root package `agent-infer`)
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
(adopt-official-first; hand-rolled only for the genuine gap).

Workspace topology source of truth: [`docs/codebase-map.md`](docs/codebase-map.md).

---

## Rules

### Execution phases (non-trivial tasks)

| Phase | Exit condition |
|-------|----------------|
| **Explore** (trace callers, grep prior art, list trait implementors) | You can name every file you will touch. |
| **Plan** (ask "how would this fail?" first; >5 files or irreversible → stop + flag) | Written approach the user accepted. |
| **Implement** (check prior art in `crates/infer-*/src/` + `docs/`; outside plan → update plan) | Diff compiles under the relevant feature set. |
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
  `cfg`-leak backend types into cross-backend modules** — everything above
  the seam (`infer-core`/`-server`/`-api`) stays device-neutral; backend
  types live only in `infer-cuda` / `infer-metal`.
- CUDA stubs on non-CUDA targets: `todo!("GPU required: ...")`.
- Pre-push type check on Mac without nvcc:
  `cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib`.

### Delegation (general-purpose subagents execute, Codex reviews, parallel by default)

Claude = **direction + integration**. Execution runs through **`general-purpose`
subagents** (Agent tool). Research/mapping runs through **`Explore`**; large
cross-cutting plans through **`Plan`**. Review runs through **`codex review`
at the Bash tool** — a shell command, not a subagent.

**DO NOT use `codex:codex-rescue` or `mcp__openmax__execute_with_codex` for
execution** — both hang ("codex hangs", observed 2026-04-19). See
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
  - **In scope:** `crates/infer-*/src/`, `crates/cuda-kernels/csrc/`,
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
  pod output, verify the pod tree is a git repo at HEAD and `strings target/release/arle | grep <symbol>`
  shows the change actually landed — the binary proves *some* tree was current
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
CUDA_HOME=/usr/local/cuda cargo build --release --features cuda              # CUDA (Linux+NVIDIA)
cargo build --release --no-default-features --features metal,no-cuda         # Metal (Apple Silicon)
cargo build --release --no-default-features --features cpu,no-cuda           # portable / CI smoke
# Multi-GPU features stack: nccl (implies cuda) → deepep (implies nccl)

# Mac CUDA-Rust typecheck without nvcc (CI-mirrored):
cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib

# Tests (CI-mirrored; see .github/workflows/*.yml for the full matrix):
cargo test -p agent-infer --release --no-default-features --features cpu,no-cuda,cli
cargo test -p cli --release --no-default-features --features metal,no-cuda   # Metal
cargo test -p kv-native-sys --release
```

**KV precision parity gate — re-ported 2026-06-10 (#58).** The monolith's
trajectory-match audit is superseded by the correct-inference gate
(`scripts/needle_gate.py` + `scripts/lever_gate.sh`): needle ladder
x3 same-config repeats vs the baseline envelope, NOT byte-identity (MoE
non-determinism). DSv4 lever verdicts
([wins entry](docs/experience/wins/2026-06-10-dsv4-lever-gate-license-or-kill.md)):
FlashMLA decode + fused-wqkv correctness LICENSED — default flips still need
a wall-clock perf license per the bench spec; pooled/contig-MoE flip KILLED
(-24%). Qwen dense KV-dtype matrix **resolved 2026-06-12 (#68)**: seam-level
kv-dtype dispatch landed (`--kv-cache-dtype`, default bf16 unchanged);
INT8/FP8 correctness LICENSED (needle exact 15/15 DET = BF16 envelope);
the initial decode −77% at B=1 was an uncached per-layer-per-step
`cudaGetDeviceProperties` in the quant decode shim — fixed same day
(static SM-count cache), post-fix −27% vs bf16+graph / −7% vs eager bf16
— opt-in only, no default flip without a perf license;
TQ4 DEFERRED (TurboQuant page_size=1 vs TileLang PAGE_SIZE=16). Verdicts:
[wins entry](docs/experience/wins/2026-06-12-cuda-quant-kv-dispatch-int8-fp8.md).

Env vars: `TORCH_CUDA_ARCH_LIST` (SM override, PyTorch convention; alt `CMAKE_CUDA_ARCHITECTURES`),
`INFER_TILELANG_PYTHON` (TileLang AOT Python), `INFER_TEST_MODEL_PATH`
(default `models/Qwen3.5-4B`). Full list: [`docs/environment.md`](docs/environment.md).
SM tier policy: [`docs/plans/sm-coverage.md`](docs/plans/sm-coverage.md).

Disk hygiene: `cargo sweep --time 30` (weekly) prunes target/ artifacts
older than 30 days. Dev profile already keeps deps DWARF-free (see root
`Cargo.toml` `[profile.dev.package."*"] debug = false`).

---

## Module Guides

Load the relevant `AGENTS.md` **before** editing inside a module. The
per-module guides under the old `infer/src/**` were deleted with the
monolith; for the `infer-*` rewrite crates the module truth is
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
