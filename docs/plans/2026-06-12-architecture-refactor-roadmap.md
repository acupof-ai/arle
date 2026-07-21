# Architecture Refactor Roadmap — Truth Surface, N×M Model Matrix, Seam Hygiene

> **Audience: executing agents.** This doc is the steering brief for tranches
> R0–R6. Read §1 Global Guardrails before touching anything. Every tranche
> names its exact scope and non-scope; work outside a named scope is drift —
> stop and report instead.
>
> **Authority chain:** derived from
> [`../projects/2026-06-10-arle-master-strategy-v2.md`](../projects/2026-06-10-arle-master-strategy-v2.md)
> §3 (per its Rule 1). This doc commissions *structural* work only; it sets no
> perf targets and re-litigates nothing in strategy v2 §5 KILL/DEFER.
> Survey date: 2026-06-12, tree at `6b482a12` + ckl WIP (see §1.1).

## §0 Verdict and key points

The rewrite's macro shape (IR → host-only seam → device-neutral engine →
executors → serving) is correct and now proven by four backend
instantiations. The structural debts are:

1. **P1 — the model×backend matrix is growing without a single source of
   truth.** DSv4 forward-order truth exists in three forms (CUDA imperative
   code, `deepseek-spec` declarative plan consumed only by HIP, HIP doc
   comments pinning CUDA *line numbers*). Unify the **contract**, not the
   kernels — a compute-graph IR is explicitly a non-goal (§7).
2. **P2 — sibling backends are laterally coupled.** `infer-vulkan` depends on
   `infer-hip` for GGUF host substrate. Extract to a neutral leaf (R1).
   **Resolved** by `infer-gguf` (`31bf4322`).
3. **P3 — the truth surface (docs) structurally lags code.** 7 workspace
   crates are undocumented; codebase-map describes deleted seam traits.
   Manual resync loses to ~15 commits/2 days; the fix is a CI gate (R0.2),
   not another one-off resync. **Resynced** (`07948a3d`) and **gated in CI**
   (`215807e3`).
4. **Sequencing is everything.** R2 (batched lowering) **is** Phase 1 — not
   new work. R3 must wait for R2's final Phase-1 gate and a fresh line-level
   brief; R5 waits for the same gate plus clean target files. R4/R6 have
   explicit *trigger conditions*; starting them early is speculative interface
   shaping and will be reverted.

## §1 Global guardrails (binding for every tranche)

### §1.1 Files you must not touch

**Run `git status` before you start — the dirty set changes daily; never
trust a snapshot list in any doc.** Anything dirty in the working tree that
your tranche did not create is ckl WIP: do **not** edit, stage, stash, or
checkout it. If your tranche needs a dirty file, the tranche is **blocked**
— report back, do not work around. Re-check `git status` and `git log`
before committing: ckl lands divergent commits mid-session.

### §1.2 Process rules

- **Pathspec commits only**: `git commit -m "..." -- <your files>`. Never a
  bare `git commit` (sweeps staged-not-yours), never `git stash`, never
  `git checkout -- <shared file>`.
- **One tranche = one commit** (or a few small ones), verification run after
  each. Commitizen format `<type>(<scope>): <subject>`.
- **Pure moves stay pure.** No drive-by renames, no "improving" passing code,
  no public-type renames beyond what the tranche spec names, no doc edits
  beyond the named stale claims (preserve by default).
- **No half-states**: finish the tranche fully or revert it. No compat shims
  / re-export aliases left behind unless the spec names one.
- **No new abstractions beyond what this doc names.** Interfaces trail
  callers (≥2 real consumers before a trait/crate is cut).
- **Bench-entry rule applies**: any `crates/infer-*/src/` diff lands with a
  dated `docs/experience/wins/` entry (structural-refactor entries are fine —
  precedent: `2026-06-12-dsv4-spec-decode-module-refactor.md`) or a
  `pending-remote` stub; docs-only diffs state `docs-only` in the commit body.
- **No GPU/bench runs locally** unless the tranche says so (no concurrent
  Metal/large-model loads on this machine).
- Verification commands for host-only work:
  `cargo check -p <crates>`, `cargo test -p <crates>`,
  `cargo clippy -p <crates> -- -D warnings`, and on Mac the CUDA typecheck
  lane needs the cudarc nvcc bypass:
  `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib`
  (bare invocation fails on a Mac without nvcc).
- **When you land a tranche, update its Status line in this doc in the same
  commit.** A stale steering doc misroutes every later agent.

## §2 Problem inventory (evidence, survey 2026-06-12)

| # | Problem | Evidence anchors |
|---|---|---|
| P1 | DSv4 forward-order truth ×3: CUDA imperative (`infer-cuda/src/dsv4.rs`), declarative `DeepSeekV4AttentionLayerPlan` (`deepseek-spec/src/v4.rs`, consumed **only** by `infer-hip/src/model.rs`), HIP doc comments pinning CUDA line numbers ("call sites at attention.rs:4372/… on today's tree" — already rotting). §0.1 mutated-buffer enumerations also per-backend. Model files: infer-cuda{model,qwen35,dsv4}, infer-metal{qwen35}, infer-hip{model}, infer-vulkan{model_qwen3,_qwen35,_qwen36,_dsv4,_gemma4} | `grep -rn AttentionLayerPlan crates` |
| P2 | Lateral backend dep: `infer-vulkan` → `infer-hip` for `{config, dequant, gguf}`. **Resolved by R1** (`31bf4322`): both consume the neutral `infer-gguf` leaf | `crates/infer-vulkan/Cargo.toml`; `crates/infer-vulkan/src/lib.rs:8` (pre-R1) |
| P3 | Truth surface stale: codebase-map missing 7 crates (`gemma-spec`, `hip-sys`, `hip-kernels`, `infer-hip`, `infer-vulkan`, `vulkan-sys`, `vulkan-kernels`); §3.2 describes deleted seam traits (Communicator/Sampler/GraphRunner/ModelArch) and misses driven `ResourceGovernor` (`infer-core/src/lib.rs:308`, `infer-api/src/loaded.rs:641`) + `KvBatchDescriptor`; kv-native-sys "zero dependents" false (infer-cuda `kv_tier.rs`, infer-metal `MetalSsdTier`); architecture.md parity matrix lacks HIP/Vulkan; CLAUDE.md/AGENTS.md workspace tree stale; ROADMAP says ROCm enters at Phase 3 while ~10k LOC landed. **Resynced in R0.1** (`07948a3d`), **CI-gated in R0.2** (`215807e3`) | this survey |
| P4 | infer-cuda re-monolithization: ~32k LOC; `attention.rs` is ~7k lines mixing 4 concerns (MLA attention, `ModelKvAdapter` `pub(crate)` with one impl, KvBatchDescriptor lowering, FlashMLA/TileLang glue); `moe.rs` ~4.3k; `dsv4.rs` ~3.6k; per-capability per-model match arms in `executor.rs` (kv-tier: Qwen only; slot-tier: DSv4 only — coverage matrix lives nowhere) | `wc -l`, executor capability match arms |
| P5 | Seam narrative vs reality: `BackendExecutor` is 15+ methods (capability default-methods: stop ids, max rows/live, prefix reuse ×2, page-tier ×4, slot-tier ×3, weight offload ×2) while docs say "two traits, submit/poll". The default-method pattern itself is **correct** (KV tier is the model example: engine drives, backends store) — the gap is governance + docs | `infer-seam/src/lib.rs:53-198` |
| P6 | Dead/residual: `xgrammar-sys` has zero code consumers (grammar decode lost in monolith deletion); `ARLE_/INFER_` env-var call sites are moving and must be re-swept before D5 cleanup (old survey said 63; latest spot check is higher); audit plan exists (`2026-06-07-dsv4-code-cleanup-audit.md`); `EchoExecutor` duplicated (`infer-server/src/lib.rs:518`, `agent-bench/src/lib.rs:167`); ~2k lines of test mocks inline in `infer-core/src/lib.rs` | greps in this survey |
| P7 | Strategy-execution divergence: D4 strict-serial vs simultaneous kv-tier SSD + MTP top-k tree + HIP/Vulkan AIPC + Gemma4 in one week; Gemma4 absent from every priority doc | git log 2026-06-10..12 |

Healthy, do not "fix": dependency direction (zero reverse deps found); KV
tier abstraction (engine-generic with per-backend stores); `KvBatchDescriptor`
at the seam; spec_decode module extraction direction; PP gap (known,
deferred).

## §3 Ideal state (acceptance shape)

1. Adding a model = neutral spec (layer plan + KV layout + mutated-buffer
   contract) + per-backend ops it lacks. Adding a backend = op vocabulary +
   the two seam traits. Forward-**order/contract** truth lives once, in the
   spec crate; imperative code consumes or is parity-checked against it.
2. Seam = core traits + documented capability families; the
   capability×backend×model coverage matrix is a table in architecture.md,
   not folklore in match arms.
3. One batched lowering path (`KvBatchDescriptor`); the sequential
   single-row split is deleted after the batched path passes its gate.
4. Truth surface mechanically enforced: CI fails when a workspace member is
   missing from codebase-map.
5. Zero dead crates; runtime knobs are CLI flags (D5).

## §4 Tranches

### R0.1 — Truth-surface resync (docs-only) — ✅ LANDED `07948a3d` (2026-06-12)

Fix exactly these stale claims; preserve everything else:

- `docs/codebase-map.md`: §1 member list + §4 add the 7 crates with
  one-line purposes (from their `lib.rs` doc headers): `infer-hip` (DSv4
  GGUF shim-portable executor, #77), `infer-vulkan` (AIPC seam-correct
  skeleton; 5 model order pins; device pending shader ABI), `hip-sys` (thin
  hand-declared HIP FFI, stub off-box), `hip-kernels` (llama.cpp-adapted
  IQ2/Q2_K mmvq corpus, hipcc-gated), `vulkan-sys` (ash-backed loader
  wrapper), `vulkan-kernels` (glslc-compiled llama.cpp vulkan-shaders
  corpus), `gemma-spec` (Gemma4 config spec). §3.2: replace the deleted
  undriven-trait sentence with the real surface (BackendExecutor +
  capability default-methods, driven ResourceGovernor, KvBatchDescriptor,
  KvCacheDtype, resource helpers). §4: fix kv-native-sys consumers.
- `docs/architecture.md`: Package Boundaries rows + dependency-direction
  entries for the 4 new backend(-adjacent) crates + gemma-spec (mark
  `infer-vulkan → infer-hip` as DEBT, dies in R1); Backend Parity Matrix:
  add HIP + Vulkan columns (experimental / skeleton); note the
  capability-default-method seam pattern as deliberate (pointer to R6).
- `CLAUDE.md` + `AGENTS.md` (keep aligned, both full docs): workspace tree
  add the 7 crates.
- `ROADMAP.md`: factual note that the AIPC HIP/Vulkan lane started ahead of
  Phase 3 ordering (#76/#77, plan docs 2026-06-10/11) — **ratification
  pending (P7, §6)**; same note for Gemma4 (in-tree, unranked).
- `docs/index.md`: add active rows for `2026-06-10-hip-backend-mvp.md` and
  this roadmap.

Exit: every claim above matches code truth. Verify: re-run the greps in §2.
Commit: `docs: resync truth surface with workspace reality` (docs-only).

### R0.2 — CI truth-surface gate — ✅ LANDED `215807e3` (2026-06-12)

Extend `scripts/check_repo_hygiene.py` (already wired in CI job
`repo-hygiene`) with one check: parse `[workspace] members` from root
`Cargo.toml`; for each `crates/<name>`, require the literal `<name>` to
appear in `docs/codebase-map.md`; report both directions (member
undocumented / documented crate not a member). Keep it literal-string dumb —
no toml/markdown parsing dependencies. No new CI job.

Exit: gate green on resynced tree; deleting any crate line from
codebase-map makes it fail locally.
Commit: `ci: gate codebase-map against workspace members` (dev-only tooling,
bench-exempt).

### R0.3 — xgrammar-sys verdict — **BLOCKED on ckl (§6)**

Do not delete or wire anything until the verdict lands.

### R0.4 — D5 env-knob cleanup — queued, not this wave

Execute per the existing audit plan
`2026-06-07-dsv4-code-cleanup-audit.md`.
Do a fresh `ARLE_/INFER_` env-var sweep first; do **not** trust the old
63-call-site count. Classify bootstrap-legit (`INFER_TP_RANK`,
`INFER_NCCL_UNIQUE_ID`, build/toolchain) vs runtime-knob→CLI-flag. Touches
hot paths → needs its own brief + bench entries; do not fold into R0.1/R0.2.

### R1 — GGUF host substrate extraction — ✅ LANDED `31bf4322` (2026-06-12; codex review: no blocking issues)

**Goal:** kill the `infer-vulkan → infer-hip` lateral dep by extracting the
GGUF host substrate to a neutral leaf crate `crates/infer-gguf`.

**Spec (follow exactly):**

1. New crate `crates/infer-gguf` (workspace member; deps: `anyhow`,
   `memmap2` (workspace), `deepseek-spec`; **no features**; everything
   always-compiled — these modules are host-only and not hip-gated today).
2. Move from `crates/infer-hip/src/`:
   - `gguf.rs` → `infer-gguf/src/gguf.rs` (format-generic GGUF v2/v3 reader)
   - `dequant.rs` → `infer-gguf/src/dequant.rs` (format-generic GGML CPU
     dequantizers)
   - `config.rs` → `infer-gguf/src/deepseek4.rs` (**rename**: this module is
     the deepseek4-arch GGUF → `DeepSeekV4Config` mapper, not generic
     config; file naming must match content semantics). Keep contents
     byte-identical apart from the module path; in-file doc comments stay.
3. `infer-gguf/src/lib.rs`: `pub mod gguf; pub mod dequant; pub mod
   deepseek4;` + crate doc header stating ownership (GGUF container reading,
   CPU dequant, per-arch GGUF→spec-config mappers; model *forward* code
   stays in backends). Spec crates stay dependency-pure: per-arch mappers
   live here (depending on the spec crate), never in `*-spec`.
4. `infer-hip`: drop the three modules; update the ~9 internal
   `crate::{config,dequant,gguf}` use sites to `infer_gguf::{deepseek4,
   dequant, gguf}`; drop any now-unused deps from its Cargo.toml
   (`memmap2` moves out if nothing else uses it); **no re-export shim** —
   `infer-hip` stops exporting `config/dequant/gguf`.
5. `infer-vulkan`: replace dep on `infer-hip` with `infer-gguf`
   (`lib.rs:8` re-export becomes `pub use infer_gguf::{deepseek4, dequant,
   gguf};` and downstream module references updated; if its public API
   re-exported `config`, the new public name is `deepseek4`).
6. Docs ride along: codebase-map + architecture.md gain the crate row
   (R0.2's CI gate will enforce); remove the DEBT marker added in R0.1.
7. Wins entry: `docs/experience/wins/2026-06-12-infer-gguf-extraction.md` —
   pure-move refactor entry (Context / What Worked / Rule), records the
   verification matrix below; no perf claim.

**Non-scope:** no changes to `executor.rs`/`model.rs`/`loader.rs` logic in
either backend beyond `use` paths; no touching `infer-metal`/`mlx-sys` GGUF
(C++-side, unrelated); no API "improvements".

**Verification:** `cargo check -p infer-gguf -p infer-hip -p infer-vulkan`
+ `cargo test -p infer-gguf -p infer-hip -p infer-vulkan` (host, default
features) + `cargo clippy -p infer-gguf -p infer-hip -p infer-vulkan -- -D warnings`
+ `python3 scripts/check_repo_hygiene.py`. Behavior must be byte-identical
(pure move; the diff is paths + Cargo plumbing only).

Exit: `grep -rn "infer-hip" crates/infer-vulkan/Cargo.toml` is empty; all
checks green.
Commit: `refactor(gguf): extract GGUF host substrate from infer-hip to infer-gguf`.

### R2 — Batched lowering closeout (= Phase 1, the keystone)

This **is** [`2026-06-07-unified-batched-kvpool-abstraction.md`](2026-06-07-unified-batched-kvpool-abstraction.md)
(authoritative; already ACTIVE as #60/#61). The only addition from this
roadmap: when the batched path passes the c-sweep gate (TTFT+ITL+tok/s per
bench spec), delete or demote the sequential single-row fallback in
`Dsv4Executor::forward_decode_batch` plus the mixed/multi-prefill split in
`Dsv4Executor::submit` (`submit_prefill_row` loop + decode sub-batch) to an
explicit named A/B arm — no silent parallel old+new paths. Do not key this work
off stale line numbers; find the functions by name in the current tree.

### R3 — DSv4 forward-order single truth — **blocked until R2 final gate + brief**

**Goal:** make `deepseek_spec::v4::DeepSeekV4AttentionLayerPlan` the single
forward-order authority for DSv4.

Approach (parity-test-first, no behavior change):

1. Add a host-only parity test in `infer-cuda` that derives the
   launcher-order table from `dsv4.rs` **as data** and asserts it against
   the spec plan walk. (Detailed line-level brief to be written after R2's
   final c-sweep/cleanup gate; do not start from this high-level sketch.)
2. Replace `infer-hip/src/model.rs` line-number doc pins with spec-plan
   references; same for `infer-vulkan/src/model_dsv4.rs`.
3. Lift the §0.1 mutated-buffer enumeration (HIP's
   `ATTENTION_LAUNCHER_WRITES` pattern) to spec-level data each backend
   proves against.

Exit: one place defines DSv4 forward order; line-number pins deleted;
parity test green; needle gate (not byte-identity — MoE non-determinism)
on the pod for any code that moved.

### R4 — second `ModelKvAdapter` + trait graduation — trigger: Qwen3.6 CUDA (Phase 3)

`ModelKvAdapter` stays `pub(crate)` in `attention.rs` until the Qwen3.6
CUDA adapter (its second impl) exists. Then: move the trait + both impls to
their own module; acceptance = Qwen3.6 CUDA serving on the shared lowering
with **zero `infer-core` changes**. Lifting earlier = speculative shaping;
don't.

### R5 — infer-cuda file splits — trigger: after #70 + R2 final gate

Pure-move `#[path]` flat splits (per repo convention, no `mod.rs`):
`attention.rs` → `attention/{mla,flashmla,adapter,batch}.rs`-shaped
siblings; then `moe.rs`, `dsv4.rs`. One file per tranche, each compiling +
tests green + committed separately. Soft bar: no file >3k LOC. Before each
split, `git status` must show the target file is clean; if not, §1.1 blocks it.

### R6 — seam capability governance — trigger-only, do nothing now

Trigger: `BackendExecutor` exceeds ~20 methods **or** a new stateful capability
family lands beyond the current documented set. Current set: scalar metadata
(`model_stop_token_ids`, row/live caps), prefix/page-tier hooks, whole-slot
tier hooks, and weight-offload hooks. If the trigger fires, regroup into
capability traits (KvTierStore / SlotTierStore / WeightOffload / new family).
Until then the default-method pattern is the documented, deliberate choice
(R0.1 notes it in architecture.md).

## §5 Sequencing and TODO

```
R0.1 ✅ ──▶ R0.2 ✅     (landed 2026-06-12: 07948a3d, 215807e3)
R1 ✅                   (landed 2026-06-12: 31bf4322)
R2 (= Phase 1, active) ──▶ R3 ──▶ R4 (= Phase 3 Qwen3.6 item)
                       └─▶ R5 (opportunistic, after WIP lands)
R6: trigger-only.   R0.3: blocked on ckl.   R0.4: queued, own brief.
```

### Remaining TODOs

| Item | Status | Next action | Stop condition |
|---|---|---|---|
| R2 Phase 1 batched lowering closeout | ACTIVE | Finish Phase 4-6 in `2026-06-07-unified-batched-kvpool-abstraction.md`; run c-sweep gate; then delete or explicitly name the single-row fallback arms | TTFT/ITL/tok/s gate not passed, or target files dirty |
| R3 DSv4 forward-order single truth | BLOCKED | After R2 gate, write a line-level brief for spec-plan parity tests + mutated-buffer table lift | R2 incomplete, no brief, or pod needle gate unavailable |
| R4 `ModelKvAdapter` trait graduation | TRIGGER-ONLY | Wait for Qwen3.6 CUDA second adapter | No second impl; lifting now is speculative |
| R5 infer-cuda file splits | TRIGGER-ONLY | After R2/#70 and clean target files, do pure `#[path]` splits one file per commit | Any non-move behavior change or dirty target file |
| R6 seam capability governance | TRIGGER-ONLY | Keep default-method pattern until method/family trigger fires | No new stateful capability family and method count below threshold |
| R0.3 xgrammar-sys verdict | BLOCKED on ckl | Decide re-port vs remove from workspace | No owner verdict |
| R0.4 D5 env-knob cleanup | QUEUED | Fresh env-var sweep, classify bootstrap vs runtime knobs, write separate brief | Touches hot path without bench-entry plan |
| P7 ratification | BLOCKED on ckl | Amend strategy v2 for AIPC/Gemma4 or re-serialize priorities | No strategy verdict |

## §6 Pending ckl verdicts

1. **xgrammar-sys**: re-port grammar-constrained decode onto the rewrite
   stack (schedule it) or move the crate out of the workspace. No silent
   keep.
2. **P7 ratification**: amend strategy v2 to admit the early-started AIPC
   lane + Gemma4 into the priority queue, or re-serialize. Until then docs
   carry the factual "ratification pending" note from R0.1.

## §7 Explicit non-goals (do not let scope creep here)

- **No compute-graph IR / no cross-device single-source forward kernels.**
  Industry-normal (vLLM/SGLang) is per-backend model code; we unify
  contracts (order, KV layout, buffer enumeration), not kernels.
- No scheduler re-coupling to any backend (the regression PR #53 exists to
  prevent).
- No FlashInfer migration, no tiered-KV readmission, no Qwen3.5 Medusa —
  strategy v2 §5 DEFER/KILL stands.
- No `infer-ops` crate, no `*-sys` splits of `cuda-kernels` (architecture.md
  anti-goals stand).
- No perf claims from any structural tranche; perf work stays
  license-or-kill under the bench spec.
