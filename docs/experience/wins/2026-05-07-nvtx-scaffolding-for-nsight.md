# 2026-05-07 · NVTX scaffolding for Nsight Systems profiling

## Goal

- M3.6 Phase 1 prerequisite: add NVTX named ranges around scheduler
  hot paths so `nsys` traces are interpretable per
  [`docs/plans/M3.6-nsight-trace-analysis-runbook.md`](../../plans/M3.6-nsight-trace-analysis-runbook.md)
  (especially the §"Step 2 — NVTX phase breakdown" table). Without
  named ranges, `nsys stats --report nvtx_sum` returns nothing and
  trace analysis falls back to kernel-name heuristics only.

## What Worked

Codex authored the scaffolding in `998bfee` (bundled commit — see
"Note on commit attribution" below). Three-piece design:

1. **Header-only NVTX v3 binding** via tiny C shim
   (`infer/src/scheduler/cuda/nvtx_scopes.c`, 9 lines):
   ```c
   #include <nvtx3/nvToolsExt.h>
   void arle_nvtx_range_push(const char *message) { nvtxRangePushA(message); }
   void arle_nvtx_range_pop(void)                 { nvtxRangePop(); }
   ```
   Avoids `dlopen` / `libloading` paths entirely. NVTX v3 is
   header-only and self-resolves the profiler injection at runtime —
   no-op when no profiler is attached.

2. **RAII Rust guard + macro** (`nvtx_scopes.rs`, 37 lines):
   ```rust
   pub(in crate::scheduler::cuda) struct NvtxScope;
   macro_rules! nvtx_scope {
       ($name:literal) => {
           let _nvtx_scope = $crate::scheduler::cuda::nvtx_scopes::NvtxScope::push(
               concat!($name, "\0").as_ptr().cast(),
           );
       };
   }
   ```
   The `concat!($name, "\0")` machinery hands NVTX a null-terminated
   string at compile time — zero runtime allocation, zero
   format-string risk. `Drop for NvtxScope` calls `nvtxRangePop` so
   panic + early-return paths still pop.

3. **`infer/build.rs`** (NEW, 61 lines) compiles the C shim through
   `cc` only when the `cuda` feature is on; ships nothing on Metal
   builds (verified by codex: `cargo check --features metal,no-cuda`
   passes, no `nvToolsExt` symbols leak).

NVTX call sites placed at the 7 anchors named in the M3.6 plan:
`step_total`, `step_admission`, `step_plan`,
`step_mixed_launch_retract`, `step_decode_kernel_launch`,
`step_prefill_kernel_launch`, `step_dispatch_emits`. These match
the runbook's "Step 2" expected-share table verbatim, so trace
reading needs no translation step.

## Note on commit attribution

The scaffolding landed inside `998bfee` whose commit message only
references `scripts/vllm_serve_control.sh`. Reason: codex had
`git add`-ed the NVTX files before invoking
`codex review --uncommitted`, intending to commit after review.
Claude (this session) ran `git add scripts/vllm_serve_control.sh`
followed by `git commit` without checking `git status` first — the
already-staged NVTX files joined the commit. The commit is correct
in content; only the message under-describes it.

This entry serves as the durable changelog for the NVTX work.
Future reviewers tracing "where did NVTX support land" should follow
the link from M3.6 Phase 1 → this entry → `998bfee` files.

## Verification (codex pre-commit)

- `cargo fmt --all --check` — clean.
- `cargo check --release -p infer --no-default-features --features no-cuda` — clean.
- `cargo check --release -p infer --no-default-features --features cuda,no-cuda` — clean.
- `cargo check --release -p infer --no-default-features --features metal,no-cuda` — clean (no NVTX symbol leak).
- `cargo check --release -p infer --no-default-features --features cuda,metal,no-cuda` — clean.
- `NVCC_CCBIN=/usr/bin/g++-14 ... cargo check --release -p infer --features cuda` — clean.
- `cargo clippy --release ... -- -D warnings` (cuda + no-cuda + metal,no-cuda variants) — clean.
- `codex review --uncommitted` — "No staged, unstaged, or untracked changes are present in the workspace, so there is no patch to review and no introduced issues to flag" (review fired post-commit so saw an empty diff).

## Bench Status

No bench. NVTX call sites are no-op when no profiler is attached
(NVTX v3 lazy-resolves the injection); zero runtime cost in
production. Per CLAUDE.md §Benchmarks "regression-check minimum"
applies after the first nsys-driven optimization lands, not for
this scaffolding-only change.

## Rule

- When committing a small targeted file, **always run `git status`
  first** to spot staged work from other agents (codex via tmux,
  pre-commit hooks, prior partial work). `git add <path>` then
  `git commit` is NOT scope-limiting — it commits the union of
  whatever is currently staged.
- Cross-agent staging conflicts during cooperative work need either
  (a) coordination via tmux ("hold off committing — I'll bundle
  yours"), or (b) `git stash` of one party's work — but `git stash`
  is forbidden by `feedback_no_git_stash_unrelated.md`, so (a) is
  the only path.
- If a wrong-message commit ships, prefer a follow-up changelog
  entry (this file) over `git commit --amend` + `push --force`.
  Force-push to main is destructive on shared history; an honest
  follow-up doc preserves the durable record.
