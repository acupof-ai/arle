# train crate systematic simplification — −4,134 LOC of pivot-orphaned dead code + single-impl traits

> Status: Shipped (4c26f1ad0, 0958e1c9f, fe342f640, c42af5fa1). Bench-exempt:
> host-side OPD-training refactor, zero serving/runtime perf surface. Verified by
> test parity (238 green) + clippy + end-to-end `--smoke` on the arle binary.

## Context

`crates/train` (31K LOC) had accreted dead scaffolding since the 2026-05-18
OPD-only pivot (pretrain/SFT/GRPO/multi-turn retired). Goal: systematically find +
delete the over-abstraction and dead paths, behavior-preserving, no serving impact.

## What Worked

**Method: workflow-mapped, evidence-gated, serial-applied.**
1. **Map** — one Workflow fanned out 8 readers over module clusters, each returning
   grep-grounded findings (caller count / identical-to location / always-same-value),
   file:line precise. 49 findings, ~4.3K LOC potential.
2. **Verify the big tickets myself** (§0 inference≠evidence) — independent grep
   corrected three agent claims before any deletion:
   - `trainer.rs`/`grad_clip.rs` are **mixed**: the generic `Trainer<O,C,S>` +
     `GradClip` trait are dead, but the free helpers (`cleanup_after_backward`,
     `extend_keep_with_params_and_grads`, `clip_grad_norm`, `compute_global_norm_f64`)
     are LOAD-BEARING in the OPD loop — surgical deletion, not whole-file.
   - `ApiTeacher`/`MultiTeacher` are NOT "never wired" — a checked-in example uses
     them → example-coupled, deferred (not clean dead code).
   - `decode_next_token` / `set_base_weight_to_fp8` are (b)-type test-only: the
     test's window into a REAL capability (LoRA sync) — deleting loses coverage →
     KEPT.
3. **Apply in 3 serial batches**, each grep-gated + built + tested + committed by
   pathspec (never `git add -A`), so a wrong "dead" claim SKIPS instead of breaking:
   - Batch 1 (−921): pivot-orphaned + uncalled surfaces (model_family synth-config,
     causal_lm dead registries, qwen35 parity-stages, tokenizer/server/swe_dataset
     dead API, metrics sink wrappers, `KlReg` always-None). 2 correct SKIPs
     (prompts fields read by examples; a loss wrapper with live test-numerics).
   - Batch 2 (−3,086): the dead generic `Trainer` subsystem + `GradAccumulator` +
     `GradClip` trait (keeping live free fns) + `MoeWithLora` + stale checkpoint
     variants. 2 correct SKIPs (ConfigJsonSource::Synthesize exercises live save
     fns via 13 tests; a loader wrapper with an out-of-crate caller).
   - Batch 3 (−127): collapsed 4 single-impl traits (`SequenceWindowedForward`,
     `TrajectoryScorer`, `TeacherWindowedForward`, `CausalLm`) into inherent/concrete
     forms — the "简化抽象" pass. Multi-impl parent `TeacherForward` left intact.

**Verification (host-side → CPU is the full gate):**
- `cargo test -p train --no-default-features --features no-cuda`: 286 → 238 green
  (delta = deleted dead-code tests), 0 failures throughout.
- `cargo clippy -p train --all-targets -- -D warnings`: clean (also cleared 2
  pre-existing test-code nits, c42af5fa1).
- End-to-end binary smoke (CPU): `arle train opd --smoke` (3 steps, ok) +
  `arle train self-opd --smoke` (loss 2.6682→2.6677, finite decreasing, EMA
  self-teacher + LoRA writeback). Colab CPU re-run byte-identical (fresh Linux).
- **CUDA-gated code verified** (Colab T4, nvcc present):
  `cargo check -p train --features cuda,no-cuda --lib --tests --examples` — all
  green in 2m (kernels stubbed by no-cuda; only the cuda-gated *Rust* compiled).
  This is the complete CUDA verification for a Rust-only host-code refactor: the
  `--examples` pass proves the collapsed `TeacherWindowedForward`/`CausalLm` and the
  deletions didn't break the example-coupled `ApiTeacher`/profiled-forward harnesses.
  No `.cu` kernel was touched, so no GPU-runtime smoke was needed to cover the change.

## Not landed — and why (the map's "dead" ≠ dead)

- **Example-coupled ~1,750 LOC is LIVE tooling, not dead code — KEPT.** qwen35
  profiled-forward tree (~900) + MoE-route diagnostics (~400) + teacher
  `ApiTeacher`/`MultiTeacher` (~457) are unreachable from `arle train` CLI, which is
  why the map flagged them "dead" — but they ARE reachable from checked-in dev
  harnesses (`opd_step_cuda_realckpt_profile.rs`, `qwen36_fp8_lora_fd_gate.rs` the
  FP8 finite-difference gradient gate, `opd_step_cuda_infer_teacher_train.rs`) all
  touched within the last month (through the 2026-07 sm_120 FP8 push). Deleting the
  code deletes maintained tooling — outside a behavior-preserving pass. The recency
  check is the SOLID gate: "unreachable from the CLI" is not "dead" when a live
  example compiles against it.
- **Hot-path extraction ~384 LOC deferred (risk > value here).** qwen35 gated-Q
  split (×5) + sparse-MLP tail (×3) + opd frozen-prompt-KV dedup only REARRANGE the
  27B forward — no dead surface removed. They need a full `test_opd_grad_check` +
  numeric A/B to license, not a smoke; the clean deletions already carried the pass.

## Rule

Pivot-orphaned dead code hides behind prominent `pub use` re-exports and big test
suites (the 693-LOC `Trainer` subsystem had 21 tests and a 6-symbol lib.rs export,
yet zero production constructor). **Grep the constructor/caller, not the export.**
A module can be MIXED — dead generic wrapper around live free helpers; delete
surgically by symbol, never by file. "Test-only" splits two ways: (a) the test
exists only to exercise the dead symbol → delete both; (b) the symbol is the test's
window into a live capability → keep it. Map with a fan-out, but verify every
big-ticket deletion yourself and apply serially with a grep-gate so a wrong "dead"
call SKIPs instead of breaking the build.
