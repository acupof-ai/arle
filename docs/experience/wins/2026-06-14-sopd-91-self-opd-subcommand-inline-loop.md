# SOPD #91 keystone — `arle train self-opd` inline EMA self-update loop

**Date**: 2026-06-14 · **Issue**: [#91](https://github.com/cklxx/arle/issues/91) (SOPD Phase-0 KEYSTONE) · **Status**: F (CLI subcommand + inline loop + held-out NLL gate) landed `31939d23`; EMA self-teacher core landed `06e69e22`/`3e372ac1`/`89f3101d` · **Bench**: `pending-remote` — on-pod needle no-regression gate (see below)

## Context

SOPD Phase 0 (#91) is the **inline G2 A1-EMA self-update loop**: the model
self-updates its own rank-r LoRA adapter at rollout time. The 4-pass-per-rollout
contract: (1) rollout decode [eager, tape off], (2) teacher score [base + EMA
adapter, tape off], (3) student recompute [base + student adapter, tape ON],
(4) KL + bwd + AdamW + EMA update [adapter-only]. The EMA self-teacher is a
second `Qwen35Model` sharing the student's **frozen base** (via
`new_lora_from_base` / `share_base_parameters_from`) with its own EMA adapter,
consumed as a `TeacherForward` by reusing `InProcessTeacher`.

F is the `arle train self-opd` standalone subcommand (ckl-selected CLI form)
wiring that loop end-to-end: load LoRA student from an HF dir → build
`EmaSelfTeacher::from_student` **before** any other scratch (its internal
`retain_ids` frees everything else) → loop `{ teacher block then ema.update }`
→ gate every N steps `{ held-out NLL → revert via ema.restore or accept →
re-snapshot }`.

## What Worked

### Cold-start needs λ>0 — pure-KL A1-EMA is dead on arrival (SOLID)

`lora_b` inits to 0 → student == EMA == base → KL(student‖EMA) == 0 → **zero
gradient**. The bootstrap gradient must come from a λ>0 GKD CE self-anchor on
the rollouts (`GkdSftAnchor::StudentRollout`); the EMA-KL term is the
*steady-state stabilizer*, not the bootstrap. F defaults `--gkd-lambda 0.5`
(the `opd.rs` `GkdLossConfig::default` λ=0.0 is wrong for cold start and is
overridden at the F call site). The smoke run confirms gradients flow at λ=0.5:
`arle train self-opd --smoke --steps 5` → loss `0.164029 → 0.163940`
monotonically decreasing, exit 0.

### Non-circular in-loop gate — held-out NLL on fixed reference text

KL(student‖EMA) is **circular** as a no-regression signal — EMA tracks the
student, so the KL shrinks regardless of capability. The honest in-process
signal is **held-out NLL** (forward-only mean next-token CE) on a *fixed*
`--eval-ids` reference sequence (defaults to the prompt), measured with the
tape off and its scratch dropped. `heldout_nll` runs every `--gate-every-n`
steps; if NLL regresses past `--gate-regress-tol` (0.02), the step is reverted.
The needle ladder (`scripts/needle_gate.py`, needs serving) stays the
**external** pod acceptance gate — no serving endpoint exists inside the inline
loop, so the in-loop gate had to be self-contained.

### Atomic rollback — {student adapter, EMA adapter, AdamW moments} as ONE unit

Partial rollback poisons the EMA (DSv4-EAGLE partial-rollback lesson). The EMA
core (`89f3101d` round-2 fix) makes `EmaSelfTeacher::restore` revert all three
buffers together from one `EmaTrainSnapshot`, and the round-1 fix (`06e69e22`)
freezes the EMA adapter + clears AdamW state on rollback. The teacher-param set
excludes the trainable student adapter (`share_base_parameters_from` folds it
into the EMA param ids — round-2 P1 root cause at `qwen35.rs:1676` wholesale
`param_ids` copy). 3 EMA-core regression tests green.

### Local verification (Mac CPU, no CUDA)

- `cargo check -p cli --release --no-default-features --features cpu,no-cuda` — PASS
- Mac CUDA-Rust typecheck (`cuda,no-cuda` lane) — PASS (the `#[cfg(feature = "cuda")]`
  `infer_rollout` arg compiles in both lanes)
- `cargo clippy -- -D warnings` — clean
- smoke loop — PASS (5 step lines, loss decreasing, exit 0; gate skipped in smoke)

F is exactly `crates/cli/src/args.rs` + `crates/cli/src/train_cli.rs` (+463/−1),
reusing the existing `build_opd_store` / `parse_prompt_ids` / `current_grad_norm`
/ `embedded_tiny_qwen35_config` / `exit_from_result` helpers — no new infra.

## Bench (`pending-remote`)

The keystone acceptance gate is the **on-pod needle no-regression** check, which
needs (a) a real Qwen3.5 HF dir (not loadable on the Mac CPU box) and (b) a
serving endpoint for `scripts/needle_gate.py`. The remote run must confirm, on
the H20 pod, all four: the loop runs end-to-end · the EMA-KL drives down ·
held-out NLL does not regress · the atomic rollback restores a clean unit. The
smoke lane is the only locally-executable surface and it passes. Cross-link:
[SOPD plan](../../plans/2026-06-14-self-training-lora-opd-sopd.md) ·
[#92 prefix-cache invalidate primitive](2026-06-14-sopd-prefix-cache-invalidate-primitive.md).

## Rule

A self-distillation loop whose teacher == an EMA of the student has **zero
gradient at cold start under pure KL** (identical adapters ⇒ KL=0) — the
bootstrap must come from a λ>0 CE self-anchor, and the no-regression gate must
be a non-circular external signal (held-out NLL on fixed text), never the
self-tracking KL. Build the EMA self-teacher before any other scratch (its
`retain_ids` frees the rest), and rollback the {student adapter, EMA adapter,
optimizer moments} triple as one unit or the EMA is silently poisoned.
