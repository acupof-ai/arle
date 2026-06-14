# SOPD #91 keystone — `arle train self-opd` inline EMA self-update loop

**Date**: 2026-06-14 (V100 verification 2026-06-15) · **Issue**: [#91](https://github.com/cklxx/arle/issues/91) (SOPD Phase-0 KEYSTONE) · **Status**: F (CLI subcommand + inline loop + held-out NLL gate) landed `31939d23`; EMA self-teacher core landed `06e69e22`/`3e372ac1`/`89f3101d`; **all 4 keystone loop-mechanic checks verified on real Qwen3.5-0.8B (V100 CPU autograd, 2026-06-15)** · **Bench**: needle-ladder *serving* gate stays `pending-remote` (no serving endpoint inside the inline loop)

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
overridden at the F call site) **and hard-rejects `--gkd-lambda ≤ 0` (incl. NaN)
at dispatch** so the no-op can't be reached even by an explicit flag (codex P2,
`0b591c78`). The smoke run confirms gradients flow at λ=0.5:
`arle train self-opd --smoke --steps 5` → loss `0.164029 → 0.163940`
monotonically decreasing, exit 0; `--gkd-lambda 0` is rejected with a cold-start
error, exit 1.

### Non-circular in-loop gate — held-out NLL on fixed reference text

KL(student‖EMA) is **circular** as a no-regression signal — EMA tracks the
student, so the KL shrinks regardless of capability. The honest in-process
signal is **held-out NLL** (forward-only mean next-token CE) on a *fixed*
`--eval-ids` reference sequence (defaults to the prompt), measured with the
tape off and its scratch dropped. `heldout_nll` runs every `--gate-every-n`
steps; if NLL regresses past `--gate-regress-tol` (0.02), the step is reverted.
A **non-finite gate NLL is treated as a regression → revert** (a `NaN > x`
comparison is false, so without the guard the accept branch would store NaN as
the baseline and permanently disable the gate; a non-finite *initial* baseline
hard-errors — codex P2, `0b591c78`).
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

### V100 verification — all 4 loop-mechanic checks on real Qwen3.5-0.8B (2026-06-15)

Ran `arle train self-opd` end-to-end on the V100 box against the real
`Qwen/Qwen3.5-0.8B-Base` HF dir (vocab=248320, hidden=1024, 24 layers,
`backend=cpu` — self-opd is pure autograd, so the GPU is irrelevant; the V100's
value is real weights + RAM + build env). Three controlled runs, all exit 0.
Rollout prompt was a **non-degenerate real-text sentence** ("The capital of
France is Paris, …", 18 tokens, encoded on the Mac's local Qwen3.5 tokenizer);
the gate `--eval-ids` was a **distinct** held-out sentence ("Water boils at one
hundred degrees Celsius, …", 14 tokens) — non-circular by construction.

| Check | Run | Flags | Result |
|-------|-----|-------|--------|
| **1 — loop runs e2e** | A′ | `--gate-every-n 0` | ✓ 3 steps, exit 0, reverts=0 |
| **2 — no-regression on the honest metric** | B | `--gate-every-n 1 --gate-regress-tol 0.02` | ✓ held-out NLL **decreases** 2.477912→2.477247→2.476554→2.475835 (−0.0021) |
| **3 — gate computes finite baseline + accepts** | B | (same) | ✓ baseline 2.477912 finite on the *distinct* held-out text; all 3 steps `gate accept`, reverts=0 |
| **4 — atomic rollback restores a clean unit** | C | `--gate-every-n 1 --gate-regress-tol=-1.0` | ✓ reverts=3; post-step NLL **byte-identical 2.477247 across all 3 reverts**, baseline frozen at 2.477912; loop completes exit 0 |

**Check 4 is the strongest signal.** `tol=-1.0` ⇒ threshold = `baseline·(1−1) = 0`
⇒ every positive NLL regresses ⇒ forced revert each step. The post-step NLL is
**identical (2.477247) on all 3 reverted steps** — each step starts from the
exact same restored state, applies the same deterministic update, lands the same
NLL, reverts again. Contrast Run B (no revert): NLL *progressed* 2.477247 →
2.476554 → 2.475835. If the rollback were **partial** (AdamW moments or EMA
adapter left drifted — the DSv4-EAGLE failure mode the atomic-unit design guards
against), steps 2/3 would diverge from step 1's 2.477247. They don't → the
`{student adapter, EMA adapter, AdamW moments}` triple restores as one clean unit.

### SOLID finding — cold-start training loss is structurally near-zero (greedy-rollout anchor)

The per-step **training loss is ~8e-6** on the real-text prompt, *not* a
degenerate-prompt artifact — it is fundamental to the current wiring. The rollout
is **greedy-argmax** (`opd.rs:342` "Greedy-argmax the last-position row"; module
doc "the student samples a rollout greedily"), and the cold-start anchor is
`GkdSftAnchor::StudentRollout` ⇒ the loss = `0.5·CE(student ‖ own greedy tokens) +
0.5·KL(student ‖ EMA)`. At cold start KL=0 (EMA==student), and CE against the
student's **own argmax** tokens is `−log p(argmax) ≈ 0` (peaked softmax). The
only gradient is "sharpen confidence on tokens you already pick" — a self-
reinforcing, near-zero bootstrap, **not capability learning**. It does generalize
weakly (held-out NLL improves −0.0021 in Run B), but a meaningful Phase-0
bootstrap needs either a **temperature-sampled rollout** (anchor tokens ≠ argmax
⇒ CE > 0) or a **real held-out corpus anchor** (`GkdSftAnchor` with
`corpus_tokens`). This is the honest license-or-kill on Phase-0 *learning*
efficacy — the keystone here is the loop *infrastructure* (4-pass · gate · atomic
rollback), which is fully verified; the bootstrap-signal redesign is a follow-up.

### Remaining `pending-remote` — needle-ladder serving gate

The needle ladder (`scripts/needle_gate.py`) needs a serving endpoint, which
does not exist inside the inline loop, so it stays the **external** pod
acceptance gate for end-to-end capability no-regression (run after a real
multi-step SOPD session writes a merged adapter). Cross-link:
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
