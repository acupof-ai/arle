# Project — OPD-only training pivot

**Status**: Active · **Decision date**: 2026-05-18 · **Driver**: ckl

## The decision

**ARLE training surface narrows to On-Policy Distillation (OPD) only.**
All other training surfaces — scratch pretrain, supervised fine-tune,
GRPO, multi-turn RL — are deleted. The train crate becomes an
OPD-focused, runtime-led teacher-student loop.

## Why now (evidence-based)

1. **Industry pretrain gap is 322×.** Today's `nanochat d12 / RTX 4070
   Ti SUPER` measurement (`docs/experience/wins/2026-05-18-bench-pretrain-nanochat-baseline-d12.md`)
   pinned the industry single-GPU pretrain throughput at **56 291 tok/s**;
   ARLE post-Wave-2.x is at **174.7 tok/s** — a 322× as-system gap. Even
   the `≥ industry × 1.3` target (73 178 tok/s) requires a ~419×
   multiplier. Single-node from-scratch pretrain is not a winnable
   product axis for ARLE.

2. **GRPO / SFT / multi-turn duplicate existing OSS.** vLLM + verl,
   SGLang + slime, HuggingFace TRL, axolotl, and unsloth already
   carry mature implementations. ARLE shipping a thin Rust
   re-implementation of the same algorithms adds no differentiation;
   the ~19 343 LOC across the four commands (`pretrain.rs` 1 807,
   `train_grpo.rs` 1 694, `train_multi_turn.rs` 1 740, `train_sft.rs`
   1 890) is product surface area without product positioning.

3. **OPD is the runtime-led training axis.** On-policy distillation
   needs (a) a strong inference runtime to host the teacher and (b)
   tight latency between student sampling and teacher scoring. ARLE
   has both: the `infer` crate is the live inference path; the
   train crate's job is the per-token KL loop on top of that. This is
   the only training surface where ARLE's pure-Rust runtime authority
   is a structural advantage, not a duplicate.

4. **Sustains the "agent / RL / self-evolving" project narrative.**
   `docs/projects/agent-rl-self-evolving.md` already positions train
   as runtime-led, not a second product. OPD-only collapses that to a
   single concrete deliverable — small student distilled from a large
   teacher running in `infer`, hot-swappable adapter just like the
   existing GRPO design promised but with a less duplicative algorithm.

## Scope

### What gets deleted

| Component | LOC | Rationale |
|---|---|---|
| `crates/train/src/commands/pretrain.rs` | 1 807 | From-scratch pretrain, 322× behind industry |
| `crates/train/src/commands/pretrain_dsv4.rs` | ~600 | DSv4-specific pretrain bootstrap, moot |
| `crates/train/src/commands/train_sft.rs` | 1 890 | Generic supervised fine-tune, OSS duplicate |
| `crates/train/src/commands/train_grpo.rs` | 1 694 | GRPO, OSS duplicate |
| `crates/train/src/commands/train_multi_turn.rs` | 1 740 | Multi-turn RL, OSS duplicate |
| `crates/train/src/commands/eval_lm.rs` | ~700 | Generic eval, will be reborn as OPD-side eval |
| `crates/train/src/commands/convert_dataset.rs` | — | Dataset prep for pretrain corpus |
| `crates/train/src/commands/download_dataset.rs` | — | Dataset prep for pretrain corpus |
| `crates/train/src/{grpo,policy,policy_support,rollout,reward,verifier,multi_turn,task_gen,sft_data,curriculum}.rs` | — | RL/SFT supporting algorithms |
| `crates/train/src/{dataset,data_adapter,hub_dataset}.rs` | — | Generic dataset adapters |
| `crates/train/src/{qwen3,qwen3_checkpoint}.rs` | — | Already targeted for retirement (Qwen3 → Qwen3.5-only) |
| `crates/train/tests/test_{grpo,sft*,multi_turn,curriculum,convergence_smoke,qwen3*,qwen_lora}.rs` | ~5 000 | Tests for deleted commands |
| `crates/train/examples/build_*_tokenizer.rs` | — | Pretrain-corpus tokenizer training |

### What stays (still load-bearing for OPD)

- `crates/autograd/` — entire crate; OPD needs forward + backward
- `crates/train/src/`:
  - `lib.rs` (slimmed)
  - `causal_lm.rs` — model trait
  - `qwen35.rs` / `qwen35_checkpoint.rs` — model arch + HF load
  - `trainer.rs` — `Trainer<O, C, S>` skeleton (still applies to OPD step)
  - `checkpoint.rs` — HF dir layout, student-checkpoint write
  - `cli_args.rs` — shared arg helpers (slim)
  - `control.rs` + `server.rs` — `/v1/train/*` control plane (still wanted for OPD progress)
  - `metrics.rs` — `MetricSink` + SharedSink + lifecycle events
  - `tokenizer.rs` — load tokenizer (BPE / WordLevel)
  - `grad_accum.rs` + `grad_clip.rs` — generic, reused
  - `loss.rs` — keep CE primitive; OPD will add `kl_divergence` here
  - `lora.rs` — LoRA on student (OPD's primary student-update mode)
  - `model_family.rs` — slimmed to Qwen3.5-only
  - `commands/` — `env.rs` + `test.rs` + `estimate-memory.rs` kept; everything else deleted

### What gets added (separate sub-project, not this commit)

A new `crates/train/src/commands/train_opd.rs` + `crates/train/src/opd.rs`
substrate:
- Teacher loader (frozen weights, calls into `infer` for forward)
- Student model (Qwen3.5 family, smaller config, optionally LoRA-only)
- Per-step loop:
  1. Sample N tokens from student (on-policy rollout)
  2. For each sampled position, run teacher forward to get teacher logits
  3. KL loss between student and teacher distributions
  4. Backward through student (LoRA params if LoRA-only)
  5. AdamW step on student
- Reuses `Trainer<O, C, S>` from this crate, just provides a different `step_fn`
- `arle train opd --teacher <path> --student <path|preset> --corpus <path>`

This add-substrate work is scoped as **separate from the deletion
tranche**. The deletion lands first; OPD substrate is the next project.

## Out of scope for this pivot

- **Inference path** (`infer` crate, `arle` CLI's `serve` / `run`
  surfaces) — unchanged. Continues as before.
- **Autograd** — unchanged. Wave Σ structural pivot remains the next
  autograd milestone.
- **Hot CUDA-kernel work** — Wave 2.x backward kernels remain in tree
  as prerequisite infrastructure for both Wave Σ and OPD (OPD needs
  the device-resident gradient path just as much as any other training).

## Execution plan (this session)

### Tranche A — Strategic alignment (Claude)

1. This project doc (`docs/projects/2026-05-18-opd-only-pivot.md`) commits first.
2. Update `CLAUDE.md` / `AGENTS.md` to declare OPD focus + remove
   mentions of pretrain/SFT/GRPO/multi-turn as supported train
   surfaces; add OPD as the one supported surface (pending substrate).
3. Update `ROADMAP.md` to reflect the train-surface narrowing.

### Tranche B — Documentation rewrite (Claude)

Update each canonical doc to remove deleted-training references and
declare OPD-only positioning:

- `README.md` — "Status at a glance" + "Models / Backends" tables
- `docs/architecture.md` — train crate description
- `docs/codebase-map.md` — train crate contents
- `docs/support-matrix.md` — §5a Training Surface Matrix (replace 5 rows with 1 OPD row + "OPD pending substrate" status)
- `docs/comparison.md` — drop "Same runtime, in-tree pretrain/sft/grpo/multi-turn/eval" → "Same runtime, in-tree OPD"
- `docs/install.md` — remove pretrain-corpus quickstart
- `docs/troubleshooting.md` — remove pretrain/SFT-specific errors
- `docs/environment.md` — remove pretrain-corpus env vars
- `docs/plans/train-runtime-architecture-v1.md` — rewrite to declare OPD-only
- `docs/projects/agent-rl-self-evolving.md` — collapse GRPO milestones into OPD milestones
- Other active plan/project docs as needed

### Tranche C — Code deletion (Codex)

Delegated to Codex via tmux session `0:1`. Codex's job:

1. Delete the files listed in §Scope→What gets deleted.
2. Trim `crates/train/src/lib.rs` `pub mod` declarations + `pub use`
   re-exports so the crate compiles after deletion.
3. Trim `crates/cli/src/train_cli.rs` (or wherever the CLI dispatch
   lives) so `arle train pretrain|sft|grpo|multi-turn|eval|...`
   subcommands are gone; only `env / test / estimate-memory` remain
   from the existing surface (and a stub for `opd` that errors with
   "OPD substrate landing next milestone").
4. Update workspace `Cargo.toml` / `crates/train/Cargo.toml` if any
   deleted-module-only dependency drops out.
5. `cargo check --workspace` + `cargo test -p train --release` green at
   each commit.
6. One commit per deletion tranche; do not batch everything into one
   monolith.

Codex receives this project doc as its directive. Build-green is the
binding gate.

### Tranche D — Verify + commit-push (Claude)

After Codex reports, Claude:

1. Pulls Codex's deletion commits into local view.
2. Re-builds `arle` (CPU + CUDA) to confirm doc claims match reality.
3. Pushes the unified `main` to `origin`.

## SOLID gates per tranche

- **Tranche A** — `cargo check --workspace` still green (only doc
  changes, should be trivially true). Project doc cites this doc.
- **Tranche B** — `cargo check --workspace` still green. All docs
  scrub-tested for stale `pretrain` / `SFT` / `GRPO` mentions outside
  of `docs/experience/{wins,errors}/` (those stay immutable per
  bench-and-trace-spec §9).
- **Tranche C** — `cargo build --workspace --release` green at each
  Codex commit. Train tests subset green
  (`cargo test -p train --release` for whatever survives).
- **Tranche D** — Branch pushed. New `arle train --help` shows only
  `env / test / estimate-memory / opd` (opd as stub).

## Risk register

| Risk | Mitigation |
|---|---|
| Codex over-deletes (touches OPD-needed support like `trainer.rs`) | This doc enumerates "What stays" explicitly; Codex's brief cites this list as the keep-set |
| Doc claims drift from code reality (claim OPD-only but pretrain code still ships) | Tranche D binding gate checks `arle train --help` output |
| Some hidden caller of deleted code in `infer` or `agent` | `cargo build --workspace` catches it; deletion tranche stops on first such caller and fixes inline |
| User changes mind on OPD focus | Cheap to revert — `git revert` the deletion commits; the autograd / kernel work this session is independent |

## Cross-references

- `docs/experience/wins/2026-05-18-bench-pretrain-nanochat-baseline-d12.md` — the 322× evidence that triggered the pivot
- `docs/projects/agent-rl-self-evolving.md` — the "train is runtime-led, not a second product" doctrine OPD inherits
- `docs/plans/train-runtime-architecture-v1.md` — to be rewritten under OPD framing
- `CLAUDE.md` / `AGENTS.md` — to be updated this session

## Cleanup of stale roadmap items

The following pending tasks (from the session task list) are now
**moot** after this pivot and should be closed:

- "写 3 篇 wins entries(pretrain/SFT/GRPO)+ 1 篇 hybrid acceptance" — the surfaces being benched are deleted
- "Hybrid linear-attn CUDA acceptance" — moot for the deleted training surfaces; relevant only for inference (separate track)

The Wave 2.x / Wave Σ / Wave 3-6 optimization tasks are **NOT** moot —
OPD needs the same autograd substrate they target.

---

**Decision authority**: ckl (project lead). **Decision recorded**:
2026-05-18 via this project doc.
