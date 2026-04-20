# Wave 3: train_sft migrated onto Trainer<O, C, S> — pending remote bench

## Context

Phase 2 of `docs/plans/train-runtime-architecture-v1.md`. The pre-migration
`train_sft` bin hand-rolled the optimizer/step/clip/zero_grad/save loop
(~500 LOC total). Wave 3 (commit 44a7e19) composed the generic
`Trainer<AdamW, NoClip, ConstantLr>` via `run_with_hooks` with the binary
keeping only the step closure, the on_step_end weight-save hook, and the
CLI wiring.

Commit ad5568b then wired the remaining Phase 2 acceptance flags:
`--lr-schedule` (constant / linear-warmup / cosine-with-warmup),
`--warmup-steps`, `--min-lr`, `--grad-accum-steps`, `--metrics-jsonl`,
`--resume-from`. Schedule is a `Box<dyn LrSchedule>` through the new
`impl<T: LrSchedule + ?Sized> LrSchedule for Box<T>` blanket in autograd.

Also landed mid-wave from codex review on 3d9125d, feae23b, bdde441,
44a7e19, bd6c871, ad5568b, 49512b1, d9eee61: P1 tape/store cleanup hook,
P2 schedule-describe persistence, P3 NoClip returning true pre-clip
norm, legacy-bare-schedule-name compat on resume, force-emit metrics on
step 1 + final, dropping the redundant `step` key from the metrics
fields array, final-step force-save in the Trainer so runs that end
between save boundaries still persist a resumable checkpoint, aligning
`train_sft`'s bf16 checkpoint directory to `step_{:06}` to share a dir
with `trainer_state.json + optimizer.safetensors`, the `--resume-from`
weight-reload step (without which a resumed SFT run combined base
`--model` weights with the resume dir's AdamW moments — a silently
corrupt resume), `SafetensorsRegistry::load_into_strict` +
`validate_resume_config` to fail fast on partial/mismatched checkpoints
instead of silently hybridising with base weights, and a stateless
`sample_index(seed, step, micro_step)` derivation so resumed runs pick
up the same data stream a single uninterrupted run would have produced
(the prior `LcgRng` position was not persisted in the checkpoint codec).

Phase 3 (this wins entry extended from Phase 2):

Commit 6bd0211 migrates `pretrain.rs` — the simplest Phase 3 binary
(~400 LOC, custom Transformer, no prior checkpoint support) — onto
`Trainer<AdamW, PretrainClip, ConstantLr>`. A local `PretrainClip` enum
wraps `NoClip` / `GlobalNorm` so `--grad-clip` and `--no-grad-clip`
collapse to one concrete `C`. `--metrics-jsonl` joins the shared
MetricSink path. Save/resume left wired to `None` pending a
`TransformerRegistry` codec (the custom `Transformer` has no
safetensors support yet; the hand-written loop also did not support
it). Smoke: `--dataset copy --steps 10 --vocab-size 300` yields step=1
+ every-2 + final=10 log lines with `grad_norm`/`tok_per_sec`/`lr`
fields populated and the expected decreasing loss.

Commit 429efc3 closes the tied/untied resume silent-corruption hole
in `pretrain_qwen3.rs` (config-match check now validates
`tie_word_embeddings`; weight load switched to `load_into_strict`)
and hardens `clip_grad_norm` at the free-function boundary so
non-finite/non-positive `max_norm` is a documented no-op across all
binaries (NaN previously bypassed the `<= 0.0` gate and poisoned
every gradient). Commit 613ff3c exposes
`Trainer::run_with_eval_and_hooks` — the missing combined surface
that binaries with their own save pipeline AND eval metrics need
(`run_with_hooks` + `run_with_eval` split the shape). Commit 8f2df76
upgrades "missing config.json" to a hard error in both `train_sft`
and `pretrain_qwen3` resume paths (silently skipping the config-match
check reopened the very hole 429efc3 fixed).

Commit 09c5c89 migrates the **SFT warm-up phase** of `train_grpo.rs`
onto `Trainer<AdamW, GrpoClip, ConstantLr>` (+157/-74 LOC). `GrpoClip`
mirrors `PretrainClip` — a local enum `None(NoClip) | Norm(GlobalNorm)`
so `--grad-clip`/`--no-grad-clip` collapse to one concrete `C`. The
GRPO phase stays hand-written (rollout_group + ref_model +
mean_sampled_kl mid-step don't fit a single `step_fn` cleanly), but
uses `args.grad_clip` via the sanitize-at-boundary
`clip_grad_norm(params, max_norm, store)` free function — no more
`GRAD_CLIP_NORM = 1.0` constant. New flags: `--grad-clip`,
`--no-grad-clip`, `--metrics-jsonl`. Smoke: `--sft-steps 1
--grpo-iters 1 --batch-prompts 2 --group-size 2 --seq 4` emits a
Trainer-format SFT metric (`step=1 loss=... lr=... grad_norm=...
ms_per_step=... tok_per_sec=...`) followed by `grpo iter 0: loss ...
mean_reward ... mean_kl ...` and a final `kl ... reward trajectory:
[...]`.

Commit 1a24db1 closes two findings from the 09c5c89 codex review:
(P1) **AdamW moments silently reset at SFT→GRPO boundary** — the
migrated `run_sft_phase` consumed its own AdamW inside the Trainer,
so the GRPO phase started from fresh moments + bias-correction step
zero. The stated blocker (missing `Trainer::into_optimizer()`) was
wrong — `Trainer::optim()` + the existing `AdamW::export_state` /
`import_state` were sufficient. Fix: `run_sft_phase` returns the
final `AdamWState`, the GRPO-phase AdamW calls `import_state` before
its first step, so moments and the bias-correction counter flow
across the phase boundary continuously. (P2) **`CliError` leaked
through `main() -> Result` with Debug formatting** — users saw
`Error: Custom("metrics sink: ...")`. Fix: mirror train_sft's
`ExitCode` wrapper so users see the plain message. Tests:
`adamw_state_roundtrip_across_trainer_boundary` (4 steps on one
AdamW vs 3 + export + fresh AdamW + import + 1 step match to 1e-6)
+ `cli_error_display_does_not_leak_debug_wrapper`.

Commit 8c11856 tightens the
`eval_final_step_forced_even_when_steps_mod_eval_every` test that
813d4f6 added: the original version fabricated `[2, 4, 5]` from the
eval call count and would have silently accepted eval firing on the
wrong steps. Fix: step_fn stashes `ctx.step + 1` into a shared cell
(StepCtx.step is 0-indexed pre-increment; the eval check in
run_inner runs on 1-indexed post-increment self.step, so the + 1
aligns them), eval_fn reads that cell — assertion now pins actual
trainer.step at each eval boundary. Codex review 813d4f6 (Low)
closed.

Commit 13afa3f closes out Phase 3 by aligning `train_multi_turn.rs`
to the shared CLI conventions — no Trainer migration (same
rollout/ref_model/mid-step mean_sampled_kl mismatch with `step_fn`
that kept train_grpo's GRPO phase hand-written; train_multi_turn is
entirely that shape, so there's no SFT portion to migrate
independently). +53/-10: deleted `GRAD_CLIP_NORM = 1.0` constant,
added `--grad-clip <f32>`/`--no-grad-clip`/`--metrics-jsonl <path>`,
wrapped `main` in the `ExitCode` pattern mirroring train_sft /
train_grpo / pretrain_qwen3. Per-iter `mt iter N: ...` println
replaced by a `MetricSample` emit through the shared sink (which
tees to stdout when `tee=true`), giving structured JSONL output
like `{"best_reward":0.125,"loss":..,"mean_kl":..,"mean_reward":..,"step":1}`.
Codex reviews of 1a24db1 and 8c11856 both came back with no
findings.

Commit 813d4f6 plugs two regressions from the bd5e277 codex review:
(1) **High** — multi-window eval in `pretrain_qwen3` cleared
`tape.entries` per window but never pruned `TensorStore`, so
`--eval-windows N` with large `--seq` accumulated N windows' forward
temporaries simultaneously. Fix: expose
`train::cleanup_after_backward` (promoted from `fn` to `pub fn` +
re-exported at the crate root), call it per window in the
`pretrain_qwen3` eval closure (re-disabling the tape after each
call), and also call it once at `Trainer::run_inner` scope after
`eval_fn` returns as the defensive single-call baseline.
(2) **Medium** — `Trainer::run_inner`'s eval branch only fired on
`self.step.is_multiple_of(eval_n)`, dropping the final-step eval
when `--steps` was not a multiple of `--eval-every`. Mirror the
save branch's `|| is_final` pattern. Two new tests
(`eval_final_step_forced_even_when_steps_mod_eval_every` +
`trainer_cleans_up_after_eval`) lock in the fix; the trainer-loop
test count is now 19 (was 17).

Commit bd5e277 completes the Phase 3 migration of `pretrain_qwen3.rs`
(~726 → ~890 LOC; actual training-step body shrank substantially —
growth is closure wiring + design-rationale comments) onto
`Trainer<AdamW, PretrainClip, ConstantLr>` using
`run_with_eval_and_hooks`. The held-out eval closure reports
`train::EvalOutcome` which surfaces as an `eval_loss` metric sample;
`on_step_end(step, store)` owns the `step_<N>/{config.json,
tokenizer.json, model.safetensors}` save pipeline, with an absolute-
step reconstruction from `start_step + trainer_relative_step` because
metric samples track the Trainer's relative counter. Trainer-side
`save_every`/`save_dir`/`resume_from` left `None`; trainer_state.json
+ optimizer.safetensors persistence deferred to a follow-up. Tests:
17 `test_trainer_loop` (+ `run_with_eval_and_hooks_drives_both_surfaces`)
+ 11 `test_grad_clip` (+ `non_finite_max_norm_is_noop` GC-7) green.

## What Worked

- 17 `test_trainer_loop` unit tests green covering step counts,
  grad-accum, LR schedule wiring, metrics (both log_every + forced),
  save (including the new final-step force-save for
  `save_every=5 total_steps=7`), eval-side `step`-field omission,
  resume (fresh / mismatch-reject / legacy-compat ×2 / rng_seed-
  mismatch), hook firing (alone + combined with eval), activation
  cleanup. Plus 11 `test_grad_clip` tests including GC-7
  `non_finite_max_norm_is_noop`.
- `test_convergence_smoke` confirms a tiny model (4-token copy task,
  AdamW lr=1e-2) still converges through the new loop: loss[0] - loss[9]
  > 0.5 with deterministic init.
- `cargo clippy --release -p train --no-default-features -- -D warnings`
  clean; `cargo check -p infer --no-default-features --features cuda,no-cuda`
  clean (CUDA-Rust typecheck on Mac).

## Status: pending-remote

End-to-end smoke + throughput compare needs a remote runner with
`Qwen3-0.6B` weights. CLAUDE.md bench policy (line 303 of plan v1):
> training-loss-curve bench; guidellm is inference-side and does not apply here

Plan:
1. On the remote, run `train_sft --model Qwen/Qwen3-0.6B --data
   <tiny-sft.jsonl> --steps 20 --batch 1 --log-every 1 --backend metal`
   and record the loss curve under the pre-migration binary (git
   checkout 41374a7^) and the post-migration binary (HEAD).
2. Expected delta: < 1% per-step loss difference — the migration is
   behavior-preserving. Anything larger is a regression.
3. Record tok/s + ms/step under both. Expected: within ±5% of pre.

Plan revisit after remote run → this file gets:
- `## Results` with both loss curves and the diff
- `## Problems` if any
- Backlinks from the plan §9 Phase 2 row.

## Rule

**Migration refactors that pass all unit tests but alter the step-loop
control flow still warrant a real-model smoke before declaring bench
regression-free.** Synthetic-loss tests exercise optim/backward but
not the forward/loss topology that the model actually takes.
