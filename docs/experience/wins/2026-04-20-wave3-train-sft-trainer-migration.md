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
