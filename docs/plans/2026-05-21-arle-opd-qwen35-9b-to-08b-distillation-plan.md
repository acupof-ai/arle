# 2026-05-21 — ARLE OPD CUDA: Qwen3.5-9B → Qwen3.5-0.8B real distillation

> **Status:** plan / coordination doc. Owns the 6-phase execution of a real
> large-to-small OPD distillation experiment using ARLE's full stack
> (teacher inference via `infer`, student OPD step via `train`),
> benchmarked head-to-head against PyTorch + TRL on industry-standard
> evaluations (MMLU + IFEval). Companion to the
> [usage manual](../projects/2026-05-21-arle-opd-cuda-usage-manual.md) and
> [industry positioning doc](../projects/2026-05-21-opd-industry-positioning-best-framework.md).

## Why this experiment

Today's 32-commit OPD CUDA cycle shipped:

- Qwen3-0.6B step at **0.164 s**, **1.71× faster** than PyTorch CUDA reference
- **2.04× faster** than HuggingFace TRL `GKDTrainer` at matched setup
- LoRA-only OPD at **3.9 GB peak** (fits 4 GB consumer cards)
- Convergence verified at lr=1e-7: held-out exact-overlap 50 → 82.8 %

All of the above is single-runtime (teacher + student in `train::TensorStore`).
The architectural claim ARLE makes — that the production-serving runtime
(`infer`) and the OPD training runtime (`train`) share the same backend +
same model code path, so the teacher inference inside an OPD loop is
production-fast — has not yet been demonstrated end-to-end on a real
large-to-small distillation task with an industry-standard eval.

This plan executes that demonstration:

1. Use **`infer` runtime** to serve **Qwen3.5-9B (AWQ Int4)** as the teacher
   (production decode path, paged-KV + CUDA Graph)
2. Use **`train` OPD step** to update **Qwen3.5-0.8B (LoRA r=16 on q/v)**
   as the student
3. Same hardware (RTX 4070 Ti SUPER 16 GB) runs both
4. Compare end-to-end against **TRL `GKDTrainer`** at matched config
5. Evaluate all 4 models (teacher / base student / ARLE-distilled student
   / TRL-distilled student) on **MMLU + IFEval** via `lm-eval-harness`

## Model selection (locked)

| Role | Model | Source | Size | Format |
|---|---|---|---|---|
| Teacher | `Qwen/Qwen3.5-9B` (or `tclf90/Qwen3.5-9B-AWQ` if BF16 doesn't fit) | ModelScope | ~17 GB BF16 / ~5 GB AWQ Int4 | safetensors |
| Student (training target) | `Qwen/Qwen3.5-0.8B-Base` | ModelScope | ~1.6 GB | safetensors BF16 |
| Student LoRA adapter | (trained in-place) | — | ~9 MB (r=16 on q/v) | safetensors |

Memory budget on RTX 4070 Ti SUPER 16 GB:

| Component | Peak |
|---|---:|
| Teacher (AWQ Int4) | ~5 GB |
| Teacher KV cache (FP8, seq ≤ 256) | ~0.5 GB |
| Student BF16 weights (base, frozen) | ~1.6 GB |
| Student LoRA adapters (trainable) | ~10 MB |
| Student AdamW state (LoRA only) | ~40 MB |
| OPD activations / tape | ~2 GB |
| **Total budget** | **~9 GB peak** (well within 16 GB) |

If `Qwen3.5-9B-AWQ` quantization isn't compatible with `infer` immediately,
fallback: download `Qwen/Qwen3.5-9B` BF16 + use ARLE's FP8 KV path (~12 GB
peak teacher) and accept tighter memory. Worst case: revert to
`Qwen/Qwen3.5-4B` BF16 teacher (~8 GB) for the initial run, then scale.

## Hyperparameters (locked per the validated SOLID-gated set)

| Knob | Value | Rationale |
|---|---|---|
| `--lr` | `1e-5` (LoRA) | LoRA tolerates higher lr than full-finetune's `1e-7` |
| `--rollout-len` | `8` | Validated by `ff931b4` |
| Prompt set | 64 diverse real-text prompts via `--prompts-file <jsonl>` | Tokenizer integration shipped in `50ef595` |
| AdamW betas / eps / wd | `(0.9, 0.999)` / `1e-8` / `0` | TRL-matched |
| Grad clip | `1.0` (fused CUDA kernel) | `92de30e` |
| LoRA target | `q_proj` + `v_proj`, `r=16`, `α=32` | TRL standard recipe |
| Steps | 1000 (initial demo); 5000 (if time) | 1000 × 0.2 s = ~3.5 min compute |
| Eval cadence | step 0, 100, 250, 500, 1000 | Continuous metrics from `cb07373` |

## Industry-standard benchmarks (locked)

- **MMLU (5-shot)** — measures knowledge retention through distillation.
  Industry baseline number for Qwen3.5-9B teacher is published; 0.8B
  student typical range 40-50 % (base) → target ≥ 1-2 pp lift post-OPD.
  Tool: `lm-eval-harness` with `tasks=mmlu`.
- **IFEval** — measures instruction-following behavior, most sensitive
  to distillation deltas. Pre/post-OPD ΔIFEval > ΔMMLU typically.
  Tool: `lm-eval-harness` with `tasks=ifeval`.
- **No GSM8K / HellaSwag in this run** — keeps the comparison focused.
  Add later if budget remains.

## Phase plan + ownership

| Phase | Work | Owner | Time | Acceptance / safety gate |
|---:|---|---|---:|---|
| **1** | Download Qwen3.5-9B (AWQ preferred) + Qwen3.5-0.8B from ModelScope; verify `infer` loads teacher (`arle --doctor --model-path <teacher>`) | codex | ~30-60 min | Teacher loads + decodes 1 token without crash; student loads via `qwen35_loader`. |
| **2** | Create `crates/train/src/teacher_infer.rs` adapter exposing `forward(input_ids, positions) -> logits` via `infer::Engine` instance, sharing the train TensorStore's `CudaBackend` arc. Modify `opd_step` to take `&dyn TeacherForward`. | codex | 1-2 days | First runnable OPD step with infer-teacher ≤ 2× the current single-runtime step time (≤ 0.4 s on Qwen3-0.6B-class). If slower, stop + diagnose. |
| **3** | Run ARLE 9B→0.8B OPD distillation: 1000 steps, LoRA r=16, lr=1e-5, real-text 64 prompts, eval cadence 0/100/250/500/1000. Save student LoRA adapter. | codex | ~3.5 min compute | Run completes without OOM; trajectory shows monotonic held-out KL decrease. |
| **4** | Mirror in TRL `GKDTrainer` (`.venv` Python). Same teacher, same student, same prompts, same 1000 steps, constant LR. Save trained student. | codex | ~30-60 min compute | TRL run completes. Report step time + peak mem. |
| **5** | Install `lm-eval-harness` (`pip install lm-eval`). Run MMLU + IFEval on 4 models: teacher / base student / ARLE-distilled / TRL-distilled. | codex | 1-2 hr compute | All 4 models evaluated. Verdict-table populated. |
| **6** | Generate 4-bar comparison PNG (ARLE OPD step / TRL OPD step / PyTorch pure inference / ARLE pure inference). Update README + ZH README. Write wins entry summarizing perf + eval. | Claude | ~1 hr | README + ZH README updated; PNG committed; wins entry committed. |
| **7** | One-click `examples/opd/run-distillation.sh` (✅ shipped 2026-05-21). Default smoke mode runs `arle train opd --smoke` end-to-end in <30 s with no download; opt-in real mode auto-resolves `ARLE_TEACHER`/`ARLE_STUDENT` (HF or ModelScope) to local cache via the `.venv`. Follow-up: extend `arle train opd` CLI itself with `--prompts-file <jsonl>` and `--model-id <hf-id>` native-Rust resolver (deferred to `hf-hub` integration). | Claude (shell wrapper); codex (native CLI later) | ~1 hr wrapper / multi-day native | Smoke path verified end-to-end without internet; real mode verified after the Phase 1 model downloads complete; native `--prompts-file` lands when codex's Phase 2 adapter is in flight. |

## Acceptance — overall

ARLE claims of "best OPD framework" require all of:
- ARLE per-step wall-clock < TRL per-step wall-clock (already proven 2.04× on Qwen3-0.6B; expect to hold or grow at 9B teacher)
- ARLE peak GPU memory ≤ TRL peak GPU memory at matched setup
- ARLE-distilled student IFEval ≥ TRL-distilled student IFEval (or within 0.5 pp tie)
- ARLE-distilled student MMLU ≥ TRL-distilled student MMLU (or within 0.5 pp tie)
- All measurements reproducible from a single shell command per phase

If ARLE loses on ANY of the four axes by > 1 pp / > 5 %, that's a real finding
— report it honestly in the wins entry. Don't massage the recipe to win.

## License-or-kill (per phase)

- **Phase 2** KILL: if first runnable infer-teacher OPD step > 0.5 s on a
  Qwen3-0.6B-scale teacher (vs. current 0.164 s in-runtime), the adapter
  is slower than expected — revert and stick with the in-runtime teacher
  path. Document the regression.
- **Phase 3** KILL: if held-out KL doesn't decrease over 1000 steps
  (anywhere in the trajectory), recipe is broken — diagnose lr / prompts /
  LoRA target set.
- **Phase 5** KILL: if BOTH ARLE-distilled and TRL-distilled IFEval/MMLU
  are within ±1 pp of the base student, the recipe is too short — extend
  to 5000 steps and re-eval.

## Hand-off protocol

- Each phase commits separately to `main`
- codex ACKs Phase 2 specifically (architecture change) before starting
- After every phase Claude pulls latest, updates this plan with results,
  and proceeds with Phase 6 docs work once Phase 5 lands
- If codex hits any blocker (download fails, infer can't load, OOM, build
  break), they post the blocker in their tmux session, Claude reads via
  the loop tick and re-briefs / pivots

## Cross-links

- Industry positioning: [`../projects/2026-05-21-opd-industry-positioning-best-framework.md`](../projects/2026-05-21-opd-industry-positioning-best-framework.md)
- Cycle wrap: [`../projects/2026-05-21-opd-cuda-cycle-wrap.md`](../projects/2026-05-21-opd-cuda-cycle-wrap.md)
- Usage manual: [`../projects/2026-05-21-arle-opd-cuda-usage-manual.md`](../projects/2026-05-21-arle-opd-cuda-usage-manual.md)
- TRL head-to-head (Qwen3-0.6B scale): [`../experience/wins/2026-05-21-arle-vs-trl-gkd-head-to-head.md`](../experience/wins/2026-05-21-arle-vs-trl-gkd-head-to-head.md)
- LoRA-only bench: [`../experience/wins/2026-05-21-arle-cuda-opd-realckpt-lora.md`](../experience/wins/2026-05-21-arle-cuda-opd-realckpt-lora.md)
