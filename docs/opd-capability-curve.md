# OPD/QAT Capability Curve — runner contract + serving blocker

The capability curve is the reproducible "did OPD/QAT make the model better?"
artifact: run the same eval suite (MMLU + GSM8K + SWE-bench Pro) against the
base model and against successive OPD/QAT checkpoints, report Δ-vs-baseline with
multi-seed confidence intervals. The grading contract is owned by
[`eval.md`](eval.md): the candidate artifact is ARLE inference output, the
scorer is deterministic, no judge-repair.

This doc pins (a) how the three eval lanes run, (b) the historical structural
blocker that stopped every non-baseline curve point, and (c) the runbook for the
H20 teacher>student curve.

**Status 2026-06-16.** Option B has landed: `arle train opd` / `self-opd`
accept `--save-checkpoint <DIR>` + `--save-every <N>` and write
full-materialized `step_NNNNNN/` HF dirs that `arle serve --model-path` loads
unchanged. The blocker section below is retained as the design record. The
ready-to-fire H20 script is [`scripts/h20_teacher_student_opd_curve.sh`](../scripts/h20_teacher_student_opd_curve.sh).

## Lanes — exact commands

### MMLU + GSM8K (`scripts/arle_capability_eval.py`)

The harness talks the OpenAI v1 **`/v1/completions`** surface (MMLU + GSM8K both
use the completions endpoint, not chat — see `ArleClient.completion`,
`arle_capability_eval.py:110`). It expects standard fields: `model`, `prompt`,
`max_tokens`, `temperature`. MMLU is 5-shot, extracts an A/B/C/D letter; GSM8K is
8-shot, extracts the final number. Datasets `cais/mmlu` (all/test+dev) and
`openai/gsm8k` (main/test+train) load via HF `datasets`.

```bash
# 1. serve the model under test (base, or a merged checkpoint — see blocker below)
arle serve --backend cuda --model-path <model-or-checkpoint-dir> --port 8123

# 2. baseline / single-shape smoke
ARLE_BASE_URL=http://localhost:8123 \
python scripts/arle_capability_eval.py \
  --backend arle \
  --base-url http://localhost:8123 \
  --model-id <served-model-id> \
  --tasks mmlu,gsm8k \
  --n-samples 200 \
  --seed 0 \
  --output bench-output/cap-base

# 3. claim-grade multi-seed (writes seed_<N>/ subdirs)
python scripts/arle_capability_eval.py \
  --backend arle --base-url http://localhost:8123 --model-id <served-model-id> \
  --tasks mmlu,gsm8k --n-samples 500 --seeds 0,1,2,3,4 \
  --output bench-output/cap-claim
```

`--model-id` is a *label only* for the arle backend — the served process holds
exactly one model, so any string is accepted and echoed into the report. The
served-model id is derived from the model dir name (`model_id_from_path`,
`infer-api/src/serve_engine.rs:465`); pass that string for clean report labels.

Multi-seed CI + paired/McNemar Δ via `scripts/analyze_multi_seed.py` (default
gate: mean ≥ 0.505, σ ≤ 0.015 — MMLU-tuned; override per task).

### SWE-bench Pro (`scripts/arle_swe_pro_eval.py`)

Three phases with a hard boundary: `prepare` (materialize instances + manifest,
**no model, no docker**), `generate` (ARLE produces one unified-diff patch per
instance — needs a served capable model), `evaluate` (official Docker/Modal
scorer — **docker or modal required, only here**).

```bash
# prepare — verified to run today (HF dataset loads, manifest + raw_samples.csv built)
python scripts/arle_swe_pro_eval.py prepare \
  --output bench-output/swepro-smoke --limit 3 --seed 0

# generate — chat surface (/v1/chat/completions); needs a served capable model
python scripts/arle_swe_pro_eval.py generate \
  --output bench-output/swepro-smoke \
  --base-url http://localhost:8123 --model-id <served-model-id> \
  --limit 3 --seed 0

# evaluate — official scorer; Docker (or Modal) required HERE ONLY
git clone https://github.com/scaleapi/SWE-bench_Pro-os /tmp/SWE-bench_Pro-os
python scripts/arle_swe_pro_eval.py evaluate \
  --output bench-output/swepro-smoke \
  --eval-repo /tmp/SWE-bench_Pro-os \
  --use-local-docker --docker-platform linux/amd64
```

**What `generate` requires:** a served model on the OpenAI v1 **chat** surface
(`ArleChatClient.chat`, `arle_swe_pro_eval.py:84` → `/v1/chat/completions`),
fields `model`/`messages`/`max_tokens`/`temperature`. Default `max_tokens=4096`,
`temperature=0.0`, `timeout=600s`. The model is prompted with only the
sanitized instance (`repo`, `base_commit`, `problem_statement`, `requirements`,
`interface`, `repo_language`) — the gold `patch`/`test_patch`/`fail_to_pass`/
`pass_to_pass` are *never* shown to the model (sanitizer enforced,
`arle_swe_pro_eval.py:167`). Output is the extracted unified diff.

**What `evaluate` requires (verified against the cloned repo):**
- The cloned evaluator repo at `/private/tmp/SWE-bench_Pro-os` — has
  `swe_bench_pro_eval.py`, `run_scripts/` (1002 `instance_*/` dirs, each with
  `run_script.sh` + `parser.py` + `instance_info.txt`), `dockerfiles/`.
- **Docker** (`pip install docker`, `--use_local_docker`) **or Modal**
  (`pip install modal`, default path). The evaluator imports both lazily; one is
  mandatory only at `evaluate`.
- Per-instance prebuilt images `{dockerhub_username}/sweap-images:{tag}` pulled
  from Docker Hub (`helper_code/image_uri.py`); default username `jefzda`
  (`arle_swe_pro_eval.py:387`). The tag is derived from the instance id.
- **Data source for the scorer is `raw_samples.csv`, NOT the HF dataset** — the
  evaluator reads the CSV with `pandas` (`swe_bench_pro_eval.py:476`), indexes by
  `instance_id`, and reads `fail_to_pass`/`pass_to_pass` from it. `prepare`/
  `generate` already write `raw_samples.csv` with those gold columns (they live in
  the CSV for the *scorer*, never in the model-visible `instances.jsonl`).
- Apple Silicon: official images are linux/amd64 → pass `--docker-platform
  linux/amd64` (the script auto-fills this when omitted with `--use-local-docker`).
- **Dataset revision:** `prepare`/`generate` pin `ScaleAI/SWE-bench_Pro`
  `split=test revision=main` (override `--dataset-revision` or
  `ARLE_SWE_PRO_REVISION`). The `evaluate` phase does not re-touch HF — it only
  needs the CSV + the matching prebuilt images, so the revision is fixed at
  `prepare` time and carried in `manifest.json`'s `sample_fingerprint`.

An instance is resolved only when every `FAIL_TO_PASS` and `PASS_TO_PASS` test
passes inside the instance image.

## The serving blocker (historical Option-B design record)

**Symptom.** Before Option B landed, the curve's baseline point (the base HF
model) served and evaled, but every OPD/QAT *checkpoint* point could not be
served, so the curve had exactly one point.

**Root cause — two gaps, both confirmed in source:**

1. **The serve loader is adapter-blind.** `arle serve` → `ServeArgs.model_path`
   (`crates/cli/src/args.rs:326`) is the only model knob — no adapter/LoRA flag.
   The CUDA loader `CudaModel::from_safetensors` (`crates/infer-cuda/src/loader.rs:31`)
   reads weights from `model.safetensors` / `model.safetensors.index.json`
   (`loader.rs:418`,`:449`) under **base HF tensor names only**; it has no LoRA
   merge step. So a served dir must already be a complete HF model — an
   adapter-only dir is unservable.

2. **The training loop saves the wrong artifact (and the CLI saves none).**
   The OPD save machinery in `crates/train/src/qwen35_checkpoint.rs` already
   supports both modes via `Qwen35StudentWeights`
   (`qwen35_checkpoint.rs:67`):
   - `FullMaterialized { bf16 }` — *merges* base+adapter and writes
     `model.safetensors` under base HF names (`save_full_materialized_weights`,
     `qwen35_checkpoint.rs:245` → `save_materialized_registry`). **This is exactly
     what the serve loader can consume.**
   - `AdapterOnly { .. }` — writes a PEFT `adapter_model.safetensors` +
     `adapter_config.json` (`save_adapter_only_weights`, `:268`). The serve loader
     **cannot** read this.

   But the *only* caller that saves checkpoints at intervals is the example
   binary `crates/train/examples/opd_step_cuda_infer_teacher_train.rs`
   (`--save-student-checkpoint DIR --save-every N`,
   `maybe_save_student_checkpoint`, example `:935`), and it hardcodes
   `AdapterOnly` for both step (`:1003`) and final (`:1022`) saves. The shipping
   CLI handlers `run_opd` / `run_self_opd` (`crates/cli/src/train_cli.rs:226`,`:583`)
   **save no checkpoint at all** — they print losses/metrics and exit.

Before the CLI save fix: nothing the curve could serve was ever written. The
merge path that would produce a servable dir existed but was unreachable from
the CLI, and the one path that did write only wrote adapter-only.

### Smallest fix — pick ONE (both are small, option B is smallest)

This is a **crates task** (out of scope for the curve driver). Stated at
file:line so a codex can copy it verbatim.

**Option B (recommended — smallest, no serve-loader change).** Make the OPD/SOPD
CLI save a *servable* checkpoint at intervals, in `FullMaterialized` mode.
- Add `--save-checkpoint <DIR>` + `--save-every <N>` to `TrainOpdArgs` /
  `TrainSelfOpdArgs` (`crates/cli/src/args.rs`, near the existing
  `lora_rank`/`lora_alpha`/`lora_target_set` fields, ~`:760`).
- In `run_opd_from_dirs` (`crates/cli/src/train_cli.rs:249`) and
  `run_self_opd_from_dir` (`:662`), after each step where `step % save_every == 0`
  and on final, call the existing
  `save_qwen35_student_checkpoint(Qwen35StepCheckpoint{..}, student, store, tape,
  Qwen35StudentWeights::FullMaterialized { bf16: true })`
  (`crates/train/src/qwen35_checkpoint.rs:204`). The example at `:950` is the copy
  template — change only the `Qwen35StudentWeights` arm from `AdapterOnly` to
  `FullMaterialized { bf16: true }`.
- Result: each `step_NNNNNN/` dir is a complete HF model (`config.json`,
  `tokenizer.json`, `model.safetensors`) that `arle serve --model-path
  step_NNNNNN` loads unchanged. The curve driver's `model_path` checkpoint points
  then work with zero serve-side change.
- Cost: `FullMaterialized` writes the full base each interval (≈ base size on
  disk per save). Acceptable for a curve at a few checkpoints; gate `--save-every`
  to keep the count small. If disk is tight, keep `AdapterOnly` as an opt-in flag
  value and default `FullMaterialized`.

**Option A (serve-side adapter load — larger, avoids re-writing the base each save).**
Teach the serve path to load base + an adapter dir and merge at load.
- Add `adapter_path: Option<String>` to `ServeArgs` (`crates/cli/src/args.rs:323`,
  the `ServeArgs` struct, alongside `model_path` at `:326`).
- Thread it through `ServeHttpOptions` (add `adapter_path: Option<String>`,
  `crates/infer-api/src/serve.rs:31`) and into
  `LoadedInferenceEngine::load_with_config` (`crates/infer-api/src/loaded.rs:290`)
  — add an `adapter_path` parameter (or carry it on `EngineLoadConfig`).
- In the CUDA load path (`load_cuda` → `CudaModel::from_safetensors`,
  `crates/infer-cuda/src/loader.rs:31`), after loading base weights, read the PEFT
  `adapter_model.safetensors` + `adapter_config.json` and fold `B·A·(α/r)` into the
  target matrices before upload (or as a fused decode add). This mirrors the merge
  math already in `save_full_materialized_weights` /
  `crates/train/src/qwen35.rs:2263` (`merged_tensor`), so the arithmetic is
  settled; the work is plumbing + a CUDA-side merge at load.
- Cost: more surface (CLI + infer-api seam + infer-cuda loader), but checkpoints
  stay tiny (adapter-only) and the curve serves any `{base, adapter}` pair.

Recommendation: **Option B now** (unblocks the curve with a 1-arm change in the
CLI save call, no new serve surface), Option A later if adapter-only checkpoint
size matters for the curve cadence.

## Driver

`scripts/opd_capability_curve.py` is the thin orchestrator: a checkpoint
manifest in, `curve.json` + a Δ-vs-baseline table out. It reuses
`arle_capability_eval.py` (MMLU/GSM8K) and `arle_swe_pro_eval.py` (SWE-bench Pro)
by subprocess and `analyze_multi_seed.py` for CI — it reimplements no eval logic.
It works against an already-running baseline serve (`base_url`) and against
checkpoint dirs (`model_path`) that it launches and tears down itself. Option B
makes each `step_NNNNNN/` a `model_path` point with no driver change. See
`--help` and `--dry-run`.

## H20 teacher>student runbook (prepared, QAT-gated)

The capability verdict must come from a teacher>student run, not the
teacher==student or cold-start V100 probes. The prepared lane is:

```bash
ARLE_ROOT=/data01/build/arle \
STUDENT_MODEL=/data01/models/Qwen3.5-0.8B-Base \
TEACHER_MODEL=/data01/models/Qwen3.5-4B \
bash scripts/h20_teacher_student_opd_curve.sh
```

Defaults: 10k pure-KL OPD steps, rollout_len=8, lr=2e-5, LoRA r16/alpha32 on
attention q/v, `--save-checkpoint $RUN_ROOT/checkpoints`,
`--save-every 2000`, then `scripts/opd_capability_curve.py` over MMLU+GSM8K
with `n_samples=500` and seeds `0,1,2,3,4`. The generated manifest lists
`base-0p8b`, each `opd-step-NNNNNN` full checkpoint, and the optional
`teacher-4b-upper` point. Outputs land under
`bench-output/h20-opd-teacher-student-*/{curve_manifest.json,checkpoints/,curve/curve.json,logs/}`.

This script is not a local smoke. It requires an H20 by default and is intended
for the post-QAT remote slot. Set env overrides only to change the controlled
run shape; setting `STUDENT_MODEL == TEACHER_MODEL` is rejected because it
recreates the confounded no-gradient setup.
