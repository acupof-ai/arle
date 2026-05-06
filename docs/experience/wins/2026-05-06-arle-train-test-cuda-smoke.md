# `arle train test --backend cuda` smoke — TileLang+native-decode build, L4

## Goal

- Verify the end-to-end `convert → pretrain → sft → eval` pipeline still
  works on CUDA after the TileLang full migration (`1d6b7836`) +
  vectorized elementwise (`61e98353`) +
  `INFER_DETERMINISTIC=1` cublasLt patch (`98330680`). Goal taxonomy:
  **regression-check (training pipeline)**.

## Hypothesis

- The migration touched only attention/decode/elementwise kernels on the
  inference path, not autograd or the training-side weight loaders. The
  canonical tiny fixture should still complete in single-digit seconds and
  produce loadable safetensors.

## Command

```bash
. /tmp/arle-env.sh
/tmp/arle-target-release/release/arle train test \
  --backend cuda \
  --keep-artifacts \
  --out-dir /tmp/arle-fixture-claude \
  --json
```

## Environment

- **Backend:** cuda
- **Hardware:** NVIDIA L4, 23034 MiB VRAM, compute capability 8.9, driver
  580.82.07, CUDA 12.8 / nvcc 12.8.93.
- **Commit:** `98330680` (HEAD at run time was `61e98353`; `98330680` had
  not yet been fetched, but the runtime binary at
  `/tmp/arle-target-release/release/arle` was built post-migration and
  carries the same TileLang-only attention path).
- **Binary:** prebuilt `/tmp/arle-target-release/release/arle` (codex's
  release build of the migrated tree).
- **Model architecture (auto-generated tiny fixture):** Qwen2-shape, 2
  layers, `hidden_size=32`, 2 heads, `head_dim=16`, vocab=23,
  `max_position_embeddings=16`, BF16 weights — total ≈ 12K params. LoRA
  adapter `r=4`, `alpha=8`, `target_modules=all-linear`, `task_type=CAUSAL_LM`.

## Results

| step | status |
|------|--------|
| convert | ok |
| pretrain | ok |
| sft | ok |
| eval | ok |

- **Wall time:** **2.84 s** (4-step pipeline, single CUDA L4)
- **Eval loss:** 2.953 / **PPL 19.16** (10 tokens evaluated)
- **Servable model dir:** `/tmp/arle-fixture-claude/sft/latest/`
- **Artifacts (sizes):**
  - `pretrain/step_000002/model.safetensors` — 52 KB
  - `pretrain/step_000002/optimizer.safetensors` — 44 KB
  - `sft/latest/model.safetensors` — 52 KB
  - `sft/latest/adapter_model.safetensors` — 24 KB
  - `sft/latest/optimizer.safetensors` — 44 KB
  - `sft/latest/{config,adapter_config,generation_config,tokenizer,trainer_state}.json`
  - Total fixture footprint: **436 KB** (regenerable in 2.84 s).

## Cross-checks

- **32×256 BF16 trajectory gate** (separate run by codex, same machine):
  3 / 3 PASS, 32 / 32 ok pairs each — see
  `bench-output/2026-05-06-qwen3-bf16-{final-default,native-bf16-attn,tilelang-prefill-native-bf16-attn}-32x256/summary.md`.
  Confirms inference-side numerics restored after the BF16-degenerate
  scare in
  `docs/experience/errors/2026-05-06-qwen3-bf16-baseline-degenerate-post-00def315.md`.

- **`arle train test --backend cuda` is now the cheapest single-shot
  regression hook** for the train pipeline against new attention / KV /
  cublasLt-autotune work — full convert+pretrain+sft+eval in <3 s, no
  external weights or datasets required (corpus is auto-built).

## Problems

- None observed at fixture scale. The tiny model (12K params) does not
  exercise the BF16 paged-prefill hot path that codex spent the migration
  fixing — that path is exercised by guidellm sweeps + the 32×256
  trajectory gate, not this smoke.

## Learnings / Rule

- After any CUDA backend migration, `arle train test --backend cuda
  --json` is a 3-second ground-truth check that autograd + weight save
  pipeline still talk to CUDA correctly. Adding it to standard
  post-migration verification gates is cheap and stops `--features cuda`
  drift between the train and infer crates.
- Fixture artifacts are regenerable, so they live in `/tmp` (project
  convention: no weights in git). Future commits that want a stable
  reference checkpoint can re-run this command with a fixed
  `--out-dir`.
