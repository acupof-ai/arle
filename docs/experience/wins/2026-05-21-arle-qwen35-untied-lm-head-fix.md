# Qwen3.5 Untied LM Head Fix

## Context

The Qwen3.5-9B-TQ4 dense-module parity harness attributed a 9B full-model
logits failure to the output projection path. Dense tensors in the TQ4
checkpoint were bit-identical to the BF16 source, and TurboQuant projection
GEMV parity was already clean, but `lm_head` parity failed badly because ARLE
always projected logits through `embed_tokens`.

Qwen3.5-9B sets top-level `tie_word_embeddings=false` and ships a separate
`lm_head.weight`. The nested `text_config` does not repeat the field, so the
Qwen3.5 spec parser defaulted it back to `true`.

## What Worked

The fix has two parts:

1. `qwen35-spec` now carries top-level `tie_word_embeddings` into nested
   `text_config` layouts.
2. `infer::model::qwen35` now loads `lm_head.weight` when the config is untied
   and routes all logits/output-projection paths through
   `output_projection()`.

Tied models preserve the existing behavior. The 0.8B-Base smoke uses the same
path with no separate `lm_head.weight`.

## Results

Artifact directories:

```text
bench-output/2026-05-21-qwen35-9b-tq4-lm-head-fix/
bench-output/2026-05-21-qwen35-08b-tied-lm-head-smoke/
```

Dense module gate:

| Model | Module | Before RMSE/ref-RMS | After RMSE/ref-RMS | Gate |
| --- | --- | ---: | ---: | --- |
| Qwen3.5-9B-TQ4 | `lm_head` | `130.502%` | `0.00176%` | PASS (`<=1%`) |
| Qwen3.5-0.8B-Base | tied output | n/a | `0.00120%` | PASS |

Verification:

```bash
cargo test -p qwen35-spec nested_config_top_level_tie_word_embeddings_overrides_default --release

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo check -p infer --example qwen35_dense_module_dump --features cuda

.venv/bin/python scripts/qwen35_tq4_dense_parity.py \
  --source-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B \
  --tq-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4 \
  --token-id 9419 \
  --output-dir bench-output/2026-05-21-qwen35-9b-tq4-lm-head-fix

.venv/bin/python scripts/qwen35_tq4_dense_parity.py \
  --source-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --tq-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --token-id 9419 \
  --output-dir bench-output/2026-05-21-qwen35-08b-tied-lm-head-smoke
```

## Problems

This licenses the lm-head/dense-module fix only. The follow-up full-model
Qwen3.5-9B-TQ4 logits gate still failed after this fix and remains documented
separately. Do not run the 9B-TQ4 OPD bench or switch user-facing headline docs
until that gate passes.

## Rule

For Qwen-family checkpoints, `tie_word_embeddings` is a model-level contract,
not a convenience default. Nested `text_config` layouts may omit it, so the
loader must preserve the top-level field before deciding whether logits are
tied to embeddings.
