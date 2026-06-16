# OPD realckpt rollout uses infer KV decode

## Context

The main `arle train opd` path already had an `InferRolloutCtx`, but the
real-checkpoint training examples still called `opd_step` directly. That kept
those entry points on the train-crate rollout fallback: one full autograd
forward per sampled token.

This tranche changes only rollout generation for the LoRA realckpt example
path. Teacher scoring and differentiable student forward/backward remain in
autograd for that example; the production CLI still uses the existing
`InferTeacher` route.

## What Worked

- `opd_step_cuda_realckpt_train.rs` now loads an `InferStudent` for LoRA mode,
  syncs the current LoRA tensors into the inference engine each step, and calls
  `opd_step_with_teacher_forward_profiled_gkd_anchor` with `InferRolloutCtx`.
- Full fine-tune mode stays on the fallback path because there is no
  full-weight sync contract for `InferStudent`; only LoRA sync is supported.
- `InferStudent` adapter parsing now covers Qwen3.6 `linear_attn.*` projections,
  so all-linear Qwen3.6 LoRA names fail less often on the rollout sync path.
- The Qwen3.6 FP8 LoRA load gate now has an optional tokenizer-backed rollout
  smoke to verify that `InferStudent::generate_rollout` returns coherent text,
  not just a successful sync status.

## Environment

- Host: `.62` / `iv-ye8is8fbi8s6iplibbg7`.
- GPU: H20 GPU7; GPU3 avoided.
- Source:
  `/data01/arle-track1-opd-rollout-infer-202606170646`, built from a clean
  `git archive HEAD` plus only the Track 1 train files.
- Target:
  `/data01/arle-target-track1-opd-rollout-infer-202606170646`.
- Build env:
  `CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090
  ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/root/tl-venv/bin/python`.
- Build result: `cargo build -p train --example opd_step_cuda_realckpt_lora_bench
  --example qwen36_fp8_lora_load_gate --release --no-default-features --features
  cuda` passed in 4m02s.

## Verification

Local gates:

```text
cargo fmt --check
PASS

cargo check -p train --release --no-default-features --features no-cuda --example opd_step_cuda_realckpt_train
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --example opd_step_cuda_realckpt_train
PASS

CUDARC_CUDA_VERSION=12090 cargo test -p train --release --no-default-features --features cuda,no-cuda --lib infer_student::tests::parse_adapter_name_covers_all_linear_targets -- --nocapture
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --example qwen36_fp8_lora_load_gate
PASS

cargo clippy -p train --release --no-default-features --features no-cuda --example opd_step_cuda_realckpt_train --example qwen36_fp8_lora_load_gate -- -D warnings
PASS
```

Remote build:

```text
Finished `release` profile [optimized] target(s) in 4m 02s
```

Remote OPD realckpt LoRA rollout-256 gate:

```text
model=/data01/modelscope-cache/Qwen/Qwen3___5-0___8B-Base
CUDA_VISIBLE_DEVICES=7
ARLE_OPD_INFER_ROLLOUT=1
steps=1
rollout_len=256
lora_target_set=attention-qv
lora_rank=16
lora_alpha=32
```

Result:

```text
infer_student_loaded seconds=0.793758 student_model=/data01/modelscope-cache/Qwen/Qwen3___5-0___8B-Base max_seq_len=304 lora_rank=16 lora_alpha=32.000000
train_step step=1 prompt_index=0 prompt=[1, 872, 198, 3456] loss=1.821148097515e-1 rollout_len=260 step_seconds=42.800533 student_rollout_seconds=1.386916 teacher_forward_seconds=6.579300 student_forward_seconds=6.599877 kl_loss_seconds=6.725442 backward_seconds=28.083849
training_summary total_steps=1 total_wall_seconds=150.837978 mean_step_seconds=42.800533
```

The rollout phase is now 1.386916s for rollout-256. The remaining step time is
autograd teacher/student scoring and backward, which is outside this tranche.
The example's eval pass added about 108-110s outside the training step.

Remote coherence / needle smoke through `InferStudent::generate_rollout`:

```text
prompt="Context: The secret code is BLUE-73-MANGO. Question: What is 2 + 3? Also repeat the secret code exactly. Answer:"
generated_text="We need to answer: What is 2 + 3? ... The secret code is BLUE-73-MANGO. ... So answer: 5. And then repeat the secret code: BLUE-73-MANGO."
smoke_seconds=0.558533
contains_expect=true
```

## Delta

| Metric | Before | After | Verdict |
|---|---:|---:|---|
| realckpt example rollout path | train-crate/autograd fallback | `InferStudent` infer-core request with KV decode | fixed |
| rollout-256 generation phase | operator-reported ~30 min/step blocker on autograd full recompute | 1.386916s | pass |
| rollout smoke | not present in qwen36 load gate | tokenizer decode + expected-substring check | pass |
| Qwen3.6 linear attention LoRA names | unsupported by `InferStudent` parser | parsed as linear projections | fixed |

## Problems

- This example still uses in-process teacher scoring, so its total step is not a
  production OPD throughput verdict. The main CLI path remains the
  `InferTeacher` authority.
- Full-weight fine-tune rollout cannot be routed through `InferStudent` until a
  full-weight sync contract exists. This patch intentionally keeps full
  fine-tune on the fallback path.
- CUDA clippy for these examples is blocked before reaching the changed train
  files by unrelated pre-existing `crates/infer-cuda/src/dsv4.rs`
  `needless_option_as_deref` lints. CUDA typecheck passes.

## Rule

OPD sampled rollout belongs on the inference engine with KV-cache decode.
Autograd should re-enter only for differentiable scoring and backward.
