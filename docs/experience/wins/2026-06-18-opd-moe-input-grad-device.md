# OPD MoE frozen input-grad fast path

## Goal

Reduce the 35B OPD rollout-256 step wall after rollout generation moved to the
infer engine. The previous smoke showed rollout generation was no longer the
wall: train/autograd backward dominated.

## Hypothesis

For `--lora-target-set attention-qv`, Qwen3.6 MoE base weights are frozen. The
MoE grouped-linear backward only needs `grad_input = grad_out @ W`, but the old
resident CUDA path still uploaded the original packed input and then copied the
base input gradient through a zero-filled merge buffer.

## Params

- Pod: `.62` (`iv-ye8is8fbi8s6iplibbg7`), GPU1.
- Model: `/data01/models/Qwen3.6-35B-A3B-FP8`.
- Command shape: `arle train opd --backend cuda --teacher-runtime infer
  --steps 1 --rollout-len 256 --prompt-max-tokens 64 --prompt-ids 1,3,8
  --logits-window-size 8 --lora-rank 1 --lora-alpha 2
  --lora-target-set attention-qv --grad-clip 1.0 --json`.
- Env: `ARLE_OPD_INFER_ROLLOUT=1`, `ARLE_OPD_ENGINE_OFFLOAD=all`,
  `ARLE_OPD_STEP_TRACE=1`, `ARLE_OPD_STEP_PROFILE=1`,
  `ARLE_MOE_GROUPED_PROFILE=1`, `CUDARC_CUDA_VERSION=12090`.
- Build: `/data01/arle-verify-ab22f727-target/release-fast/arle`,
  `FEATURES=cuda`, prebuilt CUDA kernels.

## Results

Local gates:

```text
rustfmt --edition 2024 --check <changed autograd files>
PASS
cargo test -p autograd --release
PASS
CUDARC_CUDA_VERSION=12090 cargo check -p autograd --release --no-default-features --features cuda,no-cuda --lib
PASS
cargo clippy -p autograd --release -- -D warnings
PASS
CUDARC_CUDA_VERSION=12090 cargo clippy -p autograd --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
PASS
CUDARC_CUDA_VERSION=12090 cargo clippy -p train --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
PASS
CUDARC_CUDA_VERSION=12090 cargo check -p cli --release --no-default-features --features cuda,no-cuda --lib
PASS
```

Remote CUDA gates:

```text
cargo test -p autograd --profile release-fast --features cuda cuda_matmul_bt_input_grad -- --nocapture
PASS: cuda_matmul_bt_input_grad_device_matches_cpu
PASS: cuda_matmul_bt_input_grad_accepts_frozen_bf16_rhs
```

35B rollout-256 A/B:

| Arm | rollout done | teacher fwd | student hidden | base backward | step done | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 7.735s | 12.324s | 58.621s | 94.246s | 189.790s | baseline |
| input-grad only | 7.817s | 12.323s | 59.252s | 93.181s | 189.804s | only small win |
| input-grad + no pack/merge | 7.757s | 12.324s | 58.723s | 84.808s | 180.635s | intermediate |
| final: no input host materialization | 7.785s | n/a | n/a | n/a | 181.853s | PASS |

Delta:

| Metric | Baseline | Final | Delta |
| --- | ---: | ---: | ---: |
| MoE grouped-linear profiled total | 48.271s | 38.370s | -9.901s / -20.5% |
| `base_resident_input_grad.total` | 28.496s | 26.461s | -2.035s / -7.1% |
| `pack.active_input` | 1.003s | 0.000s | removed |
| `merge.base_input_grad` | 5.976s | 0.000s | removed |
| Step to optimizer done | 189.790s | 181.853s | -7.937s / -4.2% |

The decisive fix was not the narrower GEMM primitive alone. The measured win
came from using it to make the `rank=0` frozen-MoE path skip packed input
construction and take ownership of the base input-gradient vector directly
instead of zero-fill plus add. A review pass also caught an earlier half-win:
the no-pack branch still materialized `input` via `tensor_host(input)` before it
decided packing was unnecessary. The final path checks only the input metadata
up front and pulls host input data only inside the fallback packed-input branch.

## Problems

- The remaining MoE wall is still `base_resident_input_grad.total` at 26.461s.
  It still launches and reads back per active expert. The next lever is grouping
  those per-expert input-gradient matmuls/readbacks, not more host-side merge
  cleanup.
- The run used `ARLE_MOE_GROUPED_PROFILE=1`, so absolute step times include
  profiling overhead. The A/B is same-binary/same-env and valid for the scoped
  delta.

## Learnings

For frozen-base LoRA training, do not route the common `grad_input only` case
through an API that requires the original forward input. A narrow backend
primitive keeps the mathematical contract clear and exposes the host-side
packing/merge work that can be deleted safely.

Artifacts on `.62`:

- Baseline: `/tmp/arle_opd35_r256_moe_profile.log`
- Input-grad only: `/tmp/arle_opd35_r256_moe_input_grad_after.log`
- No-pack/merge intermediate: `/tmp/arle_opd35_r256_moe_input_grad_v2.log`
- Final: `/tmp/arle_opd35_r256_moe_input_grad_v3.log`
