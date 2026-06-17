# A14 Qwen3.6 MoE Forward Active-Expert Compact

## Context

A13 localized the full-model Qwen3.6 FP8 LoRA forward wall to sparse MLP
forward: the first measured layers spent about 10-13s each in `mlp`, while
attention and linear-attention were milliseconds. The implementation still
iterated all 256 nominal routed experts in `moe_grouped_linear` forward even
when one token activates only top-8 experts.

This tranche applies the already-licensed active-expert principle to forward:
keep the public `[experts, max_rows, out_dim]` tensor shape unchanged, but only
compute route-active experts and leave inactive rows zero.

## What Worked

- `moe_grouped_linear` now builds the active expert list from routes before the
  forward expert loop.
- The route lookup and output tensor shape stay unchanged, preserving the
  backward saved context and downstream `MoeGroupedWeightedScatter` contract.
- Inactive experts no longer pay host weight reads and dense LoRA/base dot loops
  during forward.

## Environment

- Host: `.62` / `iv-ye8is8fbi8s6iplibbg7`.
- GPU: H20 GPU2 via `CUDA_VISIBLE_DEVICES=2`; GPU3 avoided.
- Model: `/data01/models/Qwen3.6-35B-A3B-FP8`.
- Source:
  `/data01/arle-track1-opd-rollout-infer-202606170646`, file checksum matched
  local `crates/autograd/src/ops/moe.rs`
  `88727889e5d585b38219ac272618dd42c57929842fc1ab738d50e5b4f5f68f3d`.
- Target:
  `/data01/arle-target-track1-opd-rollout-infer-202606170646`.
- Build env:
  `CUDA_HOME=/usr/local/cuda CUDARC_CUDA_VERSION=12090
  ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/root/tl-venv/bin/python`.

## Verification

Local gates:

```text
cargo fmt --check
PASS

cargo check -p autograd --release --no-default-features --features no-cuda
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p autograd --release --no-default-features --features cuda,no-cuda
PASS

cargo test -p train --release --test test_moe_a0 -- --nocapture
PASS, max_rel=3.407372e-3, tiny_abs_failures=0

cargo test -p autograd --release --lib
PASS, 15/15

cargo clippy -p autograd --release --no-default-features --features no-cuda -- -D warnings
PASS

cargo check -p train --release --no-default-features --features no-cuda --example qwen36_fp8_lora_fd_gate
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --example qwen36_fp8_lora_fd_gate
PASS
```

Remote build:

```text
cargo build -p train --example qwen36_fp8_lora_fd_gate --release --no-default-features --features cuda
Finished `release` profile [optimized] target(s) in 0.17s
```

Remote routed-expert finite diff with backward profile:

```text
CUDA_VISIBLE_DEVICES=2 qwen36_fp8_lora_fd_gate \
  --model /data01/models/Qwen3.6-35B-A3B-FP8 \
  --device 0 \
  --target-set all-linear \
  --target-adapter auto:routed-up \
  --mode mlp-layer \
  --layer 0 \
  --eps 1e-3 \
  --profile-backward
```

Result:

```text
qwen36_fp8_lora_fd_backward_profile total_seconds=0.266602 op_seconds=0.216708 prelude_seconds=0.000138 merge_grad_seconds=0.049578 op_kinds=12 site_kinds=11
qwen36_fp8_lora_fd_backward_profile_op rank=1 op=MoeGroupedLinear count=3 seconds=0.212961 pct_total=79.880
qwen36_fp8_lora_fd_gate_result load_seconds=13.618570 analytic_seconds=0.361241 plus_seconds=0.093031 minus_seconds=0.092101 live_host_mib=2802.2 mode=mlp-layer layer=0 target=model.language_model.layers.0.mlp.experts.210.up_proj.weight.lora_b index=186 eps=1.0e-3 loss_base=1.941415121e-6 loss_minus=1.941672963e-6 loss_plus=1.941155006e-6 analytic=-2.581575700e-7 numeric=-2.589786163e-7 rel_err=3.170e-3
qwen36_fp8_lora_fd_gate PASS
```

Remote full-model forward-only profile:

```text
qwen36_fp8_lora_fd_forward_profile total_seconds=43.499409 wall_seconds=43.499413 layers=40 output_shape=[1, 3, 248320] cache_select_seconds=0.000019 embedding_seconds=0.000317 input_rmsnorm_seconds=0.003022 attention_seconds=0.413335 attention_residual_seconds=0.000387 post_attention_rmsnorm_seconds=0.003120 mlp_seconds=43.072301 mlp_residual_seconds=0.000433 final_norm_seconds=0.000077 lm_head_seconds=0.005155
```

Layer MLP spot checks against the A13 trace:

| Layer | A13 before | A14 after | Delta |
|---|---:|---:|---:|
| 0 | 10.384718s | 1.044902s | -89.94% |
| 1 | 13.084257s | 0.947112s | -92.76% |
| 2 | 13.090730s | 0.938973s | -92.83% |
| 3 | 13.422737s | 0.940642s | -92.99% |
| 4 | 13.398635s | 0.889684s | -93.36% |
| 5 | 13.457988s | 0.955948s | -92.90% |

## Delta

| Metric | A12 active-backward | A14 active-forward | Delta |
|---|---:|---:|---:|
| Routed-expert finite-diff rel_err | 3.170e-3 | 3.170e-3 | unchanged pass |
| MLP-layer analytic phase | 0.826345s | 0.361241s | -56.28% |
| MLP-layer plus forward | 0.577678s | 0.093031s | -83.89% |
| MLP-layer minus forward | 0.578015s | 0.092101s | -84.07% |
| MLP-layer `MoeGroupedLinear` backward | 0.198487s | 0.212961s | same order |

## Next Wall

The active-forward fix is licensed for the isolated MLP gate and removes the
10-13s/layer full-forward pathology. It does not license full-model gradients:
the full-model finite-diff gate now reaches a verdict but fails, and its
backward profile is dominated by `MatmulBT` plus full-model
`MoeGroupedLinear`. See
[`../errors/2026-06-17-qwen36-full-model-fd-fails-after-forward-compact.md`](../errors/2026-06-17-qwen36-full-model-fd-fails-after-forward-compact.md).

## Rule

Sparse MoE forward must operate on active routes, not nominal expert count.
Keeping the public expert-major tensor shape is acceptable; computing 248
inactive experts per token is not.
