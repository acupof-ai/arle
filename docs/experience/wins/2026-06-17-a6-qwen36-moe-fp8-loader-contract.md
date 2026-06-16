# A6 Qwen3.6 MoE FP8 loader contract

## Context

Path A needs ARLE autograd-native Qwen3.6-35B-A3B FP8 LoRA training for OPD
35B. A5 licensed the synthetic frozen FP8 QLoRA operator substrate, but left the
real checkpoint wall open: the train-side model and loader still had to accept
Qwen3.6 MoE tensor names, config shape, and FP8 block-scaled frozen base
weights.

This tranche wires the train-side contract up to the real Qwen3.6 FP8 checkpoint
ABI. It is a loader/model contract gate, not the final 35B CUDA memory gate.

## What Worked

- Train `Qwen35Model` now builds sparse Qwen3.6 MoE MLP blocks for LoRA/frozen
  configs:
  - router gate
  - shared expert gate/up/down + shared expert gate
  - per-expert gate/up/down projections
  - grouped MoE autograd forward/backward from the A4 substrate
- `qwen35_loader` now parses Qwen3.6 MoE configs where `intermediate_size` is
  absent and the MoE fields are either flat or nested under `moe_config`.
- CUDA LoRA-student loads now accept frozen `F8_E4M3` base linear weights only
  when a matching `*.weight_scale_inv` side tensor exists with the Qwen3.6
  128x128 block shape. The uploaded autograd handle is
  `CudaFp8BlockScaled`.
- BF16 frozen-base CUDA upload coverage now includes Qwen3.6 MoE router,
  shared-expert gate, shared expert projections, and per-expert projections.

## Real Checkpoint Header Probe

Read-only probe on `.62` against `/data01/models/Qwen3.6-35B-A3B-FP8`:

```text
CONFIG hidden_size=2048 layers=40 heads=16 kv_heads=2 head_dim=256 vocab=248320
CONFIG intermediate_size_present=false num_experts=256 topk=8
CONFIG moe_intermediate_size=512 shared_expert_intermediate_size=512
CONFIG tie_word_embeddings=false layer_types_head=[linear_attention, linear_attention, linear_attention, full_attention]

embed_tokens.weight BF16 [248320, 2048]
lm_head.weight BF16 [248320, 2048]
linear_attn.in_proj_qkv.weight F8_E4M3 [8192, 2048]
linear_attn.in_proj_qkv.weight_scale_inv BF16 [64, 16]
mlp.gate.weight BF16 [256, 2048]
mlp.shared_expert.gate_proj.weight F8_E4M3 [512, 2048]
mlp.shared_expert.gate_proj.weight_scale_inv BF16 [4, 16]
mlp.shared_expert_gate.weight BF16 [1, 2048]
mlp.experts.0.up_proj.weight F8_E4M3 [512, 2048]
mlp.experts.0.up_proj.weight_scale_inv BF16 [4, 16]
input_layernorm.weight BF16 [2048]
```

The observed scale shapes match `[ceil(rows/128), ceil(cols/128)]`, which is
the loader contract now enforced.

## Verification

Local gates:

```text
cargo fmt --check
PASS

cargo check -p train --release --no-default-features --features no-cuda --lib
PASS

cargo clippy -p train --release --no-default-features --features no-cuda --lib -- -D warnings
PASS

cargo test -p train --release --no-default-features --features no-cuda --lib qwen35_loader::tests:: -- --nocapture
PASS: 22 passed

cargo test -p train --release --no-default-features --features no-cuda --lib qwen35::tests:: -- --nocapture
PASS: 5 passed
qwen36_moe_lora_fd target=model.language_model.layers.0.mlp.experts.0.up_proj.weight.lora_b[1]
analytic=-1.199122053e-3 numeric=-1.192092896e-3 rel_err=5.862e-3

CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --lib
PASS

cargo clippy -p qwen35-spec --release --no-default-features --lib -- -D warnings
PASS
```

`CUDARC_CUDA_VERSION=12090 cargo clippy -p train --release --no-default-features
--features cuda,no-cuda --lib -- -D warnings` is currently blocked by unrelated
pre-existing `infer-cuda/src/dsv4.rs` clippy findings
(`needless_option_as_deref` at the DSv4 scratch call site), before reaching this
train diff.

## Verdict

The train-side Qwen3.6 MoE LoRA + FP8 frozen-base loader contract is licensed at
the local/model level. The real checkpoint ABI has been checked against the same
fields and scale shapes the loader enforces, and the integrated MoE LoRA
finite-diff gate remains below the 1e-2 relative tolerance.

Remaining Path A wall: run the CUDA 35B loader/memory gate on `.62`, then run a
model-level finite-diff or needle-quality gate using the real Qwen3.6 FP8
checkpoint.

## Rule

Do not call synthetic QLoRA support "35B ready" until the loader consumes the
real checkpoint ABI. Check config omission, tensor names, dtype, and scale shape
first; only then spend GPU time on the full model load.
