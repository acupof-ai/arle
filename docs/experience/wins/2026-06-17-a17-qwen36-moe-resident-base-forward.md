# A17 Qwen3.6 MoE resident base forward

## Context

A16 removed the routed down-projection backward host materialization, but the
full-model forward gate still showed the real wall:

```text
qwen36_fp8_lora_fd_forward_profile total_seconds=43.041592
attention_seconds=0.395192
mlp_seconds=42.632636
```

The forward `moe_grouped_linear` path still called `tensor_host(weight)` for
every active expert and ran scalar host dot products over the frozen FP8 base
weights. That made both FD perturbation arms pay the full frozen-base
materialization/recompute cost.

## What Changed

`moe_grouped_linear` now tries a guarded CUDA fast path before the old host
path:

- pack only the active expert input rows;
- require every active expert base weight to be frozen and device-resident;
- call `Backend::matmul_bt` directly against the resident BF16/FP8 base weight
  handle;
- scatter the resident base output back to the original expert slots;
- apply only the LoRA delta on host, preserving the existing tape and gradients;
- fall back to the previous host implementation when CUDA residency is absent.

CPU and non-resident paths remain unchanged.

## Evidence

Local gates:

```text
cargo fmt --check
cargo check -p autograd --release --no-default-features --features no-cuda
CUDARC_CUDA_VERSION=12090 cargo check -p autograd --release --no-default-features --features cuda,no-cuda
cargo test -p train --release --test test_moe_a0 -- --nocapture
cargo test -p autograd --release --lib
cargo clippy -p autograd --release --no-default-features --features no-cuda -- -D warnings
cargo check -p train --release --no-default-features --features no-cuda --lib
cargo check -p train --release --no-default-features --features no-cuda --example qwen36_fp8_lora_fd_gate
CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --example qwen36_fp8_lora_fd_gate
cargo clippy -p train --release --no-default-features --features no-cuda --example qwen36_fp8_lora_fd_gate -- -D warnings
```

Remote `.62`, GPU0/GPU1, GPU3 avoided:

```text
source=/data01/arle-track1-route-frozen-fd-fast-20260617095440
target=/data01/arle-target-track1-route-frozen-fd
model=/data01/models/Qwen3.6-35B-A3B-FP8
CUDA_HOME=/usr/local/cuda
CUDARC_CUDA_VERSION=12090
ARLE_CUDA_DISABLE_FLASHMLA=1
ARLE_QWEN35_DEEPGEMM=0
```

Build:

```text
cargo build -p train --example qwen36_fp8_lora_fd_gate --release --no-default-features --features cuda
PASS: Finished release target in 32.45s
```

MLP-layer FD gate:

```text
log=/data01/arle-track1-route-frozen-fd-fast-20260617095440/a17_mlp_fd_gpu0_20260617_115312.log

qwen36_fp8_lora_fd_backward_profile total_seconds=0.090487
qwen36_fp8_lora_fd_backward_profile_op rank=1 op=MoeGroupedLinear count=3 seconds=0.034439

qwen36_fp8_lora_fd_gate_result load_seconds=12.698009 analytic_seconds=0.098665 \
  plus_seconds=0.006796 minus_seconds=0.006487 live_host_mib=2514.2 \
  target=model.language_model.layers.0.mlp.experts.210.up_proj.weight.lora_b \
  eps=1.0e-3 analytic=-2.581576268e-7 numeric=-2.587512427e-7 rel_err=2.294e-3
qwen36_fp8_lora_fd_gate PASS
RUN_EXIT=0
```

Full-model forward-only profile:

```text
log=/data01/arle-track1-route-frozen-fd-fast-20260617095440/a17_full_forward_gpu1_20260617_115312.log

qwen36_fp8_lora_fd_forward_profile total_seconds=0.832029 wall_seconds=0.832033 \
  layers=40 output_shape=[1, 3, 248320] attention_seconds=0.378071 \
  mlp_seconds=0.440192 lm_head_seconds=0.005418
RUN_EXIT=0
```

## Delta

| Metric | A16 | A17 | Delta |
|---|---:|---:|---:|
| full-model forward wall | 43.041592s | 0.832029s | -98.1% |
| full-model MLP wall | 42.632636s | 0.440192s | -99.0% |
| MLP-layer analytic arm | 0.183075s | 0.098665s | -46.1% |
| MLP-layer plus arm | 0.092355s | 0.006796s | -92.6% |
| MLP-layer minus arm | 0.091563s | 0.006487s | -92.9% |
| MLP-layer FD relative error | 3.170e-3 | 2.294e-3 | pass |

## Verdict

The measured full-model wall was not an unavoidable MoE backward issue. The
dominant remaining cost was forward-side frozen base-weight materialization.
Using the resident checkpoint handles cuts the 35B FP8 LoRA full forward gate
from 43.0s to 0.83s and keeps the real-checkpoint finite-diff gate licensed.

The next wall has moved: for this three-token full-model gate, attention is
0.378s and MLP is 0.440s. The old 42s MLP host path is gone.

## Rule

For frozen-base QLoRA, forward and backward must both consume the resident base
checkpoint handle. Host materialization is only a fallback for missing residency
or trainable base weights.
