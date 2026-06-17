# A18 Qwen3.6 dense MatmulBT device backward

## Context

A17 moved the frozen-base MoE forward path onto resident FP8/BF16 checkpoint
handles. The next full-model backward profile showed a different wall:

```text
log=/data01/arle-track1-route-frozen-fd-fast-20260617095440/a18_full_fd_profile_gpu2_20260617_115907.log

qwen36_fp8_lora_fd_backward_profile total_seconds=29.618619 op_seconds=27.511765 \
  prelude_seconds=0.005357 merge_grad_seconds=2.085479 op_kinds=18 site_kinds=1167
qwen36_fp8_lora_fd_backward_profile_op rank=1 op=MatmulBT count=1167 seconds=25.429740 pct_total=85.857
qwen36_fp8_lora_fd_gate_result analytic_seconds=30.190943 plus_seconds=0.580901 \
  minus_seconds=0.589619 live_host_mib=7863.8 rel_err=1.268e0
```

The finite-diff oracle is still noisy in full-model mode, so the load-bearing
evidence here is the controlled backward profile, not the failed full-model
relative error.

## What Changed

`matmul_bt_backward` now keeps the existing CUDA backend path reachable when
the backend is non-CPU by ensuring the saved left input, saved right input, and
upstream gradient have device handles before the existing residency check.

This is deliberately narrow:

- no new kernel;
- no new backward formula;
- CPU behavior unchanged;
- the existing host fallback remains the fallback when device residency is not
  available.

## Evidence

Local gates:

```text
cargo fmt --check
cargo check -p autograd --release --no-default-features --features no-cuda
CUDARC_CUDA_VERSION=12090 cargo check -p autograd --release --no-default-features --features cuda,no-cuda
cargo test -p autograd --release --lib
cargo test -p train --release --test test_moe_a0 -- --nocapture
cargo clippy -p autograd --release --no-default-features --features no-cuda -- -D warnings
cargo check -p train --release --no-default-features --features no-cuda --lib
cargo check -p train --release --no-default-features --features no-cuda --example qwen36_fp8_lora_fd_gate
CUDARC_CUDA_VERSION=12090 cargo check -p train --release --no-default-features --features cuda,no-cuda --example qwen36_fp8_lora_fd_gate
cargo clippy -p train --release --no-default-features --features no-cuda --example qwen36_fp8_lora_fd_gate -- -D warnings
```

Remote `.62`, GPU2/GPU4, GPU3 avoided:

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
log=/data01/arle-track1-route-frozen-fd-fast-20260617095440/a18_matmulbt_build_20260617_120351.log

cargo build -p train --example qwen36_fp8_lora_fd_gate --release --no-default-features --features cuda
PASS: Finished release target in 32.66s
```

Full-model backward profile after the change:

```text
log=/data01/arle-track1-route-frozen-fd-fast-20260617095440/a18_full_fd_profile_after_gpu2_20260617_120639.log

qwen36_fp8_lora_fd_backward_profile total_seconds=4.659262 op_seconds=2.377502 \
  prelude_seconds=0.005412 merge_grad_seconds=2.261452 op_kinds=18 site_kinds=1167
qwen36_fp8_lora_fd_backward_profile_op rank=1 op=MoeGroupedLinear count=120 seconds=1.528781 pct_total=32.812
qwen36_fp8_lora_fd_backward_profile_op rank=2 op=LinearAttention count=30 seconds=0.361495 pct_total=7.759
qwen36_fp8_lora_fd_backward_profile_op rank=5 op=MatmulBT count=1167 seconds=0.087045 pct_total=1.868
qwen36_fp8_lora_fd_gate_result analytic_seconds=5.220634 plus_seconds=0.570381 \
  minus_seconds=0.681975 live_host_mib=2517.0 rel_err=1.268e0
RUN_EXIT=1
```

MLP-layer finite-diff sanity:

```text
log=/data01/arle-track1-route-frozen-fd-fast-20260617095440/a18_mlp_fd_after_gpu4_20260617_120639.log

qwen36_fp8_lora_fd_backward_profile total_seconds=0.091317 op_seconds=0.039141
qwen36_fp8_lora_fd_backward_profile_op rank=1 op=MoeGroupedLinear count=3 seconds=0.035042
qwen36_fp8_lora_fd_gate_result analytic_seconds=0.099393 plus_seconds=0.007291 \
  minus_seconds=0.006128 target=model.language_model.layers.0.mlp.experts.210.up_proj.weight.lora_b \
  analytic=-2.581576268e-7 numeric=-2.587512427e-7 rel_err=2.294e-3
qwen36_fp8_lora_fd_gate PASS
RUN_EXIT=0
```

## Delta

| Metric | A17/A18-before | A18-after | Delta |
|---|---:|---:|---:|
| full-model backward total | 29.618619s | 4.659262s | -84.3% |
| full-model analytic arm | 30.190943s | 5.220634s | -82.7% |
| `MatmulBT` backward wall | 25.429740s | 0.087045s | -99.7% |
| `MatmulBT` share of backward | 85.9% | 1.9% | -84.0pp |
| live host memory | 7863.8 MiB | 2517.0 MiB | -68.0% |
| MLP-layer FD relative error | 2.294e-3 | 2.294e-3 | pass |

## Verdict

The dense frozen-base input-gradient path was falling back to host because the
small saved activations or upstream gradients lacked device handles at backward
time. Ensuring those tensors are resident before the existing device-path check
moves 1167 dense `MatmulBT` sites from 25.43s to 87ms.

The new full-model backward wall is no longer dense `MatmulBT`; it is
`MoeGroupedLinear` plus gradient merge. The full-model finite-diff oracle remains
unlicensed because the sparse full-model loss is still noisy, so correctness
remains anchored by the MLP-layer FD pass.

## Rule

For resident frozen-base QLoRA, saved activations and upstream gradients must be
made device-resident before deciding that a backward op cannot use the backend
fast path. The fallback should stay available, but it must not be selected just
because a small intermediate is still host-only.
