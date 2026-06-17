# A16 Qwen3.6 MoE resident base input-gradient

## Context

A15 measured the Qwen3.6 FP8 LoRA MoE backward wall inside the routed
down-projection base input-gradient path:

```text
dX = dY @ W_down
```

The base weight is frozen (`need_weight_grad=false`) and already resident on
CUDA as BF16/FP8 checkpoint storage, but the old grouped backward still
materialized active expert weights on host, transposed them, uploaded the f32
pack, then ran a small grouped matmul. The measured wall was not the GEMM:

```text
A15 call=1 pack/base_weight_t=0.279274s
A15 call=1 call_total=0.316998s
```

## What Changed

`moe_grouped_linear_backward` now has a guarded CUDA fast path for the exact
frozen-base input-gradient case:

- requires `need_input_grad && !need_weight_grad`;
- requires CUDA backend and every active expert weight to have a non-host
  device handle;
- calls `Backend::matmul_bt_backward_device` directly against the resident
  expert weight handle, including `CudaFp8BlockScaled`;
- returns to the previous host-pack grouped path if any active expert is not
  device-resident.

Shape mismatches remain hard errors; only missing residency falls back.

## Evidence

Local gates:

```text
cargo fmt --check
cargo check -p autograd --release --no-default-features --features no-cuda
CUDARC_CUDA_VERSION=12090 cargo check -p autograd --release --no-default-features --features cuda,no-cuda
cargo test -p train --release --test test_moe_a0 -- --nocapture
cargo test -p autograd --release --lib
cargo clippy -p autograd --release --no-default-features --features no-cuda -- -D warnings
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
INFER_TILELANG_PYTHON=/root/tl-venv/bin/python
```

Build:

```text
cargo build -p train --example qwen36_fp8_lora_fd_gate --release --no-default-features --features cuda
PASS: Finished release target in 35.15s
```

Profiled FD gate:

```text
log=/data01/arle-track1-route-frozen-fd-fast-20260617095440/moe_resident_grad_gpu0_20260617_111301.log

qwen36_fp8_lora_fd_gate ... --target-set all-linear --target-adapter auto:routed-up \
  --mode mlp-layer --layer 0 --eps 1e-3 --profile-backward

call=1 base_resident_input_grad total=0.001137s
call=1 call_total=0.014296s
call=2 call_total=0.010406s
call=3 call_total=0.010263s
rel_err=3.170e-3
qwen36_fp8_lora_fd_gate PASS
```

Clean timing gate without `ARLE_MOE_GROUPED_PROFILE`:

```text
log=/data01/arle-track1-route-frozen-fd-fast-20260617095440/moe_resident_grad_clean_gpu1_20260617_111334.log

qwen36_fp8_lora_fd_backward_profile total_seconds=0.089304
qwen36_fp8_lora_fd_backward_profile_op rank=1 op=MoeGroupedLinear count=3 seconds=0.033902

qwen36_fp8_lora_fd_gate_result analytic_seconds=0.183075
plus_seconds=0.092355 minus_seconds=0.091563
rel_err=3.170e-3
qwen36_fp8_lora_fd_gate PASS
```

## Delta

| Metric | A15 | A16 | Delta |
|---|---:|---:|---:|
| down-proj base path | host `pack/base_weight_t=0.279274s` | resident `base_resident_input_grad=0.001137s` | -99.6% |
| down-proj grouped call total | 0.316998s | 0.014296s | -95.5% |
| 3x MoE grouped-linear calls | ~0.346046s | 0.033902s clean profile | -90.2% |
| analytic arm wall | 0.361241s A14 baseline | 0.183075s | -49.3% |
| FD relative error | 3.170e-3 | 3.170e-3 | unchanged/pass |

## Verdict

The A15 root cause was correct: the expensive part was frozen base-weight
materialization, not the GEMM. The resident input-gradient fast path removes
the host transpose/upload for Qwen3.6 FP8 routed down-projection backward and
keeps the real-checkpoint MLP-layer finite-diff gate licensed.

This is not the final grouped-GEMM MoE backward kernel. It is the narrow
resident-weight tranche that eliminates the measured dominant wall. The next
remaining wall in this gate is gradient merge / allocation overhead, not base
weight pack.

## Rule

For frozen-base LoRA backward, do not materialize base weights just to compute
input gradients. Use the resident checkpoint handle and fall back only when
residency is absent.
