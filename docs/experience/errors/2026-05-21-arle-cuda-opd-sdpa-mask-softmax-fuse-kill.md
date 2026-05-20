# ARLE CUDA OPD SDPA Mask-Softmax Fusion KILL

## Context

Axis: replace the temporary attention middle stack
`mul_scalar(scores, scale) -> add causal mask -> softmax` with one fused CUDA
kernel, while keeping the surrounding cuBLAS `QK^T` and `PV` matmuls.

This was option B from the 2026-05-21 SDPA brief. The hypothesis was that
removing 3 attention-related launches per layer would move the Qwen3-0.6B OPD
step from the post-KV-cache region toward the `<= 0.20 s/step` target.

No implementation code is retained. The fused op passed local numerical gates,
but wall-clock did not move enough and the moderate-shape bench regressed.

## Command

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo test -p autograd --test test_cuda_lazy_ops --release --features cuda

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_moderate_bench --release --features cuda \
  | tee bench-output/2026-05-21-arle-cuda-opd-sdpa-mask-softmax-fuse/moderate-run.txt

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-sdpa-mask-softmax-fuse/nvidia-smi-before.txt

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda \
  | tee bench-output/2026-05-21-arle-cuda-opd-sdpa-mask-softmax-fuse/realckpt-profile-run.txt

# repeated twice more as realckpt-profile-run-2.txt and realckpt-profile-run-3.txt
```

## Environment

- Commit under test: `5939cc7` plus uncommitted fused-mask-softmax patch
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB
- Feature set: `--features cuda`
- Real checkpoint: `/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B`
- Real profile prompt: `[1, 872, 198, 3456]`
- Rollout length: 8

GPU memory snapshots before and after the process both reported `955 MiB`
used, because snapshots were outside the benchmark process lifetime.

## Evidence

Numerical gates passed before the KILL decision:

| Gate | Result |
|---|---|
| `cargo check --workspace` | pass |
| `cargo test -p autograd --test test_cuda_lazy_ops --release --features cuda` | pass, 29 tests |
| `cargo test -p train --test test_opd_determinism --release` | pass |
| moderate CPU/CUDA OPD loss relerr | `1.641e-6` |

Qwen3-0.6B profile, n=3:

| Run | Step seconds | Rollout forward seconds | Grad clip seconds |
|---|---:|---:|---:|
| 1 | 0.247188 | 0.094874 | 0.035844 |
| 2 | 0.249639 | 0.096157 | 0.035996 |
| 3 | 0.263433 | 0.098855 | 0.044807 |
| mean | 0.253420 | 0.096629 | 0.038882 |
| median | 0.249639 | 0.096157 | 0.035996 |
| sigma / mean | 2.822% | n/a | n/a |

Acceptance frame:

| Criterion | Result | Status |
|---|---:|---|
| Qwen3-0.6B step `<= 0.20s` | mean `0.253420s` | fail |
| Qwen3-0.6B step `0.20s-0.25s` license-with-investigation | median `0.249639s`, mean `0.253420s` | borderline, not solid |
| Qwen3-0.6B step `> 0.25s` KILL | mean `0.253420s` | KILL |
| moderate-shape non-regression vs 58.7ms | `65.936ms` | fail |

The best comparison baseline is
`docs/experience/wins/2026-05-21-arle-cuda-opd-rollout-kvcache.md`, which
reported Qwen3-0.6B profile mean `0.253540s`, median `0.253510s`, and
moderate mean `0.058702s`. The attempted fusion produced real-profile mean
`0.253420s`, effectively unchanged, while moderate regressed to `0.065936s`
(`+12.32%`).

## Root Cause

The hypothesis was too narrow. At OPD sequence lengths (`q_len <= 12`,
decode rows often `q_len=1`), the scale/mask/softmax middle stack is not the
binding wall-clock cost. Removing its launches is offset by a more complex
custom masked-softmax kernel, and the measured backward contribution of the
new fused op was only about `0.08%` of total step time.

The phase table still points at rollout forward, grad clip, backward matmul
BT, optimizer, and cleanup. A middle-stack SDPA fusion does not attack those
dominant buckets.

## Fix

Killed the implementation and reverted all autograd code changes. No fused
SDPA code path is shipped.

## Rule

For short-sequence OPD attention, do not license an SDPA middle-stack fusion
from launch-count arithmetic. The wall-clock counter says this specific B
axis is noise on Qwen3-0.6B and a regression on moderate shape.

Next viable single-variable axis: attack `grad_clip` or rollout decode launch
count directly. Pre-license: Qwen3-0.6B step `<= 0.22s`, moderate shape no
slower than `61.6ms` (`+5%` from `58.7ms`), and CPU/CUDA loss relerr
`<= 1e-4`.

## Artefacts

- `bench-output/2026-05-21-arle-cuda-opd-sdpa-mask-softmax-fuse/moderate-run.txt`
- `bench-output/2026-05-21-arle-cuda-opd-sdpa-mask-softmax-fuse/realckpt-profile-run.txt`
- `bench-output/2026-05-21-arle-cuda-opd-sdpa-mask-softmax-fuse/realckpt-profile-run-2.txt`
- `bench-output/2026-05-21-arle-cuda-opd-sdpa-mask-softmax-fuse/realckpt-profile-run-3.txt`
- `bench-output/2026-05-21-arle-cuda-opd-sdpa-mask-softmax-fuse/nvidia-smi-before.txt`
- `bench-output/2026-05-21-arle-cuda-opd-sdpa-mask-softmax-fuse/nvidia-smi-after.txt`
