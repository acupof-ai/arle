# ARLE CUDA OPD Device-Resident RoPE

## Goal

Remove the CUDA OPD rollout RoPE host fallback exposed while probing CUDA
Graph capture. Before this change, `CudaBackend` implemented `rope_forward`
for host slices but did not override `Backend::rope`; device-lazy RoPE
therefore used the default fallback:

```text
readback activation -> host rope -> upload
```

That was both a graph-capture blocker and a wall-clock cost in every rollout,
teacher, and student forward.

## Hypothesis

Reusing the existing `rope_f32` NVRTC kernel from a device-handle override
should keep Q/K activations device-resident, remove per-layer readbacks, and
reduce Qwen3-0.6B OPD step time without changing numerics.

## Command

Real-checkpoint profile, repeated three times:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda \
  | tee bench-output/2026-05-21-arle-cuda-opd-device-rope-probe/realckpt-profile-run.txt
```

Moderate non-regression:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_moderate_bench --release --features cuda \
  | tee bench-output/2026-05-21-arle-cuda-opd-device-rope-probe/moderate-run.txt
```

50-step convergence non-regression:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- \
  --lr 1e-7 --steps 50 \
  | tee bench-output/2026-05-21-arle-cuda-opd-device-rope-probe/convergence-50-lr1e-7.txt
```

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB
- Feature set: `--features cuda`
- Model: `/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B`
- Shape: hidden=1024, intermediate=3072, layers=28, vocab=151936,
  heads=16, kv_heads=8, head_dim=128, tied embeddings
- Real profile prompt: `[1, 872, 198, 3456]`
- Rollout length: 8

## Results

Qwen3-0.6B profile, n=3:

| Run | Step seconds | Rollout forward seconds |
|---|---:|---:|
| 1 | 0.207505 | 0.081800 |
| 2 | 0.209275 | 0.082761 |
| 3 | 0.209632 | 0.082771 |
| mean | 0.208804 | 0.082444 |
| median | 0.209275 | 0.082761 |
| sigma / mean | 0.455% | 0.554% |

Delta vs `2026-05-21-arle-cuda-opd-fused-grad-clip.md`:

| Metric | Before | After mean | Delta |
|---|---:|---:|---:|
| total step seconds | 0.231698 | 0.208804 | -9.88% |
| rollout forward seconds | ~0.095-0.100 | 0.082444 | about -13 ms |
| speedup | 1.00x | 1.11x | +1.11x |

Moderate OPD CUDA non-regression:

```text
summary mean_steps_per_sec=19.810302 median_steps_per_sec=19.817250 sigma_steps_per_sec=0.022759 sigma_pct=0.115 mean_step_seconds=0.050479 median_step_seconds=0.050461 max_loss_relative_error_vs_cpu=0.000001276
```

This passes the `<= 56 ms` moderate ceiling from the CUDA Graph brief and the
`<= 1e-4` CPU/CUDA loss-relative-error gate.

50-step convergence non-regression at `lr=1e-7`:

| Step | Train overlap % | Held-out overlap % | Train KL | Held-out KL |
|---:|---:|---:|---:|---:|
| 0 | 74.218750 | 50.000000 | 1.433391042756e-2 | 2.172812433186e-2 |
| 50 | 75.000000 | 50.000000 | 1.141541907359e-2 | 2.124995536075e-2 |

The trajectory matches the prior licensed lr=1e-7 step 0-50 behavior.

## Problems

This is not a CUDA Graph license. The graph probe separately showed that the
high-level rollout decode capture is still not replay-correct because captured
HtoD copies reference transient host buffers. That KILL is documented in
`docs/experience/errors/2026-05-21-arle-cuda-opd-rollout-graph-capture-kill.md`.

The new device `rope` path supports the full-head rotary shape exercised by
the existing CUDA `rope_f32` kernel. Partial-rotary CUDA RoPE remains outside
this measured path.

## Learnings

The graph probe found a real OPD bottleneck before graph replay was viable:
RoPE forward was only implemented as a host-slice helper on CUDA. Keeping RoPE
device-resident cuts about 23 ms from the Qwen3-0.6B OPD profile and moves the
step into the `0.18-0.21 s` license-with-investigation band from the graph
brief without shipping graph capture.

Next single-variable axis: remove transient host buffers from rollout decode
metadata if CUDA Graph capture remains the priority, or attack
`rollout_argmax_readback` / `post_step_cleanup` as smaller low-risk axes.

## Gates

- `cargo check --workspace`: passed
- `cargo test -p train --test test_opd_determinism --release`: passed
- `cargo test -p autograd --test test_cuda_lazy_ops --release --features cuda`: passed,
  including new `cuda_rope_device_lazy_matches_cpu`
- `cargo run -p train --example opd_step_cuda_moderate_bench --release --features cuda`: passed,
  mean `0.050479 s/step`, max CPU/CUDA loss relerr `1.276e-6`
- `cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda`: passed,
  mean `0.208804 s/step`, sigma `0.455%`
- `cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- --lr 1e-7 --steps 50`: passed,
  step50 train overlap `75.000000%`, train KL `1.141541907359e-2`

## Artefacts

- `bench-output/2026-05-21-arle-cuda-opd-device-rope-probe/realckpt-profile-run.txt`
- `bench-output/2026-05-21-arle-cuda-opd-device-rope-probe/realckpt-profile-run-2.txt`
- `bench-output/2026-05-21-arle-cuda-opd-device-rope-probe/realckpt-profile-run-3.txt`
- `bench-output/2026-05-21-arle-cuda-opd-device-rope-probe/moderate-run.txt`
- `bench-output/2026-05-21-arle-cuda-opd-device-rope-probe/convergence-50-lr1e-7.txt`
