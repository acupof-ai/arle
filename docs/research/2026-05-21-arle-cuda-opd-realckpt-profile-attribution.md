# 2026-05-21 — Real-checkpoint CUDA OPD profile attribution

## Goal

Attribute the Qwen3-0.6B CUDA OPD real-checkpoint step that tripped the
2 s kill threshold, and pick exactly one next optimization axis.

## Hypothesis

The moderate-shape CUDA path is device-resident, so the 50x slowdown at
Qwen3-0.6B should be either a production-shape host fallback or a scaling
wall in a larger matrix path. Wall-clock phase totals are the primary
evidence; nsys is auxiliary because this harness has no NVTX step range.

## Command

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
target/release/examples/opd_step_cuda_realckpt_profile \
  | tee bench-output/2026-05-21-arle-cuda-opd-realckpt-profile/run.txt
```

Control experiment:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
ARLE_OPD_REALCKPT_PROFILE_DROP_HOST_MIRRORS=1 \
target/release/examples/opd_step_cuda_realckpt_profile \
  | tee bench-output/2026-05-21-arle-cuda-opd-realckpt-profile/run-drop-host-mirrors.txt
```

nsys:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
nsys profile --force-overwrite=true --trace=cuda,nvtx --stats=true \
  -o bench-output/2026-05-21-arle-cuda-opd-realckpt-profile/realckpt-profile \
  target/release/examples/opd_step_cuda_realckpt_profile \
  > bench-output/2026-05-21-arle-cuda-opd-realckpt-profile/nsys-stats.txt 2>&1
```

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB
- Driver: 595.71.05, CUDA runtime reported by nvidia-smi: 13.2
- Feature set: `--features cuda`
- Model: `/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B`
- Shape: hidden=1024, intermediate=3072, layers=28, vocab=151936,
  heads=16, kv_heads=8, head_dim=128, tied embedding/lm head
- Prompt: `[1, 872, 198, 3456]`
- Rollout length: 8, full rollout length 12
- LR: 5e-5, AdamW betas=(0.9, 0.999), eps=1e-8, wd=0
- Perturbation: uniform 1e-3 on trainable student params

## Results

Baseline warmed profile:

| Phase | Seconds | % step |
|---|---:|---:|
| rollout student forward | 6.521386 | 62.636 |
| optimizer step | 1.696729 | 16.297 |
| teacher forward | 0.802192 | 7.705 |
| student forward | 0.790930 | 7.597 |
| rollout argmax readback | 0.239541 | 2.301 |
| grad clip | 0.175356 | 1.684 |
| backward | 0.145223 | 1.395 |
| KL loss | 0.020096 | 0.193 |
| cleanup | 0.019932 | 0.191 |
| **total** | **10.411479** | **100.000** |

Backward is not the binding constraint:

| Backward op | Count | Seconds | % backward | % step |
|---|---:|---:|---:|---:|
| MatmulBT | 197 | 0.087300 | 60.115 | 0.838 |
| RMSNorm | 113 | 0.022294 | 15.352 | 0.214 |
| Transpose | 140 | 0.012166 | 8.378 | 0.117 |
| AddBroadcast | 84 | 0.009023 | 6.213 | 0.087 |
| RoPE | 56 | 0.004158 | 2.863 | 0.040 |

nsys full-process top slice:

| Signal | Value |
|---|---:|
| top GPU kernel family | `ampere_sgemm_64x32_sliced1x4_tn` |
| top GPU kernel family share | 1069.572 ms, 54.3 % of GPU kernel time |
| total GPU kernel time, inferred from top row | ~1.970 s |
| HtoD memcpy API time | 1.684 s across 19,204 calls |
| HtoD bytes | 9.740 GB |
| max HtoD / memset transfer | 622.330 MB |
| NVTX availability | none; nsys stats are whole-process, not step-ranged |

The 622.330 MB max transfer matches a single f32
`[151936, 1024]` tied embedding/lm-head-sized tensor. This does not by
itself prove lm_head compute dominance, but it is consistent with
production-vocab host mirror churn.

Control: after warmup, drop host-side mirrors for device-resident model
params with at least 1,000,000 elements. This cleared 394 tensors and
4,767,875,072 bytes (4547 MiB) of host mirrors. The same profiled step
then measured:

| Phase | Baseline s | Control s | Δ |
|---|---:|---:|---:|
| rollout student forward | 6.521386 | 0.123208 | -98.1 % |
| teacher forward | 0.802192 | 0.016482 | -97.9 % |
| student forward | 0.790930 | 0.018063 | -97.7 % |
| total step | 10.411479 | 1.266622 | -87.8 % |

This is the decisive attribution. The binding constraint is not CUDA
GEMM throughput or the matmul-decomposed causal-SDPA path. It is CPU-side
host mirror cloning in the CUDA tensor store before every device use.

## Root Cause

`TensorStore::ensure_device` clones `tensor.data` before checking whether
the tensor already has a usable device handle:

```rust
let (dirty, has_handle, data, shape) = {
    let tensor = self.tensor(id)?;
    (
        tensor.dirty.clone(),
        tensor.device_handle.is_some(),
        tensor.data.clone(),
        tensor.shape.clone(),
    )
};

if has_handle && dirty != Dirty::Host {
    return Ok(());
}
```

Loaded real-checkpoint weights are `Dirty::Both` after their first upload:
they have a device handle and still retain the full host mirror. Every
linear projection calls `ensure_device(weight)`. At Qwen3-0.6B, the tied
embedding/lm-head mirror alone is 622 MB, so the early-return path still
burns CPU memory bandwidth by cloning large host vectors that it discards.
The moderate shape hides this because its vocab and layer count are much
smaller.

## Checks

- `lm_head`: not directly tagged in nsys, but the max 622.330 MB transfer
  and the 4.55 GiB mirror-control delta both point at large vocab tensors
  as the highest-amplification case. Treat lm-head dominance as an
  inference, not a direct per-layer measurement, until NVTX ranges or
  per-linear counters land.
- `causal_sdpa`: not the current blocker at sequence length 12. In the
  backward profile, `Matmul` is 0.015 % of step and `Softmax` is 0.010 %
  of step. In nsys, `softmax_last_axis_f32` is 0.3 % of GPU kernel time.
- Host fallback: yes. The fallback is not an explicit CPU matmul fallback;
  it is CPU host-vector cloning on the device-resident fast path.
- Memory: before/after nvidia-smi both returned to 955 MiB used after
  process exit; no OOM. Peak was not continuously sampled in this tranche.

## Recommended Next Axis

**Axis H — fix `TensorStore::ensure_device` early-return clone and host
mirror retention.**

Single-variable implementation:

1. Check `has_handle && dirty != Dirty::Host` before cloning host data.
2. For CUDA-loaded model params, either avoid retaining host mirrors after
   first upload or clear the mirror when a tensor becomes device truth and
   no host fallback is expected. Preserve small host-only tensors needed by
   train-side helpers, especially RoPE cache row selection.
3. Re-run this exact harness and the existing moderate CUDA OPD bench.

Pre-licensed kill criterion:

- **License:** warmed real-checkpoint profile step <= 1.6 s and CPU/CUDA
  first-step loss relerr remains < 1e-4.
- **Strong license:** step <= 1.3 s, matching the profile-only
  host-mirror control.
- **KILL:** step remains > 2.0 s after the clone fix, because the control
  says the fix should remove at least 80 % of current wall time.

After Axis H, the next visible bottleneck will likely be AdamW
(`0.999 s`, 78.9 % of the control step), but optimizing AdamW before
fixing host mirrors would be optimizing behind the wrong wall.

## Artefacts

- Baseline profile:
  `bench-output/2026-05-21-arle-cuda-opd-realckpt-profile/run.txt`
- Host-mirror control:
  `bench-output/2026-05-21-arle-cuda-opd-realckpt-profile/run-drop-host-mirrors.txt`
- nsys stats:
  `bench-output/2026-05-21-arle-cuda-opd-realckpt-profile/nsys-stats.txt`
- GPU before/after:
  `bench-output/2026-05-21-arle-cuda-opd-realckpt-profile/nvidia-smi-before.txt`,
  `bench-output/2026-05-21-arle-cuda-opd-realckpt-profile/nvidia-smi-after.txt`
