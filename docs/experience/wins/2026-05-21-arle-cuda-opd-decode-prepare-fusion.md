# ARLE CUDA OPD Decode Prepare/Layout Fusion

## Goal

Optimization: fuse the decode-only Qwen OPD rollout attention prepare/layout
chain after the fused decode SDPA tranche. Pre-license target was
Qwen3-0.6B OPD `<= 0.155 s/step`; `0.155-0.170 s/step` was
license-with-investigation.

## Hypothesis

After `67607a0`, fused decode SDPA was no longer the binding component. The v2
rollout-inner attribution showed the remaining attention cost was dominated by
decode prep/layout: Q/K/V split/layout, Q/K RMSNorm, RoPE, KV append, and merge.
Fusing the post-projection Q and K/V prepare chain for decode-only rollout
should remove most of the small-kernel layout/norm/RoPE cost without changing
prefill or tape-enabled training forward.

## Command

Real-checkpoint profile, serial n=3:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda
```

Moderate non-regression:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_moderate_bench --release --features cuda
```

50-step convergence non-regression:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- \
  --lr 1e-7 --steps 50
```

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB
- Driver/runtime: CUDA 13.x path, `--features cuda`
- Model: `/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B`
- Shape: hidden=1024, intermediate=3072, layers=28, vocab=151936,
  heads=16, kv_heads=8, head_dim=128, tied embeddings
- Real profile prompt: `[1, 872, 198, 3456]`
- Rollout length: 8, final rollout length 12
- LR: 5e-5 for profile, 1e-7 for convergence non-regression

## What Changed

Two decode-only CUDA kernels were added behind the existing backend trait:

- `qwen_decode_prepare_q_f32`: consumes post-projection Q, emits
  `[B, query_heads, 1, head_dim]` Q after split/layout, per-head RMSNorm, and
  absolute-position RoPE. A gated variant also emits the raw Q gate in
  head-major layout, preserving the existing `sigmoid(gate) * attention`
  order.
- `qwen_decode_prepare_kv_f32`: consumes post-projection K/V, emits head-major
  K after RMSNorm+RoPE and head-major V.

The fast path is CUDA-only, tape-disabled, `seq_len == 1`, and
`rotary_dim == head_dim`. Prefill, CPU, Metal, partial-RoPE configs, and
tape-enabled teacher/student KL forward keep the previous path.

## Results

Qwen3-0.6B profile, n=3:

| Run | Step seconds | Rollout seconds | Attention seconds |
|---|---:|---:|---:|
| 1 | 0.165288 | 0.048592 | 0.029400 |
| 2 | 0.163614 | 0.048388 | 0.029384 |
| 3 | 0.164259 | 0.048480 | 0.029435 |
| mean | 0.164387 | 0.048487 | 0.029406 |
| median | 0.164259 | 0.048480 | 0.029400 |
| sigma / mean | 0.42% | 0.17% | 0.07% |

Delta vs the v2 post-SDPA profile:

| Metric | Before | After mean | Delta |
|---|---:|---:|---:|
| total step seconds | 0.177460 | 0.164387 | -7.37% |
| rollout_student_forward | 0.063928 | 0.048487 | -24.15% |
| rollout attention | 0.044525 | 0.029406 | -33.95% |
| speedup | 1.00x | 1.08x | +1.08x |

Targeted prep/layout cluster:

| Cluster | Before | After mean | Delta |
|---|---:|---:|---:|
| `q_layout + kv_split + qk_norm + rope + append_kv + merge` | 0.028820 | 0.012738 | -55.80% |

The targeted cluster hit the `<= 0.014 s` full-license sub-target, but total
step missed the `<= 0.155 s` full-license target. Verdict:
**license-with-investigation**.

Rollout equivalence matched in all three profile runs:

```text
host=[1, 872, 198, 3456, 888, 536, 4697, 972, 262, 584, 1099, 737]
device=[1, 872, 198, 3456, 888, 536, 4697, 972, 262, 584, 1099, 737]
match=true
```

Moderate OPD CUDA non-regression:

```text
summary mean_steps_per_sec=21.230264 median_steps_per_sec=21.273692 sigma_steps_per_sec=0.080535 sigma_pct=0.379 mean_step_seconds=0.047103 median_step_seconds=0.047006 max_loss_relative_error_vs_cpu=0.000001276
```

50-step convergence non-regression at `lr=1e-7`:

| Step | Train overlap % | Held-out overlap % | Train KL | Held-out KL |
|---:|---:|---:|---:|---:|
| 0 | 74.218750 | 50.000000 | 1.433391042756e-2 | 2.172812433186e-2 |
| 50 | 75.000000 | 50.000000 | 1.141541907359e-2 | 2.124995536075e-2 |

This matches the expected 0->50 trajectory: train overlap stays stable and
train KL improves by 20.36%.

## Problems

- This is not a full license. Mean step time landed in the accepted
  `0.155-0.170 s` band but missed `<= 0.155 s`.
- The fused kernels are intentionally full-RoPE only in this tranche. A review
  pass caught that valid Qwen3.5/3.6 partial-RoPE configs must not auto-enter
  this path; those configs now stay on the previous layout/RoPE chain.
- Two attempted repeat profiles were accidentally launched concurrently and hit
  CUDA allocation failure on the 16 GiB card. Those runs were discarded; the
  table above uses three serial profile runs.
- The remaining attention time is now dominated by projection/append/merge/O
  projection and not by Q/K RMSNorm or RoPE. Fusing the requested prepare path
  moved the target, but more of the one-token layer stack must be fused to
  reach `<= 0.155 s`.

## Learnings

The decode prepare/layout hypothesis was correct: the targeted cluster dropped
55.80% and wall-clock step improved 7.37%. The next single axis should be
decode attention output/append fusion:

- Root-cause hypothesis: `append_kv + merge + o_proj` are now a larger combined
  rollout-attention share than Q/K prep. `append_kv` still allocates and
  concatenates cache tensors, and `merge` materializes head-major output before
  the O projection consumes it.
- Pre-license: Qwen3-0.6B mean step seconds `<= 0.150`, rollout
  `<= 0.042 s`, with rollout tokens unchanged and CPU/CUDA loss relerr
  `<= 1e-4`.
- Kill: mean step `> 0.160` or the targeted `append_kv + merge + o_proj`
  cluster remains above `0.008 s`.

## Gates

- `cargo check -p autograd --features cuda`: passed
- `cargo check -p train --features cuda`: passed
- `cargo check --workspace`: passed
- `cargo clippy -p autograd --features cuda -- -D warnings`: passed
- `cargo clippy -p train --features cuda -- -D warnings`: passed
- `cargo build --workspace --release`: passed
- `cargo test -p autograd --test test_cuda_lazy_ops cuda_qwen_decode_prepare_q_and_kv_match_cpu --release --features cuda`: passed
- `cargo test -p train qwen35_rollout_kv_cache_matches_full_forward_tokens --release`: passed
- `cargo test -p train --test test_opd_determinism --release`: passed
- `cargo test -p autograd --release --features cuda`: passed
- `cargo test -p train --release`: passed
- `cargo run -p train --example opd_step_cuda_moderate_bench --release --features cuda`: passed,
  median `0.047006 s/step`, max CPU/CUDA loss relerr `1.276e-6`
- `cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda`: passed,
  mean `0.164387 s/step`, sigma `0.42%`
- Post-review full-RoPE fast-path sanity profile: passed, `0.164680 s/step`,
  rollout tokens matched
- `cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- --lr 1e-7 --steps 50`: passed,
  step50 train overlap `75.000000%`, train KL `1.141541907359e-2`

## Artefacts

- Real-checkpoint profile:
  `bench-output/2026-05-21-arle-cuda-opd-decode-prepare-fused/realckpt-profile-run1.txt`,
  `bench-output/2026-05-21-arle-cuda-opd-decode-prepare-fused/realckpt-profile-run2.txt`,
  `bench-output/2026-05-21-arle-cuda-opd-decode-prepare-fused/realckpt-profile-run3.txt`,
  `bench-output/2026-05-21-arle-cuda-opd-decode-prepare-fused/realckpt-profile-final-check.txt`
- Moderate CUDA OPD:
  `bench-output/2026-05-21-arle-cuda-opd-decode-prepare-fused/moderate-bench-run.txt`
- Convergence non-regression:
  `bench-output/2026-05-21-arle-cuda-opd-decode-prepare-fused/realckpt-train-50-lr1e-7.txt`
