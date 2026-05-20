# ARLE CUDA OPD Rollout Fused Decode SDPA

## Goal

Replace the decode-time OPD rollout attention chain with one fused CUDA kernel
for `seq_q=1`, `seq_kv<=32`, and GQA K/V heads. The pre-license target was
`<= 0.155 s/step`; `0.155-0.180 s/step` was license-with-investigation. The
previous stable profile measured `0.195945 s/step`.

## Hypothesis

The rollout-inner attribution showed attention was still the largest rollout
component after KV-cache and device argmax. The decode path was paying a
repeat-KV materialization plus the decomposed `QK^T -> scale/mask -> softmax ->
PV` chain for each layer and rollout iteration. At `seq_kv<=12`, one CTA per
`(batch, query_head)` should compute the full decode attention cheaply enough
to remove launch count and repeated-KV overhead.

## Command

Real-checkpoint profile, repeated three times:

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
- AdamW betas=(0.9, 0.999), eps=1e-8, wd=0

## Results

Qwen3-0.6B profile, n=3:

| Run | Step seconds | Rollout forward seconds | Rollout attention seconds | Rollout SDPA seconds |
|---|---:|---:|---:|---:|
| 1 | 0.177680 | 0.063920 | 0.044478 | 0.003416 |
| 2 | 0.177418 | 0.063575 | 0.044276 | 0.003366 |
| 3 | 0.177058 | 0.064210 | 0.044653 | 0.003433 |
| mean | 0.177385 | 0.063902 | 0.044469 | 0.003405 |
| median | 0.177418 | 0.063920 | 0.044478 | 0.003416 |
| sigma / mean | 0.144 % | 0.406 % | 0.347 % | 0.817 % |

Delta vs the prior device-argmax entry:

| Metric | Before | After mean | Delta |
|---|---:|---:|---:|
| total step seconds | 0.195945 | 0.177385 | -9.47 % |
| rollout student forward | 0.080617 | 0.063902 | -20.73 % |
| rollout attention | 0.061670 | 0.044469 | -27.89 % |
| speedup | 1.00x | 1.10x | +1.10x |

Rollout equivalence probe from all three profile runs:

```text
host=[1, 872, 198, 3456, 888, 536, 4697, 972, 262, 584, 1099, 737]
device=[1, 872, 198, 3456, 888, 536, 4697, 972, 262, 584, 1099, 737]
match=true
```

Moderate OPD CUDA non-regression:

```text
summary mean_steps_per_sec=20.531122 median_steps_per_sec=20.602802 sigma_steps_per_sec=0.148175 sigma_pct=0.722 mean_step_seconds=0.048709 median_step_seconds=0.048537 max_loss_relative_error_vs_cpu=0.000001276
```

50-step convergence non-regression at `lr=1e-7`:

| Step | Train overlap % | Held-out overlap % | Train KL | Held-out KL |
|---:|---:|---:|---:|---:|
| 0 | 74.218750 | 50.000000 | 1.433391042756e-2 | 2.172812433186e-2 |
| 50 | 75.000000 | 50.000000 | 1.141541907359e-2 | 2.124995536075e-2 |

This matches the expected step 0-50 behavior from the lr=1e-7 convergence run:
train overlap stays stable and train KL improves by 20.36%.

## Problems

This is license-with-investigation, not a full license. The step landed inside
the accepted `0.155-0.180 s` band but missed the `<= 0.155 s` target. The fused
kernel removed the intended decode SDPA/repeat-KV cost, but the remaining
rollout attention time is now mostly projection/layout/norm/RoPE/merge/O-proj
launch overhead rather than the SDPA kernel itself. In the profile, decode
iter 1 SDPA is only `0.197 ms` out of `5.150 ms` attention.

No `nsys` safety profile was required because the first runnable fused path was
`0.177680 s`, below the `0.22 s` stop-and-profile threshold. The host-enqueue
phase table is enough to identify that the kernel dispatched and that `repeat_kv`
is zero on decode rows.

## Learnings

Full decode SDPA fusion was a real wall-clock win, but the short-sequence OPD
rollout is now bounded by the rest of the one-token layer stack. The next
single-variable axis should target decode-layer launch count outside SDPA:
fuse or graph-capture the projection/layout/norm/RoPE/merge cluster for one
decode token. Pre-license: Qwen3-0.6B step `<= 0.155 s`, n>=3, sigma `<5%`;
kill if total stays above `0.172 s`, because SDPA is no longer the dominant
subcomponent.

## Gates

- `cargo check -p autograd --features cuda`: passed
- `cargo check -p train --features cuda`: passed
- `cargo check --workspace`: passed
- `cargo clippy -p autograd --features cuda,no-cuda -- -D warnings`: passed
- `cargo clippy -p autograd --features cuda -- -D warnings`: passed
- `cargo build --workspace --release`: passed
- `cargo test -p train qwen35_rollout_kv_cache_matches_full_forward_tokens --release`: passed
- `cargo test -p autograd --test test_cuda_lazy_ops cuda_causal_sdpa_decode_gqa_matches_cpu --release --features cuda`: passed
- `cargo test -p train --test test_opd_determinism --release`: passed
- `cargo test -p autograd --release --features cuda`: passed
- `cargo test -p train --release`: passed
- `cargo run -p train --example opd_step_cuda_moderate_bench --release --features cuda`: passed,
  median `0.048537 s/step`, max CPU/CUDA loss relerr `1.276e-6`
- `cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda`: passed,
  mean `0.177385 s/step`, sigma `0.144%`
- `cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- --lr 1e-7 --steps 50`: passed,
  step50 train overlap `75.000000%`, train KL `1.141541907359e-2`

## Artefacts

- Real-checkpoint profile:
  `bench-output/2026-05-21-arle-cuda-opd-sdpa-decode-fused/realckpt-profile-run.txt`,
  `bench-output/2026-05-21-arle-cuda-opd-sdpa-decode-fused/realckpt-profile-run2.txt`,
  `bench-output/2026-05-21-arle-cuda-opd-sdpa-decode-fused/realckpt-profile-run3.txt`
- Moderate CUDA OPD:
  `bench-output/2026-05-21-arle-cuda-opd-sdpa-decode-fused/moderate-bench-run.txt`
- Convergence non-regression:
  `bench-output/2026-05-21-arle-cuda-opd-sdpa-decode-fused/realckpt-train-50-lr1e-7.txt`
