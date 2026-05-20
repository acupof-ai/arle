# ARLE CUDA OPD Rollout Inner Attribution

## Goal

Diagnosis: identify the next single optimization axis inside
`rollout_student_forward` for Qwen3-0.6B OPD after KV cache, device RoPE, and
device argmax.

## Hypothesis

The remaining rollout cost is not prefix re-encoding. It should be dominated by
short-sequence decode launch count and the temporary matmul-decomposed
attention/GQA path.

## Command

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda \
  | tee bench-output/2026-05-21-arle-cuda-opd-rollout-inner-profile/run-detail.txt

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
nsys profile --force-overwrite=true --trace=cuda,nvtx --stats=true \
  -o bench-output/2026-05-21-arle-cuda-opd-rollout-inner-profile/nsys-rollout-detail \
  target/release/examples/opd_step_cuda_realckpt_profile \
  | tee bench-output/2026-05-21-arle-cuda-opd-rollout-inner-profile/nsys-run-detail.txt
```

Raw artefacts:

- `bench-output/2026-05-21-arle-cuda-opd-rollout-inner-profile/run-detail.txt`
- `bench-output/2026-05-21-arle-cuda-opd-rollout-inner-profile/nsys-run-detail.txt`
- `bench-output/2026-05-21-arle-cuda-opd-rollout-inner-profile/nsys-rollout-top-kernels.txt`
- `bench-output/2026-05-21-arle-cuda-opd-rollout-inner-profile/nsys-rollout-top-api.txt`

## Environment

- Base commit: `5fb212a`
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, CUDA 13.x build path
- Model: `/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B`
- Shape: hidden 1024, intermediate 3072, layers 28, vocab 151936,
  heads 16, KV heads 8, head_dim 128
- Prompt: `[1, 872, 198, 3456]`
- Rollout length: 8 generated tokens, final rollout length 12

## Correctness Gate

The host greedy and device greedy rollout matched exactly:

```text
host=[1, 872, 198, 3456, 888, 536, 4697, 972, 262, 584, 1099, 737]
device=[1, 872, 198, 3456, 888, 536, 4697, 972, 262, 584, 1099, 737]
match=true
```

## Results

Primary in-process host enqueue/API wall-clock, not nsys-overhead timing and
not pure GPU elapsed time. This is still the same attribution frame used by
the existing `PhaseTotals`: elapsed wall time around Rust op calls, including
launch/API overhead and any backend synchronization inside those calls. CUDA
kernel execution is cross-checked separately with nsys below.

| Phase | Seconds | Share |
|---|---:|---:|
| total step | 0.196023 | 100.0% |
| rollout_student_forward | 0.080856 | 41.2% |
| backward | 0.027099 | 13.8% |
| optimizer_step | 0.025373 | 12.9% |
| post_step_cleanup | 0.018695 | 9.5% |
| grad_clip | 0.013414 | 6.8% |

Rollout per-iteration host enqueue/API trajectory:

| Iter | Mode | Seq len | Total ms | Attention ms | MLP ms |
|---:|---|---:|---:|---:|---:|
| 0 | prefill | 4 | 11.360 | 8.671 | 1.453 |
| 1 | decode | 1 | 10.059 | 7.673 | 1.296 |
| 2 | decode | 1 | 9.965 | 7.617 | 1.279 |
| 3 | decode | 1 | 9.952 | 7.591 | 1.290 |
| 4 | decode | 1 | 9.955 | 7.598 | 1.284 |
| 5 | decode | 1 | 9.878 | 7.512 | 1.290 |
| 6 | decode | 1 | 9.840 | 7.512 | 1.280 |
| 7 | decode | 1 | 9.845 | 7.496 | 1.284 |

The prefill row is only 1.13x a decode row. KV cache eliminated the earlier
O(N²) prefix re-encoding shape; the remaining wall-clock is per-token,
per-layer short-decode overhead.

Rollout component host enqueue/API totals:

| Component | Seconds | Share of rollout |
|---|---:|---:|
| attention | 0.061670 | 76.3% |
| MLP | 0.010456 | 12.9% |
| input RMSNorm | 0.002879 | 3.6% |
| post-attention RMSNorm | 0.002749 | 3.4% |
| all residual adds | 0.002686 | 3.3% |
| embedding + final norm + lm_head | 0.000302 | 0.4% |

Attention subcomponent host enqueue/API totals:

| Attention subcomponent | Seconds | Share of attention | Share of rollout |
|---|---:|---:|---:|
| `causal_sdpa_with_q_start` path | 0.012247 | 19.9% | 15.1% |
| `repeat_kv` materialization | 0.009739 | 15.8% | 12.0% |
| KV split layout | 0.007069 | 11.5% | 8.7% |
| RoPE | 0.006945 | 11.3% | 8.6% |
| Q/K RMSNorm | 0.004906 | 8.0% | 6.1% |
| Q layout | 0.003773 | 6.1% | 4.7% |
| merge heads | 0.003857 | 6.3% | 4.8% |
| Q/K/V/O projections combined | 0.010494 | 17.0% | 13.0% |

Single decode iter 1 has the same shape in the host enqueue/API frame:
attention is 7.673 ms of a 10.059 ms decode, with `sdpa` 1.478 ms and
`repeat_kv` 1.202 ms. Across 7 decode iters plus prefill, `sdpa + repeat_kv`
is 21.986 ms, or 27.2% of rollout and 11.2% of the whole OPD step in this
frame.

## Nsys Cross-Check

Nsys materially perturbs this short-kernel workload, so the wall-clock above is
the ground truth. The trace is still useful for launch-count attribution:

| NVTX window | Kernel instances | Kernel time |
|---|---:|---:|
| `opd_rollout_loop` | 6842 | 41.271 ms |
| `opd_rollout_iter_1_decode` | 844 | 4.186 ms |

Top kernels inside `opd_rollout_loop` were cuBLAS GEMV/GEMM variants and tiny
layout kernels. The top two internal GEMV groups alone contributed 22.072 ms
under nsys, followed by one 7.001 ms `gemv2T_kernel_val` group. The rollout
window also issued 5472 `cuMemcpyHtoDAsync_v2`, 6733 `cuMemsetD8Async`, and
4709 `cuLaunchKernel` API calls under nsys. That corroborates a launch-heavy
short-decode path rather than a long-sequence compute ceiling.

## Problems

- The profile helper uses CPU `Instant` timers around Rust op calls. Those are
  host enqueue/API attribution numbers. They include launch/API overhead and
  any explicit synchronization inside the backend, but they do not isolate pure
  GPU execution time.
- Nsys roughly doubled the step time for this workload. Its per-window timing
  is therefore diagnostic only; license-or-kill decisions should use the
  in-process wall-clock run.
- The attention subcomponents are measured from the profile-only wrapper. The
  production rollout path is not changed by this tranche.

## Binding Constraint

The next binding constraint is the GQA KV-cached one-token attention path in
the host enqueue/API frame, corroborated by the nsys kernel/API slice:
`repeat_kv` materializes repeated KV heads and then
`causal_sdpa_with_q_start` consumes the repeated tensors through the existing
decomposed attention stack. Together they account for 21.986 ms per OPD step in
the in-process attribution. That is larger than grad clip and comparable to
optimizer_step after the previous CUDA fusions.

## Recommended Next Axis

Implement a GQA-aware one-token CUDA `causal_sdpa_with_q_start` path for rollout
decode that consumes K/V as `[B, num_kv_heads, total_seq, head_dim]` directly
and maps attention head `h` to KV head `h / kv_repeat` inside the kernel. This
removes `repeat_kv` materialization and fuses the small `q_len=1` score,
softmax, and V accumulation path. Keep the current path for full-sequence
teacher/student forward and for CPU.

Pre-licensed criteria:

- License: Qwen3-0.6B OPD step <= 0.180 s, `rollout_student_forward` <= 0.065 s,
  rollout tokens unchanged, CPU determinism unchanged, CPU/CUDA loss relerr <=
  1e-4.
- License-with-investigation: step 0.180-0.190 s and `sdpa + repeat_kv` drops by
  at least 50%.
- Kill: step > 0.190 s or `sdpa + repeat_kv` remains > 15 ms. That means the
  wall-clock bottleneck is not this path, despite the subcomponent share.

Deferred: CUDA Graph capture remains plausible, but this trace shows a concrete
operation-level target first. Reducing graph launch overhead without removing
`repeat_kv` would leave the 22 ms attention/GQA work intact.
