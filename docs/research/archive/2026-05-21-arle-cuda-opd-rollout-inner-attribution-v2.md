# ARLE CUDA OPD Rollout Inner Attribution v2

## Goal

Diagnosis: refresh the Qwen3-0.6B OPD rollout attribution after fused
decode-time SDPA landed in `67607a0`, then pick the next single optimization
axis.

## Hypothesis

Fused decode SDPA should remove the old `sdpa + repeat_kv` binding constraint.
The new dominant rollout cost should be the remaining one-token decode layer
stack: tiny projection/layout/norm/RoPE/merge launches, not the attention
softmax kernel itself.

## Command

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_profile --release --features cuda \
  | tee bench-output/2026-05-21-arle-cuda-opd-rollout-inner-profile-v2/run.txt
```

Raw artefact:

- `bench-output/2026-05-21-arle-cuda-opd-rollout-inner-profile-v2/run.txt`

## Environment

- Code under test: `67607a0` (`perf(cuda): fuse OPD decode SDPA`)
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, CUDA 13.x build path
- Model: `/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B`
- Shape: hidden 1024, intermediate 3072, layers 28, vocab 151936,
  heads 16, KV heads 8, head_dim 128, tied embeddings
- Prompt: `[1, 872, 198, 3456]`
- Rollout length: 8 generated tokens, final rollout length 12

## Correctness Gate

The profile's host greedy and device greedy rollout matched exactly:

```text
host=[1, 872, 198, 3456, 888, 536, 4697, 972, 262, 584, 1099, 737]
device=[1, 872, 198, 3456, 888, 536, 4697, 972, 262, 584, 1099, 737]
match=true
```

## Results

Primary in-process host enqueue/API wall-clock, matching the `PhaseTotals`
frame used by prior OPD CUDA entries.

Top-5 phase summary:

| Rank | Phase | Seconds | Share of step |
|---:|---|---:|---:|
| 1 | rollout_student_forward | 0.063928 | 36.024% |
| 2 | backward | 0.027624 | 15.567% |
| 3 | optimizer_step | 0.025511 | 14.376% |
| 4 | post_step_cleanup | 0.016188 | 9.122% |
| 5 | grad_clip | 0.013652 | 7.693% |

Rollout per-iteration trajectory:

| Iter | Mode | Seq len | Total ms | Attention ms | MLP ms |
|---:|---|---:|---:|---:|---:|
| 0 | prefill | 4 | 11.431 | 8.699 | 1.488 |
| 1 | decode | 1 | 7.550 | 5.136 | 1.323 |
| 2 | decode | 1 | 7.545 | 5.157 | 1.309 |
| 3 | decode | 1 | 7.492 | 5.111 | 1.312 |
| 4 | decode | 1 | 7.499 | 5.120 | 1.303 |
| 5 | decode | 1 | 7.474 | 5.100 | 1.303 |
| 6 | decode | 1 | 7.483 | 5.119 | 1.302 |
| 7 | decode | 1 | 7.453 | 5.083 | 1.296 |

Rollout component totals:

| Component | Seconds | Share of rollout |
|---|---:|---:|
| attention | 0.044525 | 69.650% |
| MLP | 0.010636 | 16.637% |
| input RMSNorm | 0.002863 | 4.479% |
| post-attention RMSNorm | 0.002770 | 4.333% |
| attention residual add | 0.001391 | 2.175% |
| MLP residual add | 0.001336 | 2.090% |
| embedding | 0.000081 | 0.126% |
| final norm | 0.000101 | 0.159% |
| lm_head | 0.000105 | 0.165% |

Attention subcomponent totals:

| Attention subcomponent | Seconds | Share of attention | Share of rollout |
|---|---:|---:|---:|
| KV split layout | 0.006956 | 15.622% | 10.881% |
| RoPE | 0.006941 | 15.588% | 10.857% |
| Q/K RMSNorm | 0.004931 | 11.074% | 7.713% |
| Q layout | 0.003766 | 8.458% | 5.891% |
| merge heads | 0.003697 | 8.303% | 5.783% |
| fused decode SDPA | 0.003411 | 7.662% | 5.336% |
| O projection | 0.002984 | 6.702% | 4.668% |
| Q projection | 0.002728 | 6.127% | 4.268% |
| K projection | 0.002650 | 5.952% | 4.146% |
| append KV cache | 0.002529 | 5.679% | 3.956% |
| V projection | 0.002405 | 5.401% | 3.762% |
| repeat KV | 0.001408 | 3.163% | 2.203% |

Single decode iter 1 shows the same shape:

| Decode iter 1 subcomponent | Seconds | Share of decode attention | Share of iter |
|---|---:|---:|---:|
| RoPE | 0.000854 | 16.631% | 11.312% |
| KV split layout | 0.000852 | 16.597% | 11.290% |
| Q/K RMSNorm | 0.000621 | 12.100% | 8.230% |
| Q layout | 0.000464 | 9.034% | 6.145% |
| merge heads | 0.000457 | 8.903% | 6.056% |
| append KV cache | 0.000368 | 7.175% | 4.880% |
| O projection | 0.000361 | 7.024% | 4.778% |
| Q projection | 0.000343 | 6.684% | 4.546% |
| K projection | 0.000319 | 6.210% | 4.224% |
| V projection | 0.000288 | 5.600% | 3.809% |
| fused decode SDPA | 0.000193 | 3.758% | 2.556% |
| repeat KV | 0.000000 | 0.000% | 0.000% |

## Delta vs v1

| Metric | v1 before fused SDPA | v2 after fused SDPA | Delta |
|---|---:|---:|---:|
| total step seconds | 0.196023 | 0.177460 | -9.47% |
| rollout_student_forward | 0.080856 | 0.063928 | -20.94% |
| rollout attention | 0.061670 | 0.044525 | -27.80% |
| rollout SDPA | 0.012247 | 0.003411 | -72.15% |
| rollout repeat KV | 0.009739 | 0.001408 | -85.54% |

The fused decode SDPA axis moved the intended target. Decode `repeat_kv` is now
zero, and total `repeat_kv` is only the prefill fallback row. SDPA is no longer
the binding subcomponent.

## New Dominant Component

`rollout_student_forward` remains the top whole-step phase at 63.928 ms, and
attention remains the top rollout component at 44.525 ms. Within attention,
the new dominant actionable cluster is decode attention preparation/layout:

```text
kv_split + rope + qk_norm + q_layout + append_kv + merge =
0.028820 s = 64.7% of rollout attention = 45.1% of rollout = 16.2% of step
```

This is mostly many tiny one-token kernels and layout materializations. The
actual fused decode SDPA compute is only 3.411 ms across the whole rollout and
0.193 ms in decode iter 1.

## Recommendation

Next axis: `decode_attention_prepare_layout_fusion`.

Root-cause hypothesis: after SDPA fusion, one-token rollout decode is dominated
by launch/API overhead and device memory traffic from attention preparation and
layout transforms, not by the score/softmax/PV compute. Fuse the decode-only
Q/K/V preparation path into one or two CUDA kernels per layer:

- split projected Q/K/V outputs into head layout
- apply Q/K RMSNorm
- apply RoPE at the absolute decode position
- append K/V into the cache
- produce Q/K/V buffers in the layout consumed directly by fused decode SDPA
- optionally fold the post-SDPA merge layout into the same axis if the output
  layout can feed `o_proj` without another materialization

Keep the current full-sequence prefill and tape-enabled training forward paths
unchanged.

Pre-licensed criteria:

- License: Qwen3-0.6B mean step seconds <= 0.155, rollout_student_forward <=
  0.045 s, and the targeted prep/layout cluster above <= 0.014 s, with rollout
  tokens unchanged and CPU/CUDA loss relative error <= 1e-4.
- License-with-investigation: mean step seconds 0.155-0.165 and the targeted
  prep/layout cluster drops by at least 35%.
- Kill: mean step seconds > 0.170 or the targeted prep/layout cluster remains
  > 0.020 s. That would mean launch/layout fusion is not the wall-clock
  constraint despite the subcomponent attribution.

Deferred: MLP is the second-largest rollout component at 10.636 ms, but the
attention prep/layout cluster is roughly 2.7x larger and has a clearer
launch-fusion path. Attack MLP only after this cluster is licensed or killed.

## Problems

- This is a single diagnostic profile run, not an n=3 optimization bench. It is
  enough to pick the next axis because the top components are separated by
  millisecond-scale gaps, but the next optimization must use matched n>=3 A/B
  before licensing.
- The timers are CPU `Instant` spans around Rust op calls. They include host
  enqueue/API overhead and backend synchronizations inside those calls. They do
  not claim pure GPU kernel time.
- The workspace had an unrelated dirty CLI file during the run. The profiled
  train/autograd CUDA path was at `67607a0`; this doc-only tranche does not
  touch the dirty CLI file.

## Learnings

Fusing only the math core is no longer enough at `seq_len=1`. The OPD rollout
decode path is now limited by the surrounding per-layer preparation and layout
pipeline. The next useful experiment must reduce wall-clock launch count and
materialization around the fused SDPA kernel, then judge success by whole-step
seconds, not by a narrow attention-window percentage.
