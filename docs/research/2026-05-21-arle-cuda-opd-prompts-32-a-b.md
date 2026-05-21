# ARLE CUDA OPD Prompt Set 32 A/B

## Goal

Test whether the held-out exact-token overlap plateau from the `lr=1e-7`
real-checkpoint OPD run is caused by too little supervision diversity.

Matched control:

- same Qwen3-0.6B teacher/student setup;
- same perturbation amplitude `1e-3`;
- same AdamW lr `1e-7`;
- same `rollout_len=8`;
- same 4 held-out prompts;
- training prompt set widened from 8 prompts to 32 prompts.

Verdict: **HELD-OUT FLAT at 64.062500% exact overlap; held-out KL improves**.

The 32-prompt run completed without crash or NaN. It reaches the same
`64.062500%` held-out exact-overlap ceiling by step 250 and remains there at
step 500. However, held-out KL at step 500 is `8.63%` lower than the 8-prompt
control, so wider supervision is still improving teacher-likelihood even though
the tiny 4-prompt exact-overlap metric does not move.

## Command

```bash
mkdir -p bench-output/2026-05-21-arle-cuda-opd-prompts-32-a-b
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-prompts-32-a-b/nvidia-smi-before.txt

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- \
  --lr 1e-7 --steps 500 --rollout-len 8 --prompt-set 32 \
  --eval-steps 0,50,100,250,500 2>&1 \
  | tee bench-output/2026-05-21-arle-cuda-opd-prompts-32-a-b/run.txt

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-prompts-32-a-b/nvidia-smi-after.txt
```

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB
- Feature set: `--features cuda`
- Model: `/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B`
- Shape: hidden=1024, intermediate=3072, layers=28, vocab=151936,
  heads=16, kv_heads=8, head_dim=128, tied embeddings
- Teacher: frozen Qwen3-0.6B checkpoint
- Student: same checkpoint, all trainable params perturbed by uniform
  `[-1e-3, 1e-3]`
- Optimizer: AdamW lr=`1e-7`, betas=(0.9, 0.999), eps=1e-8, wd=0
- Rollout: `rollout_len=8`
- Decode eval: greedy 16-token suffix
- A/B variable: `--prompt-set 32` instead of the default 8 training prompts

GPU memory snapshots before and after the process both reported `955 MiB`
used, because snapshots were outside the benchmark process lifetime.

## Harness Change

`crates/train/examples/opd_step_cuda_realckpt_train.rs` now accepts:

- `--prompt-set 8`
- `--prompt-set 32`

The default remains `8`, preserving previous controls. The first eight prompts
in the 32-prompt set are byte-identical to the original 8-prompt set, so the
first rotation is directly comparable.

## Training Prompts

The 32 training prompts used for this run:

```text
0:  [1, 872, 198, 3456]
1:  [1, 198, 1512, 429]
2:  [1, 770, 3186, 25, 220]
3:  [1, 644, 374, 279, 1887]
4:  [1, 3838, 374, 264, 2077, 13]
5:  [1, 785, 594, 287, 374, 1690]
6:  [1, 3347, 11, 358, 1052, 429]
7:  [1, 2610, 527, 1139, 304, 279, 1670]
8:  [1, 888, 536, 4697, 972]
9:  [1, 374, 11, 279, 1372, 315]
10: [1, 2874, 369, 279, 31559]
11: [1, 7521, 481, 362, 5714]
12: [1, 43059, 21938, 315, 7148]
13: [1, 358, 646, 944, 1490, 432]
14: [1, 477, 11, 323, 279, 62]
15: [1, 576, 1102, 315, 264, 729]
16: [1, 291, 504, 279, 1467, 11]
17: [1, 702, 1012, 1483, 311, 7512]
18: [1, 264, 11245, 2168, 429, 702]
19: [1, 3555, 374, 264, 5714, 30]
20: [1, 19257, 311, 279, 1251, 315]
21: [1, 1156, 3019, 304, 279, 1882]
22: [1, 2701, 1467, 25, 4710, 785]
23: [1, 315, 279, 3364, 13, 576]
24: [1, 279, 897, 5927, 553, 279]
25: [1, 2055, 11, 369, 279, 1140]
26: [1, 28469, 9363, 525, 279]
27: [1, 1012, 13570, 14975, 304, 279]
28: [1, 1887, 2242, 1294, 2827, 8]
29: [1, 62, 716, 477, 11, 323]
30: [1, 1512, 429, 374, 11, 279]
31: [1, 74595, 11, 714, 279, 1467]
```

Held-out prompts were unchanged from prior runs:

```text
0: [1, 4438, 374, 279, 2768]
1: [1, 1516, 374, 264, 1296, 4339]
2: [1, 785, 1401, 315, 279, 1967]
3: [1, 3198, 279, 1296, 25, 220]
```

## Safety

The run completed without crash or NaN. The first-five sampled losses match
the prior `rollout_len=8` run because the first eight prompts were preserved:

| Step | Loss | Rollout len | Step seconds |
|---:|---:|---:|---:|
| 1 | 1.788745430531e-5 | 12 | 0.168581 |
| 2 | 3.806550739682e-5 | 12 | 0.199473 |
| 3 | 3.408079282963e-5 | 13 | 0.200170 |
| 4 | 3.934769483749e-5 | 13 | 0.199365 |
| 5 | 2.925589251390e-5 | 14 | 0.200980 |

Training wall-clock:

| Metric | Value |
|---|---:|
| total steps | 500 |
| total loop wall seconds | 151.311043 |
| mean OPD step seconds | 0.200370 |
| median OPD step seconds | 0.200648 |
| first sampled OPD loss | 1.788745e-5 |
| step 250 sampled OPD loss | 2.914809e-5 |
| final sampled OPD loss | 2.667711e-5 |

## Results

32-prompt trajectory:

| Step | Train overlap % | Held-out overlap % | Train KL | Held-out KL |
|---:|---:|---:|---:|---:|
| 0 | 59.765625 | 50.000000 | 1.415704e-2 | 2.172812e-2 |
| 50 | 60.937500 | 51.562500 | 1.267499e-2 | 2.115785e-2 |
| 100 | 61.328125 | 51.562500 | 1.154891e-2 | 2.052191e-2 |
| 250 | 69.335938 | 64.062500 | 9.527480e-3 | 1.913815e-2 |
| 500 | 73.437500 | 64.062500 | 7.759120e-3 | 1.771600e-2 |

KL reductions from step 0:

| Point | Train KL reduction | Held-out KL reduction |
|---:|---:|---:|
| 250 | 32.70% | 11.92% |
| 500 | 45.19% | 18.47% |

Matched A/B against the 8-prompt `rollout_len=8` control:

| Step | Metric | 8 prompts | 32 prompts | Delta |
|---:|---|---:|---:|---:|
| 50 | held-out overlap % | 50.000000 | 51.562500 | +1.562500 pp |
| 50 | held-out KL | 2.124996e-2 | 2.115785e-2 | -0.43% |
| 100 | held-out overlap % | 64.062500 | 51.562500 | -12.500000 pp |
| 100 | held-out KL | 2.086548e-2 | 2.052191e-2 | -1.65% |
| 250 | held-out overlap % | 64.062500 | 64.062500 | 0.000000 pp |
| 250 | held-out KL | 2.003209e-2 | 1.913815e-2 | -4.46% |
| 500 | held-out overlap % | 64.062500 | 64.062500 | 0.000000 pp |
| 500 | held-out KL | 1.938898e-2 | 1.771600e-2 | -8.63% |

The train split is not directly comparable against the 8-prompt control because
the evaluated train set is now 32 prompts instead of 8. Within the 32-prompt
run itself, train KL falls `45.19%` and train exact overlap rises from
`59.77%` to `73.44%`.

Rollout length 16 control at step 500 also reached the same held-out overlap
ceiling:

| Config | Held-out overlap % | Held-out KL |
|---|---:|---:|
| 8 prompts, rollout 8 | 64.062500 | 1.938898e-2 |
| 8 prompts, rollout 16 | 64.062500 | 1.913682e-2 |
| 32 prompts, rollout 8 | 64.062500 | 1.771600e-2 |

## Interpretation

Widening from 8 to 32 training prompts does not break the held-out
exact-overlap plateau on this 4-prompt held-out set. The final value remains
`64.062500%`, which is exactly 41/64 matched decode tokens.

The KL signal says something different:

- held-out KL improves more than the 8-prompt and rollout-16 controls;
- the improvement is monotonic across the 32-prompt trajectory;
- exact overlap lags at step 100, then catches up by step 250.

The most SOLID conclusion is therefore narrow: **supervision diversity improves
held-out teacher-likelihood, but this 4-prompt exact-token metric is too coarse
or saturated to show a new overlap win.**

## Next Axis

Change the eval surface before making another conclusion about generalization:

- keep `lr=1e-7`, `perturb=1e-3`, `rollout_len=8`, and 32 training prompts
  fixed;
- expand held-out from 4 prompts to at least 32 prompts;
- use mean held-out per-token KL as the primary metric and exact-token overlap
  as a secondary metric;
- license if held-out KL drops at least `20%` by step 500 and the larger
  held-out exact overlap improves by at least `5 pp`;
- kill if held-out KL improves less than `5%` and exact overlap remains flat.

For a user-facing eval, replace hand-picked token ids with a tokenizer-backed
text prompt corpus. The token-id harness is still useful for controlled
substrate tests, but it is now the limiting eval resolution.

## Verification

- `cargo check -p train --example opd_step_cuda_realckpt_train --features cuda`:
  passed after adding `--prompt-set`.
- 500-step CUDA run with `--prompt-set 32`: passed, no crash / no NaN.

## Artefacts

- `bench-output/2026-05-21-arle-cuda-opd-prompts-32-a-b/run.txt`
- `bench-output/2026-05-21-arle-cuda-opd-prompts-32-a-b/nvidia-smi-before.txt`
- `bench-output/2026-05-21-arle-cuda-opd-prompts-32-a-b/nvidia-smi-after.txt`
