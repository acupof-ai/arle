# ARLE CUDA OPD Real-Checkpoint Convergence 5k New Metrics

## Goal

Extend the Qwen3-0.6B real-checkpoint OPD convergence run to 5000 steps using
the continuous eval metrics added in
[`2026-05-21-arle-cuda-opd-eval-metric-fix.md`](2026-05-21-arle-cuda-opd-eval-metric-fix.md).

Setup is matched to the 500-step 32-prompt run:

- teacher: frozen Qwen3-0.6B checkpoint from ModelScope cache;
- student: same checkpoint, all trainable params perturbed by uniform
  `[-1e-3, 1e-3]`;
- prompt set: 32 training prompts, same 4 held-out prompts;
- optimizer: AdamW lr=`1e-7`;
- rollout: `rollout_len=8`;
- eval cadence: `0,100,250,500,1000,2000,3500,5000`.

Verdict: **STILL IMPROVING**. Exact held-out overlap flattened at step 3500,
but held-out KL and held-out teacher-token NLL both continued falling through
step 5000. The true KL/NLL plateau was not reached in this run.

## Command

```bash
mkdir -p bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-5k-newmetric
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-5k-newmetric/nvidia-smi-before.txt

nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
  --format=csv -l 5 \
  > bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-5k-newmetric/nvidia-smi-monitor.csv

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- \
  --lr 1e-7 --steps 5000 --rollout-len 8 --prompt-set 32 \
  --eval-steps 0,100,250,500,1000,2000,3500,5000 2>&1 \
  | tee bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-5k-newmetric/run.txt

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-5k-newmetric/nvidia-smi-after.txt
```

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB
- Feature set: `--features cuda`
- Model: `/home/ckl/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B`
- Shape: hidden=1024, intermediate=3072, layers=28, vocab=151936,
  heads=16, kv_heads=8, head_dim=128, tied embeddings
- Metrics: exact-1 greedy overlap, top-3 overlap, teacher-forced
  `KL(teacher || student)`, and teacher-token NLL

GPU memory snapshots:

| Snapshot | Used MiB | Free MiB | Utilization |
|---|---:|---:|---:|
| before | 955 | 14989 | 0% |
| peak observed by 5s monitor | 15326 | 618 | 99% |
| after | 955 | 14989 | 0% |

The monitor showed bounded sawtooth behavior during eval and training. Memory
returned to the same idle value after the run, so this run did not show a
5000-step leak.

## Results

Training wall-clock:

| Metric | Value |
|---|---:|
| total steps | 5000 |
| total loop wall seconds | 1097.298935 |
| mean OPD step seconds | 0.201210 |
| median OPD step seconds | 0.201287 |
| first sampled OPD loss | 1.788745e-5 |
| step 250 sampled OPD loss | 2.914809e-5 |
| final sampled OPD loss | 3.982126e-5 |
| train KL reduction at step 250 | 32.701483% |
| train KL reduction at step 5000 | 74.740694% |

Eval trajectory:

| Step | Train exact % | Held-out exact % | Train KL | Held-out KL | Train NLL | Held-out NLL | Train top-3 % | Held-out top-3 % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 59.765625 | 50.000000 | 1.415704e-2 | 2.172812e-2 | 1.141632 | 1.299892 | 99.609375 | 100.000000 |
| 100 | 61.328125 | 51.562500 | 1.154891e-2 | 2.052191e-2 | 1.139141 | 1.293954 | 99.609375 | 100.000000 |
| 250 | 69.335938 | 64.062500 | 9.527480e-3 | 1.913815e-2 | 1.134575 | 1.287159 | 99.609375 | 100.000000 |
| 500 | 73.437500 | 64.062500 | 7.759120e-3 | 1.771600e-2 | 1.129206 | 1.280899 | 99.609375 | 100.000000 |
| 1000 | 77.929688 | 64.062500 | 6.254898e-3 | 1.623925e-2 | 1.126169 | 1.277434 | 99.804688 | 100.000000 |
| 2000 | 79.687500 | 75.000000 | 5.026106e-3 | 1.446015e-2 | 1.123836 | 1.272165 | 99.804688 | 100.000000 |
| 3500 | 82.617188 | 82.812500 | 4.108234e-3 | 1.280286e-2 | 1.122180 | 1.264603 | 99.804688 | 100.000000 |
| 5000 | 85.742188 | 82.812500 | 3.575971e-3 | 1.179938e-2 | 1.120999 | 1.260116 | 99.804688 | 100.000000 |

Derived held-out deltas:

| Window | Held-out exact | Held-out KL | Held-out teacher NLL | Interpretation |
|---|---:|---:|---:|---|
| step 0 -> 500 | +14.062500 pp | -18.47% | -1.46% | matches the metric-fix run |
| step 0 -> 5000 | +32.812500 pp | -45.70% | -3.06% | long run keeps improving |
| step 3500 -> 5000 | +0.000000 pp | -7.84% | -0.35% | exact overlap flat, continuous metrics still down |

## Interpretation

This run answers the plateau question more precisely than exact-token overlap
could:

- exact held-out overlap improved from `50.000000%` to `82.812500%`, then stayed
  flat from step 3500 to step 5000;
- held-out KL improved monotonically at every eval point, including a further
  `7.84%` drop from step 3500 to step 5000;
- held-out teacher-token NLL also improved monotonically, though with a much
  smaller slope;
- top-3 overlap stayed saturated at `100%`, so it remains unsuitable for this
  held-out set.

The curves are visibly flattening, especially NLL, but they have not met the
plateau criterion: both continuous held-out metrics are still declining at the
last eval point.

## Verdict

**STILL IMPROVING.**

The OPD substrate continues to improve the perturbed Qwen3-0.6B student beyond
the 500-step result. The practical next axis depends on what we want to learn:

- to find the exact capacity limit of this prompt set, extend the same matched
  run to 10000 or 20000 steps;
- to improve eval confidence, widen the held-out set or switch from
  hand-picked token IDs to a real text prompt corpus before spending more
  compute on this small 4-prompt held-out set.

## Verification

- 5000-step CUDA run: passed, no crash and no NaN in the emitted trajectory.
- GPU memory monitor: bounded sawtooth, peak observed `15326 MiB`, returned to
  `955 MiB` after process exit.
- No code changed in this tranche; this is a measurement and documentation
  commit.

## Artefacts

- `bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-5k-newmetric/run.txt`
- `bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-5k-newmetric/nvidia-smi-before.txt`
- `bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-5k-newmetric/nvidia-smi-monitor.csv`
- `bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-5k-newmetric/nvidia-smi-after.txt`
