# ARLE CUDA OPD Real-Checkpoint Convergence 10k Plateau Probe

## Goal

Extend the matched Qwen3-0.6B real-checkpoint OPD convergence run from 5000
steps to 10000 steps to find whether held-out KL / teacher-token NLL reaches a
true plateau.

Setup is unchanged from the 5k run:

- teacher: frozen Qwen3-0.6B checkpoint from ModelScope cache;
- student: same checkpoint, all trainable params perturbed by uniform
  `[-1e-3, 1e-3]`;
- prompt set: 32 training prompts, same 4 held-out prompts;
- optimizer: AdamW lr=`1e-7`;
- rollout: `rollout_len=8`;
- eval cadence: `0,100,500,1000,2500,5000,7500,10000`.

Verdict: **STILL IMPROVING at 10k**. Held-out exact overlap is flat after step
5000, but held-out KL keeps falling materially through step 10000. Held-out
NLL is close to flat, but still declines.

## Command

```bash
mkdir -p bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-10k-plateau
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-10k-plateau/nvidia-smi-before.txt

# Separate monitor session while the benchmark runs:
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
  --format=csv -l 5 \
  | tee -a bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-10k-plateau/nvidia-smi-monitor.csv

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- \
  --lr 1e-7 --steps 10000 --rollout-len 8 --prompt-set 32 \
  --eval-steps 0,100,500,1000,2500,5000,7500,10000 2>&1 \
  | tee bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-10k-plateau/run.txt

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-10k-plateau/nvidia-smi-after.txt
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
| peak observed by 5s monitor | 15358 | 586 | 75% |
| after | 955 | 14989 | 0% |

The monitor showed the same bounded sawtooth pattern as the 5k run. Memory
returned to the same idle value after process exit; no 10k-step leak was
observed.

## Results

Training wall-clock:

| Metric | Value |
|---|---:|
| total steps | 10000 |
| total loop wall seconds | 2114.929817 |
| mean OPD step seconds | 0.202297 |
| median OPD step seconds | 0.202261 |
| first sampled OPD loss | 1.788745e-5 |
| step 250 sampled OPD loss | 2.914809e-5 |
| final sampled OPD loss | 2.731737e-5 |
| train KL reduction at step 10000 | 79.367987% |

Eval trajectory:

| Step | Train exact % | Held-out exact % | Train KL | Held-out KL | Train NLL | Held-out NLL | Train top-3 % | Held-out top-3 % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 59.765625 | 50.000000 | 1.415704e-2 | 2.172812e-2 | 1.141632 | 1.299892 | 99.609375 | 100.000000 |
| 100 | 61.328125 | 51.562500 | 1.154891e-2 | 2.052191e-2 | 1.139141 | 1.293954 | 99.609375 | 100.000000 |
| 500 | 73.437500 | 64.062500 | 7.759120e-3 | 1.771600e-2 | 1.129206 | 1.280899 | 99.609375 | 100.000000 |
| 1000 | 77.929688 | 64.062500 | 6.254898e-3 | 1.623925e-2 | 1.126169 | 1.277434 | 99.804688 | 100.000000 |
| 2500 | 80.664062 | 75.000000 | 4.680479e-3 | 1.384123e-2 | 1.123260 | 1.269930 | 99.804688 | 100.000000 |
| 5000 | 85.742188 | 82.812500 | 3.575971e-3 | 1.179938e-2 | 1.120999 | 1.260116 | 99.804688 | 100.000000 |
| 7500 | 87.695312 | 82.812500 | 3.145602e-3 | 1.077652e-2 | 1.120133 | 1.254977 | 100.000000 | 100.000000 |
| 10000 | 89.453125 | 82.812500 | 2.920883e-3 | 1.019376e-2 | 1.119896 | 1.252497 | 100.000000 | 100.000000 |

Derived held-out deltas:

| Window | Held-out exact | Held-out KL | Held-out teacher NLL | Interpretation |
|---|---:|---:|---:|---|
| step 0 -> 10000 | +32.812500 pp | -53.08% | -3.65% | long run improves continuously |
| step 5000 -> 7500 | +0.000000 pp | -8.67% | -0.41% | exact flat, KL still material |
| step 7500 -> 10000 | +0.000000 pp | -5.41% | -0.20% | not KL-flat under the ±0.5% criterion |

## Interpretation

The true KL plateau was not reached by step 10000:

- held-out exact overlap is flat at `82.812500%` from step 5000 onward;
- held-out top-3 overlap is saturated at `100%` for the whole run;
- held-out teacher-token NLL is nearly flat late in the run, but still moves
  down by `0.20%` from step 7500 to step 10000;
- held-out KL remains the clearest signal and drops another `5.41%` from step
  7500 to step 10000.

The acceptance definition for TRUE PLATEAU was KL flat within `±0.5%` across
at least two eval points. The final KL window is far outside that band, so the
run is not a plateau.

## Verdict

**STILL IMPROVING at 10k.**

The ARLE CUDA OPD substrate continues to improve the perturbed Qwen3-0.6B
student under the fixed 32-prompt setup. The improvement is sub-linear and
NLL is close to flat, but KL still has measurable room.

Given the small held-out set, the best next eval axis is not another same-set
extension by default. Prefer widening supervision / held-out prompts or using
a real text prompt corpus. If the sole goal is to locate this exact prompt
set's KL floor, extend to 20k with the same cadence style.

## Verification

- 10000-step CUDA run: passed, no crash and no NaN in the emitted trajectory.
- GPU memory monitor: bounded sawtooth, peak observed `15358 MiB`, returned to
  `955 MiB` after process exit.
- No code changed in this tranche; this is a measurement and documentation
  commit.

## Artefacts

- `bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-10k-plateau/run.txt`
- `bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-10k-plateau/nvidia-smi-before.txt`
- `bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-10k-plateau/nvidia-smi-monitor.csv`
- `bench-output/2026-05-21-arle-cuda-opd-realckpt-convergence-10k-plateau/nvidia-smi-after.txt`
