# ARLE CUDA OPD Eval Metric Fix

## Goal

Fix the real-checkpoint OPD eval surface after four matched runs converged to
the same held-out exact-token overlap plateau:

```text
64.062500% = 41 / 64 exact decode-token matches
```

The previous 32-prompt run showed the problem clearly: exact overlap stayed
flat, but held-out KL improved. This tranche adds continuous and more
interpretable held-out metrics to the real-checkpoint training harness.

Verdict: **use held-out KL as the primary metric; keep teacher-token NLL as a
secondary continuous metric; top-3 overlap is saturated on this held-out set**.

## Command

```bash
mkdir -p bench-output/2026-05-21-arle-cuda-opd-eval-metric-fix
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-eval-metric-fix/nvidia-smi-before.txt

NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_train --release --features cuda -- \
  --lr 1e-7 --steps 500 --rollout-len 8 --prompt-set 32 \
  --eval-steps 0,50,100,250,500 2>&1 \
  | tee bench-output/2026-05-21-arle-cuda-opd-eval-metric-fix/run.txt

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv \
  > bench-output/2026-05-21-arle-cuda-opd-eval-metric-fix/nvidia-smi-after.txt
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
- Prompt set: 32 training prompts, same 4 held-out prompts as
  [`../../research/2026-05-21-arle-cuda-opd-prompts-32-a-b.md`](../../research/2026-05-21-arle-cuda-opd-prompts-32-a-b.md)

GPU memory snapshots before and after the process both reported `955 MiB`
used, because snapshots were outside the benchmark process lifetime.

## Harness Change

`crates/train/examples/opd_step_cuda_realckpt_train.rs` now reports, for train
and held-out splits:

- `*_overlap_pct`: old exact-1 greedy decode overlap;
- `*_kl`: teacher-forced mean `KL(teacher || student)` over the teacher greedy
  suffix tokens;
- `*_teacher_nll`: student's mean NLL on the teacher greedy token at each
  teacher-forced row;
- `*_top3_overlap_pct`: whether the student's top-3 includes the teacher
  greedy token at each teacher-forced row.

The KL, NLL, and top-3 metrics use the same teacher-forced forward pass, so the
new metrics do not introduce a second decode path.

## Results

Training wall-clock:

| Metric | Value |
|---|---:|
| total steps | 500 |
| total loop wall seconds | 151.778853 |
| mean OPD step seconds | 0.200216 |
| median OPD step seconds | 0.200518 |
| first sampled OPD loss | 1.788745e-5 |
| step 250 sampled OPD loss | 2.914809e-5 |
| final sampled OPD loss | 2.667711e-5 |

Eval trajectory:

| Step | Held-out exact % | Held-out top-3 % | Held-out KL | Held-out teacher NLL |
|---:|---:|---:|---:|---:|
| 0 | 50.000000 | 100.000000 | 2.172812e-2 | 1.299892 |
| 50 | 51.562500 | 100.000000 | 2.115785e-2 | 1.297495 |
| 100 | 51.562500 | 100.000000 | 2.052191e-2 | 1.293954 |
| 250 | 64.062500 | 100.000000 | 1.913815e-2 | 1.287159 |
| 500 | 64.062500 | 100.000000 | 1.771600e-2 | 1.280899 |

Train trajectory for context:

| Step | Train exact % | Train top-3 % | Train KL | Train teacher NLL |
|---:|---:|---:|---:|---:|
| 0 | 59.765625 | 99.609375 | 1.415704e-2 | 1.141632 |
| 50 | 60.937500 | 99.609375 | 1.267499e-2 | 1.141652 |
| 100 | 61.328125 | 99.609375 | 1.154891e-2 | 1.139141 |
| 250 | 69.335938 | 99.609375 | 9.527480e-3 | 1.134575 |
| 500 | 73.437500 | 99.609375 | 7.759120e-3 | 1.129206 |

Derived held-out deltas:

| Metric | Step 0 -> 500 | Step 100 -> 500 | Interpretation |
|---|---:|---:|---|
| exact overlap | +14.062500 pp | +12.500000 pp | jumps, then plateaus by step 250 |
| top-3 overlap | 0.000000 pp | 0.000000 pp | saturated from the start |
| KL | -18.47% | -13.67% | strongest continuous signal |
| teacher NLL | -1.46% | -1.01% | continuous but weak signal |

## Interpretation

The new metrics separate three different facts:

- exact-token overlap is too coarse for this 4-prompt held-out set; it plateaus
  at the same `64.062500%` as prior runs;
- top-3 overlap is too loose; held-out is already `100%` at step 0, so it
  cannot distinguish training progress here;
- teacher-token NLL does capture progress, but the effect is small;
- full-distribution held-out KL remains the cleanest primary metric because it
  improves continuously past step 100 and is sensitive to distributional
  movement even when greedy tokens do not flip.

The right eval stack for the next OPD convergence runs is:

1. primary: held-out teacher-forced `KL(teacher || student)`;
2. secondary: held-out teacher-token NLL;
3. diagnostic only: exact-1 greedy overlap;
4. do not use top-3 on this held-out set unless the held-out prompts are made
   harder.

## Rule

For OPD eval at small held-out sizes, exact-token overlap is a coarse
decision-boundary metric, not a training-progress metric. Use a continuous
teacher-forced probability metric as the primary criterion, and only report
greedy overlap as interpretability context.

## Verification

- `cargo check -p train --example opd_step_cuda_realckpt_train --features cuda`:
  passed.
- 500-step CUDA run with `--prompt-set 32`: passed, no crash / no NaN.

## Artefacts

- `bench-output/2026-05-21-arle-cuda-opd-eval-metric-fix/run.txt`
- `bench-output/2026-05-21-arle-cuda-opd-eval-metric-fix/nvidia-smi-before.txt`
- `bench-output/2026-05-21-arle-cuda-opd-eval-metric-fix/nvidia-smi-after.txt`
