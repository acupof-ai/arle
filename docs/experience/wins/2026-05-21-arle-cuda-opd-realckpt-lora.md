# ARLE CUDA OPD Real-Checkpoint LoRA Bench

## Goal

Validate a LoRA-only OPD path for the Qwen3-0.6B real-checkpoint harness.
Production OPD users should not need full-finetune memory just to recover a
near-teacher student.

## Setup

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER 16 GB
- Checkpoint: `~/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B/`
- Teacher: frozen Qwen3-0.6B
- Student: shared frozen base weights from teacher + trainable LoRA adapters
- LoRA target set: attention `q_proj` + `v_proj`
- LoRA rank / alpha: `r=16`, `alpha=32`
- Trainable params: `2,293,760`
- Prompts: built-in 32 training prompts + 4 held-out prompts
- OPD: `rollout_len=8`, AdamW `lr=1e-5`, grad clip `1.0`, 500 steps

Command:

```bash
OUT=bench-output/2026-05-21-arle-cuda-opd-realckpt-lora
mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu \
  --format=csv > "$OUT/nvidia-smi-before.txt"
(nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
  --format=csv -l 1 > "$OUT/nvidia-smi-monitor.csv") &
MONITOR_PID=$!
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo run -p train --example opd_step_cuda_realckpt_lora_bench --release --features cuda -- \
  --lr 1e-5 --steps 500 --rollout-len 8 --prompt-set 32 --eval-steps 0,50,100,250,500 \
  2>&1 | tee "$OUT/run.txt"
kill "$MONITOR_PID"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu \
  --format=csv > "$OUT/nvidia-smi-after.txt"
```

## Results

| Mode | Trainable params | Mean step | Median step | Peak GPU memory | Held-out KL change |
|---|---:|---:|---:|---:|---:|
| Full-finetune OPD CUDA frontier | ~601 M | `0.164 s` bare step | n/a | `15358 MiB` | 10k run kept improving |
| LoRA q/v rank16 | `2.29 M` | `0.140092 s` | `0.140148 s` | `3934 MiB` | `7.63e-5 -> 4.86e-5` (`-36.39%`) |

LoRA beats the current full-finetune step time target and cuts peak observed
GPU memory by about `74%` (`15358 MiB -> 3934 MiB`). The base weights are
shared with the teacher copy; only adapter tensors are trainable and
AdamW-tracked.

Trajectory:

| Step | Train overlap | Held-out overlap | Train KL | Held-out KL | Held-out teacher-NLL |
|---:|---:|---:|---:|---:|---:|
| 0 | 96.09% | 100.00% | `6.9332e-5` | `7.6326e-5` | `1.236204` |
| 50 | 96.09% | 100.00% | `5.5905e-5` | `6.8526e-5` | `1.235781` |
| 100 | 96.09% | 100.00% | `4.8839e-5` | `6.3430e-5` | `1.235422` |
| 250 | 96.09% | 100.00% | `3.9383e-5` | `5.5536e-5` | `1.235133` |
| 500 | 98.63% | 100.00% | `3.1787e-5` | `4.8553e-5` | `1.234735` |

The exact-overlap metric is saturated for the held-out prompts because this
run starts near the teacher; KL remains the useful signal. Train KL drops
`54.15%`; held-out KL drops `36.39%`.

## Problems

The first all-linear rank16 probe was slower:

- target set: q/k/v/o/gate/up/down
- trainable params: `10,092,544`
- mean step: `0.217415 s`

That violated the "should beat full-finetune step time" goal. Root cause is
launch count and extra small adapter GEMMs across every linear in every layer.
The licensed default is therefore attention `q_proj` + `v_proj` only, matching
a common lightweight LoRA recipe and preserving the memory win.

The shared-base constructor still creates then retargets a temporary student
base on the host, so `student_load_seconds` is still about `6.15 s`. This is
startup-only and not in the OPD hot path. A future cleanup can construct LoRA
adapters directly from an existing base without the temporary host allocation.

## Correctness

- `lora_opd_step_cuda_matches_cpu_loss`: PASS, CPU/CUDA relerr
  `3.4429354e-7` (`1e-4` gate)
- `cargo test -p train --test test_opd_determinism --release`: PASS
- `cargo test -p train --release`: PASS
- `cargo check --workspace`: PASS
- `cargo clippy -p train --all-targets -- -D warnings`: PASS
- `cargo clippy -p train --all-targets --features cuda -- -D warnings`: PASS

## Rule

For OPD production recipes, prefer adapter target sets that move the held-out
KL curve while reducing both trainable params and launch count. "All linear"
is not automatically better at short sequence lengths; measure wall-clock and
peak memory before widening the adapter set.
