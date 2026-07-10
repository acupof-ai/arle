# BF16 Frozen-Base Stretch A: Rollout 16 Fits 4B OPD

## Goal

Test whether the BF16 frozen-base memory savings from
[`2026-05-22-bf16-substrate-4b-opd-memory-savings.md`](2026-05-22-bf16-substrate-4b-opd-memory-savings.md)
can buy a larger OPD rollout shape.

Single-variable change versus the licensed BF16 run:

- `rollout_len=16`
- `prompt_max_tokens=16` unchanged
- 200 steps, eval at 0/50/100/200 unchanged

License threshold:

- PASS: peak GPU memory <= 16384 MiB, no OOM, and KL remains monotonic.
- KILL: OOM or non-monotonic KL.

## Command

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
CUDARC_CUDA_VERSION=13010 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_BUILD_JOBS=1 \
cargo run -p train --example opd_step_cuda_infer_teacher_train --release --features cuda -- \
  --teacher-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B \
  --student-model /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base \
  --prompts-file examples/opd/sample-prompts.jsonl \
  --steps 200 --rollout-len 16 --lr 1e-5 \
  --eval-steps 0,50,100,200 \
  --prompt-max-tokens 16 --max-step-seconds 30 \
  | tee bench-output/2026-05-22-qwen35-4b-08b-opd-bf16-stretch-rollout16/run.txt
```

Raw artefacts:

- `bench-output/2026-05-22-qwen35-4b-08b-opd-bf16-stretch-rollout16/run.txt`
- `bench-output/2026-05-22-qwen35-4b-08b-opd-bf16-stretch-rollout16/nvidia-smi-before.txt`
- `bench-output/2026-05-22-qwen35-4b-08b-opd-bf16-stretch-rollout16/nvidia-smi-monitor.csv`
- `bench-output/2026-05-22-qwen35-4b-08b-opd-bf16-stretch-rollout16/nvidia-smi-after.txt`

## Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB class
- Driver/CUDA from `nvidia-smi`: 595.71.05 / 13.2
- Teacher: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-4B`
- Student: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base`
- Student mode: LoRA, rank 16, alpha 32, target set `attention-qv`
- Prompts: `examples/opd/sample-prompts.jsonl`
- Prompt split: 16 train, 4 held-out
- LR: `1e-5`
- CUDA graph: enabled for infer teacher runtime

## Results

The run completed without OOM or NaN. Stretch A is licensed as a larger shape
enabled by the BF16 frozen-base substrate, but it is not a quality win at 200
steps: rollout 16 fits, yet held-out KL improves slightly less than rollout 8.

| Metric | BF16 rollout 8 | BF16 rollout 16 | Delta |
|---|---:|---:|---:|
| Steps | 200 | 200 | matched |
| Rollout length flag | 8 | 16 | 2.0x |
| Mean step seconds | 5.439205 | 10.853569 | +99.54% |
| Median step seconds | 5.562066 | 10.979478 | +97.40% |
| Step sigma pct | 10.301% | 8.506% | -1.795 pp |
| Peak GPU used | 13447 MiB | 14503 MiB | +1056 MiB |
| Idle/non-bench baseline used | 1082 MiB | 1082 MiB | matched |
| Net peak over idle | 12365 MiB | 13421 MiB | +1056 MiB |

Acceptance threshold was 16384 MiB. The measured peak was 14503 MiB, leaving
1873 MiB of nominal headroom on the 16 GiB card.

KL trajectory:

| Step | Train KL | Held-out KL |
|---:|---:|---:|
| 0 | 1.510384544190e-5 | 1.739055323924e-5 |
| 50 | 1.504912705741e-5 | 1.731940187710e-5 |
| 100 | 1.500015440570e-5 | 1.724250546431e-5 |
| 200 | 1.488929962079e-5 | 1.705238992145e-5 |

Reduction at step 200:

| Metric | BF16 rollout 8 | BF16 rollout 16 |
|---|---:|---:|
| Train KL delta | -1.744% | -1.420% |
| Held-out KL delta | -2.091% | -1.945% |

Average phase attribution over 200 steps:

| Phase | Avg seconds | Share |
|---|---:|---:|
| Student rollout | 6.652289 | 61.29% |
| Backward | 3.086258 | 28.44% |
| Teacher forward total | 0.632537 | 5.83% |
| Student KL forward | 0.446260 | 4.11% |
| Post-step cleanup | 0.024114 | 0.22% |
| KL loss compute + readback | 0.010099 | 0.09% |
| Optimizer step | 0.000803 | 0.01% |
| Grad clip | 0.000749 | 0.01% |

## What Worked

The BF16 frozen-base savings bought the larger rollout shape on the same 16 GiB
card. The shape roughly doubled step time, but it did not approach OOM: peak
memory was 14503 MiB, well below the 16384 MiB license ceiling.

## Problems

Rollout 16 did not improve the 200-step KL result. Held-out KL still decreased
monotonically, but the reduction was -1.945% versus -2.091% for rollout 8.
For this prompt set, LR, and 200-step budget, longer rollout is a fit/shape win
rather than a quality win.

The sampled per-step loss increased from first to final prompt sample. That is
not the acceptance metric here because prompt lengths and prompt IDs rotate,
but it is a useful warning: rollout 16 may need retuned LR or longer training
before it turns into a quality improvement.

## Rule

Do not assume that spending saved memory on longer rollout improves OPD quality.
The wall-clock and memory evidence licenses the larger shape; the matched KL
comparison says the next quality axis should be prompt/token coverage or LR
tuning, not rollout length alone.
