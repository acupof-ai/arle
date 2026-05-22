# BF16 Stretch B KILL: Prompt 32 Was Capped by JSONL Rows

## Context

After licensing the BF16 frozen-base real OPD memory savings in
[`2026-05-22-bf16-substrate-4b-opd-memory-savings.md`](../wins/2026-05-22-bf16-substrate-4b-opd-memory-savings.md),
Stretch B was intended to spend the saved memory on longer prompt context:

- keep `rollout_len=8`
- change `--prompt-max-tokens 16` to `--prompt-max-tokens 32`
- keep the same 4B teacher, 0.8B LoRA student, prompt file, LR, and eval cadence

The run was started, then stopped after the startup logs proved the variable
was not actually changed.

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
  --steps 200 --rollout-len 8 --lr 1e-5 \
  --eval-steps 0,50,100,200 \
  --prompt-max-tokens 32 --max-step-seconds 30 \
  | tee bench-output/2026-05-22-qwen35-4b-08b-opd-bf16-stretch-prompt32/run.txt
```

Raw artefacts:

- `bench-output/2026-05-22-qwen35-4b-08b-opd-bf16-stretch-prompt32/run.txt`
- `bench-output/2026-05-22-qwen35-4b-08b-opd-bf16-stretch-prompt32/nvidia-smi-before.txt`
- `bench-output/2026-05-22-qwen35-4b-08b-opd-bf16-stretch-prompt32/nvidia-smi-monitor.csv`
- `bench-output/2026-05-22-qwen35-4b-08b-opd-bf16-stretch-prompt32/nvidia-smi-after.txt`

## Root Cause

`examples/opd/sample-prompts.jsonl` contains a per-row cap on every prompt:

```json
{"text":"Explain why small language models benefit from on-policy distillation.","max_tokens":16}
```

All 20 rows have `"max_tokens":16`. The loader honors the stricter row-local
cap, so the global `--prompt-max-tokens 32` did not expand any prompt.

Evidence:

```bash
diff -u \
  <(grep '^prompt split=' bench-output/2026-05-22-qwen35-4b-08b-opd-bf16-frozen-base/run.txt) \
  <(grep '^prompt split=' bench-output/2026-05-22-qwen35-4b-08b-opd-bf16-stretch-prompt32/run.txt)
```

The diff was empty. The supposed prompt32 stretch produced the same 20 prompt
token arrays as the licensed prompt16 run.

## Partial Run

The run was stopped after step 23 once the no-op variable was confirmed.

| Metric | Value |
|---|---:|
| Completed train steps before stop | 23 |
| Eval points emitted | 0 only |
| Peak GPU used during partial run | 13447 MiB |
| Mean partial step seconds | 5.490257 |

Startup eval matched the prompt16 run exactly:

| Step | Train KL | Held-out KL |
|---:|---:|---:|
| 0 | 1.510384544190e-5 | 1.739055323924e-5 |

## Kill Decision

KILL. This is not evidence about whether BF16 memory savings can buy longer
prompt context because the command did not actually create longer prompts.
Continuing to 200 steps would have repeated the already-licensed rollout8 /
prompt16 control and mixed up the conclusion.

## Fix

For a valid prompt-length stretch, use a prompt file whose row-local
`max_tokens` values are absent or set to 32. That should be a new matched
control with its own artefact directory, because changing the prompt file is a
different variable from only changing the CLI cap.

## Rule

For prompt-length experiments, license the tokenized prompt arrays before
licensing the run. `--prompt-max-tokens N` is only a hypothesis until the
startup log proves the emitted token arrays are actually longer.
