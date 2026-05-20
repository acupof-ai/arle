# PyTorch CUDA OPD moderate-step baseline

## Goal

Establish a PyTorch CUDA reference baseline for the OPD moderate-shape step.
This is the target ARLE's next OPD performance push should optimize toward.

## Hypothesis

A like-for-like PyTorch CUDA implementation of the existing moderate
Qwen3.5-style OPD step should be materially faster than the current ARLE CPU
moderate profile. The benchmark must match the topology and OPD semantics
directly instead of using HuggingFace Transformers.

## Params

- Backend: PyTorch CUDA
- Interpreter: `/home/ckl/projects/arle/.venv/bin/python`
- Torch: `2.11.0+cu130`
- CUDA build: `13.0`
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- Free memory at start: 15,491,465,216 bytes
- TF32: disabled for like-for-like FP32 semantics
- Shape: hidden=512, intermediate=1536, layers=12, vocab=32768
- Attention: heads=8, kv_heads=4, head_dim=64, gated q_proj, GQA, RoPE
- Prompt: `[1, 3, 8]`
- Rollout length: 2
- Optimizer: AdamW, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0
- Runs: 1 warmup, 3 measured, 10 OPD steps per measured run
- ARLE comparison point: current moderate target `0.83s/step`

Command:

```bash
timeout 1800s /home/ckl/projects/arle/.venv/bin/python \
  bench-output/2026-05-20-pytorch-cuda-opd-baseline/pytorch_cuda_opd_baseline.py \
  | tee bench-output/2026-05-20-pytorch-cuda-opd-baseline/run.txt
```

## Results

```text
run=1 wall_seconds=0.805836 per_step_seconds=0.080584 steps_per_sec=12.409469 first_loss=0.000314202 last_loss=0.000315392 peak_memory_bytes=1829115904
run=2 wall_seconds=0.854057 per_step_seconds=0.085406 steps_per_sec=11.708815 first_loss=0.000314202 last_loss=0.000315392 peak_memory_bytes=1829115904
run=3 wall_seconds=0.835476 per_step_seconds=0.083548 steps_per_sec=11.969222 first_loss=0.000314202 last_loss=0.000315392 peak_memory_bytes=1829115904
summary mean_step_seconds=0.083179 median_step_seconds=0.083548 sigma_pct=2.387 ratio_vs_arle_0p83=0.1002 speedup_vs_arle_0p83=9.9785
```

| Metric | Value |
|---|---:|
| mean step seconds | 0.083179 |
| median step seconds | 0.083548 |
| sigma / mean | 2.387% |
| ratio vs ARLE 0.83s/step | 0.1002x |
| PyTorch CUDA speedup vs ARLE 0.83s/step | 9.98x |
| peak allocated GPU memory | 1,829,115,904 bytes |

## Problems

This is a baseline reference, not an ARLE optimization. It uses a direct
PyTorch module that mirrors the train-side topology and OPD step semantics:
greedy rollout, teacher forward, student forward, KL distill loss, backward,
gradient clipping, and AdamW step. It does not validate output parity against
ARLE weights, because the target is wall-clock headroom at the same shape and
step structure.

## Learnings

The current moderate target has a roughly 10x gap to a straightforward PyTorch
CUDA implementation at the same shape. Future OPD performance work should use
this as the target frame and avoid spending time on sub-1% CPU-only cleanup
unless it is on the path to closing this CUDA baseline gap.

## Artefacts

- Script: `bench-output/2026-05-20-pytorch-cuda-opd-baseline/pytorch_cuda_opd_baseline.py`
- Raw run: `bench-output/2026-05-20-pytorch-cuda-opd-baseline/run.txt`
- JSON: `bench-output/2026-05-20-pytorch-cuda-opd-baseline/results.json`
- GPU env: `bench-output/2026-05-20-pytorch-cuda-opd-baseline/nvidia-smi.txt`
