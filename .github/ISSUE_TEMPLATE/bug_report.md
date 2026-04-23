---
name: Bug Report
about: Report a bug in agent-infer
title: "[bug] "
labels: bug
---

## Description

A clear description of the bug.

## Surface

- **Backend**: (e.g. CUDA, Metal, CPU)
- **Route / command**: (e.g. `POST /v1/chat/completions`, `arle train eval`)
- **Regression?**: (e.g. yes/no, last known good commit if known)

## Steps to Reproduce

1. Start server with `...`
2. Send request `...`
3. Observe `...`

## Expected Behavior

What should happen.

## Actual Behavior

What actually happens. Include error messages or logs.

## Environment

- **Backend**: (e.g. CUDA, Metal, CPU)
- **GPU / Metal chip / CPU**: (e.g. A100-40GB, M4 Pro)
- **CUDA / macOS / compiler version**: (e.g. CUDA 12.8, macOS 15.4)
- **agent-infer version/commit**: 
- **OS**: 
- **Model**: (e.g. Qwen3-4B)
- **Command / server flags**: (e.g. `infer --num-slots 4 --cuda-graph true`)
- **Relevant env vars**: (e.g. `INFER_CUDA_SM=90`)

## Evidence

- Logs / stack trace:
- Benchmark / trace / `/v1/stats` snapshot:
