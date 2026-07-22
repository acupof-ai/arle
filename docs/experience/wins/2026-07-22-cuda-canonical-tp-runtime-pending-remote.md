# Canonical TP runtime construction — CUDA L4, 2026-07-22

> Status: pending-remote

## Goal

Preserve CUDA model-load behavior while sharing one TP runtime constructor across Qwen3, Qwen3.5/3.6, and DSv4.

## Hypothesis

The refactor is behavior-neutral: Qwen loaders pass `pin_numa=false`; DSv4 passes `true`; multi-rank initialization retains config → ordinal → optional NUMA pin → `cudaSetDevice` → NCCL ID → NCCL init order.

## Parameters

Final matched L4 parameters are pending remote execution under `docs/bench-and-trace-spec.md`.

- Baseline: latest archived L4 champion
- Treatment: commit containing this entry
- Workload / trials: final combined L4 gate

## Environment

Mac verification uses `CUDARC_CUDA_VERSION=12080` with `cuda,no-cuda`. CUDA hardware, model, TP/EP, slots, KV, and server flags are deferred to the final L4 gate.

## Results

No wall-clock metrics claimed. Local checks and source-order proof cover the refactor only.

## Problems

A single L4 cannot runtime-exercise DSv4 multi-rank initialization; that path requires a multi-GPU remote gate.

## Learnings

`pending-remote`. Run the final L4 non-regression gate, then exercise DSv4 multi-rank startup on a multi-GPU host.
