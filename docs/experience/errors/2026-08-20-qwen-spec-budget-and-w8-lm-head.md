# Qwen speculative thinking budget and W8A16 lm_head routing — 2026-08-20

> Status: **Fixed in source, pending-remote.**

## Context

Two Qwen dispatch gaps survived because format and sampling decisions were split
across parallel entry points:

- MTP/DSpark accepted chains cannot apply forced tokens, grammar masks, logit
  bias, penalties, or a strict per-token thinking budget.
- W8A16 Marlin owned batched GEMM, while the single-row `lm_head` dispatch only
  forwarded FP4 and FP8 to their Marlin layouts.

## Root Cause

Qwen speculative routing vetoed penalties only. A thinking budget becomes
`force_next_token = </think>` after the limit, but the speculative verifier
continues through its raw argmax or filtered-probability path. A multi-token
accepted chain can also cross the limit before the engine observes its tokens.

W8A16 repack frees `qweight` and `qscales`. `gemm_batch` consumed the retained
Marlin buffers, while `gemv` fell through to the freed scalar buffers for a
quantized `lm_head`.

## Fix

- One Qwen compatibility predicate now sends every unsupported token rewrite,
  including any configured thinking budget, through one-token plain decode.
- W8A16 batch and single-row dispatch now share one raw-pointer Marlin launcher.

## Goal

Correctness: enforce the configured Qwen thinking-token ceiling under MTP and
DSpark, and serve a W8A16-quantized `lm_head` without a missing-source error.

## Hypothesis

Plain decode applies `force_next_token` before sampling. Reusing the existing
W8A16 Marlin kernel at `M=1` reads the only weight representation retained after
repack. No kernel math changes.

## Parameters

Pending remote on one H20, CUDA release build, TP=1, BF16 KV:

1. Qwen3.6 with `--spec-type mtp --mtp-draft-tokens 3`, greedy sampling,
   `enable_thinking=true`, `max_thinking_tokens=8`, one request, 64 output-token
   cap. Repeat with DSpark.
2. A Qwen3.5/3.6 W8A16 checkpoint with `lm_head.weight` quantized at group size
   128; one non-degenerate prompt, 32 output tokens.
3. `scripts/lever_gate.sh` and `scripts/needle_gate.py temp`, ladder
   512/4096/16384/32768, three repetitions, same flags against the current
   baseline envelope.

## Environment

Candidate commit and binary hash: pending. GPU: H20. Driver/CUDA, model path,
slot line, KV capacity, and exact serve commands: pending remote run.

## Results

Pending. Required pass conditions: the ninth generated thinking token is
`</think>` in both speculative configurations; W8A16 `lm_head` records a
`cuda.w8a16.marlin_tensorcore` hit; zero request errors, empty outputs, loops,
or new needle misses. This correctness change carries no performance claim.

## Problems

The local macOS `cuda,no-cuda` test binary cannot link because CUDA C symbols
are intentionally absent. `CUDARC_CUDA_VERSION=12080 cargo check -p infer-cuda
--release --no-default-features --features cuda,no-cuda --tests` passes; GPU
execution remains pending.

## Learnings

**pending-remote.** A chained decoder may run only when it implements every
token rewrite for every accepted position. Repacked formats need one shared
launcher for batched and single-row entry points.

## Rule

Represent speculative compatibility once at the executor boundary. Route every
consumer of a source-free weight through the retained layout.
