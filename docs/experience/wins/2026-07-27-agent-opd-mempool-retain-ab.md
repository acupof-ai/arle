# Agent-OPD CUDA mempool retain A/B — CUDA, 2026-07-27

> Status: pending-remote

## Goal

Determine whether the decode-oriented CUDA mempool retention policy causes the single-H20 Agent-OPD masked-writeback capacity failure at sequence length 40960, and measure its wall-time cost.

## Hypothesis

With identical binaries and inputs, `--cuda-mempool-retain false` returns freed async-pool pages to the driver, reducing `pool reserved` toward `pool used` and allowing the 40960 forward to pass its current checkpoint-group-3 OOM without changing training numerics.

## Parameters

Use the canonical remote Agent-OPD synthetic-writeback command with:

```bash
ARLE_OPD_VRAM_TRACE=1 arle train agent-opd \
  --synthetic-writeback-seq 40960 \
  --cuda-mempool-retain <true|false> \
  <unchanged canonical arguments>
```

- Baseline: current release binary, `--cuda-mempool-retain true`
- Treatment: same binary, `--cuda-mempool-retain false`
- Model: ThinkingCap-Qwen3.6-27B-FP8
- Trials: one cold process per arm initially; repeat if the arms are within measurement noise
- Record before running: commit, dirty diff, build flags, binary SHA-256, model config, CUDA version, driver, and GPU identity

## Environment

- Host / GPU: pending remote, single H20 (97508 MiB observed capacity)
- Driver / CUDA: pending remote
- Model / dtype: ThinkingCap-Qwen3.6-27B-FP8, LoRA masked-CE writeback
- Parallelism: single GPU
- Runtime flags: identical except `--cuda-mempool-retain`

## Results

| arm | release threshold verified | forward groups | backward | pool reserved peak MiB | pool used peak MiB | loss | wall time | result |
|---|---|---:|---|---:|---:|---:|---:|---|
| retain=true | pending | | | | | | | pending |
| retain=false | pending | | | | | | | pending |

For every checkpoint group, retain driver used/free, mempool reserved current, mempool used current, and live tensor count. On failure, retain the exact phase, layer/group, operation, requested bytes, free bytes, and exit code.

Raw artifacts: pending remote.

## Problems

The train CLI previously had no `--cuda-mempool-retain` control, so the causal A/B could not run. The new flag preserves `true` as the shipped default and exposes `false` only as the treatment arm. Context creation explicitly writes `u64::MAX` for `true` and `0` for `false`, then reads and logs the effective CUDA release threshold; an unverified write is a silent no-op, not an A/B.

The older reference binary is not a baseline because its exact commit, dirty state, build inputs, and allocator state are unknown.

## Learnings

Pending remote.

- PASS mechanism: treatment makes `pool reserved` track `pool used` materially more closely and moves the 40960 failure boundary or completes the writeback.
- Correctness gate: on a shorter sequence runnable in both arms, loss and gradients remain within the existing writeback tolerance.
- Default-flip license: capacity benefit is confirmed and the end-to-end wall-time cost is measured and accepted. Until then, the default remains `true`.
- If treatment leaves `pool reserved` high, locate allocations not governed by the release threshold. If `pool used` itself reaches capacity, decompose the largest live buffer before changing MLP, attention, or GDN.
