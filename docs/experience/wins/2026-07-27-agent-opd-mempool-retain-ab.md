# Agent-OPD CUDA mempool retain A/B — CUDA, 2026-07-27

> Status: pending-remote (first run rejected: treatment was a no-op)

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
| retain=true | `u64::MAX` | 3 / 64 | not entered | 91360 | 34688 | n/a | 34 s | forward OOM |
| retain=false | **read back `u64::MAX`; invalid arm** | 3 / 64 | not entered | 91360 | 34688 | n/a | 32 s | rejected |

For every checkpoint group, retain driver used/free, mempool reserved current, mempool used current, and live tensor count. On failure, retain the exact phase, layer/group, operation, requested bytes, free bytes, and exit code.

Raw artifacts: `/host/arle-runs/mempool-ab-20260727/` on the H20 host. Binary SHA-256: `b651603ccf496887ea35493908f960cc92e3187f8f5746cabf9b8d4b74f5d3b7`.

## Problems

The train CLI previously had no `--cuda-mempool-retain` control, so the causal A/B could not run. The new flag preserves `true` as the shipped default and exposes `false` only as the treatment arm. Context creation explicitly writes `u64::MAX` for `true` and `0` for `false`, then reads and logs the effective CUDA release threshold; an unverified write is a silent no-op, not an A/B.

The first remote run exposed a second writer: loading the in-process rollout engine applied an inference `CudaRuntimeFlags::default()` and reset the process-global mempool knob to `true` before creating its context. The CLI command contained `--cuda-mempool-retain false`, but the context logged `requested_retain=true`. The treatment was therefore rejected, and no 64K run was attempted. The rollout engine config now carries the OPD value explicitly; the matched A/B still requires a rebuild and rerun.

The older reference binary is not a baseline because its exact commit, dirty state, build inputs, and allocator state are unknown.

## Learnings

The first remote attempt was rejected rather than interpreted: both arms executed with `u64::MAX`, produced identical group traces, and failed at forward group 3. A process-global runtime flag must be propagated through every later runtime-config application, not only set once before the first backend.

- PASS mechanism: treatment makes `pool reserved` track `pool used` materially more closely and moves the 40960 failure boundary or completes the writeback.
- Correctness gate: on a shorter sequence runnable in both arms, loss and gradients remain within the existing writeback tolerance.
- Default-flip license: capacity benefit is confirmed and the end-to-end wall-time cost is measured and accepted. Until then, the default remains `true`.
- If treatment leaves `pool reserved` high, locate allocations not governed by the release threshold. If `pool used` itself reaches capacity, decompose the largest live buffer before changing MLP, attention, or GDN.
