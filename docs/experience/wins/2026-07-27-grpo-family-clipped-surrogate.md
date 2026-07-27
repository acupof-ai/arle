# GRPO-family clipped surrogate correctness — CUDA, 2026-07-27

> Status: correctness complete; normal online non-empty update pending-remote.

## Goal

Make the objectives exposed as GRPO, DAPO, Dr.GRPO, and GSPO use the
advantage-sign-aware clipped policy surrogate instead of a detached clamped-IS
coefficient. Preserve SAO-DIS and CISPO semantics.

## Hypothesis

The existing generation-time behavior denominator was correct, but
`A * clamp(r)` kept a non-zero gradient in PPO's saturated regions. Selecting
the active branch from the advantage sign and using `A * r` only there restores
the paper gradient without changing the fused writeback ABI.

## Parameters

Local gates:

```bash
cargo test --release -p autograd ops::fused_linear_distill::tests
cargo test --release -p train update_strategy::tests
cargo check --release -p autograd -p train
CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib
CUDARC_CUDA_VERSION=12080 cargo check -p autograd -p train --release \
  --no-default-features --features cuda,no-cuda
cargo fmt -- --check
git diff --check
```

H20 isolated source/target:

```text
source: /host/phase1-pg-denombase-src
target: /host/phase1-pg-denombase-target
GPU: H20, physical GPU 1
model: /host/arle-build-mtp/models/Qwen3.5-0.8B
```

The online diagnostic arms kept GRPO and rollout temperature `0.3`. The final
completion-cap arm used `CLAUDE_CODE_MAX_OUTPUT_TOKENS=10000`, four samples, and
a 600-second Claude session wall.

## Environment

- Final remotely tested binary before the last telemetry-only review correction:
  `8549e3b14240580c8338d3c672610b6de587defd6a7870bc764fc9e4d50b0900`.
- CUDA release build: `BUILD_EXIT=0`.
- The final local correction separates GSPO sequence-ratio telemetry from
  token-ratio k3 KL and is covered by the same shared CPU/device helper plus a
  non-zero-KL oracle. A refreshed normal-online acceptance remains pending.

## Results

Correctness:

- GRPO/DAPO/Dr.GRPO use token-level sign-aware PPO clipping.
- GSPO uses a length-normalized sequence ratio for its sign-aware clipped
  coefficient.
- CISPO remains detached soft-clamped IS; SAO-DIS remains a hard gate.
- Token ratio overflow, f32 underflow, invalid bounds, and non-finite sequence
  ratios fail closed.
- GSPO reports sequence ratio/clip telemetry while token-level k3 KL and its
  gradient remain token-ratio based.

Local:

```text
autograd focused tests: 5 passed
train strategy tests: 11 passed
release checks: passed
CUDA/no-CUDA Mac typechecks: passed
fmt/diff checks: passed
```

H20 CPU/CUDA parity:

```text
PPO A± active/clipped regions:
  loss max error:      1.9073486e-6
  d_hidden max error:  2.3841858e-7
  d_weight max error:  2.3841858e-7
  clipped:             2 / 2

CISPO:
  loss max error:      1.4305115e-6
  d_hidden max error:  1.1920929e-7
  d_weight max error:  2.3841858e-7
  clipped:             1 / 1
```

Raw artifacts:

- `/host/arle-runs/phase1-pg-main/build.log`
- `/host/arle-runs/phase1-pg-main/tests.log`
- `/host/arle-runs/phase1-pg-main/device-parity.log`
- `/host/arle-runs/phase1-pg-final-timeout-ab/cc600/`
- `/host/arle-runs/phase1-pg-final-max10k-env/`

## Problems

The normal online acceptance case did not reach a non-empty update. This is not
counted as a pass or a numerical failure:

- at a 600-second session wall, two of four Claude sessions ended naturally and
  two timed out;
- all rewards were zero and no sample edited the task;
- incomplete tool-loop requests correctly had no terminal token sidecar;
- two trajectories reached 31,641 and 31,888 tokens and were rejected by the
  30,000-token writeback guard.

The online result therefore exercised rollout/harness failure envelopes but did
not exercise the changed optimizer objective. Repeating the same workload has no
remaining information value.

## Learnings

**Correctness PASS; normal online acceptance pending-remote.** The scalar
objective, fused gradients, fail-closed ratio contract, and CUDA parity are
verified. The tranche makes no quality, convergence, or wall-clock claim. Close
the remaining gate with a normal stochastic task that reliably produces at
least two sidecar-complete, sub-cap trajectories with reward variance and a
non-zero update; do not use replay or fabricated sidecars as a substitute.
