# Suffix-LoRA + detach lever (--lora-layer-start) — landed, perf A/B pending-remote

## Context
Runtime change `c69ff813` (crates/train/src/qwen35.rs + qwen35_loader.rs, crates/cli
args, crates/autograd/src/tensor.rs): `--lora-layer-start N` restricts LoRA to the
top-layer suffix (layers ≥ N) and detaches the hidden state before the first
trainable layer, so the autograd backward stops at the suffix rather than traversing
the full frozen 35B-A3B backbone. Target: the **1333 s backward** that makes a 35B
OPD step ~2126 s (the trainer-side wall; the InferStudent rollout is only 42 s).

## What Worked (verified so far)
- **Correctness**: codex cross-review clean (detach at the right loop point
  qwen35.rs:2959/3315/3397; default `None` byte-identical; no forward-value break).
  `detach()` = tape-disconnected clone + cleared grad metadata; the tape's
  `collect_relevant` stops at the `requires_grad=false` leaf → backward genuinely cut.
- **Test green**: `qwen35_lora_layer_start_limits_adapters_and_tape_prefix` (lib unit
  test) passes — suffix-only LoRA grads, tape excludes prefix layers, default unchanged.
- **Builds** via `scripts/pod_pipeline.sh` (incremental, INCR_BUILD_EXIT=0).

## Pending-remote
The **runtime perf claim** (does the suffix-detach actually cut the 1333 s backward,
and by how much?) needs a **35B OPD A/B** (all-layer vs `--lora-layer-start K`,
same shape, wall-clock backward_seconds from the opd_step_trace). That run is gated on
the 35B-student (D) experiment, which follows the think-on agentic OPD (C) result.
Default path is byte-identical, so no regression risk to current runs. Bench entry to
be filled with the measured backward-time delta when the 35B A/B runs.

## Rule
A backward-pruning lever's correctness (tape cut + byte-identical default) can be
unit-tested on CPU, but its **speed** is a wall-clock A/B on the real shape — never
quote a backward-time cut without the opd_step_trace numbers.
