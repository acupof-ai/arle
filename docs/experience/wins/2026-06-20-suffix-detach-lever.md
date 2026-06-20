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

## Measured perf (35B A/B, 2026-06-20)
Qwen3.6-35B-A3B-FP8 student (40 layers), single-GPU autograd, rollout-649, identical
config except the lever (Arm A GPU4 all-layer attn-qv LoRA; Arm B GPU5
`--lora-layer-start 32` = top-8 layers). Same loss trajectory (0.242→0.185), both finite.

| | backward_seconds | student_forward_seconds |
|---|---|---|
| Arm A (all-layer, 40) | **340.9** (354.6 / 327.4) | 138.4 / 128.5 |
| Arm B (suffix top-8/40) | **33.2** (34.3 / 32.2) | 132.3 / 121.6 |
| **speedup** | **10.3×** backward | 1.05× (forward not detached, as expected) |

Step total drops **2.88×** (500→174 s). The lever cuts the trainer-side wall ~10×: at
the production rollout-2048 (all-layer backward was ~1333 s) the suffix path projects to
~130 s → a 35B OPD step drops from ~35 min to a few min, making 35B-student OPD
practical. Correctness intact: both arms finite + decreasing loss, **step-1
byte-identical across arms** (clean controlled A/B); step-2 diverges only because arm A
updated all-40 QV adapters vs arm B's 8 (expected). **New bottleneck: student_forward
(~127 s)** is now dominant — the next D-infra lever is the forward (reuse the
InferStudent inference forward for the KL logits). Default (None) byte-identical. Logs:
/data01/lora_lever_arm{A,B}_v2.log; needs `ARLE_OPD_STEP_PROFILE=1` (timing line is
gated off under `--json`).

## Rule
A backward-pruning lever's correctness (tape cut + byte-identical default) can be
unit-tested on CPU, but its **speed** is a wall-clock A/B on the real shape — never
quote a backward-time cut without the opd_step_trace numbers.
