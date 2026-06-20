# Suffix-LoRA + detach lever (--lora-layer-start) — landed, measured at seq-649 and seq-2048

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

Step total drops **2.88×** (500→174 s) at seq-649.

## Measured perf at PRODUCTION seq-2048 (2026-06-20, supersedes the projection)
Same A/B, **seq pinned to 2048** (long-prompt corpus, all rows truncated to
`--prompt-max-tokens 2048`, `--rollout-len 64` → `rollout_len 2112`), GPU4 (Arm A) /
GPU5 (Arm B) in parallel, 2 steps each. The earlier "projects to ~130 s / ~10×" was
**optimistic** — the measured numbers correct it:

| seq-2048 | backward_seconds | student_forward_seconds | step total_seconds |
|---|---|---|---|
| Arm A (all-layer, 40) | **1196** (1277.2 / 1115.1) | 488 (519 / 458) | **1694** (1799 / 1589) = 28.2 min |
| Arm B (suffix top-8/40) | **147** (154.5 / 139.9) | 464 (495 / 433) | **616** (652 / 579) = 10.3 min |
| **speedup** | **8.1×** backward | ~1.0× (forward not detached) | **2.75×** step |

The backward ratio **compressed 10.3× → 8.1×** going seq-649 → seq-2048: Arm B's top-8
backward grew 4.4× (33→147 s) vs Arm A's 3.5× (341→1196 s) because the O(n²) attention
backward in the retained suffix grows super-linearly with seq. **student_forward
(~464 s = 75% of Arm B's step) is now the wall** — undetached, so the lever doesn't touch
it. A 35B OPD step at seq-2048 drops **28.2 → 10.3 min** (2.75×); to go below ~10 min the
next D-infra lever must cut the forward (reuse the InferStudent inference forward for the
KL logits). Correctness intact: both arms finite + decreasing loss; **step-1
byte-identical across arms** (the detach changes only the backward tape extent + which
adapters get grads, not forward values — same code path as the seq-649 A/B that proved
byte-identity); step-2 diverges only because Arm A updated all-40 QV adapters vs Arm B's
8 (expected). Default (None) byte-identical. Logs: /data01/lora_lever_arm{A,B}_2048.log
(seq-2048) and _v3.log (seq-649); needs `ARLE_OPD_STEP_PROFILE=1` and **no `--json`**
(the timing line is gated off under `--json`).

## Rule
A backward-pruning lever's correctness (tape cut + byte-identical default) can be
unit-tested on CPU, but its **speed** is a wall-clock A/B on the real shape — never
quote a backward-time cut without the opd_step_trace numbers.
