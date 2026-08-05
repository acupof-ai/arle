# FlashQLA GDN backward default-on: 80K training step 1.99×, backward 2.14×

**Date:** 2026-08-05 · **Pod:** 8×H20, `fa742a038`, ThinkingCap-Qwen3.6-27B-FP8, LoRA attention-qv

> Status: Default flip. `--gdr-chunkwise-prefill` now defaults true; the
> recurrent arm stays reachable via `--la-backward-mono`.

## Context

The 71% `linear_attention_chunked_scan_backward_f32` row (see
`2026-08-05-80k-training-step-is-one-kernel.md`) is the whole 80K step. FlashQLA
(QwenLM, MIT, TileLang SM90+) is the official SOTA GDN chunkwise kernel; the port
landed `4846f8046` and built cold once the sm_90a fix (`4b85750e4`) and the CP
geometry table (`1b913e31e`) cleared. This entry flips the default and records the
matched A/B that licenses it.

## Result — matched A/B, seq=81920 cp=2

Same harness (`/host/fqgate.sh`), same seq, only variable is the flag.

| | FlashQLA (default) | recurrent (`--la-backward-mono`) | speedup |
|---|---:|---:|---:|
| forward | 64.12 s | 81.0 s | 1.26× |
| fused CE | 0.83 s | 1.91 s | — |
| backward | **312.64 s** | 670.28 s | **2.14×** |
| **step** | **378.72 s** | 752.96 s | **1.99×** |

Loss 4.537510, grad_norm 7.976866, RUN_EXIT=0. The 71% kernel row is gone from the
nsys profile. Forward also gains 1.26× — the chunkwise path carries the native
GDR chunk-prepare (#82).

## Tradeoff

- **Numerical:** two bf16 backward paths, so grad_norm moves at the bf16 floor
  (32K liveness saw 2.15 vs 2.26, ~4.9%). Loss is forward-only and agrees to
  2e-4. The f32 anchor is `qwen36_fp8_lora_fd_gate --gdr-chunkwise` (arm-internal
  analytic-vs-FD, ground-truth-free) — pending-remote.
- **Hardware:** FlashQLA is SM90+ (`setmaxnreg` needs sm_90a). Non-Hopper falls
  back; the flag is inert where the kernel is absent.
- **Escape hatch:** `--la-backward-mono` forces the recurrent arm for A/B.

## Rule

When the SOTA kernel exists and the port runs, flip the default on the matched
speedup — do not re-litigate that SOTA beats the hand-rolled arm. The remaining
question is correctness (f32 anchor), not whether the win is real.
