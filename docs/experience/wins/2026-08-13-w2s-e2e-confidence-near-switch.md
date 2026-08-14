# w2s 60-step e2e: the confidence threshold behaves as a near-switch

> Status: Confirmed

## Context

First full-length `arle train w2s` run after the step-budget measurement
([wins](2026-08-13-w2s-step-budget-kl-terms-dominate.md)). 60 steps,
`--confidence-threshold 0.9`, `--save-every 20`; ThinkingCap-Qwen3.6-27B-FP8
base/student, four 0.8B aux, GSM8K prompts. Binary `1c5847839`, GPU 4,
`RUN_EXIT=0`, adapters saved at steps 20/40/60.

## What Worked

The run completed end to end, and the skip statistics locate the useful
threshold range:

- 48/60 steps skipped, all with reason=Confidence. Over the skipped steps,
  max_prob is min 0.9016 / median 0.9531 / max 0.9805 on GSM8K. Every skipped
  value lies between 0.9 and 0.99, so 0.99 skips nothing and 0.9 skips 80% —
  the threshold acts as a near-switch on this workload.
- Loss on the 12 trained steps: 25.16 -> 18.80.
- VRAM drift +0.0154 GB/step, +0.49 GB over the run; cause unknown.
- Trained-step VRAM 35.60 GB vs skipped-step 40.20 GB: skipped steps exit
  early without freeing intermediates.

## Rule

On GSM8K the 27B student's last-position max_prob concentrates in
[0.90, 0.98], so the confidence threshold selects between skip-almost-nothing
(0.99) and skip-80% (0.9) with no useful gradation between. Pick the threshold
from the measured max_prob distribution of the target workload. The skipped
path holds 4.6 GB more VRAM than the trained path; free intermediates on the
early exit before relying on high skip rates for capacity.
