# OPD Fused Distill Default Path — pending remote

## Context

Windowed Route B is on by default via `--logits-window-size=32`. The pure-KL
window arm now defaults to fused `lm_head + distill loss` to avoid materializing
student full-vocab logits; `--no-fused-distill` is the local opt-out back to
dense `logits_from_hidden_window + kl_distill_loss_for_config`.

## What Worked

Local CPU/no-cuda tests gate fused vs dense equivalence at ~2e-5 and keep a
revert lever. This is not bit-identical to the previous dense path.

## Pending Remote

- H20 needle correctness gate.
- Same-binary A/B: default fused path vs `--no-fused-distill`.
- Record wall-clock memory/throughput deltas before claiming PASS.

## Rule

Default memory-path flips need an opt-out plus H20 correctness/perf evidence.
