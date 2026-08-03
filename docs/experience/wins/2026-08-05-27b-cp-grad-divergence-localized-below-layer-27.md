# The 27B CP grad divergence is depth-localized below layer 27 — and it is not depth

**Date:** 2026-08-05 · **Commit:** ddee29e59 (per-param dump) · **Pod:** 8×H20, ThinkingCap-Qwen3.6-27B-FP8, seq=32768, LoRA r16 attention-qv

## Context

cp=1 vs cp=2 on the real 27B differed 1.89× in global grad norm (#85) with
matching losses. A global norm cannot say *which* params diverge, so
`ARLE_OPD_DUMP_PARAM_GRADS` was added next to the existing grad-norm telemetry
(`opd.rs`, post-all-reduce, env-gated, inert by default).

Both arms reproduce the break exactly: **3.752082 (cp=1) vs 1.983900 (cp=2) =
1.891×**, `RUN_EXIT=0`, 64 params dumped per rank, cp=2 identical across ranks.

## What worked

Every `lora_a` grad is exactly 0 in both arms — LoRA B is zero-init, so at step 0
the whole signal lives in the 32 `lora_b` params on the 16 full-attention layers
(3, 7, 11, … 63). That makes the dump a clean depth probe.

**cp1/cp2 ratio by layer** (q_proj / v_proj):

| layer | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 | 35 | 39 | 43+ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q | 3.51 | 3.45 | 3.95 | 3.01 | 2.24 | 2.15 | 1.05 | 1.05 | 1.01 | 1.01 | ≤1.001 |
| v | 3.16 | 2.19 | 2.66 | 1.70 | 1.60 | 1.22 | 0.99 | 1.00 | 1.00 | 1.00 | ≤1.001 |

**Layers 27–63 are at parity** (≤5%, mostly ≤0.1%). The divergence appears
around layer 23 and saturates at 3–4× by layer 11. The gradient enters the
backward at layer 63 correct and degrades as it propagates *down* — this is
accumulation along the backward pass, not a per-layer bias.

## It is not depth

The obvious reading — the toy depth sweep only went to 16, inside the flat
region — was tested and **fails**. Toy hybrid, cp=2, `grad_cp_vs_f32` /
`grad_single_vs_f32`:

| depth | 16 | 32 | 48 | 64 |
|---|---|---|---|---|
| cp vs f32 | 2.18e-2 | 1.17e-3 | 4.63e-4 | 4.20e-3 |
| single vs f32 | 2.38e-2 | 2.24e-4 | 9.43e-4 | 1.12e-3 |

At the 27B's own depth of 64 the toy CP arm is 4.2e-3 from the f32 anchor — three
orders below the 89% break, and non-monotone, i.e. noise. Depth is ruled out.

## What is left

The 27B differs from the toy on: FP8 base weights, MoE, real data, seq 32768 —
and **checkpoint offload, which `[ckpt-gate] engage=true` confirms is live on the
27B run and never engages on the toy.** Offload/recompute is also the only
candidate whose effect would accumulate along the backward pass, matching the
observed shape. An earlier seq=1024 probe shrank the gap 3.3×, which points the
same way.

## Rule

A null result on an axis bounds only the range you actually swept — and when you
extend the sweep, re-check before publishing the extrapolation. "Depth 16 was
inside the flat region" was a clean, plausible story that measurement killed in
one run.

## Next

27B cp=1 vs cp=2 at a seq below the checkpoint-gate threshold (`engage=false` in
the log). If the per-layer ratios flatten, the bug is in the chunked
recompute/offload backward under CP, not in the CP collectives.
