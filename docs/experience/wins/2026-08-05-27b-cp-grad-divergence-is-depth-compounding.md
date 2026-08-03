# The 27B CP grad divergence is depth-compounding, and it starts below layer 27

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

## Why the toy gate missed it

The `nd_parallel_parity` depth sweep ran 2 / 4 / 8 / 16 layers and found no
compounding. **The 27B's divergence only becomes visible after the gradient has
travelled ~36 layers.** Depth 16 is inside the flat region — the toy could not
have seen this no matter how carefully it was read. The earlier "no depth
compounding" reading stands for depth ≤16 and does not extrapolate.

## Rule

A null result on an axis bounds only the range you actually swept. "Depth does
not compound" measured to 16 says nothing about 64; state the range with the
conclusion or the conclusion silently becomes an extrapolation.

## Next

Toy hybrid at `ARLE_ND_LAYERS` 16 / 32 / 48 / 64, cp=2 vs single vs f32 — if the
ratio climbs past depth ~24 the mechanism is reproducible off the 27B and can be
bisected cheaply. If it stays flat, the cause is 27B-specific (FP8 base weights,
MoE, seq 32768 with checkpoint offload engaged).
