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

## The 27B seq axis

Same run at `--synthetic-writeback-seq 4096`: **7.251241 vs 5.382694 = 1.347×**,
identical shape — layers 43–63 at parity (≤0.2%), 3–23 at 1.34–1.40. Both the
amplitude and the onset depth scale with sequence length (onset layer 39 at
seq 4096, layer 27 at seq 32768).

## Three axes ruled out on the toy

`nd_parallel_parity`, hybrid, cp=2, `grad_cp_vs_single`:

| | depth 16 | 32 | 48 | **64** |
|---|---|---|---|---|
| cp vs f32 | 2.18e-2 | 1.17e-3 | 4.63e-4 | **4.20e-3** |

| depth 64, seq | 512 | 2048 | 8192 | **32768** |
|---|---|---|---|---|
| cp vs single | (f32 arm) 1.1e-2 | 1.2e-2 | 3.94e-3 | **4.76e-4** |

| depth 64, `ARLE_FORCE_CHECKPOINT` | off | on |
|---|---|---|
| seq 2048, cp vs f32 | 1.225e-2 | 1.225e-2 |
| seq 32768, cp vs single | 4.760e-4 | 4.760e-4 |

**At the 27B's own depth (64) and sequence (32768), with checkpoint offload
forced on, the toy CP arm is 4.8e-4 from single card** — three orders below the
89% break, and checkpointing is a wash to 7 digits. Depth, sequence length, and
offload/recompute are each ruled out, individually and together.

## What is left

FP8 base weights, MoE MLPs, the real corpus and its target mask, the real GQA
head geometry and per-layer RoPE theta, and the 48-GDN/16-full **interleave**
(the toy stacks 63 GDN layers under one full-attn layer rather than alternating
1-in-4).

## Rule

A null result bounds only the range you swept — and when you extend the sweep,
re-check before publishing the extrapolation. "Depth 16 was inside the flat
region" was a clean, plausible story that one run killed; "checkpoint offload is
the only thing that accumulates along the backward" was the next one, killed by
a 7-digit wash.

## Next

Per-**layer** hidden-state grad norms on the 27B at cp=1 vs cp=2, not just the
every-4th-layer LoRA sample. That names the exact layer where the two arms part
and whether the step happens at a GDN layer, a full-attn layer, or the MoE MLP —
which the current instrument cannot distinguish.
