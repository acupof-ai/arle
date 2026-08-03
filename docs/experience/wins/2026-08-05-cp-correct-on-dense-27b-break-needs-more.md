# CP is correct on a dense hybrid model — the 27B's 1.89× grad break needs something else

**Date:** 2026-08-05 (rewritten 2026-08-04 after the elimination tables failed to
replicate) · **Commits:** ddee29e59 (per-param dump), 3f25b8822 (interleaved toy),
f704dbf57 + follow-up (finite-difference probe) · **Pod:** 8×H20

## Context

cp=1 vs cp=2 on the real 27B (ThinkingCap-Qwen3.6-27B-FP8) differed 1.89× in
global grad norm (#85) with matching losses. A global norm cannot say *which*
params diverge, nor which arm is wrong.

## What is established

**The divergence is real and reproducible.** Post-all-reduce, seq 32768: 3.752082
(cp=1) vs 1.983900 (cp=2) = 1.891×. seq 4096: 1.347×. seq 2048: 1.655×. Base loss
is deterministic to 6 digits across runs, so this is not noise.

**It is depth-shaped.** With LoRA on `attention-full` (adapters on all 64 layers,
both layer families) at seq 2048, the cp1/cp2 ratio is 1.00 for layers 44–63, then
climbs smoothly to ~1.66 by layer 24 and saturates. The ramp crosses GDN and
full-attention layers alike — L27 `o_proj` 1.27, L28 GDN `out_proj` 1.23, L31
`o_proj` 1.24, L32 GDN 1.28 — so no single layer family steps. It also degrades
*within* a layer (L35: `o_proj` 1.11, `q_proj` 1.50), i.e. continuously with
backward depth, not at layer boundaries.

**CP is correct on a dense model of the same architecture family.**
`qwen35-08b-clean` is 24 layers with the 27B's own `full_attention_interval: 4`
hybrid, but a dense MLP. Post-all-reduce at seq 2048:

| | grad norm |
|---|---|
| cp=1 | 3.464947 |
| cp=2 | 3.467913 |

Agreement to 8.6e-4. A directional finite difference on
`layers.3.self_attn.q_proj.weight.lora_b` certifies both arms against ground
truth — cp=1 ratio 1.034 / 0.914 (two runs), cp=2 ratio 1.067, and the two arms'
analytic norms are 0.3% apart. **The CP ring, the seq↔head all-to-all, the zigzag
shard and the grad all-reduce all produce the right gradient here.**

## The finite-difference probe

`ARLE_OPD_FD_PARAM` steps one param along its own analytic gradient, so
ΔL = 2ε‖g‖ — a single-scalar probe drowns in loss noise. `fd/‖g‖ ≈ 1` certifies
that arm. It must all-reduce itself: the probe runs with `step_optimizer=false` to
keep the weights put, and that is the same branch that skips the CP/DP reduce.

Self-test on the 27B at layer 63 (where both arms already agree): ratio 0.9887 and
0.9909 across independent runs, ΔL reproducible to four digits.

**The probe cannot be used at depth on the 27B.** Perturbing a shallow layer moves
every downstream hidden state, which flips expert selection, and the loss jumps
discontinuously — finite differences need a locally smooth function. Two identical
runs at layer 3, seq 512, gave ratio 0.301 and 0.614; the repeat spread exceeds any
effect. The unperturbed forward is deterministic to 1e-6 in the same runs, so the
non-reproducibility is created by the perturbation, not present in the model. On
the dense 0.8B the same probe at the same layer reproduces to ~12%.

## What is ruled out

- **Checkpoint recompute.** 27B layer-3 FD with `--gradient-checkpointing false`
  lands at 0.459, inside the on-arm's own 0.301–0.614 spread.
- **The fused-CE chunking.** seq 2048 = one chunk (`chunk_rows=2048`) and the
  break is still there at 1.655×.
- **A per-layer-family transport bug.** The ramp is smooth across both families.

## What is NOT ruled out, and the confound

The dense-vs-MoE comparison also changes size (0.8B vs 27B), depth (24 vs 64) and
base dtype (bf16 vs FP8). "MoE-specific" is the leading candidate, not a finding.
The MoE forward itself is per-token with no capacity limit and no sequence-global
normalization, so routing under CP should be equivalent — but the 27B's cp1/cp2
loss gap is 1.7e-4 against the dense model's 1.2e-5, 14× larger, which is what
borderline tokens routing differently would look like.

## The toy gate is not an oracle here

`nd_parallel_parity` was the basis of an earlier version of this entry that claimed
depth, sequence length and offload were each ruled out. Those tables do not
replicate. Two independent reasons:

1. **The binary was stale.** The pod tree was five commits behind local with
   hand-modified files; the snapshot the tables came from cannot be attributed to
   any commit. After `pod.sh sync` to local HEAD and a clean rebuild the same
   declared config produces entirely different numbers (f32 anchor 1413.9 →
   736877).
2. **The config was structurally blind.** `ARLE_ND_HYBRID=1` puts a single
   full-attention layer on top of an all-GDN stack, so no ring gradient ever
   crosses a layer below it. `ARLE_ND_HYBRID=k` (3f25b8822) now interleaves every
   k-th layer; 4 is the 27B's shape.

On the rebuilt binary with interleaving, both CUDA arms miss the CPU f32 anchor by
30–70% and which arm is worse flips with config (h4: single 0.43 / cp 0.90; h8:
single 0.93 / cp 0.40). The f32 norm itself swings from 13.0 to 735.5 across
configs. Each config is a different random init, and at depth ≥12 with mixed layer
types the toy is in a numerically chaotic regime. **It bounds nothing at these
depths.** Its PASS only asks "is CP as good as single card", which cannot answer
"which one is right".

## Rule

- An oracle has to be validated before its verdict counts. The f32 anchor, the
  finite difference, and the toy gate each looked authoritative and each was
  silently invalid in some regime — the f32 arm at depth ≥12, the FD probe at
  depth on a MoE model, the toy at `HYBRID=1`. The cheap validation exists in
  every case: run the identical config twice, or probe where the answer is already
  known.
- A null result measured on a binary you cannot name is not a null result. Sync
  the tree and rebuild before the sweep, not after it contradicts itself.

## Next

Separate MoE from size/depth/dtype. The in-family dense model is 0.8B and the
in-family MoE model is 27B; there is no small MoE on the pod to break the tie, so
either build one (a truncated 27B, or the toy with MoE layers enabled) or find the
mechanism directly by dumping per-layer hidden-state gradients on the 27B at both
arms.
