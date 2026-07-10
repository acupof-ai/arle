# Agent-OPD writeback backward is LinearAttention-scan-bound (47%), NOT recompute/group_size — the ckpt_group_size lever is dead

> Status: Killed — 2026-07-11

## Context

After the seq-adaptive offload win ([wins](../wins/2026-07-11-agent-opd-dspark-decode-and-seq-adaptive-offload.md)),
the 7.5s/call writeback backward became the largest remaining agent-OPD round
cost (24.5% of the round). A proposed lever: `ckpt_group_size` is pinned at 1
even at short seq because `attn_floor` (`qwen35.rs:309`) is a fixed 12 GiB that
alone exceeds the 8 GiB group budget → per-layer checkpointing → full per-layer
recompute. Hypothesis: raise group_size at short seq to cut backward recompute.

Licensed-or-killed with a real op profile before touching the memory model
(`ARLE_OPD_BACKWARD_PROFILE=1`, H20 GPU1, arm C = dspark ON + `--writeback-offload
false`, seq≈1274-1378, backward≈7.8s).

## Root Cause (measured op-split of the 7.77s backward, not inference)

| op | seconds | % backward | what |
|---|---|---|---|
| **LinearAttention** | **3.68** | **47.4%** | delta-rule chunked-**scan** backward — dominant |
| recompute-forward | 2.39 | 30.7% | Checkpoint envelope − inner ops (group=1, per-layer) |
| MatmulBT | 0.745 | 9.6% | all weight/input grad matmuls (each site ~0.01s, none material) |
| CausalSdpaRecompute | 0.608 | 7.8% | full-attn layers |
| AddBroadcast + rest | ~0.35 | <5% | |

(61 Checkpoint ops / 64 layers ⇒ group_size=1 confirmed. recompute cross-checks:
forward phase 2.54s ≈ 2.39s recompute. ✓ 4-window aggregate identical in shape.)

**The backward is LinearAttention-backward-bound**, not recompute-bound and not
grad-matmul-bound. LinearAttention is already a device chunked-scan kernel
(`crates/autograd/src/backend_cuda/kernels/linear_attention.cu` + own sub-op
profiler) — it is expensive because the delta-rule recurrence is a **sequential
scan** over chunks (cannot parallelize along seq like SDPA), not because it's
un-fused.

## Fix — KILL the group_size lever, two independent reasons

1. **Threshold:** recompute-forward is 30.7% < the 40% bar set for the lever.
2. **Mechanism (decisive):** `ckpt_group_size` cuts the **offload count** (one
   host round-trip per K layers — `qwen35.rs:282-286`), NOT recompute FLOPs
   (total recompute stays 1× forward for any group size). Offload is already OFF
   on this path, so grouping buys **nothing**. The "12 GiB floor over-conservative
   at short seq" premise is true but irrelevant — it changes group_size, and
   group_size doesn't change this path's cost.

Downstream re-ranking (round = 122378 ms arm C):
- **LinearAttention backward** — new dominant, ceiling −5.75% round at a 2× kernel
  speedup; deep sequential-scan kernel work (high effort), DEFER.
- **Recompute-skip at short seq** (`--grad-checkpointing`-style: don't checkpoint
  the writeback forward when seq is small and activations fit resident) — a
  *separate* lever from group_size, removes the 30.7% recompute (2.39s/call ≈ 7.8%
  round), numerics-identical, no capability cost. Same seq-adaptive pattern as the
  offload fix. The clean next real win.
- **`--lora-layer-start` top-half** — halves backward DEPTH (both recompute AND
  the LinearAttention scan), −11% round *if* the tape early-stops at the shallowest
  trainable layer (unverified) AND the top-half-only capability cost is acceptable.

## Rule

- **Profile the backward's op-split before touching the checkpoint memory model.**
  The "recompute-dominated" intuition was 30.7%, not the majority; the real cost
  was a sequential-scan op the memory-model lever cannot touch. A memory-config
  knob (group_size) that a source-read *seems* to gate a cost on can be
  mechanically unrelated to that cost — verify the mechanism, not just the
  threshold.
