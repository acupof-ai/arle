# MTP linear-only replay flips CUDA spec-decode to a net WIN on H20 (depth-2 0.81→1.03×)

## Context

NextN-MTP spec-decode on Qwen3.6-27B-FP8 (CUDA) was a net LOSS at every depth on
H20 real-fp8 even after the amortizing verify GEMV
([2026-06-22](2026-06-22-tile-matched-amortizing-verify-gemv.md)): depth 1/2/4 =
0.84/0.81/0.64×. `ARLE_MTP_PHASE` instrumentation of the macro-step located the
last avoidable cost — the **partial-accept REPLAY**. On partial accept (accepted
k < depth), `spec_step` restored the trunk linear state to S_{start_pos} and re-ran
a FULL 64-layer `forward_hidden(chain[0..=k])` (measured **21–47 ms**) purely to
advance the 48 GatedDeltaNet conv+recurrent states to the post-accepted position
— `next_hidden` already came from the verify's row-k hidden, so the replay's only
job was the linear-state side-effect. no-spec on H20 is 45.9 tok/s (21.8 ms/token,
still 31% HBM), so this replay (≈ a second verify) sank every depth.

## What Worked

**Deletion-style: replaced the full-trunk replay with a linear-only replay.**
Per-position snapshot is infeasible (the GDR recurrent kernel advances state
in-place; intermediate S_t never hit global memory) and a "skip the attention/MLP"
replay is wrong (each linear layer reads the full-stack residual). The correct
cheap path: during the verify forward, **cache each linear layer's GDR inputs**
(pre-conv in_proj qkv + b/a projections) for all depth+1 rows — they already
encode the full-stack residual — then on partial accept restore S_{start_pos} and
re-run ONLY conv1d + gated_delta_rule over rows [0..=k] from the cache, skipping
all 16 full-attn blocks, all MoE/MLP, and the lm_head. conv1d + GDR were factored
into a shared `advance_linear_conv_gdr` used by both the trunk forward and the
replay, guaranteeing identical kernel dispatch (recurrent-vs-chunked) → bit-equal
state. Capture is gated on the spec path, so the default decode is byte-for-byte
unchanged.

H20 (8×H20, sm_90, real FP8; GPU 1; 27B-FP8; 128-tok greedy; `ARLE_QWEN_GEMV_TILED=1`):

| depth | replay (full→linear) | speedup before → after | accept | [mtp-gate] |
|-------|----------------------|------------------------|--------|------------|
| 1 | 21–47 ms → **~2 ms** | 0.84× → **0.99×** | 67% | PASS |
| 2 | 21–47 ms → **~2 ms** | 0.81× → **1.03× (WIN)** | 61% | PASS |
| 4 | 21–47 ms → **~2 ms** | 0.64× → **0.87×** | 44% | PASS |

**depth-2 spec-decode now beats no-spec (47.6 vs 46.1 tok/s) — the first net win
for CUDA NextN-MTP on H20.** The gate (spec-vs-ref divergence == the no-spec
self-consistency floor, @128/128) confirms the linear-only replay leaves the
GatedDeltaNet state token-exact-to-greedy within the MoE non-determinism floor.

Lower depth wins (higher acceptance, less verify/replay); depth-2 is the sweet
spot. The two levers compose: amortizing verify GEMV (verify 82.7→60.7 ms at
depth-4) + replay-elimination (21-47→2 ms) together flip every depth up ~0.2-0.35×.

## Rule

When a spec-decode rollback re-runs a full forward purely to restore a recurrent
(linear-attention / SSM) state, the forward is ~all wasted — the state-advance is
a tiny fraction of it. Cache the recurrent layers' INPUTS during the verify pass
(they already carry the full-stack residual) and replay ONLY the recurrent
kernels over the accepted prefix; skip attention/MLP/lm_head. Profile the
macro-step (`ARLE_MTP_PHASE`) to find which phase dominates BEFORE optimizing —
here verify amortization alone left spec at a loss; the replay was the lever that
crossed 1.0×. Gate on recurrent-state byte-identity + the token-exact spec gate.
