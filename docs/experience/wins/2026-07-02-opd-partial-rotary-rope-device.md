# OPD backward 26.8s → 10.6s — partial-rotary RoPE was demoting whole grad chains to CPU

## Context

After the fused-SDPA chain
([2026-07-02-opd-sdpa-fused-prefill-kernel](2026-07-02-opd-sdpa-fused-prefill-kernel.md)),
backward (26.3s) dominated the OPD round, attributed only to 61 opaque
Checkpoint tape entries. Threading the backward profile through checkpoint
inner tapes (f2aaaf7c + unit test) decoded it: Transpose 79×/5.9s, Slice
32×/4.6s, MulScalar 0.8s, RMSNorm 0.8s — layout ops whose device kernels exist
— all running on HOST.

Root cause was one gate line: `rope_backward`'s device path required
`cos_rows*2 == head_dim` (kernel rotated the full head only). Qwen3.6 full
attention is **partial rotary** (`partial_rotary_factor: 0.25` — rotary 64 of
head_dim 256), so every full-attn q/k grad fell to the host rope fallback and
its host output demoted the entire upstream chain (rmsnorm → transpose →
slice backwards all gate on `upstream.dirty != Host`). The forward rope gate
had the same restriction, costing 2 host ropes per full-attn layer there too.

## What Worked

Extend `rope_f32`/`rope_backward_f32` with a `rot_half` parameter — rotate the
leading segment, pass the tail through (exact `cpu_rope_forward` semantics;
full rotary is the `rot_half == head_dim/2` special case, so one kernel serves
every model) — and relax both device gates to `<=`. Metal's host-eager
selection untouched (its partial support is unverified). Commit `c51ca26d`.

Same toy config (run-ropefix-toy1r vs run-sdpatrace-toy1r, GPU 7→1, RUN_EXIT=0):

| metric | sdpatrace | ropefix | Δ |
|---|---|---|---|
| forward_hidden_states | 3.768s | **2.835s** | **−25%** |
| backward | 26.252s | **10.557s** | **−60%** |
| Transpose bwd | 5.94s (host) | out of top-6 | eliminated |
| Slice bwd | 4.57s (host) | out of top-6 | eliminated |
| LinearAttention bwd | 4.16s | 4.17s | unchanged — next wall (40% of backward) |
| MatmulBT bwd | 0.82s | 0.65s | ✓ |
| loss | 0.282402 | 0.2827 | in 0.24–0.33 band |

Cumulative vs the pre-FP8-fix baseline (137ffb28): forward 122.1s → **2.8s
(43×)**, backward 149.1s → **10.6s (14×)**.

Next wall: LinearAttention backward — 45 calls × 93ms = 4.2s, 40% of backward.

## Rule

- A host fallback's real cost is the CASCADE, not itself: every downstream
  backward gates on `upstream.dirty != Host`, so one demotion turns a whole
  layer's grad chain into CPU work. Attribute with the inner-checkpoint op
  table before blaming the ops that merely inherited host residency.
- Kernel envelope gates (`== full head`, `== full rotary`) silently exclude
  model families; when a gate compares a dimension with `==`, ask which
  production config violates it (partial rotary, GQA, gated q) and extend the
  kernel instead of falling back.
