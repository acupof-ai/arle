# T5 small-M bf16 GEMV → cuBLAS: −5.1% decode ITL — CUDA, 2026-08-03

> Status: **Shipped, default path** (#196 T5). c=1 W8A16 decode ITL p50
> **25.08 → 23.80 ms**; cumulative vs pre-#196 baseline **26.88 → 23.80
> (−11.5%)**. Same 32k c=1 protocol (H20 GPU 6, 16×256 tok, seed 20260416).

## What shipped

`gemm_small_n_uses_gemv` (`crates/cuda-kernels/csrc/gemm/gemv.cu`) now also
requires `M >= 4096`: the hand-written GEMV keeps lm_head-class shapes
(M=152k, grid ≈ 9.5k blocks) and small-M shapes route to cuBLASLt. The only
decode GEMM affected is the T3-fused `[96, 5120]` in_proj_ba, which ran
**~52 µs/launch (~19 GB/s)** on a 6-block grid over 78 SMs; cuBLAS split-K
does the same shape in ~8 µs (SGLang side of the matched A/B measures the
identical shape at 6.4 µs via nvjet).

## Correctness: f32 anchor, not md5

The kernel swap changes fp32 accumulation order, so greedy trajectories
legitimately diverge (this thinking model flipped one prompt from
answer-first to think-first — empty `content`, coherent `reasoning_content`).
Gate used instead of byte-identity:

- **f32 anchor**: standalone cublasGemmEx OP_T on random `[96,5120]×1` vs CPU
  f64 reference — max abs err 0.033 on |y|≈11 (bf16 rounding scale).
- Generation coherence verified on multi-prompt greedy output.

Harness fix shipped with it: `scripts/bench_throughput_chat.py` counted only
`delta.content` events, so think-channel decode looked like "missing output
event" — it now counts `reasoning_content` deltas as ITL events too (ITL is
per-token regardless of channel).

## Learnings

- The gemv guard gated on N and K but never on M — a grid-fill blind spot
  that only became load-bearing when T3 created a hot [96, 5120] shape.
- Md5-gates only survive kernel-identical changes; any accumulation-order
  change needs the f32-anchor gate from the start
  ([[feedback_two_bf16_paths_need_f32_anchor_not_mutual_identity]]).
