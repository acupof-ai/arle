# Correctness gate campaign — the day's seven perf changes licensed end to end

## Context

One day of aggressive path swaps (FP8→cuBLAS GEMMs, fused SDPA, partial-rotary
device RoPE, LoRA FP8→BF16 promotion, adaptive checkpointing, chunk-parallel LA
backward, FP8 dense-GEMM floor) reduced the agent-OPD round 438s → ~13s. ckl's
challenge: 你确定训练推理的内容完全对吧? Loss-in-band alone is a weak gate —
assemble the full evidence chain to the repo's correct-inference standard.

## What Worked — four independent layers

1. **Op-level parity**: chunk-parallel LA backward vs CPU reference grads,
   Qwen3.5 + Qwen3.6-27B shapes, every gradient max_abs ≤ 1.2e-4; RoPE partial
   semantics copied verbatim from `cpu_rope_forward` (rotate segment + tail
   passthrough).
2. **Trajectory-level byte identity**: the greedy multi-turn rollout (turn0/1/2
   generations) is MD5-identical across every engine build of the day —
   including the FP8 floor change that moved 52k GEMV calls onto DeepGEMM.
3. **Train-then-infer closed loop**: every round's rollout runs on the
   previous round's freshly-synced LoRA; 10/10 rounds passed=1 at temp 0 with
   loss descending 0.2829 → 0.1740 (best floor ever; the old code's frozen-base
   drift bug capped it at ~0.24).
4. **Needle ladder (repo standard, scripts/lever_gate.sh generic profile,
   27B FP8, current binary)**: exact 3/3 at 115/300/446/2000/8000 — after
   decoding two harness artifacts, NOT model failures:
   - `max_tokens: 16` hardcoded → thinking-style preamble consumed the window
     while the model was mid-answer (now env-tunable, NEEDLE_MAX_TOKENS);
   - at 8000 the qwen3_nonthink template weakens and the model reverts to
     thinking style — with a 256-token window all three repeats state the
     exact code.

## Rule

- A "miss" aggregate from a gate harness is a case to decode, not a verdict:
  both zero-scores today were the model answering CORRECTLY outside a
  16-token window (same class as the teacher-timeout-as-abstention anchor).
- Byte-identical greedy rollouts across a kernel-route change is the
  cheapest strong equivalence gate — one grep + md5sum, no harness at all.
- The train-then-infer loop (round N samples on round N−1's weights, scored
  by pytest) is a per-round functional gate on 训练后的输入输出 for free.
