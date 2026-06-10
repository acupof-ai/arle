# Rewrite Qwen3.5/3.6 CUDA forward: degenerate output on real 35B-A3B (TP=1 AND TP=2)

## Context

First-ever real-weights run of the rewrite Qwen35 CUDA path (it could not even
LOAD this checkpoint before the stacked-expert loader `b729f8e2`; the
2026-05-30 e2e win was the deleted monolith). 8×H20 pod, built at `bddf174c`
(origin/main HEAD `d457ad1b` does not compile — references
`dsv4_mhc_enter_launch_cuda` whose FFI declaration is in uncommitted local
work). Both runs deterministic (same-config ×3 byte-consistent), both garbage:

```
TP=2 (GPUs 0,1): "The capital of France is" → "Bright正向正向薯薯薯薯确立"
                 needle FAIL; gen128 49.91 tok/s of 山羊/山脉 loops
TP=1 (GPU 0):    same prompt → "DataProvider tiny tiny tiny文学文"
                 needle FAIL; gen128 36.38 tok/s of 薯/飞扬 loops
```

Engine MECHANICS all pass: 67 GB loads in ~85 s (stacked experts, EP range
slicing), TP=2 multiproc spawn + NCCL + lockstep work, greedy sampling is
byte-identical across ranks and across repeats, decode at sane tok/s.

## Root Cause

OPEN. Killed so far:
- **TP sharding math** — TP=1 (zero sharding branches, `is_single()` arms) is
  equally garbage → the defect is in the base forward and/or stacked loader.
- **Silent tied-lm_head fallback** — untied configs load `lm_head.weight`
  loudly (`qwen35.rs:494`); engine built ⇒ it loaded. (The junk-token
  signature suggested vocab-projection misalignment; not this mechanism.)
- **c≥2 engine death** is separate and known: rewrite Qwen35/Metal executors
  are single-row-per-tick; a 2-row decode plan trips the rows==1 ensure and
  kills the engine thread ("engine thread closed" for every later request).

Open hypotheses, cheapest discriminator first:
1. **gate_up split order/orientation in the stacked expert loader** — flip
   A/B is one env-less rebuild; current gate-first follows the monolith
   `load_stacked_expert_2d` reading of this checkpoint.
2. **Qwen3.6-35B forward deltas vs whatever小 hybrid the rewrite path was
   developed against** (full-attn gate epsilon/order, gated-delta A_log/dt
   semantics, rope theta from the nested multimodal `text_config`).
3. Layer-0 activation probe vs the Metal/MLX artifact on the same prompt
   (embedding row + first attention) localizes in one run.

Note: this is a MULTIMODAL checkpoint (`model.visual.*`, `mtp.*`, top-level
`lm_head.weight`, vocab 248320) — every name resolved or build would fail,
but any *semantic* config nesting miss (text_config vs flat) is still suspect.

## Fix

None yet — needs a dedicated layer-probe session.

## Rule

- "First-runnable scope" paths are unverified against real weights until a
  real-weights needle/self-consistency gate has actually run; budget that gate
  into the FIRST real-model session, not after feature work stacks on top.
- When TP=N output is garbage, run TP=1 with the same binary+checkpoint before
  touching shard math — one control kills (or convicts) the entire axis.
