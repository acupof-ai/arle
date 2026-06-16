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

## Update 2026-06-17 — hypothesis narrowing (code-level, pre-probe)

Driven as the F2 gate for the OPD 9B+teacher work (the 35B-A3B-as-OPD-teacher
needs a *coherent* forward). Narrowed from the 3 open hypotheses; the prime
suspect is now the **gated-delta / linear-attention layer forward on CUDA**.

CLEARED as suspects (evidence):
- **gate_up split order** (hyp 1): CUDA splits the stacked expert as
  `gate = rows[0, mi)`, `up = rows[mi, 2mi)` (`loader.rs:1094-1124`) — gate-first,
  the HF-standard layout. Not the bug.
- **Config nesting / rope parameterization** (hyp 2a): the real config nests rope
  under `text_config.rope_parameters` (`rope_theta=1e7`, `partial_rotary_factor=0.25`,
  `mrope_interleaved`, `mrope_section=[11,11,10]`). qwen35-spec parses it
  correctly: `lib.rs:838 rope_theta = rope_parameters.rope_theta` (=1e7),
  `rotary_dim = head_dim(256) * 0.25 = 64` (`lib.rs:757`). The hardcoded
  `1_000_000.0` at `qwen35.rs:4273` is a test fixture, not the load path. mRoPE
  sections reduce to standard 1D RoPE for text-only positions. Not the bug.

PRIME SUSPECT (hyp 2b — the gated-delta semantics the doc flagged):
- `text_config.layer_types` = **30× `linear_attention` + 10× `full_attention`**
  (`full_attention_interval=4`); **layer 0 is `linear_attention`**. Both CUDA
  (`qwen35.rs`, 38 gated-delta refs) and Metal (`infer-metal/src/qwen35.rs`)
  implement it, but CUDA is garbage and Metal is coherent → a **subtle CUDA
  gated-delta divergence** (A_log/dt semantics, the conv1d, the delta-rule
  recurrence, or `attn_output_gate=True` on the full-attn layers).

DECISIVE NEXT (measurement, not more grep): a **layer-0 activation probe,
CUDA vs Metal**, same prompt + weights — layer 0 is `linear_attention`, so it
tests the gated-delta path immediately and localizes the first divergent op in
one run. This is the dedicated probe session the original Fix line called for.
