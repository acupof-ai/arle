# GSPO math-opd run 1: zero length compression (wrong samples had no length gradient)

**Date:** 2026-08-24
**Scope:** `crates/cli/src/train_cli/math_opd.rs`, `crates/cli/src/args.rs`
**Commit:** `39c43d5d2` (fix)

## Context

GSPO RL run on Qwen3.8-27B-NVFP4, 12 rounds, 200 train tasks / 50 eval,
K=4 samples/prompt, α=0.3 relative-within-group length penalty
(`1 - α·(len-len_min)/(len_max-len_min)` for correct, 0 for wrong).
Goal: train shorter correct reasoning on math.

Result: **zero learned.** Eval accuracy 0.48→0.48→0.46→0.52→0.48 (noise,
±7pp at n=50). Eval completion median pinned at 8192=max_tokens all 12
rounds. Train loss 0.041 (LoRA fitting train tasks) but no generalization
and no length change.

## Root Cause

Wrong samples got `reward = 0` regardless of length. The model rambles to
8192 on tasks it can't solve (~50% of eval), and those samples carried zero
length gradient. The length penalty only fired on correct samples, which
were already short (~1500 tokens on train tasks the model can solve).

The relative-within-group normalization was a secondary weakness: GSPO's
std-normalization makes relative and absolute affine-in-len rewards
equivalent within a group, so the "relative vs absolute" distinction was a
red herring. The dominant issue was the constant-0 reward on wrong samples.

## Fix

Absolute length penalty on every sample:

```
correct → max(0, 1 - α·len/L0)
wrong   → -β·len/L0
```

Defaults: α=0.5, L0=4096, β=0.05. The wrong arm is small enough that
trying (and maybe solving) always beats giving up immediately, but large
enough to create a consistent gradient toward stopping early on unsolved
tasks.

## Rule

A length penalty in RL must give gradient on **every** sample, not just
correct ones. A reward that is constant (0) for wrong samples teaches
nothing about length on the exact tasks where the model rambles. The
gradient must flow on the behavior you want to change.
