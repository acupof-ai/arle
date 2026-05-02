---
name: Train docs must separate current Qwen3 implementation from Qwen3.5 target direction
description: When describing the sole train-side model line, do not claim Qwen3.5 training already exists; current implementation is Qwen3-only and Qwen3.5 is target-direction wording only
type: feedback
---

When auditing, rewriting, or refactoring the train-side model line for this
repo, keep two facts separate:

1. **Current implementation**: `crates/train` currently has `qwen3.rs` and does **not** have a Qwen3.5 training implementation.
2. **Target direction**: the sole long-term train-side model line may converge on the **Qwen3.5 architecture family**, but that must be described as a target direction until the code exists.

Do **not** collapse these into a false present-tense claim like "the train stack is already Qwen3.5-only" or "Qwen3.5 is the current train-side implementation."

**How to apply:**
- For current-state docs, say: "Current train-side implementation is Qwen3-only."
- For architectural target docs, say: "Target direction is to converge the sole train-side model line on the Qwen3.5 architecture family."
- For implementation slices that remove legacy Transformer paths, route them
  onto the existing `qwen3.rs` train model unless/until a real train-side
  Qwen3.5 implementation lands. Do not rename `qwen3` codepaths to `qwen35`
  just to match the target-direction wording.
- If a doc currently presents TinyLM / handwritten Transformer / `pretrain.rs` / `train_multi_turn` as the active mainline, rewrite it so those are clearly legacy or transitional, not co-equal truths.
