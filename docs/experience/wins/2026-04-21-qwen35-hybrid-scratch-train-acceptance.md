# Qwen3.5 Hybrid Scratch Train Acceptance

## Context

- `Qwen3.5` hybrid linear-attention training was still documented as incomplete.
- The missing pieces were scratch pretrain construction, RL acceptance on
  `train_grpo` / `train_multi_turn`, and numeric confidence beyond "loss goes
  down".

## What Worked

- Train-side config validation now distinguishes `dense-only` acceptance from
  the broader scratch-training contract, so hybrid linear-attention is explicit
  instead of hidden behind dense-only guards.
- `linear_attention_core` now backpropagates through every trainable input
  involved in the hybrid block, and finite-difference checks cover all of them.
- CPU vs Metal parity now exists at both the op level and the model level for
  the hybrid path.
- End-to-end acceptance closed locally on CPU and Metal for:
  - `pretrain --model-family qwen35 --linear-attn-every 2`
  - `eval_lm` against the resulting checkpoint
  - `train_grpo --model-family qwen35 --linear-attn-every 2`
  - `train_multi_turn --linear-attn-every 2` for both stepwise GRPO and GSPO
- The infer HTTP surface can now proxy `/v1/train/*` to the train control
  plane, so a single OpenAI-compatible server can expose status/events/save/stop
  without forking the trainer authority.

## Rule

- Hybrid train-path work is not done until three things all exist together:
  1. finite-difference grad checks for the custom op
  2. cross-backend parity on the accepted backend set
  3. end-to-end checkpointed CLI acceptance on the real train entrypoints
