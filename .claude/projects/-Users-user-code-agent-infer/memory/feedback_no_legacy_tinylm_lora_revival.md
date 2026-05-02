---
name: Do not revive TinyLM or legacy LoRA paths
description: User explicitly rejected reusing or citing the deleted TinyLM/legacy LoRA implementation as a source line for current work. Keep the train stack on the generic Qwen-family runtime only.
---

# Rule

When implementing train-side LoRA or adapter support in `agent-infer`, do
not restore, reference as prior art, or port code from the deleted
`TinyLM`/legacy Transformer LoRA path. Build the feature directly on the
current generic Qwen-family runtime (`Qwen3` / `Qwen3.5`) and keep the
architecture model-agnostic so other families can plug in later.
