# Qwen3.5 Dense/Full-Attn Metal LoRA Validation

## Context

`M5.4` had been left open as “Mac 上跑合成 supervised fine-tune，LoRA rank=8”.
The code already had Metal-backed autograd, LoRA-only `train_sft`, and
Qwen3.5-family dense/full-attn train surfaces, but the repo still lacked one
clean Mac-local proof that the current path actually worked end to end.

## What Worked

- Added a dedicated Metal regression test for the current `Qwen3.5 + LoRA`
  training path: `crates/train/tests/test_qwen35_sft_loop_metal.rs`.
  It verifies:
  - LoRA-only optimization on `Qwen35Model::new_with_lora(...)`
  - loss falls over repeated supervised steps on `MetalBackend`
  - merged bf16 save + reload back into a non-LoRA Qwen3.5 model stays within
    expected bf16 tolerance
- Ran a Mac-local CLI chain on this `Apple M4 Pro` host:
  1. `pretrain` synthesized a tiny dense/full-attn `qwen35` checkpoint from a
     tiny pure-Rust WordLevel tokenizer
  2. `train_sft --backend metal --model-family qwen35 --lora-rank 8 --lora-alpha 16`
     fine-tuned that checkpoint for 2 steps
  3. `eval_lm --backend metal` reloaded the produced `latest/` checkpoint
  4. `train_sft --resume-from .../latest` resumed and advanced from step 2 to 3
- Concrete run results:
  - `pretrain`:
    - backend=`Cpu`
    - config=`vocab=15 hidden=64 layers=2 heads=4 kv_heads=2 head_dim=16 ffn=128`
    - `step=1 loss=2.791926 ppl=16.312414`
  - `train_sft` on Metal:
    - `step=1 loss=2.851805 ppl=17.319022`
    - `step=2 loss=2.794652 ppl=16.356928`
    - checkpoint written to `/tmp/qwen35_metal_lora_e2e/sft_metal/step_000002`
  - `eval_lm` on Metal:
    - `loss=2.837145964304606`
    - `ppl=17.066986173482515`
    - `tokens=6`
  - resume run on Metal:
    - resumed from step 2
    - advanced to `step=3`
    - wrote `/tmp/qwen35_metal_lora_e2e/sft_metal_resume/step_000003`

## Rule

When a plan item says “Mac LoRA validation”, do not count unit coverage or
CPU-only LoRA checks as closure. The acceptance bar is one Mac-local CLI chain
that proves `train_sft --backend metal` can fine-tune, save, reload, and
resume on the current active Qwen-family path.
