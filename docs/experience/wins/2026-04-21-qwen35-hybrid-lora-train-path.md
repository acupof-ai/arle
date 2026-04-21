# Qwen3.5 Hybrid Linear-Attn LoRA/Eval Path Landed

## Context

After the dense/full-attn Qwen3.5 train line and the Mac-local Metal LoRA
validation were closed, the next real train-side model gap was hybrid
linear-attn Qwen3.5. The missing piece was not scheduler wiring anymore; it
was the model/autograd path: partial-RoPE support, a train-side linear-attn
core, and a clear split between surfaces that are safe today (`LoRA` and
frozen eval) and surfaces that still are not (`scratch pretrain` and RL
acceptance).

## What Worked

- Added a train-side hybrid linear-attn core in `crates/autograd` and wired
  `Qwen35Model` to treat attention as a real enum (`Full` vs `Linear`) instead
  of rejecting non-full-attn layers up front.
- Extended the shared `qwen35-spec` contract so train-side code can validate
  two different truths without inventing parallel model definitions:
  - dense/full-attn scratch-pretrain contract
  - broader LoRA/frozen-eval contract that allows hybrid linear-attn layers
- Fixed partial-RoPE handling on the autograd path so hybrid checkpoints can
  run through the same CPU / Metal forward+backward plumbing.
- Added real regression coverage:
  - `test_qwen35_forward` now covers hybrid forward with partial rotary dims
  - `test_qwen35_hybrid_sft_loop` covers CPU LoRA training + merged save/reload
  - `test_qwen35_hybrid_sft_loop_metal` covers the same training shape on Metal
- Added hard numerical-correctness coverage instead of relying on "loss goes
  down" as proof:
  - `crates/autograd/tests/test_linear_attention.rs` now finite-difference
    checks the new `linear_attention_core` gradients for `qkv`, `z`, `b_proj`,
    and `a_proj`
  - the same test also compares CPU vs Metal results and gradients with the
    inputs forced device-resident on Metal
  - `crates/train/tests/test_qwen35_hybrid_forward_metal.rs` now compares a
    fixed hybrid `Qwen35Model::new_for_eval(...)` forward pass across CPU and
    Metal
- That grad-check immediately caught a real backward bug: the reverse pass was
  normalizing raw `q/k` preactivations while the forward path applies `silu`
  before normalization. Fixing that mismatch dropped the worst `qkv`
  finite-difference error from `4.3e-2` down to `1.1e-5`.
- Re-ran real CLI acceptance on 2026-04-21 using a tiny synthetic hybrid
  checkpoint and short assistant-only JSONL:
  - CPU:
    - `eval_lm --model-family qwen35 --model-path /tmp/qwen35_hybrid_cli_base/latest`
    - `train_sft --model-family qwen35 --backend cpu --lora-rank 8 --lora-alpha 16`
    - `eval_lm --model-path /tmp/qwen35_hybrid_cli_out_cpu_short/latest`
  - Metal:
    - `train_sft --model-family qwen35 --backend metal --lora-rank 8 --lora-alpha 16`
    - `eval_lm --model-path /tmp/qwen35_hybrid_cli_out_metal_short/latest --backend metal`
- Concrete acceptance outcome:
  - CPU and Metal both trained for 2 SFT steps without divergence
  - post-train `eval_lm` on both backends returned finite loss / perplexity
- CUDA compile surface checked via
  `cargo check -p train --release --no-default-features --features cuda,no-cuda`

## Rule

Do not describe "hybrid Qwen3.5 training" as one monolithic shipped/not-shipped
flag. There are two different support surfaces:

- `LoRA` + frozen eval can ship as soon as the model/autograd path and CLI
  acceptance are real.
- scratch pretrain and RL acceptance stay separate until they are actually
  validated on hybrid checkpoints.
