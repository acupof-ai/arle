# End-to-end training flows: SFT on public dataset + from-scratch pretrain

## Context

Follow-up to `2026-04-19-train-to-chat-end-to-end.md` (which only did a
2-step smoke test with 3 local examples). User (2026-04-19) asked:
"端到端的测试好 训练流程；跑一下 从零训练的流程也；比如 多层 的 llm
大概可能到 100m ？". Three things to verify:

1. Real public dataset → SFT → chat works with a sensible answer.
2. From-scratch pretrain loop runs on Metal and the loss actually drops.
3. ~100M-param multi-layer LLM is achievable at the current state.

## What Worked

### Flow A: Public SFT — dolly-15k → Qwen3-0.6B → chat

```bash
# 1. Download
DATA_RAW=$(target/release/download_dataset \
  --repo databricks/databricks-dolly-15k \
  --file databricks-dolly-15k.jsonl)
# → ~/.cache/huggingface/hub/datasets--databricks--databricks-dolly-15k/.../databricks-dolly-15k.jsonl

# 2. Normalize schema
target/release/convert_dataset --input "$DATA_RAW" \
  --format dolly --output /tmp/dolly.chat.jsonl
# → 15011 lines · 15011 written · 0 skipped

# 3. SFT on Metal (safe hyperparameters)
target/release/train_sft \
  --model models/Qwen3-0.6B --data /tmp/dolly.chat.jsonl \
  --out /tmp/dolly_sft_safe --steps 10 --batch 1 --seq-len 128 \
  --lr 1e-6 --save-every 10 --log-every 1 --backend metal
# → step=1 loss=4.47 … step=10 loss=2.74
# → saved /tmp/dolly_sft_safe/step_10/{config.json,model.safetensors,tokenizer.json}

# 4. Chat
echo "What is the capital of France? Answer in one word." | \
  target/release/agent-infer --model-path /tmp/dolly_sft_safe/step_10 \
    --max-tokens 24 --non-interactive
# → "Okay, the user is asking for the capital of France. I know that
#    France's capital is Paris. But"   (164 tok/s, TTFT 69 ms)
```

### Flow B: From-scratch TinyLM pretrain

```bash
cargo build --release --no-default-features --features metal \
  -p train --bin pretrain

target/release/pretrain --dataset bytes --steps 100 --batch 4 --seq 64 \
  --lr 1e-3 --d-model 256 --n-layers 6 --n-heads 8 --d-head 32 \
  --d-ff 1024 --vocab-size 256 --log-every 10 --backend metal
# → backend: Metal
#   config: vocab=256 d_model=256 n_layers=6 n_heads=8 d_head=32
#           d_ff=1024 max_seq_len=128
#   params: 4820224 (4.82M)
#   step 0:  loss 5.5913   ← ≈ ln(256) = 5.54 → ~uniform init
#   step 50: loss 2.6460
#   step 99: loss 2.5697   ← learning from random
```

Random baseline for vocab=256 is `ln(256)=5.545`. Going from 5.59 at
step 0 to 2.57 at step 99 confirms the from-scratch optimization loop
(parameter init → forward → backward → AdamW step → next batch) works
end-to-end on Metal.

### Flow C: CUDA backend — code-complete, pending remote verification

`crates/autograd/src/backend_cuda.rs` (1279 lines):

- `matmul` via `cudarc::cublas::safe::CudaBlas::gemm` and
  `gemm_strided_batched` (row-major → column-major swap trick).
- 10 custom `.cu` kernels under `backend_cuda/kernels/`: `elementwise`,
  `softmax`, `rms_norm`, `rope`, `embedding`, `reduce`, `scatter_add`,
  `silu`, `gather`, `add_broadcast`.
- All `todo!("GPU required: …")` stubs are gated by `#[cfg(feature =
  "no-cuda")]` — the Mac typecheck path. Real execution paths are
  complete.

Three Mac build gates pass:

```
cargo check -p autograd --no-default-features                    ✅
cargo check -p autograd --features metal                         ✅
cargo check -p autograd --no-default-features --features cuda,no-cuda  ✅
```

## 100M-param LLM sizing

Parameter budget for transformer LM (tied embeddings, SwiGLU):

```
embed (tied):   vocab_size × d_model
per layer:      4·d_model²   + 3·d_model·d_ff   + 2·d_model (norms)
                └ Q/K/V/O ┘   └ gate/up/down ┘
```

Three ~100M configurations that fit (vocab=32k BPE):

| d_model | d_ff | n_layers | Est. params | Shape   |
|---------|------|----------|-------------|---------|
| 768     | 2048 | 14       | ~100M       | GPT-2 small–ish |
| 512     | 1792 | 24       | ~95M        | deeper/narrower |
| 1024    | 2816 | 8        | ~103M       | wider/shallower |

**Feasibility on Mac M-series Metal**: yes but expect ~1–3 s/step at
batch=4 seq=512, so a meaningful pretraining run (10k+ steps) is on the
order of hours-to-a-day. The Backend trait + Metal path already covers
the hot ops (matmul, embed, rms_norm, rope, silu, softmax, gather) —
no infrastructure gap. The ceiling is pretraining-data scale, not the
compute path.

## Failure mode observed (preventive)

An earlier 15-step run at `lr=5e-6, batch=1, seq_len=128` on
Qwen3-0.6B collapsed the model — after 15 steps it generated `"me me
me…"` repeatedly. The instability was hyperparameter, not pipeline:
with the same dataset and model, `lr=1e-6 × 10 steps` stayed coherent
and answered "Paris". **For full-parameter SFT on 0.6B-sized HF
checkpoints with `batch=1`, keep `lr ≤ 1e-6` until LoRA lands** —
high-variance gradients at batch=1 + full-param updates can destabilize
quickly. Use LoRA (already plumbed in the trainer) for higher effective
lr without destabilization.

## Rule

End-to-end pipeline is first-class. Three canonical commands cover the
whole spine:

```
download_dataset  →  convert_dataset  →  train_sft  →  agent-infer
```

(From-scratch swaps `train_sft` for `pretrain --dataset bytes|corpus`.)
Any future change to the training or inference path must preserve:

1. Trainer writes HF-compatible `step_N/{config.json,
   model.safetensors, tokenizer.json}`.
2. `agent-infer --model-path <step_dir>` loads that directory
   unchanged.
3. On-the-wire dataset schemas (dolly/alpaca/sharegpt/chat) continue to
   normalize through `data_adapter.rs` into the canonical `messages`
   format.

If any of those break, the end-to-end smoke test
(`scripts/train_and_chat.sh`) will surface it immediately.
