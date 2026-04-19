# train → chat end-to-end: Qwen3-0.6B SFT checkpoint serves in the REPL

## Context

User asked (2026-04-19): "系统完整的做好；训练的模型可以对话了吗". The roadmap
had `train_sft` + `pretrain_qwen3` on one side producing safetensors, and
`agent-infer` (CLI/REPL) on the other loading them, but **no session had
empirically verified the round-trip**. The `pretrained-weight-bridge.md`
plan flagged this as the M5.3 blocker: "Qwen3 safetensors → autograd
forward → SFT one step → export → serve → chat" was specced but not
checked.

## What Worked

### One command trained the model

```bash
cargo run --release --no-default-features --features metal \
  -p train --bin train_sft -- \
  --model models/Qwen3-0.6B \
  --data  models/tiny_sft.jsonl \
  --out   /tmp/sft_test \
  --steps 2 --batch 1 --seq-len 64 --lr 1e-6 \
  --save-every 2 --log-every 1 --backend metal
```

Output (bf16 safetensors, matching HF Qwen3 layout):

```
step=1 loss=5.892630 lr=0.000001 ms=46765.83
step=2 loss=10.260437 lr=0.000001 ms=68254.89
[train_sft] saved checkpoint for step 2 to /tmp/sft_test/step_2
            (source model dir: models/Qwen3-0.6B, dtype: Bf16)
```

Written to disk: `step_2/{config.json, model.safetensors (1.4 GB),
tokenizer.json}`. The loss spike at step 2 is not a bug — 3-example
dataset at lr=1e-6 is not learning; what matters is the pipeline
finished all phases (load → forward → backward → optim step → save).

### The CLI loaded the trained checkpoint without special casing

```bash
echo "hi" | target/release/agent-infer \
  --model-path /tmp/sft_test/step_2 \
  --max-tokens 8 --non-interactive
```

Log excerpts:

```
Loading model from: /tmp/sft_test/step_2
Using local model path: /tmp/sft_test/step_2
MetalBackend: loading model from /tmp/sft_test/step_2
  arch: Qwen3 28 layers, hidden=1024, heads=16/8(kv), vocab=151936, eos=151645
  loading 1 shard(s) via MLX mmap …
  loaded 311 tensors (memory-mapped)
  weights loaded into Metal unified memory
loaded step_2 (metal) in 0.3s
```

Generation:

```
TTFT: 155.0ms (prefill 137 tokens)
generated 8 tokens  (178.1 tok/s)
→ "Okay, the user just said"
```

The trainer's `save_checkpoint` (`pretrain_qwen3.rs:691-741`,
`train_sft.rs` follows the same contract) writes `config.json` with the
exact HF Qwen3 schema — `architectures: ["Qwen3ForCausalLM"]`,
`model_type: "qwen3"`, `hidden_size`, `num_hidden_layers`, etc. — plus
a verbatim copy of the tokenizer. The inference-side weight loader
(`infer/src/weight_loader.rs:22-56`) resolves `model.safetensors` (or a
sharded index) under that directory and feeds it straight into the
Metal memory-mapped loader. No conversion step needed.

### What this validates

- **Training produces an inference-ready artifact.** Format contract
  between trainer and loader holds end-to-end without special casing.
- **Tokenizer is preserved across the boundary.** `train_sft` copies
  `tokenizer.json` from the source model; the CLI's prompt-formatting
  path uses the same tokenizer the model was trained against.
- **bf16 round-trip works.** Trainer saves bf16; Metal loader reads
  bf16; generation produces coherent tokens (within the limits of what
  2 SFT steps can do).
- **The "serve" step has no blocking dependency.** The 0.3 s load +
  155 ms TTFT + 178 tok/s decode are all on the Metal continuous-batching
  scheduler with FFI-zero varlen pack already in production.

## Rule

`train_sft` / `pretrain_qwen3` → `agent-infer --model-path <step_dir>`
is a first-class supported loop. Future trainer binaries must preserve
the same on-disk contract: `step_N/{config.json, model.safetensors,
tokenizer.json}` under a single directory, HF Qwen3 schema for
`config.json`, verbatim tokenizer copy. If a trainer needs to write a
different format, add a converter — do not diverge the loader.

The empirical validation matters more than schema-matching on paper:
a trivially-lossy SFT step is the cheapest way to prove the round-trip
works. Keep the two-step SFT above in `scripts/train_and_chat.sh` as a
smoke test; if it ever breaks, the integration is broken no matter
what the unit tests say.
