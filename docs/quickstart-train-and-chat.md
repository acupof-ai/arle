# Quickstart — download a dataset, SFT Qwen3, chat with the result

End-to-end recipe for going from a public HuggingFace dataset to a
trained checkpoint you can chat with via `agent-infer`. Verified on
2026-04-19 for Qwen3-0.6B on Metal and on 2026-04-21 for Qwen3-0.6B on
CUDA via `scripts/train_cuda_e2e.sh`.

## 0. Build binaries once

```bash
cargo build --release -p train --bin download_dataset --bin convert_dataset --bin train_sft
cargo build --release --no-default-features --features metal
```

## 1. Download a dataset

`download_dataset` fetches a single file from an HF dataset repo. stdout
is the resolved local path (for shell substitution); stderr is progress.

```bash
DATA_RAW=$(target/release/download_dataset \
  --repo databricks/databricks-dolly-15k \
  --file databricks-dolly-15k.jsonl)
```

Requires `HF_TOKEN` only for gated repos. Cache lives under
`~/.cache/huggingface/hub/datasets--<org>--<repo>/`; re-runs hit cache.

## 2. Convert to `messages` format

The SFT trainer reads `{"messages": [{"role", "content"}, ...]}` per
line. Most public datasets ship in a different schema — `convert_dataset`
normalizes them.

| Format     | Source schema                                                       |
|------------|---------------------------------------------------------------------|
| `chat`     | `{"messages": [...]}` — passthrough                                 |
| `dolly`    | `{"instruction", "context"?, "response"}` — Databricks Dolly-15k    |
| `alpaca`   | `{"instruction", "input"?, "output"}` — Stanford Alpaca style       |
| `sharegpt` | `{"conversations": [{"from", "value"}, ...]}` — ShareGPT/Vicuna    |

```bash
target/release/convert_dataset \
  --input "$DATA_RAW" --format dolly --output /tmp/dolly.chat.jsonl
# → 15011 lines · 15011 written · 0 skipped
```

Unknown ShareGPT roles are dropped; `instruction + "\n\n" + context`
(or `input`) becomes the user turn for dolly/alpaca.

## 3. SFT (smoke test: 2 steps)

```bash
target/release/train_sft \
  --model models/Qwen3-0.6B \
  --data  /tmp/dolly.chat.jsonl \
  --out   /tmp/dolly_sft \
  --steps 2 --batch 1 --seq-len 128 --lr 1e-6 \
  --save-every 2 --log-every 1 --backend metal
# → step=1 loss=4.47  step=2 loss=2.97
# → saved checkpoint for step 2 to /tmp/dolly_sft/step_000002
# → /tmp/dolly_sft/latest -> step_000002
```

Replace `--backend metal` with `--backend cuda` on NVIDIA hosts.

Checkpoint layout: `step_000002/{config.json, model.safetensors,
tokenizer.json, adapter_model.safetensors, adapter_config.json,
trainer_state.json, optimizer.safetensors}` plus `latest ->
step_000002`.

## 4. Chat with the trained model

```bash
echo "What is the capital of France?" | target/release/agent-infer \
  --model-path /tmp/dolly_sft/latest --max-tokens 16 --non-interactive
# → TTFT 69 ms · 167 tok/s · "Okay, the user is asking for the capital of France. I need"
```

## Known-good recipes

| Dataset                              | `--format` | File                               |
|--------------------------------------|------------|------------------------------------|
| `databricks/databricks-dolly-15k`    | `dolly`    | `databricks-dolly-15k.jsonl`       |
| `tatsu-lab/alpaca`                   | `alpaca`   | `data/train-00000-of-00001.parquet` ← parquet, not JSONL |
| `anon8231489123/ShareGPT_Vicuna_unfiltered` | `sharegpt` | varies — check repo file list |
| `allenai/tulu-3-sft-mixture`         | `chat`     | `data/train.jsonl`                 |

Parquet datasets need an extra preprocessing step (not yet wired in —
convert to JSONL with `pandas` / `datasets` first, then run step 2).
JSONL is the current fast path.

## Smoke test

`scripts/train_and_chat.sh` runs the whole loop against
`models/tiny_sft.jsonl` (3 examples, already in the repo). If that
breaks, the integration is broken.

```bash
INFER_TEST_MODEL_PATH=models/Qwen3-0.6B scripts/train_and_chat.sh
```

For the canonical CUDA smoke that also covers `eval_lm`, `agent-infer`,
and resume, run:

```bash
scripts/train_cuda_e2e.sh
```
