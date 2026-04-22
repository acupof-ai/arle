# Train Batch Forward Speedup

## Context

`pretrain` and `train_sft` still treated `--batch` as gradient accumulation.
That kept the code numerically correct, but it left the Qwen3/Qwen3.5
`forward_batch_tokens(...)` path unused on the hot training loop. On Metal
that meant higher per-step overhead and lower token throughput than necessary.

This batch changed the two active training binaries so `--batch` becomes a
real per-forward batch size, while `--grad-accum-steps` remains the explicit
knob for extra accumulation on top.

## What Worked

- `pretrain` now samples `batch * seq` packed windows and runs one
  `forward_batch_tokens(...)` call per optimizer step.
- `train_sft` now collates tokenized examples into one padded batch, runs a
  single `forward_batch_tokens(...)`, and computes the assistant-only loss in
  one batched pass.
- The new batched SFT loss is numerically pinned against the old single-example
  loss: the batch loss equals the arithmetic mean of the per-example losses.
- The dead single-sample `PretrainFamily::forward(...)` entrypoint was removed,
  shrinking the training path instead of adding a parallel codepath.

## Results

### Verification

- `cargo test -p train --release --bin pretrain --bin train_sft -- --nocapture`
- `cargo test -p train --release`
- `cargo clippy -p train --all-targets -- -D warnings`

### Metal throughput smoke

Model config:

- family: `qwen35`
- hidden/layers/heads/head_dim: `384 / 8 / 6 / 64`
- params: `25.17M`
- tokenizer: `/tmp/scratch_bpe16k_tokenizer.json`
- corpus/model: `/tmp/tinystories_256m.txt` and `/tmp/scratch_qwen35_bpe16k_25m_ctx256/latest`

`pretrain` command shape:

```bash
target/release/pretrain --model-family qwen35 --backend metal \
  --corpus /tmp/tinystories_256m.txt \
  --tokenizer /tmp/scratch_bpe16k_tokenizer.json \
  --hidden 384 --layers 8 --heads 6 --kv-heads 3 --head-dim 64 \
  --intermediate 1536 --max-pos 256 --seq 256 --steps 10 --log-every 10
```

Results:

| Job | `--batch` | Final `tok_per_sec` |
| --- | --- | --- |
| `pretrain` | `1` | `211.360430` |
| `pretrain` | `4` | `457.243673` |

Speedup: **2.16×**

`train_sft` command shape:

```bash
target/release/train_sft --model-family qwen35 --backend metal \
  --model /tmp/scratch_qwen35_bpe16k_25m_ctx256/latest \
  --data /tmp/instruction_short_mix_fit256.jsonl \
  --seq-len 256 --steps 10 --save-every 10 --log-every 10
```

Results:

| Job | `--batch` | Final `tok_per_sec` |
| --- | --- | --- |
| `train_sft` | `1` | `108.317397` |
| `train_sft` | `4` | `225.836355` |

Speedup: **2.08×**

## Rule

For active training binaries, `--batch` should mean "real batched forward".
If a path still uses `--batch` as implicit accumulation, either rename the flag
or refactor it until the batch dimension hits the model in one forward call.
