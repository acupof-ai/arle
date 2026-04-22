# Pretrain Grad Accum Surface + Generic WordLevel Tokenizer Example

## Goal

- Remove the scratch-pretrain effective-batch bottleneck caused by
  `pretrain` hard-coding `grad_accum_steps=1`.
- Keep corpus/tokenizer recipe choices out of the core runtime by moving the
  generic word-level tokenizer workflow into an example.

## Hypothesis

- Surfacing `--grad-accum-steps` on `pretrain` should let the existing
  `Trainer` accumulation path scale effective tokens per optimizer step
  without changing the default behavior.
- A generic example that builds a whitespace word-level tokenizer from any
  plain-text corpus is enough for corpus-specific vocab experiments; the
  training binaries do not need TinyStories-specific heuristics.

## Params

- Code state: local tree on `2026-04-22`
- Host: `Apple M4 Pro`
- Backend for acceptance smoke: `cpu`
- Binary: `target/release/pretrain`
- Example: `target/release/examples/build_wordlevel_tokenizer`
- Corpus: synthetic plain-text corpus with `96` tokens / `4` unique tokens
- Model: scratch `qwen35`, `hidden=16`, `layers=2`, `heads=2`, `kv_heads=1`,
  `head_dim=8`, `ffn=32`, `max_pos=16`
- Train config:
  - run A: `steps=2`, `batch=1`, `seq=8`, `grad_accum_steps=1`
  - run B: `steps=2`, `batch=1`, `seq=8`, `grad_accum_steps=4`

## Env

```bash
cargo test -p train --release --bin pretrain
cargo build -p train --release --bin pretrain --example build_wordlevel_tokenizer
cargo clippy -p train --release --bin pretrain --example build_wordlevel_tokenizer -- -D warnings
```

Smoke commands:

```bash
target/release/examples/build_wordlevel_tokenizer \
  --corpus /tmp/pretrain_grad_accum_smoke/corpus.txt \
  --out /tmp/pretrain_grad_accum_smoke/tokenizer.json \
  --vocab-size 16

target/release/pretrain \
  --model-family qwen35 \
  --corpus /tmp/pretrain_grad_accum_smoke/corpus.txt \
  --tokenizer /tmp/pretrain_grad_accum_smoke/tokenizer.json \
  --out /tmp/pretrain_grad_accum_smoke/ga1 \
  --steps 2 --batch 1 --seq 8 --lr 1e-3 --grad-accum-steps 1 \
  --log-every 1 --save-every 2 --backend cpu \
  --hidden 16 --layers 2 --heads 2 --kv-heads 1 --head-dim 8 \
  --intermediate 32 --max-pos 16 \
  --metrics-jsonl /tmp/pretrain_grad_accum_smoke/ga1_metrics.jsonl

target/release/pretrain \
  --model-family qwen35 \
  --corpus /tmp/pretrain_grad_accum_smoke/corpus.txt \
  --tokenizer /tmp/pretrain_grad_accum_smoke/tokenizer.json \
  --out /tmp/pretrain_grad_accum_smoke/ga4 \
  --steps 2 --batch 1 --seq 8 --lr 1e-3 --grad-accum-steps 4 \
  --log-every 1 --save-every 2 --backend cpu \
  --hidden 16 --layers 2 --heads 2 --kv-heads 1 --head-dim 8 \
  --intermediate 32 --max-pos 16 \
  --metrics-jsonl /tmp/pretrain_grad_accum_smoke/ga4_metrics.jsonl
```

## Results

Tokenizer example output:

| metric | value |
| --- | ---: |
| total tokens scanned | `96` |
| unique tokens | `4` |
| kept tokens | `4` |
| special tokens | `0` |

Pretrain smoke:

| run | grad_accum_steps | tokens / micro | effective tokens / optimizer step | step 1 ms | step 1 tok/s | step 2 ms | step 2 tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ga1` | `1` | `8` | `8` | `1.48` | `5417.76` | `1.10` | `7272.73` |
| `ga4` | `4` | `8` | `32` | `3.87` | `8273.99` | `3.04` | `10543.22` |

Acceptance checks:

- `run_start` now records the live accumulation setting:
  - `ga1`: `grad_accum_steps=1.0`
  - `ga4`: `grad_accum_steps=4.0`
- `pretrain` writes normal trainer + model checkpoints with the new flag.
- Default behavior is preserved when `--grad-accum-steps` is omitted.

## Problems

- The smoke corpus is intentionally tiny, so the raw `tok/s` numbers are not
  useful as a throughput conclusion. This run is a configuration-surface
  acceptance check, not a training-performance claim.

## Learnings

- `pretrain` no longer hard-locks effective batch size to one micro-batch; the
  existing `Trainer` accumulation machinery is now reachable from the CLI.
- A generic tokenizer-building example is enough for dataset-specific vocab
  experiments. The core training binary can stay corpus-agnostic while still
  supporting smaller-vocab workflows.
