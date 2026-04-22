# Train SFT Collate/Loss Cache

## Context

`train_sft` still rebuilt several batch-loss inputs inside the hot loss call:

- per-token `gather_indices`
- per-token `mask_values`
- per-row `inv_counts`
- the `[seq, 1]` all-ones tensor used for row sums

That kept the code correct, but it meant every optimizer step re-traversed the
same shifted labels and reallocated/uploaded a constant `ones` matrix even
though the collate pass had already seen the same supervision layout.

## What Worked

- `BatchedTokenizedSft` now precomputes `gather_indices`, `mask_values`, and
  `inv_counts` during collate, so the loss path consumes one batch scratch
  object instead of rebuilding those vectors every step.
- The `[seq, 1]` all-ones tensors are now allocated once per run and kept in
  the trainer retain set, so the batched loss no longer recreates that constant
  on every step.
- The numeric contract stayed fixed: `assistant_masked_causal_loss_batch`
  still matches the arithmetic mean of the single-example losses.

## Verification

- `cargo test -p train --release --bin train_sft -- --nocapture`
- `cargo clippy -p train --all-targets -- -D warnings`
- `git diff --check -- crates/train/src/bin/train_sft.rs`

## Metal Throughput Smoke

Command:

```bash
cargo run -p train --release --features metal --bin train_sft -- \
  --model-family qwen35 --backend metal \
  --model /tmp/scratch_qwen35_bpe16k_25m_ctx256/latest \
  --data /tmp/instruction_short_mix_fit256.jsonl \
  --seq-len 256 --steps 10 --save-every 10 --log-every 10 \
  --batch 4 --out /tmp/train_sft_perf_cache_reuse
```

Environment:

- Backend: `metal`
- Model: `/tmp/scratch_qwen35_bpe16k_25m_ctx256/latest`
- Dataset: `/tmp/instruction_short_mix_fit256.jsonl`
- Hardware: Apple Silicon local Metal box
- Commit: `HEAD` at run time

Result:

- Final `tok_per_sec`: `238.998078`

## Δ vs Baseline

Baseline: [2026-04-22-train-batch-forward-speedup.md](2026-04-22-train-batch-forward-speedup.md)

| metric | baseline | now | Δ |
| --- | --- | --- | --- |
| `train_sft` `--batch 4` final `tok_per_sec` | `232.056994` | `238.998078` | `+3.0%` |

## Rule

If a training loop already builds batch structure during collate, the loss path
should consume that structure directly instead of rebuilding equivalent gather
/ mask scratch inside the hot step function.
