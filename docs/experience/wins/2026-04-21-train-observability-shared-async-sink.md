# Train observability now has lifecycle events and a shared async sink

## Context

The train-side stack already had a lightweight `MetricSink`, but it only
handled scalar step metrics. That left the active binaries split:

- `Trainer`-based paths emitted plain step metrics
- RL paths appended their own schema by hand
- no run lifecycle existed
- no checkpoint/artifact events existed
- any future remote exporter would have needed to block the training loop or
  grow a second parallel logging surface

## What Worked

- Extended `MetricSample` with `phase`, so the active train paths now write
  one metric stream that distinguishes `train`, `eval`, and RL phases.
- Added generic `TrainEvent` records for lifecycle/artifact events:
  - `run_start`
  - `trainer_checkpoint`
  - `checkpoint`
  - `status`
  - `run_end`
- Added a cloneable `SharedSink` with a background worker thread. Binaries now
  keep one clone for lifecycle/artifact events while `Trainer` owns another
  clone as `Box<dyn MetricSink>`.
- Kept stdout + JSONL as the local truth while moving all sink I/O off the
  foreground training loop.
- Wired the active entrypoints:
  - `pretrain`
  - `train_sft`
  - `train_grpo`
  - `train_multi_turn`
  - `eval_lm`
- `train_multi_turn` now emits eval metrics through the same sink path instead
  of only `println!`.
- `Trainer` now emits a `trainer_checkpoint` event when it writes
  `trainer_state.json + optimizer.safetensors`.

## Verification

- `cargo test -p train --release --test test_metrics -- --nocapture`
- `cargo test -p train --release --bin train_grpo -- --nocapture`
- `cargo test -p train --release`
- `cargo clippy -p train --all-targets -- -D warnings`
- `cargo run -p train --release --bin pretrain -- --model-family qwen35 --corpus crates/train/data/sample.txt --tokenizer models/Qwen3-0.6B/tokenizer.json --out /tmp/train_obs_pretrain --steps 2 --batch 1 --seq 16 --lr 1e-4 --log-every 1 --save-every 2 --hidden 16 --layers 2 --heads 2 --kv-heads 1 --head-dim 8 --intermediate 32 --max-pos 32 --seed 7 --metrics-jsonl /tmp/train_obs_pretrain.jsonl`
- `cargo run -p train --release --bin train_grpo -- --model-family qwen35 --sft-steps 1 --grpo-iters 1 --batch-prompts 2 --group-size 2 --seq 8 --lr 1e-4 --seed 7 --metrics-jsonl /tmp/train_obs_grpo.jsonl`

Observed JSONL shape:

- `run_start`
- `metric` lines with `phase=train|eval|grpo|stepwise_grpo|sequence_gspo`
- `trainer_checkpoint` / `checkpoint`
- `run_end`

## Rule

Train observability should have one event stream, not parallel ad-hoc logging
paths. The trainer/runtime API is the shared async sink; lifecycle, metrics,
and artifacts all flow through it.
