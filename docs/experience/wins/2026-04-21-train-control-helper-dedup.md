# Train Control Helper Dedup

## Context

`pretrain`、`train_sft`、`train_grpo` 和 `train_multi_turn` 都在重复同一层训练运行时样板：开 shared metrics sink、挂 `TrainingController`、发 `run_start/run_end`、同步 `dropped_metrics`、以及按 `--serve` 起 `/v1/train/*` 控制面。

## What Worked

把这层重复逻辑收回到 `crates/train/src/control.rs` 里的 5 个小 helper：

- `open_run_metrics`
- `sync_status`
- `emit_run_start`
- `emit_run_end`
- `serve_if_requested`

这样四个活跃 bin 都只保留各自真正不同的训练逻辑，控制面和观测接线不再各写一遍。

## Rule

训练 bin 之间如果共享的是“运行时样板”而不是“训练算法”，就下沉成小 helper，不要再复制整段 controller/metrics/serve 接线代码。
