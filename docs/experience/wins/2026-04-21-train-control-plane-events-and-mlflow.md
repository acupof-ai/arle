# Train Control Plane Events and MLflow Artifacts

## Context

2026-04-21 这轮把 train-side observability 从“shared async sink + JSONL/stdout”
继续推进到可运营的形态：

- 所有活跃训练入口都走统一 `/v1/train/status|events|stop|save`
- operator `save/stop` 请求本身进入事件流
- `train_grpo` 的 `run_end` / final checkpoint 不再从 controller snapshot
  反推完成度，而是显式按已完成 SFT step 和 GRPO iter 计数
- MLflow 不再只收 metrics / run status，也会根据 `checkpoint` 事件上传
  checkpoint artifact

## What Worked

- `TrainingController` 增加 recent-record ring 后，`/v1/train/events` 可以承载
  trainer metric、lifecycle event、以及 control-plane operator intent，接口
  足够稳定，`pretrain` / `train_sft` / `train_grpo` / `train_multi_turn`
  都能复用。
- `train_grpo` 改成显式 `completed_sft_steps` /
  `completed_grpo_iters` 后，completed/stopped 两条路径都能产出正确
  `run_end` 和 final checkpoint step，不再受 controller snapshot 偶发漂移影响。
- MLflow 直接复用现有 async sink worker + REST API 足够轻量：
  `runs/create` / `runs/log-batch` / `runs/update` 负责 run metadata 与指标，
  `checkpoint` 事件驱动 `mlflow-artifacts` `PUT` 上传 checkpoint 文件，
  不需要把 Python SDK 嵌进热路径。

## Verification

- `cargo test -p train --release`
- `cargo clippy -p train --all-targets -- -D warnings`
- `pretrain --serve 19111`:
  - `GET /v1/train/status`
  - `GET /v1/train/events`
  - `POST /v1/train/save`
  - `POST /v1/train/stop`
  - graceful stop at step 77 with checkpoint + `run_end`
- `train_sft --serve 19112`:
  - `GET /v1/train/status`
  - `GET /v1/train/events`
  - `POST /v1/train/save`
  - `POST /v1/train/stop`
  - graceful stop at step 141 with checkpoint + `run_end`
- `train_grpo --serve 19115`:
  - live `GET /v1/train/status`
  - live `GET /v1/train/events`
  - `POST /v1/train/save`
  - `POST /v1/train/stop`
  - stopped during SFT warmup with final checkpoint at `step_000000` and
    `run_end { completed_sft_steps=6, completed_grpo_iters=0 }`
- `train_multi_turn --features metal --serve 19114`:
  - live `GET /v1/train/status`
  - live `GET /v1/train/events`
  - `POST /v1/train/save`
  - `POST /v1/train/stop`
  - Metal run stopped at iter 240 with checkpoint + `run_end`
- `eval_lm --metrics-jsonl ...` on a freshly trained Qwen3.5 checkpoint dir
- mock MLflow e2e:
  - `pretrain` with `TRAIN_MLFLOW_TRACKING_URI=http://127.0.0.1:19116`
  - observed `runs/create`, repeated `runs/log-batch`,
    artifact `PUT` for `model.safetensors` / `config.json` /
    `generation_config.json` / `tokenizer.json`, then `runs/update`

## Rule

Train-side observability is no longer “metrics only”. Treat `/v1/train/events`
and `checkpoint`-driven artifact export as part of the runtime contract, and
never derive completion counts from controller snapshots when the loop already
knows the exact completed work.
