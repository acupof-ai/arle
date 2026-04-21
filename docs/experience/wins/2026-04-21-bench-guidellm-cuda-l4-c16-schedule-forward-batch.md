# guidellm fast c16 schedule-forward-batch — CUDA L4, 2026-04-21

## Goal

- 验证调度器从隐式 `StepPlan` 分支重构为显式 `ScheduleBatch -> ForwardBatch` 双层语义后，c16 长上下文路径是否保持可用并给出稳定吞吐数据。

## Hypothesis

- 重构为显式控制面/执行面对象后，行为应与旧逻辑等价，不应引入功能回归（非法地址、0-token 早停、TTFT/ITL 统计短路）。

## Command

```bash
./scripts/bench_guidellm.sh \
  cuda-l4-c16-schedule-forward-batch-fresh \
  --fast \
  --target http://127.0.0.1:8017 \
  --model Qwen/Qwen3-4B \
  --processor Qwen/Qwen3-4B
```

## Environment

- Backend: `cuda`
- Model: `Qwen/Qwen3-4B`
- Hardware: `NVIDIA L4 23GB`, Driver `580.82.07`, CUDA `13.0`
- Server flags:
  - `--num-slots 16`
  - `--max-seq-len 4608`
  - `--mem-fraction-static 0.94`
  - `--chunked-prefill-size 4096`
  - `--max-prefill-tokens 16384`
  - `--enable-mixed-chunk true`

## Results (clean serial run)

| rate | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| conc16 | 3957.1 | 8064.0 | 95.95 | 106.40 | 83.25 | 0.30 |

Raw summary (`benchmarks.json`):

- `completed=10`
- `incomplete=15`
- `out_tok_s=83.2533`

## Problems

- 如果在同一服务实例上先跑高压 smoke，再跑 benchmark，会污染前缀/KV 常驻状态并显著拉高 TTFT，导致无效对比。
- 观测到 host pinned 池高水位时 demote 失败告警（`host pinned pool exhausted`），但本次 clean run 的 benchmark 可完成并输出有效指标。

## Learnings

- c16 benchmark 必须“串行 + 干净服务实例”执行；benchmark 之间需重启服务清空状态，否则结果会被前一轮流量历史污染。
- 本轮 `ScheduleBatch -> ForwardBatch` 结构重构未引入功能性回归，且在干净实例上吞吐可达稳定区间。

## Δ vs baseline

- Baseline: [2026-04-21-cuda-c16-streaming-and-guidellm-fast-validation.md](./2026-04-21-cuda-c16-streaming-and-guidellm-fast-validation.md)

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p50 | 3204.8 ms | 3957.1 ms | +23.5% |
| ITL p50 | 74.58 ms | 95.95 ms | +28.6% |
| out tok/s | 55.41 | 83.25 | +50.2% |

## Artefacts

- `bench-output/2026-04-21-cuda-l4-c16-schedule-forward-batch-fresh/benchmarks.json`
- `bench-output/2026-04-21-cuda-l4-c16-schedule-forward-batch-fresh/benchmarks.csv`
- `bench-output/2026-04-21-cuda-l4-c16-schedule-forward-batch-fresh/benchmarks.html`

