# M1 — Unified Backend Telemetry

> Backend-unification 第一个 milestone。CUDA / Metal 两端 InferenceEngine telemetry 投影到同一个
> `EngineTelemetry` snapshot,让 `/v1/stats` 和 `bench_guidellm.sh` 输出立刻可比。
> 详细规格见 [`docs/plans/backend-unification.md`](../../plans/backend-unification.md) §M1。

## Context

Pre-M1 现状:

- CUDA 侧 metrics 走 `scheduler/metrics.rs::SchedulerMetrics::snapshot()`,字段 X 个。
- Metal 侧 metrics 走 `backend/metal/runtime.rs::maybe_refresh_runtime_metrics`,字段集 Y 个,和 CUDA 不重叠。
- `/v1/stats` 各自打各自的字段,bench/wins doc 数字格式不一致。
- 任何 perf 工作都先要花时间对齐字段才能比。

主线目标:**力争世界第一 inference runtime + 统一所有后端**(2026-05-07 用户战略重定向)。M1 是 6 个
milestone 中风险最低、为 M2-M5 立 baseline 的第一个 win。

## What Worked

### 设计

- 加 `EngineTelemetry` snapshot(在 `infer/src/server_engine/types.rs`),字段 8 个:`ttft_us`、
  `itl_p50_us`、`itl_p99_us`(`Option<f64>`,缺失即 `None`),`queue_depth`、`active_requests`、
  `batch_occupancy`、`kv_tier_hit_rates: HashMap<String, f64>`、`timestamp_ms`。
- `InferenceEngine` trait 加 `fn telemetry(&self) -> EngineTelemetry { EngineTelemetry::default() }`
  ——**带默认实现**,所有现有 impl 不必立刻改,后向兼容。
- 投影**单点**:`ServerMetrics::snapshot_engine_telemetry()`(`infer/src/metrics/render.rs`),
  CUDA 和 Metal 都 push 到同一份 `ServerMetrics`,这里一次投影。
- `ttft_us = histograms.ttft.percentile(0.50) × 1e6`,
  `itl_p50/p99 = histograms.tpot.percentile(...)`(time-per-output-token,即 inter-token-latency)。
- `batch_occupancy = kv_gpu_utilization().clamp(0.0, 1.0)`(已有 KV pool 视图)。
- Tier hit rates:`T0 = prefix_hit_rate()` 总是发;`T1`/`T2`/`T3` **only emitted when has staged
  blocks**(absent ≠ 0,**不造数**)。Metal 当前没有 T1/T2/T3 wiring → key 缺席,不假阴。

### Wiring(关键)

`state.metrics`(HTTP `AppState`)是 `ServerMetrics` 不是 `SchedulerMetrics`。M1 wiring 路径:

1. `RequestHandle` trait(`infer/src/request_handle.rs`)加 `fn server_metrics(&self) -> Option<&ServerMetrics>`,
   默认 `None`。
2. `SchedulerHandle`(`infer/src/scheduler/types.rs`)加 `Option<ServerMetrics>` 字段 +
   `with_server_metrics(...)` builder + `server_metrics()` accessor。
3. CUDA 侧 `scheduler/cuda/core/construction.rs::with_config` 在创建 handle 时
   `.with_server_metrics(metrics_for_handle)`。
4. Metal 侧 `backend/metal/runtime.rs::spawn_metal_scheduler_handle_*` 同样 attach;
   并在 `MetalSchedulerHandle` 的 `RequestHandle` impl 里 forward `server_metrics()` 到
   `self.inner.server_metrics()`(round-1 review 抓到漏 forward,round-2 已修)。
5. `RequestHandleInferenceEngine::telemetry()` 调
   `self.handle.server_metrics().map(|m| m.snapshot_engine_telemetry()).unwrap_or_default()`。
6. `LoadedInferenceEngine::telemetry()` 转发到内部 engine。
7. HTTP `/v1/stats` 不需要新 wire engine 句柄:`render_stats_json` 和 `render_summary` 已经持有
   同一份 `ServerMetrics`,直接调 `snapshot_engine_telemetry()` 然后追加 `engine_*` 前缀字段。
   **老字段全部保留**,bench script regex 解析器零改动。

### Review

≥2 轮 codex review 流程:

- **Round 1** 抓到 [P1 deadlock]:`render_summary()` 顶部已 lock `histograms` mutex,line 1141 调
  `snapshot_engine_telemetry()` 内部又 lock 同一 mutex —— `std::sync::Mutex` 非 reentrant,**必死锁**。
  + [P2] Metal `MetalSchedulerHandle::RequestHandle::server_metrics()` 漏 forward → telemetry 永远空。
- **Round 1 fix**:`snapshot_engine_telemetry` 拆 helper `snapshot_engine_telemetry_with(ttft, p50, p99)`
  接受 histogram 字段作为参数(no-lock),`render_summary` 顶部一次性提取所有 histogram 值(块内 lock+drop)
  再调 helper。Metal forwarder 在 `impl RequestHandle for MetalSchedulerHandle` 里加
  `fn server_metrics(...) → self.inner.server_metrics()`。
- **Round 2** 0 个 M1-范围内的 finding。两条 [P2] 全是 Track A 的 deterministic GEMM 工作,不归 M1。

## Bench Status

**`pending-remote`** — M1 是纯 additive 观测代码,不改任何运行时决策:
- 无新 hot-path 调用
- 无新 syscall(timestamp 取一次,在 snapshot 而非 forward path)
- mutex lock 在投影点一次,投影是计算 + 字段拷贝(P50/P99 quantile 已经在 histogram 内 cached)

按 CLAUDE.md §Benchmarks 仍立 stub,实际 vs vLLM bench 在下一个有 perf-relevant 改动的 milestone
(预期 M3 unified scheduler decision layer)合并跑。验收基线:

- 验证 baseline:9833068 之前最近一次 `bench_guidellm.sh` 数字不变(±2%)
- 跟踪 plan:`docs/plans/backend-unification.md` §M3 acceptance 条款 `bench_guidellm.sh cuda-h100 不回退 ≥ 95% baseline`

## Verification

- `cargo check -p infer --no-default-features --features cuda,no-cuda` PASS(13.49 s)—— Mac CUDA-Rust
  typecheck 路径,验证 backend-isolation cfg-gate 没破。
- `cargo check -p infer --no-default-features --features metal` PASS(1.49 s,non-macOS stub 模式)。
- `cargo check -p infer --features cuda` skipped:与 M1 无关的 nvcc + GCC16 `type_traits` 不兼容
  (在 `crates/cuda-kernels/csrc/attention/decode_attention_quantized.cu`),Track A / 上游已知问题。
  M1 diff 不碰 CUDA C 任何文件,`cuda,no-cuda` 路径已经覆盖所有 Rust-cuda-feature 改动的类型检查。

## Rule

- 跨后端 telemetry 走单点 projection(`ServerMetrics`),每端只 push 内部 metric,projection 从一份 source
  pull —— **不在 backend impl 里各自构造 EngineTelemetry**,否则字段语义会再次分叉。
- trait 新方法**始终**给 default impl,降低破坏面;一次 review 漏一个不足为奇,**至少 2 轮** review。
- 持锁状态下不要调用任何会再次 lock 同一资源的方法;`std::sync::Mutex` 不会救你。
- M1 完成 = M2 起跑信号:`/tmp/codex-m2-directive.txt` 已 stage,cron `*/12 * * * *` 检测到 M1 commit
  会自动推给 codex。
