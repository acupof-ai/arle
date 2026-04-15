# MLX Metal Backend Roadmap

Reference review:

- [../reviews/2026-04-15-metal-ecosystem-route-correction.md](../reviews/2026-04-15-metal-ecosystem-route-correction.md)

## Current State

Apple Silicon 的 Rust Metal 路径现在已经不是实验性占位：

- `MetalBackend` 已通过 `mlx-sys` + C++ bridge 接通 Qwen3 / Qwen3.5 的真实加载和生成路径
- `metal_serve` 可以直接提供 OpenAI 兼容 HTTP 服务
- `metal_bench` 可以保存/比较 baseline，并做本地性能回归

当前 serving 架构仍然有一个明确边界：

- HTTP 服务走 [`backend_runtime.rs`](../../infer/src/backend_runtime.rs) 的串行 runtime
- [`metal_scheduler.rs`](../../infer/src/metal_scheduler.rs) 目前还是 CPU 调度骨架，还没接到真实执行热路径

这意味着今天的 Metal 能跑、能测、能回归，但还不具备 CUDA 路径那种连续批处理 serving 能力。

本路线现在按两个外部基线校准：

- `mlx-lm` 是 direct execution / cache behavior 的 Apple-native 参考
- `vllm-metal` / Docker Model Runner 是 Apple serving 的产品参考

结论是：Metal 的主线目标应该是 scheduler-first serving，不再是继续把单请求优化当作主线累加。

## Near-Term Work

### P0 · Serving floor

1. 把 `MetalScheduler` 接入真实执行循环，替换当前串行 runtime。
2. 把 prefix cache / KV pool 生命周期接到多请求服务路径，而不是只在单请求 fallback 中复用。
3. 暴露 Metal queue depth / prefix hit / active + peak memory / KV util 等 serving 级指标。

### P1 · Product surface

4. 完成 `/v1/responses` streaming parity，而不是只停留在 non-streaming 子集。
5. 增加结构化输出 / constrained decoding，让 tool-calling 成为一等路径。
6. 提供一条 Apple Silicon 的单命令安装 / 启动路径，避免用户理解 Cargo features。

### Background work, not main thread

7. 持续优化 Qwen3.5 decode/prefill 热路径，但前提是：
   - 不打断 `M0.2/M0.3/M0.4`
   - 有 profiler 或 benchmark 证据
   - 不把 direct-bench 提升误判成 serving 提升
8. 在 Metal 路径里把“不支持的架构”保持为显式失败，不允许静默按 Qwen 解析。

## Model Scope

当前已接通并持续优化：

- Qwen3
- Qwen3.5

后续扩展优先级：

1. GLM-4
2. Gemma 4 text path

Llama 不在这条近期路线的优先级里。

