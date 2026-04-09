# MLX Metal Backend Roadmap

## Current State

Apple Silicon 的 Rust Metal 路径现在已经不是实验性占位：

- `MetalBackend` 已接通 Qwen3 / Qwen3.5 的真实加载和生成路径
- `metal_serve` 可以直接提供 OpenAI 兼容 HTTP 服务
- `metal_bench` 可以保存/比较 baseline，并做本地性能回归

当前 serving 架构仍然有一个明确边界：

- HTTP 服务走 [`backend_runtime.rs`](../../infer/src/backend_runtime.rs) 的串行 runtime
- [`metal_scheduler.rs`](../../infer/src/metal_scheduler.rs) 目前还是 CPU 调度骨架，还没接到真实执行热路径

这意味着今天的 Metal 能跑、能测、能回归，但还不具备 CUDA 路径那种连续批处理 serving 能力。

## Near-Term Work

1. 把 `MetalScheduler` 接入真实执行循环，替换当前串行 runtime。
2. 把 prefix cache / KV pool 生命周期接到多请求服务路径，而不是只在单请求 fallback 中复用。
3. 持续优化 Qwen3.5 decode/prefill 热路径，优先做 profiler 证明确实有效的改动。
4. 在 Metal 路径里把“不支持的架构”保持为显式失败，不允许静默按 Qwen 解析。

## Model Scope

当前已接通并持续优化：

- Qwen3
- Qwen3.5

后续扩展优先级：

1. GLM-4
2. Gemma 4 text path

Llama 不在这条近期路线的优先级里。

## Historical Notes

`docs/archives/` 下的 MLX 文档保留的是历史阶段记录，很多内容描述的是更早的 Python/asyncio 原型路径，不应当再当作当前实现文档使用。
