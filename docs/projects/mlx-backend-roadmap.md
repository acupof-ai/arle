# MLX Metal Backend Roadmap

Reference review:

- [../reviews/2026-04-15-metal-ecosystem-route-correction.md](../reviews/2026-04-15-metal-ecosystem-route-correction.md)
- [../experience/errors/2026-04-27-metal-serving-gap-vs-industry.md](../experience/errors/2026-04-27-metal-serving-gap-vs-industry.md)

## Current State

Apple Silicon 的 Rust Metal 路径现在已经不是实验性占位：

- `MetalBackend` 已通过 `mlx-sys` + C++ bridge 接通 Qwen3 / Qwen3.5 的真实加载和生成路径
- `metal_serve` 可以直接提供 OpenAI 兼容 HTTP 服务
- `metal_bench` 可以保存/比较 baseline，并做本地性能回归

当前 serving 架构仍然有一个明确边界：

- 标准 `metal_serve` 已经走 live Metal scheduler runtime，不再默认走串行
  `BackendRuntimeHandle`
- Metal DFlash 仍然走串行 fallback，因为 speculative decode 还没接进新的
  scheduler runtime
- 当前 scheduler runtime 已有第一版跨请求 decode batching，但边界还很明确：
  - Qwen3 同长度 decode batch 会走共享 MLX 图
  - Qwen3.5 同长度 decode batch 已接到 batched compiled-model step
  - Qwen3.5 当前仍需每步 concat/split request-local KV / recurrent state，
    所以 quick HTTP sweep 还没有出现明显台阶
  - 变长 decode batch 仍然没有进入 batched GPU 路径
- Qwen3.5-0.8B 的单请求路线现在分成两条：MLX SafeTensors 4bit 已经
  贴到公开 Apple-native SOTA，M4 Pro 20c 上 serving-equivalent step-driver
  `1024/256` 达到 305.5 tok/s；GGUF Q4_K_M 已经从 scalar/raw 路径拉到
  MLX affine/tiled quant 主线，但同一 `1024/256` profile 仍在 202.1 tok/s
  左右，后续 GGUF 要作为独立 kernel/weight-format 缺口继续追。

当前展示数据统一以 2026-04-28 的 matched profile 为准：

| path | profile | mode | gen tok/s mean | p50 | TTFT mean | peak RSS |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| MLX SafeTensors 4bit | 1024 prompt / 256 decode | step-driver | 305.5 | 304.7 | 206 ms | 652 MB |
| MLX SafeTensors 4bit | 1024 prompt / 256 decode | direct | 289.6 | 292.5 | 225 ms | 663 MB |
| GGUF Q4_K_M | 1024 prompt / 256 decode | direct | 202.1 | 202.6 | 241 ms | 1429 MB |

GGUF 慢的直接原因不是 scheduler：单请求已经走到 C++ compiled model /
MLX bridge，但 GGUF `Q4_K_M` 不是 MLX 原生 4bit affine 权重格式。它需要
ARLE 自己的 packed K-quant bridge kernel 在 matmul 里解 block scale/min/高位
布局，lm_head/linear 热路径也不能复用 MLX 4bit 的最优 lowering；同 profile
下 RSS 也从 652 MB 提到 1429 MB。后续要追 GGUF，重点不是再调 scheduler
包装，而是重做 GGUF K-quant decode/lm_head 的 Metal kernel 或加载期转换成
MLX-native quantized layout。

这意味着今天的 Metal 已经不再是“纯串行 serving”，但还没有达到 CUDA
路径那种真正以 batched decode / prefix reuse 为核心的 serving 形态。

本路线现在按两个外部基线校准：

- `mlx-lm` 是 direct execution / cache behavior 的 Apple-native 参考
- `vllm-metal` / Docker Model Runner 是 Apple serving 的产品参考

结论是：Metal 的主线目标应该是 scheduler-first serving，不再是继续把单请求优化当作主线累加。

## Near-Term Work

### P0 · Serving floor

1. 把跨请求 batched decode 接进现有 live Metal scheduler runtime。
   当前状态：Qwen3 / Qwen3.5 同长度 decode batch 已落地；下一步是变长
   batch 和去掉 Qwen3.5 每步 batch-state concat/split，而不是继续把
   same-length 路径包装成完成态。
   补充状态：Qwen3.5 MLX 4bit 单请求 step-driver 已经贴到 oMLX M4 Pro
   20c 的 1k single-request 公开基线；Qwen3.5 GGUF 单请求 matmul/lm_head
   floor 在 matched `1024/256` profile 上是 202.1 tok/s。二者都不是 serving
   完成态，下一步仍然要把相同 kernel 收益带进 scheduler batching、变长
   decode 和 Qwen3.6/MoE 路径。
   2026-04-28 额外保留了一个 Metal-only checkpoint：Qwen3.5 C++ compiled
   session 在 prefill / scalar decode 前会先 drain 其它 request 的活动
   session，并且 scheduler 现在能输出一个 local logical serve plan。这个
   checkpoint 只作为后续 runtime-owned batched state 的基础层保存；它不代表
   continuous batching / paged KV / prefix lifecycle 已经统一完成。证据见
   [`../experience/wins/2026-04-28-bench-guidellm-metal-qwen35-session-handoff.md`](../experience/wins/2026-04-28-bench-guidellm-metal-qwen35-session-handoff.md)。
   Qwen3.6-35B-A3B 也做了 2026-04-27 的本地 quick check，确认本地
   Metal 路径仍可加载和执行。该短序列结果不作为 DFlash 优化依据；DFlash
   后续只看 long-context / 超长序列 workload。
2. 把 prefix cache / KV pool 生命周期接到多请求服务路径，而不是只在单请求 fallback 中复用。
   当前状态：Qwen3 live runtime 已接上 runtime-owned prefix cache + shared KV
   pool；admission 会先 lookup/import，再把 suffix 交给 scheduler，terminal
   prefill 会把 aligned prompt prefix publish 回共享 cache。Qwen3.5 也已进入这条
   live prefix reuse 路径，但当前实现是 replayed snapshot cache，不是 zero-copy
   shared recurrent-state ownership。
3. 暴露 Metal queue depth / prefix hit / active + peak memory / KV util 等 serving 级指标。
   当前状态：runtime-backed queue / TTFT / E2E / MLX active/peak/cache memory
   已落地；`prefix_hit_rate` 现在已在 Qwen3 live repeated-prefix smoke 中变成
   非零，`Qwen3.5` 路径仍待补齐。
   补充状态：`metal_request` / `metal_bench` / `metal_serve` 现已暴露
   `--memory-limit-bytes` / `--cache-limit-bytes` / `--wired-limit-bytes`，
   allocator control 不再只能靠 MLX 内部默认值。

### P1 · Product surface

4. 完成 `/v1/responses` streaming parity，而不是只停留在 non-streaming 子集。
   当前状态：已落地。SSE 现在稳定发出 `response.created`、
   `response.output_text.delta`、`response.completed`，然后再发 `[DONE]`。
5. 增加结构化输出 / constrained decoding，让 tool-calling 成为一等路径。
6. 提供一条 Apple Silicon 的单命令安装 / 启动路径，避免用户理解 Cargo features。
   当前状态：已落地。`scripts/start_metal_serve.sh` 是第一条推荐入口。

### Background work, not main thread

7. 继续做 Qwen3.5/Qwen3.6 decode/prefill 热路径，但只保留有 profiler 或
   benchmark 证据的改动；direct-bench 提升必须明确标注，不能写成 serving
   吞吐已经闭环。
8. 在 Metal 路径里把“不支持的架构”保持为显式失败，不允许静默按 Qwen 解析。

## Quantized KV Posture

Metal 这条线现在要把“量化 KV 是否需要做”说清楚，不再和 CUDA 能力混写：

- 当前 Metal / MLX serving **不支持** `fp8` / `int8` / `tq2-4` 这类量化 KV cache。
- 今天的 Metal KV 仍然是模型原生 dtype，通常是 `bf16` / `f16`。
- 现阶段这不是 P0，也不是 P1。Metal 的主瓶颈仍然是 batched decode、live prefix
  reuse、serving observability，以及产品级 API / DX。

什么时候才值得推进 Metal KV quant：

1. `M0.2/M0.3/M0.4` 已完成，Metal serving 已具备真正的并发调度和复用。
2. 目标 workload 明确落在 `C > 4` 且 prompt / session 长度持续超过 `8K` tokens。
3. 或者 Apple 用户明确需要在统一内存机器上塞更大的模型 / 更长上下文。

当前判断：

- `FP8 KV` 在 MLX / Apple Silicon 上不是优先路线。当前 MLX 没有一等 FP8 tensor
  dtype，Apple GPU 也没有 CUDA 那种 FP8 decode kernel 生态。
- 如果未来真的做，第一候选更像是 `INT8` 或 `TurboQuant / PolarQuant` 风格的
  压缩 KV，而不是照搬 CUDA 的 FP8 方案。

## Model Scope

当前已接通并持续优化：

- Qwen3
- Qwen3.5

后续扩展优先级：

1. Gemma 4 text path

Llama 不在这条近期路线的优先级里。
