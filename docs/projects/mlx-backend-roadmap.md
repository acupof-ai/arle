# MLX Metal Backend Roadmap

Reference review:

- [../reviews/2026-04-15-metal-ecosystem-route-correction.md](../reviews/2026-04-15-metal-ecosystem-route-correction.md)

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
- Qwen3.5-0.8B GGUF Q4_K_M 的单请求 decode floor 已经从 scalar/raw 路径
  拉到 MLX affine/tiled quant 主线：Q5_K / Q8_0 load-time repack，Q6/group16
  qmv tile 调整后，512 prompt / 1024 decode 在 M4 Pro 上达到 211.7 tok/s。

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
   补充状态：Qwen3.5 GGUF 单请求 matmul/lm_head floor 已经跨过 200 tok/s；
   这不是 serving 完成态，下一步仍然要把相同 kernel 收益带进 scheduler
   batching、变长 decode 和 Qwen3.6/MoE 路径。
   Qwen3.6-35B-A3B 也做了 2026-04-27 的本地 quick check：baseline decode
   约 63 tok/s，DFlash paired TPOT 15.23 ms -> 15.45 ms，没有显示真实收益；
   这条线下一步看 prefill/TTFT 与 MoE/GDR 执行，而不是继续堆 speculative。
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
