<p align="center">
  <strong>ARLE</strong><br>
  <em>面向 LLM 智能体的 KV-cache 优先推理引擎。纯 Rust 实现，CUDA 为主部署路径。</em>
</p>

<p align="center">
  <a href="https://cklxx.github.io/agent-infer/"><img src="https://img.shields.io/badge/website-cklxx.github.io%2Fagent--infer-D97757?style=flat-square" alt="Website"></a>
  <a href="https://github.com/cklxx/agent-infer/actions"><img src="https://github.com/cklxx/agent-infer/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://github.com/cklxx/agent-infer/releases"><img src="https://img.shields.io/github/v/release/cklxx/agent-infer?include_prereleases" alt="Release"></a>
</p>

<p align="center">
  <a href="https://cklxx.github.io/agent-infer/">官网</a> ·
  <a href="#-最新动态">动态</a> ·
  <a href="#-当前状态一览">状态</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#api">API</a> ·
  <a href="#架构">架构</a> ·
  <a href="ROADMAP.md">路线图</a> ·
  <a href="CHANGELOG.md">变更日志</a>
</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

---

## 📰 最新动态

<!-- 仅保留最近 3 条。更早历史见 CHANGELOG.md。 -->

- **2026-04-20** — Metal DFlash 长提示词 prefill 修复（`fast_forward_prefill`，commit `3bc8802`），批量终止 `eval` 通过 `async_eval` 延迟（commit `d8cb2f4`）。DFlash 在 Metal 上已成为 Qwen3.5 的默认路径，并通过 guidellm 10-strategy 套件（5400-token 提示词）验证：零 `WrongPhase` 错误，100% 请求成功。完整用法见 [`docs/resources/metal-dflash.md`](docs/resources/metal-dflash.md)。
- **2026-04-19** — DFlash 在 Metal 上默认开启（commit `47f958f`）。Qwen3.5-4B-4bit 在 B≤2 批量验证下与标量路径逐位一致，并发 c=1..8 稳定。
- **2026-04-16** — Metal packed-batch 并发解码修复：`extend_kv_cache` batch-dim bug 修好，变长加性 mask 在 MLX ≥ 0.32 SDPA 下改用 bf16 发射。4× / 8× 并发下 packed decode 稳定。

完整历史：[CHANGELOG.md](CHANGELOG.md) · 接下来：[ROADMAP.md](ROADMAP.md)

## 🚦 当前状态一览

四个维度，一个维度回答一个问题。权威的支持矩阵在
[docs/support-matrix.md](docs/support-matrix.md)；稳定性分级
（**Stable** → **Beta** → **Dev**）的定义见
[docs/stability-policy.md](docs/stability-policy.md)。

### 后端 — *在哪里运行？*

| 后端 | 平台 | 状态 | 已交付 |
|------|------|:----:|--------|
| **CUDA** | Linux + NVIDIA | **Stable** | 主部署路径。持续批处理、前缀缓存、CUDA Graph、FlashInfer。 |
| **Metal** | Apple Silicon | **Beta** | 实时调度、分块 prefill、等长 packed decode。变长批量解码进行中。 |
| **Metal DFlash** | Apple Silicon | **Beta — 默认开启** | 面向 Qwen3 / Qwen3.5 的推测解码。Qwen3-4B bf16 解码 5.9×，Qwen3.5-4B-4bit 与标量路径逐位一致，c=1..8 已验证（2026-04-20）。 |
| **CPU** | 可移植 | **仅开发** | 冒烟测试与请求路径验证，非部署目标。 |

### 模型 — *能加载哪些？*

| 模型 | 注意力 | CUDA | Metal |
|------|--------|:----:|:-----:|
| Qwen3 (0.6B – 72B) | GQA | ✅ | ✅ |
| Qwen3.5-4B | 混合（线性 + 全注意力） | ✅ | ✅ |
| GLM4 | GQA | ✅ | — |
| Llama 3 / 4 | GQA | *规划中* | *规划中* |
| DeepSeek V3 / R1 | MLA | *规划中* | *规划中* |

### HTTP API — *客户端能调用什么？*

| 端点 | 状态 | 备注 |
|------|:----:|------|
| `POST /v1/completions` · `POST /v1/chat/completions` · `GET /v1/models` | **Stable** | OpenAI 兼容。Chat 支持 SSE 流式。 |
| `POST /v1/responses` | **Beta** | 非流式子集 + SSE `output_text.delta`。 |
| `GET /metrics` · `GET /v1/stats` | **Stable** | Prometheus + 人类可读运维面。 |

### 量化 — *能压到多小？*

| 格式 | 状态 | 可用后端 |
|------|:----:|----------|
| FP8 / INT8 / TurboQuant KV | **Beta** | CUDA |
| GPTQ W4 · AWQ W4 | **Beta** | CUDA |
| Q4_K GGUF | **Beta** | CUDA |
| MLX 4-bit | **Beta** | Metal（`start_metal_serve.sh` 规范路径的默认） |

---

<!--
  下方为稳定参考资料：能力、安装、API、架构、开发流程。
  仅在架构或 API 级别发生变化时更新。项目的当前状态体现在上方两节。
-->

## 为什么要 ARLE（agent rl engine）？

智能体工作流每一轮都要付 "prefill 税"：系统提示 + 历史对话 + 工具结果都要被重新处理。上下文越长，**prefill 越主导延迟**。

ARLE（agent rl engine）把这件事当成核心问题：

| 能力 | 做的事情 | 效果 |
|------|----------|------|
| **多轮 KV 复用** | Slot-sticky 前缀匹配：原地复用上一轮 KV。共享系统提示 + 对话前缀直接跳过 prefill。跨会话的 radix 树复用会在 tiered-kv-cache M1 里落地；当前快路径是 per-slot 线性比较。 | 每轮只对新用户消息做 prefill，历史 KV 不再重算 |
| **Token 级 KV 池** | page_size=1 分页（SGLang 风格）。零碎片，即时分配/释放。 | 消除固定页面填充带来的内存浪费 |
| **GPU↔CPU 透明 offload** | 最旧的 KV 块迁移到主机内存；在注意力之前预取回来。 | 支持超出显存的上下文 |
| **Copy-on-Write 块共享** | 分页块 + 引用计数。并发请求共享相同前缀只占一份。 | N 个请求共用 1× 前缀内存 |
| **CUDA Graph 批量解码** | 每个 batch size（1–32）捕获一张图。504 次 kernel launch → 1 次 replay。 | 消除 CPU↔GPU 调度开销 |

**结果**：TTFT 比 SGLang 快 4.6×，吞吐持平 — 因为重活是缓存干的。

---

## 性能

Qwen3-4B 在 A100-SXM4-40GB 上对比 SGLang v0.5.9：

| 并发 | 吞吐 | vs SGLang | TTFT | vs SGLang |
|:----:|:----:|:---------:|:----:|:---------:|
| 1 | 119.5 tok/s | 0.99× | **8.6ms** | **快 4.6×** |
| 4 | 414.8 tok/s | 0.99× | **53.1ms** | **快 2.6×** |
| 8 | 811 tok/s | 0.92× | **68.7ms** | **快 1.1×** |

<details>
<summary>Agent 基准（多轮 + 工具调用）</summary>

| 模型 | 轮次 | 工具调用 | 跨轮 re-prefill | 平均延迟 |
|------|:----:|:--------:|:---------------:|:--------:|
| Qwen3-4B | 10 | 8 | **无**（每轮只 prefill 新 user token） | 31.9s |
| Qwen3-8B | 10 | 11 | **无**（每轮只 prefill 新 user token） | 88.5s |

_"跨轮 re-prefill = 无" 指历史 token 不会被模型重新处理：整段会话 KV 在轮次间原地复用，prefill 成本只随新用户消息长度扩展，而不是总上下文长度。_

</details>

<details>
<summary>优化路径（C=8）</summary>

| 阶段 | 改动 | 吞吐 | 增量 |
|:----:|------|:----:|:----:|
| 0 | Per-request decode 循环 | 128 tok/s | — |
| 1 | Token-level KV 池 + FlashInfer paged decode | 434 tok/s | +239% |
| 2 | 缓冲区预分配 | 681 tok/s | +57% |
| 3 | FlashInfer plan-once-per-step | 690 tok/s | +1% |
| 4 | Embedding + logits 缓冲预分配 | 700 tok/s | +1% |
| 5 | CUDA Graph 批量解码 | 756 tok/s | +8% |
| 6 | 批量 argmax + 跳过 D2D scatter | **811 tok/s** | +7% |

</details>

最新 bench 快照：[docs/experience/wins/](docs/experience/wins/) ·
自己跑一遍：[docs/plans/guidellm-integration.md](docs/plans/guidellm-integration.md)

---

## 快速开始

```bash
# Docker（推荐）
docker run --gpus all -v /path/to/Qwen3-4B:/model \
  ghcr.io/cklxx/agent-infer:latest --model-path /model --port 8000

# 或者从源码构建
git clone https://github.com/cklxx/agent-infer && cd agent-infer
cargo build -p infer --release
./target/release/infer --model-path /path/to/Qwen3-4B --port 8000
```

```bash
# 测试一下
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

**前置要求**：CUDA 12.x，Rust 1.85+，Python 3.10+ 带 `flashinfer-python`（仅构建期使用）。

## Apple Silicon 上的 Metal

```bash
# 默认：以 Metal 特性集构建并运行，绑定 127.0.0.1:8000，
#       加载 mlx-community/Qwen3-0.6B-4bit
./scripts/start_metal_serve.sh

# 覆盖模型 + 端口；额外参数放在 `--` 之后
./scripts/start_metal_serve.sh mlx-community/Qwen3-4B-bf16 8012 -- --warmup 0

# 推测解码（DFlash，Qwen3.5 目标 + 草稿，默认开启）
./scripts/run_dflash.sh           # 起服务
./scripts/run_dflash.sh bench     # 基线 vs DFlash 吞吐
```

`metal_serve`、`metal_bench`、`metal_request` 还暴露 `--memory-limit-bytes`、
`--cache-limit-bytes`、`--wired-limit-bytes` 用于对 MLX 分配器封顶。DFlash
完整参考与支持的模型对：
[docs/resources/metal-dflash.md](docs/resources/metal-dflash.md)。

---

## 支持的模型

| 模型 | 注意力 | 状态 |
|------|--------|:----:|
| Qwen3 (0.6B–72B) | GQA | :white_check_mark: |
| Qwen3.5-4B | 混合（线性 + 全注意力） | :white_check_mark: |
| GLM4 | GQA | :white_check_mark: |
| Llama 3 / 4 | GQA | 规划中 |
| DeepSeek-V3 / R1 | MLA | 规划中 |

完整计划见 [ROADMAP.md](ROADMAP.md)。

---

## API

OpenAI 兼容。当前 HTTP 表面：

- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /v1/models`
- `POST /v1/responses`（当前非流式子集）

流式目前仍在 `/v1/chat/completions`；`/v1/responses` 在 `stream=true`
时会返回明确的 `400`。

```bash
# 流式
curl http://localhost:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"解释一下 KV 缓存"}],"stream":true}'

# 补全
curl http://localhost:8000/v1/completions \
  -d '{"prompt":"The quick brown fox","max_tokens":64,"temperature":0.7}'

# 模型发现
curl http://localhost:8000/v1/models

# Responses API（非流式子集）
curl http://localhost:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"input":"用一句话概括 radix 前缀缓存。","max_output_tokens":32}'
```

<details>
<summary>完整参数参考</summary>

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `max_tokens` | int | 16 | 最大输出 token 数 |
| `temperature` | float | 0.0 | 采样温度（0 = greedy） |
| `top_p` | float | 1.0 | Nucleus 采样 |
| `top_k` | int | -1 | Top-K（-1 = 关） |
| `min_p` | float | 0.0 | Min-P 过滤 |
| `repetition_penalty` | float | 1.0 | 重复惩罚 |
| `frequency_penalty` | float | 0.0 | 频率惩罚 |
| `presence_penalty` | float | 0.0 | 出现惩罚 |
| `stop` | list | null | 停止串 |
| `seed` | int | null | 随机数种子 |
| `stream` | bool | false | SSE 流式 |

</details>

附加端点：`GET /metrics`（Prometheus）、`GET /v1/stats`（人类可读）。
Metal 下这些端点会暴露运行时的实时队列 / 延迟 / MLX 内存统计。

---

## Agent 模式

内置 agent 运行时，支持工具调用：

```bash
./target/release/agent-infer \
  --max-turns 10 --temperature 0
```

根 CLI 二进制位于 `cli` feature 之后；不带 `--features cli` 时不会构建
`agent-infer`。

当前 agent 模式的包边界：

- `agent-infer` → 薄二进制包装
- `cli` → REPL 与斜杠命令
- `infer` → `server_engine::LoadedInferenceEngine` 后端加载，
  `hf_hub::resolve_model_source` 自动发现模型
- `agent` → 会话循环与工具调用恢复
- `tools` / `chat` → 共享工具定义、执行助手、协议类型

若省略 `--model-path`，CLI 会先看 `AGENT_INFER_MODEL`，再从常用目录和
本地 HuggingFace 缓存中自动探测模型。

工具：`python`（执行 Python 片段）、`shell`（执行 bash 命令）。KV 前缀
缓存在每轮会话中原地复用完整上一轮 KV，只有新用户消息（以及任何工具
结果）走 prefill（slot-sticky 匹配；跨会话复用通过 radix 树在
tiered-kv-cache M1 里落地）。
macOS 上工具执行会在 `nsjail` 不可用时自动使用 `sandbox-exec`；Linux 在
装了 `nsjail` 时继续使用。

Apple Silicon 上以 Metal 后端构建同一个 CLI：

```bash
cargo run --release --no-default-features --features metal,no-cuda,cli -- \
  --model-path mlx-community/Qwen3-0.6B-4bit
```

CLI 跨轮保留会话历史，行历史存在 `~/.agent-infer-history`，支持斜杠命令：

- `/help` 查看命令帮助
- `/reset` 或 `/clear` 清空当前会话
- `/tools` 查看内置工具
- `/model` 与 `/stats` 查看已加载运行时
- `/save <path>` 与 `/load <path>` 以 JSON 保存或恢复会话

---

## 架构

Workspace 划分：

- `agent-infer` — 薄二进制包装
- `cli` — REPL / CLI 流程
- `agent` — 会话状态、工具调用恢复、agent 轮次循环
- `tools` / `chat` — 工具执行助手与协议类型
- `infer` — HTTP 服务、调度器、运行时、后端实现；持有唯一的
  `InferenceEngine` 契约
- `cuda-kernels` — 抽取出来的 CUDA kernel 层（csrc/、Triton AOT、Rust
  FFI）。单向依赖：`infer → cuda-kernels`。
- `mlx-sys` — Metal 后端使用的 MLX C++ 桥接

当前的包边界见
[docs/architecture.md](docs/architecture.md)、
[docs/codebase-map.md](docs/codebase-map.md)、
[crates/README.md](crates/README.md)。

```
┌──────────────────────────────────────────────────────────┐
│  HTTP API  (/v1/completions, /v1/chat/completions, /v1/models, /v1/responses)  │
└────────────────────────┬─────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Scheduler  (decode-priority, chunked prefill)           │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────────┐  │
│  │ Prefix Cache │  │ Token KV    │  │ Block Manager  │  │
│  │ (radix tree) │  │ Pool (p=1)  │  │ (CoW paging)   │  │
│  └──────────────┘  └─────────────┘  └────────────────┘  │
└────────────────────────┬─────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│  ModelForward trait                                       │
│  Qwen3 (GQA) · Qwen3.5 (混合：循环 + 注意力)             │
└────────────────────────┬─────────────────────────────────┘
                         ▼
      FlashInfer · RMSNorm · cuBLAS GEMM · CUDA Graph
         (CUDA C + Triton AOT, crates/cuda-kernels/)
```

---

## 稳定性与支持

`agent-infer` 采用显式的稳定性分级：

- **Stable**：已文档化的 HTTP 端点（`/v1/completions`、
  `/v1/chat/completions`、`GET /v1/models`）、`GET /metrics`、
  `GET /v1/stats`，以及主要的文档化构建 / 测试工作流。
- **Beta**：`POST /v1/responses`（当前非流式子集）、CLI agent 行为、
  Metal 部署路径、GGUF 加载、基准工具。
- **Experimental**：快速演进中的量化路径、tensor-parallel 脚手架、
  未文档化的 flag 或环境变量。

项目当前状态见上方 [§当前状态一览](#-当前状态一览)，以及权威的
[docs/support-matrix.md](docs/support-matrix.md)。

治理相关：

- [docs/stability-policy.md](docs/stability-policy.md)
- [docs/compatibility.md](docs/compatibility.md)
- [docs/perf-and-correctness-gates.md](docs/perf-and-correctness-gates.md)
- [docs/release-checklist.md](docs/release-checklist.md)
- [docs/environment.md](docs/environment.md)

---

## 开发

```bash
cargo test --no-default-features --features no-cuda   # 单元测试（无 GPU）
cargo clippy --workspace -- -D warnings                # Lint
cargo fmt --all -- --check                             # 格式

# CPU 后端冒烟路径（只下载 config / tokenizer 等运行时资源，不下完整权重）
cargo run -p agent-infer --no-default-features --features cpu,no-cuda,cli -- \
  --model-path Qwen/Qwen3-0.6B --max-turns 1 --max-tokens 64

# E2E（需要 GPU + 模型权重）
INFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e

# Apple Silicon 上的 Agent CLI 真模型 E2E（会自动探测本地模型）
cargo test --release --no-default-features --features metal,no-cuda,cli -- --ignored --nocapture
```

提 PR 前请过一遍：[CONTRIBUTING.md](CONTRIBUTING.md)、
[support-matrix](docs/support-matrix.md)、
[perf-and-correctness-gates](docs/perf-and-correctness-gates.md)、
[compatibility](docs/compatibility.md)、
[environment](docs/environment.md)。发版工作：
[release-checklist](docs/release-checklist.md)。

---

## 许可

[MIT](LICENSE)
