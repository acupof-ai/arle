<p align="center">
  <strong>ARLE</strong><br>
  <em>面向长上下文 LLM 智能体的 agent reinforcement learning engine。纯 Rust 实现，CUDA 为主部署路径，训练 / 评测 / agent 工作流都在同一仓库内。</em>
</p>

<p align="center">
  <a href="https://cklxx.github.io/arle/"><img src="https://img.shields.io/badge/website-cklxx.github.io%2Farle-D97757?style=flat-square" alt="Website"></a>
  <a href="https://github.com/cklxx/arle/actions"><img src="https://github.com/cklxx/arle/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://github.com/cklxx/arle/releases"><img src="https://img.shields.io/github/v/release/cklxx/arle?include_prereleases" alt="Release"></a>
</p>

<p align="center">
  <a href="https://cklxx.github.io/arle/">官网</a> ·
  <a href="#-最新动态">动态</a> ·
  <a href="#-当前状态一览">状态</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="docs/http-api.md">API</a> ·
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

- **2026-04-23** — `arle` 这张前门现在把 `train pretrain|sft|grpo|multi-turn|eval` 与 `data download|convert` 收口到同一个顶层 Rust CLI 下，不再要求用户记住一组分散的训练二进制。入口统一记录见 [`docs/experience/wins/2026-04-23-train-cli-unified-entrypoints.md`](docs/experience/wins/2026-04-23-train-cli-unified-entrypoints.md)。
- **2026-04-22** — CUDA `Qwen3.5` 现在走真正的 packed multi-request paged-prefill 路径。全注意力层直接写 paged pool，混合线性注意力层接上 packed recurrent-state 发射，paged-prefill logits 也回到统一采样面。当前 decode truth 已收口到 [`docs/plans/2026-04-23-cuda-decode-sglang-alignment.md`](docs/plans/2026-04-23-cuda-decode-sglang-alignment.md)。
- **2026-04-20** — Metal DFlash 长提示词 prefill 修复（`fast_forward_prefill`，commit `3bc8802`），批量终止 `eval` 通过 `async_eval` 延迟（commit `d8cb2f4`）。DFlash 在 Metal 上已成为 Qwen3.5 的默认路径，并通过 guidellm 10-strategy 套件（5400-token 提示词）验证：零 `WrongPhase` 错误，100% 请求成功。规范用法见 [`docs/resources/metal-dflash.md`](docs/resources/metal-dflash.md)。

完整历史：[CHANGELOG.md](CHANGELOG.md) · 接下来：[ROADMAP.md](ROADMAP.md)

ARLE 的全称是 **agent reinforcement learning engine**：同一个 Rust
workspace 同时承载 serving、agent 执行、训练、评测，以及围绕它们的
工具链。仓库仍然是 runtime-first：`infer` 是主要交付面，`arle` 与仓内
train/eval 流程是在同一套 Rust 运行时之上扩出来的集成能力，而不是另一条
独立产品线。

落到实际入口上，就是三张 runtime-led 前门：

- `infer` 负责 OpenAI 兼容 HTTP serving
- `arle` 负责本地 agent runtime，以及 `train/*` / `data/*` 工作流
- 底下共享同一套 Rust 运行时 / 模型代码，避免 serving 和 RL 工具链各走各的

## 🚦 当前状态一览

五个维度，一个维度回答一个问题。权威的支持矩阵在
[docs/support-matrix.md](docs/support-matrix.md)；稳定性分级的定义见
[docs/stability-policy.md](docs/stability-policy.md)。

### 后端 — *在哪里运行？*

| 后端 | 平台 | 状态 | 已交付 |
|------|------|:----:|--------|
| **CUDA** | Linux + NVIDIA | **Stable** | 主部署路径。持续批处理、paged KV、radix-backed reuse、tiered-KV 回迁、FlashInfer、CUDA Graph decode，以及 Qwen3 / Qwen3.5 的 packed paged-prefill。 |
| **Metal** | Apple Silicon | **Beta** | 可用的 scheduler-backed serving、chunked prefill、replay-backed prefix reuse。batched decode 与超长上下文能力仍落后于 CUDA。 |
| **Metal DFlash** | Apple Silicon | **Beta — 默认开启** | 面向 Qwen3 / Qwen3.5 的推测解码。Qwen3-4B bf16 解码 5.9×，Qwen3.5-4B-4bit 与标量路径逐位一致，c=1..8 已验证（2026-04-20）。 |
| **CPU** | 可移植 | **仅开发使用** | 冒烟测试与请求路径验证，非部署目标。 |

### 模型 — *能加载哪些？*

| 模型 | 注意力 | CUDA | Metal |
|------|--------|:----:|:-----:|
| Qwen3 (0.6B – 72B) | GQA | ✅ | ✅ |
| Qwen3.5-4B | 混合（线性 + 全注意力） | ✅ | ✅ |
| Llama 3 / 4 | GQA | *规划中* | *规划中* |
| DeepSeek V3 / R1 | MLA | *规划中* | *规划中* |

### HTTP API — *客户端能调用什么？*

| 端点 | 状态 | 备注 |
|------|:----:|------|
| `POST /v1/completions` · `POST /v1/chat/completions` · `GET /v1/models` | **Stable** | OpenAI 兼容的核心服务面。完整契约见 [`docs/http-api.md`](docs/http-api.md)。 |
| `POST /v1/responses` | **Beta** | 文本 / tool-call 子集，支持非流式和 SSE。 |
| `GET /metrics` · `GET /v1/stats` | **Stable** | Prometheus + 人类可读运维面。 |

### Agent / Train / Eval — *ARLE 自己能跑什么？*

| 面 | 状态 | 已交付 |
|----|:----:|--------|
| `arle` 本地 agent runtime | **Beta** | 默认开启工具调用、支持会话 save/load/export、`--doctor`、模型自动发现，以及基于 KV 的多轮复用。 |
| `train pretrain|sft|grpo|multi-turn|eval` | **Beta** | 统一 CLI 前门直达仓内 Rust 训练栈，包含 exact resume、HF 风格 checkpoint 目录，以及当前主线的 Qwen3.5-family train/RL 路径。 |
| `data download|convert` + 运维 DX | **Beta** | 数据集工具、`train env`、独立 eval，以及一张统一的 Rust 前门，而不是一组临时拼起来的二进制。 |

### 量化 — *能压到多小？*

| 格式 | 状态 | 可用后端 |
|------|:----:|----------|
| FP8 / INT8 / TurboQuant KV | **Beta** | CUDA |
| GPTQ W4 · AWQ W4 | **Beta** | CUDA |
| Q4_K GGUF | **Beta** | CUDA；Metal（Qwen3.5 dense GGUF 走加载时反量化） |
| MLX 4-bit | **Beta** | Metal（`start_metal_serve.sh` 规范路径默认值） |

---

<!--
  下方为稳定参考资料：能力、安装、API、架构、开发流程。
  仅在架构或 API 级别发生变化时更新。项目的当前状态体现在上方两节。
-->

## 为什么要 ARLE（agent reinforcement learning engine）？

智能体工作流每一轮都要付 "prefill 税"：系统提示 + 历史对话 + 工具结果都要被重新处理。上下文越长，**prefill 越主导延迟**。

ARLE（agent reinforcement learning engine）把这件事当成 serving 和 agent/RL 闭环里的共同核心问题：

| 能力 | 做的事情 | 效果 |
|------|----------|------|
| **多轮 KV 复用** | slot-sticky 复用让上一轮 KV 在下一轮继续热着；CUDA 还带 radix-backed tiered-KV 路径（`T0 GPU -> T1 host pinned -> T2 本地磁盘 -> T3 集群共享后端面`）做整块复用与分级回迁。 | 当前缀可复用时，每轮只对新的用户消息做 prefill |
| **分页 KV 池** | 主要 CUDA KV 形态用 `page_size=16`，支持 GPU 直接挂页，以及共享前缀上的尾页 Copy-on-Write。 | KV 记账更稳定，整块可复用，共享前缀更便宜 |
| **透明的慢层 spill / promote** | 冷块可以从 GPU spill 到 host pinned memory 和本地磁盘，再在使用前 promote 回来。当前仓内集群共享路径仍是最小共享文件系统后端。 | 让长上下文和缓存前缀复用不再受纯 GPU 驻留约束 |
| **共享前缀 CoW** | radix 路径上的共享整块保持只读；只有活跃尾页写入时才分裂。 | 并发请求共享相同前缀时，不会把基础 KV 显存按请求数线性放大 |
| **调度重叠** | CUDA 调度器把 decode launch/readback 跨迭代重叠，fetch wait 走睡眠而不是忙轮询，并用 emit worker 做流式文本解码与 stop 扫描。 | 更好的 CPU/GPU 重叠，以及更低的调度端开销 |
| **共享运行时权威** | `infer`、`arle`，以及仓内 train/eval 任务共同复用同一套 Rust 运行时 / 模型契约。 | serving、本地 agent 工作流、以及 RL 工具链不会分裂成多套实现 |

所以 ARLE 不是“一个推理引擎旁边再摆一套训练项目”。这条推理主干本身就是
更完整 agent RL 闭环的地基：同一套 Rust 模型 / 运行时权威、同一
workspace 内的训练二进制，以及一个顶层 CLI，既能当本地 agent，也能作为
train / eval / data 的统一前门，不需要再绕一层额外的 Python 控制平面。

当前 benchmark 收口重点是高并发 CUDA 与 SGLang 的剩余差距
（`c4/c8/c16`）；下面这些带日期的数字更适合作为历史快照，而不是永不过期
的公开 headline。

---

## 性能

权威 benchmark 来源是按日期归档的
[`docs/experience/wins/`](docs/experience/wins/)，由
[`scripts/bench_guidellm.sh`](scripts/bench_guidellm.sh) 产出。

下面公开展示的数字目前仍然以 serving 侧为主，因为当前 benchmark 收口的
主战场就是 CUDA runtime credibility。与此同时，同一套运行时权威也支撑
本地 `arle` 前门，以及仓内的 train/eval 栈。

当前 CUDA benchmark 计划不是“发布一个永远有效的单个 headline 数字”，而是
继续完成对 SGLang 的高并发收口：

- `c1`：接近持平
- `c2`：还有小幅吞吐差距
- `c4/c8/c16`：仍是主要优化目标

当前执行计划：
[`docs/plans/2026-04-23-cuda-decode-sglang-alignment.md`](docs/plans/2026-04-23-cuda-decode-sglang-alignment.md)

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

两张前门都是一等公民：

### `arle` — 本地 agent / train / eval 前门

```bash
git clone https://github.com/cklxx/arle && cd arle
cargo build --release --features cli -p agent-infer --bin arle
./target/release/arle --model-path /path/to/Qwen3-4B --max-turns 10
./target/release/arle train env
./target/release/arle train eval --help
```

### `infer` — OpenAI 兼容 serving

```bash
# 当前已发布的容器镜像路径
docker run --gpus all -v /path/to/Qwen3-4B:/model \
  ghcr.io/cklxx/agent-infer:latest --model-path /model --port 8000

# 或者从源码构建服务二进制
cargo build -p infer --release
./target/release/infer --model-path /path/to/Qwen3-4B --port 8000
```

```bash
# 冒烟测试 HTTP 服务面
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

`arle` 是 ARLE 这套 workspace 面向 agent、train / eval 与 dataset 工具链的
统一前门；`infer` 是专门的 OpenAI 兼容服务二进制。

**前置要求**：CUDA 12.x、[`rust-toolchain.toml`](rust-toolchain.toml)
里固定的 Rust 工具链（当前 `1.95.0`），以及 Python 3.10+ 带
`flashinfer-python`（仅构建期使用）。`crates/kv-native-sys` 需要的
Zig `0.16.0` 会由
[`scripts/setup_zig_toolchain.sh`](scripts/setup_zig_toolchain.sh) 和
`./setup.sh` 自动引导。

如果想用仓库自带流程完成本机初始化，直接跑 [`./setup.sh`](setup.sh)。
贡献者工作流与验证要求以 [CONTRIBUTING.md](CONTRIBUTING.md) 为准。

常用仓库卫生检查命令：

```bash
make hygiene        # 公共文档 / 模板 / 本地链接防漂移检查
make pre-push       # push 前的 CI 对齐快检
make check-metal    # Apple Silicon 快速检查
./setup.sh --check  # Linux/CUDA 工作站检查
```

## 文档地图

- [README.zh-CN.md](README.zh-CN.md) — 中文公共入口：安装、CLI、架构
- [README.md](README.md) — 英文公共入口
- [docs/http-api.md](docs/http-api.md) — HTTP 路由契约与流式行为
- [docs/support-matrix.md](docs/support-matrix.md) — 后端 / 模型 / 量化支持矩阵
- [docs/stability-policy.md](docs/stability-policy.md) — 稳定性分级与兼容性姿态
- [CONTRIBUTING.md](CONTRIBUTING.md) — 贡献者初始化、验证、发版要求
- [docs/index.md](docs/index.md) — 维护者内部 PARA 索引、计划与经验记录

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
| Llama 3 / 4 | GQA | 规划中 |
| DeepSeek-V3 / R1 | MLA | 规划中 |

完整计划见 [ROADMAP.md](ROADMAP.md)。

---

## API

Serving 只是 ARLE 的一个对外面。完整的服务 API 参考现在单独放在
[docs/http-api.md](docs/http-api.md)；路由总表、流式行为、边界校验、
认证 / `X-Request-Id` 约束、以及当前缺口都以那份文档为准。

核心生成接口是 `POST /v1/completions`、
`POST /v1/chat/completions`、`POST /v1/responses` 与 `GET /v1/models`。
运行探针和运维接口是 `GET /healthz`、`GET /readyz`、`GET /metrics`、
`GET /v1/stats`。会话持久化接口位于 `/v1/sessions/{session_id}/*`。

同一套 Rust 运行时权威也被本地 `arle` CLI 与训练 / 评测流程复用，所以
HTTP 服务面不是另一层单独的 Python 控制平面。

SSE 流式现在同时支持 `/v1/completions`、
`/v1/chat/completions`、`/v1/responses`。但只要请求同时带
`stream=true` 和 `tools`，chat completions / responses 仍会拒绝，
直到服务端能发出结构化的 tool-call delta。

---

## ARLE CLI

内置 ARLE 运行时，支持工具调用：

```bash
./target/release/arle \
  --max-turns 10 --temperature 0
```

```bash
./target/release/arle train env
./target/release/arle train sft --help
./target/release/arle data convert --help
```

根 CLI 二进制位于 `cli` feature 之后；不带 `--features cli` 时不会构建
`arle`。

ARLE 这张前门背后的包边界：

- `arle` → 薄二进制包装
- `cli` → REPL 与斜杠命令
- `infer` → `server_engine::LoadedInferenceEngine` 后端加载，
  `hf_hub::resolve_model_source` 自动发现模型
- `agent` → 会话循环与工具调用恢复
- `tools` / `chat` → 共享工具定义、执行助手、协议类型

若省略 `--model-path`，CLI 会先看 `ARLE_MODEL`，再回退到旧的
`AGENT_INFER_MODEL`，然后从常用目录和本地 HuggingFace 缓存中自动探测模型。

工具：`python`（执行 Python 片段）、`shell`（执行 bash 命令）。KV 前缀
缓存在每轮会话中原地复用完整上一轮 KV，只有新用户消息（以及任何工具
结果）走 prefill（slot-sticky 匹配；跨会话复用通过 radix 树在
tiered-kv-cache M1 里落地）。
macOS 上工具执行会在 `nsjail` 不可用时自动使用 `sandbox-exec`；Linux 在
装了 `nsjail` 时继续使用。

这个顶层 CLI 也同时承载 train / eval / data 子命令，让 “agent
reinforcement learning engine” 不只是 README 上的名字，而是实际统一的
日常入口。

Apple Silicon 上以 Metal 后端构建同一个 CLI：

```bash
cargo run --release --no-default-features --features metal,no-cuda,cli -- \
  --model-path mlx-community/Qwen3-0.6B-4bit
```

CLI 跨轮保留会话历史，行历史存在 `~/.arle-history`（首次运行会迁移旧的
`~/.agent-infer-history`），支持斜杠命令：

- `/help` 查看命令帮助
- `/reset` 或 `/clear` 清空当前会话
- `/tools` 查看内置工具
- `/model` 与 `/stats` 查看已加载运行时
- `/save <path>` 与 `/load <path>` 以 JSON 保存或恢复会话

---

## 架构

ARLE 不是单个二进制，而是一整个 workspace。Workspace 划分：

- `arle` — 薄二进制包装
- `cli` — REPL / CLI 流程
- `agent` — 会话状态、工具调用恢复、agent 轮次循环
- `tools` / `chat` — 工具执行助手与协议类型
- `autograd` — 从零实现的 autograd 与优化器核心，供训练栈复用
- `train` — pretrain / SFT / GRPO / multi-turn / eval 运行时与控制面
- `infer` — HTTP 服务、调度器、运行时、后端实现；持有唯一的
  `InferenceEngine` 契约
- `cuda-kernels` — 抽取出来的 CUDA kernel 层（csrc/、Triton AOT、Rust
  FFI）。单向依赖：`infer → cuda-kernels`。
- `mlx-sys` — Metal 后端使用的 MLX C++ 桥接

当前的包边界见
[docs/architecture.md](docs/architecture.md)、
[docs/codebase-map.md](docs/codebase-map.md)、
[crates/README.md](crates/README.md)。

下图聚焦的是 `infer` serving 热路径。agent/runtime 与 train/eval 这些面并
不是套在另一层 Python 服务上的壳，而是和它一起落在同一套共享 Rust 模型 /
运行时权威上。

```
┌──────────────────────────────────────────────────────────┐
│  HTTP API  (/v1/completions, /v1/chat/completions, /v1/models, /v1/responses)  │
└────────────────────────┬─────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Scheduler  (decode-priority, chunked prefill)           │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────────┐  │
│  │ Prefix Cache │  │ Paged KV    │  │ Block Manager  │  │
│  │ (radix tree) │  │ Pool (p=16) │  │ (CoW paging)   │  │
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

`arle` 采用显式的稳定性分级：

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

日常开发循环会同时覆盖 ARLE 的两侧：`infer` 这张 serving 面，以及 `arle`
这张 agent / train / data 前门。

```bash
cargo test --no-default-features --features no-cuda   # 单元测试（无 GPU）
cargo clippy --workspace -- -D warnings                # Lint
cargo fmt --all -- --check                             # 格式

# CPU 后端冒烟路径（只下载 config / tokenizer 等运行时资源，不下完整权重）
cargo run -p agent-infer --bin arle --no-default-features --features cpu,no-cuda,cli -- \
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
