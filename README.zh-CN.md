<p align="center">
  <strong>ARLE</strong><br>
  <em>纯 Rust 实现的推理运行时，覆盖 serving、本地 agent、训练与评测。<code>infer</code> 是 OpenAI 兼容的服务二进制；<code>arle</code> 是统一前门。</em>
</p>

<p align="center">
  <a href="https://cklxx.github.io/arle/"><img src="https://img.shields.io/badge/website-cklxx.github.io%2Farle-D97757?style=flat-square" alt="Website"></a>
  <a href="https://github.com/cklxx/arle/actions/workflows/ci.yml"><img src="https://github.com/cklxx/arle/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/cklxx/arle/actions/workflows/cuda-ci.yml"><img src="https://github.com/cklxx/arle/actions/workflows/cuda-ci.yml/badge.svg" alt="CUDA CI"></a>
  <a href="https://github.com/cklxx/arle/actions/workflows/metal-ci.yml"><img src="https://github.com/cklxx/arle/actions/workflows/metal-ci.yml/badge.svg" alt="Metal CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://github.com/cklxx/arle/releases"><img src="https://img.shields.io/github/v/release/cklxx/arle?include_prereleases" alt="Release"></a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="docs/http-api.md">HTTP API</a> ·
  <a href="docs/support-matrix.md">支持矩阵</a> ·
  <a href="docs/architecture.md">架构</a> ·
  <a href="ROADMAP.md">路线图</a> ·
  <a href="CHANGELOG.md">变更日志</a> ·
  <a href="CONTRIBUTING.md">贡献指南</a>
</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

---

## 快速开始

### 1. 启动一个 OpenAI 兼容的服务

**Linux + NVIDIA — 直接拉镜像，无需编译：**

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v /path/to/Qwen3-4B:/model:ro \
  ghcr.io/cklxx/arle:latest \
  serve --backend cuda --model-path /model --port 8000
```

`:latest` 跟踪 `main`；打过 tag 的版本会发布为
`ghcr.io/cklxx/arle:vX.Y.Z`，详见
[Releases](https://github.com/cklxx/arle/releases)。

**Apple Silicon — 从源码构建（Metal 后端）：**

```bash
git clone https://github.com/cklxx/arle && cd arle
cargo build --release --no-default-features --features metal,no-cuda,cli --bin arle
./target/release/arle --doctor
./target/release/arle serve --backend metal \
  --model-path mlx-community/Qwen3-0.6B-4bit --port 8000
```

### 2. 调用它

```python
# pip install openai
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "Hello from ARLE"}],
).choices[0].message.content)
```

curl 版本见 [`examples/curl_chat.sh`](examples/curl_chat.sh)，更多示例在 [`examples/`](examples/)。

### 3. 跑本地 agent

```bash
./target/release/arle                                      # 交互式 REPL，内置工具
./target/release/arle --model-path /path/to/Qwen3-4B \
  run --prompt "总结这个仓库"                              # 一次性 prompt
./target/release/arle --doctor --json                      # 自检，机器可读输出
```

仅 CPU 的冒烟构建（无需 GPU）：

```bash
cargo build --release --no-default-features --features cpu,no-cuda,cli --bin arle
./target/release/arle --doctor
```

---

## 当前状态一览

| 后端 | 平台 | 状态 | 已交付 |
|---|---|:---:|---|
| **CUDA** | Linux + NVIDIA | **Stable** | 持续批处理、paged KV、radix 复用、FlashInfer、CUDA Graph decode、Qwen3 / Qwen3.5 packed paged-prefill。 |
| **Metal** | Apple Silicon | **Beta** | 调度器驱动的实时服务、chunked prefill、replay-based prefix 复用。 |
| **Metal DFlash** | Apple Silicon | **Beta — 默认开启** | Qwen3 / Qwen3.5 推测解码。Qwen3-4B bf16 5.9× decode，Qwen3.5-4B-4bit 比特一致，c=1..8 已验证。 |
| **CPU** | 通用 | **仅开发用** | 冒烟测试与请求路径校验，不作为服务目标。 |

模型：**Qwen3 (0.6B – 72B)** 与 **Qwen3.5-4B**（混合线性 + 全注意力）在 CUDA 与 Metal 上均已支持。Llama 3 / 4、DeepSeek V3 / R1 在路线图上 —— 见 [ROADMAP.md](ROADMAP.md)。

权威矩阵（HTTP API 等级、量化、agent / train / eval 表面）：[docs/support-matrix.md](docs/support-matrix.md)。
稳定性分级：[docs/stability-policy.md](docs/stability-policy.md)。

---

## 为什么是 ARLE

agent 与 RL 工作负载里，每一轮都要付 prefill 税：system prompt + 历史 + 工具结果都要被重新处理。上下文越长，**prefill 越主导延迟**。ARLE 把这件事当成 serving 与 agent / RL 流程的共同核心问题：

- **跨轮 KV 复用。** Slot-sticky 复用让上一轮的 KV 留在原位。CUDA 还带一条 radix 支撑的分层 KV 通路（`T0 GPU → T1 host pinned → T2 本地盘 → T3 集群共享`）做整块复用与分阶段回填，只要前缀仍可复用，每轮就只需 prefill 新的 user 消息。
- **Paged KV 池。** 主 CUDA KV 格式以 `page_size=16` 为单位，直接 GPU 页面挂载、共享前缀的尾页 CoW —— 计费可预期、整块可复用、共享前缀更便宜。
- **统一的运行时权威。** `infer`、`arle`、仓内 train / eval 共用同一套 Rust 运行时与模型契约：服务、本地 agent、RL 工具链走同一条代码路径，不再各搭一套。

架构详解：[docs/architecture.md](docs/architecture.md) · [docs/codebase-map.md](docs/codebase-map.md)。
带日期的 benchmark 快照：[docs/experience/wins/](docs/experience/wins/) · 用 [`scripts/bench_guidellm.sh`](scripts/bench_guidellm.sh) 跑自己的版本。

---

## 入口面

`arle` 是用户面对的唯一二进制：

| 命令 | 含义 |
|---|---|
| `arle`（无参） | 交互式 agent REPL，内置 `python` 与 `shell` 工具（沙箱）。 |
| `arle run --prompt "…"` / `--stdin --json` | 脚本友好的一次性 agent prompt。`--no-tools` 关闭工具执行。 |
| `arle serve --backend {cuda,metal,cpu} --model-path …` | 启动 OpenAI 兼容的 HTTP 服务。 |
| `arle train {pretrain,sft,grpo,multi-turn,eval}` | 仓内训练 / RL 工作流，与服务共用同一运行时。 |
| `arle data {download,convert}` | 数据集工具。 |
| `arle --doctor [--json] [--strict]` | 自检：后端、硬件、HF 缓存、模型解析。CI 友好。 |

REPL 在 `~/.arle-history` 持久化输入历史，支持斜杠命令：`/help`、`/reset`、`/clear`、`/tools`、`/model`、`/stats`、`/models`、`/save`、`/load`、`/export`。

只想要服务二进制的运维同学可以直接用 `infer`（Linux 用 `cargo build -p infer --release --features cuda`；Apple Silicon 用 `--features metal,no-cuda`）—— 同一份 HTTP 契约，不带 agent / train / data 表面。

---

## 📰 最新动态

<!-- 仅保留最近 2 条，更早历史见 CHANGELOG.md。 -->

- **2026-04-23** — `arle` 前门把 `train pretrain|sft|grpo|multi-turn|eval` 与 `data download|convert` 收口到同一个顶层 Rust CLI。记录：[`docs/experience/wins/2026-04-23-train-cli-unified-entrypoints.md`](docs/experience/wins/2026-04-23-train-cli-unified-entrypoints.md)。
- **2026-04-22** — CUDA `Qwen3.5` 走真正的 packed multi-request paged-prefill 路径；全注意力层直接写 paged pool，混合线性注意力层接上 packed recurrent-state 发射。规划：[`docs/plans/2026-04-23-cuda-decode-sglang-alignment.md`](docs/plans/2026-04-23-cuda-decode-sglang-alignment.md)。

完整历史：[CHANGELOG.md](CHANGELOG.md)。下一步：[ROADMAP.md](ROADMAP.md)。

---

## 文档地图

- [docs/http-api.md](docs/http-api.md) —— HTTP 路由契约、流式行为、边界保证
- [docs/support-matrix.md](docs/support-matrix.md) —— 后端 / 模型 / 量化 / API 支持等级
- [docs/stability-policy.md](docs/stability-policy.md) —— 稳定性等级与兼容性策略
- [docs/architecture.md](docs/architecture.md) —— 包边界与依赖方向
- [docs/codebase-map.md](docs/codebase-map.md) —— workspace 布局与主要执行路径
- [docs/environment.md](docs/environment.md) —— 环境变量与运行时旋钮
- [docs/troubleshooting.md](docs/troubleshooting.md) —— 常见构建 / 运行时错误与解法
- [docs/comparison.md](docs/comparison.md) —— 与 vLLM / SGLang / mistral.rs / llama.cpp 的对比
- [docs/release-checklist.md](docs/release-checklist.md) · [docs/perf-and-correctness-gates.md](docs/perf-and-correctness-gates.md)
- [CONTRIBUTING.md](CONTRIBUTING.md) —— 贡献者环境、校验、发版预期
- [SECURITY.md](SECURITY.md) —— 漏洞披露策略
- [examples/](examples/) —— 可直接复制的冒烟路径（curl、OpenAI SDK、Docker、Metal、train fixture）
- [docs/index.md](docs/index.md) —— 维护者面向的 PARA 索引、plans 与经验日志

---

## 许可证

[MIT](LICENSE)
