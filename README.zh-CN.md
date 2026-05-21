<p align="center">
  <strong>ARLE</strong><br>
  <em>Pure-Rust 运行时,统一服务、本地 agent、On-Policy Distillation 训练与评测。<code>infer</code> 是 OpenAI 兼容的服务二进制;<code>arle</code> 是统一的用户入口。</em>
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
  <a href="CHANGELOG.md">变更日志</a>
</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

---

## 快速开始

```bash
# Apple Silicon — Homebrew
brew install cklxx/tap/arle

# Apple Silicon 或 Linux x86_64 — 一行安装
curl -fsSL https://github.com/cklxx/arle/releases/latest/download/install.sh | sh

# Linux + NVIDIA — Docker,无需编译
docker run --rm --gpus all -p 8000:8000 -v /path/to/Qwen3.5-4B:/model:ro \
  ghcr.io/cklxx/arle:latest serve --backend cuda --model-path /model

# 源码构建(任意后端)
cargo build --release --features cuda --bin arle     # Linux + NVIDIA
cargo build --release --no-default-features --features metal,no-cuda,cli --bin arle  # Apple Silicon
```

完整安装矩阵与卸载:[docs/install.md](docs/install.md)。

**启动服务:**

```bash
arle serve --backend cuda  --model-path /path/to/Qwen3.5-4B --port 8000
arle serve --backend metal --model-path mlx-community/Qwen3.5-0.8B-MLX-4bit --port 8000
```

**调用(OpenAI 兼容):**

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[{"role": "user", "content": "你好,ARLE"}],
).choices[0].message.content)
```

**本地 agent / 自检:**

```bash
arle                              # 交互式 REPL,内置 python/shell 工具
arle run --prompt "总结这个仓库" --model-path /path/to/Qwen3.5-4B
arle --doctor --json              # CI 友好自检
```

更多即用样例:[`examples/`](examples/)。

---

## 当前状态一览

| 后端 | 平台 | 状态 | 关键数字 |
|---|---|:---:|---|
| **CUDA** | Linux + NVIDIA | **Stable** | 持续批处理、paged KV、radix 复用、TileLang BF16 attention、CUDA Graph decode。L4 / Qwen3.5-4B BF16 + FP8 KV:**c=16 / 4k-in 197 tok/s**。 |
| **Metal** | Apple Silicon | **Beta** | 调度器驱动服务、chunked prefill、replay prefix 复用。Qwen3.6 35B-A3B 4-bit MLX:**M4 Pro 48GB 85.6 tok/s 解码 / TTFT 385 ms**。 |
| **Metal DFlash** | Apple Silicon | **Beta — 默认开启** | Qwen3.5 推测解码。Qwen3.5-4B-4bit 比特一致,c=1..8。 |
| **OPD 训练(CUDA)** | Linux + NVIDIA | **Beta** | **Qwen3.5-9B-TQ4 → 0.8B LoRA 已能在 16 GB 显卡上跑通**:100 步真实文本 run、**4.30 s/step**、**15.9 GiB 峰值**、held-out KL 单调下降。Qwen3-0.6B 同配置仍 **比 HuggingFace TRL `GKDTrainer` 快 2.04×**。详见 [最新动态](#最新动态)。 |
| **CPU** | 通用 | **仅开发用** | 冒烟测试,不作为性能目标。 |

模型:**Qwen3.5 全家族**(0.8B / 4B / 30B-A3B / 35B)在 CUDA + Metal 上支持。后续模型队列:**DeepSeek V4 (#1)** → **Qwen 3.6 (#2)** —— 见 [ROADMAP.md](ROADMAP.md#next-model-priority-order)。

权威支持矩阵:[docs/support-matrix.md](docs/support-matrix.md) · [docs/stability-policy.md](docs/stability-policy.md)。

---

## 为什么是 ARLE

agent 与 RL 工作负载每轮都要付 **prefill 税**:system prompt + 历史 + 工具结果重复处理。ARLE 把这件事当成 serving 与训练的共同核心问题:

- **跨轮 KV 复用。** Slot-sticky 复用 + radix 支撑的分层 KV(`T0 GPU → T1 host → T2 盘 → T3 集群`)保持上一轮 KV 热。
- **Paged KV 池。** `page_size=16`,直接 GPU 页面挂载 + 共享前缀的尾页 CoW —— 计费可预期、共享前缀更便宜。
- **统一的运行时权威。** `infer`、`arle`、OPD 训练共用同一套 Rust 运行时与模型契约 —— OPD teacher 就是生产服务用的同一个 runtime,不再分两套栈。

架构详解:[docs/architecture.md](docs/architecture.md) · [docs/codebase-map.md](docs/codebase-map.md)。

---

## 入口面

`arle` 是用户面对的唯一二进制:

| 命令 | 含义 |
|---|---|
| `arle`(无参) | 交互式 agent REPL,内置 `python` 与 `shell` 工具。 |
| `arle run --prompt "…"` | 脚本友好的一次性 prompt。`--no-tools` 关闭工具。 |
| `arle serve --backend …` | OpenAI 兼容 HTTP 服务。 |
| `arle train opd` | **On-Policy Distillation** —— teacher 跑 `infer`,student 跑 `train`,共享 runtime。[使用手册](docs/projects/2026-05-21-arle-opd-cuda-usage-manual.md)。 |
| `arle --doctor [--json]` | 后端 / 硬件 / 模型解析自检。 |

只要服务二进制的运维同学可直接用 `infer`(`cargo build -p infer --release --features cuda`)—— 同一份 HTTP 契约,不带 agent / train 表面。

---

## 最新动态

<!-- 最近 1-2 条,更早历史见 CHANGELOG.md。 -->

**2026-05-21 — ARLE OPD CUDA:16 GB 消费级显卡跑 9B teacher。**
Qwen3.5-9B TurboQuant teacher 在 `infer`,Qwen3.5-0.8B-Base LoRA r=16 student 在 `train`,真实文本 prompts、`rollout_len=4`、`lr=1e-5`、RTX 4070 Ti SUPER。

![ARLE OPD CUDA — 已 license 的 OPD distillation runs](docs/projects/img/2026-05-21-arle-vs-pytorch-opd-comparison.png)

| Run | Step 时间 | 显存峰值 | Held-out KL |
|---|---:|---:|---:|
| **Qwen3.5-9B-TQ4 → 0.8B LoRA r=16** | **4.30 s** | **15.9 GiB** | **100 步 -1.02 %** |
| Qwen3.5-4B BF16 → 0.8B LoRA r=16 | 5.66 s | 14.8 GiB | 200 步 -2.05 % |
| Qwen3-0.6B LoRA r=16 | 0.140 s | 3.93 GiB | 500 步 -36.4 % |
| TRL `GKDTrainer` 同 Qwen3-0.6B | 0.408 s | 12.6 GiB | 500 步 -5.5 % |

**跨 runtime 大 teacher 路径已端到端验证。** OPD teacher 现在就是生产 `infer` runtime。9B-TQ4 run 保留已验证的 dense BF16 lm head,通过 `rollout_len=4` 在 16 GB 显卡上跑通,并在 `0/25/50/100` 每个 eval 点 held-out KL 单调下降。Qwen3.5-4B BF16 保留为较小规模参考。

端到端收敛:lr=1e-7、5000 步,held-out exact-overlap **50% → 82.8%**。

证据:[9B-TQ4 headline bench](docs/experience/wins/2026-05-21-arle-cuda-opd-9b-tq4-rollout4.md) · [使用手册](docs/projects/2026-05-21-arle-opd-cuda-usage-manual.md) · [4B→0.8B 跨 runtime bench](docs/experience/wins/2026-05-21-qwen35-4b-08b-opd-infer-teacher.md) · [TRL 对照](docs/experience/wins/2026-05-21-arle-vs-trl-gkd-head-to-head.md)。

完整历史:[CHANGELOG.md](CHANGELOG.md)。

---

## 文档地图

- [docs/http-api.md](docs/http-api.md) · HTTP 契约与流式行为
- [docs/support-matrix.md](docs/support-matrix.md) · 后端 / 模型 / 量化等级
- [docs/architecture.md](docs/architecture.md) · 包边界与依赖方向
- [docs/codebase-map.md](docs/codebase-map.md) · workspace 布局与执行路径
- [docs/environment.md](docs/environment.md) · 环境变量与运行时旋钮
- [docs/troubleshooting.md](docs/troubleshooting.md) · 常见构建 / 运行错误
- [docs/comparison.md](docs/comparison.md) · vs vLLM / SGLang / mistral.rs / llama.cpp
- [CONTRIBUTING.md](CONTRIBUTING.md) · 贡献者环境与验证
- [examples/](examples/) · 即用样例
- [docs/index.md](docs/index.md) · 维护者 PARA 索引

---

## 许可证

[MIT](LICENSE)
