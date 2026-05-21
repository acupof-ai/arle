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

### 1. 安装

**Apple Silicon — Homebrew（推荐）：**

```bash
brew install cklxx/tap/arle
arle --doctor
```

**Apple Silicon 或 Linux x86_64 — 一行脚本：**

```bash
curl -fsSL https://github.com/cklxx/arle/releases/latest/download/install.sh | sh
```

脚本会从最新 GitHub Release 下载对应 tarball、校验 SHA256、解压到
`~/.local/bin`(可用 `INSTALL_DIR=...` 覆盖)。完整支持矩阵、环境变量与卸载步骤
见 [docs/install.md](docs/install.md)。

**Linux + NVIDIA — 直接拉 Docker 镜像，无需编译：**

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v /path/to/Qwen3-4B:/model:ro \
  ghcr.io/cklxx/arle:latest \
  serve --backend cuda --model-path /model --port 8000
```

`:latest` 跟踪 `main`；打过 tag 的版本会发布为
`ghcr.io/cklxx/arle:X.Y.Z`(注意：没有 `v` 前缀)。当前 v0.1.0 对应
`ghcr.io/cklxx/arle:0.1.0`。

**从源码构建**(任意后端;`cpu`、CUDA/TileLang、本地开发需要)：

```bash
git clone https://github.com/cklxx/arle && cd arle
# Apple Silicon:
cargo build --release --no-default-features --features metal,no-cuda,cli --bin arle
# Linux + NVIDIA:
cargo build --release --features cuda --bin arle
```

### 2. 启动服务

```bash
arle serve --backend metal \
  --model-path mlx-community/Qwen3-0.6B-4bit --port 8000   # Apple Silicon
arle serve --backend cuda \
  --model-path /path/to/Qwen3-4B --port 8000               # Linux + NVIDIA
```

### 3. 调用它

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

### 4. 跑本地 agent

```bash
arle                                                       # 交互式 REPL，内置工具
arle --model-path /path/to/Qwen3-4B run --prompt "总结这个仓库"           # 一次性 prompt
arle --doctor --json                                       # 自检，机器可读输出
```

仅 CPU 的冒烟构建(无需 GPU,源码构建)：

```bash
cargo build --release --no-default-features --features cpu,no-cuda,cli --bin arle
./target/release/arle --doctor
```

---

## 当前状态一览

| 后端 | 平台 | 状态 | 已交付 |
|---|---|:---:|---|
| **CUDA** | Linux + NVIDIA | **Stable** | 持续批处理、paged KV、radix 复用、TileLang BF16 attention、CUDA Graph decode。L4 / Qwen3-4B BF16 + FP8 KV：**c=16 / 4k-in 197 tok/s**。 |
| **Metal** | Apple Silicon | **Beta** | 调度器驱动的实时服务、chunked prefill、replay prefix 复用。Qwen3.6 35B-A3B 4-bit MLX HTTP serve：**M4 Pro 48GB 85.6 tok/s 解码 / TTFT 385 ms**（256/91, temp 0），与 `mlx-lm` 直跑（86.3）持平，两者均触达 273 GB/s 统一内存带宽 ~78% 上限。Qwen3.5-0.8B MLX-4bit step-driver：**M4 Pro 20c 305.5 tok/s**。 |
| **Metal DFlash** | Apple Silicon | **Beta — 默认开启** | Qwen3 / Qwen3.5 推测解码。Qwen3-4B bf16：**5.9× decode**；Qwen3.5-4B-4bit 比特一致，c=1..8。 |
| **OPD 训练（CUDA）** | Linux + NVIDIA | **Beta** | 真实 Qwen3-0.6B checkpoint + RTX 4070 Ti SUPER 上 OPD step 端到端 **0.164 s/step**（~170× 起始 naive 基线）。**对比 HuggingFace TRL `GKDTrainer` 同配置（同 checkpoint、同 prompts、同 `rollout_len=8`、同 `lr=1e-7`、同 500 步）快 2.04×**：ARLE 0.200 s/step vs TRL 0.409 s/step，held-out KL 下降 -18.5% vs -5.5%。同形态下 PyTorch CUDA 参考 83 ms → ARLE moderate **48.5 ms / 1.71× 领先**。**LoRA 模式 0.140 s/step + 仅 3.9 GB 显存峰值 —— 4 GB 消费级显卡可运行**（`r=16` 适配器在 q/v 上，2.29 M trainable params，500 步 held-out KL -36%）。`--prompts-file <jsonl>` 支持真实文本监督（commit `50ef595`，复用 checkpoint 自带的 tokenizer.json）。CPU/CUDA loss bit-equivalent（relerr ~1.3e-6）；lr=1e-7 5k 步训练上 held-out exact-overlap 由 **50 → 82.8 %**，KL/NLL 单调下降未到 plateau。 |
| **CPU** | 通用 | **仅开发用** | 冒烟测试与请求路径校验，不作为性能目标。 |

模型：**Qwen3 (0.6B – 72B)** 与 **Qwen3.5 系列**（含 0.8B GGUF Q4_K_M、4B 混合注意力）已在 CUDA + Metal 支持。**Qwen3.6 / Qwen3.5-MoE** 是窄 Metal Beta，CUDA 仍 stub。后续模型队列：**DeepSeek V4 (#1)** → **Qwen 3.6 (#2)**，见 [ROADMAP.md](ROADMAP.md#next-model-priority-order)。DeepSeek V2/V3/R1 有意不保留。

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
| `arle train opd` | **On-Policy Distillation** —— 仓内唯一训练入口（teacher 跑在 `infer`，student 用 LoRA 或全量微调，共用同一运行时）。CUDA 路径 Qwen3-0.6B **0.164 s/step**（RTX 4070 Ti SUPER）。使用手册：[`docs/projects/2026-05-21-arle-opd-cuda-usage-manual.md`](docs/projects/2026-05-21-arle-opd-cuda-usage-manual.md)。Pretrain / SFT / GRPO / multi-turn RL 于 2026-05-18 下线（[原因](docs/projects/2026-05-18-opd-only-pivot.md)）。 |
| `arle --doctor [--json] [--strict]` | 自检：后端、硬件、HF 缓存、模型解析。CI 友好。 |

REPL 在 `~/.arle-history` 持久化输入历史，支持斜杠命令：`/help`、`/reset`、`/clear`、`/tools`、`/model`、`/stats`、`/models`、`/save`、`/load`、`/export`。

只想要服务二进制的运维同学可以直接用 `infer`（Linux 用 `cargo build -p infer --release --features cuda`；Apple Silicon 用 `--features metal,no-cuda`）—— 同一份 HTTP 契约，不带 agent / train / data 表面。

---

## 📰 最新动态

<!-- 仅保留最近 2 条，更早历史见 CHANGELOG.md。 -->

- **2026-05-21** — OPD CUDA 训练栈端到端在 Qwen3-0.6B 落地。单 session 32 个 commit
  （全部 kill-or-license SOLID 闸门把守）将 OPD moderate step 推到
  **48.5 ms / RTX 4070 Ti SUPER**，比同形态的 PyTorch CUDA 参考（83 ms）**快 1.71×**；
  真实 Qwen3-0.6B step **0.164 s**（~170× 起始 naive CPU 基线）。substrate 与
  CPU bit-equivalent（relerr ~1.3e-6）。lr=1e-7 收敛验证：5000 步后 held-out
  exact-overlap **50 → 82.8 %**，KL/NLL 仍单调下降未触底。落地轴：host-mirror
  invariant 修复、AdamW in-place、rollout KV cache、RoPE/argmax device-resident、
  fused decode causal-SDPA、fused attention-prepare layout、fused grad clip。
  五条并行轴被 SOLID gate 干净 kill（forward_last_logits、merge_grad sharing、
  SDPA mask-softmax fusion、high-level CUDA Graph rollout capture、SwiGLU
  silu+multiply fusion）。证据：
  [`docs/projects/2026-05-21-opd-cuda-cycle-wrap.md`](docs/projects/2026-05-21-opd-cuda-cycle-wrap.md)、
  [`docs/projects/2026-05-21-arle-opd-cuda-usage-manual.md`](docs/projects/2026-05-21-arle-opd-cuda-usage-manual.md)、
  [`docs/experience/wins/2026-05-21-arle-cuda-opd-realckpt-convergence-5k-newmetric.md`](docs/experience/wins/2026-05-21-arle-cuda-opd-realckpt-convergence-5k-newmetric.md)。
- **2026-05-15** — DSv4 DeepEP decode 默认走 B=1 padded BF16 reduce-scatter
  combine、fused local-expert prepare kernel、广泛 scratch 复用清理。8xH20 真
  机 `DeepSeek-V4-Flash`：`decode64` 保持 **12.05 post-first tok/s**；单 token
  nsys wave **105.2 → 87.7 ms**，`cuMemsetD8Async` calls **3,640 → 544**，算
  术 (`410`/`406`) 全正确。剩余栈：NCCL SendRecv/AllReduce、FP8/FP4 expert GEMV
  （等真正 grouped GEMM/DeepGEMM）、launch churn、D2H route-count readback。
  证据：[`docs/trace-artifacts/2026-05-15-dsv4-deepep/`](docs/trace-artifacts/2026-05-15-dsv4-deepep/),
  [`docs/experience/errors/2026-05-14-dsv4-decode-nccl-bottleneck.md`](docs/experience/errors/2026-05-14-dsv4-decode-nccl-bottleneck.md)。

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
