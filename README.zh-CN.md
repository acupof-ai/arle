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
| **CUDA** | Linux + NVIDIA | **Stable** | 持续批处理、paged KV、radix 复用、TileLang BF16 attention、自定义 CUDA 量化 decode、CUDA Graph decode、Qwen3 / Qwen3.5 packed paged-prefill。**L4 / Qwen3-4B BF16 + FP8 paged KV（auto）：c=16 / 4096-in 输出 197 tok/s，peak_active=16 满槽。** |
| **Metal** | Apple Silicon | **Beta** | 调度器驱动的实时服务、chunked prefill、replay-based prefix 复用。Qwen3.5-0.8B MLX 4bit 单请求 step-driver 在 M4 Pro 20c 上达到 305.5 tok/s；GGUF Q4_K_M 在匹配的 1024/256 profile 上是 202.1 tok/s，仍作为独立 kernel/权重格式 gap 继续追。 |
| **Metal DFlash** | Apple Silicon | **Beta — 默认开启** | Qwen3 / Qwen3.5 推测解码。Qwen3-4B bf16 5.9× decode，Qwen3.5-4B-4bit 比特一致，c=1..8 已验证。 |
| **CPU** | 通用 | **仅开发用** | 冒烟测试与请求路径校验。DeepSeek V4 有慢速 Rust 参考路径用于 1B init 正确性 / HTTP smoke，不作为吞吐服务目标。 |

模型：**Qwen3 (0.6B – 72B)** 与 **Qwen3.5 系列**（包括 0.8B GGUF Q4_K_M、4B 混合线性 + 全注意力）按当前矩阵已支持 CUDA 与 Metal。**Qwen3.6 / Qwen3.5-MoE** 目前是窄 Metal Beta 路径，CUDA 仍是 stub。后续模型优先级：**DeepSeek V4（#1，V4-only substrate + CPU reference smoke 已落地）** → **Qwen 3.6（#2，规划中）**；见 [ROADMAP.md §Next-Model Priority Order](ROADMAP.md#next-model-priority-order)。当前 runtime 有意不保留 DeepSeek V2/V3/R1 支持路径。

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

- **2026-05-15** — DeepSeek V4 在 8xH20 上继续沿真实
  `/root/DeepSeek-V4-Flash` 做单 token nsys。现在 B=1 padded BF16 combine 默认
  走 `ARLE_DSV4_COMBINE_REDUCE_SCATTER=1`：expert rank 先按 origin peer 聚合
  padded route output，再用 NCCL `ReduceScatter` 直接写 owner-rank output；
  设为 `0` 可回退旧 grouped SendRecv combine。真实 HTTP smoke 保持正常中英文
  streaming 输出和算术 `410`，`decode64` 是 **12.05 post-first tok/s**；单 token
  nsys wave 从 **97.071 ms** 到 **94.923 ms**，return combine 变成
  **20.44 ms** 的 ReduceScatter，剩余 SendRecv 降到 **3.26 ms**。最新默认 DeepEP 路径让 local
  expert fallback 的 `w1`/`w3` 直接从 packed `expert_hidden` segment 读输入，
  不再先 D2D copy 到 `scratch.input`（不支持格式仍回退旧路径）。同一个
  streaming `max_tokens=2` case 输出 `霓虹`，decode-only
  `cuMemcpyDtoDAsync_v2` 从 **871 calls / 1.795 ms** 降到
  **613 calls / 1.240 ms**（每 rank range），单 token wave 是 **145 ms**。
  随后复用每层 incremental HC pre-projection / RMSNorm hidden scratch，让
  alloc/free/memset 各减少 **1,376** 次，单 token wave 进一步到
  **135 ms**。再把默认 AllGather count matrix 提前复用，同时导出 send/recv
  counts，去掉冗余 32B send-count D2H，decode-only D2H calls
  **887 → 543**，wave 到 **130 ms**。现在 B=1 padded dispatch 默认启用，并且
  在该路径跳过无用的 send-count zero/count kernel，去掉剩余 256B all-rank
  count readback 和 count AllGather，单 token wave 到 **124 ms**，
  decode-only D2H calls **543 → 344**。随后 return-side combine 在 expert rank
  先把 padded `topk` route rows 按 origin peer 聚合成一行 BF16，再回传，combine
  返回行数减少 **8×**，单 token wave 到 **112 ms**；`SendRecv` time
  **25.211 → 23.329 ms**（每 rank range）。现在 B=1 dispatch 侧也把 hidden row
  和 route metadata 合成一个 BF16 payload：3xI32 metadata 作为 raw 16-bit words
  追加到 hidden 后面一起交换，SendRecv launches **1,032 → 688**，HTTP
  `decode64` 到 **12.22 post-first tok/s**，最新单 token nsys wave 是
  **119.0 ms**，`霓彩` 和算术 `410` 都保持正确。当前主瓶颈变成 NCCL
  SendRecv/AllReduce、launch/runtime 与 allocator/memset/free churn、local-count
  D2H，以及 per-expert FP8/FP4 GEMV。最新 route-wise grouped expert 实验
  可去掉 local-count D2H，但真实 nsys 退化到 **145.7 ms**，因为 padded route
  FP4 GEMV 自身吃 **35.9 ms**（每 rank range），所以继续 default-off，主线仍是
  DeepGEMM/真正 grouped GEMM。干净 decode-only HTTP 对比也确认 pair GEMV 不能
  默认打开：默认 split expert GEMV 在 `decode64` 上是 **11.79 post-first tok/s**，
  `ARLE_DSV4_PAIR_EXPERT_GEMV=1` 只有 **7.70 tok/s**；两条路径都能正常输出序列，
  算术 case 都返回 `410`。随后把确定会被完整写入的临时 hidden buffer 从 zeroed
  allocation 切到 uninitialized allocation：HTTP `decode64` 到
  **11.99 post-first tok/s**，算术仍返回 `410`，`cuMemsetD8Async` 从
  **8,789 → 2,957 calls**，单 token wave 是 **112.7 ms**。最新 route-wise
  grouped expert 也加了 route-local `w1`/`w3` pair GEMV：
  `max_tokens=2` 仍输出 `霓彩`，单 token wave 是 **117.9 ms**，但 nsys 明确显示
  主要时间仍在 `ncclDevKernel_SendRecv`（**50.3 ms/rank-range**）、FP4 route
  pair GEMV（**19.6 ms**）、FP4 route `w2` GEMV（**10.5 ms**）、FP8 GEMV
  （**9.4 ms**），以及 allocation/free 和 launch overhead，所以这条路径继续
  opt-in，不作为默认高性能路线。
  证据：
  [`docs/trace-artifacts/2026-05-15-dsv4-deepep/`](docs/trace-artifacts/2026-05-15-dsv4-deepep/)。
- **2026-04-28** — Metal `Qwen3.5-0.8B` MLX 4bit 单请求 step-driver 在 M4 Pro 20c 上 `1024/256` 达到 **305.5 tok/s mean / 304.7 p50**，贴到 Apple-native 公开 SOTA 区间。同 profile 下 GGUF `Q4_K_M` direct 是 **202.1 tok/s**，因此 GGUF 继续作为独立 kernel/权重格式 gap 追。证据：[`docs/experience/wins/2026-04-28-bench-metal-qwen35-0p8b-mlx4bit-qknorm-default.md`](docs/experience/wins/2026-04-28-bench-metal-qwen35-0p8b-mlx4bit-qknorm-default.md)。

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
