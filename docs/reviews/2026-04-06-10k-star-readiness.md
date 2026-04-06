# 2026-04-06 · 万星项目就绪度审查

## 概述

以顶级开源作者视角，从 7 个维度审查 agent-infer 是否具备成为万星项目的条件。

**结论：技术底座扎实（9/10），开源治理和生态覆盖严重不足（3/10）。**

---

## 一、技术底座 — 9/10

**强项：**
- 纯 Rust + CUDA，零 Python 运行时开销，技术选型有差异化壁垒
- Trait-driven 架构（`ModelForward`）设计精良，扩展性好
- Scheduler（连续批处理 + 分块 prefill + CUDA Graph）是 production 级别
- Unsafe 代码极少且全部有 invariant 注释，FFI 层全检 CUresult
- 性能已达 SGLang 同级（TTFT 4.6x 快，吞吐 0.92-0.99x）

**不足：**
- 仅支持 Qwen3/3.5 两个模型族 — 最大障碍，万星项目必须覆盖 Llama、DeepSeek、Gemma 等主流模型
- 无量化支持（GPTQ/AWQ/GGUF）— 社区大多跑量化模型

## 二、文档与 README — 7/10

**强项：**
- README 结构完整：性能对比表、架构图（ASCII）、Quick Start、API 文档、Benchmark
- 内部架构文档（docs/architecture.md 51KB）非常详细
- ROADMAP.md 清晰分阶段

**致命缺失：**
- 无 LICENSE 文件 — README 写了 MIT 但文件不存在，法律上等于 all rights reserved
- README 缺少 badge 行（build status, license, crates.io）
- 无 logo / 视觉标识
- Quick Start 步骤偏多，缺少一键 Docker 方案

## 三、社区基础设施 — 3/10（最薄弱）

| 缺失项 | 影响 | 优先级 |
|--------|------|--------|
| LICENSE | 无法合法使用/贡献 | P0 |
| CONTRIBUTING.md | 贡献者无入口 | P0 |
| CODE_OF_CONDUCT.md | 社区治理缺失 | P1 |
| SECURITY.md | 安全披露无渠道 | P1 |
| Issue/PR 模板 | 反馈质量低 | P1 |
| CHANGELOG.md | 无版本历史 | P1 |
| Dockerfile | 无法一键复现 | P0 |
| Logo + Badges | 无品牌识别 | P1 |

## 四、CI/CD 与发布 — 6/10

**有：** CI 覆盖 check/test/clippy/fmt/python (CPU-only) + macOS Metal check + release workflow

**缺：**
- 从未发布过任何版本（`git tag` 为空）
- 无 CHANGELOG
- 无 crates.io 发布
- 无 GPU CI（可以理解，但应注明）

## 五、用户体验（上手成本） — 5/10

万星项目标准：5 分钟内能跑起来。

当前问题：
1. 需要 CUDA 12.x + Python 3.10 + flashinfer-python + Triton（仅为构建时依赖）
2. 需要手动下载模型权重
3. 需要设置 `LD_LIBRARY_PATH`

改进方向：
- Docker 镜像（`docker run --gpus all cklxx/agent-infer --model Qwen3-4B`）
- `setup.sh` 已有但 README 未提及
- 自动下载模型（`--model Qwen/Qwen3-4B` 直接从 HuggingFace 拉取）

## 六、代码质量 — 9/10

- Commitizen 规范严格执行
- 8258 行 Rust（infer crate），规模适中，质量高
- 模块化清晰：flat layout（`ops.rs` + `ops/`），无 `mod.rs`
- Feature gate 干净（`cuda` / `no-cuda` / `metal`）
- 错误处理：`anyhow::Result` + 结构化 `ApiError`，边界验证充分
- 测试覆盖：scheduler + HTTP + sampler 有测试，但 ops 层测试偏少

## 七、竞争定位 — 6/10

| 对比项 | agent-infer | vLLM | SGLang | llama.cpp |
|--------|-------------|------|--------|-----------|
| 模型覆盖 | 1 族 | 50+ | 30+ | 100+ |
| 量化 | 无 | FP8/INT8/AWQ/GPTQ | 同 | Q4/Q5/Q8/... |
| 多 GPU | 无 | TP/PP | TP | 无 |
| 语言 | Rust | Python+C++ | Python+C++ | C/C++ |
| 上手难度 | 高 | 低 | 低 | 极低 |

核心差异化：Pure Rust（安全、单二进制）+ 极低 TTFT。但模型覆盖和量化差距是数量级的。

---

## 万星路径：分阶段行动计划

### Phase A — 合规与可用性（1-2 周）

1. 添加 LICENSE 文件（MIT/Apache-2.0）
2. CONTRIBUTING.md + CODE_OF_CONDUCT.md + SECURITY.md
3. Dockerfile（multi-stage: build + runtime，CUDA base image）
4. 第一个 Release（v0.1.0 tag + GitHub Release）
5. CHANGELOG.md
6. GitHub Issue/PR 模板
7. README badges（CI status, license, release）

### Phase B — 模型覆盖（1-2 月，决定生死）

1. Llama 3/4 — 最大用户群
2. DeepSeek-V3/R1 — 中国开发者社区热点
3. GPTQ/AWQ 量化 — 解锁消费级 GPU 用户
4. GGUF 加载 — 兼容 llama.cpp 生态

### Phase C — 社区增长（持续）

1. Logo + 品牌设计
2. 英文 Blog post（"Why we built a pure-Rust LLM engine"）
3. 中文社区推广（知乎/V2EX/微信公众号）
4. Benchmark 自动化（CI 定期跑，公开 dashboard）
5. HuggingFace 集成（`--model Qwen/Qwen3-4B` 自动下载）
6. Discord/Telegram 社区
