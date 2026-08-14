# ARLE 新人 Onboarding

> **证据原则**：本文只汇总 canonical doc 与仓库内可验证事实；状态以
> [`support-matrix.md`](support-matrix.md) 为准，拓扑以
> [`codebase-map.md`](codebase-map.md) 为准，边界以
> [`architecture.md`](architecture.md) 为准。若本文与上述文档冲突，以 canonical 为准。

目标：30 分钟内知道「现在跑什么、代码从哪读、改完怎么验」。

---

## 1. 项目是什么

`ARLE` 是纯 Rust 推理 runtime，统一三条 surface（证据：[`architecture.md`](architecture.md) §Project framing）：

| 组件 | 职责 |
| --- | --- |
| `crates/infer-*`（plan/seam/core/cuda/metal/server/api/topo/moe/util） | serving/runtime 真相：device-neutral IR、host-only seam、Engine/scheduler、backend executor、HTTP、模型 |
| `arle`（`src/main.rs` → `crates/cli`） | 本地 CLI 入口：agent REPL、`arle serve`、`train opd` |
| `train` | OPD 训练延伸（2026-05-18 pivot 后**仅 OPD**；pretrain/SFT/GRPO 已删除） |

统一契约：`infer-api`（`crates/infer-api/src/serve_engine.rs`、`LoadedInferenceEngine`），HTTP 与 agent CLI 共用；后端通过 `infer-seam` 的 `BackendExecutor` + `KvPool` 插入同一个 `infer-core` Engine。

---

## 2. 当前支持状态（Production vs Beta vs Scaffold）

**不要凭「能编译」推断「能上线」。** 完整矩阵见 [`support-matrix.md`](support-matrix.md)。

### 2.1 后端

| 后端 | 状态 | 证据来源 |
| --- | --- | --- |
| CUDA | **Supported** — 主线 | support-matrix §1 |
| Metal | **Beta** — 本地验证可用；batched-decode 与 CUDA 仍有差距 | support-matrix §1 |
| CPU (`no-cuda`) | **Development** — smoke，非生产目标 | support-matrix §1 |

### 2.2 模型

| 模型 | 状态 | 证据来源 |
| --- | --- | --- |
| Qwen3.5 | **Supported** | support-matrix §3 |
| Qwen3.6 / Qwen3.5-MoE | **Beta (Metal)**；CUDA 为 stub | support-matrix §3 |
| DeepSeek V4 | **In progress** — CPU reference smoke；CUDA 优化 kernel 未完成 | support-matrix §3 |

### 2.3 明确是 Scaffold、不能当 Production 用的

| 能力 | 状态 | 代码位置 | 证据 |
| --- | --- | --- | --- |
| Multi-GPU TP/EP | DSv4 路径已接线（TP=8/EP=8）；通用 Qwen TP/PP 仍 staged | `crates/infer-topo`（sharding）、`crates/infer-cuda/src/{tp,deepep,dsv4}.rs` | architecture §Multi-GPU；support-matrix §0 |
| CUDA speculative decode | **Not shipped** — 未移植到 rewrite（原 legacy `infer/`-only） | 未移植到 rewrite stack | support-matrix §0、§4a |
| Qwen3.5 Medusa | **Blocked** — 需 recurrent-state rollback | — | support-matrix §4a |
| 分层 KV T1–T3 / NIXL | 未移植到 rewrite stack（原 legacy `infer/`-only） | `crates/kv-native-sys`（持久化 substrate） | support-matrix §0、§4b |
| xgrammar 结构化输出 | Scaffold Phase 1 | `crates/xgrammar-sys` | support-matrix §5 |

---

## 3. 三条执行路径（从哪读起）

摘自 [`codebase-map.md`](codebase-map.md) §2，按你的任务选一条：

### Agent CLI

```text
src/main.rs → crates/cli → infer-api (LoadedInferenceEngine) → crates/agent
```

关键文件：`crates/cli/src/lib.rs`、`crates/infer-api/src/serve_engine.rs`、`crates/agent/src/lib.rs`

### CUDA Serving（主线）

```text
infer-api → infer-core (Engine/scheduler) → infer-seam (BackendExecutor) → infer-cuda (executor) → cuda-kernels
```

关键文件：`crates/infer-cuda/src/executor.rs`、`crates/infer-cuda/src/loader.rs`、`crates/infer-cuda/src/qwen35.rs`

### OPD 训练

```text
crates/cli/src/train_cli.rs → train::opd → autograd
```

关键文件：`crates/cli/src/train_cli.rs`、`crates/train/src/opd.rs`（OPD step 逻辑）

---

## 4. Cargo Feature 决策表

来源：根 `Cargo.toml` `[features]` + [`AGENTS.md`](../AGENTS.md) §Build & run。

| 目标 | 命令 | 说明 |
| --- | --- | --- |
| Linux + NVIDIA 完整构建 | `cargo build --release --features cuda --bin arle` | 需要 nvcc |
| Apple Silicon | `cargo build --release --no-default-features --features metal,no-cuda,cli --bin arle` | 无 CUDA 链接 |
| Mac 上 CUDA Rust 类型检查 | `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --no-default-features --features cuda,no-cuda` | 无 GPU 也可过类型 |
| CPU smoke | `cargo build --release --no-default-features --features cpu,no-cuda,cli --bin arle` | 合成 token，非真实推理 |
| Multi-GPU NCCL | `cargo build --release --features cuda,nccl --bin arle` | 需 ≥2 GPU；`nccl` feature 经 cli → infer-api → infer-cuda → cuda-kernels 透传 |
| 默认 feature | `cli` only | **不含** cuda/metal；须显式选 backend |

`default = ["cli"]` — 不会隐式拉 CUDA/Metal 依赖（根 `Cargo.toml`）。

> **Tip:** 切换 feature lane 时用不同的 `CARGO_TARGET_DIR` 避免全量重编：
> `CARGO_TARGET_DIR=target-metal cargo build --release --no-default-features --features metal,no-cuda,cli`

---

## 5. 按任务选阅读清单

| 你要改… | 先读 | 模块入口 crate |
| --- | --- | --- |
| HTTP / OpenAI API | `crates/infer-server/src/coordinator.rs`、`schema.rs` | `infer-server` |
| 连续批调度 / batching | `crates/infer-core/src/planner.rs`、`lib.rs`（Engine） | `infer-core` |
| KV pool / prefix cache | `crates/infer-core/src/prefix.rs`、`radix.rs`；seam 契约 `crates/infer-seam/src/kv.rs` | `infer-core` + `infer-seam` |
| CUDA kernel | `crates/cuda-kernels/csrc/` | [`crates/cuda-kernels/AGENTS.md`](../crates/cuda-kernels/AGENTS.md) |
| Metal backend | `crates/infer-metal/src/executor.rs`、`qwen35.rs`、`kv_pool.rs` | `infer-metal` |
| 模型 forward | `crates/infer-cuda/src/qwen35.rs`（CUDA）、`crates/infer-metal/src/qwen35.rs`（Metal） | `infer-cuda` / `infer-metal` |
| Agent 对话循环 | `crates/agent/src/lib.rs` | `agent` |
| OPD 训练 | `crates/train/src/opd.rs` | [`crates/autograd/AGENTS.md`](../crates/autograd/AGENTS.md) |

---

## 6. 改动验证清单（Change Impact Map）

Runtime 改动**必须**有 bench wins/errors 条目（[`AGENTS.md`](../AGENTS.md) §Benchmarks）。下表是 minimum verification：

| 改动目录 | 最低验证 | 额外（优化/架构级） |
| --- | --- | --- |
| `crates/cuda-kernels/csrc/` | `cargo test --release -p infer-cuda --features cuda` | `scripts/bench_throughput.py` + nsys |
| `crates/infer-core/`（Engine/调度） | `cargo test --release -p infer-core` | native fixed-concurrency benchmark |
| `crates/infer-cuda/`（model/qwen35） | `cargo test --release -p infer-cuda --features cuda` | native fixed-concurrency benchmark |
| KV quant / dtype | Seam-level `--kv-cache-dtype` dispatch (BF16 default; INT8/FP8 LICENSED correctness-gated, opt-in). See [wins #68](experience/wins/2026-06-12-cuda-quant-kv-dispatch-int8-fp8.md) | native benchmark + needle gate |
| `crates/infer-metal/` | `cargo test --release -p infer-metal --no-default-features --features metal,no-cuda` | Metal Qwen3.6 bench（见 AGENTS.md §Metal canonical model） |
| `crates/agent/`、`crates/cli/` | `cargo test --release -p agent -p cli -p chat` | — |
| `crates/train/` OPD | `cargo test --release -p train` | OPD smoke on CUDA GPU |
| 文档 only | — | 无需 bench；commit body 注明 `docs-only` |

Canonical bench 流程：[`bench-and-trace-spec.md`](bench-and-trace-spec.md) + `scripts/bench_throughput.py`。

---

## 7. 不要先读什么

以下对新人默认**不是**入门材料（维护者/agent 用）：

| 目录 | 数量级 | 说明 |
| --- | --- | --- |
| `docs/experience/wins/` | ~390+ | 历史 bench 记录；按日期/标签检索，不要通读 |

需要历史 context 时，从 canonical doc 里的链接跳转，而非从 wins 目录随机读。

---

## 8. 下一步深入

| 主题 | Canonical doc |
| --- | --- |
| 包边界与依赖方向 | [`architecture.md`](architecture.md) |
| 后端能力对照 | [`architecture.md` §Backend Parity Matrix](architecture.md#backend-parity-matrix) |
| 完整文件地图 | [`codebase-map.md`](codebase-map.md) |
| 支持/量化/API 状态 | [`support-matrix.md`](support-matrix.md) |
| 贡献者操作契约 | [`AGENTS.md`](../AGENTS.md) |
| 环境变量 | [`environment.md`](environment.md) |
