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
| `infer` | serving/runtime 真相：scheduler、backend、HTTP、模型 |
| `arle` | 本地 CLI 入口：agent REPL、`serve`、`train opd` |
| `train` | OPD 训练延伸（2026-05-18 pivot 后**仅 OPD**；pretrain/SFT/GRPO 已删除，见 [`projects/2026-05-18-opd-only-pivot.md`](projects/2026-05-18-opd-only-pivot.md)） |

统一契约：`infer::server_engine::InferenceEngine`（HTTP 与 agent CLI 共用）。

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
| Multi-GPU TP/PP/EP (F0–F4) | Scaffold — NCCL smoke 通过，**forward 未接线** | `infer/src/distributed/` | architecture §Multi-GPU；`distributed.rs` 模块注释 |
| CUDA speculative decode | **Not shipped** — plumbing 存在，无吞吐收益 | `infer/src/speculative/`、`scheduler/cuda/spec_path.rs` | support-matrix §4a |
| Qwen3.5 Medusa | **Blocked** — 需 recurrent-state rollback | — | support-matrix §4a；[`plans/M_medusa-phase1b-qwen35-v2-snapshot-ring-redesign.md`](plans/M_medusa-phase1b-qwen35-v2-snapshot-ring-redesign.md) |
| ForwardBatch TP/PP 元数据 | Inert — 类型槽位，无 consumer | `infer/src/scheduler/forward_batch.rs` | 文件头注释 + architecture §Multi-GPU |
| KV T3 / NIXL | Experimental / stub | `infer/src/kv_tier/` | support-matrix §4b |
| xgrammar 结构化输出 | Scaffold Phase 1 | `crates/xgrammar-sys` | support-matrix §5 |

---

## 3. 三条执行路径（从哪读起）

摘自 [`codebase-map.md`](codebase-map.md) §2，按你的任务选一条：

### Agent CLI

```text
src/main.rs → crates/cli → infer::server_engine → crates/agent
```

关键文件：`crates/cli/src/lib.rs`、`infer/src/server_engine.rs`、`crates/agent/src/lib.rs`

### CUDA Serving（主线）

```text
infer/src/main.rs → backend/cuda/bootstrap.rs → scheduler/cuda/* → model/* → ops/*
```

关键文件：`infer/src/backend/cuda/bootstrap.rs`、`infer/src/scheduler/cuda/`、`infer/src/model/qwen35.rs`

### OPD 训练

```text
crates/cli/src/train_cli.rs → train::opd → autograd
```

关键文件：`crates/cli/src/train_cli.rs`、`crates/train/src/opd.rs`（OPD step 逻辑）

---

## 4. Cargo Feature 决策表

来源：`infer/Cargo.toml` `[features]` + [`AGENTS.md`](../AGENTS.md) §Build & run。

| 目标 | 命令 | 说明 |
| --- | --- | --- |
| Linux + NVIDIA 完整构建 | `cargo build --release --features cuda --bin arle` | 需要 nvcc |
| Apple Silicon | `cargo build --release --no-default-features --features metal,no-cuda,cli --bin arle` | 无 CUDA 链接 |
| Mac 上 CUDA Rust 类型检查 | `cargo check -p infer --no-default-features --features cuda,no-cuda` | 无 GPU 也可过类型 |
| CPU smoke | `cargo build --release --no-default-features --features cpu,no-cuda,cli --bin arle` | 合成 token，非真实推理 |
| Multi-GPU NCCL smoke | `cargo build --release -p infer --features cuda,nccl --bin infer` | 需 ≥2 GPU；`infer --nccl-smoke` |
| 默认 feature | `unified_scheduler` only | **不含** cuda/metal；须显式选 backend |

`default = ["unified_scheduler"]` — 不会隐式拉 CUDA/Metal 依赖（`infer/Cargo.toml:108`）。

---

## 5. 按任务选阅读清单

| 你要改… | 先读 | 再读模块 AGENTS.md |
| --- | --- | --- |
| HTTP / OpenAI API | `infer/src/http_server/openai_v1.rs` | [`infer/src/http_server/AGENTS.md`](../infer/src/http_server/AGENTS.md) |
| CUDA scheduler / batching | `infer/src/scheduler/cuda/runtime/scheduler_loop.rs` | [`infer/src/scheduler/AGENTS.md`](../infer/src/scheduler/AGENTS.md) |
| KV tier / prefix cache | `infer/src/prefix_cache.rs`、`infer/src/kv_tier/` | [`infer/src/kv_tier/AGENTS.md`](../infer/src/kv_tier/AGENTS.md) |
| CUDA kernel | `crates/cuda-kernels/csrc/` | [`crates/cuda-kernels/AGENTS.md`](../crates/cuda-kernels/AGENTS.md) |
| Metal backend | `infer/src/backend/metal/` | [`infer/src/backend/metal/AGENTS.md`](../infer/src/backend/metal/AGENTS.md) |
| 模型 forward | `infer/src/model/qwen35.rs` | [`infer/src/model/AGENTS.md`](../infer/src/model/AGENTS.md) |
| Agent 对话循环 | `crates/agent/src/lib.rs` | — |
| OPD 训练 | `crates/train/src/opd.rs` | [`crates/autograd/AGENTS.md`](../crates/autograd/AGENTS.md) |

**大文件提示**（单文件认知负担高，读前先查 module map）：

| 文件 | 行数（2026-06-01 `wc -l`） | 说明 |
| --- | --- | --- |
| `infer/src/backend/metal/request_state.rs` | 5267 | Metal 请求状态机 |
| `infer/src/model/deepseek/weights.rs` | 7122 | DSv4 权重加载（scaffold） |
| `infer/src/backend/metal/qwen35.rs` | 4030 | Metal Qwen3.5 forward |

---

## 6. 改动验证清单（Change Impact Map）

Runtime 改动**必须**有 bench wins/errors 条目（[`AGENTS.md`](../AGENTS.md) §Benchmarks）。下表是 minimum verification：

| 改动目录 | 最低验证 | 额外（优化/架构级） |
| --- | --- | --- |
| `crates/cuda-kernels/csrc/` | `cargo test --release -p infer --features cuda` | `scripts/bench_guidellm.sh` + nsys |
| `infer/src/scheduler/cuda/` | `cargo test --release --test e2e` | guidellm sweep |
| `infer/src/model/qwen35/` | `cargo test --release --test e2e_qwen35` | guidellm |
| KV quant / dtype | `cargo test --release -p infer --features cuda --test kv_precision_parity` | guidellm + parity JSON |
| `infer/src/backend/metal/` | `cargo test --release --no-default-features --features metal --test e2e_qwen35` | Metal Qwen3.6 bench（见 AGENTS.md §Metal canonical model） |
| `crates/agent/`、`crates/cli/` | `cargo test --release -p agent -p cli -p chat` | — |
| `crates/train/` OPD | `cargo test --release -p train` | OPD smoke on CUDA GPU |
| 文档 only | — | 无需 bench；commit body 注明 `docs-only` |

Canonical bench 流程：[`bench-and-trace-spec.md`](bench-and-trace-spec.md) + `scripts/bench_guidellm.sh`。

---

## 7. 不要先读什么

以下对新人默认**不是**入门材料（维护者/agent 用）：

| 目录 | 数量级 | 说明 |
| --- | --- | --- |
| `docs/experience/wins/` | ~390+ | 历史 bench 记录；按日期/标签检索，不要通读 |
| `docs/plans/` | ~129 | 含大量 KILL/superseded plan |
| `docs/research/` | ~195 | 假设与 survey，非 shipped 真相 |
| `docs/index.md` 顶部 session 快照 | — | 维护者 session 状态，会过时 |

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
