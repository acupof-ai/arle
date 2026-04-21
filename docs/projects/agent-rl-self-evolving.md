# Project — Agent RL 训推一体 · 自进化栈

**Status**: Active · **Started**: 2026-04-18 · **Owner**: ckl
**Locked scope**: 单机 / CUDA first / LoRA-only / GRPO / 统一训推集成
**Core main line**: 是（ROADMAP Phase 6，从 2026-04-18 起成为项目主航道之一）

---

## 0. TL;DR

把 agent-infer 从"推理引擎"升级为"**单机 Rust 原生的 agent RL 训推一体栈**"。目标是把训练与推理收敛到同一套 Rust 模型权威与权重注册表下，允许通过异步边界拆分实现。当前代码已经有独立 `train` crate 的 train-side server，并通过 `pretrain --serve` / `train_sft --serve` / `train_grpo --serve` / `train_multi_turn --serve` 暴露统一的 `/v1/train/status|events|stop|save` 控制面；当前 train-side 真相是通用 Qwen-family 训练架构，以 Qwen3.5 为默认和优化主线：`pretrain` 是通用 scratch-pretrain 入口，`train_sft` 已经能在 Qwen3 / Qwen3.5 间切换，训练默认只更新 LoRA adapter，`train_sft` checkpoint 已经是 merged-model + PEFT-style adapter 双轨导出且支持 adapter-only resume，`train_grpo` 和 `train_multi_turn` 都已经支持 exact checkpoint/resume，shared async observability 已经落到 train-side event stream + bounded async sink + `dropped_metrics` control-plane status + MLflow metrics/artifacts + OTLP HTTP log export + offline-first W&B sidecar export，hybrid linear-attn train path 还没有落地。运行面上，四个 active train binaries 现在都已有显式 backend 选择并做过 CUDA 远端验证：`pretrain` / `train_sft` 走通了 save/eval/resume，`train_grpo` 走通了 fresh/eval/resume/control-plane，`train_multi_turn` 走通了 stepwise GRPO / GSPO / eval / resume / control-plane。下面描述的是要收敛到的目标态：

```
Agent tool-use rollout  →  verifier reward  →  GRPO loss  →  AdamW step on LoRA  →  热切 adapter  →  下一轮 rollout
```

**训练端从零写**（参考 [mni-ml/framework](https://github.com/mni-ml/framework)，分析见 [`docs/research/mni-ml-framework-notes.md`](../research/mni-ml-framework-notes.md)），**推理端复用** agent-infer 现有栈（FlashInfer / Triton AOT / Paged KV / Metal runtime）。这是一次"认知提升 + 产品化"双重目的的主线工作。

> **Current implementation note**
> 下文的 workspace / 数据流 / `/v1/train/*` 更多是在定义 **目标架构**。
> 2026-04-21 当前树里的训练控制面仍然在 `crates/train`：
> `pretrain --serve` / `train_sft --serve` / `train_grpo --serve` /
> `train_multi_turn --serve` 会启动 `crates/train/src/server.rs`
> 里的 train-side HTTP control plane。要回答"今天 repo 里已经有什么"，
> 先看 [`docs/codebase-map.md`](../codebase-map.md) 和
> [`docs/plans/train-runtime-architecture-v1.md`](../plans/train-runtime-architecture-v1.md)。
> 当前 train-side 训练模型线已经变成通用 Qwen-family 控制面，且以 Qwen3.5 dense/full-attn 为默认与优化主线；`pretrain` 是唯一 canonical scratch-pretrain 入口，`train_grpo` 和 `train_multi_turn` 都已经支持 exact checkpoint/resume，`train_multi_turn` 已经支持 stepwise GRPO 和 sequence-level GSPO 两种 objective，shared async observability 已经落到 train-side event stream / MLflow / OTLP / W&B sidecar。2026-04-21 的远端 CUDA 验证已经补齐到了四个 active train binaries：`pretrain`、`train_sft`、`train_grpo`、`train_multi_turn`。其中 RL 面不只是“能训练”，还覆盖了 checkpoint reload through `eval_lm`、resume 和 live `/v1/train/*` control-plane。hybrid linear-attn 仍是后续目标，不是已经完成的现状。

---

## 1. Why（为什么做、为什么现在做、为什么这样做）

### 1.1 外部驱动
- Agent RL（GRPO/DPO/RLAIF + agent tool loop）2025–2026 成为 LLM 训练事实主流，vLLM/SGLang 生态正被 **verl / slime / AReaL** 等"训推一体"框架吸纳为 rollout backend。
- 对我们：继续做纯推理会逐渐"管道化"，下游是 Python + Megatron。主动走上训推一体是**不被管道化**的唯一出路。

### 1.2 我们的差异点
- **纯 Rust，统一权重权威**：目标态里训练 worker 和推理 worker 共享同一套 Rust 模型定义、权重注册表与 adapter 协议；实现上可以同进程，也可以通过异步 worker / 进程边界拆分，只要不引入 Python 热路径和额外的 weight-sync tax。当前实现还没完全收敛到这一步，但这仍是 Phase 6 的结构性目标。
- **agent-infer 已有基建**：FlashInfer HD128/HD256、Paged KV、Radix prefix cache、chunked prefill、continuous batching、OpenAI v1 agent loop、Metal runtime。rollout 侧几乎不用新建。
- **LoRA-only 起步**：base 冻结 → **不需要穿过 FlashInfer / Marlin / 自研 GEMM 的 backward**，这是"纯 Rust 训推一体可行"的关键前提。

### 1.3 认知目标（非交付物，但是本项目的显式动机）
从零写 autograd + AdamW + GRPO，把"LLM 训练到底怎么跑"吃到骨子里。ckl 主动声明：**认知增强 > 抄现成**。本项目的代码质量门槛是"让明年的自己读起来比 PyTorch 源码更懂"。

### 1.4 统一训推权威是护城河，异步边界是实现选项（autoplan 2026-04-18 强化）

> 背景：2026-04-18 的 `/autoplan` CEO 评审中，两路外部声音（Claude subagent + Codex outside-voice）都指出 §2 行 3 的"单进程"写法把实现方式误写成了身份约束。下面把约束收敛回真正要守住的东西：统一权重权威、统一 adapter 协议、统一 trajectory 语义。

**为什么统一训推权威是护城河，而不是某个固定进程形态**：

1. **`Arc<BaseWeights>` 零拷贝共权重**：训练 worker 和推理 worker 看到的是同一套权重 authority；如果后续用异步边界把 worker 拆开，边界上也只能传 trajectory / adapter delta / control message，不能再引入第二份模型真相。
2. **LoRA 热切是协议级 swap，不是业务 RPC**：`Arc<RwLock<LoRADelta>>` double-buffer 的 hot-swap 路径长度是一次原子指针交换；如果把 worker 拆成异步边界，协议仍然必须保持这一语义，不允许把 adapter 更新变成重状态机。
3. **没有 weight-sync tax 的认知收益**：无论是同进程还是异步 worker，只要模型 authority 是统一的，trainer 与 rollout 都不需要重复理解权重 schema、版本、对齐策略和同步周期。ckl 可以把脑容量花在 GRPO 本身和 autograd 数值正确性上。
4. **不抢"第二个 veRL"的赛道**：veRL/ProRL/SFR-RL 已在做跨进程 + 多租户 + Python 生态。我们的差异化窗口是"**单机内最简洁、Rust 原生、零 Python 热路径**"；实现上允许异步边界，但不把自己挤进它们已经赢的赛道。

**因此**：§2 行 3 的状态从"单进程实现"改成"**统一训推权威 + 可异步边界的实现约束**"。任何后续提案如果要把边界拆成多 worker / 多进程，必须先证明 §6.1 中至少一条触发条件已经被 M3 的真实运行数据满足。

---

## 2. 锁定范围（v3，2026-04-18）

| 维度 | 决策 | 砍掉的选项 |
|---|---|---|
| 硬件 | 单机单卡 NVIDIA（L40S/A100/H100 任一） | 分布式、多机、TP/PP/ZeRO |
| Metal | 本地 dev 支线，M4 里程碑再做 | 和 CUDA 并行推进 |
| 进程 | 目标态是统一 Rust 训练/推理栈；当前实现是独立 `train` crate + train-side server（`pretrain` / `train_sft` / `train_grpo` / `train_multi_turn` 都可 `--serve`），后续允许同进程或异步 worker 边界，只要模型 authority 仍然唯一 | 双栈分叉、各自维护模型真相 |
| Autograd | **从零写，参考 mni-ml/framework 结构** | candle / burn / 包 PyTorch |
| Op 集 | 只实现 LoRA+GRPO 用到的 ~7 个 op | 全量 op（conv/pool/full-attention-bwd） |
| Device 抽象 | `cudarc` 直接写；Metal 用 `mlx-sys`（支线） | 多 backend 抽象层 |
| 训练范围 | **LoRA only**，base 冻结 | 全参训练（v2 再考虑） |
| RL 算法 | GRPO（无 critic） | PPO / DPO / RLAIF（v2 可选） |
| 模型 | Qwen3.5 architecture family（规模参数化，大小只是配置，不是第二套权威） | 多架构支持（v2） |
| 权重共享 | `Arc<BaseWeights>` + `RwLock<LoRADelta>` double-buffer | 跨进程 shared memory |
| Rollout | 复用 agent-infer scheduler + agent tool loop | 重写 rollout |
| Reward | 单 verifier 起步（数学 exact-match / 代码单测） | Learned reward model |
| 数据 | 在线 self-play，verifier-grounded | 离线 dataset loader（v2） |
| Python 依赖 | **零** | HF `datasets` / tokenizers 以 tokenizers crate 替代 |

---

## 3. 目标架构

### 3.1 单机统一数据流（可同进程或异步 worker）

```
┌──────────────────────────────────────────────────────────────────┐
│  Single Rust Node · CUDA                                         │
│                                                                  │
│  ┌───────────────┐   trajectory    ┌──────────────────────────┐  │
│  │  Rollout      │──(prompt, resp, │  Trainer                 │  │
│  │  Worker       │   reward, logp) │  ┌────────────────────┐  │  │
│  │               │──────────────►  │  │ autograd tape      │  │  │
│  │  agent-infer  │                 │  │  + LoRA fwd        │  │  │
│  │  scheduler    │                 │  │  + GRPO loss       │  │  │
│  │  + tool loop  │                 │  │  + AdamW step      │  │  │
│  │  + FlashInfer │                 │  └──────────┬─────────┘  │  │
│  └───────┬───────┘                 └─────────────┼────────────┘  │
│          │                                       │               │
│          │ reads                     writes      │               │
│          ▼                            ▼          ▼               │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Weight Registry                                          │    │
│  │   ├─ Arc<BaseWeights>           (frozen, GPU buffers)     │    │
│  │   └─ Arc<RwLock<LoRAAdapters>>  (double-buffer, hot-swap) │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────┐   ┌─────────────────────────┐  │
│  │  Reward Dispatcher           │   │  Curriculum / TaskGen   │  │
│  │  verifier trait, 多实现       │   │  (self-play, M3)        │  │
│  └──────────────────────────────┘   └─────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Workspace 布局（目标态）

```
crates/
├── autograd/                  ← 新增，从零写
│   ├── src/
│   │   ├── lib.rs
│   │   ├── tensor.rs          ← TensorId + TensorStore + GpuTensor (CPU+CUDA #[cfg])
│   │   ├── tape.rs            ← Tape + BackwardOp + SavedContext + dispatch
│   │   ├── ops/
│   │   │   ├── matmul.rs      ← CPU ref + cuBLAS SGEMM
│   │   │   ├── elementwise.rs ← add / mul / sub / mul_scalar
│   │   │   ├── reduce.rs      ← sum / mean
│   │   │   ├── softmax.rs     ← log_softmax (数值稳定)
│   │   │   ├── gather.rs      ← token-index gather (取 token logp)
│   │   │   └── mod.rs
│   │   ├── optim.rs           ← AdamW (CPU + CUDA)
│   │   ├── module.rs          ← Module/Parameter trait
│   │   └── loss.rs            ← cross_entropy (ref)，policy_gradient primitives
│   └── tests/
│       ├── grad_check.rs      ← 每个 op 数值 vs 解析
│       └── adamw.rs           ← AdamW 对拍 PyTorch 参考值
│
├── train/                     ← 新增，RL 训练循环
│   ├── src/
│   │   ├── lib.rs
│   │   ├── lora.rs            ← LoRA adapter (A, B) + forward hook
│   │   ├── hook.rs            ← 与 agent-infer base forward 的接合点
│   │   ├── grpo.rs            ← GRPO loss + group-normalized advantage + KL 正则
│   │   ├── rollout.rs         ← Trajectory buffer + scheduler subscription
│   │   ├── reward.rs          ← Verifier trait + 数学 / 代码单测实现
│   │   ├── curriculum.rs      ← TaskGen (M3)
│   │   ├── weight_sync.rs     ← LoRA delta double-buffer 热切
│   │   └── trainer.rs         ← 主循环（rollout ↔ train 交替）
│   └── tests/
│
├── cuda-kernels/        ← 现有（共享给推理 + 训练 fwd）
├── mlx-sys/                   ← 现有（Metal 支线）
├── agent/chat/cli/tools ← 现有
└── ...
```

**命名**：按 ckl 指示，不加 `infer-` 前缀。`autograd` 和 `train` 作为 agent-infer workspace 的 sibling crates。

### 3.3 和现有模块的边界

| 现有模块 | 改动 | 说明 |
|---|---|---|
| `infer/src/scheduler/` | 新增 trajectory emit channel | rollout 结束时，把 (prompt_tokens, response_tokens, logp_per_step) 推给 trainer；不改调度逻辑 |
| `infer/src/model/` | 新增 LoRA merge hook | `linear_with_lora(x, W_base, A, B)` 可选路径；base 分支走现有 merged-QKV / gate-up |
| `infer/src/backend/cuda/` | 暴露 `CudaDevice` | trainer 复用同一个 context，零拷贝共权重 |
| `agent/` | 新增 trajectory 采集 callback | tool loop 每一步记 action + observation，最终形成 step-wise trajectory |
| `infer/src/http_server/` | 目标态：新增 `/v1/train/*` 控制面 | start/stop/status/checkpoint（可选，M2 之后） |

**当前控制面真相**：训练侧控制面已经在 `crates/train` 内落地，并通过所有活跃训练入口的 `--serve` 暴露；当前 surface 是 `/v1/train/status|events|stop|save`，`infer/src/http_server/` 下的 `/v1/train/*` 只是后续统一入口的目标态。

---

## 4. 里程碑（含验收 + 预估）

| M | 名称 | 交付 | 验收门槛 | 预估 |
|---|---|---|---|---|
| **M0** | Autograd 起手式 | `crates/autograd` 骨架：`TensorStore` + `Tape` + `BackwardOp::{Add, Mul, MulScalar}` + Sum reduce + CPU 路径 | `y = sum((a+b)*3); y.backward()` 对 a、b 梯度手验过；CPU-only，无 GPU | 3–5 天 |
| **M1** | Autograd 核心 op 完整 | + matmul (CPU+cuBLAS) + log_softmax + gather + AdamW (CPU+CUDA) + Module trait | 玩具 2 层 MLP 在 CPU 和 CUDA 下收敛；每 op 有 grad_check 单测；AdamW 对拍 PyTorch 参考值 | 10–14 天 |
| **M2** | LoRA 合入 agent-infer | `crates/train` 骨架 + LoRA adapter + agent-infer base forward hook + 合成数据 supervised fine-tune 路径 | Qwen3.5-family model + LoRA rank=8；合成 prompt→target 数据，train loss 明显下降；推理侧热切 adapter 后输出变化可见 | 14 天 |
| **M3** | GRPO + 单 verifier 闭环 | GRPO loss + advantage + trajectory buffer + rollout↔train 交替主循环 + 数学 verifier | 小型 GSM8K-like 数据集；reward 曲线上升（≥基线 +15% pass@1 on held-out subset）；闭环稳定跑 ≥6 小时无 OOM/崩 | 21 天 |
| **M4** | Agent 自进化 MVP | Multi-turn agent tool loop 接入 + 多 verifier（数学 + 代码单测）+ 基础 curriculum（难度上移） | 连续 N 轮自生成任务，reward 非平凡（非 reward hack）上升；冒烟：Agent 能**解决自己上轮解决不了的任务** | 4–6 周 |
| **M5** | Metal 支线对齐 | autograd 在 `mlx-sys` 上提供对应 op；Mac 上能跑 M2 的最小 demo (1.5B) | Mac M4 Pro 上 LoRA forward+backward 跑通；收敛曲线形状和 CUDA 一致 | 与 M3/M4 并行 |

**总计 M0→M4 关键路径：~12–16 周**（单人集中投入）。

---

## 5. 验收原则（每个 M 都要过）

1. **`cargo test --workspace` 全绿**，包括新增 grad_check。
2. **`cargo clippy -- -D warnings` 零警告**。
3. **数值 grad-check 双 backend 对拍**：CPU (f64 参考) vs CUDA (f32)，相对误差 < 1e-3。
4. **No 半成品**（`feedback_no_half_states.md`）：每个 M 的 crate 要么完整可用，要么不 merge。
5. **Experience 条目**：每个 M 的核心 bug / 非平凡学习写进 `docs/experience/errors/` 或 `docs/experience/wins/`。
6. **对 agent-infer 现有测试零回归**：`cargo test --release` + `cargo test --release --test e2e` 都要过。

---

## 6. 风险 & 开放问题

| 风险 | 等级 | 缓解 |
|---|---|---|
| 自写 autograd 数值精度坑 | 高 | 每个 op 强制 grad_check；CPU f64 参考路径作为 oracle |
| cuBLAS row/col-major 踩坑 | 中 | M1 第一周全在 CPU 参考实现上对数值；CUDA 路径最后接入 |
| agent-infer 的 `CudaDevice` 与 autograd crate 的 context 兼容 | 中 | M0 spike 验证，用同一个 `Arc<CudaDevice>` 注入两边 |
| LoRA 热切时推理侧读到半更新权重 | 中 | double-buffer：写到 slot B，原子 `swap` 指针，读侧 Arc 不变 |
| Rollout 和 Train 在一张卡上相互饿死 | 高 | M3 里要明确交替策略（先离线：跑 N 步 rollout → 训 K 步 → 再 rollout）；online 交替作为 M4 优化项 |
| GRPO 实现错误导致 reward 上升但是 reward hack | 高 | 每个 M 保留 held-out verifier，且人工抽检 trajectory |
| Metal MLX autograd 和 CUDA autograd 实现偏离 | 中 | 同一个 `BackwardOp` enum，实现在 `ops::*` 下分 `#[cfg(feature="cuda")]` / `#[cfg(feature="metal")]`；tape 本身共用 |
| 从零写训练栈"认知得到但交付延误" | 中 | M0 + M1 严格时间盒（< 3 周）；如果 spike 阶段就深陷 autograd 坑，退路是 **M1.5 回退用 candle 做 autograd 壳**，但 ckl 2026-04-18 明确否决该退路 |

### 6.1 异步边界演进触发条件（autoplan 2026-04-18 新增）

§1.4 把"统一训推权威"上升为项目身份。下面是**唯一允许把边界从默认实现演进成更显式 worker / 进程拆分的入口**——不是"以后看看"，而是 M3 6 小时闭环跑出真实数据后，对照下面任何一条触发：

| 触发条件 | 测量方法 | 阈值 |
|---|---|---|
| **T1**：Rollout 与 Trainer 在同卡上互相饿死，简单交替策略救不回来 | M3 闭环跑 6h，记录 `nvidia-smi dmon -s u` 利用率时序，统计 `<30%` 的窗口占比 | `>20%` 时间窗口 GPU 利用率 < 30% |
| **T2**：单卡显存压力强迫 trainer 频繁 offload，offload 本身成为瓶颈 | M3 reward 上升曲线 vs 同等 LoRA rank 在 verl/SGLang 多卡基线 | 我们到达相同 reward 的墙钟时间 > verl 基线 × 2 |
| **T3**：LoRA double-buffer 热切撞上读侧并发，热切延迟 > 推理 SLO | rollout 推理侧 P99 latency 抖动包络 vs 无热切对照 | 热切引入的 P99 抖动 > 50ms 且无法用更细 lock 解决 |
| **T4**：同行（verl/ProRL/SFR-RL）在我们 M3 之前发布"Rust agent rollout backend"接入 | 季度 OSS 扫描 | 出现且活跃维护 → 重新评估"做 substrate 而不是端到端栈"的取舍 |
| **T5**：真实 paying 用户出现，且要求多租户隔离 | 商业事件 | N=1 且需求明确 → 重审，但仍可能选择"换签到位用户、不改架构" |

**未触发任何条件之前的纪律**：
- 任何 PR / 提案中出现 `KvHandleRef`、`#[async_trait] Verifier`、`schema_version` 等"为跨进程预留接口"的字样，**默认拒绝**，除非引用本节并指出哪一条触发条件已满足。
- 触发条件的"测量数据"必须真实采自 M3 跑，不接受"我感觉""推断""同行经验"。这是对 `feedback_no_speculative_interface_shaping.md` 的硬性应用。

**触发后的处理路径**（不是预先设计，但写下来避免临时慌乱）：
- T1/T3 → 工程层优化先（chunked rollout、更细粒度 LoRA lock），仍不行再考虑跨进程。
- T2 → 优先 LoRA rank ↓ / 量化 base 释放显存，再考虑分布式。
- T4 → 重新评估"端到端 vs substrate"的战略选择，可能写新项目文档而不是改本文档。
- T5 → 商业问题，单独 ROADMAP 决策，不在本项目作用域。

---

## 7. 依赖 / 不依赖

**依赖**（必须先有）：
- agent-infer 推理栈稳定（已有，CUDA/Metal 都跑通）
- `cudarc` + `cuBLAS` 可直调（已有）
- `crates/cuda-kernels` 可被训练侧借鉴（已有）

**不依赖**（明确切断）：
- Python / PyTorch / HF transformers
- NCCL / MPI / Gloo
- 任何分布式框架
- 外部 RL 框架（verl / slime / OpenRLHF）

---

## 8. 成功的样子（1 年后回看）

> "ckl 在 2026-04-18 做了一个技术决策：不借壳，从零在 Rust 里写训练栈，然后和推理收敛到统一的训推权威。一年后 agent-infer 变成了：一个 OpenAI-compat agent 服务器 + 一个单机 RL trainer，通过异步边界对接。Qwen3-4B 上一个 demo agent 在数学 + 代码两类 verifier 上自进化 24h 后，pass@1 提升了两位数。代码只有 PyTorch+verl 栈的 1/10 行，但是 ckl 读懂了每一行，autograd 出现数值问题时 15 分钟定位。"

这是本项目的**成功画面**。如果一年后我们还在和 PyTorch/Megatron 的封装打交道，那就是失败。

---

## 9. 相关文档

- **执行计划**：[`docs/plans/rust-agent-rl-single-node.md`](../plans/rust-agent-rl-single-node.md)
- **参考分析**：[`docs/research/mni-ml-framework-notes.md`](../research/mni-ml-framework-notes.md)
- **Roadmap 入口**：[`ROADMAP.md`](../../ROADMAP.md) → Phase 6
- **现状（推理侧）**：[`docs/architecture.md`](../architecture.md)、[`docs/codebase-map.md`](../codebase-map.md)
- **Metal 支线对齐**：[`docs/projects/mlx-backend-roadmap.md`](mlx-backend-roadmap.md)
