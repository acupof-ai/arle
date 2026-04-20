# Plan — 单机 Rust Agent RL 训推一体（M0–M5）

**Status**: Active · **Opened**: 2026-04-18 · **Project**: [agent-rl-self-evolving.md](../projects/agent-rl-self-evolving.md)
**Scope lock**: 单机 / CUDA first / LoRA-only / GRPO / 统一训推集成（当前实现是独立 `train` crate + train-side server；`train_multi_turn --serve` 是当前控制面真相） / 训练端从零写 / sole train-side model line is the Qwen3.5-family dense/full-attn path with HF-style checkpoint dirs; handwritten Transformer/TinyLM runtime compat deleted; hybrid linear-attn path not landed yet

---

## 0. 阅读顺序

1. 本文件（任务 + 验收门槛 + 每日颗粒度）
2. [`docs/projects/agent-rl-self-evolving.md`](../projects/agent-rl-self-evolving.md) — 架构 + Why + 风险
3. [`docs/research/mni-ml-framework-notes.md`](../research/mni-ml-framework-notes.md) — 参考分析 + 裁剪清单

> **Current implementation note**
> 本计划主要描述 Phase 6 的执行路径和目标收敛方向。
> 2026-04-20 当前 repo 里的训练控制面仍然是 `crates/train`
> 自己的 train-side server：`train_multi_turn --serve` →
> `crates/train/src/server.rs`。如果问题是当前 runtime / checkpoint /
> metrics / server 的真实边界，先看
> [`train-runtime-architecture-v1.md`](train-runtime-architecture-v1.md)
> 和 [`docs/codebase-map.md`](../codebase-map.md)。
> 当前 train-side 训练模型实现已经是 Qwen3.5-family dense/full-attn 路径；HF-style checkpoint 目录已在用。手写 Transformer/TinyLM runtime compatibility 路径已经删除，不再是并列主线；hybrid linear-attn 还没落地。

---

## 1. 前置工作（开工前一次性做完）

### 1.1 确认硬件可用性

- 本地 Mac M4 Pro（开发、CPU 路径 grad-check）
- CUDA 机器（L40S / A100 / H100 任一；远程或本地）
- `nvidia-smi` 跑通，`cargo test --release` 现有 CUDA 测试全绿

### 1.2 Clone 参考仓库

```bash
git clone https://github.com/mni-ml/framework /tmp/mni-ml-framework
```

- **纯只读**参考，不 vendor。
- 读取顺序：`src/native/src/autograd.rs` → `tensor.rs` → `ops/matmul.rs` → `ops/elementwise.rs` → `ops/reduce.rs` → `ops/norm.rs` → `ops/optimizer.rs`。
- 每读一个文件，在 `docs/research/mni-ml-framework-notes.md` 对应节里补笔记（本次已完成初版）。

### 1.3 Feature flag 设计（workspace 级）

- `autograd` crate 的 features：`cpu`（默认，f32 CPU 参考实现）、`cuda`、`metal`
- `train` crate 的 features：继承 `autograd` + `rollout` flag 控制是否拉 agent-infer scheduler
- 与现有 agent-infer features 不冲突：现有 `cuda` / `metal` / `no-cuda` 不变

---

## 2. M0 — Autograd 起手式（3–5 天）

### 2.1 Scope

最小可运行 autograd：能跑 `y = (a+b)*3; y.sum().backward()` 并算出对 a、b 的梯度。CPU only，不触 GPU。

### 2.2 任务分解

| # | 任务 | 文件 | 验收 |
|---|---|---|---|
| M0.1 | 起 `crates/autograd/` 骨架 + `Cargo.toml` + 加到 workspace | `crates/autograd/Cargo.toml` | `cargo build -p autograd` 绿 |
| M0.2 | `TensorId = usize`, `GpuTensor` (CPU-only fields), `TensorStore` (slot + free_ids) | `crates/autograd/src/tensor.rs` | 单测：alloc/free/shape/to_host |
| M0.3 | `SavedContext` enum (Add/Mul/MulScalar/Sum 四个变体) + `BackwardOp` enum + `TapeEntry` + `Tape` 结构 | `crates/autograd/src/tape.rs` | 结构体编译通过 |
| M0.4 | `Tape::backward()`：DFS relevant set + 后序拓扑 + grad 累加 HashMap | `crates/autograd/src/tape.rs` | 单测：空 tape backward 不崩 |
| M0.5 | `ops::elementwise::add` (fwd+bwd), `mul_scalar` (fwd+bwd), `mul` (fwd+bwd) | `crates/autograd/src/ops/elementwise.rs` | 每个 op 有 grad_check 单测 |
| M0.6 | `ops::reduce::sum` (fwd+bwd) | `crates/autograd/src/ops/reduce.rs` | grad_check 单测 |
| M0.7 | 集成测试：`y = ((a+b)*3).sum(); y.backward();` 验证 `a.grad == 3.0` | `crates/autograd/tests/m0_toy.rs` | 测试绿 |

### 2.3 验收门槛

- ✅ `cargo test -p autograd` 全绿
- ✅ `cargo clippy -p autograd -- -D warnings` 零警告
- ✅ grad_check 工具（`tests/helpers.rs`）可复用：`fn num_grad<F: Fn(&[f32]) -> f32>(f: F, x: &[f32], eps: f32) -> Vec<f32>`
- ✅ 每个 op 在数值梯度 vs 解析梯度上 `max_abs_err < 1e-4`（f32 CPU）

### 2.4 认知产出

M0 结束时 ckl 应该能用一句话说清：
- TensorId 为什么是 usize 而不是 `Arc<Tensor>`
- SavedContext 为什么是 enum 而不是 `Box<dyn Any>`
- backward 的拓扑序为什么能用 DFS 后序得到

如果说不清，**不要进 M1**，回去重读 `autograd.rs` 和 `tape.rs`。

---

## 3. M1 — Autograd 核心 op 完整（10–14 天）

### 3.1 Scope

完成 LoRA+GRPO 所需的全部 op，CPU + CUDA 双路径，AdamW 可用。玩具 MLP 能在 CPU 和 CUDA 上收敛。

### 3.2 任务分解

| # | 任务 | 文件 | 验收 |
|---|---|---|---|
| M1.1 | CPU naive matmul (fwd+bwd)，支持 batched 广播 | `ops/matmul.rs` | grad_check ≤ 1e-4，支持 `[B, M, K] @ [B, K, N]` |
| M1.2 | CUDA matmul via `cudarc::cublas::safe::Gemm` (fwd+bwd) | `ops/matmul.rs` | CPU/CUDA 结果相对误差 ≤ 1e-3 |
| M1.3 | `log_softmax` (fwd+bwd)，数值稳定 | `ops/softmax.rs` | grad_check；和 PyTorch `F.log_softmax` 结果比对 ≤ 1e-4 |
| M1.4 | `gather` (fwd+bwd)：按 `indices: [B, S]` 从 `[B, S, V]` 取出 token logp → `[B, S]` | `ops/gather.rs` | grad_check |
| M1.5 | `mean` (fwd+bwd) | `ops/reduce.rs` | grad_check |
| M1.6 | `Module` / `Parameter` trait（不用 proc macro，保持简洁） | `module.rs` | `Linear::new(in, out)` + `forward(x)` + `parameters()` 可用 |
| M1.7 | `AdamW`（CPU）| `optim.rs` | 对拍 PyTorch `torch.optim.AdamW` 10 步参数值，相对误差 ≤ 1e-4 |
| M1.8 | `AdamW`（CUDA fused kernel） | `optim.rs` | CPU/CUDA 对拍 10 步 ≤ 1e-3 |
| M1.9 | **集成**：玩具 2 层 MLP 分类 (MNIST-like 合成数据) | `tests/m1_mlp.rs` | 100 步 CPU loss 从 2.3 降到 < 0.5；CUDA 收敛曲线形状一致 |

### 3.3 验收门槛

- ✅ `cargo test -p autograd --features cuda` CUDA 机器全绿
- ✅ `cargo test -p autograd --no-default-features --features cpu` Mac 全绿
- ✅ 所有 op 有 grad_check 单测
- ✅ AdamW 对拍 PyTorch 参考值（保存在 `tests/fixtures/adamw_reference.json`）
- ✅ 玩具 MLP CPU 100 步从 loss 2.3 → < 0.5

### 3.4 避坑要点

- **cuBLAS 的 row-major 陷阱**：`cudarc::cublas` 默认 col-major，传 row-major 数据必须 transpose flag + 交换 operand 顺序。参考 mni-ml `ops/matmul.rs` CUDA 路径的现成配置。
- **log_softmax 数值稳定**：减去行 max 再 log-sum-exp，不要先 exp 再 log。
- **grad 累加**：多次 backward 同一 tensor 应该**累加**，不是覆盖。Store 里 `accumulate_grad` 逻辑要测多路径图。

---

## 4. M2 — LoRA 合入 agent-infer（14 天）

### 4.1 Scope

`crates/train` 启动，LoRA adapter 定义清楚，和 agent-infer 的 base forward 成功拼在一起，能跑 supervised fine-tune（离 RL 还有一步）。

### 4.2 任务分解

| # | 任务 | 文件 | 验收 |
|---|---|---|---|
| M2.1 | 起 `crates/train/` 骨架 | `crates/train/Cargo.toml` | workspace build 绿 |
| M2.2 | `LoRAAdapter { A: Parameter, B: Parameter, rank, alpha }`，`forward(x) -> B @ (A @ x) * scale` | `train/src/lora.rs` | 单测：rank=8，alpha=16，forward shape 正确，grad 流通 |
| M2.3 | **Hook 设计**：在 `infer/src/ops/linear.rs` 加 `linear_with_lora` 可选路径，base 侧 `W @ x` 走 agent-infer 现有 kernel，LoRA 分支独立 cuBLAS 小 GEMM，结果相加 | `infer/src/ops/linear.rs` + `train/src/hook.rs` | 单测：frozen base + LoRA 前向与"base 单独跑 + LoRA 手算相加"数值一致 |
| M2.4 | `Arc<BaseWeights>` 零拷贝共享：autograd `GpuTensor` 支持"frozen view"模式，不参与 tape | `autograd/src/tensor.rs` | LoRA forward 不克隆 base weight，显存不翻倍 |
| M2.5 | 合成数据 supervised fine-tune loop：随机生成 `(prompt_tokens, target_tokens)` pairs，cross-entropy loss，AdamW 更新 LoRA | `train/src/trainer.rs`，`train/tests/supervised.rs` | Qwen3.5-family model + LoRA rank=8，100 步 loss 明显下降（>50%） |
| M2.6 | **热切**：LoRA delta 写完后，推理侧用新 adapter；double-buffer `Arc<RwLock<LoRAAdapters>>` 切换 | `train/src/weight_sync.rs` | 集成测试：训练 100 步后，推理同一 prompt 输出 token 分布明显变化 |
| M2.7 | 只训练 LoRA（base 不更新）验证：跑 1 epoch 后，`BaseWeights` 的 CUDA 指针指向数据 bitwise 不变 | `train/tests/base_frozen.rs` | 测试绿 |

### 4.3 验收门槛

- ✅ Qwen3.5-family checkpoint 加载后 + LoRA rank=8，显存占用 ≤ base 的 1.05x（LoRA 参数 < 1% base）
- ✅ 推理侧热切 LoRA 后，同一 prompt 的 top-1 token 或 logits 发生非平凡变化
- ✅ Base weights bitwise 不变（grad 流只到 LoRA）
- ✅ `cargo test --workspace --release` 无回归

### 4.4 风险 & 缓解

- **Hook 改到 `infer/src/ops/linear.rs` 是入侵点**。缓解：默认 `lora: Option<&LoRAAdapter>`，生产推理路径 `None`，开销零；训练开启时才走 LoRA 分支。
- **显存翻倍**：base + LoRA 共存，如果不共享会翻倍。M2.4 的 frozen view 是硬性门槛。
- **adapter 热切时推理中请求读到半更新**：double-buffer 写 slot B，原子 swap `Arc` 指针，正在执行的 forward 继续用旧 Arc 直到该请求结束。

---

## 5. M3 — GRPO + 单 verifier 闭环（21 天）

### 5.1 Scope

真正的 RL：rollout → verifier reward → GRPO loss → AdamW → 热切 → 下一轮。单 verifier（数学 GSM8K-like exact-match 起步），非 agent（单轮 prompt→response）。

### 5.2 任务分解

| # | 任务 | 文件 | 验收 |
|---|---|---|---|
| M3.1 | Trajectory 结构：`Trajectory { prompt_ids, response_ids, step_logprobs, reward }` | `train/src/rollout.rs` | 单测：从一次 agent-infer 完成请求能构造 Trajectory |
| M3.2 | Scheduler 侧新增 trajectory emit channel（tokio mpsc）：每个请求完成时 emit | `infer/src/scheduler/` | 改动最小化，不影响推理 throughput（基准测试证明 < 1% 回归） |
| M3.3 | **Group advantage**：对同一 prompt 采样 G 个 response，reward 归一化 `A_i = (r_i - mean) / std` | `train/src/grpo.rs` | 单测：输入 (G, rewards)，输出 advantages 有界、方向对 |
| M3.4 | **GRPO loss**：`L = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] + β * KL(π_θ || π_ref)` | `train/src/grpo.rs` | grad_check（ratio 的 autograd）；KL 项数值稳定 |
| M3.5 | **KL to reference policy**：ref policy = 初始 LoRA-off 的 base model；ref logp 在 rollout 时算一份冻住 | `train/src/grpo.rs` | KL 在 well-behaved 范围（< 1.0 per token） |
| M3.6 | **Rollout↔Train 交替主循环**：N=16 prompts × G=4 samples → 收齐 → GRPO 1 个 epoch K=4 步 → 热切 → 下一轮 | `train/src/trainer.rs` | 一轮 cycle 跑通，wall clock 测 |
| M3.7 | **数学 verifier**：GSM8K 风格 prompt，exact-match final number；`fn verify(prompt: &str, response: &str) -> f32` | `train/src/reward/math.rs` | 手造 20 样本测试集，verifier 判断正确率 100% |
| M3.8 | **Held-out 评估 harness**：每 10 轮训练后在 held-out 100 prompts 上跑 pass@1 | `train/src/eval.rs` | 每次训练 log 出 held-out reward |
| M3.9 | 端到端训练 6 小时不崩，held-out reward 上升 | `train/tests/e2e_grpo.rs` (ignore by default) | GSM8K-like held-out pass@1 比 base 高 ≥15% |

### 5.3 验收门槛

- ✅ 训练与推理共享同一模型权威，6 小时稳定；实现上可同进程，也可通过异步 worker 边界落地
- ✅ held-out reward 单调不降（允许短时波动，窗口均值上升）
- ✅ 不是 reward hack：人工抽检 20 个 sample，至少 15 个解题路径合理（而不是"蒙对答案"）
- ✅ KL 到 ref policy 在合理带内（β 调到 KL 1.0 附近）
- ✅ `infer_prefix_hit_rate` 在 rollout 阶段 > 0.3（同一 prompt 的 G 个 sample 共享 prefix）

### 5.4 学习预期

M3 是**自证"训推一体"概念能跑的标志**。ckl 应能自主判断：
- Rollout 和 Train 谁是瓶颈（看 GPU util 分布）
- KL 系数应该往哪调（log KL 曲线 + reward 曲线形状）
- verifier 是否过紧或过松（bucket 掉的 sample 数量）

---

## 6. M4 — Agent 自进化 MVP（4–6 周）

### 6.1 Scope

从"单轮 prompt→response RL"扩到"agent 多轮 tool use RL + 多 verifier + 基础 curriculum"。M4 交付后这个项目从"训练器"变成**"自进化 agent 框架"**。

### 6.2 任务分解

| # | 任务 | 要点 | 验收 |
|---|---|---|---|
| M4.1 | **Multi-turn trajectory**：agent tool loop 的每一步 (action, observation) 都进 trajectory，step logp 按 action-token 计算 | 参考 agent 现有 tool loop，不改逻辑，加 callback | 多轮 trajectory 能回到 trainer，logp 和 forward 一致 |
| M4.2 | **Stepwise reward assignment**：最终 reward 按 step 反向折扣（γ=1 或 0.99）；工具调用失败的 step 单独加小惩罚 | `train/src/reward.rs` | 手造 3 种失败样例，惩罚正确触发 |
| M4.3 | **多 verifier**：数学 + 代码（pytest-like 单测 Rust 原生实现）+ 工具调用成功率 | `train/src/reward/{math,code,tool}.rs` | 每个 verifier 独立单测 |
| M4.4 | **Reward aggregation**：多 verifier 加权；权重作为 config，方便调 | `train/src/reward/aggregate.rs` | 文档化每个 verifier 的 scale |
| M4.5 | **基础 curriculum**：任务池分级（easy/medium/hard），base pass@1 > 0.8 的 easy 自动 retire，引入新 hard 任务 | `train/src/curriculum.rs` | 训练 24h 后任务池分布向 harder 移动 |
| M4.6 | **Task generator**（self-play 雏形）：让 model 自己生成新任务 + 匹配的 verifier（verifier 用模板生成，限定在数学 / 代码 DSL 内） | `train/src/curriculum/gen.rs` | 生成任务可 verifier-grounded（不接受无 verifier 任务） |
| M4.7 | **当前控制面**：train-side server + `train_multi_turn --serve`；**目标统一入口**：`/v1/train`（可选） | `train/src/server.rs` / `infer/src/http_server/train.rs` | curl 调通 |
| M4.8 | **冒烟自进化**：固定 100 个 base 解决不了的 hard 任务，训练 24h 后能解决 ≥ 30% | `train/tests/e2e_self_evolve.rs` | 验证通过 |

### 6.3 验收门槛

- ✅ 24h 自进化训练：起始 base pass@1 on hard-set 记为 P0；训练后 P24 ≥ P0 × 1.3 且绝对值 ≥ 0.3
- ✅ 抽检 10 个 Agent 解题 trajectory，每个都有**多轮工具调用**且路径合理
- ✅ Curriculum 分布可见向 hard 移动（log + snapshot）
- ✅ 不崩：24h 内 0 panic，0 OOM，0 死锁

### 6.4 开放问题（M4 期间解决）

- Reward hack 检测机制（抽样人审 + 异常 reward 波动告警）
- Self-play 生成的 verifier 可信度（templated verifier 的边界）
- 多轮 trajectory 的 GAE / TD 选择（第一版 MC return，若方差大再上 GAE）

---

## 7. M5 — Metal 支线（与 M3/M4 并行）

### 7.1 Scope

把 `autograd` 的 op 在 `mlx-sys` 上提供 Metal 实现，让 Mac 本地能跑 M2 的最小 demo（1.5B 模型 LoRA fine-tune）。

### 7.2 任务分解

| # | 任务 | 备注 |
|---|---|---|
| M5.1 | ✅ 2026-04-18 调研完成：`mlx::core::grad` 存在但 mlx-sys bridge 只暴露 forward | 结论见下方 M5.2 |
| M5.2 | ✅ 2026-04-18 路线锁定 = **(b)**：在我们的 tape 上用 mlx-sys forward 调 MLX op，bwd 公式自己写，和 CUDA 同 tape。commit `a46fc00` 落地 `Backend` trait + `CpuBackend`/`MetalBackend`/`CudaBackend`（per-call upload/compute/download） | 选 (b) 的原因：一致性 > 便利；和 CUDA 同 tape |
| M5.3 | 🟢 **M5.3a LANDED 2026-04-20** — device-resident tensor + lazy eval boundary shipped. Commits `17b3599`（M5.3a.1: Tensor.device_handle + Dirty enum + ensure_host/ensure_device）、`7d024d8`（M5.3a.2: lazy matmul + add）、加 matmul-backward-on-GPU follow-up（见 `docs/experience/wins/2026-04-20-matmul-backward-gpu.md`）。验收：`cargo test -p autograd --release --features metal` 48 路全绿，`metal_single_forward_backward_step_has_bounded_eval_count` 断言 forward matmul=0 eval / sum=1 eval / backward ≤2 eval（对照 pre-M5.3a 每 op 一次 eval 的退化路径）。bench_step_matmul 批量 32 时 d_model=128 Metal = 1.74× CPU（gate ≥1.1×），d=256 5.78×，d=512 12.34×；批量 4 仍 CPU 主场（d=128 0.42×）。详细规格：[`docs/plans/m5.3-device-resident-tensor.md`](m5.3-device-resident-tensor.md)；验收快照：[`docs/experience/wins/2026-04-20-m5.3a-device-resident-tensor-acceptance.md`](../experience/wins/2026-04-20-m5.3a-device-resident-tensor-acceptance.md)。**M5.3b 进行中**：✅ M5.3b.1 `sum_all` lazy（commit `b28aa36` 前后）。✅ M5.3b.2 `softmax`/`log_softmax` lazy forward（commit `93447e0` — forward 链 `matmul → log_softmax → sum` 0 eval，`metal_log_softmax_forward_backward_records_current_eval_cost` 断言 after_log_softmax==0 / after_backward≤3）。✅ M5.3b.3 `silu` lazy forward（commit `875c935` — `mlx_sigmoid * x` 组合，MLP 每层 2 次 silu 从 eager→lazy，Qwen3.5 28 层潜在减少 56 evals/token；新增 `metal_silu_forward_stays_lazy` 断言）。✅ M5.3b.4 `exp` lazy forward（`mlx_exp` 单节点接入 MLX graph，Dirty-dispatch 同 silu 模板；`exp_backward` 依赖 `tape.backward` 的 pre-walk batch-flush 读 saved output，forward 保持 lazy 仍安全；新增 `metal_exp_forward_stays_lazy` 断言 forward 链 0 eval）。✅ M5.3b.5 `rope` lazy forward（`mlx_slice` → `mlx_multiply` → `mlx_subtract`/`mlx_add` → `mlx_concatenate_axis` 全链 lazy，借 `x_handle` 不重新 upload；cos/sin 仍 host 侧 upload 每次；dispatch 放宽到 `device_handle.is_some() && dirty != Host`，兼顾 Dirty::Device 和 Dirty::Both（ensure_device 后重入场景）；Qwen3.5 每层 q/k 两次 rope × 28 层 = 56 次 eval/token 潜在消除；新增 `metal_rope_forward_stays_lazy` 断言 forward 0 eval + CPU 参考对拍 ≤ 1e-5）。✅ M5.3b.6 `rmsnorm` lazy forward（`mlx_fast_rms_norm` 单节点，weight 每次 upload 成 `[hidden]` 临时 array；`SavedContext::RMSNormCtx` 加 `eps: f32` 字段，lazy 路径存空 `inv_rms` vec 作 sentinel，backward 直接用已 destructure 的 eps 从 host-flushed x 重算 inv_rms；Qwen3.5 每层 3 次 rmsnorm（input/post-attn/post-mlp）× 28 层 = 84 evals/token 潜在消除；新增 `metal_rmsnorm_forward_stays_lazy` 断言 forward 链 `matmul → rmsnorm → sum` 0 eval + CPU 参考对拍 ≤ 1e-4）。✅ M5.3b.7 `embedding` lazy forward（`mlx_take_axis(table_handle, ids_i32, axis=0) → mlx_reshape [seq,hidden]→[1,seq,hidden]` 组合，ids 每 step upload 成 `[seq]` int32 临时 array；dispatch 同 rope 模板覆盖 Dirty::Both，故 AdamW 上传新表后下一次 forward 仍走 lazy；embedding 是 forward pass 第一个 op，上游 lazy 打开后全链从 token id 起就驻留在设备；新增 `metal_embedding_forward_stays_lazy` 断言 `embedding → rmsnorm → sum` 链 0 eval + CPU 参考对拍 ≤ 1e-5；backward 仍走 host scatter-add，依赖 tape 批量 flush 提供 host upstream）。✅ M5.3b.8 `gelu` lazy forward (erf 形式，`mlx_multiply(INV_SQRT_2, x) → mlx_erf → mlx_add(1, ·) → mlx_multiply(0.5x, ·)` 组合；关键：用 erf 而**非** `gelu_forward` 的 tanh 近似，因为 `gelu_backward` 用 saved `x` 算 erf 导数，forward 若切 tanh 会和 backward 偏离 ~1e-3；新 `Backend::gelu` device 方法正好就是 erf 契约，tanh `gelu_forward` 留给别处；新增 `metal_gelu_forward_stays_lazy` 断言 forward 链 0 eval + 与 CPU erf 公式对拍 ≤ 1e-4）。剩余：gather/AdamW 同模板逐个 port；eval_count 目标收敛到严格 1 per step（当前 ≤3：1 tape 批量 flush + 1 matmul_backward FFI eval + 1 log_softmax backward host-mul readback slack，后者需 softmax backward 上设备才能归零）。 | 架构就位后每个 op 数值对拍 CPU f64 参考 ≤ 1e-3 |
| M5.4 | ⏳ 未开工：Mac 上跑 M2 的合成 supervised fine-tune，Qwen 1.5B，LoRA rank=8 | 前置 = M5.3a ✅ + M5.3b 完成；loss 曲线形状和 CUDA 一致 |

### 7.3 验收门槛

- ✅ `cargo test -p autograd --no-default-features --features metal` Mac 上全绿
- ✅ Metal / CUDA / CPU 三路径的 grad_check 互相对拍

---

## 8. 每日颗粒度（M0 示范，M1+ 开工前当天再拆）

```
M0 Day 1:
  - Create crates/autograd/ + Cargo.toml + add to workspace
  - Write src/lib.rs (empty module decls)
  - Write src/tensor.rs: TensorId + shape + strides + TensorStore skeleton
  - First `cargo build -p autograd` green

M0 Day 2:
  - Implement TensorStore: alloc/free/to_host/from_slice
  - Unit tests for alloc+free with slot recycle
  - Implement Tape + TapeEntry + SavedContext (minimal enums: None, Tensor, TensorAndScalar, Shape)
  - BackwardOp enum: Add, Mul, MulScalar, Sum

M0 Day 3:
  - Tape::backward() with DFS relevant + topo-order
  - Implement ops::elementwise::add + add_backward
  - grad_check helper (tests/helpers.rs)
  - First grad_check test on Add

M0 Day 4:
  - Implement mul + mul_backward, mul_scalar + mul_scalar_backward
  - Implement ops::reduce::sum + sum_backward
  - grad_check for all four ops

M0 Day 5:
  - Integration test: y = ((a+b)*3).sum(); y.backward(); assert a.grad == 3
  - clippy clean
  - Commit: "feat(autograd): M0 minimal tape + add/mul/sum"
```

**M1–M4 开工前一天**按此颗粒度拆。超出时间盒 30% 以上时 → 暂停，评估是否要调整 scope。

---

## 9. 测试 / CI 约定

### 9.1 测试分层

| 层 | 命令 | 运行环境 |
|---|---|---|
| Unit（op-level） | `cargo test -p autograd` | Mac CPU / CUDA 任一 |
| Unit（train-level） | `cargo test -p train` | 需要 Qwen 小模型权重 |
| grad_check | `cargo test -p autograd grad_check -- --include-ignored` | CPU + CUDA |
| E2E supervised (M2) | `cargo test --release --test m2_supervised -- --ignored` | CUDA，~5 min |
| E2E GRPO (M3) | `cargo test --release --test e2e_grpo -- --ignored` | CUDA，~1 h |
| E2E self-evolve (M4) | `cargo test --release --test e2e_self_evolve -- --ignored` | CUDA，~24 h |

### 9.2 CI 增量（不改 agent-infer 现有 CI）

新增 job：
- `cargo test -p autograd`（每次 push）
- `cargo test -p autograd --features cuda`（需要 CUDA runner，可手动触发）
- `cargo clippy -p autograd -- -D warnings`
- `cargo clippy -p train -- -D warnings`

---

## 10. 文档 / 提交纪律

### 10.1 Commit 规范

按 agent-infer 现有 commitizen：`<type>(<scope>): <subject>`。新增 scopes：
- `autograd` — autograd crate 本体
- `train` — train crate 本体
- `rl` — 跨 crate 的 RL 循环级改动

### 10.2 每个 M 结束必须产出

1. **Experience wins 条目**：`docs/experience/wins/YYYY-MM-DD-agent-rl-m<N>-<slug>.md`，按 [`TEMPLATE`](../experience/wins/) 格式
2. **如遇非平凡 bug（>1 次尝试失败）**：`docs/experience/errors/YYYY-MM-DD-<slug>.md`
3. **ROADMAP.md Phase 6 状态更新**：勾选对应 milestone
4. **本 plan 文件**：在 §11 "Progress log" 追加一行

---

## 11. Progress log

| 日期 | 里程碑 | 状态 | 备注 |
|---|---|---|---|
| 2026-04-18 | Plan + project doc + research note 提交 | ✅ | 锁定 scope v3；准备开工 M0 |
| 2026-04-18 | M0–M1 Autograd + 核心 op + AdamW | ✅ | TinyLM (~8.4M) CPU SFT 收敛到位；保留为历史 scaffolding，不是当前 train-side model line |
| 2026-04-18 | M2a LoRA on TinyLM (self-contained) | ✅ | frozen base + rank-r adapters, grad 仅流向 A/B；保留为历史 scaffolding |
| 2026-04-18 | M2b LoRA hook into Qwen3 `linear.rs` | ✅ | 走 [`m2b-blocker-analysis.md`](m2b-blocker-analysis.md) 选项 (b)：`LoRAAdapter { a/b: DeviceMatrix }` 落在 `infer/src/model/qwen3/lora.rs`（不与 train `TensorStore` 共享），PEFT loader + additive apply ops + prefill/decode hot-path wiring + synthetic safetensors integration test；这仍是 Qwen3-era 历史实现记录，不是当前 train-side 主线；CUDA Graph decode 在 LoRA 激活时自动降级为 eager（`supports_cuda_graph_decode`），warmup 仍跑两遍以预热 cublasLt autotune cache；train↔infer gradient loop 仍走选项 (a)/M1-CUDA, 未在此 phase 内 |
| 2026-04-18 | M3 GRPO 单 verifier 闭环 | ✅ | rollout_group + group_advantages + PPO-clip surrogate |
| 2026-04-18 | M3.5 PPO clip + multi-verifier scaffolding | ✅ | host-space active-mask；Copy/ReverseCopy/Palette/WeightedEnsemble |
| 2026-04-18 | M4.1 Multi-turn episode scaffolding | ✅ | Episode / TurnSpec / Environment / rollout_episode |
| 2026-04-18 | M4.2 Stepwise returns wired through GRPO | ✅ | discounted_returns + group_normalize + returns_to_per_position + `grpo_loss_per_position` |
| 2026-04-18 | `train_multi_turn` 二进制 (stepwise RL loop) | ✅ | smoke: mean_reward 0.09 → 0.31 over 30 iters (vocab=16, 2 turns × 2 agent tokens, group=8) |
| 2026-04-18 | M4.3 多 verifier（真实 math/code/tool archetype） | ✅ | `ArithmeticVerifier`（digit-token 加/乘解码评分）+ `MonotonicVerifier`（严格递增代码风格）+ `ToolSuccessVerifier`（sentinel 工具成功代理）；`VerifierKind` 扩展 + 测试；真实 tokenizer 对接时只换解码器（11 test 全过） |
| 2026-04-18 | M4.4 Reward aggregation config | ✅ | `RewardConfig` + `VerifierKind` 数据驱动；`WeightedEnsemble::from_config` 与 fluent builder 语义等价 (测试对拍) |
| 2026-04-18 | M4.5 基础 curriculum（task pool + auto-retire） | ✅ | `TaskPool` 滚动 pass@1 窗口 + `min_samples_before_retire` 门槛；`sample` 排除 retired；`active_distribution` 导出分级存活数（8 test 全过） |
| 2026-04-18 | M4.6 Task generator（self-play scaffold） | ✅ | `TaskGenerator` + `TierSpec`：每个 `GeneratedTask` 结构性绑定 `VerifierKind`（verifier-grounded invariant），参数落在 tier bounds 内，权重分布对拍 ±5%（5 test 全过） |
| 2026-04-18 | M4.7 /v1/train HTTP control plane | ✅ | `train::control::TrainingController`（`Arc<Mutex<TrainingStatus>>` + `AtomicBool` stop/save，save 为 `swap(false)` 边沿触发）+ `train::server`（std `TcpListener`，0 新依赖，8 KiB 请求上限）；routes `GET /v1/train/status` / `POST /v1/train/stop` / `POST /v1/train/save`；`train_multi_turn --serve PORT` 启动控制面，`iter`/`mean_reward`/`best_reward`/`last_kl`/`last_loss`/`wall_secs` 每步回写；curl 三端点 + 404 端到端冒烟通过（5 test 全过） |
| 2026-04-18 | M4.8 Self-evolve 冒烟 + 训练速度 baseline | ✅ | `train_multi_turn` 内建 `bench:` 行（wall / iter/s / episode/s / token/s）；CPU vs Metal snapshot → `docs/experience/wins/2026-04-18-bench-train-multi-turn.md`；TinyLM-scale CPU 833 tok/s、d_model=256 Metal 1.40× 超 CPU；24h full self-evolve 待 Arithmetic verifier 驱动硬任务后再跑 |
| 2026-04-18 | M5 Backend trait + Metal matmul + CUDA matmul (标记待验证) | ✅ | `Backend` trait + `CpuBackend`/`MetalBackend`/`CudaBackend`；Metal 对 CPU 参考 ≤1e-3（4 test 全过）；CUDA 路径 Mac typecheck 通过，等 GPU 机器验证；`train_multi_turn --backend metal` 端到端通过 |

---

## 12. 不要做的事（反向清单）

1. **不要在 M0 阶段碰 GPU**。CPU 路径跑通前不看 CUDA。
2. **不要先写 LoRA 再写 autograd**。autograd 是地基，地基不稳后面全塌。
3. **不要 vendor mni-ml 代码**。读、理解、重写。
4. **不要引入 candle / burn / tch**。这是项目的核心否定约束。
5. **不要做多 backend 抽象层**（Backend trait + 实现）。CUDA / CPU / Metal 直接 `#[cfg]`，照 mni-ml 的做法。
6. **不要在 M2 之前碰 agent-infer `infer/src/ops/linear.rs`**。M0/M1 纯粹在新 crate 里。
7. **不要在 M3 之前碰 agent tool loop**。先单轮闭环，再多轮。
8. **不要做 checkpoint / resume**。M1–M4 全程"进程不崩就不 checkpoint"。M5+ 再说。
9. **不要做分布式**。任何 NCCL / MPI / Gloo / process-group 相关提案，本 plan 外。
10. **不要为未来扩展预留抽象**。YAGNI，写到用到再抽。
