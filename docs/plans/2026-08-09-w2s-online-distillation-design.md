# W2S 在线蒸馏工程设计

> 来源：[w2s 在线蒸馏实验设计](https://bytedance.larkoffice.com/docx/MIRKdxiKeou1nxxgK23c1ItKnwx)
> 目标：在 ARLE 训练 substrate 上复现 weak-to-strong 在线蒸馏，验证弱辅助模型的策略偏移量能否在线提升强主模型。

## 1. 可行性评估

### 1.1 ARLE 已有能力（可直接复用）

| 能力 | 现有实现 | 位置 |
|------|---------|------|
| KL 蒸馏损失 | forward / reverse KL，温度 T，T² 补偿，chunked | `loss.rs::kl_distill_loss` |
| LoRA student | rank / alpha / target set / layer start | `lora.rs` |
| AdamW 优化器 | host path + device path | `autograd::optim::AdamW` |
| 多 teacher | `MultiTeacher` + routing | `teacher_infer.rs` |
| Teacher runtime | in-process / infer-api / HTTP API | `teacher_infer.rs` |
| Checkpoint | v2: `trainer_state.json` + `optimizer.safetensors` | `checkpoint.rs` |
| 并行 | TP / CP / PP / DP | `tensor_parallel.rs` 等 |
| 显存优化 | gradient checkpointing, offload | `runtime_flags.rs` |
| Infer rollout | CUDA graph + paged KV，LoRA sync | `infer_student.rs` |
| 自训练 + EMA | SOPD: EMA self-teacher, snapshot/revert | `ema_self_teacher.rs` |

### 1.2 需要新增的能力

| 能力 | 说明 | 复杂度 |
|------|------|--------|
| 策略偏移量 ΔT | pre-RL 与 post-RL log-prob 差 | 中 |
| Proxy teacher | `z_proxy = z_s.detach() + α·T·(ΔT₁+ΔT₂)/2` | 低 |
| 双辅助模型 | 每个有 pre/post-RL 版本 | 中 |
| 一致性门控 | `cos(ΔT₁, ΔT₂) > threshold` | 低 |
| 置信度过滤 | `max(softmax(z_s)) < threshold` | 低 |
| Shadow adapter | 训练 shadow，原子切换到 serving | 中 |
| 局部 + 全局 KL 正则 | `π_new vs π_old`, `π_new vs π_base` | 低 |
| 在线更新流程 | buffer → forward → filter → gate → accum → switch | 高 |

**结论：可行。** 核心算法（ΔT + proxy teacher + reverse KL）可在现有 OPD substrate 上扩展实现。主要工程量在 shadow adapter 机制和在线更新流程。

## 2. 核心算法

### 2.1 策略偏移量

对每个辅助模型 i，计算 post-RL 与 pre-RL 的 log-prob 差：

```
ΔTᵢ = log_softmax(z_post_rlᵢ) - log_softmax(z_pre_rlᵢ)
```

ΔTᵢ 是 token 级别的 log-prob 差向量，形状 `[seq_len, vocab]`。

### 2.2 Proxy teacher

```
z_proxy = z_student.detach() + α · T · (ΔT₁ + ΔT₂) / 2
```

- `z_student` 必须 detach，梯度只流过 student
- α 是蒸馏强度（0.1–1.0），T 是温度（2–4）
- 有效强度为 α·T·‖ΔT‖，需与 ‖z_student‖ 可比

### 2.3 蒸馏损失

```
L_kd = T² · KL( softmax(z_proxy / T) || softmax(z_student / T) )
```

reverse KL（学生为 q，proxy teacher 为 p），对弱老师的预测不确定性更鲁棒。

### 2.4 灾难性遗忘正则

```
L = L_kd + β₁ · KL(π_new || π_old) + β₂ · KL(π_new || π_base)
```

- `π_old`：上一版本 adapter，局部约束
- `π_base`：主模型初始权重，全局锚点，防止无界漂移

### 2.5 门控

1. **置信度过滤**：`max(softmax(z_student)) > 0.9` → 跳过（主模型已确定）
2. **一致性门控**：`cos(ΔT₁, ΔT₂) < threshold` → 跳过（两弱模型方向不一致）

## 3. 架构设计

### 3.1 模块划分

```
crates/train/src/
├── w2s.rs                    # 新增：w2s 核心
│   ├── W2sAuxModel           # 单个辅助模型（pre-RL + post-RL）
│   ├── W2sConfig             # 配置
│   ├── W2sStepOutcome        # 单步结果
│   ├── w2s_step              # 单步训练
│   └── ShadowAdapterManager  # shadow adapter 管理
├── teacher_infer.rs          # 扩展：W2sAuxTeacher
├── lora.rs                   # 扩展：双 adapter slot
└── ...
```

CLI：
```
crates/cli/src/
├── args.rs                   # TrainW2sArgs
└── train_cli.rs              # run_w2s
```

### 3.2 数据流

```
请求 x
  │
  ├─ Student forward (shadow adapter) → z_s
  ├─ Aux1 pre-RL forward  → lp_pre₁
  ├─ Aux1 post-RL forward → lp_post₁ → ΔT₁ = lp_post₁ - lp_pre₁
  ├─ Aux2 pre-RL forward  → lp_pre₂
  ├─ Aux2 post-RL forward → lp_post₂ → ΔT₂ = lp_post₂ - lp_pre₂
  │
  ├─ 置信度过滤: max(softmax(z_s)) < confidence_threshold ?
  ├─ 一致性门控: cos(ΔT₁, ΔT₂) > consistency_threshold ?
  │
  ├─ z_proxy = z_s.detach() + α·T·(ΔT₁ + ΔT₂) / 2
  │
  ├─ L_kd = T²·KL(softmax(z_proxy/T) || softmax(z_s/T))
  ├─ L_local = β₁·KL(π_new || π_old)
  ├─ L_global = β₂·KL(π_new || π_base)
  ├─ L = L_kd + L_local + L_global
  │
  └─ Backward → shadow LoRA adapter
```

### 3.3 Shadow adapter 机制

```
┌──────────────────────────────────────────────────────┐
│                  Shadow Adapter 流程                   │
├──────────────────────────────────────────────────────┤
│                                                      │
│  serving adapter (infer engine)                      │
│       ↑                                              │
│       │ 原子切换 (eval 通过)                          │
│       │                                              │
│  shadow adapter (training store)                     │
│       ↑                                              │
│       │ gradient update                              │
│       │                                              │
│  每 N 步:                                             │
│    1. eval shadow adapter on validation set          │
│    2. 指标提升且统计显著 → 原子切换 serving ← shadow  │
│    3. 否则 → 丢弃 shadow，重置为 serving              │
│                                                      │
│  回滚: 保存上一版本 serving adapter 快照              │
│                                                      │
└──────────────────────────────────────────────────────┘
```

## 4. 组件详细设计

### 4.1 W2sAuxModel

```rust
/// 单个辅助模型：持有 pre-RL (base) 和 post-RL (instruct) 版本。
/// 两个版本权重都冻结，只做 forward，不参与梯度。
pub struct W2sAuxModel {
    pre_rl: Qwen35Model,
    post_rl: Qwen35Model,
}

impl W2sAuxModel {
    /// 计算策略偏移量 ΔT = log_softmax(post_rl) - log_softmax(pre_rl)。
    /// 返回的 tensor 是常量（tape 上无 grad）。
    pub fn forward_delta(
        &self,
        input_ids: &[u32],
        positions: &[u32],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let lp_post = log_softmax(self.post_rl.forward(input_ids, positions, store, tape)?, store, tape)?;
        let lp_pre = log_softmax(self.pre_rl.forward(input_ids, positions, store, tape)?, store, tape)?;
        sub(lp_post, lp_pre, store, tape)
    }
}
```

**Runtime 选项**：
- `in-process`：autograd 加载，适合小模型
- `infer-api`：通过 `LoadedInferenceEngine` forward，适合大模型
- `api`：HTTP 调用独立服务的 `/v1/raw_logits`

### 4.2 W2sConfig

```rust
pub struct W2sConfig {
    /// 蒸馏强度 α。α·T·‖ΔT‖ 应与 ‖z_student‖ 可比。
    pub alpha: f32,
    /// 温度 T（2–4）。
    pub temperature: f32,
    /// 置信度过滤阈值。student max prob > 此值则跳过。
    pub confidence_threshold: f32,
    /// 一致性门控阈值。cos(ΔT₁, ΔT₂) < 此值则跳过。
    pub consistency_threshold: f32,
    /// 局部 KL 正则权重 β₁（vs π_old）。
    pub beta_local: f32,
    /// 全局 KL 正则权重 β₂（vs π_base）。
    pub beta_global: f32,
    /// 梯度累积步数（16–64）。
    pub grad_accum_steps: usize,
    pub rollout_len: usize,
    pub grad_clip: f32,
}

impl Default for W2sConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            temperature: 2.0,
            confidence_threshold: 0.9,
            consistency_threshold: 0.0,
            beta_local: 0.01,
            beta_global: 0.001,
            grad_accum_steps: 32,
            rollout_len: 8,
            grad_clip: 1.0,
        }
    }
}
```

### 4.3 w2s_step

```rust
pub fn w2s_step<O: Optimizer>(
    student: &Qwen35Model,        // 主模型，LoRA = shadow adapter
    student_old: &Qwen35Model,    // 上一版本（局部 KL），LoRA = previous serving
    student_base: &Qwen35Model,   // base 版本（全局 KL），无 LoRA
    aux1: &W2sAuxModel,
    aux2: &W2sAuxModel,
    prompt_ids: &[u32],
    cfg: &W2sConfig,
    shadow_params: &[TensorId],
    optimizer: &mut O,
    store: &mut TensorStore,
    tape: &mut Tape,
) -> Result<W2sStepOutcome> {
    // 1. Student forward → z_s
    let z_s = student.forward(prompt_ids, positions, store, tape)?;

    // 2. 置信度过滤
    let student_probs = softmax(z_s, store, tape)?;
    let max_prob = max_last_dim(student_probs, store, tape)?;
    if max_prob > cfg.confidence_threshold {
        return Ok(W2sStepOutcome { skipped: true, reason: SkipReason::Confidence, .. });
    }

    // 3. 辅助模型 ΔT
    let delta1 = aux1.forward_delta(prompt_ids, positions, store, tape)?;
    let delta2 = aux2.forward_delta(prompt_ids, positions, store, tape)?;

    // 4. 一致性门控
    let sim = cosine_similarity(delta1, delta2, store, tape)?;
    if sim < cfg.consistency_threshold {
        return Ok(W2sStepOutcome { skipped: true, reason: SkipReason::Consistency, .. });
    }

    // 5. 构建 proxy teacher
    let avg_delta = mul_scalar(add(delta1, delta2, store, tape)?, 0.5, store, tape)?;
    let scaled_delta = mul_scalar(avg_delta, cfg.alpha * cfg.temperature, store, tape)?;
    let z_proxy = add(detach(z_s), scaled_delta, store, tape)?;

    // 6. KL 蒸馏损失（reverse）
    let loss_kd = kl_distill_loss(
        z_s, z_proxy, num_positions,
        cfg.temperature, KlDirection::Reverse, store, tape,
    )?;

    // 7. 局部 + 全局 KL 正则
    let z_old = student_old.forward(prompt_ids, positions, store, tape)?;
    let z_base = student_base.forward(prompt_ids, positions, store, tape)?;
    let loss_local = kl_distill_loss(z_s, detach(z_old), ..., KlDirection::Reverse, ...)?;
    let loss_global = kl_distill_loss(z_s, detach(z_base), ..., KlDirection::Reverse, ...)?;

    let loss = add(loss_kd,
        add(mul_scalar(loss_local, cfg.beta_local, ...),
            mul_scalar(loss_global, cfg.beta_global, ...), ...), ...)?;

    // 8. Backward + optimizer step
    tape.backward(loss)?;
    optimizer.step(shadow_params, store, tape)?;

    Ok(W2sStepOutcome { loss, skipped: false, .. })
}
```

### 4.4 Shadow adapter

复用现有 `LinearWithLora`，增加双 adapter slot：

```rust
pub struct LinearWithLora {
    base_weight: TensorId,
    lora_a: TensorId,   // 当前 active adapter
    lora_b: TensorId,
    // shadow adapter 权重存在单独的 tensor 中
}

impl LinearWithLora {
    /// 原子切换：将 shadow adapter 权重拷贝到 active adapter。
    /// 通过 swap tensor 数据实现，O(1) 指针交换。
    pub fn swap_shadow_to_active(&mut self, store: &mut TensorStore) -> Result<()>;

    /// 重置 shadow adapter 为 active adapter 的副本。
    pub fn reset_shadow_from_active(&mut self, store: &mut TensorStore) -> Result<()>;
}
```

**切换策略**：
- 每 `eval_every_n_steps` 步，用 shadow adapter 在验证集上评估
- 指标提升且统计显著（p < 0.05）→ `swap_shadow_to_active`
- 否则 → `reset_shadow_from_active`，丢弃 shadow 训练

**回滚**：每次切换前保存 active adapter 快照；切换后回归 benchmark 下降超阈值 → 恢复快照。

### 4.5 在线更新流程

```
┌─────────────────────────────────────────────────────────────┐
│                    Online Update Loop                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  请求 buffer (容量 = grad_accum_steps)                        │
│       │                                                      │
│       ▼                                                      │
│  批量 forward: student + aux1(pre/post) + aux2(pre/post)     │
│       │                                                      │
│       ▼                                                      │
│  过滤 + 门控 → 有效样本加入梯度累积                            │
│       │                                                      │
│       ▼ (累积满 grad_accum_steps)                            │
│  optimizer.step → shadow adapter                             │
│       │                                                      │
│       ▼ (每 eval_every_n_steps)                              │
│  eval shadow on validation set                               │
│       │                                                      │
│       ├─ 提升显著 → 原子切换 serving ← shadow                 │
│       └─ 否则 → 重置 shadow                                   │
│                                                              │
│  回归监控: 每次切换后跑回归 benchmark                          │
│    MMLU ↓ > 2pp | HumanEval ↓ > 3pp | GSM8K ↓ > 3pp → 回滚   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 5. 显存与延迟估算

### 5.1 显存

| 组件 | 显存 |
|------|------|
| 主模型 8B FP16 | 16 GB |
| 辅助模型 4B INT4 × 2 (post-RL) | 4 GB |
| 辅助模型 pre-RL 4B INT4 × 2 | CPU 内存 4 GB（按需加载） |
| KV cache | 14 GB |
| LoRA 激活 + 优化器 | 1 GB |
| **总计** | **~35 GB** |

单张 A100 40GB 可满足（余量 5GB）。80GB 卡可增大 batch size 或序列长度。

### 5.2 延迟

三模型并行推理权重流量：8B(FP16) + 4B(INT4) + 4B(INT4) = 20 GB。
单 8B 权重流量 16 GB。
延迟增量 ≈ 20/16 − 1 = 25%。目标 ≤ 30%，INT4 量化辅助模型可满足。

## 6. 实现阶段

### Phase 1: 离线 w2s 训练（核心算法）

**目标**：跑通 W0–W2 实验，验证 ΔT + proxy teacher + reverse KL 有效。

- `W2sAuxModel`（in-process）
- `w2s_step`：ΔT、proxy teacher、reverse KL
- 置信度过滤 + 一致性门控
- 局部 + 全局 KL 正则
- CLI：`arle train w2s --student ... --aux1-pre ... --aux1-post ... --aux2-pre ... --aux2-post ...`
- 基线 B0/B1/B2 对比

### Phase 2: Shadow adapter + 原子切换

**目标**：实现训练-评估-切换闭环。

- 双 LoRA adapter slot（serving + shadow）
- `swap_shadow_to_active` / `reset_shadow_from_active`
- 验证集评估 + 统计显著性检验
- 回滚机制（adapter 快照）

### Phase 3: 在线更新流程

**目标**：接入真实流量，在线更新。

- 请求 buffer + 梯度累积
- 定时评估 + 原子切换
- 回归 benchmark 监控 + 自动回滚
- 数据漂移监控（PSI）

### Phase 4: 多 runtime 辅助模型

**目标**：支持大辅助模型，降低显存。

- infer-api 辅助模型（`LoadedInferenceEngine`）
- HTTP API 辅助模型（独立服务）
- 多模型并行 forward

## 7. 关键设计决策

### 7.1 ΔT 计算位置

辅助模型的 pre/post-RL forward 不参与梯度（detach）。ΔT 作为常量输入 proxy teacher。梯度只流过 student。

### 7.2 Proxy teacher 构建

不能复用现有 `TeacherForward`（teacher 看不到 student logits）。在 `w2s_step` 内直接构建：先 student forward，再 aux forward，再组合。

### 7.3 Shadow adapter 实现

复用 `LinearWithLora`，增加 shadow adapter slot。原子切换 = 交换 adapter 权重 tensor（O(1) 指针交换，非拷贝）。

### 7.4 辅助模型放置

- post-RL 辅助模型放 GPU（每步都要 forward）
- pre-RL 辅助模型放 CPU 内存（按需加载，因为 ΔT = post − pre，pre 是固定锚点）

### 7.5 一致性门控的局限

一致性只检验方向一致，不检验方向正确。同构模型的 ΔT 天然一致，需用异构模型或结合置信度过滤。

## 8. 实验复现路径

按文档 W0→W1→W2→W3→W5→W4→W6 顺序：

| 实验 | ARLE 实现 |
|------|----------|
| W0 表现差距诊断 | `arle eval` 对比 base vs instruct |
| W1 有效性 | `arle train w2s` vs B0(no train)/B1(self-train)/B2(pre-RL distill) |
| W2 蒸馏方法 | 方法一(forward KL 最终输出)/方法二(log-ratio)/方法三(proxy+reverse KL) |
| W3 模型多样性 | 同构/异构辅助模型 |
| W5 门控阈值 | 扫描 consistency_threshold ∈ {−∞, 0, 0.3, 0.5} |
| W4 硬标签损失 | 离线加 CE loss |
| W6 组件消融 | 去掉门控/置信度/KL 正则 |
