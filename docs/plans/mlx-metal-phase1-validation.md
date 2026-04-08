# Plan: MLX Metal Phase 1 统一验证方案

> 适用范围：`docs/projects/mlx-backend-roadmap.md` 阶段 1 的首批落地代码
> 目标：先把基础设施验证做实，再谈后续接线到 `MetalBackend`
> 创建：2026-04-08

---

## 1. 验证范围

本方案只验证三类内容：

1. `MetalKVPool` 的元数据层行为是否正确
2. `MetalScheduler` 的纯 CPU 调度骨架是否符合阶段 1 设计
3. 在尚未接线到真实 Metal 推理热路径之前，基础设施是否已经达到可继续集成的门槛

明确不在本轮验证范围内的内容：

1. 真实 `mlx-rs` / Metal kernel 性能
2. `MetalBackend` 的完整 forward / decode 接线
3. 自定义 fused shader、Flash Decoding、KV 量化等阶段 2 内容
4. 模型覆盖、自动下载、REPL 体验等阶段 3/4 内容

当前验证的核心原则是：**先验证调度和资源账本正确，再验证 GPU 热路径**。如果基础设施账本都不稳定，后续接线只会把问题放大。

---

## 2. 测试矩阵

### 2.1 `MetalKVPool` 元数据层

| 场景 | 预期结果 | 验证重点 |
|------|----------|----------|
| 初始化池 | 空闲 token 数 = `max_total_tokens` | 配置字段、容量统计 |
| 连续分配 | 分配顺序稳定，`available_tokens` 递减 | LIFO 空闲栈是否工作 |
| 释放请求 | slot 归还后可再次分配 | 回收路径是否完整 |
| 重复释放 | 不崩溃，不重复回收 | 幂等性 / 边界处理 |
| 前缀共享 | 多个请求可引用同一批物理 slot | refcount / shared prefix 账本 |
| 共享后释放 | 只有最后一个引用释放时才真正回收 | 引用计数正确性 |
| 超额分配 | 返回错误，不破坏已有状态 | 容量保护 |

### 2.2 `MetalScheduler` 纯 CPU 骨架

| 场景 | 预期结果 | 验证重点 |
|------|----------|----------|
| 队列提交 | 请求进入 waiting 队列 | `submit` / 入队逻辑 |
| decode 优先 | active decode 存在时先调度 decode | 阶段 1 的优先级规则 |
| chunked prefill | 大 prompt 被切成固定 chunk | `prefill_chunk_size` 行为 |
| decode + prefill 交错 | 同一轮步进可同时返回 decode / prefill | continuous batching 骨架 |
| 生命周期迁移 | waiting -> active -> finished | 状态机是否闭环 |
| 资源上限 | 达到上限时拒绝或回退 | 账本不越界 |
| preemption 预留位 | 仅验证 CPU 决策，不验证 GPU swap | 接线前边界 |

### 2.3 接线前门槛

| 门槛 | 通过标准 |
|------|----------|
| API 稳定 | `MetalKVPool` 和 `MetalScheduler` 的公开方法足够表达阶段 1 需求 |
| 状态可解释 | 所有关键状态都能从 CPU 层读出并断言 |
| 测试可重复 | 同一组测试在 CI 和本地都可重复跑 |
| 不依赖 GPU | 基础设施测试能在 `no-cuda` 环境下执行 |

---

## 3. 命令

### 3.1 CPU-only 基础验证

```bash
cargo test --manifest-path infer/Cargo.toml --no-default-features --features no-cuda
```

用途：

1. 验证 `MetalKVPool` 的纯 Rust 元数据逻辑
2. 验证 `MetalScheduler` 的纯 CPU 调度单测
3. 验证当前仓库在无 CUDA / 无 Metal 环境下仍可编译和测试基础设施

### 3.2 Metal 依赖编译验证

```bash
cargo test --manifest-path infer/Cargo.toml --no-default-features --features metal,no-cuda
```

用途：

1. 验证 `metal` feature 下新增类型和条件编译没有断裂
2. 验证 `MetalKVPool` / `MetalScheduler` 在 Apple Silicon 环境中的编译面
3. 为后续接 `MetalBackend` 热路径提前暴露编译错误

### 3.3 精准回归验证

```bash
cargo test --manifest-path infer/Cargo.toml --no-default-features --features no-cuda scheduler::tests
cargo test --manifest-path infer/Cargo.toml --no-default-features --features no-cuda metal_kv_pool
```

用途：

1. 在调度器或 KV pool 改动后，快速跑对应回归集
2. 缩短开发迭代反馈时间

---

## 4. 通过标准

本阶段的通过标准分成三层：

1. 功能正确：所有新增单测通过，且旧的 scheduler / prefix cache / block manager 测试不回退
2. 编译正确：`no-cuda` 和 `metal,no-cuda` 两条构建路径都能通过
3. 设计正确：`MetalKVPool` 能表达共享前缀账本，`MetalScheduler` 能表达 decode 优先和 chunked prefill 的调度顺序

更具体一点：

1. `MetalKVPool` 不能在共享前缀和释放路径上出现“重复归还”或“提前归还”
2. `MetalScheduler` 不能把 decode 请求饿死在长 prompt prefill 后面
3. chunked prefill 的边界必须和配置一致，不能出现 off-by-one
4. CPU 层状态必须可观测，测试可以直接断言，不依赖日志

---

## 5. 暂不验证项

这轮不验证以下内容：

1. 真实 Metal GPU throughput
2. `MetalBackend` 真正的 prefill / decode 计算正确性
3. `MetalKVPool` 的 scatter/gather 读写性能
4. prefix cache 和 MetalKVPool 的物理接线
5. 多请求并发下的端到端 TTFT / tok/s
6. 任何阶段 2 的 fused kernel、Flash Decoding、KV 量化

原因很直接：这些都依赖后续把基础设施接到热路径上。现在先把账本和调度机理验证干净，后面才能做性能归因。

---

## 6. 接线后补充验证

当 `MetalKVPool` 和 `MetalScheduler` 接入 `MetalBackend` 后，再补下面这些验证：

1. 单请求回归：与当前 Metal 单请求路径结果一致
2. 多请求回归：请求 B 的 TTFT 不能随 A 的长生成线性增长
3. prefix cache 回归：共享 system prompt 时，第二次请求应明显减少 prefill 成本
4. 资源回收回归：请求结束后 slot 必须完整归还，不能泄漏
5. 并发压力回归：连续提交多个短请求和长请求时，调度器不能卡死或饥饿
6. 性能基线回归：接线前后分别记录 TTFT、prompt TPS、generation TPS，作为后续优化起点

---

## 7. 执行顺序建议

1. 先跑 `no-cuda` 单测，确认纯 Rust 基础设施稳定
2. 再跑 `metal,no-cuda` 编译验证，确认条件编译没有把 Metal 相关符号弄坏
3. 最后只在接线完成后补端到端验证，不把 GPU 热路径问题提前掺进基础设施阶段

