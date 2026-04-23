# 2026-04-21 SGLang 控制面/执行面对齐重构计划

> Status (2026-04-23): **Historical architecture precursor.** The `ScheduleBatch -> ForwardBatch` direction in this file is still useful, but the current decode-alignment truth and execution order now live in [`2026-04-23-cuda-decode-sglang-alignment.md`](2026-04-23-cuda-decode-sglang-alignment.md).

## 目标

- 把当前 CUDA 调度路径重构为明确的两层语义：
  - 控制面：`ScheduleBatch`（请求状态、KV 复用、admission、chunk/mixed 决策）
  - 执行面：`ForwardBatch`（GPU 前向所需低层输入，供模型执行）
- 统一成单一 tick 主路径，删除并行旧路径和中间态分叉。
- 对齐 SGLang 的 phase-aware 思路：prefill/decode/mixed 的调度与执行分层明确、可观测、可验证。

## 非目标

- 本轮不实现独立 PD 集群部署（prefill 节点/ decode 节点网络拆分）。
- 本轮不引入新 kernel backend（如 TRTLLM/FA3/NSA 新后端接入）。
- 本轮不改 HTTP 协议语义，仅改调度与执行编排。

## 背景问题

- 当前代码虽已有 `plan_step()`，但控制面对象与执行面对象未显式分层，执行入口仍以函数分支驱动，语义边界不够清晰。
- 高并发下问题定位成本高：调度决策、lowering、执行失败在日志中耦合，不易快速归因到 admission / KV / kernel / sampling 任一环节。
- 与 SGLang 的核心实践相比，缺少“ScheduleBatch -> ForwardBatch”的一等结构体语义。

## 目标架构图

```text
HTTP/OpenAI API
   ↓
Tokenizer + Request parser
   ↓
Req (waiting / running / finished)
   ↓
Scheduler Control Plane
   ├─ waiting queue + running decode set
   ├─ radix prefix match + reusable-slot / direct-attach
   ├─ KV alloc / demote / evict / chunk budget
   ├─ mixed eligibility + admission
   └─ build ScheduleBatch
   ↓
Lowering Layer
   └─ ScheduleBatch -> ForwardBatch
      (decode tokens/slots, prefill slices, metadata plan)
   ↓
Execution Plane (ModelForward)
   ├─ prefill / decode / mixed dispatch
   ├─ attention + linear/GEMM + KV ops + sampling
   ├─ optional CUDA Graph path
   └─ returns tokens/logprobs + prefill completion marks
   ↓
Scheduler Reconcile
   ├─ token append + stop/finish
   ├─ KV commit/release/cache publish
   └─ stream delta emit
```

## 工作分解

### W1. 调度主循环分层（控制面）

- [ ] 新增 `ScheduleBatch` 一等对象（替代隐式 `StepPlan` 分支语义）。
- [ ] `plan_step` 重命名/重构为 `plan_schedule_batch`，只做控制面工作。
- [ ] 明确 `ScheduleBatch` 字段：
  - [ ] phase: `Idle | DecodeOnly | Mixed | PrefillOnly`
  - [ ] admitted prefill candidates
  - [ ] 预算与约束快照（必要字段）

### W2. Lowering 层（ScheduleBatch -> ForwardBatch）

- [ ] 新增 `ForwardBatch` 一等对象，作为执行面唯一输入。
- [ ] 实现 `lower_schedule_batch`：
  - [ ] 解析 decode token/slot
  - [ ] 解析 prefill slices
  - [ ] 执行前 metadata 准备（保持行为不变）
- [ ] 删除主循环中的散落分支，统一由 `ForwardBatch` 分发执行。

### W3. 执行入口统一（执行面）

- [ ] 保留现有 kernel 调用细节（先不换 kernel），只统一入口路径。
- [ ] 单点执行入口：
  - [ ] `execute_forward_batch(forward_batch)`
  - [ ] decode launch + optional mixed + readback
  - [ ] prefill-only 批次执行
- [ ] 保证 pending/readback 生命周期仍为单一状态机。

### W4. Phase-aware 算子可观测性

- [ ] 为 prefill/decode/mixed 增加统一 execution label（日志 + stats）。
- [ ] 增加“算子类别耗时”基础指标位：
  - [ ] attention
  - [ ] linear/GEMM
  - [ ] KV metadata/ops
  - [ ] sampling/readback
- [ ] 不改变算子实现，仅先打通观测面。

### W5. 删除式清理

- [ ] 删除被 `ScheduleBatch/ForwardBatch` 取代的中间态分支与重复入口。
- [ ] 删除死代码、重复 helper 和仅旧路径使用的状态字段。
- [ ] 更新 `infer/src/scheduler/AGENTS.md` 对应不变量说明。

### W6. 文档与后端复用边界

- [ ] 新增/更新架构文档，说明该分层如何复用于 Metal/CPU：
  - [ ] 控制面（ScheduleBatch）可复用
  - [ ] Lowering 需要按后端实现
  - [ ] 执行面（ForwardBatch -> kernels）后端特化
- [ ] 明确“可复用 vs 不可复用”边界，避免 backend cfg 泄漏。

## 验证与基线

### 构建/测试

- [ ] `cargo fmt --all`
- [ ] `cargo test -p infer --release --lib scheduler:: -- --nocapture`
- [ ] `cargo check -p infer --no-default-features --features cuda,no-cuda`

### 运行验证

- [ ] c16 streaming smoke（`16 slots / 4608 seq / 4096-in`）：
  - [ ] `4096-in / 64-out`
  - [ ] `4096-in / 256-out`
  - [ ] 确认无 `CUDA_ERROR_ILLEGAL_ADDRESS`
  - [ ] 确认无异常 0-token 提前完成（非 client cancel 场景）

### Benchmark

- [ ] `scripts/bench_guidellm.sh <label> --fast ...` 回归检查
- [ ] 输出 TTFT/ITL/out tok/s 与请求完成率
- [ ] 新增 wins 条目（含命令、环境、结果、问题、学习）

## 验收标准

- [ ] 代码层已形成显式 `ScheduleBatch -> ForwardBatch` 分层。
- [ ] `step()` 主循环只保留单一计划/降级/执行路径。
- [ ] 不出现旧路径并行保留（no half-states）。
- [ ] c16 smoke 通过，bench 指标可解释且无统计短路（TTFT/ITL 非 0）。
- [ ] 文档与 wins 落盘，commit/push 到 `main`。

## 风险与回滚策略

- 风险 1：重构后 mixed 路径隐性退化为 decode-only。  
  应对：在 `ForwardBatch` 层加 phase 断言与日志计数。

- 风险 2：lowering 过程中 metadata 复用条件变化导致 graph 抖动。  
  应对：保留原 metadata 更新逻辑，不在本轮改 graph capture 条件。

- 风险 3：高并发下 KV demote/evict 行为变化引发吞吐倒退。  
  应对：bench 结果对比前一条 wins；若倒退超阈值，回滚本轮行为改动，仅保留结构重构。

## 里程碑顺序

1. 先做 W1+W2（结构落地，不改语义）。  
2. 再做 W3+W5（删除旧路径，统一执行入口）。  
3. 最后做 W4+W6 + 验证/bench/wins。  
