# infer 原子 Rust 库拆分方案（面向超长 Agent 序列）

## 目标

把当前偏“单大 crate”的 infer，升级为**清晰、可演进、训推一致**的 Rust 工作区组织：
- 超长序列场景保持高吞吐（continuous batching + prefix cache + paged KV）。
- 训练语义与推理语义一致（mixed attention / state transition 可验证）。
- 新模型/新后端接入成本线性，不因代码复制导致熵增。

---

## 一、建议的工作区组织（Atomic Crates）

> 设计原则：**核心语义最小化**、**后端细节外置**、**策略可插拔**、**接口向内收敛**。

```text
agent-infer/
├── Cargo.toml                      # workspace root
├── crates/
│   ├── infer-core/                 # 纯领域层：请求、token、step、错误模型、共享 trait
│   ├── infer-model-api/            # ModelForward / State trait + model capability 描述
│   ├── infer-backend-api/          # Backend trait（设备、stream、buffer、kernel capability）
│   ├── infer-scheduler-core/       # 与后端无关的调度状态机（slot/request lifecycle）
│   ├── infer-policy/               # admission/chunking/eviction/prefix 策略接口与默认实现
│   ├── infer-kv/                   # paged KV 抽象、地址空间、回收与压缩协议
│   ├── infer-observability/        # metrics/event/trace schema（统一事件模型）
│   ├── infer-runtime-cuda/         # CUDA adapter（实现 backend-api）
│   ├── infer-runtime-metal/        # Metal adapter（实现 backend-api）
│   ├── infer-http/                 # OpenAI-compatible API 层（只依赖 façade）
│   ├── infer-engine/               # 组合层：把 model+scheduler+backend 组装为可运行引擎
│   └── infer-cli/                  # CLI/bench 入口
└── infer/                          # 兼容层（过渡期）：re-export + legacy glue
```

---

## 二、依赖方向（必须单向）

```text
infer-core
  ↑
infer-model-api      infer-backend-api      infer-observability
  ↑         \\         ↑                   ↑
  └─────── infer-scheduler-core ─────── infer-policy
                  ↑            \
               infer-kv         \
                    ↑            \
          infer-runtime-cuda   infer-runtime-metal
                    \            /
                     \          /
                      infer-engine
                           ↑
                   infer-http / infer-cli
```

约束：
1. `runtime-*` 不能反向依赖 `http`、`cli`、`scheduler-core` 的实现细节。
2. `scheduler-core` 不允许出现 CUDA/Metal 私有类型。
3. `model-api` 不感知网络协议与后端执行细节。
4. `infer-engine` 是唯一“装配点”；其他 crate 不做跨层偷连。

---

## 三、各原子库职责边界（Do / Don’t）

### 1) infer-core
**Do**：通用 ID、请求上下文、生命周期枚举、通用错误码、共享 Result。  
**Don’t**：任何 GPU、HTTP、模型权重加载逻辑。

### 2) infer-model-api
**Do**：`ModelForward`、`ModelState`、`ModelCaps`（如 max_seq, supports_sliding_window）。  
**Don’t**：调度策略、batch 拼装。

### 3) infer-backend-api
**Do**：设备能力抽象（stream、graph、alloc、kernel dispatch）。  
**Don’t**：具体 CUDA/Metal 调用。

### 4) infer-scheduler-core
**Do**：request/slot 状态机、prefill/decode 编排、公平性与 deadline 框架。  
**Don’t**：显存分配细节、内核 launch 细节。

### 5) infer-policy
**Do**：策略 trait + 默认策略（admission/chunking/eviction/prefix）。  
**Don’t**：直接修改 scheduler 内部字段（仅通过策略输入/输出）。

### 6) infer-kv
**Do**：paged KV 的逻辑页管理、映射、碎片控制、回收协议。  
**Don’t**：绑定具体后端 alloc 接口（通过 backend-api 注入）。

### 7) infer-observability
**Do**：统一事件协议（`enqueued`, `prefill_started`, `decode_step`, `evicted`）。  
**Don’t**：业务逻辑分支。

### 8) infer-runtime-cuda / infer-runtime-metal
**Do**：实现 backend-api + 性能特化（graph capture、kernel tuning）。  
**Don’t**：调度主循环决策。

### 9) infer-engine
**Do**：统一 façade（submit/poll/cancel），组装 model+scheduler+backend。  
**Don’t**：暴露底层后端私有类型给上层。

### 10) infer-http / infer-cli
**Do**：协议转换、参数校验、服务化入口。  
**Don’t**：直接操控 KV、直接调用 runtime 私有接口。

---

## 四、训推一致如何嵌入到分层中

- `infer-model-api` 定义一致性探针接口：层级摘要、token 级摘要。
- `infer-observability` 承载 Golden Trace 事件与 first-diff 报告 schema。
- `infer-engine` 提供 `consistency_mode`（线上关闭、CI/nightly 开启）。
- `runtime-*` 只提供执行，不拥有一致性判定逻辑。

这样可保证：一致性机制是**平台能力**，不是散落在模型实现中的临时代码。

---

## 五、落地迁移顺序（避免大爆炸）

### Phase 1（先拆 API，不搬重逻辑）
1. 抽出 `infer-core` / `infer-model-api` / `infer-backend-api`。
2. 在现有 `infer` 中通过 re-export 保持外部接口不破坏。
3. 建立 crate 级 CI（每个 crate 至少有 compile + unit）。

**完成标准**：编译通过、外部调用零破坏、依赖方向满足单向规则。

### Phase 2（拆调度与策略）
1. 把 scheduler 状态机迁到 `infer-scheduler-core`。
2. 把 admission/chunking/eviction 迁到 `infer-policy`。
3. prefix cache / paged KV 协议迁到 `infer-kv`。

**完成标准**：新增策略无需改 scheduler 主循环。

### Phase 3（拆后端与装配层）
1. CUDA/Metal 分别进入 `infer-runtime-cuda` / `infer-runtime-metal`。
2. `infer-engine` 成为唯一装配层。
3. `infer-http` / `infer-cli` 只依赖 façade。

**完成标准**：新增后端时不改 scheduler-core 与 http。

---

## 六、仓库治理规则（防止再次长成单体）

1. 新增模块前先判断能否归入现有原子库；不能才新建 crate。
2. 跨 crate 调用必须通过公开 trait，不允许直接 import 私有实现。
3. `cargo udeps` + `cargo hakari`（或等价）定期清理依赖污染。
4. 每次 PR 必填“影响层级”与“是否破坏依赖方向”。
5. 任何 `runtime-*` 到 `http/cli` 的反向依赖直接拒绝。

---

## 七、达到“合理 Rust 仓库组织”的验收标准

- **结构清晰**：看目录就能判断职责，不需要读 10 个文件猜边界。
- **依赖健康**：无循环依赖、无跨层偷连、无后端细节上渗。
- **性能可控**：优化主要发生在 `runtime-*` 与 `kv/policy`，不污染领域层。
- **一致性可测**：CI/nightly 能产出 first-diff 与一致性门禁报告。
- **演进可持续**：新增模型/后端以“加 crate/加 adapter”为主，而非改核心主干。

> 这是“原子化 + 装配化 + 策略化”的组织方式：保持 infer 在高性能前提下长期可维护。
