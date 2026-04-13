# infer 原子 Rust 库拆分方案（面向超长 Agent 序列）

## 目标

把当前偏“单大 crate”的 infer，升级为**清晰、可演进、训推一致**的 Rust 工作区组织：
- 超长序列场景保持高吞吐（continuous batching + prefix cache + paged KV）。
- 训练语义与推理语义一致（mixed attention / state transition 可验证）。
- 新模型/新后端接入成本线性，不因代码复制导致熵增。

当前仓库实际处在 **Phase 1 的收尾/稳定阶段**：控制面 crate 已经开始拆出，但 scheduler / KV / runtime 的原子化还没有进入主拆分。

在重新对照 Tokio、Tracing、Serde、rust-analyzer、Cargo、Bevy 这些顶级 Rust 工作区之后，方案需要一个重要纠偏：

- 不按“抽象名词”预建 crate，而按 **稳定边界、复用证明、发布/测试面、进程边界** 拆。
- `core` / `api` crate 只有在 **至少两个直接消费者** 需要更强稳定承诺时才成立；这更接近 `tracing` / `tracing-core`，而不是先把所有层都拆空。
- 像 rust-analyzer 那样的大量内部 crate，只在 **架构不变量已经非常清晰** 时才值得引入；否则会先引入维护成本，再引入边界收益。
- 像 Tokio、Serde、Cargo 那样，workspace 里的 crate 更多是 **可独立使用/发布/测试** 的单元，而不是每个内部概念都独立成包。
- 像 Bevy 一样，很多“组合能力”优先通过 **feature/profile** 收拢，再决定哪些边界最终沉淀成独立 crate。

因此，下面不再把“未来所有可能存在的 crate”当作默认终局，而是把 **当前推荐拓扑** 和 **后续候选拆分** 分开描述。

---

## 一、当前推荐工作区组织（Near-term Workspace）

> 设计原则：**先拆稳定边界，再拆内部层；先证明复用，再沉淀 API**。

```text
agent-infer/
├── Cargo.toml                      # workspace root / shared metadata
├── infer/                          # 运行时兼容层与当前数据面主体
│   ├── scheduler / kv / backend / http / bootstrap
│   └── feature-gated CUDA / Metal / CPU paths
├── crates/
│   ├── infer-core/                 # 小而稳定的共享领域类型
│   ├── infer-policy/               # scheduler/runtime 可复用的策略接口
│   ├── infer-observability/        # 统一事件 schema + sink trait
│   ├── infer-chat/                 # 协议类型
│   ├── infer-tools/                # builtin tools + shared tool policy hooks
│   ├── infer-agent/                # agent loop（不拥有 UI / runtime 私有细节）
│   ├── infer-cli/                  # CLI / REPL / slash commands
│   └── infer-engine/               # 运行时装配边界 / model discovery / backend loading
└── xtask/ or tools/                # 可选；放 Rust 自动化而不是 shell glue
```

这个布局更接近 Tokio / Cargo / Serde 的拆法：只有那些已经形成稳定消费面的部分才单独成 crate；真正仍在高速迭代的 runtime 内核先留在 `infer/`。

### 候选后续 crate（满足条件才拆）

- `infer-scheduler-core`
  - 只有当 CUDA / Metal / CPU 至少两条路径共享同一套状态机、并且 golden tests 可以跨后端复用时再拆。
- `infer-kv`
  - 只有当 paged KV 协议被多个 runtime/backend 共享，而不只是当前 `infer/` 内部细节时再拆。
- `infer-runtime-api`
  - 只有当 “runtime trait + state trait + capability 描述” 已经被多个具体 backend 和至少一个上层装配点稳定消费时再拆。
- `infer-runtime-cuda` / `infer-runtime-metal`
  - 只有当 backend 适配层已经从 `infer/` 中自然长出来，而不是为了“看起来分层”硬切时再拆。
- `infer-http`
  - 只有当 HTTP server 需要被独立复用/发布/测试，而不是当前 `infer/` 的一部分时再拆。

---

## 二、当前推荐依赖方向（必须单向）

```text
infer-core      infer-policy      infer-observability      infer-chat
    ↑                ↑                    ↑                   ↑
    └──── infer-tools / infer-agent / infer-engine ───────────┘
                           ↑
                        infer-cli
                           ↑
                      agent-infer

infer/  <---- 当前仍然承载 scheduler / kv / backend / http / bootstrap
  ↑
infer-engine  只依赖 infer 的运行时装配边界，不反向依赖控制面
```

约束：
1. `infer-engine` 不得反向依赖 `infer-agent` / `infer-cli`。
2. `infer-agent` 不得拥有 UI 输出，也不直接拥有 runtime/backend 私有细节。
3. `infer-tools` / `infer-chat` 提供共享协议与默认语义，但不反向感知 CLI。
4. 只有当 runtime 内部边界被两个以上实现共同消费时，才允许从 `infer/` 中拆出新 crate。

---

## 三、各原子库职责边界（Do / Don’t）

### 1) infer-core
**Do**：通用 ID、请求上下文、生命周期枚举、通用错误码、共享 Result。  
**Don’t**：任何 GPU、HTTP、模型权重加载逻辑。

### 2) infer-policy
**Do**：策略 trait + 默认策略（admission/chunking/eviction/prefix）。  
**Don’t**：直接修改 scheduler 内部字段（仅通过策略输入/输出）。

### 3) infer-observability
**Do**：统一事件协议（`enqueued`, `prefill_started`, `decode_step`, `evicted`）。  
**Don’t**：业务逻辑分支。

### 4) infer-engine
**Do**：统一 façade（submit/poll/cancel），组装 model+scheduler+backend。  
**Don’t**：暴露底层后端私有类型给上层。

### 5) infer-agent / infer-cli
**Do**：控制面回合循环、协议转换、参数校验、服务化/交互入口。
**Don’t**：直接操控 scheduler / KV / backend 私有实现。

### 6) infer（当前 runtime 主体）
**Do**：继续承载 scheduler、KV、backend、http、bootstrap 的主逻辑，直到共享不变量被证明。
**Don’t**：继续向控制面反向长依赖。

---

## 四、后续阶段：训推一致如何嵌入到分层中

> 这一节描述的是后续阶段希望落地的能力，不代表当前 Phase 1 已经具备独立的一致性接口 crate、`consistency_mode` 或 Golden Trace 管线。

- 如后续确实需要训练/推理一致性探针，再抽一个更窄的 model-facing probe boundary，负责层级摘要、token 级摘要。
- `infer-observability` 承载 Golden Trace 事件与 first-diff 报告 schema。
- `infer-engine` 提供 `consistency_mode`（线上关闭、CI/nightly 开启）。
- 具体 runtime/backend 只提供执行，不拥有一致性判定逻辑。

这样可保证：一致性机制是**平台能力**，不是散落在模型实现中的临时代码。

---

## 五、落地迁移顺序（避免大爆炸）

### Phase 1（当前实际范围：控制面拆分与边界收敛）
1. 保持已拆出的控制面 crate 可编译、可测试、可回退：`infer-agent`、`infer-cli`、`infer-chat`、`infer-tools`、`infer-core`、`infer-engine`、`infer-observability`、`infer-policy`。
2. 让 `infer-engine` 只保留运行时装配与后端适配边界，不再反向依赖控制面实现。
3. 维持 `infer` 的兼容层职责，先不把 scheduler / KV / runtime 的主逻辑继续拆散。
4. 把 observability、policy 先收敛成稳定边界，后续再接入更深的 runtime 语义。

**当前未完成**：
- 训练/推理一致性探针边界还没有收敛到足够稳定，因此还不该单独成 crate。
- scheduler、KV、runtime-* 的原子化仍在后续阶段。
- observability 目前已经在 batch/Metal 路径收敛为 action-oriented 事件边界；CUDA runtime 的同级事件 sink 仍在后续阶段，更深的 trace / consistency 语义也还没继续下沉。
- policy API 已经包含 admission + chunking 边界，且当前 batch/CUDA/Metal 路径已经消费 enqueue admission 与 decode-aware chunking；更深的 eviction / KV wiring 仍在后续阶段。

### Phase 2（证明运行时内部共享不变量）
1. 先让 CUDA / Metal / CPU 的 chunking、admission、event 语义尽量收敛到同一套 policy/observability contract。
2. 为 scheduler / KV / backend 边界补跨实现 golden tests，而不是先新建 crate。
3. 只有当共享状态机被证明稳定时，再考虑抽 `infer-scheduler-core`。

**完成标准**：新增/修改策略主要改 policy 层与测试，而不是到处散改 runtime 主循环。

### Phase 3（按复用证明继续拆 runtime）
1. 当 scheduler 状态机跨 backend 共享时，再抽 `infer-scheduler-core`。
2. 当 paged KV 协议被多个 runtime/backend 共同消费时，再抽 `infer-kv`。
3. 当 backend trait 真正稳定时，再决定是否抽 `infer-runtime-api` 与 `infer-runtime-*`。

**完成标准**：新增 backend/变体时，主要新增 adapter 或 feature，不需要回改控制面与共享 contract。

### Phase 4（可选：独立 HTTP / 发布面）
1. 只有当 HTTP server 需要独立发布或独立复用时，才从 `infer/` 中抽 `infer-http`。
2. 只有当外部消费者真的需要更稳定的 runtime API 时，才进一步压缩并公开这些边界。

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
