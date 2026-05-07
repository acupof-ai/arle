# ARLE 战略 pivot — Rust-native Coding/Agent 推理 runtime

> 2026-05-07 战略全局分析(3 pass review 后定稿)。用户提示:服务核心目标是
> coding & 通用 agent。重新定义 ARLE 的 product / market / moat / 优先级。

## 0. 核心论断(一句话)

**ARLE 是双产品双第一要义**:
- **推理侧 = Rust-native coding/agent runtime**(支持 Cursor / Claude Code /
  Aider / Continue),**moat = 5 项 capability 组合**(2 ✓ + 3 plan)。
- **训练侧 = DSV4 架构 from-scratch repro**(`dsv4-small-repro.md`,~1.05B
  param dsv4-mini,16GiB 单卡,Muon+AdamW,FP8 dense + BF16 master)。

**两侧战略独立,工程互补**:推理侧 MoE 服务(Qwen3.5-MoE / DeepSeek)和训练侧
DSV4 共用 mlx-sys / cuda-kernels MoE 路径。

**本文档专注推理侧**;训练侧详见 [`dsv4-small-repro.md`](../plans/dsv4-small-repro.md)。

---

**推理侧核心论断**:**ARLE 不是 generic LLM 服务,是 Rust-native AI 开发工具推
理 runtime**(支持 Cursor / Claude Code / Aider / Continue 这类 coding agent
在 Linux 服务器和 Apple Silicon Mac 上跑)。**Defensible moat 是 5 项 capability
的组合**(其中 2 项已具备,3 项待 land);**单点最强**不是策略目标。**Long-ctx
prefill TTFT 是当前 binding constraint**(agent multi-turn loop 复合受影响);
**Prefill CUDA Graph capture 必要但不充分**,需与 TileLang HD128 / FP8 paged KV
组合才能闭合 4k 缺口。

## 1. Workload reframing — 谁是真用户

### 1.1 现"world #1 lead at 4 canonical shapes" 的隐含假设

`M_world1-30-percent-lead-roadmap.md` 用 4 个 canonical workloads
对标 vLLM/SGLang/TRT-LLM,假设的目标用户是 generic LLM 服务运营商(批量服务
匿名用户、混合 prompt 形态)。

### 1.2 Coding/Agent workload 的真实形态

每 request 形态:
- **输入** 5k-32k tokens(system prompt + 文件内容 + 历史 + 工具结果)
- **输出** 50-2000 tokens(JSON tool call 或代码)
- **并发** c=1-8 典型(单用户 agent loop,**不是**多租户 c=64 批服务)
- **延迟** TTFT 在 multi-turn 复合;一个 session 10-50 turn,每轮 TTFT 都吃
- **复用** system prompt 跨 turn 不变 → prefix cache 命中率 80%+
- **结构** tool call JSON 占输出 70%+ → grammar 约束生成
- **解码** 输出短(50-500 tok),speculative decoding 杠杆大

### 1.3 部署模式分两类(优化目标不同)

| 模式 | 例子 | 主要优化 | ARLE 当前 |
|---|---|---|---|
| **本地单用户** | Cursor 本地 / Claude Code Mac | 单 request TTFT + ITL p99 | Metal backend(mlx-sys),CUDA 16GB 卡 |
| **服务多租户** | Cursor cloud / 自托管 API | 综合 throughput + p99 + 成本 | CUDA 服务 |

ARLE 同时服务两类 → 必须 Metal + CUDA 双 backend 都强。当前战略文档仅讨论
CUDA gap,**Metal backend 维度需要单独评估**(本文档外)。

### 1.4 Workload-相关性重排 4 canonical shapes

| Canonical shape | Coding/Agent 相关度 | 备注 |
|---|---|---|
| high-conc 1k/256/c=64 | LOW | agent 不批服务 64 路 |
| **long-ctx 4k/c=4** | **HIGHEST** | 典型小代码库 agent turn |
| **long-ctx 8k/c=4** | **HIGH** | 中等代码库 + 历史 |
| multi-tenant prefix | MEDIUM-HIGH | system prompt 复用,接近真实但 c 值仍偏批 |

**未覆盖的 agent 真相**:
- **multi-turn 同 system-prompt** session(10× turn,每轮 1k user query)
- **32k-128k 大代码库**(Claude Code 级)
- **结构化输出 JSON tool call** workload(grammar 约束)
- **speculative-decode-friendly** 短输出(50-500 tok)

## 2. ARLE 现状(workload-重排后)

### 2.1 Bench rank by shape(再加权)

| Shape | ARLE | #2 | Δ vs #2 | Coding/agent 权重 | 净评估 |
|---|---:|---:|---:|---|---|
| high-conc 1k/256/c=64 | #1 (843 tok/s) | SGLang 499 | **+69%** | LOW | 浪费 lead |
| **long-ctx 4k/c=4** | #3 (TTFT 1976) | SGLang 973 | **−51%** | **HIGHEST** | **关键 gap** |
| **long-ctx 8k/c=4** | #2 (tok/s 持平,TTFT) | vLLM 2362 ms | TTFT −48% | **HIGH** | gap |
| multi-tenant prefix | #1 (TTFT 318ms) | vLLM 573 | +80% | MEDIUM-HIGH | hold |

**真相**:**ARLE 的现 lead 不在 agent 痛点上,落后的 shape 才是**。

### 2.2 架构资产盘点

#### 已具备且对路 ✓

| Capability | 文件位置 | Coding/agent 价值 |
|---|---|---|
| Continuous batching scheduler | `infer/src/scheduler/cuda/` | 多 request 复用 |
| RadixCache prefix cache | `infer/src/kv_tier/` | system prompt 跨 turn 复用 |
| Decode CUDA Graph capture(B=1..8) | `infer/src/model/qwen3/batch_decode.rs:1703` | 短输出(tool call)latency |
| Paged KV + FP8 | `infer/src/kv_tier/`,`crates/cuda-kernels/csrc/quant/` | 长上下文内存效率 |
| TileLang HD128 attention(custom) | `crates/cuda-kernels/tools/tilelang/` | 自定义 kernel,无 Triton/CUTLASS 依赖 |
| Rust hot path(整 stack) | `infer/`,`crates/` | 无 Python 解释器 overhead |
| Metal backend(Apple Silicon) | `infer/src/backend/metal/`,`crates/mlx-sys/` | 本地 Mac coding/agent |
| Qwen3.5 / MoE substrate | `crates/qwen35-spec/`,`crates/deepseek-spec/` | MoE coding 模型支持 |

#### 缺口(对 coding/agent 而言)

| 缺口 | 影响 workload | 当前状态 |
|---|---|---|
| **Prefill CUDA Graph capture** | long-ctx TTFT(每 turn 都吃)| 已 plan(commit `939008f`,Phase 0 ~200 LOC) |
| **Speculative decoding (EAGLE/Medusa)** | tool call 短输出 tok/s × 2-3(条件:acceptance ≥70%) | 未实现,Qwen3.5 spec scope 在 roadmap |
| **Grammar 约束生成** | tool call JSON 强制有效 | 未实现 |
| **32k+ long-ctx 验证** | Claude Code class | 未 benched |
| **Tool call 快路径** | 结构化短输出 latency | 通用 streaming 不够优 |
| **agent-aligned bench harness** | 验证以上改动是否真对 agent 有效 | 未实现 |

#### 已 KILLED — 不要重做

| 实验 | 原因 |
|---|---|
| M_pf-gemm Phase 0(cuBLAS algo 选择) | top-1 已最优,−1.3% 在噪声内 |
| M_pf-fuse Phase 0(gate-up fusion) | +1.5% regression(cuBLAS 在 22k N 选 algo 反而吃亏) |
| M_b.2.2 split-KV BF16(opt-in) | bench regression(ITL +31.6%, tok/s -18.8%)+ e2e hang 33m |

## 3. Defensible moat — 5 项组合(forward-looking)

ARLE **现具备 2 项**,**plan 中 3 项**;**没有任何竞品同时具备这 5 项**:

| Capability | ARLE | SGLang | vLLM | TRT-LLM |
|---|:-:|:-:|:-:|:-:|
| **Rust hot path**(整 stack) | ✓ | ✗ Python | ✗ Python | C++ engine,Python frontend |
| **Custom TileLang HD128 prefill+decode attention** | ✓ | (Triton) | (Triton/CUDA) | (CUTLASS) |
| Piecewise Prefill CUDA Graph | **plan**(`939008f`) | ✓ default ON | partial(piecewise_cuda_graph) | partial(engine 编译) |
| Speculative decoding(EAGLE/Medusa) | **plan** | ✓ | ✓ | ✓ |
| Grammar 约束(xgrammar 等) | **plan** | ✓ xgrammar | ✓ xgrammar | partial |

### 3.1 Moat 真相 — 不是单点最强是组合

- 单点比 → SGLang 也强(piecewise prefill graph + xgrammar + spec decode 都默认开)
- ARLE **独家**:**Rust hot path + TileLang custom kernel** 这两项 + 上面 3 项 plan
- 5 项全 land 后 → ARLE 是**唯一一个 Rust-native + 自定义 attention kernel + piecewise prefill graph + spec decode + grammar** 的 stack

### 3.2 但要诚实 — 5 项中 3 项还在 plan

如果这 3 项不 land,moat 就是 paper moat。Forward-looking moat ≠ 现 moat。
执行风险评估:

| Plan | 风险 | 替代 |
|---|---|---|
| Prefill graph(`939008f`)| 中(scope 已限,kill criteria 7 条)| 无 |
| Spec decode(EAGLE)| 高(draft model 数据/训练依赖,kernel + 调度复杂)| 用 Medusa 多头 simpler |
| xgrammar(Rust 重写)| **高**(~1000+ LOC FSM)| **FFI 包 xgrammar C++**(降风险) |

## 4. ROI 真相 — Prefill Graph 单点不够

### 4.1 Codex 3:1 plan 给的诚实数学

```
Launch floor saved: 36 layers × 7 ops/layer × 2 chunks × 7.5us = 3.8 ms
ARLE 4k TTFT 1976ms → 1969-1972ms (only)
SGLang 4k TTFT = 973ms
SGLang gap to ARLE = 1003 ms
Graph capture 解释: 3.8 ms / 1003 ms = 0.38%
```

**Graph capture 单点解释 < 1% of SGLang lead**。其余 ~999ms 来自:
- Cuda graph 还消除 host-side dispatch / cudarc 调用 / event 流量 / 动态调度 — 但 ARLE Rust hot path **本来就少这些 overhead**(对比 Python),所以 ARLE 受益小于 SGLang
- FlashInfer paged prefill kernel vs ARLE TileLang HD128 — 算法性能差异
- Attention 实现(FA2/FA3 vs TileLang HD128)
- KV layout / page boundary / 调度 overhead

### 4.2 Phase 0 license-or-kill 阈值(per codex plan)

- **<10ms TTFT 改善 + nsys 显示无 launch overhead 减少** → KILL Phase 0
- **1850-1950ms** → license Phase 1(piecewise 42 buckets)
- **<1700ms** → 立即 promote
- **graph + TileLang/FP8 不能 < 1500ms** → KILL M_world1 4k 闭合 claim

### 4.3 真闭合 4k gap 的路径

如果 Phase 0 数据显示 graph 单点不够(预期),**真路径是组合**:
1. Phase 0 graph capture(removes launch + scheduling overhead)
2. Phase 2 + TileLang custom prefill GEMM(per `M_pf-gemm` Phase 2.5)
3. Phase 2 + FP8 paged KV in prefill(已具备 substrate)
4. 可能还需要研究 SGLang FlashInfer 对比 TileLang HD128 在 prefill 的 kernel 时间差

**这意味着 M_pf-gemm Phase 2 不应当被 demote**,只是要在 graph capture **之后**评估。

## 5. 主线路径 — P-priority 排序(SOLID 版)

### P0(本 sprint,sequence-critical)

**P0.0 — Agent-aligned bench harness**(必须先做)

- Custom Python runner(guidellm 不直接支持 multi-turn)
- 三个 agent shapes:
  1. **8k-system + 1k-user → 200-tok JSON tool call**(典型 agent turn)
  2. **multi-turn 10× 同 system-prompt + 增量 user query**(prefix cache 复用)
  3. **32k-codebase + 2k 代码生成输出**(Claude Code class)
- Acceptance: 三个 shape 都对 SGLang / vLLM / ARLE 跑通,建立 baseline 表
- 没这个 harness → P0.1 落地后**不知道是否对 agent 真有效**
- LOC 估:200-300 Python(custom runner)+ docs
- Owner: Claude(细节)+ general-purpose 助攻
- Trigger: 立即(独立于 GPU bench,可纯 CPU)

**P0.1 — Prefill CUDA Graph capture Phase 0**(继续 codex 3:1 plan `939008f`)

- 已 plan,scope ~200 LOC,opt-in `INFER_PREFILL_GRAPH=1`
- 单 bucket 2048-token,只 capture body(GPU-only),其余 prepare/finish 留 eager
- Bench:4k/c=4 + agent shapes(P0.0 提供)
- License threshold: 1850-1950ms 进 Phase 1; <1700ms 立即 promote
- Phase 1: 42 buckets piecewise; Phase 2: + TileLang HD128 + FP8 paged KV
- Owner: codex (impl) + Claude (review/bench/wins)
- Trigger: P0.0 harness 跑通 + 可以双 stack 对比验证

### P1(下一 sprint)

**P1.1 — Speculative decoding(Medusa-multi-head 优先,EAGLE 后置)**

- **Medusa**(单模型多头 draft)优先于 EAGLE(独立 draft 模型),risk 更低
- 复用 ARLE Rust runtime + TileLang verify kernel
- 目标:tool call 短输出(50-500 tok)tok/s × 2-3(条件:acceptance ≥70%)
- ROI:agent tool call 占输出 70%+,speedup 直接体现
- 风险点:acceptance rate 不到 70% 时 ROI 大幅下滑
- Owner: codex(impl)+ Claude(plan + review)
- Trigger: P0.1 license 通过

**P1.2 — Grammar 约束生成(xgrammar FFI 优先)**

- **FFI 包现成 xgrammar C++ 库**(降风险),不重写 Rust(~1000+ LOC FSM)
- Hook 到 ARLE sampling(`infer/src/scheduler/cuda/sampler.rs` 等价路径)
- 目标:JSON tool call 100% 有效 + 解码 overhead < 10%(简单 schema)/ < 30%(复杂)
- 风险点:xgrammar Python/C++ ABI 不稳定 → 需要 vendor lock 版本
- Owner: codex(FFI 实现)+ Claude(plan)
- Trigger: P0.1 license + P1.1 plan 草

### P2(之后)

- **32k-128k long-ctx 优化**(Claude Code class)— ARLE 在这个 shape 未 benched
- **Tool call fast path**(短结构化输出 lazy KV write 等优化)
- **MoE 模型支持**(Qwen3.5-MoE / DeepSeek V4)— substrate 已有,需要 production 验证
- **量化(AWQ/GPTQ/INT8)**— 用户为塞小卡常用
- **Metal backend coding/agent 优化**(Mac 用户)— 单独 track,本文档外

### KILL / DEFER(明确停做)

| 项 | 决定 | 原因 |
|---|---|---|
| ❌ M_pf-gemm Phase 0 | KILLED | top-1 cuBLAS 已最优 |
| ❌ M_pf-fuse Phase 0 | KILLED | regression |
| ❌ M_b.2.2 split-KV BF16 | KILLED | regression + e2e hang |
| ⏸ M_pf-gemm Phase 2 / 2.5(custom prefill GEMM) | **DEFER 到 Phase 0 graph 后** | 不是 binding constraint(R1 证明 SGLang 用 cuBLAS 也赢);但闭合 4k 仍需要 |
| ⏸ "Win at every canonical shape" 框架 | **REPLACE** | 换成 "win at agent shapes" |
| ⏸ Generic high-conc 持续优化 | **DEFEND only** | 已 +69%,守住即可 |
| ⏸ Multi-tenant prefix 持续优化 | **DEFEND only** | 已 +80%,守住即可 |

## 6. Multi-dimension 维度(本 pivot 范围外但要标记)

以下维度本文档**不深入分析**,但需要单独 track 评估:

1. **Metal backend coding/agent 优化** — Apple Silicon Mac 用户大量,需要单独
   bench + 优化策略;ARLE Metal scheduler runtime 已存在(`infer/src/backend/metal/`)
2. **MoE 模型支持** — Qwen3.5-MoE / DeepSeek V4 的 EP / 专家路由优化
3. **量化路径**(AWQ/GPTQ/INT8/FP4)— 8B/14B/32B 在 16GB-24GB 卡运行需求
4. **Vision/Multi-modal**(Qwen3-VL)— coding agent 截图分析
5. **生产指标**(p99/p99.9 / 错误处理 / 监控)
6. **Tool use 协议适配**(Anthropic XML / OpenAI v1 / 原生 JSON)
7. **Distributed inference**(TP/PP)— 大模型(Qwen3-72B / DeepSeek)

**默认假设**:本 pivot 先聚焦 CUDA + 单卡 + Qwen3-4B 4B-32B BF16 的场景;
其他维度后续单独 plan。

## 7. 文档/计划级 actions

1. **Update [`M_world1-30-percent-lead-roadmap.md`](../plans/M_world1-30-percent-lead-roadmap.md)**:
   - 加 §"Coding/Agent workload reframing"
   - acceptance criteria 加 agent-aligned shapes
   - 标注 high-conc 是 "defended" 不是 "primary target"
   - kill criteria 加 "graph + TileLang/FP8 不到 1500ms 4k → 重新评估"

2. **新建 plans**:
   - `docs/plans/M_agent-bench-harness.md`(P0.0,Claude 起草)
   - `docs/plans/M_spec-decode-medusa.md`(P1.1,Claude 起草,Medusa 优先 EAGLE 后置)
   - `docs/plans/M_xgrammar-ffi.md`(P1.2,Claude 起草,FFI 优先重写后置)

3. **现 plan 不动**:
   - `docs/plans/M_pf-graph-prefill-capture.md`(`939008f`,顶级质量)— 继续执行
   - `docs/plans/M_pf-gemm-cublaslt-autotune.md` Phase 2/2.5 — DEFER 状态保留

4. **新 ROADMAP 段**:加 "Coding/Agent runtime" 主线,标注 5 项独家组合(2 ✓ + 3 plan)

## 8. 决策请求 — 排序后的 Top 5(等用户拍板)

按战略关键度排序:

### 决策 D1(关键度 ★★★★★)— 接受产品重定义?

> ARLE 是 "Rust-native AI 开发工具推理 runtime"(coding/agent 优先),
> 不是 generic LLM serving。这影响所有后续优先级。

- [ ] 接受
- [ ] 不接受(请说明替代定位)

### 决策 D2(关键度 ★★★★★)— P0.0 agent bench harness 先做?

> 没这个 harness 无法验证 P0.1 prefill graph 在 agent shape 下是否真有效。
> P0.0 → P0.1 序列必须严格。

- [ ] 接受 P0.0 先做
- [ ] 不接受(并行 / 跳过 / 其他)

### 决策 D3(关键度 ★★★★)— Spec decode 选 Medusa 优先 EAGLE 后置?

> Medusa 单模型多头风险低,EAGLE 独立 draft 风险高(数据/训练依赖)。

- [ ] 接受(Medusa 先,EAGLE 评估后再决定)
- [ ] 不接受(直接 EAGLE / 都做 / 其他)

### 决策 D4(关键度 ★★★★)— xgrammar FFI 优先 Rust 重写后置?

> FFI 包 C++ 风险低 LOC 少;Rust 重写 ~1000 LOC FSM 风险高。

- [ ] 接受(FFI 先)
- [ ] 不接受(直接 Rust 重写 / 其他)

### 决策 D5(关键度 ★★★)— M_pf-gemm Phase 2 DEFER 不 KILL?

> R1 证明 SGLang 用 cuBLAS 也赢 → 不是 binding constraint。但闭合 4k 仍需要,
> 留 DEFER 状态 graph capture 之后再评估。

- [ ] 接受 DEFER(plan 保留)
- [ ] 不接受(KILL 整个 plan / 现在就做 / 其他)

## 9. 已知不确定性(透明列出)

以下点**仍有不确定性**,本文档基于现有 evidence 做了最佳判断,但需要更多数据:

| 不确定性 | 当前判断依据 | 解除条件 |
|---|---|---|
| Prefill graph 在 ARLE Rust runtime 的真实 ROI | Codex plan 数学 + R1 evidence | Phase 0 实测 |
| Spec decode acceptance rate(Qwen3-4B,coding 输出)| 行业典型 70-80%,coding tool call 可能更高 | 训练 / 选 draft 后实测 |
| 32k-128k long-ctx ARLE 表现 | 未 benched | 后续 P2 工作 |
| FlashInfer paged prefill vs TileLang HD128 kernel-time | 未对照 bench | nsys A/B + ncu |
| Metal backend 在 coding/agent 的现状 | 未深入分析 | 单独 Metal track plan |

## 10. 一句话总结

ARLE 的产品是 **Rust-native AI 开发工具推理 runtime**;护城河是 **5 项 capability
组合**(Rust hot path + 自定义 TileLang attention 已具备;Prefill graph + Spec
decode + xgrammar 待 land);**Long-ctx 4k TTFT 是 binding constraint 但 graph
capture 单点不够,必须组合 TileLang + FP8 KV** 才能达到 +30% past SGLang(<748ms);
**P0.0 agent bench harness 是 sequence-critical**(没它就没法验证后续改进);
**M_pf-gemm Phase 2 不 KILL 而 DEFER**(graph 落后再评估)。
