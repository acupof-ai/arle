# 32k–128k 长上下文吞吐 — World #1 by ≥30% Mission

**Status:** Active — 2026-04-30 立项；项目级 mission，覆盖 4 phase
**Owner:** ckl
**Supersedes:** 旧 `docs/plans/2026-04-30-longctx-32k-throughput.md`（其工程内容吸收为本文 §7 Phase 1）
**Bench discipline:** 每个 phase 独立 wins 条目；累乘后 margin 数据进 README

---

## 1 · Mission

```
在 32k–128k 长上下文 throughput 上做到 world #1，领先第二名 ≥30%。
```

不是补课。**foundation (Phase 1) + lossless 累乘 (Phase 2-4)**。

**对外可声明的 success 形态**：

> "On Qwen3-4B at 32k input, ARLE delivers <数字> tok/s — **≥1.30× the strongest open-source competitor** (SGLang / vLLM / TRT-LLM / Mooncake) on max-throughput AND long-decode workloads, on L4 + H100 + Apple M-series."

低于 1.30× → "leading"，**不能讲 "world #1"**。

---

## 2 · "World #1" 的精确定义

四个维度同时钉死，缺一不可：

### 2.1 Workload panel — **2 个**（全打）

| Workload | shape | 强敌 | 我们差异化 |
|---|---|---|---|
| **W1 · max-throughput** | prompt=32k / output=256 / c=4 | SGLang + FlashInfer | Phase 1 split-KV varlen + Phase 3 disagg + Phase 4 sparse |
| **W2 · long-decode** | prompt=32k / output=2048 / c=4 | TriForce / MagicDec | Phase 2 spec decode 是杀手锏 |

**显式不打**：agent-loop / RAG (prefix-cache 主导) — Mooncake 已强，本 mission 不正面冲突；128k 单请求 — 归 W2 的扩展，不单独立 workload。

### 2.2 Hardware tier — **3 档**

| 档 | 角色 | 卡型 |
|---|---|---|
| **H1 · 消费级 foundation** | 项目当前 wins/ 主基线 + CI | L4 (sm_89, 24GB) |
| **H2 · 服务器 SOTA 战场** | 行业公开基线集中地 | H100 (sm_90, 80GB) |
| **H3 · 独占差异化** | 项目结构性优势 | Apple M3/M4 Pro/Max (Metal + 统一内存 + NVMe T2) |

每档独立 wins 条目；mission success 要求**至少 H1 + H2 同时达标**，H3 是锦上添花。

### 2.3 Baseline panel — **4 家**（commit pinned）

| 系统 | Anchor commit | 启动参数对齐策略 |
|---|---|---|
| **SGLang** | `214c35b03184c354acf1f86f99746799e1c9b3a9` | `--kv-cache-dtype fp8_e4m3 --max-running-requests 16 --mem-fraction-static 0.85` |
| **vLLM v1** | tag `v0.10.x`（mission 启动时锁定）| `--kv-cache-dtype fp8 --max-num-seqs 16 --gpu-memory-utilization 0.85` |
| **TensorRT-LLM** | tag 锁定 mission 启动版本 | FP8 KV、`--max_batch_size 16` |
| **Mooncake** (Kimi 生产) | `kvcache-ai/Mooncake` HEAD（mission 启动时锁定）| 默认 disagg config |

无 commit pin / 无启动参数表 → wins 条目无效。**这一条是不容讨价的 reproducibility 硬门**。

### 2.4 Margin target — **≥30%**

每个 (workload × hardware) 格子的 success 公式：

```
success(W, H) := ARLE.tok/s(W, H) ≥ 1.30 × max(SGLang, vLLM, TRT-LLM, Mooncake).tok/s(W, H)
```

Mission 整体 success := `success(W1, H1) ∧ success(W1, H2) ∧ success(W2, H1) ∧ success(W2, H2)` 同时为真。

**4 个格子全绿才是 world #1**。绿 1-3 个 = "leading on subset"，wins 条目可发，但不能宣称 world #1。

---

## 3 · 4-Phase 结构（顺序执行，不并行）

```
Phase 1 ─► Phase 2 ─► Phase 3 ─► Phase 4
foundation   long-decode   max-throughput   叠加
              杀手锏          scale            放大器
```

每个 phase 是一份独立 plan/ 文档（Phase 1 的工程细节嵌入本文 §7；Phase 2-4 启动时各自 spawn plan/）。

| Phase | 目标 workload | 杠杆 | 累乘倍率 | 仓库现状 |
|---|---|---|---|---|
| **1** | W1 内功 | split-KV varlen FP8 + Mixed wire | 1.0× （catch-up）| 50% — varlen kernel 已写 |
| **2** | W2 杀手锏 | Long-ctx spec decode (MagicDec/TriForce) | × 2.0–2.5 | 30% — `speculative.rs` CPU 框架在树 |
| **3** | W1 scale | Disaggregated prefill/decode (Mooncake-aligned) | × 1.5 | 40% — `kv_tier`+`kv-native-sys` 基础在 |
| **4** | W1+W2 放大 | Sparse near-lossless (DuoAttention/Quest) | × 1.3 | 0% — 全新 |

**累乘数学**：

| Workload | Phase 1 | × Phase 2 | × Phase 3 | × Phase 4 | vs SGLang |
|---|---:|---:|---:|---:|---:|
| W1 max-throughput | 1.0 | (skip, 不影响) | 1.5 | 1.3 | **1.95×** |
| W2 long-decode | 1.0 | 2.5 | (skip) | 1.3 | **3.25×** |

两 workload 累乘后**远超 1.30× 门槛**。Phase 4 是保险，Phase 1-3 任一不达标 mission 仍能成立。

---

## 4 · 依赖图（phase 级）

```
Phase 1 ──► Phase 2 ──► Phase 4
   │                       ▲
   └─────► Phase 3 ────────┘
```

- **Phase 1 是所有后续的前置**（split-KV varlen + Mixed wire 是 KV format 基石）
- **Phase 2 与 Phase 3 互不依赖**，但本 mission 强排序：先 Phase 2（杀手锏数据，对外故事强）再 Phase 3（多 GPU 工程量大）
- **Phase 4 依赖 Phase 2 + Phase 3 全部落地**（叠加放大器，单走价值低）

---

## 5 · Hardware tier 分阶段策略

| Phase | H1 (L4) | H2 (H100) | H3 (Apple) |
|---|---|---|---|
| 1 | **必落** wins | 需要远端机器；落 wins | 不在 Phase 1 范围 |
| 2 | **必落** wins | 必落 wins | 评估 Metal DFlash 后续路径 |
| 3 | 必落 wins（多卡需 2× L4 或 1× H100）| **必落 wins**（disagg 真实场景）| 不适用 |
| 4 | 必落 wins | 必落 wins | 评估 Metal sparse 可行性 |

**H3 (Apple) 单独路径**：Metal DFlash speculative + 统一内存 + NVMe T2 是项目独占，与 H1/H2 走的不是同一 kernel 战场，独立立项归于本 mission §7 但 phase 节奏自定。

---

## 6 · Baseline 跑通的工程成本（不可回避）

每个 baseline × 每个 phase 节点 × 每个 hardware tier × 每个 workload = 一次 reproducible 跑：

```
4 baselines × 4 phase 检查点 × 3 hardware × 2 workload = 96 次 baseline 跑
```

不全打，但**至少**：

- Phase 1 末：4 baseline × H1 × W1 = 4 跑（确认 catch-up 起点）
- Phase 2 末：4 baseline × (H1+H2) × W2 = 8 跑（杀手锏 headline）
- Phase 3 末：4 baseline × (H1+H2) × W1 = 8 跑（scale headline）
- Phase 4 末：4 baseline × (H1+H2) × (W1+W2) = 16 跑（最终 mission 验证）

**总计 ~36 reproducible baseline 跑**。每跑必须有 commit pin + 启动参数 + headline 表，进 wins/。

这是项目级承诺，不是顺手能搞。专门 owner / runtime budget。

---

## 7 · Phase 1 — Foundation: split-KV varlen FP8 + Mixed wire

吸收自原 `docs/plans/2026-04-30-longctx-32k-throughput.md`，关键工程细节如下。完整版见本文末尾历史归档（如需要可从 git 找回原 plan：`git log --all -- docs/plans/2026-04-30-longctx-32k-throughput.md`）。

### 7.1 Phase 1 单一目标

```
Qwen3-4B FP8 KV, L4, prompt=32k / output=256 / c=4
ARLE tok/s within 5% of SGLang  (≥0.95×)
```

**Phase 1 不是 mission**——它是 mission 的入场券。Phase 1 不达标，Phase 2-4 累乘没有意义（基石不稳）。

### 7.2 Phase 1 工程切片

每一刀独立 commit，依赖图见原 plan §8.2。

| 序 | 名称 | Blocked by | 关键改动 | 验收 |
|---|---|---|---|---|
| **S1** | Kernel: varlen FP8+INT8 加 split-KV | — | `decode_attention_varlen_fp8.cu` 拆 phase-1/2，加 `K_scales/V_scales` 可空指针 | KV_SPLIT∈{1,8} max-abs-diff < 1e-3，varlen qlen=1 ≤ `quantized:307` + 5% |
| **S2** | Wire-up: Mixed plan 接 FP8/INT8 | S1 | `forward.rs:585` 扩 BF16\|FP8\|INT8；`batch_decode.rs:481` 删 BF16-only 早 return；`:720-750` 加 FP8/INT8 dispatch | TileLang on/off 两种 feature 各跑 e2e；`StepPlan::Mixed` 计数 > 0 |
| **S3** | 数值 gate 四层防线 | S2 | e2e long-prompt + FP8/BF16 16×64 + ARLE-BF16/SGLang-BF16 16×64 + 32×256 长尾扫 | ≥70% pass / 60-70% degraded / <60% stop；divergence_p50 ≥ token 30 |
| **S4** | Bench harness ARLE + SGLang baseline 脚本 | — | `scripts/bench_guidellm.sh --workload longctx-32k`；`scripts/bench_sglang_longctx.sh` 含 commit pin | 5s 烟测跑通，default workload 不回归 |
| **S5** | Bench 实跑 + Phase 1 wins | S2+S3+S4 | 无代码改动；c=4/300s + c=1/360s supplementary；remaining baseline panel 另按 §6 闭环 | c=4 ARLE/SGLang ≥ 0.95 closes SGLang row；c=1 记录 no-regression/parallel track；vLLM/TRT-LLM/Mooncake pending before full Phase 1 endpoint；plan 分布 `Mixed > 0`、`Split = 0` |
| **S6** | Qwen3-4B Split 守护 + ROADMAP | S5 | `execution.rs:362` 加 `debug_assert!(model_family != Qwen3)`；ROADMAP 更新 | 复跑 S5 不触发；clippy clean |

### 7.3 Phase 1 关键设计决策（已通过 4 subagent critique）

- **KV_SPLIT 动态**：`clamp(kv_total_len / 4096, 1, 16)`，抄 `decode_attention_quantized.cu` 现策略
- **INT8 顺手做**（+30 LoC）：varlen kernel 加 `K_scales` 可空指针，`nullptr` = FP8 自描述
- **Phase-2 reduction 模板**：抄 `decode_attention_quantized.cu:282-298`，禁止从 paper 重写
- **TileLang feature 盲区**：S2 acceptance 强制两种 feature 都构建并跑 e2e
- **数值 gate 阈值分层**：≥70% pass / 60-70% degraded / <60% stop（统一 §3 与 §5.4 阈值）
- **SGLang 对照 1st-class**：S4 单独脚本，commit pin `214c35b...`，启动参数表写进 wins
- **Split kernel 真实保留**：`debug_assert!` 守 Qwen3-4B 不进 Split，物理代码服务 LoRA + Qwen3.5（不假装删除，触发条件见 §11）

### 7.4 Phase 1 主要风险（最大裂缝）

**Prefill 段 FP8 长 qlen 单 CTA 模式无 tensor core**——chunked_prefill 切 16 段 × qlen=2048 全走 FP8 single-CTA-per-(q_token, q_head)，SGLang 用 FlashAttention-2 + tensor core，结构性快 1.5-2×。

触发动作：S5 prefill TFLOPs < SGLang 50% → 接受差距 < 5% 进 wins；≥ 5% 时**开 errors/ 立项独立 "FP8 prefill tensor-core kernel" 工程**，Phase 1 接受 degraded 收尾。

---

## 8 · Phase 2 — Long-ctx Speculative Decode (MagicDec/TriForce)

### 8.1 目标

W2 (long-decode) 上 ARLE/SGLang ≥ 2.0×（保守预期），stretch ≥ 2.5×。

### 8.2 杠杆来源（论文实证）

- **MagicDec** (ICLR 2025)：Llama3.1-8B 在 batch 32-256 下 **2.51×**，反直觉发现"长 ctx + 大 batch 下 spec 收益反升"
- **TriForce** (COLM 2024)：Llama2-13B 128k 在 2× RTX 4090 上 0.22s/token，**8× faster than offloading**
- **LongSpec** (2025-02)：32k+ 1.8-2.5×

### 8.3 仓库已有半成品

- `infer/src/speculative.rs` (631 行) — SpecConfig / TokenProposal / Verifier 全在
- `infer/src/speculative/cuda.rs` (157 行) — CUDA 集成 stub
- `crates/agent/` — 同步 spec config 已留位置

### 8.4 Phase 2 工程切片（高阶）

完整 plan 在 Phase 1 完成时 spawn `docs/plans/YYYY-MM-DD-longctx-spec-decode.md`。本节只列骨架：

1. CUDA verifier kernel 接进 mixed batch（复用 Phase 1 的 split-KV varlen kernel）
2. Draft model 路径：选 MagicDec 风格的 self-speculation + sparse KV，不引入第二个 model
3. SpecConfig 进 HTTP 层 + scheduler 接受率自适应
4. 数值正确性：拒绝采样保证 target 分布 bit-identical（按 LongSpec 的 3-prompt × 1024-token verifier）
5. W2 wins on H1 + H2

### 8.5 Phase 2 严格无损保证

理论上 verifier 拒绝采样保证 target 分布 bit-identical（仅受硬件浮点精度限制）。**不是近似，不是经验**。

---

## 9 · Phase 3 — Disaggregated Prefill/Decode (Mooncake-aligned)

### 9.1 目标

W1 (max-throughput) 在多卡上 ARLE/SGLang ≥ 1.5×。

### 9.2 杠杆来源

- **DistServe** (OSDI 2024)：**7.4× more requests / 12.6× tighter SLO**
- **Mooncake** (FAST 2025 Best Paper)：长 ctx 真实流量容量 **+59-498%**
- **Sarathi-Serve** (OSDI 2024)：chunked prefill + stall-free batch，**2.6-5.6×**

### 9.3 仓库已有半成品

- `infer/src/kv_tier/` — T0/T1/T2 完整
- `crates/kv-native-sys/` — 持久化 substrate
- `infer/src/kv_tier/transport/` — RDMA / 文件系统 backend 接口
- `docs/projects/tiered-kv-cache.md` — Phase E 远端验证待做

### 9.4 Phase 3 工程切片（高阶）

Spawn `docs/plans/YYYY-MM-DD-disaggregated-prefill-decode.md` 时展开。骨架：

1. Prefill GPU + Decode GPU 物理拆分（最小 2 GPU 配置）
2. KV 通过 NVLink / RDMA / 共享内存传输（NIXL 或本仓库 transport 抽象）
3. Scheduler 双向：prefill 完成 push KV → decode 起步
4. 命中 RadixCache 时跳过 prefill GPU，直接 decode + tier readmission
5. W1 wins on H1×2 + H2 单卡（H100 80GB 单卡可装下 prefill+decode 隔离）

### 9.5 Phase 3 严格无损保证

只切分阶段，attention 数学不变。bit-identical。

---

## 10 · Phase 4 — Sparse Near-Lossless（叠加放大器）

### 10.1 目标

把 Phase 1+2+3 累乘后再放大 ~1.3×。

### 10.2 候选方法

| 方法 | LongBench 退化 | 节省 | 优先 |
|---|---|---|---|
| **DuoAttention** (ICLR 2025) | <1pp | KV 2× / decode 2× | **★** 首选——MIT-Han Lab 开源代码现成，head 离线分类 |
| Quest (ICML 2024) | <1pp | decode 2-3× | 备选——页级 top-k 与现 paged 路径自然契合 |
| MInference 1.0 (NeurIPS 2024) | <1pp | prefill 10× @ 1M | 仅 W2 prefill 段相关，与 Phase 1 prefill 缺口同向 |
| SnapKV / PyramidKV | 持平 | KV 4-8× | 备选——prefill 一次性裁 KV |

**默认上 DuoAttention**——head 分类离线一次，运行时 attention 路径分流，对 Phase 1-3 stack 改动最小。

### 10.3 严格性免责

⚠ **Phase 4 不是严格无损**。LongBench ≤1pp 退化是经验保证，不是数学保证。

如果用户语境严格要求 bit-identical → **跳过 Phase 4**，仅靠 Phase 1+2+3 累乘 (W1: 1.5×, W2: 2.5×) 也可以宣称 "leading by ≥30% in lossless setting"。

实际部署：以 feature flag 形式提供，default off。

### 10.4 严格无损路线（Phase 4 跳过时的替代）

如果跳 Phase 4：
- **vAttention-style memory layout** 是另一可选，但与 paged KV 投资冲突，工程成本 > 收益
- **Async pipeline gap closure**（`wins/2026-04-29-scheduler-overlap-gap-instrumentation.md` 已量化）— 单数 % 收益，仅在 Phase 1-3 stack 还差 0.95-0.99 时启用

---

## 11 · 显式不做（Out of Scope）

| 不做 | 触发条件 |
|---|---|
| **agent-loop / RAG workload** | Mooncake 已强；本 mission 不正面冲突。**触发加入**：完成 Phase 4 后 ARLE 综合优势明显 + 用户场景需要时 |
| **128k 单请求专门 workload** | 归 W2 长-decode 的扩展 |
| **vAttention** (ASPLOS 2025) | 与 paged KV 投资冲突，迁移成本 > 收益 |
| **`StepPlan::Split` enum 整体删除** | 触发：(a) Qwen3.5 mixed 落地 + (b) LoRA mixed 落地 + (c) 一周生产日志 Split 计数 = 0 |
| **`BatchAttention` trait/enum** | 触发：≥2 模型族同型 dispatch + 至少一族 ≥3 实现 + 新增分支需碰 ≥2 call site |
| **Sparse method 进 default** | 触发：Phase 4 wins 落地 + 用户接受 ≤1pp 退化 |
| **再 push beyond 1.5× margin per phase** | 物理上限考虑（HBM 带宽、L4 22GB 容量），过度优化收益递减 |

---

## 12 · 风险（mission 级）

| 风险 | 触发信号 | 回退动作 |
|---|---|---|
| Phase 1 catch-up 失败 (ARLE/SGLang < 0.95) | S5 bench | 按 §7.4 prefill kernel 缺口处理；mission 推迟 6 个月 |
| baseline 4 家不能全部跑通 | mission 启动时跑不起 vLLM v1 / TRT-LLM / Mooncake | 降级为 SGLang-only baseline；wins 表注明 "limited baseline panel"，mission success 标准变弱（仅 2 家）|
| Phase 2 spec accept rate 低 | MagicDec self-speculation 在 Qwen3-4B 上 < 2.0× | 试 LongSpec 的 retrieval+滑窗 KV draft；若仍不达，跳 Phase 2 用 Phase 3+4 |
| Phase 3 disagg 在 L4 上不可达（容量限制） | 2× L4 跑不动 prefill+decode 隔离 | H2 (H100) 单卡 80GB 跑 disagg；H1 跳过 Phase 3 wins |
| Phase 4 LongBench 实测退化 > 1pp | DuoAttention head 分类对 Qwen3-4B 不稳 | 切 Quest / SnapKV 备选；若都不达，跳 Phase 4，仅靠 Phase 1+2+3 累乘 |
| 4 baseline 中某家发布更新追平 | 行业进展正常 | 升级 baseline anchor commit；wins 注明对照升级 |

---

## 13 · 当前位置（Phase 1 状态 = SGLang-row closed; baseline panel pending）

2026-05-01 本地 L4 已完成 Phase 1 W1/c4 SGLang-row mission-critical close：

| run | successful requests | total output tokens | effective out tok/s | vs SGLang c4 | TTFT p50 | ITL p50 | artifact |
|---|---:|---:|---:|---:|---:|---:|---|
| r1 | 32 | 8192 | 27.307 | 1.678x | 32225.9 ms | 178.4 ms | `bench-output/2026-05-01-phase15-evictable-c4-r1/benchmarks.json` |
| r2 | 28 | 7168 | 23.893 | 1.469x | 33888.9 ms | 116.2 ms | `bench-output/2026-05-01-phase15-evictable-c4-r2/benchmarks.json` |
| r3 | 32 | 8192 | 27.307 | 1.678x | 33879.3 ms | 117.5 ms | `bench-output/2026-05-01-phase15-evictable-c4-r3/benchmarks.json` |
| mean | - | - | 26.169 | 1.609x | - | - | `docs/experience/wins/2026-05-01-phase1-close-evictable.md` |

Phase 1 SGLang-row close 结论：

- **§7.1 SGLang-row entrance gate closed:** W1/c4 mean `1.609x` SGLang,
  above the `0.95x` entrance criterion.
- **§2.4 SGLang-row margin secured for W1/H1:** worst run `1.469x` SGLang,
  above the `1.30x` mission margin target. Full `success(W1,H1)` still
  requires the remaining vLLM / TRT-LLM / Mooncake pinned baselines.
- c=4 deadlock/bimodal mode removed in the three-run validation set.
- c=1 supplementary measurement remains a parallel single-stream decode
  optimization track: c=1 TTFT is slightly outside the §7.4 5% watch line
  (`12540.6 ms` vs SGLang secondary `11862.86 ms`, `+5.7%`), and the larger
  residual gap is decode ITL (`56.84 ms` vs `43.10 ms`). This is tracked in
  `docs/experience/errors/2026-05-01-c1-single-stream-decode-gap-parallel-track.md`
  and does not block W1/c4 mission closure or Phase 2 W2 spec-decode start.

已完成：

- L4 环境、Qwen3-4B 权重、本地 CUDA release build。
- S3 单目标 32k long-prompt smoke。
- S4 harness smoke：ARLE 5s smoke，SGLang 60s smoke。
- S5 SGLang pinned baseline run。
- P1.0 plan-label counters：`wins/2026-04-30-bench-guidellm-longctx-32k-phase1-s5-plan-label.md`
  记录 `Mixed=16`、`Split=0`。
- Phase 1 SGLang-row c=4 close wins:
  `docs/experience/wins/2026-05-01-phase1-close-evictable.md`。

下一刀 = Phase 2 spec decode plan
`docs/plans/2026-05-01-longctx-spec-decode-phase2.md`。

---

## 14 · 关联文档

- 父框架：本 ROADMAP P0
- 父审计：[`2026-04-29-throughput-gap-analysis.md`](2026-04-29-throughput-gap-analysis.md) — K2 (50 tok/s lever)
- Pipeline map：[`2026-04-29-scheduler-pipeline-map.md`](2026-04-29-scheduler-pipeline-map.md)
- 父对齐计划：[`../plans/2026-04-23-cuda-decode-sglang-alignment.md`](../plans/2026-04-23-cuda-decode-sglang-alignment.md)
- 多 GPU 计划：[`../plans/2026-04-28-single-node-multi-gpu.md`](../plans/2026-04-28-single-node-multi-gpu.md) — Phase 3 disagg 的 TP/PP/CP 骨架
- Tiered KV：[`tiered-kv-cache.md`](tiered-kv-cache.md) — Phase 3 数据通路
- Bench 协议：[`../bench-and-trace-spec.md`](../bench-and-trace-spec.md)
- Bench 矩阵设计：[`../plans/bench-matrix-design-2026-04-29.md`](../plans/bench-matrix-design-2026-04-29.md)
- 数值漂移历史：[`../experience/errors/2026-04-30-arle-fp8kv-numerical-drift.md`](../experience/errors/2026-04-30-arle-fp8kv-numerical-drift.md)
- TileLang 盲区先例：[`../experience/errors/2026-04-28-tilelang-prefill-short-qlen-nan.md`](../experience/errors/2026-04-28-tilelang-prefill-short-qlen-nan.md)
- 反速 模式：[`../../memory/feedback_no_speculative_interface_shaping.md`](../../memory/feedback_no_speculative_interface_shaping.md)、[`../../memory/feedback_no_half_states.md`](../../memory/feedback_no_half_states.md)

### 论文 / 系统引用

**Phase 2 (spec)：**
- [MagicDec, ICLR 2025](https://arxiv.org/abs/2408.11049) · [project](https://infini-ai-lab.github.io/MagicDec-part1/)
- [TriForce, COLM 2024](https://infini-ai-lab.github.io/TriForce/) · [code](https://github.com/Infini-AI-Lab/TriForce)
- [LongSpec, 2025-02](https://arxiv.org/abs/2502.17421)

**Phase 3 (disagg)：**
- [DistServe, OSDI 2024](https://arxiv.org/abs/2401.09670) · [code](https://github.com/LLMServe/DistServe)
- [Mooncake, FAST 2025 Best Paper](https://arxiv.org/abs/2407.00079) · [code](https://github.com/kvcache-ai/Mooncake)
- [Sarathi-Serve, OSDI 2024](https://arxiv.org/abs/2403.02310)
- [Disaggregated Inference 18-month retro (UCSD)](https://haoailab.com/blogs/distserve-retro/)

**Phase 4 (sparse)：**
- [DuoAttention, ICLR 2025](https://arxiv.org/abs/2410.10819) · [code](https://github.com/mit-han-lab/duo-attention)
- [Quest, ICML 2024](https://arxiv.org/abs/2406.10774)
- [MInference 1.0, NeurIPS 2024](https://arxiv.org/abs/2407.02490)
- [SnapKV, NeurIPS 2024](https://arxiv.org/abs/2404.14469)
- [PyramidKV / PyramidInfer](https://arxiv.org/abs/2406.02069)

**Baseline 系统：**
- [SGLang](https://github.com/sgl-project/sglang)
- [vLLM v1](https://github.com/vllm-project/vllm)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [Mooncake](https://github.com/kvcache-ai/Mooncake)

**Kernel 基础：**
- [FlashAttention-3, NeurIPS 2024](https://pytorch.org/blog/flashattention-3/)
- [FlashInfer 0.2, MLSys 2025](https://arxiv.org/abs/2501.01005)
