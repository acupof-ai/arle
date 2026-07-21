# ARLE 战略主文档 v2 — 唯一真理(2026-06-10)

> **本文 supersede `2026-05-07-arle-master-strategy.md`(v1)的全部战略结论。**
> 任何与本文冲突的旧 doc 以本文为准。v1 保留作历史推理记录,header 已打 SUPERSEDED 横幅。
> 进展跟踪:[umbrella #55](https://github.com/cklxx/arle/issues/55) ——
> Phase 0 = #56–#59(✅ 全关 2026-06-10;残留 #68 model-generic KV-quant gate,不再 block
> Phase 1),Phase 1 = #60–#61(ACTIVE),Phase 2 = #70(kernel-base 收敛,先行)+ #62,
> Phase 3 = #63–#65;off-path infra = #69(cold-boot)。
>
> 写作纪律:结论先行 + evidence 引用;hypothesis 显式标注(§6);不复述 per-date
> wins/errors 的细节,只 link。

---

## §0 一句话核心论断

**ARLE = Rust-native、设备中立的推理 runtime + runtime-led OPD 训练轴。**

- **产品主线 = coding-agent runtime**:本地单用户(Metal/Apple Silicon)+ 自托管多租户
  (CUDA),serve coding/agent workload(W3/W4 shape:长 prompt、短结构化输出、
  prefix 复用 80%+、spec-decode 杠杆大)。v1 的这条产品论证至今没有被任何证据推翻,
  只是被 6 月的 DSv4 冲刺饿着。
- **DSv4-Flash 8×H20 = 技术 wedge + 引擎锻造场,不是独立产品线**。它的不对称价值:
  ARLE 能 serve 的 DSv4-Flash FP8 checkpoint **SGLang 加载不了**(2026-06-04 同 pod
  A/B 只能用 V3.2 做 proxy),"第一个能 serve DSv4 的 runtime" 是宣传卡 + 最难的 KV
  layout 逼出了 engine-generic 抽象。正面对抗 SGLang 主场(多 GPU frontier serving)
  不是 ARLE 的产品定位。
- **训练 = OPD-only**(2026-05-18 pivot,不变)。结构性理由依然成立:OPD 需要强 teacher
  serving + 低延迟 student 打分,这正是 `infer-*` 的本体;其他训练面是 OSS 重复。

## §0.1 真护城河(修订 v1 的"5 项组合")

| Moat | 状态 | Evidence |
|---|---|---|
| **设备中立 Engine seam**:同一 `Engine<E,K>` 驱动 CUDA 连续批处理 + Metal MLX 两个结构迥异后端;第三后端 = 实现两个 host-only trait | ✅ 已证明 | final report §4;`agent-bench` 双后端实例化。vLLM/SGLang 无 Metal 故事 |
| **Rust hot path**(全栈无 Python)| ✅ | 整个 workspace |
| **DSv4-Flash 首发覆盖**(DSA/CSA/HCA/mHC/MTP 全架构)| ✅ 正确性已证,perf 在途 | needle@long-ctx + GSM8K=72;SGLang 加载不了 V4-Flash |
| **OPD 训推一体**(teacher 即 serving 路径)| ⏳ substrate 已通,GPU 实验饿着 | [OPD pivot](2026-05-18-opd-only-pivot.md);`arle train opd` e2e wired |

**v1 moat 列表中被撤销的项**:TileLang 自研 attention kernel 不再是 moat —— 6 月
官方-kernel 采纳弧(FlashMLA / 官方 DSA indexer / DeepGEMM)实证 adopt-official 全面
胜过 hand-rolled(retro),
TileLang prefill 还有 FlashInfer 迁移计划
在案。自研 kernel 是手段不是壁垒;**`先用最好的再自己写` 是纪律**。

---

## §1 v1 → v2:哪些结论被推翻(防混淆清单)

| v1 论断 | v2 裁定 | 依据 |
|---|---|---|
| 训练侧 = DSV4 from-scratch repro(v1 §5/§8 全部)| **DEAD** | 2026-05-18 OPD pivot(322× 预训练 gap 实测)|
| "❌ Distributed 大模型(72B+)是后期" | **被现实推翻**:DSv4-Flash TP8/EP8 是 6 月主战场,且锻造出了 engine-generic 抽象 | 6 月全部 commits/wins |
| 单体 `infer/` crate 是 runtime 本体 | **DELETED**(~167k LOC,`e81b98fb`):serving truth = `infer-plan→seam→core→cuda/metal→server/api` | final report §1 |
| Moat = Rust+TileLang+graph+spec+grammar 5 项组合 | **修订为 §0.1 四项**;TileLang 撤销 | 上文 |
| W3/W4 跨引擎 baseline 是 P0.0 "必须先做" | **依然成立但从未执行**(2026-05-02 至今);移入 Phase 3,排在 batched lane 之后 | 无 W3/W4 跨引擎数据存在 |
| Medusa 是 spec-decode 主路径 | **修订**:DSv4 走 frozen-KV MTP(checkpoint 自带 draft head,免训练);Qwen3.5 Medusa 继续卡 recurrent-rollback gate | frozen-KV 设计 |
| 4k/8k prefill 落后 SGLang 50%(sm_89 单卡数据)| **过期数据**,不再指导决策;当前 perf 锚 = H20 同 pod A/B | H20 reference baseline |

---

## §2 已确立的事实(evidence,非 hypothesis)

1. **架构**:设备中立重写已 land 且唯一(`infer/` 删除);两后端共享 scheduler /
   RadixCache / chunked-prefill / sampling / streaming。扩展性已论证非断言。
2. **DSv4-Flash 8×H20 性能坐标**:decode 236.8 → ~26-27 ms/token(结构弧 + 官方 kernel
   弧);同 pod SGLang(V3.2 proxy)= **15.89 ms no-spec / 8.24 ms +EAGLE**;
   **5-6 ms 是 H100/H800 数字,不是 H20 目标**。当前 no-spec gap ≈ 1.6-1.7×,剩余
   slice 已点名(MLA attention、FP8 fused call form、mHC fuse)。
3. **长上下文**:900K milestone 端到端(718K tok / 270s);per-layer RoPE theta 修复
   (`fa355315`);**残留**:seq≥241 尾数丢失(§6)、256K admission band-aid(`39be5f83`)。
4. **B=1 per-kernel 优化已死**(8 次 wash 实证):只有 ①更少 GPU 计算 ②通信 overlap
   ③摊销(batching/MTP)能动 B=1 的墙。
   CONCLUSIVE。
5. **c≥2 serving lane 刚刚存在**(`cd421794`,2026-06-10):executor 把 mixed/multi-prefill
   plan 拆成串行 single-row 子步 + decode 子批 —— 这是**正确性/可用性修复,不是吞吐**;
   c-sweep bench pending-remote。此前 c≥2 直接杀引擎线程,锁死 deepep_ll A/B、MTP
   acceptance workload、一切吞吐叙事
   (errors 2026-06-10)。
6. **正确性 gate 语义**:正确推理 ≠ 与基线 byte-identical(MoE run-to-run 非确定性);
   gate = needle + same-config-twice 地板 + 自洽。KV-precision-parity 审计**尚未移植**到
   `infer-cuda`,是一切 KV/quant default-flip 的前置。

---

## §3 演进路径(Phase 0-3,严格串行)

```
Phase 0 还债(正确性 + 真相面)
   └─▶ Phase 1 批量化 serving lane license(钥匙石)
          └─▶ Phase 2 spec decode 默认有且默认好用
                 └─▶ Phase 3 产品回归(W3/W4 + OPD + Qwen3.6)
```

### Phase 0 — 还债(~1-2 周,廉价,不可跳)

| 项 | 出口条件 |
|---|---|
| 长上下文正确性收口:seq≥241 尾数残留 root-cause + same-config-twice 控制 | needle 全长度段通过;非确定性地板量化 |
| 256K admission band-aid → 真 fix | dummy-KvPool sizing 删除,admission 按真实 KV 预算 |
| KV-precision-parity 审计移植到 `infer-cuda` DSv4 | parity harness 在重写栈跑通;三个 gated lever(FlashMLA/fused-wqkv/contig-MoE)的 default-flip 解锁 |
| 真相面重同步(本文 + ROADMAP + CLAUDE.md/AGENTS.md + index.md)| 本 commit 系列完成 |

**理由**:在脏基线上做 Phase 1 = 自欺(§0.1 既有教训);CLAUDE.md 还在描述已删除的
`infer/`,每个 session 的认知起点都是错的。

### Phase 1 — 批量化 serving lane license(钥匙石,~2-6 周)

执行已有的 authoritative 计划
[`unified-batched-kvpool-abstraction`](../plans/2026-06-07-unified-batched-kvpool-abstraction.md)
(`KvBatchDescriptor` + 每模型 `ModelKvAdapter`,DSv4 先行)。`cd421794` 的顺序拆分是
起点不是终点:真批量 lowering 才有吞吐。

**为什么是钥匙石**:同时解锁所有停泊项 —— guidellm c-sweep(吞吐叙事)、deepep_ll
翻案 A/B(它唯一能赢的 lane)、MTP acceptance workload、Metal 收敛(同为 single-row
guard)、Qwen/Gemma 的 engine-generic 复用。wall-clock @4096 实测 c=8 仅 1.63× scaling
(pd-systematic-analysis)。

**出口条件**:c-sweep 不 crash 且 TTFT+ITL+tok/s 全指标过
[bench-and-trace-spec](../bench-and-trace-spec.md)(distilled lesson:`plan_label=mixed`
可达性 ≠ license);然后 deepep_ll-vs-allreduce 在真实 lane A/B,license-or-kill。

### Phase 2 — spec decode 默认有且默认好用(ckl directive)

路径:frozen-KV MTP(draft+verify
读冻结 target KV,不重跑 compressor;checkpoint 自带 `mtp.0.*` draft head,免训练)。

- **排序在 kernel base 收敛后**:×1.93 摊在 16ms 上 = 8ms,摊在 30ms 上只有 16ms。
- **第一步是廉价实测**:coherent workload(GSM8K/ShareGPT,非退化 prompt)上的
  acceptance —— 这是当前最大的未验证 hypothesis(§6),license-or-kill。
- Qwen3.5 Medusa 不抢队(recurrent rollback gate 未解)。

**出口条件**:spec-on 为默认,B=1 +长 ctx 两个 shape 上 wall-clock 净赢,正确性 gate
按 §2.6 语义(非 byte-identity)。H20 目标 ~8-10 ms/token。

### Phase 3 — 产品回归(带着能打的 runtime)

1. **W3/W4 跨引擎 baseline**(v1 P0.0,欠账 5 周+):ARLE/SGLang/vLLM 三方,
   `scripts/bench_agent_trace.py` 已在;mission 阈值 ≥1.30 维持。
2. **长上下文领导力 mission 重启**:在新 substrate + frozen-KV 设计上重做 Phase 2
   spec-decode(当年 −62.8% regression 的两个根因均已有解)。
3. **OPD 拿回 GPU 时间**:rollout O(n²) 线程重启;capability 评估按 multi-seed gate。
4. **Qwen3.6 CUDA serving**(Next-Model #2):`ModelKvAdapter` 第二个实现,验证抽象的
   engine-generic 承诺。
5. **AIPC 路线**(ckl directive 2026-06-10,#71):Metal 单用户收敛(继承 Phase 1 批量
   抽象,c=1 SLO gate)+ HIP/ROCm 第三后端 —— `infer-hip` 只实现两个 host-only trait,
   zero `infer-core` 改动是验收线(moat 主张变成可测命题)。本地统一内存硬件
   (M-series / Ryzen AI Max 级),不与 8×H20 pod 抢占;§5 的 ROCm DEFER 仅指
   Phase 0-2,在此入队。

---

## §4 决策记录(default 在 ckl 未否决前生效)

| # | 决策 | 状态 |
|---|---|---|
| D1 | 产品 = coding-agent runtime;DSv4 = wedge + 锻造场,非独立产品线 | **default YES**(2026-06-10 分析获 ckl OK)|
| D2 | 吞吐轴唯一路径 = unified batched KvPool 计划;不再做绕过 seam 的 DSv4 专用旁路 | YES |
| D3 | spec decode 必须默认有且默认好用;DSv4 走 frozen-KV MTP,排在 kernel base 后 | YES(ckl directive)|
| D4 | 三线(DSv4 substrate / agent workload / OPD)严格串行,不并行争 GPU 与注意力 | YES |
| D5 | 运行时 knob 一律 CLI `--flags`,env var 仅 build/toolchain | YES(ckl directive,清理单在 code-cleanup-audit)|

## §5 KILL / DEFER(不要重做)

| 项 | 裁定 | 依据 |
|---|---|---|
| B=1 decode 的 per-kernel / launch / alloc / host-overhead **micro**-lever | **KILLED**(8 wash)。例外:whole-step graph + lockstep step-start 两项已于 2026-06-10 依本节翻案规则 RE-LICENSED([skew anatomy](../experience/wins/2026-06-10-dsv4-nsys-skew-anatomy-rewrites-lever-board.md) 实测 launch-gap 29% + start-offset 18% of wall),跟踪 #70 | `2026-06-08-dsv4-decode-6ms-FINAL`;翻案 evidence 见 skew anatomy |
| deepep_ll default-on | **BLOCKED**:B=1 −55%;翻案唯一条件 = Phase 1 后批量 lane A/B 赢 | errors 2026-06-10 |
| classical spec decode(Leviathan 自/外 draft)| **KILLED**(α≤0.25 三连)| v1 §7.4 evidence,继续有效 |
| pooled/contiguous decode 当 B=1 默认 | **KILLED**(28.4 vs 37.6 tok/s)| memory + A/B |
| DSA-skip lever | **KILLED**(−3.7%,压缩是必要计算)| `5b10d6b5` |
| 5-6 ms/token 当 H20 目标 | **KILLED**(H100 数字;H20 真目标 ~16ms no-spec / ~8ms +MTP)| 同 pod SGLang A/B |
| hand-rolled kernel 先行(vs adopt-official)| **纪律性 KILL** | adopt-official retro |
| FlashInfer 迁移 / tiered-KV readmission / ROCm 第三后端 / Qwen3.5 Medusa | **DEFER**,不进 Phase 0-2(ROCm 自 Phase 3 起以 AIPC 路线入队,#71)| 各自 plan 在案 |

## §6 不确定性(显式,license-or-kill 解除)

| 不确定性 | 解除条件 |
|---|---|
| **MTP acceptance(coherent workload)** —— 最大未验证 hypothesis,Phase 2 的成立前提 | Phase 2 第一步廉价实测(GSM8K/ShareGPT,≥2 prompt)|
| `cd421794` c≥2 lane 实测行为(吞吐、稳定性)| pending-remote c-sweep(wins entry 待补)|
| 长上下文 seq≥241 尾数残留的 root cause | Phase 0 控制实验 |
| OPD capability 增益真实性 | multi-seed(≥5)+ Wilson CI gate(2026-05-28 规则)|
| 8×H20 pod 单点依赖(全部 P0 证据来源)| 无缓解,显式接受 |
| W3/W4 上 ARLE 真实竞争力(从未实测)| Phase 3 跨引擎 baseline |

## §7 Rules

1. **本文是 ARLE 战略唯一信息源**;其他文档只 link 不复述。新 plan 必须从 §3 派生。
2. **新实验必须 cite §5 KILL 列表**;重做 KILLED 项需先推翻其 evidence。
3. **Supersede 规则**:被本文超越的旧 doc 在 header 加
   `> ⚠️ SUPERSEDED by docs/projects/2026-06-10-arle-master-strategy-v2.md`,不删正文。
4. **进展跟踪走 GitHub issues**(umbrella + per-Phase 子 issue);Phase 出口条件变更须
   同步本文。
5. 新发现的不确定性立即进 §6;新 KILL 立即进 §5。
