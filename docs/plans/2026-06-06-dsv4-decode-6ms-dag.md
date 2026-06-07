# DSv4 decode → 6ms — 抽丝剥茧 DAG（原子任务 + 依赖图 + 预算）

## Superseded by later evidence

**The whole DAG below is built on the SMOKE-SHAPE lever ranking and the
"6ms-via-EAGLE-now" critical path — both overturned the same day.** The decode-side
critical path here (`G1 → A1 → A2 → A3` with D-branch comm/GEMV/mhc compressing the
base) was anchored on the 8-token decode profile (comm 32.4%, GEMV 14.4%, mhc 12.2%)
from [`2026-06-06-dsv4-decode-6ms-remaining-levers.md`](2026-06-06-dsv4-decode-6ms-remaining-levers.md),
which is itself superseded. The end-to-end **wall-clock** trace at the 4096 SLO shape
([`2026-06-06-dsv4-pd-systematic-analysis.md`](2026-06-06-dsv4-pd-systematic-analysis.md)
§3) found the real bottleneck is `dsv4_csa_select` (the E2 entry in this doc's
prefill branch was actually the #1 decode AND prefill cost), and the fix was to
**adopt the official DeepSeek DSA indexer**, not hand-roll a parallel selector:
[`../experience/wins/2026-06-07-dsv4-official-dsa-default-on.md`](../experience/wins/2026-06-07-dsv4-official-dsa-default-on.md).

The 6ms target itself is re-anchored: base no-spec decode is ~20-35ms on H20 and 6ms
**requires** MTP/EAGLE spec — see the
[H20 reference baseline](2026-06-06-dsv4-h20-reference-baseline.md). The Branch-A
EAGLE math here is also overturned: A1 (per-token rollback) landed correct but −32%;
A2 (s_q=K) was killed then un-killed via the frozen-KV redesign
([`2026-06-06-dsv4-frozen-kv-mtp-redesign.md`](2026-06-06-dsv4-frozen-kv-mtp-redesign.md));
MTP is now parked at the **draft-quality wall** (39% accept vs SGLang 68%):
[`../experience/errors/2026-06-06-dsv4-mtp-perf-acceptance-workload-blockers.md`](../experience/errors/2026-06-06-dsv4-mtp-perf-acceptance-workload-blockers.md).
The forward-looking program is the
[unified batched-decode/paged-KV abstraction](2026-06-07-unified-batched-kvpool-abstraction.md).
Kept for history (the §0.1 atomic-task decomposition + correctness-model framing are
valid process records).

---

> **⚠️ UPDATE 2026-06-06: EAGLE 主干 A2/A3 KILLED.** A1(per-token rollback)
> 落地正确(`25a92e8a`,needle 过)但 −32%。A2(s_q=K)实测 **12.85 tok/s(3× 慢)
> 且与 autoregressive 系统性不等价**(scalar control 也分叉 → 是 DSv4 stateful 压缩
> 注意力,非 FlashMLA glue),见 `errors/2026-06-06-dsv4-eagle-sqk-no-amortize-kill.md`。
> **EAGLE 的 K-token 摊销对 DSv4 不成立**:per-query prepare-chain(csa_select top-512
> + compressor + indexer)昂贵且随 K 线性,K-token forward ≈ K× attention 不是 1×
> (同 FlashMLA-prefill kill 的墙)。**新 critical path = prepare-chain 优化 + D kernel
> levers**(下文 A 分支作废,prepare-chain 见 §新)。6ms-via-EAGLE 受阻。

**Date:** 2026-06-06. **方法**:把"compressed-attention + EAGLE + 6ms"拆成原子任务,
标出依赖,画 DAG,critical path 自现。**判据已纠正**(ckl 2026-06-06):gate 是
**正确推理**(needle 取回 + 连贯 + 质量),**不是** byte-identity 复刻 s_q=1 基线。

## 0. 纠正后的正确性模型(贯穿所有节点的前置)

byte-identity 判据是错的,且被 DSv4 MoE **run-to-run 非确定性**(atomic-scatter 浮点序
→ near-tie argmax 翻转,见 `reference_dsv4_moe_nondeterminism_confounds_4096_parity`)
**confound**。greedy 投机的正确性保证是"输出 == verify kernel 自身的自回归 greedy",
near-tie 上和 s_q=1 不同**两者都对**。

**G1 — 正确推理 gate(所有验证的前置,cross-cutting):**
- `needle`:植入事实的 prompt,spec-ON 必须取回(证明 attention 数值正确,非"看着像")。
- `same-config-twice 控制`:同 config 跑两遍取 token 分叉点 = **非确定性地板**;任何
  spec-vs-baseline 差异只有**超过这个地板**才算 bug。
- `自洽`:spec-ON 输出 == verify-kernel 自回归 greedy(verify kernel 是参照,不是 s_q=1)。
- 退化 prompt(greedy 自己就循环,如 [344] 的 `343,67,11`)**不能**当 correctness 测例。

## 1. 原子任务清单（id · what · files/sites · gate · risk · impact）

### Branch A — EAGLE/MTP 投机解码（乘数项,通往 6ms 的主干）

| id | what | files/sites | gate | risk | impact |
|----|------|-------------|------|------|--------|
| **A1** | **compressor+indexer running-state 回滚（snapshot/restore 全 5 字段 × 2)— 无条件,代码已证 bug 真**:`truncate_decode_len` 只写 `compressed.seq_len`,从不还原 `pending_kv`/`prev_overlap_*`;rejected draft 撞 compression 边界(每 `ratio` token)时,它触发的压缩(更新 prev_overlap + reset pending)**全没回滚** → 真损坏。draft 不触发压缩时才自愈(故 [11111] 过、长 prompt 不过) | `attention.rs` `Dsv4CompressorState`(79)、`Dsv4LayerAttentionState`(reset 437)、`truncate_decode_len`(456) | needle + same-twice 地板(非 byte-identity) | 中 | 解锁正确 depth-1 |
| A0 | (机制确认,非 gate)dump pending_kv reject 前后 + 是否撞 compression 边界 + same-twice 非确定性地板 | `attention.rs:728` dump 已建 | — | 低 | 确认机制/floor,不决定是否修 |
| **A2** | **s_q=K FlashMLA verify**(真正的加速,depth-1 ~1.5×) | `attention.rs` `DSV4_FLASHMLA_S_Q`(25)、`Dsv4FlashMlaDecodeState`(134)、`dsv4_flashmla_decode_build_indices.cu` | needle + 自洽 | 高 | **26.6→~18ms** |
| A2a | Q 排布 `[1,K,h_q,d_qk]` | `forward_tokens_verify`(dsv4.rs:766) | — | 中 | A2 子项 |
| A2b | per-query top-k 索引 — **核心难点,tranche2 在此挂**:`dsv4_flashmla_decode_build_indices.cu` kernel 是为 `s_q=1` 写的,只填 row 0;s_q>1 要扩 kernel 给每个 query 位填各自 top-k(镜像 prefill 多查询 builder `arle_flashmla_csa/hca_build_indices`)。tranche2 做了 Rust glue(`query_start_pos`/`indices[max_s_q*topk]`)但 kernel 很可能没扩 → rows 1..s_q 是垃圾/回退 scalar → 结构性发散 + 3× 慢。**这是 .cu kernel 改,非纯 glue** | `dsv4_flashmla_decode_build_indices.cu` | — | **高** | A2 子项 |
| — | **A2 详设前置**:必须在 A1 干净基线上做(broken s_q=K 跑在带 rollback bug 的基线 → 垃圾被 confound)。A1 落地后重跑 `/tmp/tranche2_sqk_broken.diff` 隔离纯 glue bug,再细化 A2 实现级方案 | — | — | — | 顺序约束 |
| A2c | K 内因果 + cached-prefix mask | 同上 | — | 中 | A2 子项 |
| A2d | `sched_meta` for s_q=K(`get_meta(h_q,s_q)` 已支持) | `sparse_decode.h:41/383` | — | 低 | A2 子项 |
| A2e | s_q=K 回滚(若 A1 需要,snapshot 扩到 K) | `attention.rs` | — | 中 | A2 子项 |
| **A3** | **depth>1 EAGLE**(chain/tree,**复用单个 mtp.0 head 自回归,无需新权重**) | `mtp_forward`(dsv4.rs:1247);`num_nextn_predict_layers==1` | needle + 接受长度统计 | 高 | **~18→~8ms** |
| A3a | K 步自回归 draft chain(draft 回喂 mtp head) | `mtp_forward` | — | 中 | A3 子项 |
| A3b | tree verify + tree-attention mask(s_q=tree_size FlashMLA) | A2 之上 | — | 高 | A3 子项 |
| A3c | 接受长度记账(scheduler 多 token 已就绪) | `infer-core/lib.rs apply_output` | — | 低 | A3 子项 |

### Branch D — 单次 forward 的 kernel 成本（与 A 正交,每项乘进 EAGLE)

| id | what | files/sites | gate | risk | impact |
|----|------|-------------|------|------|--------|
| D1 | residual `wo` GEMV → DeepGEMM | `attention.rs` `dsv4/linear/wo`(设计已出:`2026-06-06-dsv4-decode-residual-gemv-fusion.md`) | needle + env-A/B | 低(proven) | 14.4% |
| D2 | mhc-fuse(TileLang `T.gemm(f32)`→bf16 `x_smem_16`) | `norm_fn_kernel.py:107`;3 call-sites | needle + A/B | 中 | 12.2% |
| D3 | comm:AllReduce overlap(已 `1b0222e7`)+ **AllGather/EP one-shot all-reduce** | `tensor.rs` comm_stream;TP all_reduce | needle + A/B | 中 | ~16%(AllGather) |
| D4 | decode CUDA graph(可捕获,launch overlap)— 仅配合 D3 one-shot 才有意义 | 已有 graph 路径 | needle + A/B | 中 | 小(B=1 launch 本已 overlap) |

### Branch E — prefill（"prefill 高性能"另一半,与 decode 正交)

| id | what | files/sites | gate | risk | impact |
|----|------|-------------|------|------|--------|
| E1 | fused-wqkv → 多 token prefill | 设计已出 `2026-06-06-dsv4-prefill-fused-wqkv-extend.md` | needle + prefill_ms A/B | 低(proven) | 22.8% |
| E2 | csa-select fused top-k(SGLang Indexer / skip_topk 跨层复用) | `dsv4_csa_select_kernel` | needle + A/B | 高(novel) | 17.7% |
| E3 | prepare-chain overlap(compressor/indexer 走 comm_stream 藏在 attention 后) | comm_stream fence | needle + A/B | 中 | 4.2%+ |

## 2. DAG（依赖图 + critical path）

```
            ┌─────────────────────────────────────────────────────────────┐
            │  G1  正确推理 gate（needle + same-twice + 自洽）              │  ← 所有验证的前置
            └─────────────────────────────────────────────────────────────┘
                      │ (gates every node below)
   DECODE 主干 ───────┼───────────────────────────────────────────────────────
                      ▼
   A1 compressor/indexer 回滚（无条件,代码已证 bug 真）──┐
      （A0 只是机制确认/floor,不是是否修的 gate)          │
                                                          ▼
                                                   A2e s_q=K 回滚
   A2a Q排布 ┐                                            │
   A2b 索引   ┼─▶ A2 s_q=K verify (~1.5×, →18ms) ◀─────────┘
   A2c mask   ┤          │
   A2d sched  ┘          ▼
                A3a chain ┐
                A3b tree  ┼─▶ A3 depth>1 EAGLE (~2.5-3×, →~8ms)
                A3c 记账  ┘
                              ▲
   DECODE kernel(并行,正交)──┘  乘数
   D1 wo-DeepGEMM ─┐
   D2 mhc-fuse ────┼─▶ 单次 forward 成本 ↓（26.6→~16-18ms 底座）
   D3 EP one-shot ─┤      （D3+D4 一起才解锁全 decode graph）
   D4 decode graph ┘

   PREFILL（完全正交,另一条线）
   E1 fused-wqkv→prefill ─┐
   E2 csa-select top-k ───┼─▶ TTFT ↓
   E3 prepare overlap ────┘
```

**Critical path 到 6ms**:`G1 → A1 → A2 → A3`(A1 无条件,代码已证 bug 真),**D 并行压底座**。
A0(dump/same-twice)是机制确认与非确定性地板,**不**决定 A1 是否做。

## 3. 6ms 预算（数字说话,license-or-kill 用 wall-clock）

| 阶段 | 机制 | tok/s | ms/token |
|------|------|-------|----------|
| 现状(已提交最优) | masked + fused-wqkv + on-device route | 37.6 | 26.6 |
| + D1+D2+D3(全实现,~40% 叠加打折) | 单次 forward 成本 ↓ | ~63 | ~16 |
| + A2(depth-1 s_q=K, α≈0.6, ~1.5×) | EAGLE 摊销 | ~94 | ~10.6 |
| + A3(depth-3 tree, α 高, ~2.5×) | 多 token 摊销 | ~160 | **~6.3** |

**结论**:6ms ≈ **D 全压底座 + EAGLE depth-3-class**。depth-1 单独只到 ~18ms;
**A3(depth>1)是真正跨过 6ms 的那一步**,且**不需要新权重**(单 mtp.0 head 自回归)。

## 4. 执行波次（拓扑序;⚠️ 都触 attention.rs,串行过 Codex 避免冲突)

- **Wave 0(现在)**:A1 compressor/indexer 回滚 snapshot/restore(**无条件,代码已证**);
  A0 dump + same-twice 仅作机制确认 + 非确定性地板。**Codex 在跑**。
- **Wave 1(并行无冲突)**:E1(prefill,proven)、D1(wo,proven)— 都触 attention.rs,
  rollback 决议后逐个串行落。
- **Wave 2**:A2(s_q=K verify,真加速;tranche2 diff 存 `/tmp` 可参照,但索引 A2b 要重写)。
  仅在 A0 给出回滚结论后。
- **Wave 3**:A3(depth>1 tree)、D2(mhc)、D3(EP one-shot)。
- **Wave 4**:D4 + 全 decode graph;E2/E3 prefill 收尾。

## 5. 立即下一步(Wave 0,Codex 在执行)

**A1 是无条件的**(代码已证 `truncate_decode_len` 不还原 running buffer = bug 真)。Codex:
(1) 实现完整 snapshot/restore(全 5 字段 × compressor+indexer,镜像 `reset()` 字段表),
draft forward 前快照、reject 时还原;(2) dump 仅作机制确认(哪些 buffer 残留 + draft 是否撞
compression 边界);(3) **用正确推理验证**:needle 取回 + same-config-twice 非确定性地板,
**不是** byte-identity 复刻基线。这区分了"修 bug"(无条件)与"验证 gate"(correct-inference)。
