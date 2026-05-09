---
title: Prefill axis 重开 — multi-key bucket capture(非 prior single-key 实现)
date: 2026-05-09
type: research
status: hypothesis-license
depends_on:
  - docs/experience/wins/2026-05-09-bench-sglang-reverify-post-p1.0-p1.2.md
  - docs/experience/errors/2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md
  - docs/research/2026-05-09-chunked-prefill-already-exists-correction.md
---

# Prefill graph capture 重开 — SGLang 42-bucket multi-key,非 prior single-key

> 接续 codex N=3 SGLang reverify(`wins/2026-05-09-bench-sglang-reverify-post-p1.0-p1.2.md`)
> 落地的 SOLID gap 数据 + cross-reference M_pf-graph Phase 0 KILL 实际原因。
> **结论**:prefill graph axis **未架构性 KILL**,prior KILL 是 single-key
> cache + tail-1-token fallback 实现失败。SGLang `PiecewiseCudaGraphRunner`
> 42-bucket multi-key 完整实现是 Phase 0 KILL 反面。

## 1. SOLID 实测 gap(同机 N=3,σ < 1-2%)

| Workload | ARLE TTFT p50 | SGLang TTFT p50 | Δ | ARLE 强/弱 |
|----------|----:|----:|----:|----|
| **4k/256 c=4(prefill-dominant)** | **1639.3 ms** | **928.4 ms** | **+76.6%** | ❌ ARLE 弱 |
| 256/256 c=1(decode-dominant) | 13.2 ms | 36.1 ms | -63.4% | ✓ ARLE 强 |
| 256/256 c=4(decode-dominant) | 32.6 ms | 111.0 ms | -70.6% | ✓ ARLE 强 |

数据来源 `wins/2026-05-09-bench-sglang-reverify-post-p1.0-p1.2.md`(codex N=3,
package versions pinned `sglang==0.5.11 + sglang-kernel==0.4.2.post1+cu130 +
flashinfer-python==0.6.8.post1`,RTX 4070 Ti SUPER)。

**关键 single number**:**4k/c=4 prefill-dominant gap = +76.6% slower**,
**short decode 反向快 60-70%** → unequivocal:next axis **必须 in prefill path**。

## 2. M_pf-graph Phase 0 KILL cross-reference(避免重 KILL)

`errors/2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md` 实际 KILL 原因:

> Phase 0 did not remove enough launch overhead to matter for the real
> 4097-token request shape. The implementation captured valid 2048-token
> chunks, but the workload still had three prefill parts per request:
> 2048 + 2048 + 1. **The final 1-token tail fell back to eager**, and
> **the graph cache held only one key at a time**, so the runtime
> recaptured when alternating between `start_pos=0` and `start_pos=2048`
> across request groups.

→ Phase 0 KILL 是 **2 个 implementation failure**:
1. **Tail-1-token fallback**(每 request 第 3 chunk 是 1 token,落 eager → 失去 graph 收益)
2. **Single-key cache invalidation**(只 cache 1 key,start_pos=0 / =2048 alternate 必重 capture)

per `kernel-optimization` skill anti-pattern #6:**"License on 'capture exists'
not 'capture reused'"** — Phase 0 KILL 的根本是 **capture 没真的 reuse**。

**SGLang `PiecewiseCudaGraphRunner` 实现是 42-bucket multi-key**(`python/sglang/srt/model_executor/cuda_graph_runner.py`):
- 42 个 token-count buckets(覆盖 1, 2, 4, 8, 16, 32, …, 2048, 4096)
- 每 bucket capture 1 个 graph,**reuse across requests**
- Tail token 走最近的小 bucket(不是 eager fallback)

→ SGLang impl 完全 hits Phase 0 KILL 的两大 failure mode 反面。

## 3. 候选 axis ranking(post-codex-baseline)

### 🥇 P0 候选 — Multi-key bucket prefill graph(SGLang `PiecewiseCudaGraphRunner` port)

| 维度 | 评估 |
|------|------|
| **优先级** | P0 — 唯一 evidence-direct 4k/c=4 +76.6% gap 的 axis |
| **ROI 数字基础** | hypothesis:close 30-50% gap(if launch overhead 是真 binding constraint),i.e., TTFT 1639 → 1100-1300 ms |
| **LOC 估计** | **400-700**(SGLang impl ~600 lines,ARLE 现已部分 graph substrate 在 decode path)|
| **Wall-clock** | 1-2 weeks codex(实施 + multi-bucket lifecycle + tail handling)|
| **Risk** | 中(架构改动,但 SGLang 实现可参 + Phase 0 已学 single-key 失败教训)|
| **Negative case / Kill criteria** | TTFT 4k/c=4 Δ < 10% → **KILL**,Δ ≥ 25% → **strong proceed**(per `M_pf-graph-prefill-capture.md` license)|
| **Confounder** | Phase 0 KILL 同 4k/c=4 0 改善;新 impl 必须验证 **bucket cache hit rate** 单独 counter,避免重蹈 single-key |

**Phase 0 重启 license**(必须 SOLID 闭环):
1. ✅ 评估 SGLang 42 bucket coverage 对 ARLE Qwen3-4B 实际 prefill seq distribution(workload-aware bucket sizing)
2. ✅ 实施前确认 bucket cache hit-rate counter(per request)— 不是只 capture exists,要测 reuse
3. ✅ Tail-1-token handling — 走最小 bucket OR 改 chunk size 避免 1-token tail
4. ✅ Matched-control bench:**same KV format(BF16 vs FP8)**,**same admission policy** — 避免 Phase 0 contaminated 对照

### 🥈 P1 候选 — `SchedulerMixedPolicy::Mixed` A/B(low-cost,可立即 bench)

| 维度 | 评估 |
|------|------|
| **优先级** | P1 — 0 LOC,可立即 GPU 验证 |
| **ROI 数字基础** | hypothesis:c=4 多并发 mixed mode 让 decode 不 stall 等 prefill chunk;但 SGLang 默认 false 暗示 perf 不一定显著 |
| **LOC** | **0**(已有 substrate `--scheduler-mixed-policy mixed`)|
| **Wall-clock** | 1 hour bench × N=3 |
| **Kill criteria** | TTFT 4k/c=4 Δ < 5% with σ overlap → KILL |
| **Risk** | 低(纯运行时 flag,可 revert)|

**License threshold**:Δ ≥ 10% with σ < 5% across n=3 → wins,否则 KILL with errors entry。

### 🥉 P2 候选 — chunked_prefill_size sweep(diminishing returns)

| 维度 | 评估 |
|------|------|
| **优先级** | P2 — SGLang 默认 2048 与 ARLE 同,sweep 1024/2048/4096 估计 < 5% |
| **LOC** | 0 |
| **Wall-clock** | 2 hours sweep × N=3 |
| **Kill criteria** | 任 sweep 点 Δ < 5% 全 KILL |
| **预估 gain** | < 5%(SGLang 早 sweep 过同 GPU class)|

## 4. 不再考虑的 axis(per codex 决策树)

- ❌ **Trivial ops-layer dispatch wire**:已 KILL P1.3/P1.4/P1.6 三个,不再 attempt
- ❌ **Decode-only 优化**:codex 实测 ARLE decode-dominant 256/256 反向快 60-70%,decode 不是 binding
- ⚠ **Multi-tenant burst axis**(σ > 5% unstable,not strategic evidence,需 stable rerun before pickup)

## 5. Claude 立即可执行(this tick — GPU 释放 + codex 在自审 docs)

**Single-variable A/B,zero LOC**:
- ARLE B7 4k/c=4 baseline(已 1639 ms)vs `--scheduler-mixed-policy mixed`(P1 候选)
- 1 run × 60s × n=1(scout)→ if directional ≥ 10%,trigger N=3 full A/B

**或者** delegate codex(估计 review finishes in 15-30min):
- Codex pickup #36 PrefixAwareAdmission wiring(已在 task list)
- Or queue prefill-graph multi-key Phase 1 plan(估计 1-2w codex)

## 6. License-or-kill explicit thresholds(skill rule 8)

| Plan | License threshold | Kill threshold |
|------|------------------|----------------|
| **Multi-key prefill graph Phase 1** | TTFT 4k/c=4 Δ ≥ +10% with σ < 5% n≥3 | < +5%, OR ITL/tok-s regression > 5%, OR bucket cache hit < 50% |
| **Mixed policy A/B** | TTFT 4k/c=4 Δ ≥ +10% with σ < 5% n≥3 | < +5% within noise band |
| **Chunked size sweep** | 任一 size Δ ≥ +5% | 全 sweep < +5% |

## 7. 状态 + 下一步

**SGLang gap evidence-driven P0 axis = multi-key prefill graph(SGLang `PiecewiseCudaGraphRunner` port)**。
非 prior Phase 0 KILL 的 single-key 失败 axis,impl path 必须避开两大 failure mode。

**Claude 立即工作**:本 tick 写 P0 plan brief 后 commit,等 GPU + codex review finish 后再 trigger
mixed policy A/B(P1,low-cost)+ 让 codex pickup multi-key prefill graph Phase 1 implementation(P0,高 ROI)。

## Cross-references

- Codex SOLID baseline:`wins/2026-05-09-bench-sglang-reverify-post-p1.0-p1.2.md`
- M_pf-graph KILL 实际原因:`errors/2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md`
- Chunked prefill 已存在纠错:`docs/research/2026-05-09-chunked-prefill-already-exists-correction.md`
- SGLang `PiecewiseCudaGraphRunner`:`/tmp/sglang-chunked-src/python/sglang/srt/model_executor/cuda_graph_runner.py`(待 review)
- skill anti-pattern #6:"License on 'capture exists' not 'capture reused'"
