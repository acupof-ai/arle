---
title: #37 multi-key cache vs device-memory start_pos — Path A vs Path B design
date: 2026-05-09
type: research
status: design-decision-pending
depends_on:
  - docs/research/2026-05-09-prefill-graph-w4-prereq-architecture.md
  - docs/experience/wins/2026-04-22-bench-guidellm-qwen35-paged-prefill-graph-start-pos-stability-guard.md
  - docs/experience/errors/2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md
---

# #37 design choice — multi-key cache vs device-memory start_pos

> Phase 0 source-grep `qwen35/prefill.rs` 的 paged-prefill graph 实际机制:
> commit `5c8aa81`(2026-04-22 wins)保留 single-key cache + invalidates on
> start_pos change → Phase 0 KILL 同根因。**两个 fix 路径**,选 B 更彻底。

## 现状(2026-04-22 wins entry §What Worked)

```
infer/src/model/qwen35/prefill_buffers.rs 现在 tracks the start_pos 
baked into the currently captured paged-prefill graph and clears 
that marker on graph invalidation.

infer/src/model/qwen35/prefill.rs 现在 invalidates and recaptures the 
full-forward paged-prefill graph when start_pos changes.
```

→ 当前 graph **single-key**,**start_pos 不同 = 重 capture**。这正是 Phase 0
KILL 病:c=4 4097-token 拆 2048+2048+1 → start_pos {0, 2048} 交替 → graph
反复 invalidate + recapture → graph 收益 = 0。

**Wins entry §Rule** 自己点出:
> graph reuse is only claimed for matching captured `start_pos` values 
> **until the prep kernel becomes parameter-stable across chunk offsets**.

→ Path B(prep kernel param-stable)在 wins entry 已被 acknowledge 为更彻底的
解,只是 deferred at that time。

## Path A — Multi-key cache(naive solution)

| 维度 | 细节 |
|------|------|
| **机制** | LRU 持 30 keys,key = `(token_count, start_pos, num_pages, page_size)` |
| **LOC 估计** | 80-150 |
| **当 c=4 workload** | start_pos {0, 2048} 各 1 key,**第二个 request 之后 cache hit** ✓ |
| **缺陷** | 1) start_pos = 4097(tail-1)仍需单 key,要求 chunk_size 严格分。2) Cache 占 GPU memory(每 graph capture ~MB scale)。3) Eviction 策略影响 hit rate(LRU OK,但 num_keys 选 30 vs 50 vs 64 需 sweep)。 |
| **预估 close gap** | 30-50%(if cache hit > 80%,hypothesis)|

## Path B — device-memory start_pos(SGLang 实际策略)

| 维度 | 细节 |
|------|------|
| **机制** | start_pos move 到 device tensor,prep kernel reads from device memory(not launch scalar)。Graph capture once,**replay refreshes device tensor before each launch**。 |
| **LOC 估计** | 100-200(prep kernel 改 + caller device-tensor lifecycle + replay refresh hook)|
| **当 c=4 workload** | **Single graph reused across all start_pos values** ✓ |
| **缺陷** | 1) Prep kernel 改 = 跨语言改动(CUDA C / TileLang DSL — depends on which prep kernel is on hot path)。2) Numerical correctness 验证 — device-mem read vs launch-scalar 数值等价 needs `cargo test --test greedy_consistency`。3) Replay 路径加 device-tensor refresh 调用 ≈ 1 small launch overhead per replay(< 1 μs typically)。 |
| **预估 close gap** | **40-70%** of +76.6%(more reuse = more savings)|

## Path 比较

| 维度 | A 多 key | B device start_pos |
|------|---------|---------------------|
| LOC | 80-150 | 100-200 |
| 风险 | 中(LRU 策略 + memory)| 中(prep kernel 改,数值验证)|
| Cache hit 上限 | 80-95%(start_pos 类别 fits 30-key cache)| **100%**(无须 cache,single graph)|
| Memory 占用 | 30 graph captures × ~1MB = 30 MB | 1 graph capture × ~1MB = 1 MB |
| 真实 close gap 预估 | 30-50% | **40-70%** |
| 后续可叠加 | -- | + bucket cache by token_count (chunked workload) → 可能再 + 5-10% |
| SGLang 实际策略 | -- | **B**(per `PiecewiseCudaGraphRunner` 实现:metadata 全 device-tensor)|

→ **Path B 推荐**(更彻底 fix + 更大 close gap + SGLang 已 validate)。

## #37 scope 修正(if Path B selected)

| 子 task | LOC | 说明 |
|---------|-----|------|
| start_pos device tensor 在 PrefillGraphState 持有 | 30-50 | `start_pos_dev: cudarc::DeviceMemory<u32>` |
| Prep kernel(prefill metadata 准备)改 read device | 50-100 | locate prep kernel(`prep_paged_prefill_metadata` etc)→ scalar param 换 device pointer |
| Replay refresh hook(每 launch 前 host→device copy start_pos)| 30-50 | 1 H2D copy per replay,≪ 1 μs |
| Tail handling(避免 1-token chunk OR 走 small bucket)| 30-50 | adjust chunk_size 避免 4097 / 2048 = 2 + 1-tail (e.g. chunk_size 2049 让 4097 / 2 = 2 even chunks) |
| Bucket cache for token_count(可选 v2)| 60-100 | optional second axis,if Path B 单独不够 close gap |
| 单元测试 + greedy consistency | 30-50 | 验证 device-mem start_pos 数值等价 |
| **总(Path B 单独)** | **170-300** | |

→ #37 LOC 范围 **170-300**(原估计 160-300 ≈ same,但 Path B 更彻底)。

## License-or-kill criteria 修正

| 维度 | Path A | **Path B(推荐)** |
|------|--------|---------------------|
| Cache hit rate counter | bucket cache hit > 80% | N/A(single graph,trivially 100%)|
| Throughput gate | TTFT 4k/c=4 Δ ≥ +10% σ < 5% n=3 | TTFT 4k/c=4 Δ ≥ **+15%** σ < 5% n=3(strong threshold,因 Path B 预估 close gap 大)|
| Strong proceed | Δ ≥ +25% | Δ ≥ **+35%** |
| KILL | Δ < +5% OR cache hit < 50% | Δ < +10%(Path B 已无 cache hit fallback,纯 graph reuse measure)|
| Numerical correctness | greedy_consistency PASS | greedy_consistency PASS(device-mem read 数值等价 critical)|

## Action items

1. **Update #37 task description**:specify Path B preferred,LOC 170-300,license Δ ≥ +15%
2. **不动手实施**(等 codex #24 完成后 brief 给 codex pickup)
3. **本 brief commit + push** 作 #37 codex pickup 时 reference

## Cross-references

- 当前 single-key 实现:`infer/src/model/qwen35/prefill_buffers.rs` + `prefill.rs`(per 5c8aa81)
- Phase 0 KILL same root cause:`docs/experience/errors/2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md`
- Wins entry 自 acknowledge defer Path B:`docs/experience/wins/2026-04-22-bench-guidellm-qwen35-paged-prefill-graph-start-pos-stability-guard.md`
- #24 prereq architecture:`docs/research/2026-05-09-prefill-graph-w4-prereq-architecture.md`
- Phase 0v3 validation protocol:`docs/plans/2026-05-09-prefill-graph-phase0v3-validation-protocol.md`

## 状态

#37 design 选 **Path B**(device-memory start_pos,SGLang `PiecewiseCudaGraphRunner` 模式)
over Path A(multi-key cache)。LOC 170-300,license Δ ≥ +15% 4k/c=4 TTFT。
不阻塞 #24 codex 当前进度,**等 #24 commit + Phase 0v3 validation 通过 → brief #37 Path B**。
