---
title: Prefill graph capture for W4-hybrid 实际 prerequisite — #24 不是 #37
date: 2026-05-09
type: research
status: architecture-correction
depends_on:
  - docs/research/2026-05-09-prefill-axis-reopened-multi-key-bucket.md
  - docs/experience/errors/2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md
---

# 前序 #24 W4A8 prefill graph capture hoist 是 #37 multi-key cache 的 hard prerequisite

> 接续 `2026-05-09-prefill-axis-reopened-multi-key-bucket.md` P0 = #37
> multi-key bucket prefill graph。Phase 0 source-grep `qwen35/prefill.rs`
> 发现 Qwen3.5 已有 graph capture substrate,但 **graphsafe gating
> `is_dense_bf16()` 拒绝 W4 weights**。codex baseline W4-hybrid model
> 即使 multi-key bucket port 也**不会 enter graph path**。

## 1. Phase 0 source audit

### Qwen3.5 已有 prefill graph capture substrate

`infer/src/model/qwen35/prefill.rs:301-310`:

```rust
let use_graph = self.supports_paged_prefill_graph();
if use_graph {
    let mut graph_state = std::mem::replace(&mut bufs.graph_state, CudaGraphState::new());
    graph_state.run_or_capture(&self.ctx, || {
        self.prefill_forward_paged_kernels(pool, recurrent, bufs, true)
    })?;
    bufs.graph_state = graph_state;
} else {
    self.prefill_forward_paged_kernels(pool, recurrent, bufs, false)?;
}
```

`infer/src/model/qwen35/prefill.rs:825-852`:

```rust
fn supports_paged_prefill_graph(&self) -> bool {
    self.enable_cuda_graph
        && self.layers.iter().all(|layer| {
            // gate every per-layer weight via graphsafe_batched_weight
            ...
        })
}

fn graphsafe_batched_weight(weight: &DeviceMatrix) -> bool {
    weight.is_dense_bf16()    // ← 只允许 BF16 dense weights
}
```

→ Qwen3.5 substrate **仅适用 BF16 weights**。W4 Marlin packed weights = **NOT
graphsafe**。

### Qwen3-4B-W4-hybrid-zpfix(codex baseline)实际路径

- 模型 family = qwen3(non-3.5),用 `infer/src/model/qwen3/`
- W4 hybrid Marlin weights → `is_dense_bf16() == false`
- Even if qwen3.5 substrate ported to qwen3 → `graphsafe_batched_weight` returns
  false for all W4 layers → graph capture path 永远 false

## 2. 真 architecture chain — 必须 #24 在前

| Stage | 内容 | 是否完成 | LOC |
|-------|------|----------|-----|
| **#24 W4A8 prefill graph capture hoist**(prerequisite)| Marlin W4A8 scratch hoist + graphsafe gating allow W4 packed weights | **pending(codex queue)** | 200-400 |
| **#37 Multi-key bucket cache + tail handling** | Cache 30+ graph keys,tail-1 token 走 small-bucket | pending | 200-300(scope 后) |
| **(post)benchmark 验证 close 4k/c=4 gap** | TTFT ≥ +10% with σ < 5% n=3 | pending | bench only |

**先 #24 → 后 #37**:#24 让 W4 graphsafe,#37 多 key cache 复用 capture。

#37 单独 land **without #24** = **零效果 on W4-hybrid 模型**(graphsafe 仍拒
W4)。

## 3. #24 hoist 工作详细 scope(基于源码 audit)

### 3.1 W4 weight graphsafe 化条件

需修 `qwen3/forward.rs` 或 `qwen3/prefill.rs`(看 W4 path 在哪):

| 条件 | 检查点 | 当前 | 改动 |
|------|--------|------|------|
| weight pointer 不每 call 变 | Marlin W4 weights `&self`(已 immutable pool-shared)| ✓ | 0 |
| scratch buffer 不每 call alloc | `run_marlin_w4a8_linear` line 1307+ allocs 5 buffers per call | ✗ | hoist scratch into `MarlinPrefillScratch`(类似 decode `MarlinDecodeScratch` per ca0673b)|
| activation tensor stride deterministic | prefill BF16→INT8 quant stride | ✓(deterministic per shape)| 0 |
| metadata refresh in replay | sequences / page_indices / start_pos | ✓(qwen3.5 already does)| port pattern |

**真 LOC 重点**:`MarlinPrefillScratch` lifecycle(类似 `MarlinDecodeScratch`)
+ `graphsafe_batched_weight` 加 W4 packed accept(W4 hybrid mark)。

### 3.2 LOC 估计修正(vs prior brief)

| 子 task | LOC | 说明 |
|---------|-----|------|
| `MarlinPrefillScratch` struct + alloc/lifecycle | 100-150 | hoist Marlin W4 scratch 进可 graph-capture 的容器 |
| `try_gemm_with_phase_into` 路径接受 prefill scratch | 30-50 | linear.rs:1872 None → Some(&mut prefill_scratch) |
| `graphsafe_batched_weight` extend W4 packed | 20-50 | 加 packed weight detection + graphsafe certify |
| qwen3 prefill.rs 加 `supports_paged_prefill_graph` + `run_or_capture` 套壳 | 50-100 | 借鉴 qwen35/prefill.rs:301-310 模式 |
| Phase 0 envelope clamp removal(per KILL fix #4)| 10-30 | scheduler.rs 不强 clamp prefill admission to 1 request |
| **总** | **210-380** | original #24 200-400 estimate ✓ |

### 3.3 KILL #2 (multi-key cache)scope

`#37` LOC 估计修正:

| 子 task | LOC |
|---------|-----|
| Multi-key cache(LRU 或 fixed 30-key)| 80-150 |
| Tail handling — chunk_size 改 OR 走 smallest bucket | 50-100 |
| Bucket-cache-hit-rate counter(per Phase 0 KILL anti-pattern #6)| 30-50 |
| **总** | **160-300** |

**Combined #24 + #37 = 370-680 LOC**(≈ SGLang `PiecewiseCudaGraphRunner` 600
lines original estimate)。

## 4. License chain(Phase 0v3 → Phase 0v4)

### Phase 0v3(#24 alone)license — graph capture 单独效果

- Bench:ARLE W4-hybrid baseline 1639 ms vs `INFER_PREFILL_GRAPH=1` Phase 0v3
- Predicted Δ:**< 5%**(因为 single-key cache + tail eager 仍存在,Phase 0
  KILL 同 issue)
- → 单独 license 不 expected。**作为 prerequisite,license skip**,只跑 functional smoke
  + nsys evidence 证明 W4 weights 进入 capture path

### Phase 0v4(#24 + #37 stack)license — 完整 close gap

- Bench:同 baseline vs Phase 0v4
- License threshold:TTFT 4k/c=4 Δ ≥ +10%, σ < 5%, n=3
- Strong proceed:Δ ≥ +25%(close 半 76.6% gap)
- KILL:Δ < +5% OR bucket cache hit rate < 50% (counter)
- Matched-control:**same KV format**(W4-hybrid auto KV mode),same admission
  policy,same num_slots(≠ Phase 0 BF16-forced contamination)

## 5. Codex pickup 推荐 batching

| 选项 | LOC 总 | wall-clock | 风险 |
|------|--------|-----------|------|
| **A: #24 + #37 一起 batch**(单大 PR)| 370-680 | 2-3 weeks codex | 高(大 PR 难审)|
| **B: #24 first → bench gating → #37 second** | #24:200-400 → #37:200-300 | 1.5-2.5 weeks codex(分两 wave)| 中(分批 audit)|
| **C: #24 only,#37 evaluate after**(若 #24 functional ok 再评估 #37 LOC) | 200-400 | 1 week | 低(分批 license) |

→ **B 推荐**(Phase-by-phase license-or-kill,符合 §0 SOLID Phase audit 原则)。

## 6. Action items

1. **更新 task #37 description**:加 blockedBy #24
2. **更新 task #24 description**:include this brief 的详细 scope + Phase 0v3 functional
   smoke 作 first license gate(non-throughput,只验 W4 进 capture path)
3. **不 commit code**(纯 docs research),不 attempt implementation 因为 codex queue work
4. **本 brief commit + push** 作为 codex pickup 时 reference

## 7. 总结 — strategic axis ranking 修正

之前 brief(`2026-05-09-prefill-axis-reopened-multi-key-bucket.md`)P0 = #37
multi-key bucket。**修正**:**#37 是 phase 2,#24 是 phase 1 hard prerequisite**。

| Axis | Phase | LOC | License gate | ROI |
|------|-------|-----|--------------|-----|
| **#24 W4A8 prefill graph capture hoist** | **Phase 1**(必先)| 210-380 | functional smoke + nsys 验 W4 进 capture path | prerequisite |
| **#37 multi-key bucket + tail handling** | **Phase 2**(必后)| 160-300 | TTFT 4k/c=4 Δ ≥ +10% σ < 5% n=3 | close 半 +76.6% gap(预估)|

## Cross-references

- Phase 0 KILL evidence:`errors/2026-05-08-m_pgc-phase0-killed-ttft-under-threshold.md`
- Qwen3.5 substrate:`infer/src/model/qwen35/prefill.rs:301-310, 825-852`
- Marlin W4A8 prefill alloc-bound:`infer/src/ops/linear.rs:1307+`
- Marlin decode scratch hoist precedent:`ca0673b` `MarlinDecodeScratch`
- 上一 brief:`docs/research/2026-05-09-prefill-axis-reopened-multi-key-bucket.md`
- Codex SOLID baseline:`wins/2026-05-09-bench-sglang-reverify-post-p1.0-p1.2.md`

## 状态

#37 作 P0 是错的;**#24 是 hard prerequisite**(W4 weights graphsafe certification)。
分阶 license:Phase 1 (#24) functional + nsys → Phase 2 (#37) throughput gate。
