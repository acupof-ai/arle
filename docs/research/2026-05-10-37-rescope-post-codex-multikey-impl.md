---
title: #37 re-scope post codex's #24 multi-key impl — only throughput bench remains
date: 2026-05-10
type: research
status: scope-reduced
depends_on:
  - docs/experience/wins/2026-05-10-bench-p24-w4a8-prefill-graph-hoist.md (codex draft, uncommitted)
  - docs/research/2026-05-09-37-multikey-vs-device-startpos-design.md
---

# #37 re-scope post codex's #24 multi-key impl — only throughput bench remains

> Codex's #24 W4A8 prefill graph hoist draft wins entry shows scope **超出**
> 我 brief 中的 Phase 1 prerequisite。已经实现 **multi-key graph cache 含
> page-layout + start_pos in key**,即 #37 Path A 路径(多键 cache)的核心
> substrate。**#37 真实剩余 = throughput bench validation + 可选 Path B
> device-mem 优化**。

## 1. Codex #24 实际实施(per uncommitted wins entry)

> "Added Qwen3 paged prefill graph resources keyed by exact layout: token
> count, page size, page index length, prefix rows, batch size, sequence
> lengths, start positions, and page count."

→ **multi-key cache 已实现**(8 维 key: token / page_size / page_idx_len /
prefix_rows / batch / seq_lens / start_positions / page_count)。

> "Added prefill-lifetime Marlin scratch using the existing decode scratch
> arena type, but with a prefill-specific config. Hybrid decode uses W4A16,
> while hybrid prefill uses W4A8, so decode scratch config is insufficient."

→ MarlinPrefillScratch + phase-specific config(已 hoist 到 prefill 路径)。

> "Allowed graph-safe batched weights for dense BF16, W4A16 Marlin, W4A8
> Marlin, and W4-hybrid with INFER_HYBRID_W4A8_PREFILL=1."

→ graphsafe gating extends to W4-hybrid。

**Functional smoke evidence**:
```
Qwen3 prefill graph capture key: tokens=8 batch=1 pages=1 prefix_rows=0 marlin_scratch=true
```

**Test gates**(全 PASS):cargo check / clippy / e2e(2/2)/ greedy_solo /
greedy_w4a8_marlin / W4-hybrid INFER_PREFILL_GRAPH=1 HTTP 200。

## 2. #37 原 brief vs 实际剩余

### 原 brief Phase 2(per `2026-05-09-prefill-axis-reopened-multi-key-bucket.md`)

| 子 task | 原 LOC | 实际状态(post codex #24)|
|---------|-------:|---------------------------|
| Multi-key bucket cache + tail handling | 200-300 | **已 codex 实施(8-维 key)**|
| Bucket cache hit-rate counter | 30-50 | **TBD**(codex 未提 counter) |
| Tail-1-token handling(2048+2048+1)| 50-100 | **TBD**(可能 multi-key 自动 cover,需测)|

### 原 brief Path B(per `2026-05-09-37-multikey-vs-device-startpos-design.md`)

| Path | LOC | Cache hit | 说明 |
|------|-----|-----------|------|
| **A** Multi-key cache(已 codex 实施)| 80-150 | 80-95% | start_pos 类别 fits 8-维 cache |
| B Device-memory start_pos | 100-200 | 100% | Single graph,SGLang upstream pattern |

→ Codex 选 Path A(更简单,direct fix)。Path B 仍可作 后续 优化(if Path A
cache hit < 80% 实测发现)。

### #37 真实剩余 work

```
Phase 1 (codex #24 done):
  ✓ Multi-key graph cache (8-d key)
  ✓ MarlinPrefillScratch hoist
  ✓ Graphsafe gating W4-hybrid
  ✓ Functional smoke + correctness PASS

Phase 2 (#37 真实剩余):
  ⏳ Throughput bench validation:matched-control 4k/c=4 vs codex baseline 1639ms
     - License threshold: TTFT Δ ≥ +10% with σ < 5% n=3
     - 必须 INFER_PREFILL_GRAPH=1 vs default OFF baseline
  ⏳ Bucket cache hit-rate counter(if Δ < +10%,确认 cache reuse 是否 binding)
  ⏳ Tail-1-token check(per Phase 0 KILL anti-pattern):测 c=4 4097-token
     workload 实际 chunk 边界(2048+2048+1 vs 2049+2048 vs other)
  ⏳ (optional Path B)Device-memory start_pos:if Path A cache hit < 80%,
     wire device-tensor pattern for further reduction

Phase 3 (defer):
  ⏳ Metal-side qwen35.rs / dflash.rs prefill graph mirror(needs Mac)
```

## 3. 修正 #37 task description

```
原: Multi-key bucket cache + tail handling impl ~160-300 LOC
新: Throughput bench validation + bucket cache hit counter + tail check
    ~50-100 LOC + 1-2 day bench
    Phase 2 hard prereq #24 已 done by codex (multi-key cache substrate)
    Path A vs Path B decision deferred to bench evidence
```

LOC reduction:160-300 → **50-100 LOC**(只剩 counter + tail-handling refinement
+ bench infra)。

Wall-clock reduction:1-2w → **2-3 days**(主 bench + analysis)。

## 4. 立即可执行(等 codex commit 后)

post-codex-commit immediate work(Claude OR codex):
1. Apply Phase 2 step 3 RoPE patch(`f9ad134`)→ Phase 2 step 3 closed
2. Phase 0v3 5-gate validation(`acb32ca`)— independent verification
3. **#37 throughput bench**:
   ```bash
   PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
     scripts/bench_guidellm.sh p37-w4hybrid-prefill-graph-on \
     --concurrencies 4 --max-seconds 120 --warmup 10 \
     --data 'prompt_tokens=4096,...,output_tokens=256,...'
   # OFF baseline: same command without INFER_PREFILL_GRAPH=1
   ```
4. License-or-kill per #37 thresholds

## 5. ROI 综合

**Codex 的 #24 implementation = Phase 1 + Phase 2 substrate 一并 land**(超 brief 范围)。
**Wall-clock 节省**:#37 从 1-2w → 2-3 days。
**LOC 节省**:#37 从 160-300 → 50-100。
**Risk reduction**:multi-key cache 已 ship,只剩 bench validation 简单 task。

## Cross-references

- Codex #24 wins draft:`docs/experience/wins/2026-05-10-bench-p24-w4a8-prefill-graph-hoist.md`(uncommitted as of 2026-05-10 ~00:48)
- 原 #37 brief:`docs/research/2026-05-09-prefill-axis-reopened-multi-key-bucket.md`
- 原 #37 Path A vs B 设计:`docs/research/2026-05-09-37-multikey-vs-device-startpos-design.md`
- Phase 0v3 validation protocol:`docs/plans/2026-05-09-prefill-graph-phase0v3-validation-protocol.md`
- Phase 2 step 3 RoPE patch:`docs/plans/2026-05-10-phase2-step3-qwen3-caller-optin-patch.md`

## 状态

#37 scope **大幅缩减** post codex #24 multi-key cache 已实施。从 implementation
work 转 bench validation work。LOC 50-100,wall-clock 2-3 days。等 codex
#24 commit 后立即可启 bench(matched-control 4k/c=4 with INFER_PREFILL_GRAPH=1
on/off)。
