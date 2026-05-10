# Post-W4A8 strategic axis ranking — chunked prefill vs Medusa vs xgrammar

> **2026-05-10 later update**: this ranking predates the Qwen3.5 Medusa
> rollback audit. Medusa remains a strategic non-classical spec-decode
> direction, but Qwen3.5 execution is blocked on recurrent-state
> accepted-length rollback before training/runtime work.

> 接续 `2026-05-09-w4a8-axis-saturation-audit.md`(W4A8 5 axis 3 KILL)。
> 评估 W4A8 后下一 strategic axis ROI,empirical-driven。
> **结论**:**chunked prefill > Medusa > xgrammar**(by ROI / wall-clock /
> evidence support)。

## 3 candidate axis 对比

| Axis | LOC | wall-clock | 风险 | 预估 gain | evidence 支持 |
|------|----:|-----------|-----|----------|---------------|
| **A. Chunked prefill** | 300-500 | 1-2 week codex | 中 | TTFT **-15-30%**(if prefill compute dominant 假设 hold)| ✅ nsys 实测 prefill 97% active GPU(`aaf0b55`)|
| **B. Medusa spec decode** | 500 + dataset + 1 week training | **2-3 weeks**(数据 + 训练 + 集成 + bench)| 高(α 真能 0.7?需训练才知)| tok/s **2-3×**(if α=0.7-0.85 hold)| ⚠ prediction-only,training risk |
| **C. xgrammar FFI** | 400-600 | 1 week codex | 低 | structured-output latency,不是 throughput axis | ⚠ 不同 metric,不直接对比 |

## Phase 0 evidence 支持

### Axis A — Chunked prefill

**直接 evidence**:nsys 60s decomposition `aaf0b55` 实测:
- prefill::compute = **327 ms / 337 ms active = 97% of active GPU time**
- decode::compute = 3% of active
- admission = 0.06% wall-clock

→ TTFT 改善的 #1 leverage 在 prefill compute path。

**Chunked prefill 原理**:
- 把长 prefill(seq=4096)拆成 K 个 chunks(K=4 → 1024 tokens each)
- 每 chunk prefill 后,scheduler 可以插入 decode steps(不必等全 prefill done)
- 长 prompt 的 first token 仍要等全 prefill,但 多并发场景 decode 可以 overlap

**预估 gain**:
- B7 c=4 4096-in 当前 TTFT 1614 ms → 2-3 chunks × ~500ms each + decode overlap
- 理论 TTFT 1100-1400 ms = **15-30% TTFT 改善**
- 多并发 throughput +5-10%

**风险**:
- 实施复杂度 中(scheduler dispatch 改 + KV cache update logic)
- vLLM/SGLang 已实现,可借鉴(Apache 2.0)

### Axis B — Medusa spec decode

**间接 evidence**(基于 Medusa-2 paper Vicuna 数据):
- α 0.7-0.85 → throughput 2.0-3.5×
- 但 4 KILL classical Leviathan(α≤0.25)显示 ARLE/Qwen3-4B 模型 spec 难度大
- Medusa **trained heads** 可能 break α ceiling,但需训练才知

**Wall-clock 障碍**:
- Phase 1a 数据下载 — blocker on `#34` HF Hub library
- Phase 1b 训练 — 1 week H100 / 2 weeks 4080S
- Phase 2 substrate 集成 — codex 1 week
- 总:**2-3 weeks before bench**

**风险**:
- 训练失败 risk(α < 0.5)→ 全部 wall-clock 浪费
- 集成复杂(tree attention + 4 heads + verification logic)

### Axis C — xgrammar FFI

**不同 metric 轴**:
- xgrammar = 结构化 generation(JSON schema enforcement)
- 不直接改 throughput / TTFT
- 价值在 **structured output use case**(API tool calls,SQL query gen 等)

**不直接 compete**:
- xgrammar 落地不影响 chunked prefill OR Medusa
- 可作为并行 axis(codex own,不阻塞其他)

## 推荐 strategic ranking

### 🥇 P0 — Chunked prefill(单 axis,最高 evidence-driven ROI)

- **直接 target nsys 97% prefill 主导信号**
- 中 LOC,1-2 week wall-clock(短)
- 风险中,有 vLLM/SGLang 参考实现
- **15-30% TTFT 改善 expected**

→ 这是当前 evidence-driven 最高 ROI axis。Codex pickup 推荐。

### 🥈 P1 (parallel) — xgrammar FFI

- 不同 metric,可并行
- 长期 product feature value
- 1 week codex 工作

### 🥉 P2 — Medusa(待 #34 解锁后再启)

- High potential gain(2-3× tok/s)
- 但 2-3 weeks wall-clock + 高 risk(α 不达预期)
- **推荐 等 P0 chunked prefill 落地后再启**(资源轮替)

## 综合 codex pickup 推荐

**当前 codex 在跑 SGLang baseline**(数据基准对照)。SGLang baseline 完成后:

1. **codex P0 pickup**:**Chunked prefill scaffold**(从 SGLang 抄 chunk 切分 logic)
2. **codex P1 parallel**:xgrammar FFI scaffold(已 plan)
3. **codex P2 后续**:Medusa(等 #34 + 数据准备完)

Claude 持续:
- nsys 实测 chunked prefill 假设(if codex 实施后)
- ncu profile when wrapper unblocks
- pickup queue housekeeping + audit

## Cross-references

- W4A8 axis 饱和:`docs/research/2026-05-09-w4a8-axis-saturation-audit.md`
- nsys 实证 prefill 97%:`docs/research/2026-05-09-eod113-p1a-nsys-decomposition-evidence.md`
- Medusa plan:`docs/plans/M_medusa-required-path.md`
- 4 spec decode KILL:`docs/experience/errors/2026-05-08-spec-decode-*-kill.md`
- xgrammar:`docs/plans/...xgrammar*`(待 plan)
- Chunked prefill plan:**未写**(本 brief 推荐写)

## 状态

W4A8 axis 饱和后 strategic axis ranking:**chunked prefill = P0 next**,
xgrammar = P1 parallel,Medusa = P2 后续(wall-clock 长 + 高 risk)。
建议 codex SGLang baseline 完成后直接 pickup chunked prefill scaffold。
