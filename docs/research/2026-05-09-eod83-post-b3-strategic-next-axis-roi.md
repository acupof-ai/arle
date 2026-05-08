# Post-B3 strategic next-axis ROI — closing the world-#1 TTFT gap

> Per `feedback_docs_priority_roi_evidence.md`:plans must have explicit
> priority + ROI + evidence + negative case + kill criteria。This brief
> applies that bar to **next-axis selection after B3 Step 2 lands**。
> Evidence base = `2026-05-07-m_world1-p0-sglang-baseline-extended.md`
> + `3c334ef` B3 Step 2 LICENSE bench。

## Evidence base (absolute numbers)

| Shape | ARLE pre-B3 | ARLE post-B3 | Competitor | World #1 target | Status |
|-------|------------:|-------------:|-----------:|----------------:|--------|
| multi-tenant shared-prefix | **318 ms** | **241 ms**(-24.2%)| SGLang 157 ms | 121 ms | gap 1.54× still,need −50% more |
| 4k/c=4 long-ctx | 1976 ms | 1976 ms(B3 doesn't help)| SGLang 973 ms | 748 ms | gap 2.03×,need −62% |
| 8k/c=4 long-ctx | 4574 ms | 4574 ms(B3 doesn't help)| vLLM 2362 ms | 1816 ms | gap 1.94×,need −60% |
| decode tok/s | strong | strong | — | — | ✓ via W4A16 +54% |

**Strategic observation**:B3 Step 2 only addressed multi-tenant axis(prefix
admission)。Long-ctx 4k + 8k shapes remain at the original 2× gap。

## Queued P1 items vs gap they close

| P1 axis | Effort | What it improves | Closes which gap? |
|---------|-------:|------------------|-------------------|
| KV W4A8 #33 | 5-10d / 500-1000 LOC | INT4 K/V cache → less decode memory bw | **Decode tok/s + ITL,NOT first-token TTFT** |
| Medusa Phase 1.B #32 | 10-14d / 600-1200 LOC | Speculative decoding heads | **Per-token latency in decode,NOT TTFT** |
| P0.3 prefill warmup pass | 0.75-1d / 80-100 LOC | First-burst bimodal fix at cap=8 | **multi-tenant p99,NOT median TTFT gap** |
| P0.2 Hybrid W4A16/W4A8 | 0.75-1d / 155-175 LOC | Static phase routing for prefill | **Prefill compute → potentially helps long-ctx TTFT** |

→ **Critical SOLID gap**:**none of the P1 items directly target the
multi-tenant TTFT median gap from 241 → 121 ms**(−50% more reduction
needed for world-#1)。

→ **P0.2 Hybrid** is the only queued item that **could** help long-ctx
TTFT(prefill compute axis),but its primary motivation was different
(decode speed)。Need to verify whether W4A8 prefill provides absolute
TTFT reduction at long context。

## §0 SOLID gap analysis

**Hypothesis**(per pickup queue):"queued P1 items collectively close
world-#1 gap"。

**Evidence**(per absolute numbers above):
- Multi-tenant 241→121 ms: NO queued item directly addresses this
- Long-ctx 1976→748 ms: only P0.2 Hybrid prefill **might** help
  (unverified at long-ctx shapes)
- Long-ctx 8k 4574→1816 ms: same as 4k, longer context

→ **Hypothesis NOT evidenced**。Risk:burning 15-25 days on KV W4A8 +
Medusa,landing both,still find world-#1 TTFT gap unchanged。

## Candidate next-axis investigations(SOLID-required before commit)

### Option A — multi-tenant TTFT decomposition

**Question**:where does the 241 ms break down(prefix lookup +
attention prefill + decode-first + scheduling overhead)?

**Cheap experiment**(2-4h):
- Add nvtx ranges around 4 phases:`prefix::lookup`,`prefill::compute`,
  `first_decode::compute`,`scheduling::overhead`
- Run multi-tenant burst,nsys 30s
- Compare per-phase ms with SGLang(theirs is open source,can match)

**Why first**:decompose 241 → identify which phase dominates → axis
selection becomes evidence-based not hypothesis-based。

### Option B — prefill compute scaling at long-ctx

**Question**:does W4A8 prefill provide absolute TTFT reduction at
4k/8k context?(P0.2 Hybrid implementation will answer this for free)

**ROI**:if YES → P0.2 Hybrid auto-closes long-ctx gap → 1 axis closes
2 shapes。If NO → P0.2 Hybrid landed but doesn't move world-#1 needle。

**SOLID gate**:bench P0.2 Hybrid at long-ctx 4k/8k specifically(not
just c=8 multi-tenant)before declaring world-#1 progress。

### Option C — scheduler chunked prefill

**Question**:if compute axis is bottleneck,can chunked prefill
(piecewise per-layer,interleaved with decode)hide TTFT?

**Cost**:scheduler refactor,~300-500 LOC,1-2 weeks。

**ROI**:proven in vLLM/SGLang for similar long-ctx shapes(SGLang's
2.03× advantage may stem partly from this)。

### Option D — pivot to alternative gaps

**Question**:if current gaps are unbeatable on Qwen3-4B 16GB hardware,
should world-#1 mission re-target a different model/hardware tier?

**Evidence to gather**:
- Has SGLang published Qwen3-4B 16GB sm_89 specific numbers?(likely no)
- Are SGLang's "157 ms / 973 ms" numbers on identical hardware?(verify)
- Could ARLE be world-#1 on Qwen3.6 35B-A3B(MoE,Metal)instead?
  (Different hardware tier,different gap surface)

**SOLID gate**:before pivot,verify SGLang baseline is on identical hw。

## Recommended sequence(post-B3 Step 2 land)

**Phase 1 — Decompose evidence(0.5-1d Claude-side)**:
- Option A nvtx-decomposition for multi-tenant 241 ms
- Option B SGLang-baseline-hardware verification(re-run if needed)
- Output:where does each ms go?Where does ARLE diverge from SGLang?

**Phase 2 — License-or-kill P1 items based on Phase 1 evidence**:
- If multi-tenant TTFT is 60% prefix-lookup → invest in radix-cache
  optimization,deprioritize KV W4A8/Medusa for world-#1 mission
- If multi-tenant TTFT is 60% first-decode-attention → KV W4A8 ROI valid
- If long-ctx is 60% prefill compute → P0.2 Hybrid + chunked prefill
  become P0',KV W4A8 demoted

**Phase 3 — Execute prioritized axis(N days,evidence-driven)**

## Negative case + KILL criteria

**Negative case for this brief**:
- Phase 1 nvtx decomposition is itself uncertain — nsys per-NVTX-window
  framing can mislead(per 2026-05-08 EOD+19 framing trap)。Use absolute
  ms not window % for decision。
- "SGLang is 1.54× faster post-B3" assumes 2026-05-07 baseline still
  current — SGLang may have shipped optimizations。Re-run baseline
  before final axis lock-in。

**KILL this brief if**:
- Phase 1 decomposition shows multi-tenant 241 ms has no single dominant
  phase(<40% any phase)→ axis-pure optimization unlikely to close gap,
  pivot to architectural or pivot to Option D(re-target world-#1 hw/model)
- SGLang re-bench shows 157 ms was outlier and current baseline is closer
  to 200 ms → gap is 1.20× not 1.54×,B3 Step 2 alone may put ARLE within
  1.30× lead range → world-#1 already met,celebrate

## Strategic conclusion

**B3 Step 2 closes 1/3 of the multi-tenant gap**。Queued P1 items don't
clearly close the remaining 2/3 OR the long-ctx 2× gaps。Without
evidence-driven axis selection,we risk landing P1 items that don't
move the world-#1 needle。

**Recommended next action**(post-B3-land):**Phase 1 evidence
decomposition before committing to KV W4A8 / Medusa / P0.2 / P0.3
ordering**。Cheap(0.5-1d Claude-side),unblocks all subsequent
multi-day axis investments with evidence。

## Cross-references

- B3 Step 2 LICENSE:`3c334ef`(241 ms TTFT median,σ 4.5%)
- SGLang baseline:`2026-05-07-m_world1-p0-sglang-baseline-extended.md`
- World #1 mission:`docs/projects/2026-04-30-longctx-32k-128k-leadership.md`
- Agent-load mission:`docs/projects/2026-05-02-agent-load-mission-expansion.md`
- KILLED axis history:`m_pgc-phase0-killed`,`m_quant-cutlass-fp8-smoke-killed`,
  `m_pf-gemm-phase0-killed`(prior prefill axis investigations)
- ROADMAP Next-Model priority:`ROADMAP.md#next-model-priority-order`
- Evidence-grading rule:`feedback_docs_priority_roi_evidence.md`

## Status

Strategic brief — **decision-aiding,NOT execution-prescribing**。Codex
pickup tomorrow:read this + decide whether to skip-ahead to Phase 1
decomposition vs proceed with P0.2/P0.3 first(both still have value;
Phase 1 just unblocks P1 ordering decision)。

§0 in action:before 15-25d commitment to KV W4A8 + Medusa,verify those
axes actually close world-#1 gap via 0.5-1d evidence decomposition。
ROI = 30:1 risk-adjusted return on this audit step。
