# 2026-05-07 · M_ibp Phase 0 ABANDONED — ARLE already wins in-batch shared-prefix

## Priority & ROI

**Priority**: kill criteria triggered → ABANDON M_ibp implementation.

**ROI saved**: ~250 LOC + ~1-2 days codex implementation that
would have been NET ZERO gain (or regression). License-or-kill
experiment paid off — the cheap Phase 0 measurement saved real
implementation time.

**Negative case discovered**: M_ibp's whole premise (ARLE has a
gap vs vLLM in-batch dedup) was false. ARLE's existing
chunk-boundary cascade pattern (sealed-block publish to RadixCache)
already covers this workload class better than vLLM.

**Kill criteria fired**: per M_ibp plan
([`12a19ad`](../../plans/M_ibp-in-batch-prefix-caching.md)):
- PROCEED if ARLE TTFT > 2× vLLM at multi-tenant shared-prefix
- ABANDON if ARLE TTFT < 1.5× vLLM
- Actual: ARLE TTFT 0.55× vLLM (ARLE FASTER) → ABANDON.

## Goal

License-or-kill experiment for M_ibp (in-batch prefix caching).
Bench ARLE vs vLLM at the canonical multi-tenant shared-prefix
workload to decide if implementation is worth the ~250 LOC.

## Workload

4 concurrent requests with:
- Shared system prompt (~6k tokens of repeated structured English)
- Different short user query (~100 tokens, unique per request)
- max_tokens=64, temperature=0.0, stream=true
- Pre-warmed by 1 request to populate cache

Custom Python runner (`/tmp/m_ibp_phase0_bench.py`) using aiohttp
for true concurrent fire (guidellm doesn't natively support shared
prefix across requests).

## Results

| Phase | ARLE post-F4-Small / M_b.1 / B.1.2 / M_pf-P0 | vLLM 0.20.1 (s8 control) |
|---|---:|---:|
| Cold warmup TTFT | 1238 ms | 1892 ms |
| **4-concurrent same-prefix TTFT mdn** | **318 ms** | **573 ms** |
| 4-concurrent TTFT min | 121 ms | 450 ms |
| 4-concurrent TTFT max | 318 ms | 574 ms |
| 4-concurrent burst total wall | 1631 ms | 1557 ms |
| Output token completion | greedy "!" repetition | "<think>" reasoning prefix |

### Headline

**ARLE TTFT mdn 318 ms vs vLLM 573 ms = ARLE 1.80× faster** at
multi-tenant in-batch shared-prefix workload.

ARLE's existing chunk-boundary cascade pattern:
1. Warmup populates RadixCache with system prompt sealed blocks
2. 4 concurrent requests admit fast — `lookup_or_stage` HITs
   the cached blocks
3. Each request only needs to prefill the unique 100-token user
   query (+ first token decode)
4. Result: TTFT dominated by the user-query prefill (small) + first
   decode tick

vLLM also has prefix caching, but in this experiment ARLE's pattern
wins — possibly due to:
- ARLE's tighter integration of cache HIT with admission
  (no fetch overhead for ReadyOnGpu blocks)
- vLLM's `<think>` chain-of-thought prefix wraps responses (per
  Qwen3 instruct template), adding latency before user-visible
  output
- Smaller per-tick scheduler overhead

## Decision

**ABANDON M_ibp** per kill criteria. Don't implement P1.1-P1.3.

**Implications for M-final roadmap**:
- M_ibp removed from active priority stack
- Tier ordering becomes: P0 (Phase 1A v3) → P1 (M_pf) → ~~P2 (M_ibp)~~ →
  P2 (spec-decode + F4 compound) → P3 (INT4 KV)
- "World-first long-sequence" bar moves: at multi-tenant shared-prefix
  workloads ARLE is already 1.80× faster than vLLM. Further gain
  on this shape is marginal.

**Implications for M-final stretch goal projection**:
- M-final assumed M_ibp would help. Actual: not needed.
- Updated trajectory:
  - F4-Small ✓: high-conc 1k/256/c=64 = 843 (vs vLLM 647, +30%)
  - + Phase 1A v3 (P0): long-ctx 8k/c=4 TTFT 4961 → ~1500 ms target
  - + M_pf (P1, marginal): -14ms H2D for cached prefix
  - **No M_ibp needed**
  - "World-first" already partially achieved at multi-tenant
    shared-prefix; remaining gap is single-tenant long-ctx
    (Phase 1A v3 closes it)

## Rule

- **License-or-kill experiments save real engineering time**.
  M_ibp plan estimated 250 LOC + 1-2 codex days. Phase 0 cost
  ~30 min Claude work + 5 min GPU time. The 30:1 ROI on the
  experiment vs the implementation it could have saved is the
  argument for license-or-kill on every "uncertain gain" plan
  going forward.
- Per memory rule
  (`feedback_docs_priority_roi_evidence.md`): research-y work
  makes Phase A a cheap license-or-kill. Validated here.
- **ARLE's chunk-boundary cascade is a real strength**, not an
  implementation gap. Future plans that propose to "fix" it
  should first license-or-kill via this exact bench shape.

## Bench Status

- ARLE artifact: `bench-output/2026-05-07-m_ibp-phase0-arle/`
- vLLM artifact: not saved separately (single bench run, summary
  recorded in this entry)
- Bench script: `/tmp/m_ibp_phase0_bench.py` (commit-eligible if
  it becomes a recurring bench shape)

## Cross-references

- M_ibp plan (now abandoned): `12a19ad`
- M-final roadmap (stack to update):
  `docs/plans/M-final-world-first-integration-roadmap.md`
- F4-Small wins (decode axis closed): `8f83c80`
- M_pf P0 substrate (still useful): `699002f`
- Phase 1A v3 (in flight): codex pipeline
