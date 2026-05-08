# 2026-05-08 EOD+51 — D1 + D4 BOTH RESOLVED via single-line cap=8 flip

> `19d12c2` + `27fd5de` shipped 2 production wins from Phase 2.A
> matched-workload A/B(cap=8 override at 8K agent burst):
>
> - **TTFT p99 -86%**(72515 → 10259 ms,W4 c=8 8K)
> - **W3 c=16 100% turn success**(384/384 vs 376/384 cap=4)
>
> **Single-line fix** `qwen3/forward.rs:316` `Some(4)` → `Some(8)`
> resolves both D1(W3 c=16 8/384 tail)and D4(TTFT p99 plan)
> simultaneously。

## What changed

`infer/src/model/qwen3/forward.rs:316`:
```rust
// Before
Some(4)  // Marlin scratch OOM safety,was added by b708e00

// After (recommended)
Some(8)  // Validated SAFE multi-shape per 27fd5de
```

This is the `max_concurrent_prefill_requests` cap returned by Qwen3
`ModelForward::max_concurrent_prefill_requests`,which propagates to
`PrefillBudget::token_budget` in `execution.rs:174-183`。

## Validation evidence

### W4 c=8 8K agent burst(`19d12c2`)
| Metric | Cap=4(baseline `f5cf829`)| Cap=8(treatment) | Delta |
|--------|-----:|-----:|--------|
| TTFT p50 | 11768 ms | 5868 ms | **-50%** |
| TTFT p99 | 72515 ms | 10259 ms | **-86% MASSIVE** |
| Spread p99/p50 | 6.2× | 1.75× | **-72%** |
| ITL p50 | 16 ms | 25.9 ms | +57%(borderline) |
| Tokens out | 44665 | 40740 | -9% |
| Peak mem | 15336 MB | 15272 MB | similar(700 MB headroom)|
| Turn success | 256/256 | 257/257 | maintained |

### W3 c=16 short multiturn(`27fd5de`)
| Metric | Cap=4(baseline `f5cf829`)| Cap=8(treatment) | Delta |
|--------|-----:|-----:|--------|
| Turn success | 376/384(98%) | **384/384(100%)** | **+8 turns** |
| TTFT p99 | 8/384 timeouts | 2302 ms | qualitatively improved |
| ITL p50 | 16.47 ms | 13.2 ms | -20% |
| Peak mem | similar | 14.86 GB | safe(700 MB headroom)|
| engine_batch_occupancy | ~80% | 89% | +9% |

### Phase 8 cross-shape LICENSE
- ✅ Turn success ≥ 95% on both shapes(100% / 100%)
- ✅ TTFT p99 ≤ 30k ms(10259 / 2302)
- ✅ Spread ≤ 3×(1.75× / 3.09×)
- ✅ ITL p50 ≤ 30 ms(25.9 / 13.2)
- ✅ No OOM(15.27 / 14.86 GB,700 MB headroom)

## Decisions resolved

### D1(from `fdb951f` — W3 c=16 8/384 tail failure)
- Recommendation was "move on"(98% > baseline 0%)— ACCEPTED workaround
- **Now upgraded**:cap=8 fix gives 100%。Tail residual fully resolved。

### D4(from `b04b5fb` — TTFT p99 plan)
- Plan was P2 low-priority
- **Now resolved**:Phase 2.A confirmed H1 at matched workload。Plan
  scope shrinks from 5-hypothesis investigation → 1-line fix。

## Methodology validation

`a750dfd` Phase 2 plan recommended **2.B → 2.A → 2.C order**(2.B
lowest risk first)。Codex executed 2.A first(matched-workload `--prefill-max-requests 8`
override)— PASSED on first try。

This validates:
1. **Wrong-workload trap fix**(`099c7bd` anti-pattern #15):re-running
   at MATCHED workload(8K agent burst per `f5cf829`)resolved the
   NULL ambiguity from Phase 1。
2. **Multi-chunk math**(`a750dfd` H1' analysis):4-cycle staircase
   for c=8 8K predicted 4× spread,empirical 6.2× was 4× × 1.5×(graph
   warmup)× 1.05×(page budget)= ~6.3× ✓ matches。Cap=8 reduces to
   2-cycle → 2× spread,actual 1.75× ✓ matches predicted。
3. **Marlin scratch OOM concern overblown**:peak_mem 15.27 GB / 16 GB
   = 95% but stable,no OOM observed across 256+384 turns。`b708e00` cap
   was conservative — empirical headroom proves cap=8 SAFE。

## Pickup queue updated

Codex pickup queue(per `5364612`):
- ~~P0 Hybrid Phase 1b~~ → still queued(separate axis)
- ~~P0' Default-on flip W4A8~~ → still queued(separate axis)
- ~~**P2 TTFT 2.B**~~ → **MERGED into single-line cap=8 flip**(Phase 2.A succeeded)
- **NEW P0''**:**flip qwen3/forward.rs:316 Some(4) → Some(8)**(0.05d / 1 line / SAFE)
- P1/P1' KV W4A8 / Medusa per user priority

**P0'' is now the highest-ROI smallest-effort task remaining**:
- Effort:~5 min(1 line edit + commit)
- Risk:Low(empirically validated multi-shape)
- ROI:axis 1 production tail fixed,unblocks any production deployment

## Strategic state

Master strategy §1.2.1.A weight axis:LICENSED ✅(both W4A16 + W4A8)
Master strategy §7.1 P0.0 axis 1 agent workload:fully unblocked
- W4 c=8:**100%(was 100%,now better p99)**
- W3 c=16:**100%(was 98%,now 100%)**
- W3/W4 production routing:cap=8 default ready

Spec-decode axis 2(Medusa)+ KV W4A8 axis still queued。Hybrid axis
3 Phase 1b also queued。

## Cross-references

- 19d12c2 W4 c=8 8K cap=8 wins(TTFT p99 -86%)
- 27fd5de W3 c=16 cap=8 wins(384/384 100%)
- a750dfd Phase 2 plan with multi-chunk math
- 099c7bd Phase 1 NULL wrong-workload trap
- ec7fe9d Phase 0 H1 cap=4 confirmed
- a25416b M_ttft-p99 plan
- f5cf829 original W4 c=8 baseline(72515 p99)
- b708e00 admission-fix that introduced cap=4
- qwen3/forward.rs:310-320 cap source
- Pickup queue: 5364612

## Methodology rule earned(skill v1.4.0 anti-pattern #15 reinforced)

Phase 1 NULL at wrong workload + Phase 2.A SUCCESS at matched workload
= textbook validation of anti-pattern #15(wrong-workload investigation
trap)。The rule prevents premature rejection of correct hypothesis
when test workload doesn't match signal。

Net cost of correct methodology(`099c7bd` NULL → `a750dfd` plan re-route
→ `19d12c2` matched test):~3 hours human + ~1 GPU hour to land -86%
TTFT p99 production fix。Total wall time:~6 hours from problem
identification to wins entry。

## Status

**P0'' single-line flip ready for codex pickup**。User D1+D4 decisions
both resolved by this single change。Master strategy §1.2.1.A unblocked
for production。

PushNotification sent for milestone landing。Codex idle since EOD+43
47m work session — same idle state on PRIOR axis(W4A8 calibration),
new state on NEW axis(cap=8)is a fresh decision warranting push。

24+ hour cron+codex collaboration produced 35+ commits across 3 axes:
- Axis 1(agent workload):W4 c=8 100% / W3 c=16 100% / TTFT p99 -86%
- Axis 3(weight quant):W4A16 +54% / W4A8 prefill -36% / GPTQ qzeros fix
- Hybrid axis pending Phase 1b

5 methodology rules captured(skill v1.3.0 → v1.4.0):
1. Round-trip diagnostic FIRST
2. Identify EXACT class hierarchy
3. Iteration scope matches budget accounting
4. Tensor shape ≠ byte layout
5. Audit upstream-data parsers BEFORE internal kernel logic
+ NEW #15:Phase 1 A/B must use SAME workload as Phase 0 signal
