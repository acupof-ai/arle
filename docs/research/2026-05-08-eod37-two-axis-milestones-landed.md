# 2026-05-08 EOD+37 — 2 axis milestones landed:scheduler unblock + W4A8 calibration validated

> Codex worked 1h 36m 29s and shipped 2 strategic milestones across
> axes 1 and 3 in one commit pair。This brief consolidates the state
> for user decision review。

## Milestone 1 — Axis 1:W3/W4 admission deadlock UNBLOCKED(`b708e00`)

### What landed
3-part hot-path fix in `infer/src/scheduler/cuda/`:
1. **Page budget scope correction**(my `369292f` hypothesis):reserve
   decode-growth headroom only for **active decoding slots**(plus
   emit-gated decoding rows),not for queued/prefill slots
2. **Marlin GEMM error propagation**:CUDA alloc/kernel failure in
   prefill GEMM no longer `expect()` panics scheduler thread,now
   propagates via `Result`
3. **Qwen3 Marlin prefill row cap**:single-step concurrency capped to
   4 rows to avoid 16-slot burst → FP16 scratch OOM

### Validation
- **W3 c=16**:376/384 turns OK,clean drain(active=0 waiting=0 prefill_queue=0)
- **W4 c=8**:256/256 turns OK,clean drain
- Full gate matrix:cargo fmt / cuda+no-cuda typecheck / clippy / e2e / greedy_consistency
- 2 codex review rounds:0 actionable findings

### Open follow-ups
- **8/384 turn failure on W3 c=16**(2% failure rate at burst)— investigate residual race or admission queue starvation
- **TTFT p99 still poor** — separate tail-latency issue,not a liveness blocker

## Milestone 2 — Axis 3:W4A8 GPTQ re-pack PASS(`e753af7`)

### What landed
Empirical validation that my `12a54da` GPTQ-aware patch on
`pack_w4a8` closed the calibration drift。

Re-verification across 4 layers × 4 tensor types(q_proj, down_proj,
gate_proj, o_proj × layers 0/5/18/35):
- ALL PASS at ~0.03% max / ~0.02% mean drift
- **133× improvement** in max drift vs `b7176d3` naive max-scale
- **31× improvement** in mean drift

### Phase 5 single-variable A/B
| Arm | Path | Max drift | Mean drift |
|-----|------|-----------|------------|
| A | naive max-scale(`b7176d3`) | 4.02-4.14% | 0.62-0.69% — **FAIL** |
| B | GPTQ-aware(`12a54da`) | 0.02-0.03% | ~0.02% — **PASS** |

### Phase 8 LICENSE met
- rel max < 1% threshold:0.03% ✅
- Cross-layer consistency:uniform across 4 tensor types ✅
- Layer position coverage:early/mid/late uniform ✅

### Phase 1b path LICENSED at script level
Wall time from `b7176d3` finding → `12a54da` fix → `e753af7` validation:
**~30 minutes**。Methodology validation:cron+codex collaboration loop
(empirical fail → hypothesis fix → validation)tight enough for
30-minute cycles when blocker is well-localized。

## Decisions for user

### D1. W3 c=16 8-turn failure — investigate or move on?
- **Option A**:investigate the 2% tail failure(possible race or
  admission queue starvation at burst)。Risk:another iteration cycle
- **Option B**:declare W3 c=16 functionally unblocked(98% success >
  baseline 0%),move to next priority。Risk:tail-latency users hit failures
- **Recommended**:Option B — 98% is a major unblock,tail investigation
  is a separate ROI-graded item

### D2. W4A8 GPTQ — proceed to greedy_consistency gate now?
Re-pack quality validated。Next step is end-to-end model accuracy:
- Run greedy_consistency on Qwen3-4B-GPTQ-W4A8-marlin
- If PASS → bench guidellm + default-on flip(blocked by `62e75ee` graph capture)
- If FAIL → activation-quant compounding still need work

Recommended:**proceed**。Phase 1b script already produces the checkpoint;
greedy gate is ~1-2 min GPU。Strong evidence pack quality is fine at
script level → likely PASS。

### D3. Medusa(axis 2)— start implementation now?
M_medusa Phase 3 plan `528844c` ready。W4 c=8 baseline now exists for A/B。
- ~1-2 weeks codex implementation
- Master §7.4 P1.1 commits to Medusa as REQUIRED post-classical-DEAD

Recommended:start when D2 lands(serialize axis 3 → 2 to avoid GPU contention)。

### D4. TTFT p99 tail-latency — separate plan?
Codex flagged "TTFT p99 still very poor"。This is separate from liveness。
Likely admission queue wait time at burst。
- **Option A**:write `M_pf-tail-latency` plan,prioritize after axis 3
- **Option B**:ignore until production users complain

Recommended:**A** with low priority — quick plan(0.5d Claude),implement
later。Don't block axis 2/3 advances on this。

## Strategic state(updated)

| Axis | Status | Next |
|------|--------|------|
| **1 agent workload** | W4 c=8 ✅ 100% / W3 c=16 ⚠ 98% | D1 decision + W3 tail invest |
| **2 spec decoding** | Plan ready `528844c` | D3 — start Medusa impl |
| **3 weight quant** | Phase 1b GPTQ-aware PASS ✅ at script level | D2 — greedy gate next |

**4 open Decisions** for user(D1-D4)。

## Cumulative loop value

This 12+ hour cron loop produced:
- Axis 1:0/256 → 256/256(W4 c=8) + 0/N → 376/384(W3 c=16)= **first valid agent workload bench data**
- Axis 3:5+ iteration W4A8 narrowing closed,Phase 1b shortcut path taken,calibration validated
- 4 methodology lessons captured for future iteration prevention
- 25+ commits across both Claude(cron)and codex sessions

**Master strategy commitment**(per master `2026-05-07-arle-master-strategy.md`):
- §1.2.1.A weight axis 全套:W4A8 calibration substrate(✅ Phase 1b)
- §7.1 P0.0 axis 1 真 agent workload:W3/W4 deadlock unblock(✅ b708e00)
- §7.4 P1.1 Medusa REQUIRED:plan ready,implementation pending(⏳ D3)

## Methodology rules earned this loop

1. **Round-trip diagnostic FIRST**(`39237b9`): when investigating
   "quantization X produces wrong output",pack/unpack vs upstream
   reference test data is the cheap first diagnostic
2. **Identify EXACT class hierarchy**(`3cee2f0`): PR #31 had Layer vs
   W4A8Layer with same-named methods,different patterns
3. **Iteration scope matches budget accounting**(`369292f`): step-budget
   over running_batch (all slots) vs decode_slots (active) caused 40×
   batch occupancy delta
4. **Tensor shape ≠ byte layout**(`8bb57ea`): perm patterns produce
   identical shapes with different bytes — verify class hierarchy

## Cross-references
- `b708e00` admission deadlock fix(scheduler hot path)
- `e753af7` GPTQ re-pack PASS validation
- `f5cf829` W4 c=8 substrate LICENSED
- `12a54da` GPTQ-aware patch
- `36830bf` EOD+34 loop synthesis
- `369292f` page_budget hypothesis(verified)
- `39237b9` W4A8 calibration root cause
- `662cbbb` M_quant AutoGPTQ→Marlin plan
