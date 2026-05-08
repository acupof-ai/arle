# cap=8 bimodal correction — memory IS axis but NOT the trigger(`9596566` partially walked back)

> Per `e5f9d86` slots=12 isolation:reducing memory headroom made degraded
> mode WORSE,not better。Disproves my `9596566` hypothesis that memory
> pressure was the bimodal TRIGGER。
>
> Memory IS an axis(governs degraded-mode floor)but bimodal SWITCH is
> something else(scheduling / harness / graph-cap race)。

## What the `e5f9d86` empirical shows

`e5f9d86` ran `--num-slots 12`(vs default 16,~25% smaller KV pool):
- Result:**145/256(56.6%)degraded**,kv_util 99.5% saturated

Compared to slots=16 default:
| Run | slots | Turn % | kv_util |
|-----|------:|-------:|--------:|
| b1mm1k0r7 | 16 | **92%** | 86.7% |
| b4kaqdrmj | 16 | **56%** | 84.7% |
| b5i3467ad | 16 | **76%** | 86.7% |
| **byfqsbviy** | **12** | **56.6%** | **99.5%** |

Pattern:
- slots=16:**bimodal 56-92%**(~67% normal mode,33% degraded)
- slots=12:**all degraded 56.6%**(no bimodal,deterministic floor)

→ If memory pressure was the bimodal **trigger**,less headroom should
  make MORE runs degraded。Empirically TRUE — slots=12 forces all
  degraded。
→ But that doesn't explain WHY slots=16 is **bimodal**(some 92%,some
  56%)at SAME memory state。
→ Conclusion:**memory governs degraded-mode floor,not bimodal switch**

## Updated hypothesis space

REFUTED:H_mem(memory pressure is bimodal trigger)
CONFIRMED:H_mem'(memory governs degraded-mode floor)

Remaining bimodal-switch candidates:
- **H_sched** Scheduling ordering(WB scheduler chunk dispatch sequence varies)
- **H_harness** Bench harness retry budget per-run
- **H_grcap** Graph-capture race(some sessions land before warmup completes)
- **H_alloc** GPU allocator slop(CUDA mempool fragmentation)

## Walking back `9596566`

My `9596566` brief recommended bumping **#33 KV W4A8 to P0** because
"memory pressure is bimodal trigger"。This reasoning is now REFUTED。

But:**#33 KV W4A8 still has independent value**:
- 4× KV pool capacity → improves DEGRADED-MODE FLOOR(more slots
  feasible)
- Unblocks c=16 hybrid memory budget(per `1959a21` Phase 0)
- Doesn't address bimodal SWITCH but addresses related axis

**Updated #33 priority**:
- Original P1
- `9596566` bumped to P0(based on REFUTED memory-trigger reasoning)
- **NOW**:**P1 again**(independent value valid,but not on critical
  path for cap=8 bimodal fix)

Master priority list reverts to:
- P0 Hybrid Phase 1b(`6be30ce` directive,~155-175 LOC per `9dc32d6`)
- P0' bimodal trigger investigation(per `e5f9d86` Next steps)
- P1 #33 KV W4A8(reverted from P0 bump)
- P1' Medusa(per `afdddec`)

## P0' bimodal investigation plan

Per `e5f9d86` Next steps + my hypothesis refinement:

### Test 1 — slots=24 + max-seq-len=6144(same memory budget, different dim)
- KV pool ≈ 24 × 6144 × bytes_per_token ≈ 16 × 8192 × bytes_per_token
- Same memory bucket but more slots concurrent
- If still bimodal:trigger is NON-memory(scheduling/harness/graph)
- If now 100% turn:memory-headroom is bimodal sub-axis

### Test 2 — W4 c=4 instead of c=8(halve concurrent pressure)
- Halves prefill admission per step
- If still bimodal at c=4:trigger is per-session,not per-cohort
- If 100% at c=4:cohort-level pressure is binding

### Test 3 — Server log diff between bimodal modes
- Run bench in both modes(maybe 5 runs)
- Diff RUST_LOG=info output line-by-line
- Look for graph-cap timing,allocator state,scheduler decisions
- Identify which deterministic state diverges

### Test 4 — Sleep + nvidia-smi --gpu-reset between runs
- Fresh GPU state per run
- If bimodal disappears:GPU driver/allocator state persistence is binding
- If bimodal persists:state is NOT GPU-driver-side

## Cross-references

- `9596566` original memory-trigger hypothesis(refuted)
- `e5f9d86` slots=12 isolation refutation
- `a0a3f42` 6-run bimodal characterization
- `fc41e7e` deterministic byte-identical signature
- `f05ea3a` skill v1.5.0 anti-pattern #17(bimodal masks single-run)
- `c20b1ce` warmup fix(necessary but not sufficient for 100%)
- `12300c5` cap=8 flip(production fix landing)

## Methodology lesson

`9596566` priority bump was based on **single-data-point hypothesis**
(memory peak 97% GPU)without controlled experiment to verify。Two ticks
later `e5f9d86` ran the controlled experiment(slots=12 reduction)and
refuted the hypothesis。

**Anti-pattern reinforced**(skill v1.4.0):**hypothesis priority bump
without controlled experiment trap**。Don't bump strategic priority
based on plausible mechanism alone — wait for controlled A/B(or run
quick A/B if cheap)。

Cost:
- 1 cron tick wrote `9596566` recommendation
- 1 cron tick wrote this correction
- Net:no priority shift on master,but methodology rule reinforced

## Status

Walks back `9596566` priority bump。#33 reverts to P1。**Bimodal trigger
investigation is now P0'**(not memory pressure as previously claimed)。

P0 Hybrid Phase 1b stays P0 — independent of bimodal investigation。

Codex idle since EOD+43 47m work session。Pickup queue:
- P0 Hybrid Phase 1b(`6be30ce` corrected scope per `9dc32d6`)
- P0' bimodal investigation(this brief proposes 4 tests)
- P1 #33 KV W4A8
- P1' Medusa
