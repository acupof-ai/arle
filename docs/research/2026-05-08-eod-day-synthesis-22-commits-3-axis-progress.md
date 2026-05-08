# 2026-05-08 EOD synthesis — 22+ commits across 3 master axes,methodology v1.5.0 codified

> Day's MEGA progress synthesis(EOD+~14h Claude session + codex
> collaboration)。22+ commits LANDED across W4 quantization axis,
> agent workload axis,and methodology codification。**Production
> path clear** — codex substrate pickup queue prioritized for next
> session。

## §1 Today's landings by axis

### Axis 1 — Agent workload(master §2.1 binding production)

| Commit | Effect |
|---|---|
| `b708e00`(codex) | W3+W4 admission deadlock substrate fix |
| `12300c5`(codex) | cap=4 → cap=8 default flip |
| `c20b1ce`(codex) | warmup respects model.max_concurrent_prefill_requests |
| `f5cf829`(Claude) | W4 c=8 8K admission-fix LICENSED 100% turn success |
| `19d12c2`(Claude) | TTFT p99 -86% via cap=8 override(warm server) |
| `27fd5de`(Claude) | cap=8 multi-shape SAFE(2 shapes) |
| `8281047`(Claude) | warmup fix LICENSED 91.8% turn success(N=1) |
| 6 cap=8 chain research entries(`db20d34`,`3cd3494`,`fc41e7e`,`a0a3f42`,`e5f9d86`,etc) | bimodal distribution characterization,memory floor distinction |

**Net axis 1 result**:
- W3+W4 deadlock: SOLVED
- TTFT p99: **-86% w/ cap=8**(robust across all 6 cap=8 runs)
- Turn success: bimodal 67% normal(76-92%)+ 33% degraded(56%)
- Production-readiness: **TTFT-LICENSED**;turn-success requires bimodal investigation OR 95%+ workload tolerance to 67%

### Axis 3 — Weight quantization(master §1.2.1.A)

| Commit | Effect |
|---|---|
| `2a3a6f0`(codex) | qzeros +1 fix in convert_gptq.py(REAL ROOT CAUSE for W4A16+W4A8) |
| `12a54da`(codex) | pack_w4a8 GPTQ-aware mode(Phase 1b enabler) |
| `09869bc`(Claude) | convert_gptq_w4a16_to_w4a8_marlin.py(141 LOC) |
| `b6502f7`(Claude) | merge_w4_hybrid_checkpoint.py(105 LOC) |
| `b5889b3`(Claude) | W4A8 prefill TTFT -36% LICENSED |
| `bc15eca`(Claude) | W4A16 GPTQ-zpfix CLOSES Round 4 implementation gap(1.06×→1.64×=+54%) |
| `c4fae17`(Claude) | 3-shape grid hybrid ROI 14-15% stable |
| `8588f6a`(Claude) | W4A8 vs W4A16 c=4→c=8 sweep |
| `1959a21`(Claude) | hybrid Phase 0 reconnaissance |
| `9dc32d6`(codex) | hybrid Phase 1b §2.1 scope correction |

**Net axis 3 result**:
- **qzeros bug RESOLVED** — W4A16+W4A8 production accuracy unblocked
- W4A16:LICENSED 1.64× via TWO routes(naive sym + GPTQ-zpfix)
- W4A8:LICENSED prefill(-36% TTFT vs W4A16),DEFERRED decode(no batch crossover at production c)
- Hybrid path:Phase 0 done,Phase 1 substrate pending(155-175 LOC codex)

### Axis 2 — Speculative decode(master §7.4)

| Commit | Effect |
|---|---|
| `aa00c6a`(Claude) | 4th classical-spec KILL evidence(W3 c=4 production-shape) |
| `528844c`(Claude) | M_medusa Phase 3 formula corrected + 4-KILL evidence |
| `afdddec`(Claude) | Medusa Phase 0 reconnaissance — substrate ready,scope -38% LOC |

**Net axis 2 result**:
- Classical-spec axis: DEAD across 4 shapes
- Medusa REQUIRED path: Phase 0 done,Phase 1 substrate(~500 LOC + 1 wk training)pending codex

## §2 Methodology contributions(skill v1.4.0 → v1.5.0)

`6c627c4` skill v1.4.0:anti-pattern #14 added(upstream parser silent corruption)
`f05ea3a` skill v1.5.0:anti-patterns #15-17 added:
- #15:"Warm-server" implicit dependency trap
- #16:Implicit-coupling-via-shared-default trap
- #17:Bimodal failure distribution masks single-run LICENSE

**Net methodology**:17 codified anti-patterns + 6 mantra rules。Today's
~6 ticks of variance investigation(cap=8 chain)yielded 3 institutional
rules preventing recurrence。

## §3 Codex pickup queue priority(EOD)

Updated from `5364612` codex pickup queue with today's evidence:

### P0 — Hybrid Phase 1(`9dc32d6` scope:155-175 LOC,~0.75-1 day)

- Loader §2.1 detection patch(25-35 LOC)
- §2.2 tensor read both formats(~50 LOC)
- §2.3 DeviceMatrix Option A extension(~50 LOC)
- §2.4 constructor(~30 LOC)
- e2e gate via `Qwen3-4B-W4-hybrid-zpfix` checkpoint(`b6502f7` ready)

**ROI**:unlocks -14% E2E latency at production c=4-8(per `c4fae17`
3-shape grid)。Memory budget: 7.15 GB / 16 GB = 45%(c=4)to 87%(c=8)。

### P0' — KV W4A8 axis(M_quant-kv-w4a8.md task #33)

- Phase 0a smoke kernel(1 h codex)
- Phase 0b implementation(if smoke passes)
- New kernel `decode_attention_w4_a_fp8.cu`(~300-400 LOC)
- KV pool 21k → 84k tokens(4× capacity)

**ROI**:
- Memory pressure relief → cap=8 bimodal degraded-mode floor lifts
- Long-context decode bandwidth -75%
- Unblocks c=16 hybrid deployment(currently NOT FEASIBLE per Phase 0)

### P1 — cap=8 bimodal residual investigation

- Step 2.B bench harness retry isolation(per `fc9bea9`)
- Step 2.C server admission tracing
- Identify why 33% of cap=8 fresh runs hit degraded mode despite warmup fix

**ROI**:closing the 33% degraded mode lifts production turn success
from bimodal 67-92% to consistently 95%+。

### P1' — Medusa Phase 1.A-1.E(per `afdddec`)

- 1.A Training data prep(~50 LOC)
- 1.B Head architecture + training(~100 LOC PyTorch + 150 LOC `crates/train/src/medusa.rs`)
- 1.C ARLE inference integration(~150 LOC)
- 1.D Test gate(~50 LOC)
- 1.E Bench gate(1 day Claude)
- Total:~500 LOC + 1 week training

**ROI**:2.0-3.0× tok/s on production agent W3/W4(if α=0.7-0.85 Medusa paper holds)。

### P2 — xgrammar FFI scaffold(task #26,master §7.5)

- 400-600 LOC codex
- Combined with Medusa for grammar-constrained spec decode

### P3 — TRT-LLM bench(task #21 deferred)

- Cross-engine 4-axis matrix completion

## §4 Strategic insights from today

### W4 quantization axis 全套 — production CLEAR

Pre-EOD:W4A16 marginal,W4A8 garbage,hybrid theoretical
Post-EOD:**W4A16 LICENSED both routes,W4A8 prefill LICENSED,hybrid Phase 0 done**

The qzeros +1 bug(`2a3a6f0`)was the key blocker — single line fix
unblocks both W4A16 AND W4A8 production accuracy。Hybrid integration
becomes mechanical(155-175 LOC)post-fix。

### Agent workload — production CLEAR with caveat

Pre-EOD:W3 c=16 deadlock,W4 c=8 deadlock,no agent baseline
Post-EOD:**256/256+ turn success at cap=4,TTFT p99 -86% with cap=8**

Bimodal characterization is the production caveat — for tail-bound
workloads cap=8 is shippable;for turn-bound 95%+ workloads,cap=4
remains conservative default until bimodal investigation lands。

### Spec decode — Medusa REQUIRED finalized

Pre-EOD:classical-spec in question
Post-EOD:**4 KILLs across 4 shapes confirms classical DEAD,Medusa Phase 0 ready**

Phase 0 reconnaissance saved 38% LOC scope vs naive estimate(800 → 500 LOC)
by inventorying existing substrate(speculative.rs 721 LOC + train + autograd
crates ready)。Medusa is fresh delta on top,not green-field。

### Methodology — 3 new anti-patterns codified

Today's cap=8 chain investigation produced 3 anti-patterns
institutionally codified into skill v1.5.0:
- #15 warm-server implicit dependency
- #16 implicit-coupling-via-shared-default
- #17 bimodal failure distribution

These prevent recurrence on FUTURE config-change axes(num_slots,
max_seq_len,kernel batch sizes,admission thresholds,etc)。

## §5 Next session pickup checklist

For codex(in priority order):
1. ☐ Hybrid Phase 1 substrate(`9dc32d6`,155-175 LOC,~1 day)
2. ☐ KV W4A8 Phase 0a smoke(M_quant-kv-w4a8.md §3,1h)
3. ☐ Medusa Phase 1.A-1.E sequenced(~500 LOC + 1 week training)
4. ☐ Cap=8 bimodal residual Step 2.B/2.C(per `fc9bea9`)
5. ☐ xgrammar FFI scaffold(task #26)

For Claude(parallel, single-file ≤100 LOC):
1. ☐ Medusa Phase 1.A training data prep stub(once codex 1.B head architecture lands)
2. ☐ Bench any new substrate codex ships
3. ☐ Continue research entries on bimodal investigation(if codex pickup delayed)

## §6 Production deployment status

**SHIP READY**(no codex blocker):
- W4A16 production decode default(LICENSED 1.64× both routes)
- W4A8 prefill bench data
- W3 c=16 + W4 c=8 substrate fix(cap=4 stable,cap=8 conditional)

**CONDITIONAL**(needs additional verification):
- cap=8 default(turn-bound workloads need bimodal investigation)
- W4A8 GPTQ-zpfix checkpoint(local only,not yet pushed to HF)

**PENDING SUBSTRATE**(codex blocker):
- Hybrid prefill-decode dispatch(Phase 1 pickup)
- KV W4A8 INT4 kernel(Phase 0a smoke)
- Medusa multi-head spec(Phase 1.B+)
- xgrammar FFI(scaffold)

## §7 Cumulative count

- Day commits:22+(this synthesis itself = 23)
- Wins entries:6
- Research entries:14+
- Skill versions:v1.4.0 → v1.5.0(3 new anti-patterns)
- Plan documents:Hybrid + Medusa + KV W4A8 + TTFT p99 + xgrammar + ...

Methodology cost-benefit:massive Claude+codex collaboration produced
clean axis-by-axis production picture with full evidence trail。Tomorrow's
pickup is unambiguous — codex Hybrid Phase 1 → KV W4A8 Phase 0a →
Medusa Phase 1.A in priority order。

## Cross-references

- Skill v1.5.0:`f05ea3a`(`.claude/skills/kernel-optimization/SKILL.md`)
- Master strategy:`docs/projects/2026-05-07-arle-master-strategy.md`(may need EOD update)
- Codex pickup queue:`5364612`(needs refresh per this synthesis)
- All 22+ commits:see `git log` between `b708e00` and this entry
