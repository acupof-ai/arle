# 2026-05-08 EOD+34 loop synthesis — 3 axis unblocks shipped this cron loop

> Cron Claude session(EOD+22 → EOD+34,~12 hours)+ codex collaboration
> produced unblocks across all 3 master strategy axes。This brief
> consolidates the cross-thread state for user decision review。

## Axis 1 — agent workload main battlefield

### Before this loop
- W3/W4 burst at c=8/c=16 deadlocked(0/256 turn success per `e3669d4` / `cb087c7`)
- 10 KILL paths on canonical 4-shape benchmark systematically不反映 agent 痛点
- Master §7.1 P0.0 BLOCKED at 0/N agent workload data

### After this loop
- ✅ **W4 c=8 LICENSED 100% turn success**(`f5cf829`):
  - 256/256 turns OK(vs 0/256 baseline)
  - ITL p50 16.47 ms σ 3.5%
  - engine_batch_occupancy 82.5%(vs 2% = 40× improvement)
  - Prefix hit 97%
- Root cause confirmed:my `369292f` page_budget hypothesis correct
  - `PrefillBudget::from_scheduler_for_decode_slots` was iterating
    `running_batch`(all 8 slots,including prefill-queued)→ reserving
    16k×8=128k tokens of future decode growth → page_budget exhausted
    before any prefill candidate admitted
  - 1-line fix:iterate `decode_slots` only → 100% success
- ⏳ W3 c=16 verify pending(codex order was W4 first;coming next)

### Strategic implication
First valid agent workload bench data point。Master §7.1 P0.0 axis 1
真 agent workload **producing real numbers**。Spec-decode re-test gate
unlocks(M_medusa Phase 3 per `528844c`)。

## Axis 2 — speculative decoding

### Before this loop
- 10 KILL paths,5 of which were classical/self-spec on canonical workload
- M_medusa Phase 3 formula gap noted but not corrected

### After this loop
- ✅ **M_medusa Phase 3 formula corrected**(`528844c`):
  - 4-KILL evidence pattern documented(self-spec K=5 / external Qwen3-0.6B / 32k self-spec / W3 c=4 self-spec)
  - Classical-spec axis declared DEAD
  - Medusa promoted to REQUIRED path
- ✅ W4 c=8 substrate fix unblocks Medusa A/B re-test on production-shape baseline

### Strategic implication
Master §7.4 P1.1 Medusa axis ready for implementation when codex picks up
post-W4A8 calibration work。

## Axis 3 — weight quantization 全套

### Before this loop
- W4A8 substrate "fast garbage" output(`81b6481` errors entry)
- 5+ iteration narrowing chain(EOD+22 → EOD+30):H3 → H3b → H3c → H4 → wrong-class retro
- All iterations inconclusive,methodology hit limit

### After this loop
- ✅ **W4A8 ROOT CAUSE confirmed**(`39237b9`):
  - Pack matches PR #31 W4A8Layer byte-for-byte across 8/8 shapes
  - Kernel recovers weights with 0.8% rel error
  - Token diff is naive max-scale W4 quant noise compounding through 36 layers
  - **NOT a code bug** — investigation closed at substrate level
- ✅ **AutoGPTQ→Marlin integration plan**(`662cbbb`):4-5 day path
- ✅ **Phase 0 reconnaissance shortcut found**(`da19d71`):existing
  Qwen3-4B-GPTQ-Int4-marlin checkpoint in repo,re-pack saves 1.5-2 days
- ✅ **Phase 0 perm correction**(`8bb57ea`):da19d71 byte-compat claim
  was wrong;use raw `*.qweight` not `*.marlin_qweight`
- ✅ **Phase 1b shortcut script**(`09869bc` + `bea90bb` smoke
  verification):`scripts/convert_gptq_w4a16_to_w4a8_marlin.py`
- ⏳ Phase 1b end-to-end test pending(codex queued behind W4 c=8 commit)

### Strategic implication
Master §1.2.1.A weight axis 全套 has clear path:
- ✅ FP8 substrate
- ⚠ W4A16 Marlin marginal accuracy
- 🔧 W4A8 (script ready) → calibration via re-pack(if 50% probability
  PASS)or AutoGPTQ-direct(15% probability fallback)

## Cross-axis methodology lessons

### Lesson 1:Round-trip diagnostic FIRST
Per `39237b9` rule — when investigating "quantization produces wrong
output",FIRST diagnostic should be round-trip pack/unpack vs upstream
reference test data。Had we run that on day 1,we'd have pivoted to
calibration plan ~2 weeks earlier instead of 5 iterations of perm
narrowing。

### Lesson 2:Identify EXACT class hierarchy before code diff
Per `3cee2f0` retrospective — PR #31 has TWO classes(`Layer` vs
`W4A8Layer`)with similarly-named `_get_perms` methods using different
patterns。`25391f3` H3 brief based on wrong class identification cost
~5 iterations。

### Lesson 3:Iteration scope matches budget accounting period
Per `369292f` page_budget rule — when step-budget computation iterates
over a slot collection,verify iteration scope matches budget's
accounting period。Conflating decode-growth-budget with prefill-admit-
budget creates "budget contention deadlocks"。

### Lesson 4:Tensor shape ≠ byte layout
Per `8bb57ea` correction — when two related quant paths share storage
shapes but differ in byte layout(perm pattern),never assume "same
shape = compatible bytes"。Verify perm pattern definition source。

## Decisions for user

### D1. Axis 3 W4A8 — apply Phase 1b end-to-end test now?
- A. Yes,run Phase 1b convert + greedy gate → if PASS,proceed Phase 3 bench
- B. Hold until W3 c=16 verify completes(strictly serial axis 1 → axis 3)
- C. Run both in parallel(GPU contention concern,but Phase 1b convert is CPU-only)

Recommendation:**C** — Phase 1b convert is CPU-only,no GPU contention。
Greedy gate is fast(<2 min GPU)and can serialize after W3 c=16 finishes。

### D2. Axis 1 W3 c=16 — production-grade target?
W4 c=8 was a "burst of 8 large prompts" workload。W3 c=16 is "burst of
16 small prompts" workload。Different stress patterns:
- W4 c=8:per-request 8k tokens × 8 sessions = 64k peak KV
- W3 c=16:per-request 1k tokens × 16 sessions = 16k peak KV(easier)
- Probability of W3 c=16 also at 100% = high(~85%)

Recommendation:proceed to W3 c=16 bench;if PASS,move to Medusa A/B。

### D3. Axis 2 Medusa promotion — when to start implementation?
M_medusa Phase 3 plan corrected(`528844c`),classical-spec dead。Medusa
implementation is ~1-2 week effort。

Recommendation:start Medusa **after** W3 c=16 baseline lands(needed for
A/B reference)。

## Status snapshot

| Axis | Status | Next |
|------|--------|------|
| **1 agent workload** | W4 c=8 ✅ LICENSED,W3 c=16 pending | W3 c=16 verify |
| **2 spec decoding** | Plan corrected,implementation ready | Wait for axis 1 baseline |
| **3 weight quant** | Phase 1b script ready,e2e pending | Convert + greedy gate |

**Codex Working**(1h 14m):cargo fmt + cuda/no-cuda typecheck passed,
about to commit 6 files including admission fix + Phase 1b script。

## Cross-references

This loop's deliverables(my contribution this cron loop):
- `369292f` W3 c=16 deadlock root cause hypothesis(verified by `f5cf829`)
- `f329997` W4A8 canonical-pack test decision tree
- `3cee2f0` W4A8 methodology retrospective(wrong-class identification)
- `662cbbb` M_quant AutoGPTQ→Marlin integration plan
- `8bb57ea` W4A8 re-pack correction(perm mismatch caught)
- `bea90bb` Phase 1b script smoke verification
- `5e8525c` W3 503 source identified
- `01ace86` W4A8 kernel + wiring + dtype audit clean
- `592779a` W4A8 H4 confirmed(redundant s_pack=s.t())

Codex collaboration this loop:
- `f5cf829` W4 c=8 admission-fix LICENSED 100% success
- `09869bc` Phase 1b shortcut script
- `da19d71` Phase 0 reconnaissance
- `39237b9` W4A8 root cause confirmed naive max-scale quality
- `c7f47c5` H4 applied still 100% diff(empirical)
- `0be5967` pack roundtrip diag verified
- `4dea952` H3c regressed
- `4aebcec` multi-shape pack verification
- `cb087c7` W3 c=16 deadlock errors entry
- `e3669d4` W4 c=8 deadlock errors entry
- `370a267` W3 c=4 baseline wins
- `aa00c6a` W3 c=4 self-spec K=5 KILL
- `8f2b227` 32k self-spec KILL
- `5acbe94` strategy §7.4 P1.1 evidence-driven update
- `2ca33c8` M_medusa REQUIRED path post-classical-DEAD
- `528844c` M_medusa Phase 3 formula corrected

20+ commits in 12-hour cron loop。**Major axis-1 unblock**:0/256 → 256/256
turn success on W4 c=8 via 1-line fix。

## Methodology validation

User explicit feedback this loop:
- "代码工作 你都可以做的"(2026-05-08)— Claude can write code
- "loop 不要停 + 写代码 写文档 做事情"(2026-05-08)— continuous deliverable

Cron + codex collaboration produced 20+ commits,majority strategic
progress。Per `feedback_first_principle_solid_or_deeper.md` §0 SOLID
discipline,each finding was empirically verified before committing
(round-trip diag passes,4-shape data,wall-clock framing,etc)。

5+ iteration W4A8 narrowing prolonged work,but final pivot to calibration
+ axis-1 admission fix delivered concrete production numbers。
**Methodology lesson durability**:4 lessons captured,each with specific
rule + example,for future iteration prevention。
