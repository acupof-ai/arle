# Codex Pickup Queue — 2026-05-08 EOD+47

> Codex idle since `2a3a6f0` 47m work session(EOD+43)。Multiple plans
> queued for pickup。This doc orders them by ROI + dependency graph
> for when codex resumes work。

## Priority order(highest ROI first)

### P0 — Hybrid Phase 1b loader patch(`6be30ce` directive)
**Why first**:smallest concrete next step,unblocks entire hybrid axis。
- Effort:0.5d / ~100-150 LOC
- Risk:Med(loader changes need careful testing)
- ROI:enables -14% E2E at c≤8 production
- Dependency:NONE — Phase 0 + Phase 1a already done(`1959a21` + `b6502f7`)
- Smallest verification step:`weight_loader.rs:514` add `"marlin_w4_hybrid"`
  detection,build + run existing tests,then proceed

### P0' — Default-on flip W4A8 prefill path(D5)
**Why next**:W4A8 prefill TTFT -36% LICENSED but not yet default。Could
flip independently of hybrid,then revisit when hybrid lands。
- Effort:0.25d / ~30 LOC scheduler/linear.rs config changes
- Risk:Low(both arms LICENSED individually,no kernel changes)
- ROI:immediate -36% TTFT for production users
- Dependency:graph capture wiring(`62e75ee` plan,which was KILLED per master
  §7.7 — needs re-evaluation)
- **Decision needed from user**:flip default-on now,or wait for hybrid Phase 4?

### P1 — KV W4A8 plan execution(task #33)
**Why P1**:paired axis with weight W4A8(master §1.2.1.B),c=16 hybrid
blocked on this。Independent gain:21k → 84k tokens KV pool capacity(4×)。
- Plan:`M_quant-kv-w4a8.md` already exists(`1e713de`)
- Effort:5-10 days(new kernel `decode_attention_w4_a_fp8`)
- Risk:High(new kernel from scratch)
- ROI:long-context decode + multi-tenant memory headroom + unblock c=16 hybrid
- Dependency:NONE,but P0 hybrid Phase 4 bench would inform priorities

### P1' — Medusa axis 2 implementation(D3,task #32)
**Why P1**:classical-spec axis dead per `aa00c6a` 4-KILL evidence,Medusa
REQUIRED path per master §7.4 P1.1。Production-shape baseline now exists
post-`b708e00` admission unblock。
- Plan:`M_medusa.md`(`528844c` Phase 3 corrected)
- Effort:~1-2 weeks
- Risk:High(complex spec coordination)
- ROI:potentially +20-30% throughput for short-output workloads
- Dependency:axis 1 baseline ✓ ready

### P2 — TTFT p99 tail-latency plan(D4)
**Why P2**:codex flagged "TTFT p99 still very poor" in W4 c=8 license。
Separate from liveness,don't block axes 2/3。
- Plan:not yet written(need 0.5d Claude)
- Effort:Variable(once root cause known)
- Risk:Med
- ROI:improves production tail-latency UX
- Dependency:NONE,can run in parallel

### P2' — SGLang multi-tenant prefix cache investigation(task #30)
**Why P2**:M_world1 P0.2 multi-tenant baseline showed SGLang 2× ARLE。
Investigation could reveal optimization。
- Plan:not yet written
- Effort:0.5d Claude code grep + write up
- Risk:Low(research)
- ROI:potentially identifies +50-100% multi-tenant gain
- Dependency:NONE

### P3 — TRT-LLM bench(task #21,deferred)
**Why P3**:M_world1 P0.3 — full 三方 baseline includes TRT-LLM。
Operationally heavier(separate venv install)but completes 4-axis matrix。
- Effort:0.5d setup + run
- Risk:Low(infra)
- ROI:cross-engine parity claims
- Dependency:NONE

### P3' — 3 KILLED substrate cleanup(task #24)
**Why P3**:1-week observation period from KILL dates(2026-05-XX)。
Today 2026-05-08 — observation period extends to ~2026-05-14。NOT YET DUE。
- Effort:0.5d cleanup commits
- Risk:Low(deletions only)
- ROI:repo cleanup,no functional gain
- Dependency:wait until 2026-05-14+

## Recommended sequence(if codex picks up next)

1. **P0 Hybrid Phase 1b**(0.5d)— smallest concrete step,unblocks chain
2. **P0' Default-on flip W4A8**(0.25d)IF user approves D5 — immediate prod gain
3. **P0/P0' bench**(0.25d)— ship wins entry per CLAUDE.md mandatory rule
4. After P0 chain lands → choose between P1(KV W4A8)or P1'(Medusa)
   based on user priority(both ~1-2 week efforts,independent gains)

## Open decisions for user

- **D3 Medusa start timing**:after P0 lands?(rec yes — P0 is 1d,Medusa 1-2 weeks)
- **D4 TTFT plan**:write in parallel with P0/P1?(rec yes — 0.5d Claude side)
- **D5 default-on flip**:do BEFORE hybrid Phase 4,or wait?(rec do BEFORE — independent of hybrid,direct user gain)
- **D6.1 hybrid pursue**:GREEN-LIGHT(rec proceed per `9754aca` plan)
- **D6.2 side-by-side weights vs dynamic conversion**:Option A(side-by-side)for ≤4B,Option B(dynamic)for ≥7B

## Cross-references

- Hybrid plan: [`9754aca`](M_quant-w4a16-w4a8-hybrid-prefill-decode.md)
- Phase 1b directive: [`6be30ce`](M_quant-hybrid-phase1b-loader-directive.md)
- Phase 0 reconnaissance: `1959a21`
- Phase 1a tool: `b6502f7`
- Concurrency sweep: `8588f6a`
- W4A8 prefill LICENSED: `b5889b3`
- W4A16 LICENSED: `bc15eca`
- qzeros fix root cause: `2a3a6f0` + `5593865`
- KV W4A8 plan: `M_quant-kv-w4a8.md`
- Medusa Phase 3 plan: `528844c`
- W3/W4 admission unblock: `b708e00` + `f5cf829`

## Status

This queue is the **single source of truth** for next-pickup ordering
when codex resumes。Updated as plans land or priorities shift。

Cron+codex collaboration has produced **35+ commits across axis 1 + axis 3**
in past 24 hours。Codex idle awaiting direction — push notification was
sent at EOD+43 milestone landing。Waiting for user to GREEN-LIGHT next
pickup direction。
