# Codex Pickup Queue — 2026-05-09 EOD

> Successor to [`codex-pickup-queue-2026-05-08.md`](codex-pickup-queue-2026-05-08.md)
> after another 43+ commit day。Codex idle EOD,7 explicit pickup items
> with dispatch directives。**Designed so the next cron-fired Claude can
> paste-buffer directly to codex without rebuilding context**。
>
> 2026-05-09 EOD state:W4A8 LANDED + W3+W4 admission unblock + cap=8 default
> + B3 Step 1 admission_allows refactor + skill v1.6.0(18 anti-patterns)
> + 6 wins entries + 21+ research entries。

## Tonight's progress(2026-05-08 → 2026-05-09)

| Axis | Status | Commit / Entry |
|------|--------|----------------|
| W4A8 accuracy(GPTQ qzeros) | ✅ LICENSED | `2a3a6f0` |
| W3+W4 admission deadlock | ✅ SOLVED(codex page_budget) | `b708e00` |
| cap=8 default flip | ✅ LANDED + caveat | `12300c5` + `db20d34` |
| W4A16 LICENSED 1.64× | ✅ 2 routes(naive sym + GPTQ-zpfix) | `bc15eca` |
| W4A8 prefill LICENSED -36% TTFT | ✅ both arms verified | `b5889b3` |
| TTFT p99 -86% | ✅ via cap=8 | `cap8-ttft-tail.md` |
| B3 Step 1 admission_allows | ✅ byte-identical regression | `7c8fd61` + `c30e298` |
| Skill v1.4.0 → v1.6.0 | ✅ +5 anti-patterns | `125f795` |
| Codex pickup directives | ✅ this doc | (current) |

## P0 — Pickup directives(cron-Claude paste-buffer ready)

### P0.1 — B3 Step 2 PrefixAwareAdmission CUDA-runtime gate(refined per `1217375`)

**Why first**:smallest concrete next step in production scheduler chain。
SGLang multi-tenant 2× gap closure。Architecture refined to reuse existing
`runtime/admission.rs:187` lookup_or_stage(no SchedulerHandle field needed)。

- **Effort**:~100 LOC,**0.5 day**
- **File**:`infer/src/scheduler/cuda/runtime/admission.rs`
- **LOC site**:after line 187 `lookup_or_stage` returns
- **Risk**:Low(reuses existing lookup,backend isolation preserved)
- **ROI**:enables PrefixAwareAdmission → SGLang multi-tenant gap close
- **Dependency**:✅ A1 RadixCache production-wired(`1217375`)+ ✅ B3 Step 1 admission_allows(`7c8fd61`)

**Dispatch directive**(paste-buffer to codex tomorrow):
```
B3 Step 2 — PrefixAwareAdmission CUDA-runtime integration

File: infer/src/scheduler/cuda/runtime/admission.rs
Site: after existing lookup_or_stage call (line ~187)

Implementation outline:
1. After `let lookup = self.prefix_cache.lookup_or_stage(...)` returns
2. Construct SchedulerSignals {
     queued_requests: scheduler.waiting_count(),
     active_decodes: scheduler.active_count(),
     prefix_hit_tokens: lookup.matched_len,
     session_affinity_slot: session_slot_hold.as_ref().map(|h| h.slot_idx()),
     turn_depth: req.turn_depth,
   }
3. Construct policy:
   PrefixAwareAdmission::with_cold_headroom(
     scheduler.config.max_waiting,
     scheduler.config.cold_headroom.unwrap_or(scheduler.config.max_waiting / 4),
   )
4. Gate: if !policy.allow(signals) { return AdmissionResult::Rejected(...) }

Tests:
- Integration test: warm-vs-cold session ordering
- Bench: multi-tenant 4-conc 6k-system burst → expect TTFT 318ms → 157ms

Reference:
- A1 audit: docs/research/2026-05-09-a1-audit-radix-cache-production-wired.md
- Step 2 architecture: docs/research/2026-05-09-b3-step2-architecture-acknowledged.md
- PrefixAwareAdmission: infer/src/scheduler/policy.rs:98-130
- Backend isolation rule: CLAUDE.md §Backend isolation
```

### P0.2 — Hybrid Phase 1b loader patch(`6be30ce` directive)

**Why**:smallest concrete next step in hybrid axis。已 Phase 0 + Phase 1a done。
Codex pickup pending since 2026-05-08。

- **Effort**:155-175 LOC,**0.75-1 day**
- **File**:`infer/src/model/weight_loader.rs:514`(detection site)
- **Risk**:Med(loader changes need careful testing)
- **ROI**:enables -14% E2E at c≤8 production
- **Dependency**:NONE — Phase 0(`1959a21`)+ Phase 1a(`b6502f7`)done

**Dispatch directive**:
```
Hybrid Phase 1b — loader patch for marlin_w4_hybrid checkpoint

Reference plan: docs/plans/M_quant-hybrid-phase1b-loader-directive.md (commit 6be30ce)

File: infer/src/model/weight_loader.rs:514
Add: detection for "marlin_w4_hybrid" config field
Wire: load both W4A16 + W4A8 weight tensors per Phase 0 reconnaissance
Verify: cargo test + scripts/bench_guidellm.sh hybrid-phase1b-smoke

Acceptance criteria:
- Loader successfully reads merged W4A16/W4A8 checkpoint
- Per-layer dispatch metadata preserved
- No regression on existing W4A16-only or W4A8-only checkpoints
- Bench wins entry: docs/experience/wins/2026-05-XX-bench-hybrid-phase1b-loader.md
```

### P0.3 — cap=8 prefill pre-warm fix(`56dbd1c` Step 2.B')

**Why**:bimodal failure distribution(67% normal / 33% degraded 56%)
identified per `db20d34`。Cold-start prefill warmup gap。Single-tick fix
unblocks ALL cap=8 benefit。

- **Effort**:100-150 LOC,**1-1.5 day**
- **File**:`infer/src/scheduler/cuda/core/warmup.rs` + adjacent
- **Risk**:Med(scheduler hot path)
- **ROI**:eliminates bimodal distribution → cap=8 stable -86% TTFT p99
- **Dependency**:NONE

**Dispatch directive**:
```
cap=8 prefill pre-warm — Step 2.B' implementation

Reference: docs/research/2026-05-08-cap8-default-h4-warmup-cap-rootcause.md (db20d34)

Root cause: c20b1ce warmup covers DECODE only, not PREFILL cold-start.
First fresh-server cap=8 prefill burst hits cold kernel cache → bimodal regression.

File: infer/src/scheduler/cuda/core/warmup.rs
Add: PREFILL warmup pass
- Trigger same paths exercised by first prefill at startup
- Prime kernel cache for cap=8 admission default
- Cold-start adds ~250ms but unblocks -86% TTFT p99 stable

Acceptance criteria:
- W3 c=4 cap=8 fresh-server bench: 5/5 trials within σ < 5%
- TTFT p99 stays at -86% relative to cap=4
- Bench entry: docs/experience/wins/2026-05-XX-bench-cap8-prefill-warmup.md
```

## P1 — Larger substrate(later in tomorrow's queue)

### P1.1 — KV W4A8 Phase 0a smoke kernel(#33)

- **Effort**:100-400 LOC,**1-3 days**
- **Plan**:`docs/plans/M_quant-kv-w4a8.md`(`1e713de`)
- **ROI**:21k → 84k tokens KV pool(4×)+ unblock c=16 hybrid

### P1.2 — Hybrid Phase 1-3 dispatch substrate(#30)

- **Effort**:155-175 LOC scaffolding,**1 day**
- **ROI**:complete hybrid axis after Phase 1b loader

### P1.3 — W4A8 graph capture hoist(#24)

- **Effort**:200-400 LOC
- **ROI**:enables W4A8 production default-on(currently gated by graph capture)

## P2 — Long-horizon(this week)

### P2.1 — M_medusa scaffold(#28)

- **Effort**:600-1200 LOC + 1 week training
- **Plan**:`docs/plans/M_medusa.md`(`528844c`)
- **ROI**:potentially +20-30% throughput,short-output workloads

### P2.2 — M_xgrammar FFI scaffold(#26)

- **Effort**:400-600 LOC FFI bridge
- **ROI**:JSON-mode + structured output speedup

## P3 — Deferred / blocked

### P3.1 — arle data download HF Hub blocker(#34)

- **Status**:demoted P3(workaround works)
- **Issue**:`hf-hub` Rust crate or `ureq` HTTP client internal "unexpected end of file"
- **Workaround**:manual `wget` + `pandas read_parquet` + `arle data convert`

## Recommended cron-Claude tomorrow flow

```
Tick 1 (cron 12min after first wake):
  1. capture-pane tmux 0:0 → confirm codex idle
  2. paste-buffer P0.1 dispatch directive (B3 Step 2)
  3. tmux send-keys Enter Enter Enter Enter
  4. verify "Working" appears
  5. ScheduleWakeup 1800s
  6. report state

Tick 2 (cron next):
  - if codex still working P0.1 → Claude does Phase 0 audit on P0.2 (hybrid loader)
  - prepares P0.2 dispatch directive in advance for next tick

Tick 3+ (codex finishes P0.1):
  - bench verify P0.1 (Claude or codex)
  - paste-buffer P0.2 immediately
```

## Cross-references

- B3 Step 1 LANDED: `7c8fd61` + `c30e298`
- A1 audit: `1217375`
- B3 Step 2 architecture: `c097b2b` + `637701b`
- Skill v1.6.0 anti-pattern #18: `125f795`
- Predecessor pickup queue: `codex-pickup-queue-2026-05-08.md`

## Status

**Single source of truth** for tomorrow's codex pickup ordering。Each P0
item ships with paste-buffer-ready dispatch directive — cron-Claude can
re-enter session and dispatch in <1 minute without rebuilding context。

Hand-off mechanism preserves momentum across cron-fired session boundaries
that would otherwise lose substrate-level state。

## Rule

**EOD pickup queue with explicit dispatch directives is the management
artifact that lets cron-fired Claude function as a continuity layer
across session boundaries**。Without it,each cron tick spends 5-10 min
rebuilding pickup context before dispatching。With it,dispatch is sub-minute。

This is the practical realization of "顶级管理者推进 codex/Claude 工作分配"
under cron-loop session model。
