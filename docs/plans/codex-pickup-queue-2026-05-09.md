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
| **B3 Step 2 PrefixAwareAdmission** | ✅ **LICENSED -24.2% TTFT** σ/mean=4.5% | `b85929b` + wins entry `docs/experience/wins/2026-05-09-bench-b3-step2-prefix-aware.md` |
| **P0.2 Hybrid Phase 1b loader** | ✅ **LANDED**(loader-only,Phase 2 dispatch deferred) | `232aed5` + wins entry `docs/experience/wins/2026-05-09-bench-hybrid-phase1b-loader.md`(TTFT p50 68.4ms regression gate PASS) |
| Skill v1.4.0 → v1.7.0 | ✅ +6 anti-patterns | `c768b70` |
| **Skill v1.8.0 anti-pattern bank**(deferred batch) | 🟡 **5 candidates ready,batch trigger pending** | #20 hypothesis-inheritance(`c076aae`)+ #21 recipe-itself audit(`b55bfcd` + `153fd93`)+ #22 twin-commit attribution(`919c0fb` + `8d91d20` + `3fea979`)+ #23 truncated-output partial-view(`156d2c2`)+ #24 cell-collapse blindness(`1ccb448`) |
| **Wins-entry re-attribution gap**(c20b1ce NO-OP)| 🟡 3 entries need invalidation/re-attribution | `8d91d20` empirical triage |
| **M_quant Round 4 #6 W4A16BatchGemv override** | ❌ **KILLED** via refined bench(+37% ITL regression even at decode batch=4)| `3b9cc06` — Marlin tensor-core wins over launch-overhead savings;hypothesis empirically eliminated;skill v1.8.0 #25 grounded |
| Codex pickup directives | ✅ this doc | (current) |

## 🌀 Bidirectional audit cycle status(2026-05-09 EOD,11+ commits)

This session demonstrated **bidirectional audit pattern** at unprecedented density:

| # | Stage | Commit | Layer |
|---|-------|--------|-------|
| 1 | Claude — Phase 0 audit warmup.rs | `1fdd763` | Source audit |
| 2 | Codex — audit-of-audit gaps | `c076aae` | Audit-of-audit |
| 3 | Claude — adopt gates + LOC reduction | `8b1a913` | Integration |
| 4 | Codex — Phase 0.5 cheap-experiment recipe | `3456f8f` | Recipe |
| 5 | Claude — cite recipe in queue | `43b2115` | Reference |
| 6 | Codex — strategic axis-ROI brief | `d2c2c17` | Strategic |
| 7 | Claude — P0.0 Phase 1 decomposition gate | `9e964c9` | Strategic gate |
| 8 | Codex — LICENSE staleness pre-merge gate | `ec5c37c` | Meta-SOLID |
| 9 | Claude — annotate gate in queue | `183bd60` | Reference |
| 10 | Codex — B3 Step 2 substrate LANDED | `b85929b` | Implementation |
| 10' | Codex — Phase 1.A nvtx recipe | `2fafa9e` | Recipe(forward) |
| 11 | Claude — Phase 1.A scoping fix | `b55bfcd` | Recipe-itself audit |
| 12 | Codex — recipe-itself audit codification | `153fd93` | Methodology(skill #21 candidate) |
| 13 | Claude — c20b1ce incoherent fix discovery(via codex P0.2 mid-flight) | `919c0fb` | Implementation-impact audit |
| 14 | Codex — c20b1ce was NO-OP not silent-fail(deeper)| `8d91d20` | Empirical-impact triage(skill #22 candidate refinement) |
| 15 | Claude — pickup queue NO-OP triage acknowledgment | `921f313` | Reference + cycle codification |
| 16 | Codex — `12300c5` was actual fix,c20b1ce dead code | `3fea979` | **CLOSURE — 7-layer audit chain CLOSED + anti-pattern #22 final framing** |
| 17 | Codex — P0.3 framing correction post closure | `82eda25` | Surgical doc edits prescribed |
| 18 | Claude — apply 3 surgical edits to pickup queue P0.3 | `e247c4f` | Cycle compounds into anti-pattern #22 sub-rule:**downstream-doc citation pollution** |
| 19 | Codex — annotate 2 wins entries + **Layer-8 confound discovery** | `655accf` | `bwa4piqqx`(num_slots=4)vs `b1mm1k0r7`(num_slots=16)= multi-variable confound,NOT c20b1ce isolation |
| 20 | Codex — annotate 3rd wins entry(`cap8-chain-final-synthesis`)| `9bc4729` | **3-doc attribution scan COMPLETE** — all c20b1ce-cited wins entries now have corrected attribution |
| 21 | Codex — task #24 BF16 split-KV pre-audit(forward planning) | `78ccbb6` | Pre-staging task #24 cleanup(due 2026-05-14):1 of 3 substrates grep-confirmed,~30-50 LOC delta |
| 21' | Codex review — Phase 1b decode-panic finding | `156d2c2` precursor | "decode step for hybrid checkpoint will panic instead of serving" → gate option chosen |
| 22 | Codex — targeted-test interpretation verified SOLID | `156d2c2` | Pre-commit empirical-impact verification per skill v1.7.0 #19 + v1.8.0 #22 discipline |
| 23 | **Codex — P0.2 SUBSTRATE LANDED**(`feat(cuda): load hybrid W4 Marlin side tensors`)| `232aed5` | **Phase 1b loader-only,reject hybrid at linear dispatch,warmup.rs reverted to `num_slots.min(256)`,5 src + wins entry**;bench TTFT p50 68.4ms regression gate PASS |
| 24 | Claude — current main = cell (c) state verified by code-grep | `717b304` | Partial H7-A evidence collected:cell (c) at c=1 PASS + cell (b) historical 100% turn success(annotated) |
| 25 | Codex — cell (d) recipe with kill criterion(12300c5 attribution) | `1bf408d` | Step-by-step revert recipe + Layer-8 num_slots=8 gate enforcement + restore step;most informative single remaining experiment |
| 26 | Codex — **cell (a) ≡ cell (d) post-P0.2** — 4-cell collapses to 1 experiment | `1ccb448` | Cell-collapse blindness anti-pattern #24 candidate;saved tomorrow's pickup duplicate experiment; **net empirical work = 1 experiment** |
| 27 | Codex — pickup queue 4-cell table surgical update for cell-collapse | `05dbceb` | Reflects post-P0.2 reality:cell (a) ≡ (d),cell (b) non-reproducible,cell (c) = current main,cell (d) only experiment;preserves Layer-8 gate |
| 28 | Codex — docs/index refresh EOD+101(skill v1.8.0 5-candidate batch + cell-collapse codified) | `d738040` | Index resumption pointer current to EOD+101 milestone |
| 29 | Claude — auto-memory `feedback_bidirectional_audit_cycle.md` codified for cross-session persistence | (memory file,not in repo)| Pattern triggering conditions + anti-patterns + rule for next session;persists locally for future Claude sessions |
| 30 | Claude/User — auto-memory **natural-closure heuristic** added | (memory file annotation)| **STOP signal codified**:3+ maintenance-only ticks → pivot to (a) wait for state change,(b) orthogonal work,(c) minimal status updates;**Don't force audit briefs**。Cycle CLOSED at stage 30。 |

### 🌀 R4#6 micro-cycle on truly-orthogonal axis(post main-cycle closure)

Spawned per natural-closure heuristic option (b) "truly orthogonal work"。Demonstrates
fresh micro-cycle on different axis with full Phase 1-8 skill application:

| # | Stage | Commit | Layer |
|---|-------|--------|-------|
| R1 | Claude — R4#6 Phase 0 audit(linear.rs:67-99 dispatch site)| `6ade2d4` | Source audit |
| R2 | Codex — audit-of-audit verified 6/6 SOLID + 3-line refinement | `5bb99d7` | Audit-of-audit |
| R3 | Claude — env-gated override implementation | `02209f4` | Substrate |
| R4 | Codex — wins entry pending-remote stub | `a377e57` | Bench-stub discipline |
| R5 | Claude — preliminary 60s bench + Layer-9 SOLID gap discovered(hypothesis-context vs implementation-context)| `fe9ea8a` | **Empirical bench eliminates audit blindspot both code-level audits missed** |
| R6 | Claude — refined `batch ∈ 2..=8` gate + 120s re-bench | `3b9cc06` | **Phase 8 license-or-kill:KILL +37% ITL regression** |
| R7 | Claude — wins entry pending-remote → KILLED | `3b9cc06`(co-commit) | Closure |

**Outcome**:R4#6 W4A16BatchGemv override hypothesis EMPIRICALLY KILLED via 120s
N=52 bench(σ/mean<0.5%)。Marlin tensor-core utilization beats W4A16BatchGemv
launch-overhead savings even at decode batch=4。Hypothesis eliminated from
M_quant Round 4 active queue。

**Skill v1.8.0 anti-pattern #25**(hypothesis-context vs implementation-context
mismatch)EMPIRICALLY GROUNDED via this micro-cycle:both Phase 0 audit
(`6ade2d4`)+ audit-of-audit(`5bb99d7`)missed the prefill-context leak;only
empirical bench(`fe9ea8a`)caught it。**Bench is the truly orthogonal SOLID
layer that catches what code-level audits both miss**。

**Methodology insight**:7-stage R4#6 micro-cycle proves the bidirectional audit
pattern + natural-closure heuristic generalizes beyond the original c20b1ce
chain。Same compounding rigor produces empirical KILL on different axis with
same skill discipline applied。

**Pattern**:each prescription layer audited by other side。Compounding rigor。
**Outcome**:B3 Step 2 LANDED LICENSED -24.2% TTFT,**3 skill v1.8.0
candidates(batch ready)**,P0.2 dispatched + active(near-LANDED),Phase 1.A
recipe verified-correct,**c20b1ce dead-code revealed via 7-layer chain closure**。

### NEW pre-Phase-1.A action — 4-cell A/B → **1 experiment** on post-P0.2 main(per `3fea979` + `1ccb448` cell-collapse)

**Goal**:empirically confirm 12300c5 is real fix(c20b1ce dead code already reverted by P0.2 232aed5)。
**Effort**:~30 min Claude-side(SINGLE experiment per cell-collapse finding `1ccb448`)

| cell | revert c20b1ce | revert 12300c5 | post-P0.2 reality | predicted turn success |
|------|:------:|:------:|--------|:---:|
| (a) | YES | YES | **same as (d)** — c20b1ce already reverted | 76% |
| (b) | NO | NO | **non-reproducible** post-P0.2(c20b1ce removed permanently)| historical 100% only |
| (c) | YES | NO | **current main**(P0.2 reverted c20b1ce + 12300c5 kept) | **100%(c=1 PASS partial confirmed `232aed5`)** |
| (d) | NO | YES | **revert 12300c5 only**(c20b1ce stays reverted) | **76%(predicts H7-A,run via `1bf408d` recipe)** |

**Net empirical work**:**1 experiment** = run cell (d) recipe `1bf408d`(~30 min wall-clock,sed Some(8)→Some(4))。
Result serves as BOTH cell (a) AND cell (d)per `1ccb448` collapse finding。

If (d) shows ~76%/300ms TTFT:H7-A confirmed → 12300c5 is real fix → annotate 3 wins entries(✅ done) + skill v1.8.0 anti-pattern #22(twin-commit attribution)transitions to empirically-grounded codified rule。
**Codex completed 3-doc attribution scan**(stages 19+20):
- ✅ `2026-05-08-warmup-fix-c20b1ce-verified-92pct-turn-success.md`(annotated `655accf`)
- ✅ `2026-05-08-w3-c4-cap8-default-clean-100pct-tt-improved.md`(annotated `655accf`)
- ✅ `2026-05-08-cap8-chain-final-synthesis.md`(annotated `9bc4729`)

⚠ **Layer-8 SOLID requirement on the 4-cell A/B itself**(per `655accf`):
**FIX num_slots constant across all 4 cells**。Original "validation bench"
had `bwa4piqqx`(num_slots=4)vs `b1mm1k0r7`(num_slots=16)= multi-variable
confound。4-cell A/B MUST run with `--num-slots 8`(production-default)across
ALL cells,changing ONLY the c20b1ce/12300c5 revert axes。Otherwise we
repeat the original confound trap。

Ordering:**4-cell A/B BEFORE Phase 1.A nvtx decomposition** — corrects
the bimodal-root-cause assumption baseline before Phase 1 invests effort。

### Concrete cell-by-cell pickup-ready recipes(per stages 24-25)

| Cell | Status | Recipe |
|------|--------|--------|
| (b) | ✅ Historical bench data + annotation | Pre-`232aed5` main 100% turn success per annotated wins entries |
| **(c)** | ✅ Code-grep verified post `232aed5` | Current main = cell (c)。c=1 partial bench PASS。**W3 c=4 cap=8 verify needed**(~10 min Claude bench) |
| **(d)** | 📋 **Recipe ready** | `docs/research/2026-05-09-eod98-cell-d-recipe-12300c5-attribution-kill-criterion.md`(`1bf408d`)— sed `Some(8)→Some(4)`,build,bench,restore(~30 min) |
| (a) | ❌ **Redundant per `1ccb448`** | Cell (a) ≡ Cell (d) post-P0.2(c20b1ce already reverted by `232aed5`)→ same single experiment closes both cells |

## ⚠ STRATEGIC RE-ORDERING(2026-05-09 EOD)— per codex `d2c2c17`

**Per codex strategic axis-ROI brief**(`docs/research/2026-05-09-eod83-post-b3-strategic-next-axis-roi.md`,`d2c2c17`):

> B3 Step 2 closes **only 1/3 of multi-tenant gap**(241ms vs world-#1 121ms target = still 1.99× gap)。
> Long-ctx 4k/8k 2× gaps **unchanged**(B3 doesn't help)。
> **SOLID gap**:none of P1(KV W4A8 / Medusa / P0.2 / P0.3)directly targets multi-tenant median TTFT gap。
> **Risk**:burning 15-25d on KV W4A8 + Medusa,still find world-#1 gap unchanged。

### P0.0 — Phase 1 evidence decomposition(NEW priority,blocks P1 axis lock-in)

**Effort**:0.5-1d Claude-side
**ROI**:30:1 risk-adjusted return — unblocks 15-25d P1 axis decisions
**File**:None;analysis-only(produces research entry + axis re-prioritization)

**Phase 1.A** — multi-tenant 241ms TTFT nvtx decomposition:
- Add nvtx ranges around 4 phases:`prefix::lookup`,`prefill::compute`,
  `first_decode::compute`,`scheduling::overhead`
- Run multi-tenant burst,nsys 30s
- **Use absolute ms not NVTX-window % per §0 SOLID rule 6**(2026-05-08 EOD+19 framing trap)

**Phase 1.B** — SGLang baseline verification:
- Re-verify SGLang's 157ms multi-tenant + 973ms 4k long-ctx baselines on
  identical hw(possible SGLang shipped optimizations 2026-05-07 → today)
- If SGLang baseline drifts → recompute world-#1 gap math

**Decision tree post-Phase 1**:
- If multi-tenant TTFT 60% prefix-lookup → invest radix-cache opt,deprio KV W4A8/Medusa
- If multi-tenant TTFT 60% first-decode-attention → KV W4A8 ROI valid
- If long-ctx 60% prefill compute → P0.2 Hybrid + chunked prefill = P0',KV W4A8 demoted

**KILL criteria**:Phase 1 shows no single dominant phase(<40% any phase)→ pivot to architectural / Option D(re-target hw/model tier — ROADMAP §Next-Model)。

---

## P0 — Pickup directives(cron-Claude paste-buffer ready)

### P0.1 — B3 Step 2 PrefixAwareAdmission CUDA-runtime gate ✅ LICENSED(pending re-bench gate per `ec5c37c`)

**Status:LICENSED -24.2% multi-tenant TTFT(318 ms → 241 ms median,σ/mean=4.5%)**

⚠ **License bench staleness gate**(codex `ec5c37c` 2026-05-09 meta-SOLID):
`3c334ef` bench(-24.2%)was measured on **Round-2-P2 codepath**。Round 4 P2
tightened warm signal to **runnable-only**(removed wrong-warm classification
leak)。Bench number may drift ±5% on actual ship codepath。Pre-commit checklist:
- [ ] Re-bench(option a)or explicit caveat in wins entry(option b)
- [ ] Update wins entry numbers if drift > σ band
- [ ] Add Round-4 attribution in wins entry Problems/Learnings

§0 meta-application:license-or-kill on the LICENSE itself。Bench number must
match committed codepath。



Codex implementation 193 LOC across 5 files,EXCEEDS dispatch directive
with senior-quality fail-open guard against admission deadlock。Wins
entry:`docs/experience/wins/2026-05-09-bench-b3-step2-prefix-aware.md`。
Default policy preserved as `queue-bound`(prod-safe);prefix-aware
opt-in via `--admission-policy=prefix-aware`。

GuideLLM `turns=3, prompt=6000, session=4` shape produces 12k-18k
actual tokens > 8192 max-seq-len → invalid zero-output data filtered。
LICENSED via separate `scripts/bench_multitenant_burst.py` shared-prefix
warm-cache n=5 (244/241/218/239/249 ms) median 241 ms。

Next-tick directive(after codex commit):dispatch P0.2 hybrid loader。

**Original directive preserved below for record**:

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
- **File**:`infer/src/weight_loader.rs:514`(detection site,top-level not model/)
- **Risk**:Med(loader changes need careful testing)
- **ROI**:enables -14% E2E at c≤8 production
- **Dependency**:NONE — Phase 0(`1959a21`)+ Phase 1a(`b6502f7`)done

**Dispatch directive**:
```
Hybrid Phase 1b — loader patch for marlin_w4_hybrid checkpoint

Reference plan: docs/plans/M_quant-hybrid-phase1b-loader-directive.md (commit 6be30ce)

File: infer/src/weight_loader.rs:514 (verified 2026-05-09 — top-level, not in model/)
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

**Why**(corrected post `3fea979` 7-layer closure):bimodal failure
distribution(67% normal / 33% degraded per `db20d34`)POST-12300c5
may persist if first-burst prefill GEMM cold-state contributes。
**P0.3 prefill warmup pass closes RESIDUAL bimodal variance,
gated on Phase 0.5 evidence**(per `3456f8f` recipe)— NOT a
"cold-start gap" framing(that was the c20b1ce-era stale claim)。

- **Effort**:**80-100 LOC**(revised down from 150 per codex `eod81` cross-check — `forward_prefill_batch` already exists),**0.75-1 day**
- **File**:`infer/src/scheduler/cuda/core/warmup.rs` + adjacent
- **Risk**:Med(scheduler hot path)
- **ROI**:eliminates bimodal distribution → cap=8 stable -86% TTFT p99
- **Dependency**:NONE
- **SOLID gates added 2026-05-09 by codex `eod81` audit-of-audit + `3456f8f` recipe**:
  - **Phase 0.5**:verify prefill IS the degraded-path root cause。
    **Recipe ready at [`docs/research/2026-05-09-eod82-p0.3-phase0.5-cheap-experiment-recipe.md`](../research/2026-05-09-eod82-p0.3-phase0.5-cheap-experiment-recipe.md)**
    (`3456f8f`)— 3 options:**recommended Option 1**(log counter,~10 min,absolute
    ms evidence not NVTX framing),Option 2(nsys 30s trace),Option 3(dummy curl pre-warm A/B)。
    **Without this,risk burning 80-100 LOC on wrong axis**
  - **Phase 0.5b**:5-min grep `infer/src/ops/` to confirm prefill GEMM
    routing(cublasLt vs TileLang AOT)。If TileLang AOT cubins,warmup buys
    ~0 — pivot needed
- **`forward_prefill_batch` already exists** at `infer/src/model/qwen3/forward.rs:415`
  — call directly,no new model trait method needed

**Dispatch directive**:
```
cap=8 prefill pre-warm — Step 2.B' implementation

Reference: docs/research/2026-05-08-cap8-default-h4-warmup-cap-rootcause.md (db20d34)

Root cause hypothesis (corrected post 3fea979 7-layer closure):
warmup_cuda_graphs covers DECODE batch sizes 1..num_slots via the existing
slot loop (production-default num_slots=8 already covers 1..8 regardless
of c20b1ce no-op). Prefill GEMM at varying PROMPT lengths (M dimension)
is NOT warmed — first burst's prefill shape may hit cold cublasLt
heuristic state. HYPOTHESIS, not yet evidenced — Phase 0.5 cheap
experiment required first.

File: infer/src/scheduler/cuda/core/warmup.rs (verified 2026-05-09 — exists at this path, 296 LOC)

Phase 0 audit findings (2026-05-09 by Claude):
- Current warmup_cuda_graphs() at line 26 ONLY handles decode-shaped paths
  (line 102 comment: "Pass 1: drive forward for each warmup batch size.
   Populates the cublasLt heuristic algo cache for all GEMM shapes used by decode.")
- Decode paths warmed for batch sizes 1..num_slots via existing slot
  loop in warmup_cuda_graphs (production-default num_slots=8 → batch 1..8,
  no c20b1ce dependency per EOD+90 7-layer audit closure 3fea979)
- Prefill kernel paths (different GEMM shapes from decode, varying M
  dimension across prompt lengths) are NOT warmed
- First fresh-server burst at varying prompt lengths may hit cold
  cublasLt heuristic for prefill GEMM shapes → potential residual
  bimodal contribution (HYPOTHESIS — Phase 0.5 cheap-experiment
  per 3456f8f recipe required to evidence)

Add: dedicated PREFILL warmup pass (NEW, not just bump max_bs further)
- Either: append a Pass 2 in warmup_cuda_graphs() that exercises prefill kernel
  paths with prefill-shaped dummy data
- Or: separate warmup_prefill_paths() function called after warmup_cuda_graphs()
- Trigger paths: prefill GEMM (q/k/v/output projections), prefill attention kernel,
  prefill RMSNorm if shape differs from decode

Implementation hint:
- Existing decode warmup uses dummy_tokens=vec![0; max_bs] + slot_indices=0..max_bs
- Prefill warmup needs dummy prompts at varying lengths (e.g., representative
  short=128, mid=512, long=2048 prompt token counts) to populate prefill GEMM
  shapes
- forward_prefill_batch already exists at infer/src/model/qwen3/forward.rs:415
  → call directly, mirror decode warmup pass structure (warmup.rs:190-266) at
  varying prompt lengths. NO new model trait method needed (per codex eod81 cross-check).

§0 SOLID gates (per codex eod81 audit-of-audit, 2026-05-09):

Phase 0.5 — Verify prefill is degraded-path root cause BEFORE writing code:
- "33% degraded path = cold prefill GEMM" is HYPOTHESIS not evidence
- Could equally be: cublasLt heuristic / TileLang cubin disk-load /
  paged-KV alloc pattern / first-burst L2 cold
- Cheap experiments (pick one):
  1. Log counter on first 10 requests post-startup: per-layer
     (prefill_time_ms, decode_time_ms, alloc_time_ms). Compare to steady-state.
  2. nsys 30s trace of fresh-server first burst — identify if p99 ITL
     spike is dominated by prefill::matmul vs decode::* vs paged_kv::alloc
  3. Pre-warm prefill via dummy curl (1-token prompt) before bench →
     if degraded path disappears, prefill warmup confirmed root cause
- WITHOUT this verification, risk burning 80-100 LOC on wrong axis

Phase 0.5b — Verify prefill GEMM routing (5min grep):
- grep prefill GEMM dispatch in infer/src/ops/
- If routes through cublasLt → P0.3 prefill warmup pays off (populate
  algo cache for new M=prompt_length shapes)
- If routes through TileLang AOT cubins → cubins pre-built at compile
  time, no runtime warmup needed/possible, P0.3 axis is wrong

Acceptance criteria:
- W3 c=4 cap=8 fresh-server bench: 5/5 trials within σ < 5%
- TTFT p99 stays at -86% relative to cap=4 (NO bimodal regression)
- Cold-start adds ~250ms (acceptable) but unblocks -86% TTFT p99 stable
- Bench entry: docs/experience/wins/2026-05-XX-bench-cap8-prefill-warmup.md
```

## P1 — Empirically RE-RANKED post nsys decomposition(`aaf0b55` EOD+113)

⚠ **EVIDENCE-LOCKED P1 axis**(per nsys 60s decomposition,prefill::compute = 97% of active GPU time):

### P1.0 ✅ — **Hybrid Phase 1-3 dispatch substrate(#30)** — PROMOTED

- **Effort**:155-175 LOC scaffolding,**1 day**
- **ROI**:**directly targets prefill::compute,the empirically dominant TTFT phase**
- **Evidence**:nsys decomposition `aaf0b55` shows prefill 327ms / 337ms active = 97% of compute time
- **Decision matrix per `d2c2c17`**:prefill dominant → P0.2 Hybrid Phase 2 dispatch wiring is right axis
- **This is the ONLY P1 item directly targeting prefill compute**(KV W4A8/Medusa target decode)

### P1.1 ⏬ — KV W4A8 Phase 0a smoke kernel(#33) — DEMOTED post evidence

- **Effort**:100-400 LOC,**1-3 days**
- **ROI**:**targets decode bandwidth(0.9% of active GPU time per nsys)**,not dominant phase
- **Evidence-based decision**:demoted from P1 priority to P2 axis until prefill optimization saturates
- **Original plan**:`docs/plans/M_quant-kv-w4a8.md`(`1e713de`)— still valid for memory headroom but not for TTFT closure

### P1.2 ⏬ — W4A8 graph capture hoist(#24) — adjacent to P1.0

- **Effort**:200-400 LOC
- **ROI**:enables W4A8 production default-on(currently gated by graph capture)
- **Note**:helps prefill compute(W4A8 prefill -36% TTFT per `b5889b3`)→ aligns with P1.0 evidence-axis

## P2 — Long-horizon(this week,empirically demoted post `aaf0b55`)

### P2.1 ⏬ — M_medusa scaffold(#28) — DEMOTED post nsys evidence

- **Effort**:600-1200 LOC + 1 week training
- **Plan**:`docs/plans/M_medusa.md`(`528844c`)
- **ROI claim**:potentially +20-30% throughput,short-output workloads
- **Evidence-based note**:targets per-token decode latency(decode = 3% of active GPU per nsys),
  NOT prefill compute(97% dominant)。15-25 day commitment risks not moving world-#1 needle。
  Reconsider AFTER P0.2 Phase 2 dispatch closes prefill axis OR if next nsys shows decode-
  dominance with different workload shape。

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
