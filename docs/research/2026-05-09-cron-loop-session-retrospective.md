# 2026-05-09 cron-loop session retrospective — Claude-as-manager methodology

> 100+ commit day(2026-05-08 → 2026-05-09)concluded with codex
> picking up B3 Step 2 PrefixAwareAdmission and shipping it LICENSED
> -24.2% TTFT(σ/mean=4.5%)。This retrospective captures the
> methodology that worked,for future cron-loop session continuity。

## Session topology

| Role | Activity | Commits |
|------|----------|--------:|
| **Codex**(substrate implementor)| B3 Step 2:193 LOC across 6 files + wins entry | 1(pending)|
| **Claude**(manager-auditor)| Pickup queue + audit research entries + skill anti-patterns + path fixes | 11+ |
| **Cron loop**(scheduler)| Triggered ~12 ticks across 90+ minutes | n/a |

## Methodology that worked

### 1. Pickup queue with paste-buffer-ready dispatch directives

**Artifact**:[`docs/plans/codex-pickup-queue-2026-05-09.md`](../plans/codex-pickup-queue-2026-05-09.md)

7 codex pickup items prioritized P0/P1/P2/P3,each P0 ships with a
dispatch directive containing:
- File paths(verified at write-time per skill v1.7.0 #19)
- LOC + day estimate
- Acceptance criteria
- Code skeleton + bench expectation

**Value**:cron-fired Claude can paste-buffer the directive to codex
in <1 minute,no context-rebuild。Tonight executed P0.1 successfully
via this exact path。

### 2. Mid-flight read-only audit during codex execution

**Pattern**:while codex `Working`(build/test/bench cycles),Claude
performs read-only `git diff` + code-grep to verify codex's
implementation matches the directive。

**Value**:catches drift early without disrupting codex flow。On B3
Step 2,my audit at codex `12m 12s` confirmed:
- Architecture matches refined plan(reuse existing `lookup_or_stage`)
- `turn_depth: 0` hardcode is intentional + safe
- **codex EXCEEDS directive** with senior-quality fail-open guard
  against admission deadlock(I did NOT specify this — codex
  identified the risk independently)

The audit research entry(`f41d7c9`)preserves the architectural
reasoning that codex's commit message field would not capture。

### 3. Cross-tick continuity via fresh-day pickup queue

**Insight**:cron-fired Claude sessions lose conversation context but
retain repo state。Cross-session artifacts(pickup queue,EOD anchor,
audit research entries)become the continuity layer。

**Tonight's loss-vector tests**:
- `14116c1` docs(index) had broken link → `8935851` 2-char fix
- pickup queue P0.2 had wrong path → `de8b4dc` fix
- These are exactly the kind of cross-session-artifact stale-path
  failures that anti-pattern #19 codifies(skill v1.7.0)。

### 4. Bench shape validation discipline

**New rule from B3 Step 2 wins entry**:bench generators(guidellm,
chat templates,turns expansion)can produce request shapes whose
**actual token count exceeds server max-seq-len** → zero-output
invalid data → false license attempt。

**Diagnostic**:service trace `Prefix hit rate: peak 0.0%` when bench
named `prefix-aware` but server is queue-bound default → instant red
flag for label/config mismatch。

This is special-case of anti-pattern #8(production default ≠ A/B
baseline)applied to admission-policy server-flag axis。

## Skill anti-pattern accumulation today

| # | Title | Source incident |
|---|-------|------|
| 14 | Upstream-data parser silent corruption | qzeros off-by-one(`5593865`) |
| 15 | Warm-server implicit dependency trap | cap=8 bench(`db20d34`)|
| 16 | Implicit-coupling-via-shared-default | warmup-max=4 hardcode(`db20d34`)|
| 17 | Bimodal failure distribution masks single-run LICENSE | cap=8 67/33 split |
| 18 | Phase 0 substrate audit before scoping new wiring | A1 RadixCache audit(`1217375`)|
| 19 | Dispatch directive path verification | broken-path tonight(`8935851`,`de8b4dc`)|

**v1.4.0(14)→ v1.7.0(19)= 5 anti-patterns codified in single 100+
commit day**。Each is empirically grounded in tonight's incidents
caught during cron-loop ticks。

## Phase 8 license-or-kill — B3 Step 2 record

| Acceptance criterion | Result | Status |
|----------------------|--------|--------|
| TTFT improvement(B3 multi-tenant burst) | 318 ms → 241 ms median = **-24.2%** | ✅ >10% threshold |
| σ/mean across n=5 runs | 4.5% | ✅ <5% threshold |
| cargo fmt --all --check | PASS | ✅ |
| cargo check --release --features cuda | PASS | ✅ |
| cargo check --release --features metal,no-cuda | PASS | ✅ |
| cargo clippy --release --features cuda -- -D warnings | PASS | ✅ |
| cargo test scheduler::types::tests | PASS | ✅ |
| cargo test --features cuda scheduler:: | PASS | ✅ |
| 565 lib tests PASS | PASS | ✅ |
| metal_eval_audit | FAIL(pre-existing,unrelated per `f41d7c9` audit) | 🟡 documented |
| Multi-tenant TTFT improvement σ < 5% | 4.5% | ✅ |
| Wins entry cited | `2026-05-09-bench-b3-step2-prefix-aware.md` | ✅ |

**Outcome**:LICENSED with explicit gap reporting(no false claim of
hitting 157ms early target)。Production-safe via opt-in default。

## Commits this session(Claude-side)

| Commit | Title |
|--------|-------|
| `f6a4869` | docs(plans): codex pickup queue 2026-05-09 EOD |
| `8935851` | docs(index): fix broken pickup queue link |
| `125f795` | docs(skill): kernel-opt v1.6.0 — anti-pattern #18 |
| `de8b4dc` | docs(plans): pickup queue P0.2 path fix |
| `c768b70` | docs(skill): kernel-opt v1.7.0 — anti-pattern #19 |
| `f41d7c9` | docs(research): B3 Step 2 codex mid-flight audit |
| `24f3bef` | docs(research): B3 Step 2 bench baseline-regression analysis |
| `3c334ef` | docs(plans): pickup queue P0.1 LICENSED -24.2% |
| `(this commit)` | docs(research): cron-loop session retrospective |

**Codex pending commit**:B3 Step 2 implementation(193 LOC,6 files
+ wins entry)。

## Tomorrow's pickup readiness

P0.2 dispatch brief pre-staged at `/tmp/codex-brief-p0.2.txt`(63
lines,paste-buffer-ready)。File path verified `infer/src/weight_loader.rs:514`
(top-level)。Phase 0 reconnaissance + Phase 1a tool ✅ done。

Estimated:155-175 LOC,0.75-1d codex effort。Acceptance via existing
test suite + new hybrid loader unit test + bench entry。

## Cross-references

- Pickup queue: [`codex-pickup-queue-2026-05-09.md`](../plans/codex-pickup-queue-2026-05-09.md)
- B3 Step 2 wins entry(codex pending): `docs/experience/wins/2026-05-09-bench-b3-step2-prefix-aware.md`
- B3 Step 2 audit chain: `f41d7c9` + `24f3bef` + `3c334ef`
- Skill v1.7.0: `c768b70`
- Phase 1b directive: `docs/plans/M_quant-hybrid-phase1b-loader-directive.md`(`6be30ce`)

## Rule

**Cron-loop session as continuous-delivery substrate**:
- Pickup queue(EOD)= continuity layer between sessions
- Mid-flight audit(during codex Working)= management value extraction
- Path verification at write-time = anti-pattern #19 discipline
- Bench shape validation(token count vs max-seq-len)= matched-control discipline applied to bench shapes,not just policy axes

Each cron tick must:
1. Capture 3 status(parallel)
2. Pick path per table(don't idle wait)
3. Execute commit + push(NULL result also commits)
4. Report state + ScheduleWakeup

This methodology turns 90 minutes of asynchronous codex work into
**11+ Claude knowledge-accumulation commits** at zero marginal idle
cost — the cron loop's substrate-management value。
