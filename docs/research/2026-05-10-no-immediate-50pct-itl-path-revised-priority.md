---
title: REVISION — no immediately-actionable -50%+ ITL path on sm_89 W4 decode (M_spec classical KILLED at 4k)
date: 2026-05-10
type: research
status: strategic-revision-supersedes-61c9666-priority
---

# REVISION — no immediately-actionable -50%+ ITL path on sm_89 W4 decode (M_spec classical KILLED at 4k)

> Per skill v1.10.0 #28 anti-hallucination rule applied to my own
> prior `61c9666` synthesis. While surveying spec decode infra to plan
> #28 pickup, found M_spec plan tail evidence I missed: classical
> external-draft spec decoding was tested twice and KILLED at -73%
> and -46% tok/s. My "P0 = #28 spec decoding -50%+ ITL" framing in
> `61c9666` was DIRECTIONALLY CORRECT but oversimplified — the actual
> available path is Medusa (~500 LOC + 1 week training), not the
> immediate fix the priority matrix implied.

## §0 Direct evidence (raw `tail` + `wc -l` this tick, NOT memory recall)

### M_spec plan tail (raw quote)

```bash
$ tail -40 docs/plans/M_spec-decode-classical-bench-first.md
```

> "Phase 0 (this plan, classical bench) — **CLOSED at 4k random text**.
>  Hold for re-test on W3/W4 shape before further classical work."
>
> "Both KILLs honor anti-pattern #13 (NULL is real elimination)...
>  Without methodology, 'spec-decode tested twice, got -73% and -46%
>  tok/s, KILL the axis' would be the natural reaction — wrong.
>  Workload distinction matters."
>
> "Plan execution order revised:
>  - Phase 0 (this plan, classical bench) — CLOSED at 4k random text.
>  - Phase 1 — pivot to either Medusa training OR W3/W4 harness validation.
>  - Phase 2+ — depends on Phase 1 evidence."

### Spec decode substrate inventory

```bash
$ wc -l infer/src/speculative.rs
721 infer/src/speculative.rs

$ grep -nE "^pub fn|^pub struct|DraftMode" infer/src/speculative.rs | head
30: //! - DraftModel trait (GPU stub)
47: pub struct SpecConfig {
91: pub struct TokenProposal {
154: pub struct VerificationResult {
201: pub fn verify_tokens(...)
250: pub fn verify_tokens_greedy(...)
303: pub struct AcceptanceTracker {
369: // DraftModel trait (GPU stub)        ← Medusa would replace stub
376: pub trait DraftModel: Send + Sync {
392: pub struct MockDraftModel {           ← test only
436: pub fn expected_speedup(...)
```

```bash
$ grep -n "spec_" infer/src/main.rs | head
142: spec_enabled: bool,
146: spec_draft_k: usize,
150: spec_acceptance_threshold: f32,
154: spec_draft_model: String,
158: spec_sparse_kv_enabled: bool,
162: spec_sparse_recent_tokens: usize,
166: spec_sparse_top_k_pages: usize,
```

```bash
$ ls -d infer/models/Qwen3-0.6B
infer/models/Qwen3-0.6B    ← external draft model on disk
```

### Task list state

```
#27. [completed] M_spec external draft Qwen3-0.6B bench — KILLED at 4k random
#28. [pending]  M_medusa scaffold (codex own, ~500 LOC + 1 week training)
```

## §1 What this means for the prior `61c9666` priority

`61c9666` claimed:

> Real ITL win mechanisms on sm_89 W4 decode (revised priority):
> P0  Spec decoding (#28, blocked on #34 HF Hub) — 500 LOC + training,
>     **-50%+ ITL** via amortized weight read

**Correction**: spec decoding has TWO sub-paths:
1. **External small-model draft** (#27, classical) → KILLED at 4k
   random text (-73% / -46% tok/s, two bench runs)
2. **Medusa multi-head** (#28, target hidden states) → pending,
   ~500 LOC + 1 week training, NOT yet attempted

So #28 Medusa is the ONLY remaining spec-decode path. -50%+ ITL is
still the theoretical ceiling (per amortized weight read formula),
but EXTERNAL draft already proved this doesn't work at 4k random
text — Medusa shares target hidden states (per M_spec plan §1.5),
which the plan predicts gives "naturally aligned" higher acceptance
rate, but this hypothesis is UNPROVEN until codex executes #28.

## §2 Honest revised priority (supersedes 61c9666 §7)

| Priority | Path | LOC | Wall | Predicted gain | Risk |
|----------|------|-----|------|----------------|------|
| **P0 (in flight)** | Phase 1 dequant.h port (#42) | 687 | 1.5-2 days | **-3-8% ITL** (modest, real) | low |
| **P1 (queued)** | NEW prefill-only FP8 (P0.A 5.21× evidence) | ~700 | 2 days | **-8-16% TTFT** (separate from ITL) | medium |
| **P2 (queued small)** | #34 CLI surface (~30-50 LOC, 7feb260) | ~50 | 1h | unblocks downstream | low |
| **P3 (research)** | W3/W2 quantization | TBD | 1 wk research | -25-50% ITL ceiling per quant level | unknown |
| **P4 (long-term)** | #28 Medusa scaffold | 500 | 1 wk + training | -50%+ ITL IF acceptance ≥ 70% (UNPROVEN) | high |
| **P5 (deferred)** | Re-test classical spec on W3/W4 shape | 0 (just bench) | 0.5 day | -? unknown | low |

**Key honest revision**: P0 is now the modest Phase 1 (-3-8% ITL),
not the speculative -50%+. The -50%+ target requires Medusa work
that has 1-week training cost + UNPROVEN acceptance rate at our
shapes.

## §3 What user's stated "-20-40% ITL" target maps to (honest assessment)

| Target | Actual achievable mechanism on sm_89 W4 decode | Status |
|--------|------------------------------------------------|--------|
| **-3-8% ITL** | Phase 1 dequant.h port | in flight #42 |
| **-8-16% TTFT** | Prefill-only FP8 (codex P0.A 5.21× signal) | new directive needed |
| **-20-40% ITL via FP8** | NONE — structurally infeasible per `61c9666` (W4 decode HBM-bound) | killed |
| **-25-50% ITL via W3/W2** | Research path, no immediate impl | P3 |
| **-50%+ ITL via Medusa** | Hypothesis only, 1-week training + UNPROVEN acceptance | P4 |
| **0% ITL via classical spec (Qwen3-0.6B)** | Already tested twice, KILLED at -73%/-46% on 4k random | dead |

**Honest message to user**: the -20-40% ITL target via "Machete-from-vLLM"
or any FP8 path is structurally infeasible on sm_89 W4 decode. The
path that COULD reach the target floor is Medusa (#28), but that's
1-week training cost with UNPROVEN acceptance rate at our workload
shape.

Realistic 2-tick deliverable: Phase 1 (-3-8% ITL) + prefill-only FP8
(-8-16% TTFT). Combined: meaningful but doesn't reach -20-40% ITL.

## §4 What this changes for cooperative work

- **Codex's current Phase 1 work (in flight)** is correctly the P0
  pickup — modest -3-8% ITL but actually achievable
- **#34 unblock** still useful but its strategic value DROPPED:
  unblocks #28 which is now P4 (long-term + uncertain), not P0
- **Prefill-only FP8 directive (~700 LOC)** is the correct P1 next
  pickup post Phase 1 — codex's P0.A 5.21× prefill signal is real,
  delivers TTFT axis -8-16%
- **W3/W2 quant research** (P3) still has the ceiling argument but
  needs significant calibration work — not immediate

## §5 Recommended PushNotification to user

> "Strategic revision: my prior 61c9666 'P0 = #28 spec decoding
> -50%+ ITL' was oversimplified. M_spec plan classical external draft
> already KILLED at 4k random (-73%/-46% tok/s, two prior runs).
> Only Medusa (#28) remains for spec decode = 1 week training +
> UNPROVEN acceptance rate. Honest revised priority: Phase 1 in
> flight (-3-8% ITL), prefill-only FP8 next (-8-16% TTFT). User's
> -20-40% ITL target via FP8 path is structurally infeasible per
> 61c9666 architectural analysis. -50%+ ITL via Medusa is hypothesis
> only with 1-week cost. Real near-term wins are modest. Documenting
> for transparency."

NOT dispatching this tick — codex still mid-Phase 1, this is for
post-Phase 1 commit when there's a natural decision point.

## §6 Cross-references

- M_spec plan: `docs/plans/M_spec-decode-classical-bench-first.md` (the source of the KILL evidence)
- Existing spec substrate: `infer/src/speculative.rs` (721 LOC)
- CLI flags wired: `infer/src/main.rs:142-166` (--spec-enabled, --spec-draft-k, etc.)
- Draft model on disk: `infer/models/Qwen3-0.6B`
- Phase 0 P0.A KILL: `docs/research/2026-05-10-phase0a-decode-kill-architectural-implication.md` (61c9666) — superseded for priority section, architectural analysis still valid
- Phase 1 in flight: `docs/research/2026-05-10-phase1-substep1.1-codex-impl-audit-clean.md` (70b4d7b) + `docs/research/2026-05-10-phase1-substep1.2-atomic-add-brief-pre-draft.md` (43bda9c)
- #34 rescope: `docs/research/2026-05-10-task-34-rescope-substrate-exists-only-cli-surface-missing.md` (7feb260)
- Skill v1.10.0 anti-pattern #28: `.claude/skills/kernel-optimization/SKILL.md`

## §7 Status

61c9666 priority section SUPERSEDED. Honest revised priority in §2.
Real near-term ITL wins on sm_89 W4 decode are modest (Phase 1 -3-8%,
prefill FP8 -8-16% TTFT). -50%+ ITL via Medusa is uncertain
hypothesis with 1-week training cost. User's -20-40% ITL target may
need horizon-extension OR pivot to architectural change (W3/W2 quant
or different model size).

Per skill v1.10.0 #28: every claim verified by raw `tail`/`wc -l`/
`grep`/`ls` output this tick, NOT memory recall.
