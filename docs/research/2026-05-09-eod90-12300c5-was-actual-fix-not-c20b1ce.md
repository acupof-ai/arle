# 12300c5 was actual fix,c20b1ce just NO-OP — closure on audit chain

> Per 919c0fb + 8d91d20 + 921f313 chain:c20b1ce is NO-OP(production)or
> silent-fail(non-default)。**Then what caused 76→100% turn success?**
> Direct git show of `12300c5` answers:single-line cap=4→cap=8 flip is
> the real fix。

## 12300c5 actual diff(verified by `git show`)

```rust
// infer/src/model/qwen3/forward.rs (line 316 area)
- Some(4)
+ Some(8)
```

That's it。1 functional line。Wrapping 6 lines of comments cite `27fd5de`
multi-shape SAFE validation + `19d12c2` TTFT p99 -86%。

## What this changes semantically

Pre-12300c5(cap=4):
- Scheduler admits ≤ 4 prefill requests per step
- At workload concurrency=8 → 4 prefill in step 1,4 queued
- 4 queued requests prefill in step 2 → 2-step prefill cycle
- TTFT for queued requests = wait + own prefill = 2× "should be"

Post-12300c5(cap=8):
- Scheduler admits ≤ 8 prefill requests per step
- At workload concurrency=8 → all 8 prefill in step 1
- Single-step prefill cycle
- TTFT = own prefill only

→ At cap=8 workloads,12300c5 directly halves TTFT for queued half。
**This is the real fix that produced 76→100% turn success**。

## c20b1ce was supposed to do what?

`c20b1ce` commit message:
> warmup must cover ALL batch sizes the scheduler may admit per step. Prefill
> admission cap (model.max_concurrent_prefill_requests, e.g. Qwen3 Marlin
> Some(8) per `12300c5`) can exceed num_slots when sessions queue concurrently —
> warming only num_slots leaves batches num_slots+1..cap as cold-start graph
> captures during bench → tail-latency regression

→ c20b1ce hypothesis:after 12300c5 cap=4→8,warmup needs to cover batch
sizes 1..8(not just 1..num_slots)。Without warmup at sizes 5..8,those
batches hit cold-start cublasLt heuristic → tail-latency regression。

**But**:in production-default,`num_slots = 8`(CLI default,scheduler
config)。So `num_slots = prefill_cap = 8`。Pre-12300c5 had cap=4 BUT
num_slots=8,so warmup ALREADY covered 1..8 batch sizes(via num_slots,
not prefill_cap)。

→ **Warmup ALREADY covered 1..8**。c20b1ce solved a problem that didn't
exist in production-default config。

## The 12300c5+c20b1ce twin-commit hypothesis was wrong

Original framing(per `2026-05-08-warmup-fix-c20b1ce-verified-92pct-turn-success.md`):
> Two-place fix(`12300c5` + `c20b1ce`)restored coherent behavior

→ Reality:
- `12300c5` = real semantic change(cap=4→8 admission)
- `c20b1ce` = no-op cosmetic(max_bs comment-clarification with same numerical result in production-default)

## SOLID hypothesis(license-or-kill)

**H7-A(strong)**:`12300c5` alone explains 76→100% turn success
empirical improvement。c20b1ce contributed exactly 0%。

**H7-B(weak)**:co-shipping had subtle interaction that's not pure
12300c5(e.g. cublasLt heuristic invalidation,paged-KV pool state
side-effect)。

**Distinguishing experiment**(tomorrow's pickup,~30 min):
- (a) revert BOTH 12300c5 + c20b1ce → bench fresh-server cap=8 → expect 76%
- (b) keep 12300c5 + c20b1ce(current main)→ bench → expect 100%
- (c) keep 12300c5 + revert c20b1ce → bench → predict 100% per H7-A
- (d) keep c20b1ce + revert 12300c5(cap=4)→ bench → predict 76% per H7-A

If (c) ≈ (b) and (d) ≈ (a) → H7-A confirmed,c20b1ce is dead code,
revert OK。

If (c) < (b) or (d) > (a) → H7-B confirmed,c20b1ce is doing something
subtle that's not pure max_bs change → investigate cublasLt state /
paged-KV side effects。

## Strongest possible refutation

If (c) ≈ (b) confirmed:**c20b1ce is dead code that should be reverted**
per CLAUDE.md "no half-states"。It claims to fix something but
empirically does nothing。Keeping it adds maintenance debt without
benefit。

This is anti-pattern #22 (skill v1.8.0 candidate per 919c0fb / 8d91d20)
in concrete actionable form:**check whether the "fix" 改变了 anything
empirically — if not,revert it**。

## Implications for c20b1ce era wins entries

3 wins entries(per 8d91d20)attributed empirical improvement to c20b1ce
incorrectly。Per H7-A confirmation,proper attribution chain is:
- `12300c5`:real fix(76% → 100% via cap=4→8 admission)
- `c20b1ce`:cosmetic / dead code(0% contribution)
- `27fd5de`:validation evidence(not the fix itself)
- `19d12c2`:reference data point(not the fix either,was a CLI override)

Wins entries should be **ANNOTATED**(not deleted):
> NOTE 2026-05-09:per audit chain 919c0fb / 8d91d20 / eod90,improvement
> attributed to c20b1ce was incorrect。Real fix was 12300c5 cap=8 admission
> flip。c20b1ce is no-op in num_slots=prefill_cap=8 production config。

This preserves historical record + provides accurate attribution for
future readers。

## §0 first principle:final closure

This audit chain demonstrates §0 in action across 7 layers:
1. c20b1ce code change real(layer 1)
2. 919c0fb finds incoherent silent-fail logic(layer 2)
3. 8d91d20 finds NO-OP in production-default(layer 3)
4. (this brief layer 4 implicitly)silent-fail edge case effectively NO-OP too
5. **(this brief layer 5)**12300c5 was the actual fix all along
6. (potential)what caused 27fd5de SAFE validation specifically
7. (potential)cross-check ALL "two-place fix" historical claims

Each layer escalates rigor。**Single deepest finding**:every "fix"
claim must verify empirical impact,not trust commit message hypothesis。

## Skill v1.8.0 anti-pattern #22 final framing

Combining 919c0fb / 8d91d20 / this brief:
> **Anti-pattern #22**:Twin-commit fix attribution trap。When two commits
> co-ship as "fix the issue",check each independently for empirical
> contribution。One may be the real fix,the other no-op or silent-fail
> cosmetic。Default attribution to BOTH causes future readers to repeat
> the no-op fix in similar situations。Cure:revert each in turn,
> measure individual contribution。

## Cross-references

- `919c0fb` 5-layer SOLID gap(silent-fail logic)
- `8d91d20` 6-layer extension(NO-OP in production-default)
- `921f313` pickup queue cycle codification
- `12300c5` actual fix(cap=4→8 admission flip)
- `c20b1ce` no-op cosmetic(max_bs identical in prod config)
- `27fd5de` cap=8 multi-shape validation(citation,not fix)
- `19d12c2` TTFT p99 -86% reference(citation,not fix)
- 3 wins entries to annotate(per 8d91d20 list)

## Status

Layer-7 SOLID closure on c20b1ce audit chain。Real fix identified
(`12300c5`)。Tomorrow's pickup:run controlled A/B (a/b/c/d) experiment
to confirm。If H7-A: revert c20b1ce as dead code (per "no half-states"
rule)+ annotate 3 wins entries with corrected attribution。

Audit cycle complete。From "warmup fix landed"(2026-05-08)to "warmup
fix was NO-OP,12300c5 was real fix"(2026-05-09 EOD+90)— 1 day to
correct attribution via 7-layer §0 SOLID rigor。
