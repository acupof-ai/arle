# B3 Step 2 LICENSE bench staleness — pre-merge gate concern

> Per §0 SOLID 第一原则 + CLAUDE.md "MANDATORY — every runtime change
> produces a bench entry":bench numbers must reflect the codepath being
> committed,not an earlier iteration of it。
>
> Round 4 P2 fixes(2026-05-09 EOD+84,codex still iterating)tightened the
> warm-signal definition,which **may invalidate the `3c334ef` LICENSE
> bench numbers**。This brief flags the gate before merge。

## What changed in Round 4 P2

Per codex pane status(EOD+84):
> 两个 runtime P2 已修:warm 信号和 waiting hint **都只看实际可运行的
> reuse/staged/attach prefix**;全冷 fail-open 现在按空闲 slot 数放行,
> 而不是只放一个。加了一个 hint 单测锁住"raw match 但不可运行不算 warm"。

**Translation**:
1. **warm signal**:was raw lookup match → now only runnable
   reuse/staged/attach prefix
2. **waiting hint**:same fix
3. **fail-open behavior**:全冷 admit count 1 → 按空闲 slot 数放行
4. New regression test:`waiting_hint_ignores_non_runnable_lookup_matches`

→ All 3 changes are SOLID correctness improvements that **change runtime
admission behavior**。

## Why this invalidates `3c334ef` LICENSE bench

The LICENSE bench at `3c334ef`:
- ARLE multi-tenant TTFT p50: 318→**241 ms (-24.2%)**
- σ/mean 4.5%,N=5 paired

But that bench ran on the **PRE-Round-4 P2 codepath** where:
- warm signal was **too permissive**(counted raw lookup matches that
  weren't actually runnable)
- → more requests classified as warm
- → fewer cold_headroom rejections
- → more cold admits → potentially DIFFERENT TTFT than post-fix codepath

→ **Post-Round-4 codepath behavior**:
- More requests correctly classified as cold
- More cold_headroom rejections at queue saturation
- Could produce **higher OR lower** TTFT depending on workload mix
- Direction unknowable without re-bench

## SOLID gate per CLAUDE.md

Per CLAUDE.md "MANDATORY — every runtime change produces a bench entry":
> "A change is not 'done' until a dated entry lands under
> `docs/experience/wins/` ... no bench entry → not shipped."

→ The B3 Step 2 final commit must either:
- **(a) Re-bench**:run `scripts/bench_multitenant_burst.py` against
  Round-4-final codepath,update wins entry with new numbers
- **(b) Explicit caveat in wins entry**:document that "-24.2% measured
  on Round 2 P2 codepath; Round 4 P2 tightening behavior could shift
  numbers ±5%, scheduled re-bench tracking-ticket #..."

Option (a) is preferred per SOLID。Option (b) is acceptable IF:
- Workload is low-cold-request-volume(small impact from fail-open
  behavior change)
- AND the post-fix unit test `waiting_hint_ignores_non_runnable_lookup_matches`
  proves the pre-fix had wrong behavior(license bench measured wrong-
  warm classification leak,not the actual policy benefit)

## Why this matters

If we ship `3c334ef` -24.2% claim with Round-4 codepath and the actual
production behavior is +5% TTFT(regression),we'd be making a false
claim in the wins entry。The B3 Step 2 axis would still be valid(prefix-
aware admission helps in principle),but the empirical % would be wrong。

If we ship with -30% TTFT actual(better than -24.2%),we'd be
under-claiming and missing celebration room。

**Either way:bench number must match committed codepath**。

## Recommended action

**Pre-commit checklist for B3 Step 2 final batch**:
- [ ] Run `scripts/bench_multitenant_burst.py` against Round-4-final
      build,N=5 paired runs(matching `3c334ef` protocol)
- [ ] Update wins entry numbers if drift > σ band(>±5% per recorded
      σ/mean 4.5%)
- [ ] Add Round-4-fix attribution line in wins entry "Problems" or
      "Learnings" section explaining the warm-signal correctness
      tightening
- [ ] If drift in expected direction(e.g. tighter cold semantics →
      slightly worse TTFT under heavy cold load,better TTFT under
      heavy warm load),narrative-explain why

## §0 first principle in action

This is a **license-or-kill on the LICENSE itself** — meta-application
of §0:
- 推断 ≠ evidence:can't claim -24.2% on codepath we didn't bench
- 混淆变量必须隔离:Round-2-P2 vs Round-4-P2 are different codepaths,
  same bench number can't span both
- Root cause 假设也要 license-or-kill:"24.2% was solely from PrefixAware
  policy" is hypothesis;some of it might have been from the wrong-
  warm classification leak(now removed by Round 4)

→ Pre-commit re-bench = single SOLID step that resolves all three §0
concerns。

## Cross-references

- `3c334ef` LICENSE bench commit(pre-Round-4)
- Wins entry:`docs/experience/wins/2026-05-09-bench-b3-step2-prefix-aware.md`
- Round 4 P2 fix description:codex pane EOD+84(`warm signal` runnable-only)
- §0 first principle:CLAUDE.md "求真务实,追求极致"
- Bench requirement:CLAUDE.md "MANDATORY — every runtime change produces
  a bench entry"
- 5-min framing trap:`2026-05-08 EOD+19 nsys framing trap`(separate but
  related — measurement-vs-actual semantics)

## Status

Pre-merge gate brief。Codex should treat this as a Round 4 review
finding(if review didn't already catch it),resolve before push:

**Path A**(recommended): re-bench Round-4-final build → update wins
numbers → commit batch with accurate bench

**Path B**(acceptable with caveat): commit batch + wins entry with
prominent "Round-4-P2 may shift ±5%" note + tracking ticket for
post-merge re-bench within 24h

Either path resolves §0 SOLID concern。Silent ship of pre-Round-4 numbers
on post-Round-4 code = SOLID violation。
