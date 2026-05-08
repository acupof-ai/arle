# cap=8 chain — N=2 second-order issue caveats `8281047` validation + `1fce03f` recommendation

> Per `fc41e7e` N=2 fresh run reveals **deterministic 144/256(56%)
> failure pattern** post-warmup-fix。`8281047` 91.8% claim was
> single-run #1 outlier。My `1fce03f` synthesis recommendation needs
> caveat。

## What changed

`fc41e7e` ran the SAME cap=8 default + `c20b1ce` warmup fix bench again
(N=2)。Result was **eerily byte-identical** to pre-fix run #2:

| Metric | Pre-fix run #2(`b4r8fha82`)| Post-fix run #2(`fc41e7e`)|
|--------|----:|----:|
| Turn success | 144/256(56%)| 144/256(56%)|
| tokens_out | 23424 | 23424(byte-exact)|
| Wall total | 1409 s | 1401 s |
| Peak mem | 15911 MB | 15880 MB |

→ **Deterministic failure pattern**,NOT variance。Warmup fix helps
run #1(76→92%)but run #2 converges to 144/256 regardless。

## Run #1 vs Run #2 binding factor

Cross-N matrix:
| Run | Pre-fix turn % | Post-fix turn % |
|-----|---------------:|----------------:|
| #1 | 76% | **92%** ✓ warmup fix helped |
| #2 | 56% | **56%** ✗ warmup fix didn't help |

Warmup fix's improvement is **CONDITIONAL on Run #1-style initial
conditions**。Run #2-style conditions deterministically fail at 56%
regardless of warmup state。

## Hypothesis space(per `fc41e7e`)

### H1 — Bench harness state(retry budget per-run)
Run #1 wall = 2356 s,Run #2 wall = 1401 s — 1.7× longer。If retries
are time-bound,longer wall = more retries succeed。

But this can't explain BYTE-IDENTICAL tokens_out across pre/post fix。
Same byte count means same exact tokens emitted → same exact session
flow → suggests harness deterministic seed/ordering rather than retry
budget difference。

### H2 — Server-side state persistence across restarts
Maybe state survives `cargo run` restarts(e.g. tmpfs / `~/.cache` /
`/tmp` artifacts)。If yes,Run #2 inherits Run #1 state → different
behavior。

### H3 — Run #1 favorable initial conditions
Run #1 had longer wall + warmup fix → more cold-start opportunities to
capture batches。Run #2 short wall → less retry headroom even with
warmup pre-captured。

### H4 — Bench seed determinism
Same workload spec + same seed → same session ordering → same failure
mode for specific session。If session N deterministically fails,Run #2
hits it earlier in shorter wall budget。

## Implications for my `1fce03f` recommendation

`1fce03f` recommended **A. Keep cap=8 + warmup fix as production
default** based on `8281047` 91.8% turn success。Per `fc41e7e`,actual
expectation is **bimodal 56-92% depending on which "run mode" production
hits**。

Updated trade-off:
- A. Keep cap=8 + warmup:**56-92% turn success bimodal**,p99 ~10s
- B. Revert cap=4:**100% turn success deterministic**,p99 ~72.5s
- A still beats B on TTFT but **turn success unpredictable**
  - Worst-case A:56% × 9.5 + 44% × 19 retry = **13.7s amortized p99 worst-case**
  - vs B:72.5s deterministic
  - A still 5-7× better even at worst-case bimodal

→ **Recommendation B vs A trade-off remains A,but with bimodal caveat**。
Production users will see TTFT improvement consistently,but turn-success
% will fluctuate run-to-run。

## Investigation needed(P1)

Per `fc41e7e` codex action:
- Step A — bench harness deterministic-failure isolation:
  - Which session indices fail in run #2 vs run #1?
  - Are they the same sessions or random?
- Step B — server log diff between run #1 and run #2 starts:
  - Compare RUST_LOG=info output line-by-line
  - Look for memory pool init differences,KV pre-alloc patterns
- Step C — Sleep/reset between runs:
  - Add `sleep 60 && nvidia-smi --gpu-reset` between bench runs
  - If run #2 → 92% with reset,confirms server-side state persistence

## Updated production policy

Until investigation resolves:

**Conservative production deployment**:
- Keep `12300c5` cap=8 + `c20b1ce` warmup fix on main(current)
- Document expected behavior:turn success **bimodal 56-92%**,p99
  consistently better than cap=4
- Recommend bench customers run N=3-5 to characterize their workload's
  mode
- If users see 56% reliably → suggest temporary `--prefill-max-requests 4`
  override

**Don't revert** — TTFT win is solid。Investigation should target the
deterministic failure mode,not the cap value。

## Master strategy update needed

§1.2.1.A weight axis(per `5dc27a2` + `182e084` + planned EOD+58 update)
should now read:
```
Schedule cap: cap=8 + warmup fix per `12300c5` + `c20b1ce`
              TTFT p99 -87%(72515 → ~10000 ms)consistently
              Turn success:bimodal 56-92%(N=2 verification)
              Investigation pending — server/harness state interaction
```

## Methodology lesson

`8281047` was a single-run "validation" that I prematurely synthesized
into `1fce03f` recommendation。Skill v1.3.0 LICENSE rule requires N≥3 σ
check — `fc41e7e` provided N=2 and revealed bimodal behavior single-run
missed。

**Anti-pattern reinforced**:single-run "VALIDATED" claim is never
sufficient for production-flip recommendation。Always require N≥3
runs with σ analysis before LICENSE。

Cost:`1fce03f` synthesis was premature → caveat brief here forced 1
extra cron tick + risk of misleading user。Should have waited for N=3
verification before writing synthesis。

## Cross-references

- `8281047` single-run validation(misleadingly framed as LICENSED)
- `1fce03f` synthesis built on `8281047`(needs caveat = THIS brief)
- `fc41e7e` N=2 deterministic failure
- Pre-fix runs:`b4r8fha82`(56%)+ `bwa4piqqx`(76%)
- Post-fix runs:`8281047`(92%)+ `fc41e7e`(56%)

## Status

**Recommendation**:keep cap=8 + warmup fix on main(no revert),BUT
update expectations:bimodal 56-92% turn success,investigation pending。

**Next concrete action**:codex Step A bench harness instrumentation to
identify deterministic-failure session indices。~0.5d codex,then
Steps B-C as decision tree dictates。

Codex idle since EOD+43。This brief blocks the master strategy
§1.2.1.A update until investigation resolves the bimodal behavior。
