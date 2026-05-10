---
title: Self-correction on da7f5a2 — Task #43 Arm A "server survived" framing dramatically understated 16× TTFT slowdown / 26× throughput loss
date: 2026-05-10
type: research
status: closed (Claude self-correction sediment, da7f5a2 framing supersession)
related_tasks: [#43 (DISPROVEN), #47 (H1' design now has performance-regression A/B gate too)]
related_skills: [#29 (Claude self-application n=5), #36, #38]
---

# Self-correction on da7f5a2 — "server survived" framing was misleading

> **Purpose**: Last commit (`da7f5a2`) claimed Arm A's 70-OOM cascade was
> "saved by SKILL #38" and "server functional". Bench CSV parse reveals
> the server was 16× slower TTFT / 26× lower throughput than Arm B.
> "Survived" hides the actual catastrophic degradation. §0 SOLID rule 1
> violation: "推断 ≠ SOLID" — I claimed survival from log-grep evidence
> without parsing the bench output.

## §1 What da7f5a2 claimed

> "Both arms produced bench output. Server in Arm A actually serviced
> requests successfully despite the 70-OOM cascade. The graceful warmup
> clamp degraded Pass 3 budget without killing the process."

This framing implied Arm A was functional-but-noisy. It was actually
functional-but-catastrophically-degraded.

## §2 What the CSV actually shows

```python
# bench-output/2026-05-10-task43-{A,B}-*/benchmarks.csv row 3 (data row)
```

| Arm | Successful | Errored | TTFT median | server tok/s |
|---|---:|---:|---:|---:|
| A (scratch ON, 70 OOMs in Pass 3) | 71 | 0 | **1501.85 ms** | **1.23** |
| B (scratch OFF, 1 OOM in Pass 3) | 56 | 0 | **93.85 ms** | **32.41** |
| **Δ% (A vs B)** | +27% | 0 | **+1500% (16× slower)** | **−96.2% (26× lower)** |

Arm A produced more "successful" requests in the same window because
each one took so long, but actually delivered far fewer tokens overall.

## §3 Why this matters

### §3.1 da7f5a2's SKILL #38 "load-bearing" claim is OVERSTATED

I wrote: "Without #38 the cascade would have killed the server".
Source check on `warmup.rs:280-310` shows:

- Pass 3 retry-at-half is implemented as `tokens_per_row = next_tokens; continue`
- Total failure (`tokens_per_row <= 1`) triggers `break 'prefill_sizes` — no panic
- Pass 3 abort means "no Pass 3 graphs", NOT "server panic"

So without SKILL #38 retry-at-half:
- Pass 3 would abort early on first OOM
- Server would still serve requests (without Pass 3 graphs)
- Performance would likely STILL be degraded (no graphs at conc=4 4k W4A16)
- BUT — possibly faster than Arm A's 1502ms because no failed-and-retried
  graph capture work

The "load-bearing" framing should be: **graceful degradation, not survival**.

### §3.2 Real load-bearing for Arm A's perf is NOT #38 — it's that the dispatch path eventually settles

The server Arm A served 71 requests at ~526ms prefill chunks (per log). The
question is WHY each request takes so long. Hypotheses:

1. Pass 3 graphs partially captured at small token counts → most prefill
   chunks fall through to non-graph dispatch with high launch overhead
2. The static-scratch buffer is held throughout → less VRAM headroom for
   request-time KV cache → admission backpressure
3. Failed Pass 3 attempts left some state inconsistent → some other slow
   path

Per §0 SOLID rule 1, all are HYPOTHESES not evidence. Cheap experiment
to disambiguate: log breakdown shows
`prefill=525692us decode=208us total=526310us batch=1` — the prefill is
the cost, not graph dispatch.

### §3.3 Task #47 H1' design: now needs PERFORMANCE-regression A/B gate too

Prior `da7f5a2` correction: H1' needs OOM-regression A/B gate. This
correction adds: **also needs TTFT/tok-s regression A/B gate** at conc=4
4k W4A16 sustained. Otherwise H1' could land with "looks fine in
single-request smoke" but actually 16× slower under load.

**Updated H1' acceptance criteria for Task #47**:
1. PF8 TTFT/ITL improvement at conc=1 (the original goal)
2. **No** OOM-cascade regression at conc=4 4k W4A16 (per da7f5a2)
3. **No** TTFT regression > 5% at conc=4 4k W4A16 (this correction)
4. **No** tok-s regression > 5% at conc=4 4k W4A16 (this correction)
5. greedy_consistency 0.0% diff preserved

## §4 SKILL implications

### §4.1 #29 (default broken fixtures / framing) → n=5 evidence

Adding to n=4 evidence:
| n | Source | Pattern |
|---|---|---|
| 4 | `b956f3a` Claude self-app via test/fixture mismatch | applied own |
| **5** | **`da7f5a2` "server survived" framing without CSV parse (this correction)** | **applied own — claimed survival from log-grep without verifying performance via bench output** |

Strengthened wording: "Default test/bench artifact framing (`benchmarks.html`,
`benchmarks.csv`, server logs) requires multi-source cross-check; single
artifact (e.g. log-grep without CSV parse) can produce misleading
'success' framing for catastrophic degradation."

### §4.2 #34 (greedy single-request not sufficient) reinforced

The Arm A outcome is exactly what #34 warns about — single-source
artifact (server log) suggested "running fine"; the actual perf metric
told the real story. **Multi-artifact cross-check** is the rule.

### §4.3 #38 evidence count

**Per §3.1, da7f5a2's #38 n=5 claim is also overstated.** The correct
n=5 reading is "graceful-not-panic" not "load-bearing for survival".

Recounting #38 evidence (corrected):
| n | Source | Verified claim |
|---|---|---|
| 1-4 | (prior, retained) | warmup target shape clamp graceful degradation |
| 5 | Task #43 Arm A | **graceful-not-panic** during 70-OOM cascade (NOT "saved server"; server perf degraded 16×) |

Net: #38 still n=5 evidence, but the framing of n=5 changes from
"load-bearing for survival" to "graceful-not-panic with potentially
catastrophic perf cost".

## §5 Procedural lesson — "trust but verify" applied to my own commits

I committed `da7f5a2` 30 minutes ago with a strong claim ("server
functional", "load-bearing"). At that time I had:
- log evidence (70 OOMs visible)
- log evidence (server continued to admission/prefill lines)
- CSV files existed (`benchmarks.csv`)
- I did NOT parse the CSVs

This is the same anti-pattern as SKILL #34's "greedy single-request not
sufficient" — sufficient log evidence was treated as sufficient global
evidence. Multi-artifact verification (especially numerical metrics)
should precede strong-claim commits.

**Procedural rule for future ticks**: when a research note claims
"server survived" / "test passed" / "fix worked", REQUIRE numerical
metric verification (TTFT/ITL/tok-s) from the bench output before
commit, not just log grep.

## §6 Net for da7f5a2 + this correction

`da7f5a2` is NOT retracted — its core findings stand:
1. ✅ 70:1 OOM ratio between scratch=ON and scratch=OFF (TRUE)
2. ✅ Codex `83fc5d0` INVERSE finding confirmed independently (TRUE)
3. ✅ Task #47 H1' needs OOM-regression A/B gate (TRUE)
4. ⚠ "Server functional" / "load-bearing for survival" — **OVERSTATED**, see this correction

This correction adds:
1. **Arm A is 16× slower TTFT / 26× lower throughput than Arm B**
   (the actual cost of scratch=ON at this workload)
2. **Task #47 H1' also needs TTFT/tok-s regression A/B gate**
3. **SKILL #29 to n=5** with Claude self-application
4. **SKILL #38 framing correction** ("graceful-not-panic" not "saved")
5. **Procedural rule**: parse CSVs before committing strong claims

## §7 Cross-references

- `da7f5a2` original commit (this corrects/strengthens, doesn't retract)
- `83fc5d0` codex Task #43 INVERSE (still confirmed)
- `1ba06f0` Claude original DISPROVEN hypothesis
- `2cc608a` H1' design — now has 2 A/B gates (OOM + perf)
- `warmup.rs:280-310` Pass 3 retry-at-half source (verified)
- `bench-output/2026-05-10-task43-{A,B}-*/benchmarks.csv` (newly parsed)
- SKILL `kernel-optimization` v1.11.0+ #29 (now n=5)
- SKILL `kernel-optimization` v1.13.0+ #38 (n=5 framing corrected)

## §8 Status

**Closed — Claude self-correction sediment.** `da7f5a2` framing strengthened.
H1' acceptance criteria expanded. Procedural rule for future strong-claim
commits added. SKILL #29 to n=5 with this self-application instance.
