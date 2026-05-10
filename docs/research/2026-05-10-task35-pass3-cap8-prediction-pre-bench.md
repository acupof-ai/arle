---
title: Task #35 Pass 3 cap=8 first-burst — Claude formula prediction BEFORE codex bench numbers land (SKILL #34 mantra rule 1 verification setup)
date: 2026-05-10
type: research
status: open (prediction made BEFORE bench result; will be reconciled with codex's wins entry when it lands)
related_tasks: [#35 (cap=8 prefill warmup, codex Working ~36min)]
---

# Task #35 Pass 3 cap=8 first-burst — formula prediction PRE-bench

> **Purpose**: codex is currently running W4 c=8 agent trace cold-start
> bench for Task #35 acceptance. Per SKILL kernel-optimization v1.12.0
> mantra rule 1 ("Predict with formula, not vibes"), Claude commits a
> prediction NOW (before numbers come back) so reality vs prediction
> can be reconciled in the wins entry. This is the SOLID
> license-or-kill discipline applied to a substrate change.

## §1 What Pass 3 does

Per `M_warmup-prefill-pass-directive.md` + codex Task #35 implementation:

- **Pass 1**: model load + weights pinning (always)
- **Pass 2**: graph-capture mode only — re-captures graphs with autotuned algorithms
- **Pass 3** (NEW): cap=8 prefill batch pre-warm at server startup
  - default-on with `INFER_PREFILL_WARMUP=0` opt-out (per `60f114f`
    matched-control discipline doc)
  - Eager warmup of cap=8 prefill kernels so first cap=8 burst doesn't
    hit cold codegen / capture / JIT paths
  - Observed startup cost: +282.7ms (per codex's sustained-load A/B
    summary this session)
  - Pass 3 body itself: 308ms stable

## §2 Baseline (prior cap=8 bench numbers)

Per `docs/experience/wins/2026-05-08-ttft-p99-cap8-fix-86pct-reduction.md`
(Task #31 closure, `bench_agent_trace.py agent-w4-tool-resume`):

| Metric | cap=4 (pre-#31) | cap=8 (#31 default-on) | Δ |
|---|---|---|---|
| TTFT p50 | 11768 ms | 5868 ms | −50% |
| TTFT p99 | 72515 ms | 10259 ms | −86% |
| p99/p50 spread | 6.2× | 1.75× | −72% |

The cap=8 numbers above are the **without-Pass-3 baseline** for Task
#35. Codex's Task #35 "off" arm should reproduce ~5868 / 10259 ms
(within σ).

## §3 Formula prediction for Pass 3 cap=8 first-burst

### §3.1 Mechanism

First cap=8 burst (request set 1, fresh server, cold kernels):

```
TTFT_first_burst = T_kernel_cold + T_graph_capture + T_prefill_compute
                 + T_decode_first_token

Without Pass 3:
- T_kernel_cold = full JIT/codegen on first cap=8 prefill = ~hundreds of ms
- T_graph_capture = cuda graph capture for cap=8 shape = ~tens of ms
- T_prefill_compute = warm-path baseline (~hundreds of ms for 8×512=4096 tokens)
- T_decode_first_token = warm-path baseline (~10 ms)

With Pass 3:
- T_kernel_cold = ~0 (paid at server startup via Pass 3 +282.7ms)
- T_graph_capture = ~0 (already captured during Pass 3)
- T_prefill_compute = warm-path baseline (unchanged)
- T_decode_first_token = warm-path baseline (unchanged)
```

### §3.2 Magnitudes

The current cap=8 baseline (5868 ms p50 / 10259 ms p99) IS the
"steady-state cap=8" — these numbers are aggregated over many bursts
in the agent trace, including warm ones. The cold first-burst is
amortized.

If Pass 3 eliminates the cold first-burst overhead:
- p50 over the trace: should change LITTLE (most bursts are warm
  already in steady-state)
- p99 (the first burst tail): could drop more visibly since first
  burst's tail is the worst case
- p99/p50 spread: should tighten further

### §3.3 Predicted Δ ranges

| Metric | Predicted Δ vs `off` arm | Reasoning |
|---|---|---|
| TTFT p50 | **−2% to +2%** (essentially noise) | Most agent-trace bursts are warm; first-burst small fraction |
| TTFT p99 | **−5% to −20%** | First-burst tail is what Pass 3 eliminates |
| TTFT p99/p50 spread | **−5% to −15%** | Tight tail = better spread |
| First-burst TTFT (if measured separately) | **−30% to −60%** | This is what Pass 3 actually optimizes |
| Steady-state ITL p50 | **0% (unchanged)** | Pass 3 doesn't touch decode |
| Server startup time | **+282.7ms** (observed, paid once) | Already measured |

### §3.4 License threshold (Claude's pre-bench gate)

Per SKILL v1.12.0 Phase 8 license-or-kill:

| Outcome | Verdict |
|---|---|
| TTFT p99 Δ > −10% AND first-burst Δ > −30% AND ITL Δ ≈ 0 | LICENSE Pass 3 default-on |
| TTFT p99 Δ < −5% (regression in tail) OR first-burst Δ > −10% | KILL Pass 3 (effect too small to justify +282.7ms startup) |
| Between | REVIEW (might want to flip default-on/off based on workload) |

## §4 What could falsify this prediction

Per skill #1 "推断 ≠ SOLID — 推断 = hypothesis, evidence = bench numbers":

1. **bench_agent_trace.py shape ≠ first-burst-heavy**: if the agent
   trace replays many warm bursts and only 1-2 cold first-bursts, the
   first-burst Δ would be diluted in aggregates. **Check codex's
   wins entry for first-N-burst breakdown.**
2. **Pass 3 doesn't actually warm the right kernels**: implementation
   bug — Pass 3 might warm shape A but production hits shape B.
   Symptom: no cold-vs-warm difference in numbers. **Check codex
   wins for whether off/on first-burst differs at all.**
3. **Cold first-burst on this shape is already small**: maybe the
   agent trace's first burst doesn't trigger expensive JIT (e.g. TileLang
   AOT cache hit). Symptom: Δ ≈ 0 across all metrics.
4. **Variance dominates**: σ across N=3 runs > Δ → not a real win.

## §5 Reconciliation plan

When codex's wins entry lands:

1. Read the actual TTFT p50 / p99 / first-burst Δ
2. Compare to §3.3 predicted ranges
3. If WITHIN range → SKILL #1 mantra holds, formula was right
4. If OUTSIDE range:
   - Above range (over-delivered) → revise formula upward, learn
     that JIT cost was bigger than estimated
   - Below range (under-delivered) → check §4 falsification list
5. Either way, document the actual Δ in this doc's §6 (added by next
   tick)

## §6 Actual results — reconciled vs codex wins entry

Source: `docs/experience/wins/2026-05-10-bench-35-cap8-prefill-warmup.md`
(codex-authored, untracked at this moment, content read this tick).

### §6.1 Sustained-load arms (codex measured n=3, σ < 5% all)

| Metric | Predicted Δ | Actual Δ | Within range? |
|---|---|---|---|
| TTFT p50 conc=1 | -2% to +2% | +0.6% (66.0 → 66.4 ms) | ✅ within |
| TTFT p50 conc=2 | -2% to +2% | +2.3% (79.0 → 80.8 ms) | ⚠ marginal (above) |
| TTFT p50 conc=4 | -2% to +2% | +0.3% (157.2 → 157.7 ms) | ✅ within |
| ITL p50 conc=1 | 0% | 0.0% (5.80 → 5.80 ms) | ✅ exact |
| ITL p50 conc=2 | 0% | 0.0% (7.44 → 7.44 ms) | ✅ exact |
| ITL p50 conc=4 | 0% | -3.4% (8.31 → 8.03 ms) | ✅ within (slight improvement) |
| Out tok/s conc=1 | ~0% | -0.1% (159.76 → 159.53) | ✅ noise |
| Out tok/s conc=4 | ~0% | +1.6% (423.82 → 430.58) | ✅ noise |
| Server startup overhead | +282.7 ms (observed) | **+282.7 ms exact** (1077.7 → 1360.3 mean) | ✅ exact match |
| Pass 3 body time | 308 ms (observed) | **308 ms** stable n=3 | ✅ exact match |

**Sustained-load verdict: all predictions held.** The +2.3% conc=2 TTFT
is marginally above the predicted ±2% band but within statistical noise
(CV = 3.85% baseline / 3.36% treatment, so the difference is < 1×σ).

### §6.2 First-burst arms — NOT measured

Per codex's wins entry §Problems:
- First W4 agent-trace attempt used `--model default` → 404 for every
  request (model name issue, codex caught + corrected)
- Corrected attempt was long-running rather than short smoke; codex
  stopped after `requests=127, active=8, kv_util=100%` to free GPU
- **Did not count as license data**

So the metric Pass 3 actually optimizes (first-burst TTFT, predicted
-30% to -60%) is **STILL UNMEASURED**. TTFT p99 (predicted -5% to -20%)
also not in the wins entry — sustained-load reports p50 only.

Codex's own §Rule confirms: "Startup warmup changes need two gates: a
short sustained-load regression smoke for conc 1/2/4, and a separate
full first-burst workload for the workload that originally exposed the
bimodal failure. Do not substitute one for the other." — **identical
to my §4.1 falsification concern.**

### §6.3 License-or-kill verdict per §3.4 threshold

| Gate | Status |
|---|---|
| TTFT p99 Δ > -10% | ❓ not measured |
| First-burst Δ > -30% | ❓ not measured (the actual gate metric) |
| ITL Δ ≈ 0 | ✅ confirmed (0% on conc=1/2, slight improvement conc=4) |

→ **REVIEW** (verdict deferred until first-burst metric measured).

Pass 3 is sustained-load-safe (regression-free) but the cap=8
first-burst optimization claim is unproven. Codex correctly refuses
to claim Task #35 acceptance based on sustained-load smoke alone.

### §6.4 Formula validity assessment

Per SKILL kernel-optimization v1.12.0 mantra rule 1 ("predict with
formula, not vibes"):

- **Sustained-load mechanism prediction**: ✅ formula correct.
  Pass 3 is per-server-startup; once warm, steady-state unaffected.
  All 3 concurrencies × 3 metrics (TTFT/ITL/tok-s) within or marginal
  to predicted ranges.
- **First-burst mechanism prediction**: ⏳ unverified. Falsification
  reason §4.4 ("variance dominates") doesn't apply since we have no
  measurement; falsification reason §4.1 ("bench shape ≠
  first-burst-heavy") explains why codex's chosen smoke shape (5-second
  warmup + 30-second sustained) doesn't expose the cold first burst.
- **Startup cost prediction**: ✅ exact match (codex's measured
  +282.7ms identical to the observation Claude based the prediction
  on).

**Net**: formula PARTIALLY validated. Sustained-load predictions
exact; first-burst predictions await the dedicated agent-trace bench
codex stopped for time.

### §6.5 Next steps

- Task #35 wins entry NOT closing the cap=8 first-burst gate by
  itself. Either:
  1. User runs the long-form `bench_agent_trace.py
     agent-w4-tool-resume` (the Task #31 8k W4 cap=8 burst shape,
     30+min) to measure first-burst TTFT
  2. Accept Pass 3 as "sustained-load-safe + +282.7ms startup, no
     first-burst measurement yet" — license on regression-guard alone,
     defer first-burst proof

### §6.6 CRITICAL CAVEAT — bench numbers came from BUGGY Pass 3

Per codex's tmux output 2026-05-10 ~46min, codex review caught **3
substantial bugs** in Task #35 diff (not 1 as 5f3f58f initially
claimed):

1. **sync() called twice** (mentioned earlier, codex review tick)
2. **Pass 3 needed to derive per-row token count from
   `chunked_prefill_size` / token budget** (was wrong, fix in flight)
3. **Pass 3 was warming graphs to a TEMPORARY context that gets
   dropped** — graph prefill resources were being warmed then thrown
   away. **This means Pass 3 was effectively a no-op for the
   graph-prefill case.**

Implication for reconciliation: the bench numbers above (§6.1) measured
"BUGGY Pass 3 (effectively no-op for graph case)" vs "Pass 3 disabled".
Both arms had ≈0 functional Pass 3, hence no improvement observed.

The post-fix bench (when codex commits + re-bench) should show:
- Sustained-load: still ~no improvement (mechanism prediction unchanged
  — Pass 3 doesn't help c=1/2/4 even when working)
- First-burst: NOW the actual prediction range (-30% to -60%) becomes
  testable. Pre-fix bench couldn't have distinguished prediction-true
  from no-op, since Pass 3 was no-op either way.

This **does NOT invalidate** the §6.1 sustained-load reconciliation
(predictions held within ±2.5% noise) but **DOES change the
interpretation**: it doesn't prove "Pass 3 sustained-load-safe" because
Pass 3 was effectively absent on the graph path. It proves "no-op Pass
3 sustained-load-safe" — a weaker claim.

**Re-bench needed after codex's fix lands** to validate Pass 3
sustained-load claim with functional Pass 3.

### §6.7 Updated SKILL #33 evidence count

Task #35 codex review caught **3 real bugs** (sync, chunked_prefill_size,
temporary-context), not 1. That brings the n=2 evidence to:

| Session | Diff | Formal gates PASS | Codex review caught |
|---|---|---|---|
| `ace3cbe` PF8.3 | 12 files | build+clippy+greedy+e2e | 3 real bugs |
| Task #35 (codex pending) | 6 files | build+clippy+greedy+sustained-load | **3 real bugs** |

Both diffs: substrate, all formal gates PASS, **codex review caught 3
bugs each**. Pattern is consistent magnitude, not just consistent
existence. Strong argument that codex review is the highest-yield
verification step for non-trivial diffs.

### §6.12 W4 startup cost MEASURED with new methodology — 8.2s total

Per codex tmux ~1h 02min, log line captured:

```text
CUDA Graph warmup done in 8242ms (decode=8 batch sizes, prefill=7 batch sizes, max decode 8)
```

**W4 actual Pass 3 startup cost: 8.2 seconds** (decode=8 batch sizes
+ prefill=7 batch sizes).

Reconciliation table (UPDATED):

| Metric | Predicted | Pre-fix BUGGY (no-op) | BF16 B=4 only (~56min tmux) | **W4 REAL (8242ms)** |
|---|---|---|---|---|
| Server startup overhead | +282.7ms | +282.7ms | ~60,000ms B=4 alone | **+8242ms total (8 decode + 7 prefill)** |
| Ratio vs prediction | 1× | 1× | ~212× | **~29×** |

**Revised understanding**:
- Prediction was ~29× too low (anchored to buggy substrate's no-op cost)
- BF16 60s number was likely a single-shape outlier or test-window
  artifact, not the production cost
- Real W4 production Pass 3 cost is **8.2 seconds** = significant but
  not catastrophic
- Server boot now adds ~8s for production-grade Pass 3 vs ~ms for
  empty Pass 1+2

**Updated license verdict (§3.4 threshold revision):**

| Gate | Status |
|---|---|
| ITL Δ ≈ 0 | ✅ confirmed (§6.1) |
| TTFT p99 Δ > -10% | ❓ first-burst still unmeasured per §6.2 |
| Server startup acceptable | ✅ 8.2s is well within "acceptable for production deployments that boot once and serve thousands of requests" |
| Dev-iteration friction | ⚠ 8.2s adds friction but `INFER_PREFILL_WARMUP=0` escape hatch covers it |

→ **REVIEW** still (first-burst measurement still pending), but the
startup cost no longer flips the "default-on prod / default-off dev"
recommendation toward "always opt-in" — 8.2s is tolerable, just costly
enough to want the escape hatch for tight test loops.

**SOLID lesson reinforced**: my §6.9 alarm at "200-1000× larger than
predicted" was based on extrapolating from BF16 60s. Single
extrapolation point ≠ ground truth. **Codex's discipline to measure
W4 directly per the actual production target shape was correct**;
~29× over prediction is bad enough to call out but not catastrophic.

### §6.11 Codex applying SKILL #29 to its own prior data (pre-fix logs contaminated)

Per codex tmux ~1h 01min:

> 旧 on 日志是 review 修复前的 64-token warmup，不能再作为当前实现的数据。
> 我改用日志里的 CUDA Graph warmup done 作为 readiness 标志，重新采 W4
> 当前实现的 Pass 3 startup cost。
>
> [translation]: Old on-arm logs are pre-review-fix 64-token warmup
> data, can no longer count as current implementation data. I'll
> switch to using the "CUDA Graph warmup done" log line as readiness
> marker, re-collect W4 current implementation Pass 3 startup cost.

This is **codex independently applying SKILL #29** (default broken
fixtures may be known-broken) to its OWN prior bench data — the
recognition that data collected before the 4 review fixes (sync +
chunked_prefill_size + temp-context + max_seq_len) is from a
fundamentally different (BUGGY) substrate and must be discarded /
re-collected.

This validates SKILL #29 as a generalizable pattern beyond test
fixtures:
- Original #29 (codex caught W4A8 default greedy_consistency PASS
  vs PF8 INACTIVE) — test fixture broken default
- Today's case (codex catching its own pre-fix bench data is no
  longer current) — substrate-state broken default

Both are "data may be from a substrate snapshot you don't intend to
generalize from". The pattern: **after substantial code change
(refactor / fix / dispatch path adjustment), prior bench data may
not transfer to current substrate even if methodology was identical**.

**Skill candidate v1.13.0+ #39 (single evidence, not codifying yet)**:
"After applying review fixes / dispatch refactor / non-trivial code
change, ALL prior bench numbers from that file/path are stale and
must be re-collected. Do NOT mix pre-fix + post-fix data in same A/B.
The substrate identity changed."

Watch-list: when Task #47 H1' refactor lands, any prior PF8 chain
bench numbers (v3-v10 attempts) become STALE — must re-collect at
the new substrate. Codex is already applying this discipline here in
real-time.

### §6.10 W4 startup A/B attempt — server not ready in 5-10s (W4 likely > 60s BF16)

Per codex tmux ~57min: codex started 6-arm W4 startup A/B (3 off-arm
runs + 3 on-arm runs) but first arm not ready within expected 5-10s
window. Codex's discipline statement: "如果 120s 超时，会直接保留已有
sustained-load 和 targeted greedy 证据，不把这个失败尝试写成成功数据".

Translation: if 120s timeout reached, keep existing evidence + NOT
claim Pass 3 W4 numbers in the wins entry — exactly the SKILL #34
multi-shape discipline (refuse to substitute partial data for full
validation).

**Implication for §6.9 startup cost**: BF16 B=4 = 60s already. W4
startup is even slower (model weights require quantization unpack +
graph capture for W4 paths). If W4 Pass 3 takes > 120s per arm,
the practical Pass 3 cost on production W4 deployments may be
multiple minutes per server boot — large enough that "default-OFF
for dev / default-ON for production" decision flips toward "always
opt-in, never default-on".

**Codex demonstrating SKILL #34 in real-time**: refusing to claim
data that doesn't exist. This is exactly the discipline #34 codified
("greedy single-PASS NECESSARY but NOT SUFFICIENT") applied to an
adjacent decision (don't backfill missing data with optimistic
assumptions). Reinforces the rule.

### §6.9 DRAMATIC startup cost surprise — real Pass 3 is ~200-1000× more expensive than predicted

Per codex tmux ~56min: targeted greedy PASSED post-fix, codex measured REAL
post-review Pass 3 behavior:

- **BF16 test window B=4 warmup: ≈ 60 SECONDS** (60,000 ms)
- BF16 test window B=3 warmup: ≈ 36 seconds
- Codex's stated plan: update wins entry replacing old 64-token startup
  cost (308ms) with CURRENT real data, plus a "quick W4 startup A/B as
  pre-commit bench evidence"

**Reconciliation against my §3.3 prediction (940c7cc):**

| Metric | Predicted | Reported pre-fix (BUGGY) | Reported post-fix (REAL) |
|---|---|---|---|
| Server startup cost | +282.7ms | +282.7ms (matched exactly, but was BUGGY no-op Pass 3) | **+~60,000ms BF16 B=4 alone** (≥200× larger) |

The pre-fix bench number `+282.7ms` was the cost of the BUGGY Pass 3
that warmed graphs to a temporary context immediately discarded — i.e.,
essentially a no-op. The 308ms was just function-call + minor setup.

Post-fix REAL Pass 3 actually warms each (B=1..=8) shape on the production
context. BF16 B=4 alone takes ~60s. If all 8 batch sizes scale similarly,
total startup overhead could be 5-10+ minutes for fully warm Pass 3.

**This is a NEW class of falsification per §4** that I didn't enumerate:
- Not §4.1 (shape mismatch — codex IS warming the right shapes)
- Not §4.2 (Pass 3 not warming right kernels — it IS, just SLOW)
- Not §4.3 (cold first-burst already small)
- Not §4.4 (variance — this is signal not noise, ~200× over prediction)
- **NEW falsification mode**: prediction underestimated absolute warmup
  cost by orders of magnitude. Formula assumed "Pass 3 = pre-warm
  kernels" without modeling that pre-warming requires running ACTUAL
  PRODUCTION INFERENCES at each shape, each of which takes seconds.

**Implication for license-or-kill (§3.4 threshold revision):**

Original §3.4 thresholds assumed +282.7ms startup. With +60,000ms
(or more), the cost-benefit calculation changes:

- **First-burst improvement gate**: my -30% to -60% prediction was for
  a relatively quick warmup (282ms). Now first-burst saves 60s+ per
  batch-size-not-yet-seen. Per-burst saving could be MUCH larger
  absolute number (saves 60s on a request that would otherwise take
  60s+ to first-burst).
- **Amortization**: with +5-10 min server startup, Pass 3 only pays
  off if server runs N>>1 requests AND first-burst latency matters
  (UX constraint). Production servers handle thousands of requests
  → amortizes over thousands of requests easily.
- **Test/dev workflows**: server startup cost +5-10 min is significant
  — codex's `INFER_PREFILL_WARMUP=0` escape hatch becomes load-bearing
  for dev iteration speed.

**Updated license verdict (pending codex's W4 A/B numbers):**
- LICENSE if: first-burst improvement + amortizable cost over realistic
  production traffic
- KILL if: first-burst improvement minor AND startup cost prohibitive
- REVIEW: most likely outcome — Pass 3 should be opt-in default-OFF for
  dev (escape hatch reverse), default-ON for production deployments

**SOLID lesson for future prediction-reconciliation cycles**: when
predicting startup cost, account for whether the prediction-anchor
data was from a FUNCTIONAL or BUGGY substrate. The ~200× error here
came from anchoring to a no-op measurement (BUGGY Pass 3 reported
308ms, prediction inherited that anchor). Per skill v1.12.0 #29
(default broken fixtures), the "prior measurement may be on broken
substrate" pattern applies to prediction inputs too, not just bench
outputs.

### §6.8 Re-verification layer caught 4th issue (post-review)

Per codex tmux ~50min: while running targeted greedy_consistency to
validate the post-review fixes, codex flagged a 4th issue:

> 测试配置 max_seq_len=512 但 chunked_prefill_size=4096
> [test config max_seq_len=512 but chunked_prefill_size=4096]
> ...
> 把 warmup token cap 再夹到 effective_max_seq_len，避免测试/小窗口
> 启动时 warming 不可能进入的形状
> [cap warmup token to effective_max_seq_len, avoid warming shapes
> the test/small window can't reach at startup]

Codex applied the cap, rebuilt + clippy PASS in 7.02s (~52min).

This is the **4th substantial issue** caught in Task #35 cycle:
- Issue #1-#3: codex review (sync, chunked_prefill_size, temp-context)
- Issue #4: codex re-verification (max_seq_len cap on warmup budget)

**Layered discipline pattern emerges**: codex review catches design
bugs that formal gates miss; codex re-verification (re-running tests
on the post-review code) catches integration bugs that emerge from
fix interaction. Both have value. Both are load-bearing.

**Generalization (single-evidence skill candidate #38, not codifying
yet per accumulation policy)**: "warmup target shape budget should be
clamped to effective workload shape budget; warming unreachable shapes
is dead work that may also expose perf/correctness anomalies in
test-config small-window setups."

Watch-list: when codex implements Task #47 H1' refactor, check whether
PF8Scratch sizing is similarly clamped (max_m, max_n, max_k vs
effective workload max_seq_len etc). If H1' design needs the same
clamp guard → n=2 evidence for #38.

## §7 SKILL #33 reinforcement candidate (n=2 evidence)

Originally codified in `0be7220` v1.12.0 from PF8.3 substrate session
(`ace3cbe` codex review caught 3 bugs that build/clippy/greedy/e2e all
PASSED). Today's Task #35 reinforces:

| Session | Diff | Formal gates | codex review caught |
|---|---|---|---|
| 2026-05-10 PF8.3 (`ace3cbe`) | 12 files, +3936/-13 LOC | build+clippy+greedy+e2e PASS | 3 real bugs (parallel-M loop, max_par/lock workspace, graph-capture interaction) |
| 2026-05-10 Task #35 (codex pending commit) | 6 files, ~?LOC | build+clippy+greedy+sustained-load PASS | sync() double-call (1 real bug) |

Both diffs were "non-trivial substrate" by SKILL #33's criteria
(≥3 files / FFI boundaries / cross-feature interactions). Both had
all formal gates PASS. Both had codex review catch real bugs that
would have slipped to production.

This is **n=2 evidence** for SKILL #33 — sufficient to upgrade from
"recently codified anti-pattern" to "battle-tested canonical
practice". No revision needed; the rule is reinforced, not refined.

## §7 Cross-references

- `M_warmup-prefill-pass-directive.md` (the directive Task #35
  implements)
- `2026-05-08-ttft-p99-cap8-fix-86pct-reduction.md` (Task #31 cap=8
  baseline, the "off" arm reference)
- `60f114f` matched-control escape-hatch discipline (Task #35 design)
- `0be7220` SKILL kernel-optimization v1.12.0 mantra rule 1 +
  Phase 8 license-or-kill
- `infer/tests/greedy_consistency.rs:test_greedy_solo_vs_concurrent`
  (codex's correctness gate, PASSED)
- This doc: `2026-05-10-task35-pass3-cap8-prediction-pre-bench.md`

## §8 Status

**Prediction committed BEFORE codex bench result lands.** This is
the verification setup — formula vs reality. Outcome reconciled in
§6 next tick.
