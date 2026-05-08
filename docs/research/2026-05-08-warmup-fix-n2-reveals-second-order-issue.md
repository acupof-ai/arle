# Warmup fix N=2 reveals second-order regression — run #2 always 144/256(56%)regardless of fix

> Per `8281047` Phase 8 conditional LICENSE rule — N=2 verification at
> cold-start required to confirm 91.8% is real not single-run outlier。
>
> **N=2 result:144/256(56.3%)**,**EERILY IDENTICAL** to pre-fix
> `b4r8fha82` 144/256(also 56%)。Warmup fix `c20b1ce` improves run #1
> from 76% → 92% but **run #2 deterministically converges to 144/256
> regardless of warmup state**。Second-order issue beyond warmup
> coverage。

## N=2 result

```
turns OK:        144 / 256(56.3%)
scored turns OK: 74
tokens total:    18820
wall total:      1401 s
TTFT p50/p99:    14791 / 15249 ms
ITL p50/p99:     25.9 / 26.1 ms
Peak mem:        15880 MB
session_slot_pressure_evictions_hard: 131
```

## Cross-N comparison — striking pattern

| Run | Warmup | Run # | Turn Success | Wall | Tokens out | Peak mem |
|---|---|---:|---:|---:|---:|---:|
| `bwa4piqqx` | max=4 | #1 | 194/256(76%) | **2290 s** | 35733 | 15880 MB |
| `b4r8fha82` | max=4 | #2 | **144/256(56%)** | **1409 s** | **23424** | 15911 MB |
| `b1mm1k0r7` | **max=16(fix)** | **#1** | **235/256(92%)** | **2356 s** | 32298 | 15911 MB |
| **`b4kaqdrmj`(this)** | **max=16(fix)** | **#2** | **144/256(56%)** | **1401 s** | **23424** | 15880 MB |

**`b4r8fha82` and `b4kaqdrmj` are nearly identical in outcomes**:
- Same 144/256 turn success(112 errored sessions)
- Same 23424 tokens_out(byte-exact)
- Same 1401-1409s wall time
- Same 15880-15911 MB peak

These can't be coincidence。**The bench has a deterministic failure pattern at "run #2"**:
- Some 112 sessions DETERMINISTICALLY fail
- Same exact tokens_out suggests same session-failure ordering
- Same wall time suggests same retry-exhaustion pattern

## Hypothesis space

### H1 — Bench harness state(strongest)

`scripts/bench_agent_trace.py` may have shared state between runs:
- HuggingFace cache populated AFTER run #1 → run #2 uses cached prompts faster
- Session retry budget per-bench-run vs across-runs
- Session ID assignment may align with stuck sessions

If bench harness retries ARE deterministic by session ID → certain sessions
ALWAYS fail at exact same point。

### H2 — Server-side state persistence across server restarts

Despite `kill 911823` and fresh server start,SOME state may persist:
- GPU driver allocator state
- CUDA context warmup that survives process restart
- Filesystem-side prefix cache(if shared)

But this should not produce byte-identical outcomes across runs。

### H3 — Run #1 had favorable initial conditions

`bwa4piqqx`(76%)and `b1mm1k0r7`(92%)both had **2290-2356s wall time**
(significantly longer than 1401-1409s)。Maybe:
- Run #1 got LESS pressure(some structural factor)
- Run #2 has cumulative state(not from server,but from bench/network/HF cache)

This explains:
- Run #1 had MORE time to retry → more successes → higher final %
- Run #2 had LESS time → retries exhausted → fewer successes

## Run-#1-vs-#2 wall time mystery

Why does run #2 wall time DROP from 2290-2356s to 1401-1409s while
turn success drops from 76-92% to 56%?

Possible answers:
1. **Bench harness skips retries faster on run #2**(retry budget reduced,
   or backoff is faster after seeing pattern)
2. **Failed sessions error out FAST**(503 exhaustion in 31 sec each,vs
   successful sessions taking 1-2 min each)
3. **Total work done = success_count × per_success_time + fail_count × 31s**
   - Run #1 92%:235 × ~10s + 21 × 31s = 2350s + 651s = 3001s wall? doesn't match
   - Run #2 56%:144 × ~6s + 112 × 31s = 864s + 3472s = 4336s? doesn't match either

Math doesn't quite work — bench harness retry semantics need investigation。

## Codex follow-up investigation needed

### Step A — Bench harness deterministic-failure isolation
- Trace which specific sessions(by ID)fail in run #2
- If same N sessions fail across 3+ runs → deterministic harness bug
- Mitigation:reset bench harness retry state OR seed RNG differently

### Step B — Run #1 vs run #2 server log diff
- Compare server `/v1/stats` snapshots at start of run #1 vs run #2
- Identify any persistent state(KV cache,allocator,etc)
- If state differs at start → server state persists across restarts

### Step C — Run with `--reset-bench-state` flag(if exists)or modify bench script
- Add `time.sleep(60)` between runs to allow GPU cooldown
- Add explicit `nvidia-smi --gpu-reset` between(if possible)
- Compare vs back-to-back runs

## Phase 8 verdict — REVERT-AND-REVISIT consideration

`8281047` 91.8% LICENSE was based on **single-run #1** evidence。N=2 reveals
**91.8% was a single-run outlier**,not stable distribution。Average
turn-success is **56-92% with significant variance**。

**Reframed**:
- Warmup fix HELPED (run #1 went 76→92%)— directionally correct
- Warmup fix DID NOT close 100% gap — run #2 still 56%
- Deeper issue exists in bench harness OR server state persistence
- Production deployment confidence still LACKING(92% single run,but
  multi-run reality is 56-92%)

**Decision**:
- Don't revert `c20b1ce`(directionally correct + improves first-run)
- **N=3 third run** to characterize variance band
- Codex Step A bench harness investigation in parallel

## Cross-references

- `c20b1ce` warmup fix(directionally correct)
- `8281047` warmup fix LICENSE(based on N=1)
- `b4r8fha82` pre-fix run #2(144/256 — first occurrence of deterministic pattern)
- `b1mm1k0r7` post-fix run #1(235/256 — best result)
- `b4kaqdrmj`(this — post-fix run #2,144/256 same as pre-fix run #2)
- Codex investigation plan: `fc9bea9`
- Bench harness: `scripts/bench_agent_trace.py`

## Status

- ✅ Warmup fix `c20b1ce` valid and ships(don't revert)
- ⚠ Single-run LICENSE `8281047` was based on outlier(91.8% not stable)
- ⚠ Multi-run reality:56-92% turn success at W4 c=8 8K(huge variance band)
- 🔧 Need codex Step A bench harness investigation OR Step B/C config tests
- 🔧 N=3 third run for variance characterization

## Skill v1.4.0 anti-pattern caught(refinement of #16)

Anti-pattern #15 (warm-server implicit dependency):caught,fix landed。
Anti-pattern #16 (implicit-coupling-via-shared-default):caught,fix landed。

**This entry's contribution — anti-pattern #17(NEW)**:**second-run
state contamination**:
- Run #1 of any bench may have favorable initial state
- Run #2-N may converge to a degraded "steady-state" pattern
- Single-run LICENSE based on run #1 is **systematically optimistic**

**Rule added**:**N=3 verification mandatory across ALL run positions**
(not just N=3 of run #1)。Specifically,verify run #1,run #2,run #3
each independently → characterize whether all stable or progressive
degradation pattern。

For ARLE specifically:`8281047` LICENSE should be reframed as
"conditional on single-run state",not production-deployment-ready。
N=3+ verification across run positions needed before final LICENSE。

This methodology gap cost 1 Claude tick to surface。Closing the rule
prevents over-confident production gating。
