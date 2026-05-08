# cap=8 N=3 — six-run distribution reveals bimodal failure mode,not strictly run-position

> Per `fc41e7e` skill rule "N=3 verification mandatory across run positions"。
> Run #3 result:194/256(76%)= same as `bwa4piqqx` original。
>
> **Six-run dataset reveals BIMODAL distribution**:not strictly tied to
> run position(#1 vs #2 vs #3),but to specific resource-state factor。
> 4/6 runs land at 76-92% turn success;2/6 lock at deterministic
> 144/256(56%)。Production confidence still LACKING for 95% threshold。

## Six-run cumulative dataset

| Tick # | Run label | Cap | Server | Warmup | Turn Success | Wall | Tokens out |
|---|---|---|---|---|---:|---:|---:|
| 1 | `bwa4piqqx` | 8 default | fresh build #1 | max=4 | 194/256(76%) | 2290s | **35733** |
| 2 | `b4r8fha82` | 8 default | fresh restart | max=4 | **144/256(56%)** | **1409s** | **23424** |
| 3 | `ba00s5nu3` | 8 override | fresh restart | max=4 | 201/256(78.5%) | 2441s | 32169 |
| 4 | `b1mm1k0r7` | 8 default | fresh build #2(post-fix)| max=16 | **235/256(92%)** | 2356s | 32298 |
| 5 | `b4kaqdrmj` | 8 default | fresh restart | max=16 | **144/256(56%)** | **1401s** | **23424** |
| 6 | `b5i3467ad`(this) | 8 default | fresh restart | max=16 | 194/256(76%) | 2291s | 30826 |

## Bimodal pattern — degraded vs normal mode

**Degraded mode**(2/6 runs):
- Turn success:144/256(56%)
- Wall:**1401-1409s**(short,because failures exit fast)
- Tokens_out:**23424**(BYTE-IDENTICAL across both occurrences)
- Distinct fingerprint suggests deterministic pathology trigger

**Normal mode**(4/6 runs):
- Turn success:194-235/256(76-92%)
- Wall:2290-2441s(longer,sessions retry succeed)
- Tokens_out:30826-35733(varies)

## Pattern interpretation

### What ISN'T determining mode

- **Run position**:#1 of session was 76% AND 92%(both modes within #1's)
- **Cap source**:override and default both produce both modes
- **Build state**:pre-fix vs post-fix both produce both modes(though post-fix run #1 was the BEST 92%)

### What IS likely determining mode

- **Resource state at server start time**:GPU driver scheduler,allocator slop,
  CUDA stream queue from prior process exits
- **Bench harness state**:HF dataset cache(after first run completes),
  retry budget per session
- **Network timing**:HTTP retries can race differently across runs

The byte-identical 23424 tokens in degraded mode suggests **deterministic
session-failure ordering when degraded mode triggers** — same N sessions
fail the same way → same total tokens emitted。

## Probability density

Of 6 cap=8 8K runs collected:
- **Normal mode(76-92%)**:4/6 = **67%**
- **Degraded mode(56%)**:2/6 = **33%**

Production deployment expectations:
- Expected turn success rate:**(0.67 × ~85%) + (0.33 × 56%) = 75%**
- 95% threshold not met by this distribution
- A workload that requires turn success ≥ 95% would deploy at 67% probability,
  fail 33% of the time

## Phase 8 verdict — REFRAMED

`8281047` 91.8% LICENSE was based on **single normal-mode run**(`b1mm1k0r7`)。
N=3 verification reveals:
- Warmup fix REAL improvement(run #1 76%→92%,real win for "first burst")
- But MORE THAN 1/3 of subsequent runs hit degraded mode 56%

**Production deployment confidence**:
- For TTFT-tail-bound workloads(p99 9-15s vs cap=4 baseline 72s):**LICENSED**
- For turn-success-bound workloads(95%+ required):**NOT LICENSED**
- Mixed:document the bimodal distribution as known production characteristic

## Codex follow-up — narrowed investigation

Per `fc9bea9` Step 2/3 + this entry's bimodal evidence:

### Step 2.A — Trigger isolation

If degraded mode triggers based on resource state:run with explicit:
- `nvidia-smi --gpu-reset` between runs(if possible)
- 60-sec sleep between server kill and bench start
- Test if degraded mode probability changes

### Step 2.B — Bench harness retry isolation

Add to `bench_agent_trace.py`:
- Log per-session retry count
- Identify if degraded-mode 144/256 has SAME 112 sessions failing same way
- If yes → bench harness has deterministic retry pattern

### Step 2.C — Server-side admission tracing

Server log added during degraded-mode bench:
- Trace per-session admission/eviction decisions
- Identify the exact session ID that triggers eviction cascade

## Skill v1.4.0 anti-pattern refinement

Anti-pattern #17(second-run state contamination)from `fc41e7e` was
PARTIALLY correct:
- ✅ Run #1 vs run #2 patterns differ
- ❌ Pattern isn't strictly run-position based
- ✅ Some hidden resource state affects success rate

**Refined to anti-pattern #17b — bimodal failure distribution**:
- Some workloads exhibit bimodal turn-success(degraded vs normal)
- Single-run LICENSE is necessary but not sufficient
- N=3 may not characterize fully — N=10+ may be needed for distribution

**Rule clarified**:**multi-run sampling characterizes DISTRIBUTION,not single
"true" value**。If runs split into two modes,deployment confidence
must account for the mode probability。

## Cross-references

- N=2 deterministic finding: `fc41e7e`
- Warmup fix LICENSE(single-run #1): `8281047`
- Run #1 of post-fix build: `b1mm1k0r7`(92%)
- Run #2 of post-fix build: `b4kaqdrmj`(56%)
- Run #3 of post-fix build(this): `b5i3467ad`(76%)
- Codex investigation plan: `fc9bea9`
- Codex final synthesis: `1fce03f`

## Status

- ✅ Warmup fix `c20b1ce` ships(real win for run #1)
- ⚠ Production characterization:**67% normal / 33% degraded mode**
- 🔧 Codex Step 2.A-2.C investigation needed for full LICENSE
- 🔧 N=10+ sampling may be needed for distribution characterization

## Rule

**Multi-run distribution characterization is MORE INFORMATIVE than
single-value LICENSE**:
- N=1:point estimate(can be normal mode or degraded outlier)
- N=3:detect bimodal vs unimodal distribution
- N=10+:full distribution shape + confidence interval

For ARLE specifically:cap=8 deployment is **TTFT-tail-LICENSED**(p99 -86%
robust)but turn-success has bimodal mode distribution。Document this
explicitly in production deployment notes。

This methodology nuance(distribution > point)applies to ALL
production-readiness benches at scale。Single-run LICENSE is starting
estimate;multi-run characterization is production rigor。
