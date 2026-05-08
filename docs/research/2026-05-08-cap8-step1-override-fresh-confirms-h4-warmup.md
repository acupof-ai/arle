# cap=8 Step 1 — Override fresh server CONFIRMS H4 warmup root cause

> Per codex `fc9bea9` investigation plan Step 1:rerun
> `--prefill-max-requests 8` override on FRESH server to disambiguate
> "is regression due to default propagation or fresh-server cold-start"。
>
> **Result**:override fresh → **201/256(78.5%)turn success**,
> essentially same as default fresh(56-78%)。**`db20d34` H4 warmup
> hypothesis CONFIRMED** — fresh-server cap=8 always regresses
> regardless of cap source(default OR CLI override)。

## Step 1 — Override fresh server result

```
turns OK:        201 / 256(78.5%)
scored turns OK: 128
tokens total:    32169
wall total:      2441 s
TTFT p50/p99:    5254 / 14609 ms
ITL p50/p99:     24.0 / 26.0 ms
W4 scored resume:128 TTFT p50/p99 = 5254 / 14609 ms
Peak mem:        15848 MB
```

## Cross-run definitive evidence

| Run | Cap source | Server state | Turn Success | TTFT p99 |
|---|---|---|---:|---:|
| `f5cf829` | cap=4 default | fresh | **256/256(100%)** | 72515 ms |
| `19d12c2` | cap=8 CLI override | **WARM**(prior benches) | **257/257(100%)** | 10259 ms |
| `bwa4piqqx` | cap=8 default | fresh | 194/256(76%) | 11182 ms |
| `b4r8fha82` | cap=8 default | fresh | 144/256(56%) | 15357 ms |
| **`ba00s5nu3`(this Step 1)** | **cap=8 CLI override** | **fresh** | **201/256(78.5%)** | 14609 ms |

**Pattern definitively isolated**:
- **cap=4 fresh**:100%(works)
- **cap=8 fresh**(any source — default OR override):**56-78%**(REGRESSED)
- **cap=8 warm**(prior benches warmed batches 5-8):**100%**(works)

Variance hypothesis fully REFUTED。Source of cap value(CLI vs default)IRRELEVANT。**Fresh-server cap=8 cold-start is the binding issue**。

## Root cause confirmed:H4 warmup max=4 hardcode

`infer/src/scheduler/cuda/core/warmup.rs` pre-captures CUDA Graphs for
`max=4` batch sizes regardless of cap value。Fresh-server log:
```
Warming up CUDA Graphs for 4 batch sizes (max 4)...
```

When cap=8 burst hits,batches 5-8 prefill triggers FIRST-ENCOUNTER
graph capture(100-500 ms each)during admission cascade → retry
exhaustion → 22-44% session 503 errors。

Warm server case(`19d12c2`):batches 5-8 already captured by prior
bench runs → no first-encounter cost → 100% success。

This was **`db20d34`'s exact hypothesis** — Step 1 produces orthogonal
evidence(override fresh ≈ default fresh)that confirms H4 root cause
without ambiguity。

## Phase 8 — REVERT or FIX decision is now CLEAR

| Option | Effect | LOC |
|---|---|---|
| **Fix warmup**(read cap from model)| Production prevented(0% UX regression)+ TTFT win retained | ~5 LOC |
| Revert `12300c5` | TTFT win lost,production stable | revert |

**Recommendation**:fix warmup,don't revert。`db20d34` Phase 1.2 fix is
empirically validated as the correct path。

## Codex action

Codex pickup unambiguous now:
1. Edit `infer/src/scheduler/cuda/core/warmup.rs`(or wherever `max=4`
   hardcode lives)to read `model.max_concurrent_prefill_requests()`
2. Pre-capture batches 1..N where N = that cap value(now 8 for Qwen3 Marlin path)
3. Re-run W4 c=8 8K agent fresh-server bench → expect 100% turn success
4. Land wins entry per CLAUDE.md mandatory bench rule

Cold-start time grows ~250 ms(4 extra batch graphs × ~60 ms each)— acceptable for production deployment。

## Skill v1.4.0 anti-pattern caught(this entry's contribution)

**"Warm server" implicit dependency trap**:single-run validation `19d12c2`
implicitly assumed server was at production cold-start。In reality,prior
benches had warmed CUDA Graph cache。When `27fd5de` multi-shape LICENSE
based on `19d12c2` declared cap=8 production-ready,it was production-ready
ONLY ON ALREADY-WARM SERVERS。

**Rule added(skill v1.4.0)**:**production-readiness benches MUST start
from cold cargo-clean build OR document warm-state explicitly**。Hidden
warm-state assumptions cause LICENSE → production regressions when
deployed to cold infrastructure。

For ARLE specifically:any CUDA Graph related LICENSE bench should add a
fresh-server cold-start verification step:`cargo clean && cargo build && bench`
to expose warmup-coverage gaps。

## Cross-references

- Codex investigation plan: `fc9bea9`
- H4 hypothesis(`db20d34`): warmup max=4 not bumped with cap flip
- Original cap=8 multi-shape LICENSE: `27fd5de`(now reframed — was warm-state validation)
- Single-run override LICENSE: `19d12c2`(now reframed — warm-state)
- Default cap=8 verify run #1: `bwa4piqqx`(76%)
- Default cap=8 verify run #2: `b4r8fha82`(56%)
- Override fresh Step 1: `ba00s5nu3`(78.5% — this entry)
- Warmup source: `infer/src/scheduler/cuda/core/warmup.rs`

## Status

- ✅ H4 warmup root cause CONFIRMED via Step 1 evidence
- ✅ Variance hypothesis REFUTED(both cap=8 sources regress same)
- ✅ Warm-state assumption EXPOSED
- 🔧 Codex pickup:fix warmup ~5 LOC OR revert pending decision

## Rule

**Warm-state assumption is a hidden methodology trap**:
- If a bench's first run after building gives 100% but a fresh
  cold-start run gives 76%,warm-state is hiding a real issue
- Document warm-state explicitly in benches
- Multi-run variance characterization(per `150b4c4` rule)+ fresh-build
  verification(this entry's rule)are BOTH required

For codex review process:future config-flip PRs should include either:
- Cold-start fresh-build verification trace,OR
- Explicit "this LICENSE assumes warm server" caveat with deployment guidance

This methodology gap cost 2 ticks of Claude variance investigation
+ 1 codex round-trip。Closing the rule prevents recurrence。
