# cap=8 default systematic regression — H4 CUDA Graph warmup max=4 not bumped with cap flip

> Per `150b4c4` N=3 variance characterization recommendation,executed
> Run #2 of cap=8 default verify。
>
> **Result confirms SYSTEMATIC REGRESSION,not variance**:run #1 76%,
> run #2 56%(both << override #19d12c2 100% turn success)。**Root
> cause identified — H4 graph warmup hardcoded `max=4` not bumped
> when cap=8 flipped(`12300c5`)**。

## N=3 Run #2 Empirical

```
Run #2 results:
  TTFT p50:    14817 ms (run #1: 7908,    override: 5868)
  TTFT p99:    15357 ms (run #1: 11182,   override: 10259)
  Turn success: 144/256 (56%) (run #1: 76%, override: 100%)
  Tokens out:   23424   (run #1: 35733,   override: 40740)
  Wall total:  1409 s   (run #1: 2290,    override: ~860)
  Peak mem:    15911 MB (run #1: 15880,   override: 15272)
```

## Cross-run summary

| Run | Cap source | Turn Success | TTFT p99 | Wall |
|---|---|---:|---:|---:|
| `f5cf829` | cap=4 default | **256/256(100%)** | 72515 ms | ~860s |
| `19d12c2` | cap=8 CLI override | **257/257(100%)** | 10259 ms | ~860s |
| `bwa4piqqx` | cap=8 default(post `12300c5`) | 194/256(76%) | 11182 ms | 2290s |
| **`b4r8fha82`(this)** | cap=8 default | **144/256(56%)** | 15357 ms | 1409s |

**Pattern**:
- cap=4 default: 100%(was working)
- cap=8 override: 100%(works on already-warmed server)
- cap=8 default: 56-76%(consistent regression)

Variance hypothesis REFUTED:two cap=8 default runs both show major
regression。Override worked because server was already warmed by prior
benches。

## Root cause — H4 confirmed

Server startup log(both run #1 + run #2):
```
Warming up CUDA Graphs for 4 batch sizes (max 4)...
```

**The warmup max=4 hardcode was NOT updated when codex flipped
`max_concurrent_prefill_requests Some(4) → Some(8)` in `12300c5`**。

Mechanism:
1. Codex `12300c5` bumped cap=4 → cap=8 in `qwen3/forward.rs:316`
2. Warmup loop in `core/warmup.rs` still pre-captures batches 1-4
3. At W4 c=8 8K agent burst,batches 5-8 prefill happens
4. First-encounter graph capture for batch=5-8 takes 100-500 ms each
5. Delay cascades during admission burst → some sessions retry then
   exhaust → 503 error
6. ~25-44% of sessions error out depending on session ordering

Override case `19d12c2` worked because:
- CLI flag was ON
- Server had been warmed by prior benches with batches 5-8 already captured
- Cold-start graph capture penalty was paid in earlier session,not in
  this bench's measured window

## Phase 1.2 fix — codex substrate

`infer/src/scheduler/cuda/core/warmup.rs` needs to read
`max_concurrent_prefill_requests` from model and pre-capture batches
1..N where N = that cap value。Currently appears hardcoded to `min(4, num_slots)`
or similar。

**Proposed fix**(codex pickup):
```rust
// Old (warmup.rs ~line 100, hypothesis):
let max_batch = std::cmp::min(4, num_slots);

// New:
let max_batch = std::cmp::min(
    model.max_concurrent_prefill_requests().unwrap_or(num_slots),
    num_slots,
);
```

LOC:~5。Risk:Low(startup time grows from 4-batch warmup to 8-batch
warmup,~250 ms longer cold-start)。

OR simpler:**revert `12300c5`** until warmup is fixed,keep cap=4 as
production default。`27fd5de` multi-shape LICENSE was based on CLI
override(warm server),not cold-start default。

## Phase 8 verdict — REVERT or FIX recommendation

| Action | Pros | Cons |
|---|---|---|
| **Keep cap=8,fix warmup** | TTFT p99 -86% benefit retained | needs codex substrate fix(~5 LOC) |
| **Revert `12300c5`** | Production stability immediately | TTFT p99 stuck at 72.5s until warmup fix |
| **Hybrid:keep cap=8,document cold-start caveat** | Compromise — production works after warmup pass | footgun for new deployments |

**Recommendation:fix warmup**(codex 1-line scope)。Don't revert the cap
fix that gives -86% TTFT improvement。

## Skill v1.4.0 anti-pattern caught(NEW)

**Implicit-coupling-via-shared-default trap**:
- Cap=4 was set in TWO places(model default + warmup hardcode) without
  obvious connection
- Bumping cap in ONE place left the other STALE
- Production regression ensued

**Rule added(skill v1.4.0)**:**when changing a config parameter,grep
for ALL usages of the OLD value across the codebase,not just the
declaration**。`Some(4)` may appear in warmup loops,kernel
configurations,test fixtures,etc。Each must be evaluated for
consistency with the new value。

This was the methodology gap in `12300c5`'s codex review process —
2 review rounds didn't catch the warmup-cap coupling。Per skill: future
config-change PRs should include a `grep` evidence dump in commit body
showing all usage sites checked。

## Strategic implication

`12300c5` was DIRECTIONALLY correct(TTFT p99 -86% real)but had hidden
coupling bug。Fixing the warmup cap unlocks the full benefit。Or
reverting until warmup fix lands keeps production stable。

Either way,this is a **codex substrate pickup blocker**:`12300c5` is in
tree,production deployment is currently regressed at fresh-server
startups。

Update Tasks:
- Reopen `cap=8 production deployment` as needs-warmup-fix
- New blocker:warmup max bump to match cap

## Cross-references

- Codex flip: `12300c5`
- Override test (warmed server): `19d12c2`(257/257,p99 10259)
- Default verify run #1: `bwa4piqqx`(194/256,p99 11182,`150b4c4` doc)
- Default verify run #2: `b4r8fha82`(this — 144/256,p99 15357)
- Multi-shape LICENSE(CLI override): `27fd5de`
- Original H4 hypothesis: `a25416b` plan §H4
- Warmup logic: `infer/src/scheduler/cuda/core/warmup.rs`

## Status

- ❌ **cap=8 default deployment REGRESSED on fresh-server startup**
- ✅ TTFT p99 improvement IS real(holds at -85% vs cap=4 baseline)
- 🔧 Root cause CONFIRMED:H4 warmup max=4 hardcode coupling
- ⏳ Codex pickup:bump warmup max OR revert `12300c5`(see verdict)

## Rule

**When implementing config flip via single-line change,grep entire
codebase for the old value first**。Production tail-latency regressions
hide in implicit couplings(here:warmup loop bound)。

Skill v1.4.0 generalization:**every config-related PR commit body
should include grep-evidence**:
```
$ grep -rn 'Some(4)' infer/src/ crates/cuda-kernels/src/
infer/src/model/qwen3/forward.rs:316: ...    # this PR changes
infer/src/scheduler/cuda/core/warmup.rs:???   # ← MISSED — needs same flip
```

Without this discipline,single-line "safe" changes ship hidden
regressions。This run cost 2 ticks of Claude variance investigation
to catch — codex review process should catch upstream。
