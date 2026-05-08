# cap=8 default verify — TTFT p99 holds but turn success variance flagged

> Per `12300c5` codex bumped Qwen3 `max_concurrent_prefill_requests`
> default `Some(4) → Some(8)` post `27fd5de` multi-shape LICENSE。
> Production verify ran fresh build + W4 c=8 8K agent burst WITHOUT
> `--prefill-max-requests` flag(default cap=8)。
>
> **Result**:TTFT p99 holds at −85%(11182 vs `f5cf829` 72515 ms)but
> **turn success regressed to 194/256(75.8%)**(vs `19d12c2` override
> test 257/257 100%)。Cap=8 fix is fundamentally working;turn-success
> variance needs follow-up before declaring production-deployment
> green-light。

## Phase 5 — Single-variable A/B(default vs CLI override)

**Variable**:flag mechanism(CLI `--prefill-max-requests 8` vs no flag,
let model default `Some(8)` propagate)。

All else identical:
- Model:`Qwen3-4B-W4A16-sym-g128-marlin`(post-zpfix corrected)
- Workload:`agent-w4-tool-resume`(128 sessions × 2 turns,8K + 256)
- Concurrency:c=8
- Hardware:sm_89 RTX 4070 Ti SUPER

```bash
# Baseline override (19d12c2)
./target/release/infer ... --prefill-max-requests 8

# Treatment default (this — post 12300c5 flip)
./target/release/infer ... # no --prefill-max-requests flag
```

## Results — TTFT win HOLDS,turn success VARIES

| Metric | `f5cf829` cap=4 baseline | `19d12c2` cap=8 override | **`12300c5` cap=8 default(this)** |
|---|---:|---:|---:|
| **TTFT p50** | 11768 ms | 5868 ms | **7908 ms** |
| **TTFT p99** | **72515 ms** | **10259 ms** | **11182 ms** |
| **TTFT spread p99/p50** | 6.2× | 1.75× | **1.41×** |
| **TTFT vs cap=4 baseline** | — | **−86%** p99 | **−85%** p99 |
| ITL p50 / p99 | 16.5 ms / n/a | 25.9 / 26.1 ms | 25.9 / 26.0 ms |
| **Turn success** | **256/256(100%)** | **257/257(100%)** | **194/256(75.8%)** |
| Tokens out | 44665 | 40740 | **35733** |
| Wall total | ~860 s | ~860 s | **2290 s** |
| Peak mem | 15336 MB | 15272 MB | **15880 MB** |
| engine_batch_occupancy | 0.825 | 0.833 | **0.867** |
| session_slot_pressure_evictions_hard | 218 | 240 | 184 |

## Phase 7 tradeoff — turn success variance hypothesis

Across 3 W4 c=8 8K runs:
- `f5cf829`(cap=4):**100% turn success**,p99 72.5s
- `19d12c2`(cap=8 CLI override):**100% turn success**,p99 10.3s
- **This run`12300c5`(cap=8 default)**:**76% turn success**,p99 11.2s

Same workload,same nominal cap value,**different turn-success outcome**。
Possible explanations:

1. **Run-to-run variance**:bench duration grew(2290s vs 860s)→ more
   sessions hit cumulative scheduling pressure。Could be deterministic
   to specific session ordering that this run encountered。
2. **Fresh build cold-start**:CUDA Graph captures during early sessions
   tax tail TTFT。Override test may have warmed the graph state from
   prior runs。
3. **Memory pressure**:peak 15880 MB(+608 MB vs override)closer to
   16384 MB ceiling。If transient peak hit limit,evictions cascade →
   session 503 cascade。
4. **Build artifact difference**:rebuild for `12300c5` may have produced
   slightly different binary(though only 1 LOC changed)。

The TTFT p99 metric IS robust(both override + default produce −85% to
−86% reduction)。Cap=8 fix is fundamentally working。

## Phase 8 verdict — partial LICENSE,follow-up needed

| Threshold | Override `19d12c2` | **Default(this)** |
|---|---|---|
| TTFT p99 ≤ 30k ms | 10259 ✅ | 11182 ✅ |
| Spread ≤ 3× | 1.75× ✅ | 1.41× ✅ |
| Turn success ≥ 95% | 100% ✅ | **76% ❌** |
| No OOM | 700 MB headroom | 504 MB headroom |
| ITL p50 stable | 25.9 ✅ | 25.9 ✅ |

**Mixed verdict**:
- TTFT improvement LICENSED at default(holds at -85% p99)
- Turn-success threshold FAILED at default(76% < 95% needed)
- **Follow-up needed**:re-run default cap=8 to determine if 76% is
  variance or systematic regression

## Action items

1. **Re-run default cap=8 verify 2-3× to characterize variance**
   - If next runs all > 95% turn success → previous run was outlier
   - If next runs persist 70-80% turn success → systematic regression,
     codex investigate
2. **Investigate why peak_mem +608 MB vs override**
   - Build artifact difference?
   - Session ordering / KV admission timing?
3. **Trace specific failed session timing**
   - Which sessions errored?
   - Did they hit 503 retry exhaustion or KV eviction?

## Skill v1.4.0 anti-pattern caught(NEW)

**Variance window methodology limit**:single bench run(even with σ-tight
metrics)is insufficient evidence for production deployment if **turn
success variance** is in play。Per skill v1.4.0:

- TTFT distribution(p50,p99,spread)— validated via 1 σ-tight run
- Turn success rate — needs **multi-run validation**(N=3 minimum)to
  characterize variance band

**Rule added(skill v1.4.0)**:**before declaring config change
production-LICENSED,verify N=3 runs across the binding shape**;
single-run turn success ≥ 95% is necessary but not sufficient。

This was the methodology gap in `27fd5de` — it had only 1 cap=8 multi-shape
run。This default-cap=8 verify shows variance was missed at single-run
LICENSE。

## Strategic implication

Codex `12300c5` cap=8 flip is **likely correct in spirit**(TTFT improvement
robust)but **deployment confidence not yet at production-grade**。
Recommend:
- Hold cap=8 in tree(don't revert)
- Run N=3 verify benches over next ticks
- If consistent ≥ 95% turn success → close as production-deployment
- If consistent 70-80% → revert to cap=4 OR add memory-pressure guard

## Cross-references

- Codex flip:`12300c5`(`fix(scheduler): bump Qwen3 max_concurrent_prefill_requests Some(4) → Some(8)`)
- TTFT p99 -86% override test:`19d12c2`(257/257 success)
- Multi-shape LICENSE(included this regression-prone gap):`27fd5de`
- W3 c=16 cap=8 verify:included in `27fd5de`(384/384 — different workload,not affected)
- Original deadlock baseline:`f5cf829`(cap=4,256/256 success)
- Skill v1.4.0:`6c627c4`
- Bench artifact:`bench-output/2026-05-08-arle-w4-c8-cap8-default-verify.json`(local)

## Status

- ✅ TTFT p99 -85% holds with default cap=8(robust)
- ❌ Turn success 76% in this run vs 100% in override(variance gap)
- ⏳ N=3 verify pending(Claude actionable next 3 ticks)
- 🔧 If variance confirms:keep cap=8 in tree;if regression:revert or add guard

## Rule

**Multi-run variance characterization is mandatory before global
production default flip**。Single run with σ-tight metrics validates
distribution shape but not run-to-run reproducibility of binary
outcomes(turn success/fail)。

For ARLE specifically:any model-level default change in
`forward.rs:max_concurrent_prefill_requests`(or similar admission
caps)should be N=3 verified at the binding shape before merging。
This run's 76% vs prior 100% is the cost of single-run LICENSE。
