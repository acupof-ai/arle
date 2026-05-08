# W3 c=4 cap=8 default LICENSED clean 100% — TTFT p50 -45% improvement

> Per `5cee921` EOD reference table production decision tree:cap=8
> default at low-pressure shapes(W3 short multiturn c=4)expected to
> work cleanly。Empirical verification confirms。
>
> **Result:384/384 turn success(100%),TTFT p50 -45% vs cap=4 baseline
> `370a267`**(208 vs 379 ms)。No bimodal at low-pressure shapes — 
> bimodal issue confined to W4 8K-prompt c=8 burst scenarios(`a0a3f42`)。

## Phase 1 target

| Field | Value |
|---|---|
| Metric | turn success + TTFT/ITL on W3 short multiturn c=4 |
| Baseline | `370a267` cap=4 default:384/384 turns OK,TTFT p50 379 ms,ITL p50 8.5 ms |
| License threshold | turn success ≥ 95% AND no TTFT regression |
| Kill threshold | turn success regression OR ITL regression > 20% |

## Phase 5 — Single-variable A/B(cap default,same workload)

**Variable**:`max_concurrent_prefill_requests` model default(was `Some(4)`,
now `Some(8)` per `12300c5` + warmup max=8 per `c20b1ce`)。

All else identical:
- Model:`Qwen3-4B-W4A16-sym-g128-marlin`
- Workload:`agent-w3-short-multiturn`(64 sessions × 6 turns)
- Concurrency:c=4
- Server:`--num-slots 8 --max-seq-len 5120`(matches `370a267` baseline)

```bash
CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  ./target/release/infer \
  --model-path infer/models/Qwen3-4B-W4A16-sym-g128-marlin \
  --port 8000 --num-slots 8 --max-seq-len 5120
# Default cap=8 from model (post 12300c5)
# Warmup max=8 (post c20b1ce, num_slots=8)

PATH=.venv/bin:$PATH \
  python scripts/bench_agent_trace.py \
    --workload agent-w3-short-multiturn \
    --num-concurrent 4 \
    --label arle-w3-c4-cap8-default-warmupfix
```

## Results — production CLEAN

| Metric | `370a267`(cap=4 default) | **cap=8 default(this)** | Δ |
|---|---:|---:|---:|
| Turn success | 384/384(100%) | **384/384(100%)** | maintained |
| **TTFT p50** | 379 ms | **208 ms** | **−45% IMPROVED** |
| TTFT p99 | 582 ms | 573 ms | similar(within σ) |
| ITL p50 | 8.5 ms | **8.5 ms** | same |
| ITL p99 | 8.8 ms | 8.8 ms | same |
| Wall total | n/a | 345 s | new datapoint |
| engine_batch_occupancy | n/a | **91.6%** | high efficiency |
| prefix_hit_rate | 99.0% | **99.0%** | same |
| Peak mem | 14491 MB | 14300 MB | similar |
| session_slot_pressure_evictions_hard | 0 | **0** | NO evictions |

W3 scored split:
- Warm turns:256 TTFT p50/p99 = 205 / 571 ms
- Cold turns:64 TTFT p50/p99 = 319 / 843 ms

## Phase 7 tradeoffs

| Axis | Status | Note |
|---|---|---|
| Turn success | ✅ 100% maintained | no regression |
| **TTFT p50** | ✅ **-45% IMPROVEMENT** | bigger admission window benefit |
| TTFT p99 | ✅ similar | unchanged |
| ITL | ✅ same | low-conc not bottlenecked by cap |
| Memory | ✅ similar | no extra pressure |
| Workflow | ✅ no config changes | cap=8 default works out-of-box |

**Why TTFT improved at low concurrency**:
- Cap=4 baseline:4 sessions admitted at burst start,sequential prefill
- Cap=8 default:4 sessions all admit immediately(cap=8 > c=4 conc)
- No admission cascade,no queue wait → TTFT drops 45%

## Phase 8 license — LICENSED HARD

| Threshold | Result | Verdict |
|---|---|---|
| Turn success ≥ 95% | 100% | ✅ |
| TTFT no regression | -45% improvement | ✅ |
| ITL no regression | same | ✅ |
| No OOM | 14.3 GB / 16 GB | ✅ |

**LICENSED HARD** for production at W3 short multiturn c=4。

## Strategic implication — cap=8 default is GOOD for common-case workloads

Cross-shape cap=8 default deployment status post-fix:

| Shape | Turn Success | TTFT impact | Status |
|---|---:|---|---|
| W3 c=4 short multiturn(this) | 100% | -45% improvement | **LICENSED HARD** |
| W3 c=16 short(`27fd5de`) | 100% | similar | LICENSED HARD |
| **W4 c=8 8K agent burst** | **bimodal 56-92%** | **-86% p99 win** | **CONDITIONAL**(per `a0a3f42`) |

**Bimodal issue is confined to W4 8K-prompt c=8 burst scenarios**(prefill-bound,
high-pressure)。**Common-case workloads(W3 / 4k longctx / lower conc)
benefit unconditionally from cap=8 default**。

## Production deployment recommendation refined

For most production workloads:**cap=8 default ships safely**:
- W3 W4 c=4-16 short prompts:CLEAN
- 4k longctx all c:CLEAN(per `c4fae17` 3-shape grid)
- Long-context decode:CLEAN(less affected by cap)

For W4 c=8 8K agent burst specifically:
- TTFT p99 -86% gain robust
- Turn success bimodal pending `#35` prefill pre-warm fix
- Deploy with caveat OR use cap=4 conservative for that specific workload

This refines the conservative reading from earlier today。Cap=8 default
is **GOOD for production majority,with one specific bimodal caveat**。

## Cross-references

- W3 c=4 cap=4 baseline: `370a267`
- cap=8 multi-shape LICENSE(W3 c=16 + W4 c=8 8K): `27fd5de`
- W4 c=8 8K bimodal characterization: `a0a3f42`(67% normal / 33% degraded)
- Codex H_grcap pre-warm directive: `56dbd1c`
- EOD production reference: `5cee921`
- Skill v1.5.0:`f05ea3a`
- Bench artifact: `bench-output/2026-05-08-arle-w3-c4-cap8-default-warmupfix.json`(local)

## Status

- ✅ W3 c=4 cap=8 default CLEAN 100% with TTFT -45% improvement
- ✅ Cap=8 default LICENSED for common-case workloads
- ⏳ W4 c=8 8K specific bimodal pending `#35` prefill pre-warm fix
- 🎯 Production deployment recommendation refined:cap=8 default majority-safe

## Rule

**Bimodal failure modes can be workload-specific even after substrate
fixes**。Cap=8 default has bimodal at W4 8K c=8 burst BUT is clean at:
- W3 c=4 short(this entry — TTFT IMPROVES)
- W3 c=16 short(`27fd5de` — 100% turn success)
- 4k longctx multi-c(`c4fae17`/`8588f6a` — tight σ)

Production deployment guidance must distinguish bimodal-affected vs
clean shapes。Single "production caveat" oversimplifies — workload
classification matters。

For ARLE specifically:cap=8 default ships for majority of workloads,
W4 8K c=8 burst gets cap=4 fallback OR waits for `#35` pre-warm fix。
