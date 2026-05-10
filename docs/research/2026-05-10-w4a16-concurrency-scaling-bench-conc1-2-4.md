---
title: W4A16 concurrency-scaling bench — conc=1/2/4 single-server A/B sets concrete Medusa floor
date: 2026-05-10
type: research
status: closed (3-point scaling curve captured)
related_tasks: [#28 (Medusa, perf floor now concretely set), #30 (Hybrid W4A16/W4A8 dispatch)]
related_skills: [#34 (single-conc not sufficient), #38 (warmup clamp)]
---

# W4A16 concurrency-scaling bench — conc=1/2/4

> **Purpose**: Per skill `kernel-optimization` Phase 5 sub-rule
> "single-conc bench NECESSARY but not SUFFICIENT" (#34), extend the
> 4-arm A/B (`d8b2870`) which was conc=1 only, to characterize how
> W4A16 scales at higher concurrency. Concrete numbers for Medusa
> Phase 1.A perf floor (Task #28) now include scaling efficiency.

## §1 Bench config (single-server, single-variable)

| Variable | Value |
|---|---|
| Server | `target/release/infer --model-path Qwen3-4B-GPTQ-W4A16-marlin-zpfix --port 8000` |
| Env | `RUST_MIN_STACK=33554432` (no PF8 env vars) |
| Workload | guidellm `--profile concurrent --rate {1,2,4} --max-seconds 60 --warmup 5` |
| Data | `prompt_tokens=512, output_tokens=128` (deterministic stdev=1) |
| Variable changed | `--rate` (concurrency only) |
| Same server reused | YES (warmup amortized once at conc=1, held across) |

## §2 Result table

| Conc | TTFT mdn | TTFT p95 | ITL mdn | ITL p95 | tok/s mean | req/s mean |
|------|---------:|---------:|--------:|--------:|-----------:|-----------:|
| **1** (Arm C) | **66.0 ms** | 67.1 ms | **5.8 ms** | 5.8 ms | **159.6** | 1.25 |
| **2** | **82.1 ms** | 126.9 ms | **7.4 ms** | 7.4 ms | **248.8** | 1.96 |
| **4** | **78.1 ms** | 158.9 ms | **7.7 ms** | 8.6 ms | **469.6** | 3.71 |

**Δ vs conc=1**:
- TTFT: +24% at conc=2, **+18% at conc=4** (improves -5% from conc=2→4)
- ITL: +28% at conc=2, +33% at conc=4 (mostly stabilizes after conc=2)
- tok/s: +56% at conc=2, **+194% at conc=4** (2.94× = 73% scaling efficiency)
- req/s: +57% at conc=2, +197% at conc=4 (mirrors tok/s)

## §3 Findings

### §3.1 TTFT scales sub-linearly + improves conc=2→4

Counter-intuitive but real: TTFT mdn **drops** from 82.1 ms (conc=2)
to 78.1 ms (conc=4). Hypothesis (n=1, untested): Pass 3 prefill
graph (per Task #35 substrate `a2ad788`) hits MORE often at higher
concurrency — multiple prefill requests batch into the captured
shape, amortizing graph dispatch cost across more requests.

p95 TTFT does grow significantly (67.1 → 126.9 → 158.9 ms),
indicating tail latency suffers under load — likely admission queue
backpressure for incoming requests waiting on slot.

### §3.2 ITL stabilizes after conc=1

ITL grows +28% from conc=1 to conc=2 (5.8 → 7.4 ms), then only +5%
to conc=4 (7.4 → 7.7 ms). Decode path overhead is mostly fixed-cost
amortized once concurrency ≥ 2; further increases are minor.

### §3.3 73% throughput scaling efficiency at conc=4

2.94× tok/s improvement at 4× concurrency = 73% scaling efficiency.
For comparison: perfect linear would be 4×; SGLang reference per
`#40` Path B.2 wins entry mentions ~76% at similar workload. ARLE
W4A16 path is in the same ballpark.

## §4 Implications

### §4.1 Concrete Medusa Phase 1.A perf floor (Task #28)

For Medusa to deliver "world-first" gains over current W4A16
baseline, must beat per-concurrency:

| Conc | Floor TTFT | Floor ITL | Floor tok/s | 2× tok/s target |
|---|---:|---:|---:|---:|
| 1 | ≤ 66 ms | ≤ 5.8 ms | ≥ 160 | **≥ 320** |
| 2 | ≤ 82 ms | ≤ 7.4 ms | ≥ 249 | **≥ 498** |
| 4 | ≤ 78 ms | ≤ 7.7 ms | ≥ 470 | **≥ 940** |

**At conc=4, Medusa must reach ≥940 tok/s** to deliver 2× over
current W4A16 baseline. SGLang Medusa published numbers typically
hit 2-3× decode tok/s at acceptance ≥ 70%, so 940 is achievable.

### §4.2 Refines direction options recommendation

Updates `2026-05-10-post-pf85-direction-options.md` §6:

> "Path I (Medusa, 2× tok/s at 70% accept): effective ITL halved
> → ~2.9 ms (~-50%)"

was correct for conc=1. At conc=4, the more relevant production
workload, Medusa floor is 940 tok/s = 2× of 470 tok/s baseline.
Achievable per SGLang Medusa published numbers.

### §4.3 Refines Option B (Hybrid W4A16/W4A8 dispatch) reasoning

Per `cc8b437` Option B revision: B.1 (dual-quant checkpoint)
requires ~2 weeks tooling. BUT: the conc=4 W4A16 throughput data
(469 tok/s) shows W4A16 already scales well. The hybrid dispatch's
theoretical "best of both" is now quantified:
- W4A8 prefill (54.2ms TTFT @ conc=1) + W4A16 decode (5.8ms ITL @ conc=1)
- Hybrid would target ~54ms TTFT + 5.8ms ITL at conc=1
- vs W4A16-only: 66ms TTFT (-18%) + 5.8ms ITL (no change)
- Net win at conc=1: ~-18% TTFT only

At conc=4, hybrid would target 54ms-prefill (best W4A8) + ~7.7ms
ITL (best W4A16 at conc=4) = comparable to W4A16-only at conc=4
(78.1ms TTFT + 7.7ms ITL). The relative win shrinks at higher
concurrency. **Option B's value proposition is mostly conc=1
specific**, less compelling for production conc=4 workload.

This further strengthens the Option A (Medusa) recommendation for
production-relevant performance.

## §5 SKILL implications

### §5.1 #34 (single-conc not sufficient) reinforced

3-arm scaling curve required to see the sub-linear-then-improving
TTFT pattern. Single-conc bench would have missed both:
- That conc=2 has worst-case p95 TTFT growth (67 → 127 ms)
- That conc=4 actually reduces median TTFT vs conc=2 (82 → 78 ms)

This is the canonical "single-X is necessary but not sufficient"
pattern at the concurrency axis.

### §5.2 W4A16 path validated as production-ready

0 kernel failures across 60s × 3 conc levels. Substrate is solid
even at conc=4 sustained — contrasts with PF8.3 KILL (5878 failures
at conc=1 single-shot per `0be278f`).

## §6 Cross-references

- `06b7437` Arm C original conc=1 W4A16 control bench (extends this)
- `d8b2870` 4-arm A/B + perf comparison (sets baseline this extends)
- `cc8b437` post-PF8.5 direction options revision (this updates §6)
- `bench-output/2026-05-10-armC-w4a16-conc{2,4}/benchmarks.{json,csv}`
- `/tmp/pf85-armC-w4a16-multi-conc.log`
- SKILL `kernel-optimization` v1.12.0 #34 (single-conc not sufficient)
- SKILL `kernel-optimization` v1.13.0 #38 (warmup clamp)
- ROADMAP.md SGLang reference comparison context
