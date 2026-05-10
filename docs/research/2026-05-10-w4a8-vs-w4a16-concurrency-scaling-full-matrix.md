---
title: W4A8 vs W4A16 full concurrency-scaling matrix conc=1/2/4 — Hybrid Option B value quantified at conc=4 (~3.5% gain, sub-Machete-class)
date: 2026-05-10
type: research
status: closed (full 6-cell matrix captured, Option B ROI quantified)
related_tasks: [#28 (Medusa P1), #30 (Hybrid B option)]
related_skills: [#34 (multi-conc essential), #38 (warmup clamp)]
---

# W4A8 vs W4A16 — full concurrency-scaling matrix

> **Purpose**: extends `8d32576` (W4A16 scaling) with parallel W4A8
> scaling at same conc=1/2/4. Now have 6-cell perf matrix at the
> production-relevant conc range. Quantifies Option B (Hybrid
> W4A16/W4A8 dispatch) value precisely.

## §1 Bench config

Same as `8d32576` W4A16 except model:
- Model: `infer/models/Qwen3-4B-GPTQ-W4A8-zpfix`
- Server: `target/release/infer --model-path ... --port 8000` (no PF8 env)
- Workload: guidellm `--profile concurrent --rate {2,4} --max-seconds 60 --warmup 5`
- Data: `prompt_tokens=512, output_tokens=128`
- Same server reused (warmup amortized), only `--rate` changes
- conc=1 reused from Arm D (`d8b2870`)

## §2 6-cell perf matrix

| Conc | Path | TTFT mdn | TTFT p95 | ITL mdn | ITL p95 | tok/s mean | req/s mean |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | W4A16 | 66.0 | 67.1 | 5.8 | 5.8 | 159.6 | 1.25 |
| 1 | W4A8 | **54.2** | 55.0 | 11.9 | 11.9 | 81.7 | 0.64 |
| 2 | W4A16 | 82.1 | 126.9 | 7.4 | 7.4 | 248.8 | 1.96 |
| 2 | W4A8 | 83.2 | 83.7 | 12.7 | 12.7 | 149.1 | 1.20 |
| 4 | W4A16 | 78.1 | 158.9 | 7.7 | 8.6 | 469.6 | 3.71 |
| 4 | W4A8 | **52.8** | 112.9 | 13.0 | 13.6 | 289.4 | 2.33 |

**W4A8 vs W4A16 deltas**:

| Conc | TTFT Δ% | ITL Δ% | tok/s Δ% |
|---|---:|---:|---:|
| 1 | **-18%** | +105% | -49% |
| 2 | +1% | +72% | -40% |
| 4 | **-32%** | +69% | -38% |

## §3 Findings

### §3.1 W4A8 TTFT advantage is concurrency-bimodal

W4A8 wins TTFT at conc=1 (-18%) and conc=4 (-32%) but **ties at conc=2**
(+1%). Why? Hypothesis (n=1, untested): at conc=2 the prefill batching
benefit balances W4A8's per-prefill compute advantage (FP8 mma faster);
at conc=4 prefill batching saturates and W4A8's compute advantage
dominates.

### §3.2 W4A16 throughput dominates everywhere

At every conc, W4A16 has ~1.6-1.95× higher tok/s than W4A8. Decode
ITL is the binding constraint and W4A16 wins decode (per-token
quant overhead of W4A8 dominates at single-token decode).

### §3.3 W4A8 ITL stays consistently 70-100% worse than W4A16

Across all 3 concs: W4A8 ITL is ~70-100% higher than W4A16. Not
concurrency-dependent — it's an architectural per-token cost.

## §4 Option B (Hybrid W4A16/W4A8 dispatch) value quantified

### §4.1 End-to-end request latency calculation

For 128-token output requests, end-to-end latency =
TTFT + (output_tokens - 1) × ITL = TTFT + 127 × ITL:

| Conc | W4A16 latency | W4A8 latency | Hybrid (W4A8 prefill + W4A16 decode) |
|---|---:|---:|---:|
| 1 | 66.0 + 127×5.8 = **802 ms** | 54.2 + 127×11.9 = 1565 ms | 54.2 + 127×5.8 = **791 ms** (-1.4%) |
| 2 | 82.1 + 127×7.4 = **1022 ms** | 83.2 + 127×12.7 = 1696 ms | 82.1 + 127×7.4 = 1022 ms (no win) |
| 4 | 78.1 + 127×7.7 = **1056 ms** | 52.8 + 127×13.0 = 1704 ms | 52.8 + 127×7.7 = **1031 ms** (-2.4%) |

**Hybrid Option B perceived-latency value**:
- conc=1: -1.4% (marginal)
- conc=2: 0% (W4A8 has no TTFT advantage here)
- conc=4: -2.4% (the largest win, still marginal)

### §4.2 Comparison to user's stated -20-40% Machete-class target

User's directive: "预估 -20-40% ITL vs current Marlin"

Option B Hybrid delivers AT MOST **-2.4%** end-to-end latency
improvement at conc=4. **One order of magnitude below the user's
target**. Hybrid does NOT deliver Machete-class gains.

This **strongly confirms `cc8b437` recommendation revision**: Option
B is no longer competitive vs Option A (Medusa). The earlier framing
("hybrid combines best of both = -50% ITL net") was based on naïve
"add the wins" math, not end-to-end latency analysis.

### §4.3 Path forward for Machete-class gains

Per §6 of `2026-05-10-machete-framing-re-disambiguation-post-pf85-kill.md`,
only Path I (Medusa) reliably exceeds the -20-40% target via 2-3×
tok/s improvement at acceptance ≥ 70%.

Path I (Medusa) effective ITL improvement:
- W4A16 baseline conc=4: 7.7 ms ITL × ~3.71 req/s
- Medusa 2× tok/s at 70% accept: effective ITL halved → ~3.85 ms (-50%)
- Even 1.5× tok/s: ITL -33%

**Both Medusa scenarios exceed -20% target. Hybrid does not.**

## §5 SKILL implications

### §5.1 #34 (single-X not sufficient) — n+1 evidence at concurrency axis

Conc=2 W4A8 TTFT measurement (+1% Δ vs W4A16) is qualitatively
different from conc=1 (-18%) and conc=4 (-32%). Without all 3 conc
levels, the bimodal pattern would have been missed → wrong conclusion
about whether W4A8 prefill advantage holds.

### §5.2 New SKILL candidate: end-to-end latency math IS load-bearing

Naïve "add the wins" math: W4A8 prefill -18% + W4A16 decode unchanged
= "Hybrid wins big". End-to-end math: -1.4% at conc=1, -2.4% at conc=4.
The framing decay between "best of both" and "actual perceived
latency" is exactly the SKILL #29 framing-decay pattern at the
metric-aggregation axis.

Detection rule: when proposing hybrid/dispatch optimizations,
calculate end-to-end perceived latency = TTFT + (out_tokens - 1) ×
ITL across the relevant concurrency band. The "best of both" framing
without this calculation is hypothesis-grade evidence.

n=1 evidence: this Option B revision (cc8b437 → §4.2 here).

## §6 Status

**Closed — Option B (Hybrid) ROI definitively quantified at sub-
Machete-class (-2.4% max).** Option A (Medusa) remains the only
recommended path for user's stated -20-40% goal.

Direction options doc `2026-05-10-post-pf85-direction-options.md`
should be updated with this evidence to make the recommendation
ironclad.

## §7 Cross-references

- `8d32576` W4A16 scaling conc=1/2/4 (parallel to this for W4A8)
- `d8b2870` Arm D W4A8 conc=1 baseline
- `cc8b437` Option B revision (this strengthens with end-to-end math)
- `ed2aaa3` Machete framing re-disambiguation (this confirms Path I = Medusa)
- `bench-output/2026-05-10-armD-w4a8-conc{2,4}/benchmarks.{json,csv}`
- `/tmp/armD-w4a8-multi-conc.log`
- SKILL `kernel-optimization` v1.12.0 #34 (multi-conc not sufficient)

## §8 W4A8 long-ctx extension (added EOD+1820)

Per follow-up: W4A8 prompt=2048 conc=1 with `--max-seq-len 8192`,
parallel to W4A16 long-ctx series (`2048eca` §10).

### §8.1 W4A8 vs W4A16 at long context (prompt=2048)

| Metric | W4A8 prompt=2048 | W4A16 prompt=2048 (8d32576+§10) | W4A8 vs W4A16 |
|---|---:|---:|---:|
| Successful (60s) | 32 | 51 | -37% |
| TTFT mdn | **191.3 ms** | 272.1 ms | **-30%** |
| ITL mdn | 12.6 ms | 6.4 ms | **+97%** |
| tok/s mean | 71.8 | 117.6 | -39% |
| Kernel failures | 0 | 0 | ✓ |
| Cache demotions | **0** | 1 (at 4k+) | W4A8 less mem pressure |

### §8.2 W4A8 prefill-advantage WIDENS at long context

At prompt=512 (Arms C+D): W4A8 TTFT -18% vs W4A16
At prompt=2048 (this): W4A8 TTFT **-30%** vs W4A16

Prefill is a larger fraction of total compute at long context, so
W4A8's FP8 mma advantage compounds. ITL stays ~+100% (architectural,
per-token cost).

### §8.3 Hybrid Option B value GROWS with context length

End-to-end latency at conc=1 prompt=2048 (output=128):

| Path | TTFT + 127×ITL | E2E | vs W4A16 |
|---|---|---:|---:|
| W4A16 | 272.1 + 127×6.4 | 1085 ms | baseline |
| W4A8 | 191.3 + 127×12.6 | 1791 ms | +65% (worse, ITL dominates) |
| **Hybrid** (W4A8 prefill + W4A16 decode) | 191.3 + 127×6.4 | **1004 ms** | **-7.5%** |

Compare to short-ctx (prompt=512) hybrid value:
- conc=1 prompt=512: -1.4% perceived latency
- conc=4 prompt=512: -2.4% perceived latency
- **conc=1 prompt=2048: -7.5% perceived latency** (this row)

**Hybrid Option B value 3-5× higher at 2k context vs 512.** Still
sub-Machete-class (-20-40% target) but no longer sub-1% noise.

### §8.4 Updated direction options recommendation

The original ironclad "A (Medusa) only" recommendation per
`92813dc` + `12e0c07` was based on prompt=512 data. With long-context
data, **Option B becomes more viable** — but only for long-context
workloads where prefill dominates.

Refined recommendation matrix:
- **Short-ctx workloads (prompt ≤ 512)**: Option A (Medusa) ironclad
- **Long-ctx workloads (prompt ≥ 2048)**: Option B (Hybrid) viable
  with -7.5% gain; Option A still better for throughput
- **Mixed workloads**: depends on prompt distribution; favor whichever
  is dominant

For "world-first 长序列推理引擎" goal — Option B becomes more
attractive than the prior short-ctx analysis suggested. Worth
re-evaluating Option B Phase 1 cost (B.1 dual-quant checkpoint
~2 weeks tooling) vs Option A (Medusa 2-3 days) given long-ctx
target audience.

### §8.5 Cross-references (added)

- `bench-output/2026-05-10-w4a8-longctx-prompt2048/benchmarks.{json,csv}`
- `/tmp/w4a8-longctx-2048.log` (server log, 0 kernel failures, 0 cache demotions)
- `2048eca` W4A16 long-ctx prompt=2048 (parallel measurement)
- `12e0c07` direction options strengthening (this complicates the
  recommendation for long-ctx workloads)

## §9 W4A8 long-ctx prompt=4096 extension (added EOD+1900)

Extended W4A8 long-ctx series to 2 points (2k + 4k), parallel to
W4A16 series.

### §9.1 W4A8 long-ctx 2-point curve

| Metric | prompt=2048 | prompt=4096 | scaling |
|---|---:|---:|---:|
| TTFT | 191.3 ms | 409.4 ms | **2.14×** (linear with 2× prompt) |
| ITL | 12.6 ms | 13.8 ms | +9.5% |
| tok/s | 71.8 | 59.5 | -17% |
| Successful (60s) | 32 | 26 | -19% |
| Cache demotions | 0 | 1 | +1 |
| Kernel failures | 0 | 0 | ✓ |

### §9.2 W4A8 vs W4A16 comparison at prompt=4096

| Metric | W4A8 4k | W4A16 4k (`4e2f39a`) | W4A8 vs W4A16 |
|---|---:|---:|---:|
| TTFT | **409.4 ms** | 577.6 ms | **-29%** |
| ITL | 13.8 ms | 7.4 ms | +86% |
| tok/s | 59.5 | 84.6 | -30% |

Pattern from §8.1 confirmed at 4k: W4A8 TTFT advantage holds at
~-30%, ITL stays ~+80-100% worse.

### §9.3 Hybrid Option B value across context lengths (now n=4 contexts)

E2E latency calculation (TTFT + 127 × ITL for output_tokens=128):

| Workload | W4A16 E2E | W4A8 E2E | Hybrid E2E | Hybrid vs W4A16 |
|---|---:|---:|---:|---:|
| conc=1 prompt=512 | 802 ms | 1565 ms | 791 ms | **-1.4%** |
| conc=4 prompt=512 | 1056 ms | 1704 ms | 1031 ms | **-2.4%** |
| conc=1 prompt=2048 | 1085 ms | 1791 ms | 1004 ms | **-7.5%** |
| **conc=1 prompt=4096** | **1517 ms** | 2162 ms | **1349 ms** | **-11.1%** |

**Hybrid value progression**: -1.4% → -2.4% → -7.5% → -11.1% as
context grows. Approaching but not yet Machete-class (-20-40% target).

### §9.4 Extrapolation to 8k context

Per §11.4 of `2048eca` long-ctx wins entry: W4A16 prompt=8192 = TTFT 1335ms.
Predicted W4A8 prompt=8192: TTFT ≈ 1335 × (1 - 0.29) = ~948 ms (assuming
-29% W4A8 advantage holds).

E2E at 8k (predicted):
- W4A16: 1335 + 127×8.9 = 2466 ms
- W4A8: 948 + 127×~14 = 2726 ms (W4A8 ITL also grows to ~14 at 8k)
- Hybrid: 948 + 127×8.9 = 2078 ms
- Hybrid vs W4A16: **-15.7%** (predicted, would test next bench)

At 16k context: predicted hybrid ~ -18 to -20% (approaching Machete-
class threshold).

### §9.5 Strategic insight for "world-first 长序列推理引擎"

Hybrid Option B value structurally GROWS with context length per the
pattern:
- prompt=512: -1.4% (sub-noise)
- prompt=2048: -7.5% (becomes meaningful)
- prompt=4096: -11.1% (clearly above noise)
- prompt=8192 (predicted): -15.7%
- prompt=16384 (extrapolated): ~-18 to -20% (Machete threshold)

**For long-context-dominant workloads**, Option B's checkpoint format
investment (~2 weeks tooling) starts to pay off vs Option A (Medusa)
which mostly improves throughput at all context lengths.

The key strategic question becomes: **what context length distribution
does the user's "world-first" target audience have?**
- If primarily short-ctx (≤512): Medusa (Option A) clearly wins
- If primarily 2-4k context: Both A and B viable, A faster to results
- If primarily 8k+ context: Option B becomes Machete-class competitive

### §9.6 Cross-references (added)

- `bench-output/2026-05-10-w4a8-longctx-prompt4096/benchmarks.{json,csv}`
- `/tmp/w4a8-longctx-4096.log` (server log, 0 kernel failures, 1 cache demotion)
- `4e2f39a` W4A16 prompt=4096 baseline for comparison
- `b340e2c` W4A16 prompt=8192 source for §9.4 extrapolation
- SKILL `kernel-optimization` Phase 4 formula (validated EXACT MATCH at 8k)
