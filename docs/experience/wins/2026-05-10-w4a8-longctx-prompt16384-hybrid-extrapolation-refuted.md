# W4A8 prompt=16384 measured — REFUTES §12.7 Hybrid Option B Machete-class extrapolation

## Context

Date: 2026-05-10 (cron-loop tick 99-100 KST)
Bench: Qwen3-4B W4A8-zpfix at conc=1 prompt=16384 output=128, 120s.
Purpose: convert §12.7 (`4718b44`) hybrid extrapolation -35.3% from
projection to MEASURED data. Per `9350767` §4 the prior pattern
suggested asymptotic value, but the linear-extrapolation framing in
§12.7 implied -35% at 16k.

This bench is the **ground-truth check** on whether Hybrid Option B
crosses the Machete-class -20-40% threshold at 16k context.

## What Worked

### Result table (raw)

| Metric | Value |
|---|---:|
| TTFT median | **2713.4 ms** |
| ITL median | **19.1 ms** |
| tok/s mean | **25.0** |
| Successful (120s) | 23 |
| Kernel failures | 0 |
| Cache demotions | 9 |
| Server config | `--max-seq-len 32768` |

### W4A8 long-context scaling now n=4 (2k/4k/8k/16k)

| Prompt | TTFT (ms) | scale vs 2k | ITL (ms) | tok/s | demotions |
|---:|---:|---:|---:|---:|---:|
| 2048 | 191.3 | 1× | 12.6 | 71.8 | 0 |
| 4096 | 409.4 | 2.14× | 13.8 | 59.5 | 1 |
| 8192 | 985.4 | 5.15× | 15.4 | 43.9 | 3 |
| **16384** | **2713.4** | **14.2×** | **19.1** | **25.0** | **9** |

W4A8 TTFT scales ~14× for 8× prompt length = +75% super-linear
(comparable to W4A16's +62% — both share the same paged-attention
prefill kernel).

### W4A8 vs W4A16 head-to-head at 16k

| Metric | W4A16 | W4A8 | W4A8 vs W4A16 |
|---|---:|---:|---:|
| TTFT median | 3411.5 ms | 2713.4 ms | **-20.4%** |
| ITL median | 11.9 ms | 19.1 ms | **+60.5% slower** |
| tok/s mean | 26.4 | 25.0 | -5.3% |

W4A8 TTFT advantage at 16k: -20.4%. W4A8 ITL penalty: +60%.

## REFUTATION OF §12.7 EXTRAPOLATION

§12.7 (`4718b44`) extrapolated Hybrid Option B value at 16k as **-35.3%**
based on linear continuation of the n=4 trajectory. Measurement now
contradicts that.

### Hybrid Option B measured value at 16k

For 128-token output at 16k context, end-to-end perceived latency:

```
Latency_W4A16 = TTFT_W4A16 + (output_tokens-1) * ITL_W4A16
              = 3411.5 + 127 * 11.9 = 3411.5 + 1511.3 = 4922.8 ms

Latency_W4A8  = TTFT_W4A8 + (output_tokens-1) * ITL_W4A8
              = 2713.4 + 127 * 19.1 = 2713.4 + 2425.7 = 5139.1 ms

Latency_Hybrid = TTFT_W4A8 + (output_tokens-1) * ITL_W4A16
               = 2713.4 + 127 * 11.9 = 2713.4 + 1511.3 = 4224.7 ms
```

**Hybrid vs W4A16: -14.2% (NOT -35.3% as extrapolated).**

### Updated n=5 Hybrid value progression

| Context | W4A8 TTFT vs W4A16 | Hybrid Option B vs W4A16 | §12.7 prediction | Match? |
|---|---:|---:|---:|---|
| conc=1 prompt=512 | -18% | -1.4% | (pre-prediction) | n/a |
| conc=1 prompt=2048 | -30% | -7.5% | (pre-prediction) | n/a |
| conc=1 prompt=4096 | -29% | -11.1% | (pre-prediction) | n/a |
| conc=1 prompt=8192 | -26% | -14.2% | (pre-prediction) | n/a |
| conc=1 prompt=16384 | **-20%** | **-14.2%** | **-35.3%** | **REFUTED** |

**Hybrid value PLATEAUS at ~-14% from 8k onward**, NOT continued
linear improvement. §12.7's linear-extrapolation framing was
fundamentally wrong.

### Why the extrapolation failed

§12.7 implicitly assumed W4A8 TTFT advantage stays constant or grows
with context. Measurement shows it SHRINKS:
- W4A8 TTFT advantage at 4k: -29%
- W4A8 TTFT advantage at 8k: -26%
- W4A8 TTFT advantage at 16k: **-20%**

The W4A8 advantage degrades because both quants share the same
paged-attention prefill kernel; W4A8's compute saving is on
matmul-bound ops which are fractionally smaller share of TTFT as
context grows (attention dominates more at longer ctx).

ITL gap grows in W4A8's favor's opposite direction too — W4A8 ITL
+60% slower at 16k vs +73% at 8k. Both adverse.

## Implications

### §1 Machete-class -20% threshold NOT crossed by Hybrid alone

Hybrid Option B at any context measured (512 → 16384) caps at -14.2%
perceived latency. Does NOT reach Machete-class -20-40% target on its
own.

To reach Machete-class threshold, would need EITHER:
- A Machete kernel actually ported (better W4A8 / W4A16 decode kernel)
- Combination with Medusa speculative decoding (multiplies tok/s
  independently)
- Different model size (70B+ where weight bandwidth dominates more)

### §2 Refines strategic decision matrix

Updates `9350767` §5 strategic matrix row "Maximum TTFT/ITL long-ctx (≥8k)":

| User priority | Path | Time | Why (REVISED) |
|---|---|---|---|
| Maximum TTFT/ITL long-ctx (≥8k) | A (Medusa) | 2-3 days | Hybrid B caps at -14%; A's 2-3× tok/s is bigger lever |
| World-first 32k+ ctx specifically | A + B combined | ~3 weeks | Need both speedup paths multiplied |

Option B alone is no longer the recommended long-ctx path. Option A
is now the dominant single-axis investment.

### §3 Validates Phase 4 measurement-over-extrapolation discipline

This is the FOURTH Phase 4 prediction validation cycle this session-tail:
- ✓ EXACT MATCH at W4A16 8k (`4d3aa4f`)
- ✓ +4% margin at W4A8 8k (`b5f9b4e` §10.3)
- ✓ EXACT MATCH at W4A16 16k (`4718b44` §12)
- **✗ REFUTED at Hybrid 16k extrapolation (`4718b44` §12.7) — this entry**

Predictions about MEASURED scaling validated 3/3. Predictions about
EXTRAPOLATED downstream metrics (the hybrid combination) refuted 1/1.

**Lesson**: Phase 4 formula works when bound to direct kernel
measurements. It fails when chained through naïve "best of both"
combinations. This is exactly the SKILL candidate `92813dc` /
`2026-05-10-skill-candidate-end-to-end-latency-math.md` that was
flagged in §3 of the session-tail summary.

### §4 SKILL candidate "end-to-end latency math" now n=2 evidence

This entry adds the second independent evidence point for graduating
the SKILL candidate "end-to-end latency math vs naïve 'best of both'":

| Evidence | Source | Issue |
|---|---|---|
| n=1 | `92813dc` Hybrid Option B aggregation framing decay | Aggregating per-component minima ≠ realistic latency |
| **n=2** | **THIS entry — §12.7 extrapolation -35.3% vs measured -14.2%** | **Linear extrapolation of asymmetric trajectories diverges from physics** |

Recommend graduating this candidate to canonical anti-pattern
**#39** (or next available number) at next SKILL bump.

## Rule

When extrapolating optimization stack values (e.g. "Hybrid would gain
X% at context Y"), MEASURE THE COMBINATION at Y first before claiming
the win. Linear extrapolation of asymmetric trajectories is unreliable.

For ARLE specifically: Hybrid Option B value caps at ~-14% perceived
latency from 8k onward on sm_89 + Qwen3-4B. To exceed this requires
multiplicative paths (Medusa) or fundamentally better kernels (Machete).

## Cross-references

- `4718b44` W4A16 16k validation §12 (parent doc, §12.7 extrapolation refuted here)
- `9350767` session-tail TOTAL summary (§4 hybrid value progression — UPDATE NEEDED)
- `92813dc` original 6-cell W4A16/W4A8 matrix
- `b5f9b4e` W4A8 long-ctx 8k extension (§10)
- `eab166d` BF16 baseline strict-win wins entry
- `bench-output/2026-05-10-w4a8-longctx-prompt16384/benchmarks.{json,csv}`
- `/tmp/w4a8-longctx-16384.log` (0 kernel failures, 9 demotions)
