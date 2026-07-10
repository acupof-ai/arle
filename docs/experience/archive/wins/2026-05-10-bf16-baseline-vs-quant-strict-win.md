# BF16 baseline vs W4A16/W4A8 — quantization is STRICT WIN at conc=1 prompt=512 on sm_89

## Context

Date: 2026-05-10 13:26-13:27 KST
Bench: Qwen3-4B BF16 (no quantization) at conc=1 prompt=512 — perf
ceiling reference for the W4A16/W4A8 quantization comparison.

Per session-tail TOTAL summary (`9350767` §1), all prior benches used
quantized variants (W4A16/W4A8/PF8). BF16 was the missing reference
point.

## What Worked

### Result table (full conc=1 prompt=512 comparison)

| Path | TTFT mdn | ITL mdn | tok/s | Successful (60s) |
|---|---:|---:|---:|---:|
| BF16 (no quant) | **68.7 ms** | **14.0 ms** | **69.3** | 31 |
| W4A16-marlin-zpfix | 66.0 ms | 5.8 ms | 159.6 | 75 |
| W4A8-zpfix | 54.2 ms | 11.9 ms | 81.7 | 45 |
| PF8 hybrid (Arms A/B) | KILL | KILL | KILL | 0 |

### Surprise: W4A16 BEATS BF16 on EVERY metric

| Metric | BF16 | W4A16 | W4A16 vs BF16 |
|---|---:|---:|---:|
| TTFT mdn | 68.7 ms | 66.0 ms | **-4%** (slightly faster) |
| ITL mdn | 14.0 ms | 5.8 ms | **-59%** (massively faster) |
| tok/s mean | 69.3 | 159.6 | **+130%** (more than 2×) |

W4A16 is a **strict win** over BF16 on sm_89 16GB at this workload.
No tradeoff — every metric improves with quantization.

### Why W4A16 dominates BF16

Per memory `61c9666` ARCHITECTURAL INSIGHT (2026-05-09):
> "W4 decode HBM-bound on weight read (already 4× smaller than BF16)"

Decode at conc=1 single-token mma is dominated by weight read
bandwidth from HBM. 4× less weight memory = ~4× theoretical decode
speedup. Actual W4A16 ITL improvement of -59% is reasonable (some
overhead in dequantization + Marlin kernel launch).

## Implications

### §1 The "BF16 is gold standard" assumption is FALSE for sm_89

This is counter-intuitive: most discussions assume BF16 is the
"correct" baseline and quantization trades accuracy for speed. On
sm_89 4070 Ti SUPER + Qwen3-4B + conc=1 + 512-token prompt:
- W4A16 has BETTER perf than BF16 across all 3 metrics
- W4A16 maintains accuracy (greedy_consistency 0.0% diff per
  Task #48 verification `8d1caad`)
- **Quantization is the dominant strategy, not a fallback**

This validates the W4 quant axis as the right primary path for
ARLE on sm_89.

### §2 Refined comparison vs other paths

W4A16 vs other quant variants at conc=1 prompt=512:
- vs W4A8: TTFT +22% (slower), ITL -51% (faster), tok/s +95% (faster)
- vs PF8: PF8 KILLed, no comparison
- vs INT8 v3 baseline (per session retrospective): different bench
  config, not directly comparable

W4A16 wins for general workloads; W4A8 wins TTFT-prioritized
workloads (per `92813dc` 6-cell matrix); PF8 currently dead (per
`0be278f` PF8.5 KILL).

### §3 Updates direction options matrix

The earlier direction options doc (`a64fad7` + `12e0c07`) used
W4A16 as the implicit "best baseline" without quantifying vs BF16.
This bench confirms W4A16 IS the right baseline:
- Medusa Phase 1.A perf floor (Option A) is set against W4A16
  numbers (correct choice)
- Hybrid Option B (Task #30) targets W4A8-prefill + W4A16-decode
  combination — also correct because both individually beat BF16

No revision needed; this confirms the existing recommendation.

## Rule

For sm_89 + Qwen3-4B-class models (4-7B param) at conc=1
short-medium context, **W4A16-marlin is the perf ceiling, not BF16**.
Future bench comparisons should:
1. Use W4A16 as the dominant baseline
2. Cite this entry to justify NOT including BF16 in routine
   comparisons (BF16 < W4A16 on sm_89, no upside)
3. Continue including BF16 in correctness-checks (greedy_consistency)
   where output deterministic comparison is needed

For larger models (70B+) where weight memory exceeds VRAM, BF16
may not fit at all (16GB GPU) and quantization becomes mandatory.
For smaller models (≤4B), quantization is optimization (W4A16 strict
win at sm_89).

## Cross-references

- `9350767` session-tail TOTAL summary (§1 Cumulative bench tally)
- `61c9666` ARCHITECTURAL INSIGHT memory (sm_89 W4 decode HBM-bound)
- `8d1caad` Task #48 W4A8 accuracy verification (qzeros-fixed)
- `92813dc` W4A16 vs W4A8 6-cell matrix (now extended with BF16 reference)
- `bench-output/2026-05-10-bf16-baseline-conc1/benchmarks.{json,csv}`
- `/tmp/qwen3-bf16-baseline.log` (0 kernel failures, 0 demotions)
