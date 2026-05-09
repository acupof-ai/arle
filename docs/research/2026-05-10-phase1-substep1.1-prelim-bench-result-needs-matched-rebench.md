---
title: Phase 1 Substep 1.1 prelim bench — STRIKING ITL -36% signal but workload-size mismatch confounds (needs σ-tight re-bench)
date: 2026-05-10
type: research
status: prelim-result-pending-matched-control-rebench
---

# Phase 1 Substep 1.1 prelim bench — STRIKING ITL -36% signal but workload-size mismatch confounds (needs σ-tight re-bench)

> Codex ran W4A16 4k/c=4 regression bench post Phase 1 Substep 1.1
> dequant.h port (`bench-output/2026-05-10-path-b-p1-newdequant-r1`).
> Single-arm n=1 vs 2026-05-09 W4A16 n=3 baseline shows striking ITL
> improvement, but workload-size differs 4.2×. Per skill v1.10.0 #5
> (wall-clock framing) + #28 (raw evidence): document the signal,
> flag the confound, recommend σ-tight matched re-bench before
> license claim.

## §0 Direct evidence (raw `head` on bench output files this tick, NOT memory recall)

### newdequant-r1 (Phase 1.1 treatment)

```bash
$ head -3 bench-output/2026-05-10-path-b-p1-newdequant-r1/headline_table.md
| rate | TTFT mean | TTFT std | TTFT p50 | TTFT p99 | TPOT mean | ITL mean | ITL std | ITL p50 | ITL p95 | ITL p99 | ITL max | E2E mean | E2E p99 | conc p50 | out tok/s | total tok/s | in tok/s | total in | total out | req/s actual |
| conc4 | 2453.5 | 94.2 | 2386.3 | 2574.3 | 20.93 | 11.39 | 0.04 | 11.38 | 11.5 | 11.51 | 11.51 | 5.36 | 5.48 | 4 | 195.17 | 3318.71 | 3208.05 | 344148 | 21504 | 0.764 |
```

### baseline B5 W4A16 c=4 4k (2026-05-09, n=3)

```bash
$ head -3 bench-output/2026-05-09-baseline-B5-w4a16-c4-4k-r1/headline_table.md
| conc4 | 2423.7 | 42.6 | 2415.7 | 2486.5 | 27.17 | 17.77 | 0.03 | 17.76 | 17.82 | 17.89 | ...

$ head -3 .../baseline-B5-w4a16-c4-4k-r2/headline_table.md
| conc4 | 2394.3 | 27.4 | 2387.6 | 2450.7 | 27.06 | 17.77 | 0.03 | ...

$ head -3 .../baseline-B5-w4a16-c4-4k-r3/headline_table.md
| conc4 | 2343.9 | 12.6 | 2347.5 | 2360.3 | 26.85 | 17.77 | 0.03 | ...
```

### Direct comparison (n=3 baseline mean vs newdequant n=1)

| Metric | Baseline mean (n=3) | newdequant n=1 | Δ | Δ% |
|--------|--------------------:|---------------:|---:|---:|
| TTFT mean (ms) | (2423.7 + 2394.3 + 2343.9) / 3 = 2387.3 | 2453.5 | +66.2 | **+2.8% regression** |
| TTFT p50 | (2415.7 + 2387.6 + 2347.5) / 3 = 2383.6 | 2386.3 | +2.7 | flat |
| ITL mean (ms) | 17.77 (all 3 runs) | 11.39 | **-6.38** | **-35.9% ⭐** |
| ITL p50 | 17.76 | 11.38 | -6.38 | -35.9% |
| TPOT mean (ms) | (27.17 + 27.06 + 26.85) / 3 = 27.03 | 20.93 | -6.10 | **-22.6% ⭐** |
| Throughput out tok/s | (158.01 + 158.56 + 159.74) / 3 = 158.77 | 195.17 | +36.4 | **+22.9% ⭐** |
| E2E mean (s) | (6.96 + 6.93 + 6.87) / 3 = 6.92 | 5.36 | -1.56 | **-22.5% ⭐** |
| **total in toks** | **81940** | **344148** | **4.2× MORE** | **WORKLOAD CONFOUND** |
| **total out toks** | **5120** | **21504** | **4.2× MORE** | **WORKLOAD CONFOUND** |

## §1 The confound

newdequant-r1 ran 4.2× more total requests/tokens than baseline. This
likely means a longer wall-clock window or different `--max-seconds`/
`--data` config. Confounders this introduces:

1. **Warmup amortization**: longer runs amortize one-time setup costs
   (cudagraph capture, KV pool warmup, prefix cache fill) better, so
   per-token metrics improve mechanically without any kernel change.
2. **Steady-state vs transient**: longer windows reach steady-state
   batch occupancy, while shorter ones are transient-dominated.
3. **Cache state**: longer run accumulates prefix cache, may show
   higher hit rate even on "random" prompts due to repeated structure.

Per skill v1.10.0 #5 wall-clock framing: cannot license a -35% ITL
claim from this data alone. The TTFT regression (+2.8%) within
baseline σ (~50ms) is consistent with same kernel; the dramatic
ITL improvement is more likely an artifact than a real win.

## §2 What this evidence DOES support

- **Substrate works**: 21504 tokens generated, 0 errors. dequant.h
  port (09ae5a5 + 994a294) is functionally correct at production
  workload size.
- **Numerical stability**: ITL std 0.04ms is extremely tight (single
  run!), suggesting the dequant function is deterministic and
  consistent across requests.
- **No catastrophic regression**: TTFT +2.8% within σ noise; not a
  KILL signal.

## §3 What's NEEDED before license claim

Per kernel-optimization skill v1.10.0 Phase 5 + Phase 8 + skill #5
(wall-clock framing):

**Re-run newdequant at SAME workload size as baseline (n=3, σ-tight)**:
- Same `--data` spec
- Same `--max-seconds` window
- Same total token count (≈81940 in / 5120 out per run)
- 3 runs, compute mean + σ
- License gate: ITL Δ ≥ -3% with σ < 5% across n=3

If matched re-bench shows ITL Δ ≥ -3% (even modest): license per
e59beb5 conservative -3-8% estimate.
If matched re-bench shows ITL Δ near 0 or regression: confirms current
1.1 result was workload-confound artifact, mark Phase 1 as
correctness-only (no perf license, but substrate landed for future
upstream cherry-picks).

## §4 Suggested codex pickup brief (post current investigation)

```
Codex pickup: Phase 1.1 σ-tight matched re-bench (3 runs)

Current bench bench-output/2026-05-10-path-b-p1-newdequant-r1 has
4.2× more workload than baseline-B5 n=3, confounding ITL comparison.

Re-run at MATCHED workload size:
  --data 'prompt_tokens=4096,prompt_tokens_stdev=512,output_tokens=128,output_tokens_stdev=32'
  --max-seconds 60     (matches baseline-B5 windowing)
  --warmup 10
  --concurrencies 4

Run 3 times, compute mean + σ, write wins entry citing both:
  Baseline n=3 mean (2026-05-09 historical)
  Treatment n=3 mean (this re-bench)
  Δ% per metric (TTFT, ITL, TPOT, throughput)

License if ITL Δ ≥ -3% with σ < 5%.

Use prior server-restart pattern (setsid bash -c 'exec ...' per
4b30c15 unstick + correct ARLE endpoints per c3bb82b: /readyz NOT
/health).
```

## §5 Pattern lesson (sediment for future bench discipline)

**Workload-size matching matters more than concurrency matching for
per-token metrics** (ITL, TPOT). When comparing benches:
- Same concurrency ✓ (already standard)
- Same data spec (prompt/output token counts) ✓ (already standard)
- Same total token count / wall-clock window — ✗ often forgotten

If newdequant-r1 had used `--max-seconds 60` matching baseline-B5,
the comparison would be apples-to-apples. The 4.2× workload size
suggests codex either:
(a) Used different `--max-seconds`
(b) Used different `--data` total tokens
(c) Server warmup window differed

Check codex's `command.txt` if needed to nail down the cause.

## §6 Cross-references

- Phase 1.1 substrate (codex 09ae5a5 + Claude 994a294 build-restore):
  `crates/cuda-kernels/csrc/gemm/marlin_dequant.cuh` (651 LOC)
- e59beb5 Phase 1 plan: conservative -3-8% ITL estimate (this result
  -36% is 4-12× larger if real, suggesting confound)
- 09ae5a5 strategic revision: real near-term wins = Phase 1 + prefill-only FP8
- 2026-05-09 baseline-B5 W4A16 c=4 4k n=3 dirs (this entry §0)
- newdequant-r1 dir: `bench-output/2026-05-10-path-b-p1-newdequant-r1`
- Skill v1.10.0 #5 wall-clock framing, #28 verify raw output

## §7 Status

Prelim bench result captured (raw evidence quoted §0-1). Striking
ITL signal -36% but workload-size confound. NOT licensed yet.
Recommended σ-tight matched re-bench (n=3 at baseline-B5 workload
size) before any wins claim. Per skill v1.10.0 #5 + #28: every
number in §0 verified by raw `head` on bench `headline_table.md`
files, NOT memory recall.

If matched re-bench confirms ≥-3% ITL: PASS Phase 1 with conservative
modest win (matches e59beb5 estimate). If matched re-bench shows
near 0 or regression: Phase 1 correctness-substrate only (still
useful for future cherry-picks). Either way, no immediate -36%
claim until verified.
