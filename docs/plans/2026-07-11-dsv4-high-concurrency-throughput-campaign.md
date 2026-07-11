# DSv4 high-concurrency agent-workload throughput campaign

> Status: Active

**Goal (ckl 2026-07-11)**: maximize **TTFT · TPOT · throughput** for the
high-concurrency agent workload on DSv4-Flash-FP8. Each optimization is
LICENSED only by a `scripts/bench_guidellm.sh` sweep at production concurrency
(agent-shape prompts, c-sweep incl. c≥4) showing a TTFT/TPOT/throughput win —
per-metric, vs the current-best baseline. Order: **1 → 2 → 3 → 5 → 6**.

## Measurement (the license gate)
- Harness: `scripts/bench_guidellm.sh` (canonical), agent-shape prompt mix,
  c-sweep {1,4,8,16}. Report TTFT p50/p99, ITL/TPOT p50/p99, output tok/s.
- Each optimization: matched-baseline A/B (flag/knob OFF vs ON, same binary,
  same boot config), Δ% per metric. A flip/default needs ≥2 binding shapes.
- Pod-gated: all measurement runs on the H20 (TP=4, the session-standard).

## Optimizations (in order)

1. **Reuse hit-rate — `adopt_canonical` frontier-tail preservation** (#159,
   LANDED `372365ef4`). A continuation after radix dedup was losing the sub-page
   tail → less decode-region reuse → more re-prefill → worse multi-turn TTFT.
   Fixed on the opt-in path. Measure: multi-turn reuse hit-rate + TTFT under
   c≥4 with `--dsv4-decode-reuse true`.
2. **E6 c=4 +3.8% residual** — the Phase-3b demand-paging regression at
   concurrency 4 (wins/2026-07-10-dsv4-band-demand-paging-phase3b). Real
   TPOT/throughput debt; unattributed. nsys-attribute (slots ruled 0.9pp;
   zeroing/growth-storm ablated out) → fix the residual. Default-path.
3. **Decode-region reuse default flip** — `--dsv4-decode-reuse` on by default,
   IF the c-sweep clears TTFT AND TPOT AND throughput on ≥2 agent shapes (the
   finish-capture D2H must not cost more than the reuse saves on short turns).
   Gated on #1+#5 landing first (they reduce the capture/restore cost).
4. **(skipped this round — page-granular mid-generation reuse, deferred.)**
5. **Pinned DRAM for the L2 pool — KILLED (bad ROI, 2026-07-11).** cudarc's
   `alloc_pinned()` is write-combined (correct for H2D, pathological for the
   D2H-then-host-read the codec needs); the correct path needs net-new unsafe
   `malloc_host(_,0)` FFI + a capture return-path refactor (the reuse capture is
   a deferred single-sync, incompatible with one reusable buffer), for a
   second-order stall win on the opt-in path. If finish-stall ever matters,
   async off-engine capture is the right design, not pinned. Not built.
6. **Aggressive admission watermark — KILLED as a knob (unsafe, 2026-07-11).**
   DSv4's concurrency ceiling is a real invariant, not a conservative guard:
   each slot reserves its full band on the host, and `band_extend` hard-errors
   on device-pool exhaustion with NO DSv4 recovery (no preempt/park/demote;
   `retract_decode_to_fit` is device-blind). A watermark<1 removes the
   safety invariant with no cascade to catch it; its benefit regime ==
   its unsafe regime. The REAL lever is the band-exhaustion recovery cascade —
   **#160** (#154 follow-on infra), not a flag.

## Verdict (2026-07-11)
#1 landed + measured (+1 page reuse, no single-shot regression). #5/#6 both
KILLED as specced — the honest throughput lever is #160 (DSv4 band-exhaustion
cascade), a real infra project. The multi-turn concurrent harness
(`eval_harness multiturn_concurrent`) is built and ready to quantify the current
high-concurrency multi-turn TTFT/TPOT/throughput + the reuse-under-concurrency win.

## Non-goals
- Page-granular mid-generation reuse (deferred — bigger, concurrent-share case).
- #150 substitution (independent near-tie noise, separate track).

## Blocker (current)
The pod is contended: a sibling `fp8probe` build LOOP owns `/host/arle-build`
(new build every ~2 min, 35 compiler procs) — no clean window to build without
clobbering it. GPUs are free (our leftover TP=8 serve was killed). The campaign
measurements queue until the build tree frees.
