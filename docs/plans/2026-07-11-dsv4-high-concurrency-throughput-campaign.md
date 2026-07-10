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
5. **Pinned DRAM for the L2 pool** — capture/restore D2H uses pageable host
   memory (blocking staging copy). Page-locked host buffers cut the
   capture/restore stall → TTFT (multi-turn) + throughput (less engine stall
   under concurrency). Opt-in reuse path; measure the restore-latency Δ.
6. **Aggressive admission watermark** — the band demand-paging admission
   watermark is conservative → under-admits slots → caps concurrency. Loosen
   (behind a knob first; default-path so needs multi-shape verify): higher slot
   count → higher aggregate throughput without OOM. Measure slot count +
   throughput at c=16.

## Non-goals
- Page-granular mid-generation reuse (deferred — bigger, concurrent-share case).
- #150 substitution (independent near-tie noise, separate track).

## Blocker (current)
The pod is contended: a sibling `fp8probe` build LOOP owns `/host/arle-build`
(new build every ~2 min, 35 compiler procs) — no clean window to build without
clobbering it. GPUs are free (our leftover TP=8 serve was killed). The campaign
measurements queue until the build tree frees.
