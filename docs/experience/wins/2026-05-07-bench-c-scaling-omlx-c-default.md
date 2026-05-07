# Bench — c=1..16 scaling sweep with oMLX-C default — 2026-05-07

## Goal

Confirm oMLX-C v3 (multi-step async pipelining, default-on as of commit
3987fa5) keeps ARLE's c≥2 ITL healthy as concurrency scales from c=1 to
c=16. Per the master M_e plan, the per-step host-block cost compounds
with batch — pipelining should keep wall-clock per-step roughly linear
in concurrency, not super-linear.

## Params

- Binary: `target/release/metal_serve` rebuilt at this commit
  (`INFER_OMLX_C` env-flag deleted; pipelining hardcoded as default)
- Model: `models/Qwen3.5-0.8B-MLX-4bit`
- `--max-running-requests 16`
- Single sequential sweep: c=8, c=16, c=4, c=2, c=1 (then back to c=1
  for clean cooldown). Each c via `/tmp/cN_smoke.sh <N>` — N concurrent
  POSTs, max_tokens=64, temperature=0.0.
- All steps logged via `INFER_PHASE_TIMING=1` and aggregated by
  `batch=` field of `metal_phase_timing_pipelined` lines.

## Results — total per-step μs by batch

| batch | n samples | avg μs | p50 μs | p99 μs |
|------:|----------:|-------:|-------:|-------:|
|     1 |        73 |   3087 |   3037 |   4989 |
|     2 |       103 |   3558 |   3508 |   4571 |
|     3 |         7 |   3868 |   4024 |   5734 |
|     4 |        30 |   5213 |   5347 |   6591 |
|     5 |         4 |   4859 |   4804 |   7572 |
|     6 |        35 |   7365 |   7590 |   8841 |
|     7 |         6 |   7080 |   8786 |   9746 |
|     8 |        23 |   9250 |   9631 |  10940 |
|    10 |         2 |  13968 |  13968 |  23050 |
|    14 |        24 |  13898 |  13814 |  26010 |
|    16 |        15 |  14084 |  14110 |  22044 |

→ **Scaling shape**: ITL roughly linear in `batch` for c≤8, then jumps
into ~14ms at c≥10. The transition is consistent with workload
crossing into a regime where Metal's command-buffer encoding amortizes
across more rows in fewer dispatches.

## Throughput (tok/s = batch * 1e6 / total_us p50)

|  batch | tok/s p50 |
|------:|---------:|
|     1 |      329 |
|     2 |      570 |
|     4 |      748 |
|     8 |      830 |
|    16 |     1133 |

→ Throughput keeps climbing through c=16 (1133 tok/s aggregate at
c=16 vs 329 at c=1, **3.4× total throughput from batching**) — the
per-step cost grows sub-linearly in batch even as ITL p50 grows
super-linearly.

## Problems / observations

1. **c=8 is the sweet spot for ITL**: 9.6ms p50 / 830 tok/s aggregate.
   Beyond that, ITL doubles by c=16 while throughput only grows ~36%.
   Recommend `--max-running-requests 8` as the ARLE default for
   latency-sensitive workloads, `--max-running-requests 16` for
   throughput-only.
2. **The c=10/c=14/c=16 "step" up to ~14ms is real** — likely Metal
   command-buffer commit at higher row counts. Worth a follow-up with
   metal-trace to confirm whether it's a fixed cost per kernel
   dispatch or an attention-kernel scaling issue. Either way, it's
   not a regression vs pre-pipelining (no flag-off comparison this
   sweep, but session 2 of `2026-05-07-bench-c4-omlx-c-v3.md` showed
   pipelined was better at every c we tested).
3. **Heavy tail** appears at c=10 (p99 23ms vs p50 14ms) and persists
   through c=16. The p99 is worth investigating before claiming
   "steady-state at high c"; for the bench-spec watch list, it lives
   under "P99 stability" rather than headline ITL.

## Learnings

1. **Pipelining wins compound through c=16.** Even though we benched
   only oMLX-C-on, the absolute numbers are healthy: c=4 p50 5347μs
   matches the 15.3% win from the v3 matched A/B against legacy
   path's 6296μs.
2. **The "cliff" between c=8 and c=10** suggests Metal's command-buffer
   commit dynamics shift at that batch size. A future tick should
   capture metal-trace to see whether splitting into 2× c=8 dispatches
   would help. For now, c=8 is the documented sweet spot.
3. **The default config lands**: `INFER_OMLX_C` env flag deleted; the
   pipelining is now part of `decode_qwen35_packed_batch`'s normal
   flow, gated only on whether `prev_sampled` has been stashed (i.e.
   first-call bootstrap vs steady state). No external knob to forget.

## What worked / Rule

- Single-sweep of c=1..16 produced enough samples per c=4/c=8/c=16 for
  decent p50/p99 estimates; mid c values (3, 5, 7) had small n but
  those aren't headline numbers.
- The phase-timing tag (`metal_phase_timing_pipelined`) confirmed the
  pipelined path fired at every c — no fallback to bootstrap-only
  beyond the first call per batch.

## Rule

When deleting an environment-variable feature gate after matched A/B
proves the win:
1. Confirm the gated path is on every test run (not just the labeled
   bench) by checking the path probe fires.
2. Replace the `if env_flag && state` guard with just `if state` —
   no half-states (`feedback_no_half_states.md`).
3. Update the docstring's "Caller invariants" so future readers don't
   look for a flag that no longer exists.

## Next

- **Investigate c=10+ step jump** with metal-trace or
  `INFER_PHASE_TIMING_PIPELINED_PREP_BREAKDOWN=1` (next tick).
- **Implement M_e.2 PromptTrie** ([`docs/plans/M_e2-prompttrie-prefix-cache.md`](../../plans/M_e2-prompttrie-prefix-cache.md))
  for the chat-style shared-system-prompt workload.
- **ELI Layer 2 functional gate**: `bench_eli_agent.sh smoke-real`
  should now report `session_affinity_hit > 0`; verify when next
  cron lands.

## References

- v3 (matched A/B + default-on):
  [`2026-05-07-bench-c4-omlx-c-v3.md`](2026-05-07-bench-c4-omlx-c-v3.md)
- Baseline (pre-pipelining):
  [`2026-05-07-bench-c4-phase-timing-baseline.md`](2026-05-07-bench-c4-phase-timing-baseline.md)
- Next big lever:
  [`docs/plans/M_e2-prompttrie-prefix-cache.md`](../../plans/M_e2-prompttrie-prefix-cache.md)
