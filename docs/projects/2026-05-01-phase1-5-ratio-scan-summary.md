# Phase 1.5 Ratio Scan Summary

## Context

Phase 1.5 tested the first SGLang-inspired ARLE admission patch:
`decode_headroom_ratio` reserves a ratio-scaled decode tail for live decode
rows and only charges future decode growth on final prefill chunks.

The pre-patch anchor is
`docs/experience/wins/2026-04-30-bench-guidellm-longctx-32k-phase1-baseline-anchor-0464fb3e.md`.

## Longctx-32k c=4 Scan

| ratio | commit at run | result | output tok/s | peak kv_util | key signal | entry |
|---:|---|---|---:|---:|---|---|
| 5% | `b2c1f62c` | hang | n/a | 97.0% | `active=4`, `decode_rows=3`, `prefill_rows=1`, `tokens_out=1032` | `docs/experience/errors/2026-04-30-bench-guidellm-longctx-ratio05-hang-b2c1f62c.md` |
| 10% | `a7cfcb1e` | regression | 0.53 | 99.0% | one c=4 completion, 501 incomplete output tokens | `docs/experience/errors/2026-04-30-bench-guidellm-longctx-ratio10-regression-a7cfcb1e.md` |
| 15% | `cf1d3b81` | hang | n/a | 96.4% | `active=4`, `decode_rows=3`, `prefill_rows=1`, `tokens_out=1032` | `docs/experience/errors/2026-05-01-bench-guidellm-longctx-ratio15-hang-cf1d3b81.md` |
| 20% | `4b2e563a` | hang | n/a | 96.4% | `active=4`, `decode_rows=3`, `prefill_rows=1`, `tokens_out=1032` | `docs/experience/errors/2026-05-01-bench-guidellm-longctx-ratio20-hang-4b2e563a.md` |

Baseline anchor:

| metric | value |
|---|---:|
| c4 output tok/s mean | 3.02 |
| c4 valid runs | 2/3 |
| c4 no-success runs | 1/3 |
| SGLang c4 reference | 16.27 |
| ARLE/SGLang ratio | 0.186x |

## Interpretation

- The ratio-only patch is rejected. It does not clear the entrance gate and is
  worse than the already-weak baseline.
- 5%, 15%, and 20% share the same final state: c=4 longctx requests remain
  active with mixed decode+prefill rows and do not drain.
- The failure is not only absolute pool fullness. The hang reproduces at
  `95-97%` peak KV utilization, below the 99% edge seen in some baseline and
  10% runs.
- `split=0` across the scan means the scheduler still does not expose the
  SGLang-like split/retraction behavior needed to keep decode moving while
  admitting long prefill chunks.

## Decision

Revert the ratio-only runtime patch before implementing the next policy. The
next implementation should follow the SGLang research gap table more directly:

- admission budget must reason about `free + evictable`, not free pages alone;
- decode progress and retraction pressure must feed back into admission;
- mixed decode+prefill batches need an explicit drain/retry rule when active
  rows stop making progress;
- mixed-mode benchmarks should wait until this replacement policy exists,
  because the current ratio-only patch cannot pass the simpler longctx c=4
  gate.

## Follow-Up Patch Direction

The next patch should not be another static ratio scan. It should change the
admission invariant:

1. Compute prefill growth against available KV pages plus evictable prefix
   pages, with eviction cost visible in the budget.
2. Reserve decode progress using live request state and recent retraction/hang
   pressure, not only a fixed clipped-tail percentage.
3. If mixed decode+prefill overlap reaches active non-drain state, stop new
   prefill admission and prefer decode-only progress until rows drain.
4. Then rerun longctx c=4 plus the required mixed-mode matrix.
