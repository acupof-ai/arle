# Metal MTP Observability

## Context

The SGLang survey made one thing non-optional: speculative decode results are
not interpretable from latency alone. Acceptance length, acceptance rate, and
fallback shape must be visible next to TTFT, TPOT, and throughput.

ARLE's Metal MTP path already logged per-request cleanup details, but the
server stats and `metal_bench` artifacts did not carry enough MTP counters to
explain the 2026-06-02 long-output regressions. That created a risk of making
performance claims from wall-clock tables without knowing whether MTP was
actually accepting useful draft suffixes or falling back through scalar rows.

## What Worked

- Added server-level Metal MTP counters for block count, accepted input sum,
  draft input sum, suffix acceptance rate, utilization, and scalar-row count.
- Flushed per-request MTP block stats into `ServerMetrics` at request
  completion, preserving the current scalar-only scheduler behavior.
- Exposed the same counters through Prometheus text, `/v1/stats?format=json`,
  and the human summary line.
- Taught `metal_bench` to emit `[mtp]` summaries and JSON fields for block
  count, block size, average accepted inputs, and suffix acceptance rate.
- Kept this as observability only: no decode math, scheduler selection,
  default enablement, or packed verify behavior changed.

## Verification

- `cargo test -p infer --no-default-features --features no-cuda server_metrics_ -- --nocapture`
- `cargo check -p infer --no-default-features --features no-cuda`
- `cargo check -p infer --no-default-features --features metal --bin metal_bench`
- `cargo fmt --check`
- `git diff --check`

No performance benchmark is claimed for this tranche because it does not alter
the decode hot path. The next MTP performance run should use the control plan's
long-output matrix and report these acceptance counters alongside latency.

## Rule

MTP performance evidence must report acceptance and scalar fallback next to
latency. A speculative decode speedup without acceptance counters is not a
licensed win.
