# numa-pin is single-rank-inert by design

2026-08-23 · CUDA · bench (null result)

## Context

`--numa-pin` (default ON) pins each TP worker's threads to a disjoint core
slice of its GPU's NUMA node. The supporting evidence (2026-06-12 trace,
8×H20 B=1 decode) was multi-rank: unpinned ranks wander across sockets and
arrive late at collectives. Whether single-rank serves also benefit was
untested, so a same-binary A/B was run on Qwen3.6-35B-A3B-FP8, 1×H20,
fp8 KV, decode graph on, c=1/16/32, 120 s/cell.

## Result

| c | on (pin) itl_p50 ms | off (no pin) itl_p50 ms |
|---|---|---|
| 1 | 7.213 | 7.193 |
| 16 | 36.737 | 36.593 |
| 32 | 64.591 | 64.569 |

Wash at every concurrency. **But the wash is uninformative**: neither serve
log contains a `[numa-pin]` line. The pin call (`loader.rs:60`) is gated on
`!cfg.is_single()` — single-rank never enters the branch, so `--numa-pin`
is inert for single-rank serves. Both arms ran unpinned.

## Verdict

No action. The flag is multi-rank-only by design; the 2026-06-12 evidence
already supports default ON there. Single-rank decode is GPU-bound, so even
if the pin were applied, H2D/scheduler placement is unlikely to matter.
The flag stays as the multi-rank opt-out.

## Rule

Before benching a flag, grep its call site for the gate that reaches it.
A flag whose consumer is behind a `!is_single()` (or similar) gate cannot
move single-rank numbers — the A/B will wash regardless of the flag's merit.
