# DSv4 prefill chunk 128→2048 default — c1 TTFT −64%, c16 TTFT p50 −87% — CUDA, 2026-07-17

> Status: Shipped

## Verdict

Default flip LICENSED (plan Phase 3,
plan). Every
DSv4 prefill tick was hard-capped at 128 tokens (planner one-unit cap ×
sliding_window alignment; the CLI flag was cosmetic). Effective chunk is now
2048 by default (`ARLE_DSV4_PREFILL_CHUNK` overrides both ways).

## Gates (4×H20 TP=4/EP=4, cap32 serve, bench-prompts-64 ~2.8k-tok docs)

- t(chunk) curve (c1 cold TTFT p50): 128→3031ms · 512→1479 · 1024→1208 ·
  2048→1088 · 4096→1018. Monotone; flattens past 1024 (~1.0s residual floor
  — next attribution target).
- Needle: both depths ×2 passes (incl. prefix-restore lane) zero-miss at
  chunk 2048; partials identical on the 128 control (baseline behavior).
- Final c-sweep at the flipped default (needle 9/9 exact):

| c | complete | TTFT p50/p99 ms | ITL p50/p99 ms |
| --- | --- | --- | --- |
| 1 | 14 | 1093 / 1133 | 21.6 / 42.0 |
| 4 | 28 | 1794 / 4205 | 44.0 / 89.6 |
| 16 | 64 | 5855 / 16026 | 71.2 / 120.9 |
| 32 | 96 | 4519 / 4558 | 133.0 / 191.5 |

vs 128-era controls: c16 TTFT p50 −87%, ITL p99
120.9 vs 416ms (**better** — prefill queueing, not chunk stalls, dominated
ITL). The 2026-05-25
ITL-kill precedent inverts at long-prompt shapes.

## Notes

- SW ring same-slot write race at chunk>window fixed host-side
  (`sw_ring_tail_slice`, 0e9f687f8) — plausible co-conspirator in historical
  >2048 needle failures.
- c16 TTFT p50 varied 2.4× between same-code runs (2403 vs 5855ms) —
  arrival/cache-state sensitivity, flagged unattributed.
- prefix_reuse.py harness broken on all arms (#166, pre-existing).
- Raw: pod `bench-output/2026-07-17-p2-*`, `2026-07-17-p3-sweep/`.
