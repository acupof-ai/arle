# DSv4 KV tier L2-only vs L2+L3 — guidellm, TP=4, 2026-07-09

> Partial PASS: matched-A/B bounded-concurrency comparison clean (no cost);
> canonical `sweep` profile blocked by a reproduced TP=4 capacity ceiling;
> zero tier activity observed in either run, so this does not yet test L3
> under real spill pressure. Side-finding: Route A's new shared KV pools
> shrink the TP=4 slot ceiling significantly vs the pre-Route-A baseline.

## SLO-shape probed? N

Bounded-concurrency (1/4/8), 45s window, matched A/B — not the canonical
`--profile sweep`. See "Problems" for why.

## Goal

Compare DSv4-Flash-FP8 serving throughput with the KV tier system's L2
(host DRAM) tier alone vs. L2+L3 (host DRAM + NVMe disk spill), same
binary/TP/model, canonical `scripts/bench_guidellm.sh`.

## Hypothesis

Enabling an L3 spill path with no actual spill pressure should show
near-zero overhead — confirm nothing spills, so the comparison is
meaningful.

## Environment

- **Backend:** CUDA, TP=4, `INFER_CUDA_DEVICES=4,5,6,7` (GPU1 held 51 GB by
  a foreign tenant the whole session, untouched; GPU0/2/3 free, untouched).
- **Model:** DeepSeek-V4-Flash-FP8.
- **Commit:** `edb88c976`.
- **Feature set:** `cargo build --release --features cuda,nccl,deepep --bin arle`.
- **Non-default flags:** `--max-total-tokens 4608 --max-prompt-tokens 4200`
  (bisected down from 16384 — see Problems); L2+L3 arm adds bare `--kv-disk`.

## Command

```bash
scripts/bench_guidellm.sh dsv4-l2-only --concurrencies 1,4,8 --max-seconds 45
scripts/bench_guidellm.sh dsv4-l2-l3   --concurrencies 1,4,8 --max-seconds 45
```

## Results — headline table

| conc | metric | L2-only | L2+L3 | Δ% |
|---|---|---:|---:|---:|
| 1 | TTFT p50/p99 (ms) | 4979.6 / 4986.3 | 5024.1 / 5073.4 | +0.9% |
| 1 | ITL p50/p95 (ms) | 22.17 / 22.19 | 22.59 / 22.67 | +1.9% |
| 1 | TPOT p50 (ms) | 41.53 | 42.18 | +1.6% |
| 1 | out tok/s | 27.25 | 26.86 | −1.4% |
| 4 | TTFT p50/p99 (ms) | 19848.2 / 19848.8 | 20024.1 / 20024.5 | +0.9% |
| 4 | ITL p50/p95 (ms) | 45.74 / 48.77 | 46.25 / 49.49 | +1.1% |
| 4 | out tok/s | 80.23 | 79.34 | −1.1% |
| 8 | TTFT p50/p99 (ms) | 36626.3 / 36626.3 | 36950.0 / 36950.0 | +0.9% |
| 8 | ITL p50/p95 (ms) | 77.83 / 77.83 | 78.93 / 78.93 | +1.4% |
| 8 | out tok/s | 12.90 | 12.72 | −1.4% |

conc=8's `req/s actual=0`/`total out=256` in both arms: only 1 request
completed in the 45s window at this concurrency in both runs (TTFT already
exceeds the window under queueing) — identical artifact in both arms, still
a valid matched comparison, not a canonical-sweep-scale result.

## Results — service-side KV/tier stats (before → after, both arms)

| metric | L2-only | L2+L3 |
|---|---|---|
| requests_completed | 2 → 23 | 2 → 23 |
| kv_free_pages | 1560 → 1140 | 1580 → 1140 |
| resident_pages / resident_evictable_pages | 0 → 440 / 0 → 440 | 0 → 440 / 0 → 420 |
| `host_demoted_pages` | 0 → **0** | 0 → **0** |
| `disk_pages` | 0 → **0** | 0 → **0** |
| `reuse_hit_{resident,host_demoted,disk}` | all 0 | all 0 |
| L3 sparse-file real disk usage (`du`, not apparent size) | n/a | **0 bytes** (4× 103 GB sparse `kv.mmap`, fully unwritten) |

## Problems

- **Route A's new shared KV pools shrink the TP=4 slot ceiling.** On this
  HEAD, the default `--max-total-tokens 16384` (the same value that got 121
  slots on the pre-Route-A `ba36fbd39` baseline, 2026-07-06) clamps to
  `pool-band-affordable 1` slot — the new compress-state + SW-ring shared
  pools (`Dsv4CompressStatePool`/`Dsv4SwRingSnapshotPool`, always
  GPU-resident, `COMPRESS_STATE_TIER_BUDGET_BYTES` placeholder sizing) eat
  into the same VRAM budget `kv_budget_plan()` used to size KV slots.
  Bisected to `--max-total-tokens 4608` for 79 slots, used identically for
  both arms so the A/B stays matched, but this is a real capacity
  regression worth its own follow-up bench against the pre-Route-A baseline.
- **Canonical `--profile sweep` hit the documented TP=4 capacity ceiling**
  (`docs/experience/wins/2026-07-06-dsv4-mtp-tp4.md`) — unbounded
  concurrency floods past what TP=4 on half a node can batch; lockstep
  ticks degraded to ~10-12s each with GPUs pinned 100% (real compute grind,
  confirmed via a standalone timed single-request curl showing normal
  per-request latency — not a deadlock). Switched to bounded
  `--concurrencies 1,4,8 --max-seconds 45`, matched identically across both
  arms.
- **Neither arm actually exercised L2 host-demotion or L3 disk-spill** — KV
  pressure at 79 slots / 23-24 completed 4096-in/256-out requests never got
  tight enough to evict. This is a zero-activity-path comparison: it shows
  "no cost when nothing spills," not "spilling itself is free."

## Learnings

- Enabling `--kv-disk` with zero actual spill activity carries no
  measurable throughput cost (all deltas ±0.9-1.9%, noise-level) — but this
  is only established for the *idle* L3 path. A real L3-overhead
  measurement needs a workload that actually forces host-demotion and disk
  spill (much higher concurrency or a KV budget tight enough to evict).
- Confirm tier activity via BOTH the stats counters (`host_demoted_pages`/
  `disk_pages`) AND actual disk usage (`du`, not apparent/sparse-file size)
  before trusting a "zero cost" tier comparison — a counter bug or a sparse
  file never being written could each independently mask real spill
  activity from the other check alone.
- Route A's shared pools need their own KV-budget-vs-slot-count bench
  against the pre-Route-A baseline before their capacity impact is
  considered acceptable — this run only stumbled onto the regression while
  trying to get enough slots for a tier bench, not as its primary goal.

## Follow-ups

- Rerun at a KV budget/concurrency tuned to force real L2 demotion + L3
  spill, to measure L3's actual overhead under load.
- Bench Route A's shared-pool VRAM footprint against the pre-Route-A
  baseline (`ba36fbd39`) at matched `--max-total-tokens`, to quantify the
  slot-count regression precisely.
