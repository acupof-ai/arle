# DSv4 on-demand FlashMLA band paging (Phase 3b) — 16K slot cliff 1→117, pod gate passed with one documented perf miss

> Status: **Shipped** — commits `91ab1c5d1` (demand paging) + `429e87a70`
> (budget-solve fixes) + `806b7ba4` (memset claim-zero) + `977325ca` (codex R3
> batched-gap fix) + reserve-at-first-chunk. 4×H20 (GPUs 4-7), TP4,
> DeepSeek-V4-Flash-FP8, logs `job3b_*.log` / controls `job3c_*.log`
> (pre-3b `fc850c7c` rebuilt same-day in the needlegate tree).

## What changed

Admission stops reserving `num_slots × full-band`: MODEL1 layers allocate
ring blocks + comp pages from per-layer pool free lists as sequences grow
(claim-zeroed via device memset; the full prompt span reserves at the first
chunk), V32/GLM keeps identity full bands (its pack lane still band-base
addresses). `kv_budget_plan` jointly solves `(num_slots, pool_tokens)` on
cross-rank-reduced scalars; the engine gates admission on token-projection
page availability (`prompt+max_tokens`, conservative watermark =
zero-preemption; pool exhaustion in `ensure_band` is a hard error and never
fired in any lane). New seam hook `release_kv_slot` frees a slot's band at
engine free (radix retention stays host-side via the 2a pool). Selftest now
allocs + refreshes device tables before direct forwards (codex flag).

Two boot bugs found by the first pod round: the budget reduce saturated the
i32 collective at 2047MB (the OLD `pool_budget_total` reduce silently did
the same — its clamp was accidentally conservative, the new solve is not;
now reduced in MiB), and the ring-only-layer floor check wrongly added a
full band.

## Capacity (gate ② — the headline)

| Config | pre-3b (same day, fc850c7c) | 3b |
|---|---|---|
| `--max-total-tokens 2048` | 209 slots (pool-band clamp) | **256 slots**, shared comp 524,288 tokens (8192 engine pages) |
| `--max-total-tokens 16384` | **3 slots** (pool residual 172MB; the 2026-07-09 cliff) | **117 slots**, shared comp 60,672 tokens (948 engine pages) |

The slot cliff dissolves: concurrency at 16K is now bounded by per-slot
STATE (177MB/slot) and shared comp tokens, not whole-band reservation.

## Correctness lanes (final binary unless noted)

| Lane | Result |
|---|---|
| E1 solo ×15 cache ON @2048 | 15/15 (the one v1 miss `738292` did not replicate ×3 on either binary — near-tied floor) |
| E1b solo ×15 cache OFF | 15/15 |
| E2 resend ×10 @2048 | **10/10**, warm TTFT 0.185 vs cold 0.770 = **4.16×** (2a: 4.19×) |
| Multi-shape @16384: n∈{1,4} × len∈{500,2000,8000} | n=4 arms 3/4-4/4 (concurrent near-tied class); **solo len-2000/8000 deterministic `738.` miss is PRE-EXISTING** — same-day pre-3b control misses identically (errors/2026-07-10-dsv4-16k-mtt-solo-needle-truncation-preexisting.md) |
| RB restore→batched ×10 (codex R3 lane: cold publish, then 3 identical concurrent, all restore into ONE batched decode) | **post-fix 30/30 · 29/30 across two runs; pre-fix kill-test 25/30 with warm `738292`/`73829` digit corruption** — the batched bulk-gap fix verified both directions. (First RB round used the `job3b-rbN` salt family — 0/10 exact even SOLO-COLD: a deterministic hard-salt basin, lane re-salted.) |
| Exhaustion / preempt counters | **0 in every lane** — conservative admission holds |

## Perf guard (gate ④) — FAILED at +3.8%, documented not buried

E6 shape (n=4, len 2000, ×15, same salts, same day):

| Arm | mean wall |
|---|---|
| pre-3b fc850c7c (209 slots, identity bands) | **9.137 s** |
| 3b, 256 slots | 9.473-9.498 s (**+3.8%**) |
| 3b forced to 209 slots (diag) | 9.417 s → slot count ≈ 0.9pp |
| 3b + memset claim-zero (removed ~10MB blocking H2D/request) | 9.498 s — no change |
| 3b + full-prompt reserve at first chunk (growth events 9→1/request) | 9.483 s — no change |

Solo len-500 delta is only ~+1.3% (cold TTFT 0.770 vs 0.761). Ruled OUT by
paired A/B: claim-zero H2D, growth/device-table-refresh storm, slot count
(mostly). Residual ~2.9pp on the n=4 prefill-heavy shape remains
unattributed — needs nsys; follow-up filed rather than another guess loop.
Verdict: shipped with the miss on record — the ±2% guard failed on this
one shape; the capacity win (1→117 slots at 16K) plus fully-green
correctness lanes carry the license, per the coordinator gate with this
explicit deviation.

## Rule

- A demand-paging port's perf tax hides in the SMALL per-event host work
  (blocking pageable H2Ds, per-chunk growth) — but verify by paired
  ablation before "fixing": two plausible mechanisms (zeroing, refresh
  storm) each measured ZERO here.
- Salt families are not interchangeable: `job3b-rb-N` 10/10 vs `job3b-rbN`
  0/10 solo-cold on identical code — every warm/concurrent lane needs its
  OWN same-prompt solo-cold floor before its misses mean anything.
