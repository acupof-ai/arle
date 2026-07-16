# DSv4 FP32 scratch hoist — per_slot 9618→338 MB, slots 2→59, rate16 +48% — CUDA, 2026-07-16

> Status: Shipped

## Goal

Restore the DSv4 slot budget collapsed by the FP32 compressor promotion: the
transient probe scratch (`fp32_kv_raw`/`fp32_score_raw`, `2 × width ×
max_seq_len × 4 B`) was allocated per (layer, slot) inside
`Dsv4CompressorState`, inflating per_slot to 9618 MB → `256 slots clamped
to 2`.

## Change (`672b8ac08`)

`Dsv4CompressorFp32Scratch` — one model-wide shared pair per rank (pattern:
#85 P3 FlashMLA hoist), threaded `Option<&mut>` through every prefill lane;
decode lanes pass `None` (probe unreachable there, fail-loud otherwise).
Per-slot ledger and budget fixed-term stay bit-exact.

## Ledger (serve log, verbatim deltas)

| | arm B (2e635eda3) | arm C (672b8ac08) |
|---|---|---|
| per_slot | 9618 MB (slot-state 9596) | **338 MB** (slot-state 316) |
| shared compressor FP32 | — | 256 MB per rank (once) |
| slot clamp | 2 | **59** (affordable 60) |
| shared comp capacity | 65536 tok | 84736 tok |

## Correctness (needle gate, ship gate PASS)

Depth 0.0: 27/27 exact, all DET. Depth 0.5: zero miss (115/1000 partial ×3,
180 exact 2 + partial 1 — known mid-depth behavior). Logs:
`needle-armC-d0{5,0}.log` (pod).

## A/B vs arm B (rolling-champion comparison, same GPUs/config/dataset/seed)

| point | B total tok/s | C total tok/s | B out tok/s | C out tok/s | B TTFT p50 | C TTFT p50 |
|---|---:|---:|---:|---:|---:|---:|
| rate 1 | 4117 | 4514 | 19.6 | 21.4 (+9%) | 521 ms | 446 ms |
| rate 4 | 5339 | 6663 | 25.4 | 31.7 (+25%) | 1467 ms | 1270 ms |
| rate 8 | 5605 | 7985 | 26.6 | 37.9 (+42%) | 3921 ms | 2612 ms |
| rate 16 | 5481 | 8130 | 26.0 | 38.6 (+48%) | 6698 ms | 5397 ms |
| var-c1 | 854 | 850 | 4.9 | 4.9 (wash) | 3019 ms | 3031 ms |
| var-c32 | 41 done/120 s | **59 done/101 s (+72% req/s)** | — | — | 51.8 s | 32.6 s |

var-c1 is a byte-level wash — clean null control (the fix moves concurrency
headroom only). c32 caveats: arm C `total tok/s` (826) is deflated by a
successful-set prompt-mean accounting artifact (1311 vs 2763) — completed
count and TTFT are the robust columns; ITL p50 rises 160 ms → 2079 ms because
59 slots actually interleave 32 decodes + chunked prefill (clamp 2 had
queue-serial tight ITL). **The c32 run ended at ~101 s in a fatal
`HostPagedKvPool out of pages` teardown** — see
[errors/2026-07-16-dsv4-c32-hostpagedkvpool-fatal.md](../errors/2026-07-16-dsv4-c32-hostpagedkvpool-fatal.md);
unreachable at clamp 2, now the top blocker for the high-concurrency regime.

## Environment

8×H20, driver 535.161.08, CUDA 12.9, DSv4-Flash-FP8, TP=4/EP=4 GPUs 0–3,
eager serve port 8000, build `--release --features cuda,nccl` (plain `cuda`
cannot serve TP=4). HEAD carried unrelated train/docs commits; infer-cuda
delta vs arm B is 672b8ac08 only. Raw:
`bench-output/2026-07-16-fp32slots-*` (pod).

## Learnings

- Transient per-call scratch inside a per-(layer,slot) state struct multiplies
  by layers × slots — audit any "pre-allocated to avoid per-call alloc" fix
  for WHERE the buffer lands, not just that it's pre-allocated.
- The slot-state remainder is now 316 MB — per_slot is no longer the
  concurrency binding constraint; the shared comp pool (84736 tokens) and the
  fatal-alloc path are.
- var-c1 wash + rate-grid gains together attribute the win entirely to
  concurrency capacity, matching the mechanism — no kernel-speed claim made.
