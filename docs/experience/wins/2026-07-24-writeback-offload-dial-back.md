# Writeback-offload threshold 4096 → 16384: offload buys zero peak headroom below 12K and costs −29…−38% backward wall

> Status: Shipped (#172, default flip). Pod: 8×H20, GPU 2 only, 27B FP8
> student, LoRA r16 α32 qv, masked-CE writeback via `agent-opd
> --replay-records` with synthetic fixed-length records,
> `ARLE_OPD_VRAM_TRACE=1`. Base residency 28.5 GiB.

## Context

`writeback_offload_for_seq` engaged host-offload at seq≥4096, anchored on a
pre-fused-CE "resident checkpoints OOM at seq≈9600" measurement. Fused-CE
(no [seq,vocab] tile) and the batched-LA device path (ecc058b20) changed the
headroom; #172 asked for a measured re-sweep.

## What Worked

| seq | ON backward s | OFF backward s | Δ | ON peak MiB | OFF peak MiB | OOM |
|---|---|---|---|---|---|---|
| 5120 | 60.7 | 40.0 | −34% | 43,177 | 43,881 | no |
| 8192 | 122.0 | 75.1 | −38% | 53,129 | 53,321 | no |
| 10240 | 147.8 | 105.4 | −29% | 57,001 | 56,617 | no |
| 12288 | 204.4 | 140.6 | −31% | 61,961 | 61,897 | no |
| 16384 | — | 224.7 | — | — | 72,489 | no |
| 20480 | — | 333.7 | — | — | 82,505 | no |
| 24576 | — | 463.4 | — | — | 93,449/97,508 | no |
| 28672 | — | fwd ok, alloc fail | — | — | 85,481 at fail | **yes** |

- **Peak VRAM ON vs OFF is a wash at every measured seq** — in this regime
  offload buys no headroom, only serializes H2D re-uploads (−29…−38% backward).
- New resident OOM boundary: last-good **24,576**, fail 28,672 — ~2.6× the old
  9,600 anchor. Resident peak scales ≈ +2.4 MiB/token.
- **Flip: `WRITEBACK_OFFLOAD_MIN_SEQ` 4096 → 16384** — 1.5× under
  last-proven-good, 25 GiB peak headroom at the threshold, recovers the
  −29…−38% backward tax across the 4K–16K band.
- Margin rationale: the replay lane lacks live rollout-engine scratch remnants;
  the 1.5× covers that. 20480 is plausible but 16384 is the SOLID pick.

Evidence: pod `/host/ob172/` (records, per-arm logs, maxmem files).

## Rule

- An offload/spill default is re-measured after any change that shrinks the
  resident footprint it was protecting — a stale threshold silently taxes the
  common case (here −29…−38% backward for 2 weeks of the 4K-16K band).
