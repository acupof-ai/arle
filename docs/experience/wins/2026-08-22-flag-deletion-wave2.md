# Flag deletion wave 2 — three more proven flags deleted, Metal warmup default fixed

> Status: Accepted. `5b880f2f8`, 9-agent workflow (6 investigators → edit → verify).
> Serve flags 56 → 53.

## Context

Wave 1 (2026-08-22) deleted 10 flags. Wave 2 ran per-flag verdict
investigations on the remaining uncharacterized flags: a flag is deleted
only when its off-arm has a dated losing verdict or the flag is proven
inert; live unmeasured knobs stay.

## What Worked

Deleted (hardcoded winner, full chain removed — flag, seam field, static,
accessor, read sites, doc rows):

| flag | hardcoded | verdict |
|---|---|---|
| `--qwen35-fa3` | on | off-arm 2.76× decode loss (2026-07-27), −36 % prefill (2026-06-11) |
| `--qwen35-deepgemm` | on | decode-inert by construction (R≤256 < 1024 floor); prefill 9.10→2.32 s (2026-06-11) |
| `--qwen35-moe-decode-kernel` | on | off-arm 15.4× at the real shape, 10.8× at c=16 |

The non-sm90 / head_dim≠256 fallback kernels stay — they are the permanent
capability lane, not a flagged arm. `--qwen35-fa3 false` also silently
disabled the BF16 paged decode graph, a second unmeasured regression.

**Metal warmup default fixed:** `MetalRuntimeFlags::default()` had
`warmup: true` while the shipped default is false (2026-08-20) — programmatic
(non-CLI) Metal loads silently warmed up. Seam Default and serde default
now false.

**Kept, with reasons:** `--qwen35-deepgemm-min-routes` (real per-workload
knob; the 256<R<1024 crossover is unmeasured and it encodes the decode-band
invariant); `--dspark-confidence-threshold`, `--mtp-adaptive`,
`--mtp-min-accept` (live, no verdict; deletion surface sits in peer-dirty
`executor/dsv4.rs` — finishable in a later wave); `--low-impact` (load-bearing
Metal governor; on CUDA it is exactly `--max-running-requests 1` + a
`--chunked-prefill-size` clamp).

**Stale docs fixed (12 refs):** environment.md rows for the deleted flags
and env names; baselines.md / perf-qwen36-27b.md default-on wording;
`ARLE_QWEN35_QUANT_PROFILE`→`ARLE_QWEN35_PROFILE` renames.

Verified: cuda-lane clippy `-D warnings` (cli + infer-api), metal check,
cpu smoke 5/5.

## Rule

Per-flag verdict investigations before deleting: "no entry" means keep, not
delete. A live unmeasured knob is a bench backlog item, not dead wiring.
