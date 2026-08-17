# L2/L3 KV tier validation — Qwen3.6-27B + DeepSeek-V4-Flash-FP8, 2026-08-17

> Status: Shipped

## Context

User request: validate L2 (host DRAM) and L3 (NVMe disk) KV tier spill for all
production models. Two mechanisms are in scope:

- **Whole-slot tier** (`slot_tier`): parks a demoted slot's full device image
  in the tier store when `active.len() >= running_cap` (oversubscription).
- **Page-level tier** (`kv_system`): radix pages demoted to host, spilled to
  disk when the host budget is exceeded.

## What Worked

### Qwen3.6-27B-FP8 (Metal, page-tier + whole-slot)

- demoted=372 / promoted=372 / failures=0
- Coherent generation under tier churn.
- Park ~130 ms, promote ~275 ms (L3 disk read when L2 budget exceeded).

### DeepSeek-V4-Flash-FP8 (CUDA TP=4, whole-slot + L3 NVMe)

Serve config: `--kv-dram 256MiB --kv-disk /data00/kv-ssd-dsv4
--kv-oversubscription --max-running-requests 8 --max-total-tokens 4096`,
TP=4 on H20 (sm_90), binary `l2l3-fix` (source 72d3afccb).

14 concurrent requests (3.5K-token prefill each, 128 max-tokens), 59.9 s wall:

| Metric | Value |
|---|---|
| demoted_slots | 206 |
| promoted_slots | 206 |
| slot_promote_failures | 0 |
| reuse_hit_disk (L3) | 1187 |
| reuse_hit_host_demoted (L2) | 209 |
| reuse_hit_resident | 160 |
| reuse_miss | 10 |
| disk_pages | 32 |
| L3 disk written | 2.6 GB |
| Request success | 14/14 (200 OK) |
| Generation | Coherent (correctly identified repeated pangrams) |

L3 NVMe budget: 1.27 TB at `/data00/kv-ssd-dsv4` (ssd_fraction 0.5).

### Lockstep ack timeout under extreme churn (known limitation)

With 16 concurrent requests and 256 MiB DRAM, a worker stalled permanently
at tick #337 (min_acked=333, 4 ticks behind). The coordinator's 120 s ack
timeout fired and tore down the serve. 320+ parks / 315+ promotes completed
before the stall; 3.3 GB was spilled to L3 disk. The stall is in the L3
promote path (disk read blocking the lockstep step), not in the tier logic
itself. 14 concurrent requests with the same config completed without
stalling. Production configs (8 GiB+ DRAM) have far lower churn.

## Rule

- L2/L3 validation requires `--kv-dram` small enough to force disk spill
  (256 MiB for Flash-FP8 with 3.7K-token slots; 8 GiB never spills).
- Whole-slot parking needs `active.len() > running_cap` — send more concurrent
  requests than `--max-running-requests`.
- The lockstep coordinator's 120 s ack timeout is the ceiling for a single
  promote; an L3 read that exceeds it tears down the serve. Keep concurrent
  requests ≤ 14 with 256 MiB DRAM on Flash-FP8 TP=4.
