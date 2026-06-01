# Metal 12K Decode Tail Fix

## Context

ARLE 12K/256 serial measurements showed TPOT spikes that MLX did not show.
The bad samples were not caused by parallel load or KV pool dual-write:
`pool_dual_write_us` was effectively zero in `INFER_PHASE_TIMING=1`.

## What Worked

Root cause was an ARLE-only duplicate cache policy. The Qwen3/Qwen3.5
scheduler paths called `clear_cache()` at every `KV_CACHE_CHUNK=256`
decode boundary, while M_e.11 already provides the global 1024 generated-token
residency clear. On 12K/256 this boundary landed inside the measured request
and produced 50-332 ms `clear_us` spikes plus decode instability.

Fix: boundary clears now go through `INFER_METAL_KV_BOUNDARY_CLEAR`, default
off. M_e.11 remains default-on.

## Evidence

Pre-fix 12K cache-neutral phase timing:

| metric | value |
|---|---:|
| `clear_us` max | 332.463 ms |
| `clear_us > 1ms` | 1 per request |
| measured TPOT samples | 24.0, 15.7, 32.1, 32.8, 16.1 ms |

Post-fix 12K cache-neutral, serial, `enable_thinking=false`, prefix hit 0:

| metric | value |
|---|---:|
| prompt tokens | 12,334-12,339 |
| TTFT measured | 15.08, 14.87, 14.85 s |
| TPOT measured | 14.52, 14.42, 14.88 ms |
| `clear_us` max | 0.009 ms |
| `clear_us > 1ms` | 0 |
| process RSS high-water | 17.77 GiB |

Verification:

```bash
cargo test -p infer --no-default-features --features metal enable_thinking -- --nocapture
cargo check -p infer --no-default-features --features metal
cargo build -p infer --release --no-default-features --features metal --bin metal_serve
```

Raw logs:

- `/tmp/arle_12k_phase_timing_default.log`
- `/tmp/arle_12k_after_fix_cache_neutral_exact.log`

## Rule

Do not stack periodic Metal cache-clears. One measured, centralized residency
cadence is enough; extra boundary clears are p99 sync points.
