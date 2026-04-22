# Host-tier idle drain local trace still shows no store submit

## Context

- Change under test: `84a3b0c` adds an explicit idle-path host-tier drain gate
  and exposes cumulative staged-store counters in `/v1/stats`.
- Local validation used the canonical CUDA `Qwen3-4B` long-prompt shape on the
  L4 host:
  - `--num-slots 16`
  - `--max-seq-len 4608`
  - `--mem-fraction-static 0.94`
  - `--chunked-prefill-size 4096`
  - `--max-prefill-tokens 16384`
- The canonical `scripts/bench_guidellm.sh` run is currently blocked locally by
  the existing `/v1/completions` rejection of
  `stream_options.continuous_usage_stats`, so local diagnosis used a direct
  `32` request / `16` worker manual load plus `/v1/stats` polling.

## Root Cause

- The idle drain entrypoint is no longer the main problem. The scheduler now has
  a reachable idle-path hook, but the local trace still shows **zero staged
  store submissions**.
- Manual load summary:
  - `32` requests total
  - prompt tokens per request: `4100`
  - completion tokens per request: `255`
  - wall time: `84.52s`
- During that run, the server log clearly shows host-tier demotion pressure:
  - `bench-output/infer-qwen3-4b-l4-c16-84a3b0c-idle-host-drain-server/infer.log:567`
    reaches `host usage 85%`
  - repeated fallback drops continue after that, e.g.
    `bench-output/infer-qwen3-4b-l4-c16-84a3b0c-idle-host-drain-server/infer.log:826`
    and `bench-output/infer-qwen3-4b-l4-c16-84a3b0c-idle-host-drain-server/infer.log:1061`
- But the new `/v1/stats` counters stay flat for the whole run and the full
  `15s` idle tail:
  - `kv_store_q=0/16`
  - `kv_store_ops=0/0/0`
  - `tier_store_wait=0.0ms`
- So the deeper fault is now below the idle sleep gate: `spill_host_blocks_if_pressured()`
  still does not produce a real `submit_store()` path under this workload.

## Fix

- Keep the idle-path drain hook and cumulative store counters; they are still
  useful because they prove the old observability surface was insufficient.
- Do **not** claim host-tier background drain is working end-to-end yet.
- Next diagnosis should focus on why `spill_host_blocks_if_pressured()` skips all
  candidates even once host usage reaches the `0.85` high-water mark.

## Rule

- When a background tier path is suspected, add **cumulative** queue activity
  counters before drawing conclusions from sampled queue depth alone.
- If host usage reaches the configured high-water mark but staged-store counters
  remain at zero, the bottleneck is `submit_store` eligibility, not idle wait
  reachability.
