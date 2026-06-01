# Metal In-Memory KV Cache Auto Budget

## Context

The Metal prefix cache had two separate residency tiers:

- SSD snapshots: already bounded by the 20 GiB LRU disk budget.
- In-memory snapshots: still budgeted in token-equivalent units, not resident
  bytes.

With `--max-batch-tokens 4096`, the old in-memory limit expanded to more than a
million token-equivalent units. That was effectively unbounded for Qwen3.6
serving and could retain completed-request KV arrays until process memory grew
far beyond the visible live request set.

## What Worked

- Replaced token-equivalent accounting with actual retained MLX array bytes
  (`kv_flat` + `gdr_flat` `nbytes()`).
- Added LRU eviction for the byte budget and a drop path for snapshots larger
  than the budget.
- Added `--kv-memory-max-bytes`; `0` disables the in-memory snapshot tier while
  leaving SSD snapshots enabled.
- Default `metal_serve` now auto-sizes the memory snapshot budget from:
  available memory, model KV bytes per token, `max_running_requests *
  max_batch_tokens`, and system headroom.
- Model weight file size is logged but not hard-reserved in the default
  non-wired mode. MLX maps Qwen3.6 weights through mmap/lazy unified memory, so
  the 19 GiB checkpoint size is not the same as current RSS. If
  `--auto-wired-limit` or a positive `--wired-limit-bytes` is used, the auto
  budget reserves the corresponding weight residency.

## Evidence

Targeted release tests:

```text
cargo test --release -p infer --no-default-features --features metal auto_budget -- --nocapture
  4 passed

cargo test --release -p infer --no-default-features --features metal --bin metal_serve kv_memory -- --nocapture
  4 passed

cargo test --release -p infer --no-default-features --features metal memory_prefix_runtime -- --nocapture
  2 passed
```

Real Qwen3.6 startup smoke:

```text
target/release/metal_serve \
  --model-path mlx-community/Qwen3.6-35B-A3B-4bit \
  --port 9088 \
  --max-running-requests 1 \
  --max-batch-tokens 4096 \
  --warmup 0
```

Relevant log lines:

```text
Metal in-memory KV snapshot auto-budget: budget=8589934592 bytes (8.00 GiB), available=24003575808 bytes (22.36 GiB), total=51539607552 bytes (48.00 GiB), model_weights=20402204271 bytes (19.00 GiB), weight_reserve=0 bytes (0.00 GiB), live_kv_estimate=83886080 bytes (0.08 GiB), live_kv_reserve=167772160 bytes (0.16 GiB), system_headroom=5153960755 bytes (4.80 GiB), spare_after_reserve=18681842893 bytes (17.40 GiB), kv_bytes_per_token=20480
Metal live prefix cache enabled for Qwen3.5 snapshot replay: block_size=16, max_cached_bytes=8589934592
Metal SSD KV cache configured: dir=/path/to/.cache/arle/metal_kv max_bytes=21474836480 watermarks=0.90/0.75 fsync_each_block=false
```

Dev/test build artifacts were cleaned after targeted tests:

```text
cargo clean --profile dev
  Removed 19124 files, 4.2GiB total
```

## Rule

Memory cache budgets must account retained resident bytes, not token counts.
For MLX mmap-backed weights, checkpoint size is a risk bound and a wired-mode
reserve, not default-mode RSS.
