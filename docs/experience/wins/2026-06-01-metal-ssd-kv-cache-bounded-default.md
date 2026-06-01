# Metal SSD KV Cache Bounded Default

## Context

The README Metal retest exposed a bad default: `metal_serve` auto-enabled
`~/.cache/arle/metal_kv` without a size cap. The local cache grew to 47 GiB and
the filesystem had only 315 MiB free. At 12k prompts, `metal_serve` logged
`No space left on device`, and latency/RSS became a disk-pressure artifact.

## What Worked

- Kept SSD prefix snapshots default-on for Metal, but added a default 20 GiB
  `max_bytes` budget.
- The existing LRU high/low watermark eviction now has a budget to enforce.
- `--no-kv-disk` still disables the cache.
- `--kv-disk-max-bytes` overrides the default budget.
- README now states the 20 GiB bounded default and the Metal/CUDA KV quant split.

## Evidence

- Deleted the generated cache: `~/.cache/arle/metal_kv` went from 47 GiB to 0 B,
  and the Data volume recovered from 315 MiB free to 46 GiB free.
- Unit tests:
  `cargo test -p infer --no-default-features --features metal --bin metal_serve kv_disk -- --nocapture`
  passed 3/3.
- Eviction unit test:
  `disk_prefix_index_evicts_under_bounded_budget` covers LRU eviction under
  high/low watermarks.
- Release build:
  `cargo build -p infer --release --no-default-features --features metal --bin metal_serve`.
- Real startup smoke:
  `target/release/metal_serve --model-path mlx-community/Qwen3.6-35B-A3B-4bit --port 9088 --max-running-requests 1 --max-batch-tokens 4096 --warmup 0`
  logged `max_bytes=21474836480 watermarks=0.90/0.75`.

## Rule

Any default-on disk cache must have a finite size budget and eviction evidence.
Unbounded local caches are not allowed as defaults.
