# Metal Qwen3.5 Snapshot Payload Pending Bench

## Context

P0-1A SSD KV persistence needs a stable byte payload for Qwen3.5 live prefix
snapshots before the runtime can place those snapshots in `DiskStore`.

## What Worked

- `Qwen35PrefixSnapshot` now encodes to a postcard payload with magic, version,
  model fingerprint, token ids, cache length/capacity, and named MLX array
  records.
- Decode rejects model-fingerprint mismatches before importing arrays.
- Unit tests round-trip real MLX arrays through the payload format.

## Benchmark

Status: `pending-remote`

No guidellm run was executed for this tranche because the payload helpers are
not yet called by `MetalQwen35PrefixRuntime`. The next tranche that writes and
reads these payloads through `DiskStore` must attach the first TTFT before/after
table.

## Rule

Keep the snapshot payload Qwen3.5-specific until block-level Metal KV/Radix
persistence replaces this bridge path.
