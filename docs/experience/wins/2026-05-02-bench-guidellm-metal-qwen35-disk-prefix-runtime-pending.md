# Metal Qwen3.5 Disk Prefix Runtime Pending Bench

## Context

P0-1A SSD KV persistence now connects Qwen3.5 live prefix snapshots to
`DiskStore` from the Metal scheduler runtime.

## What Worked

- Publish writes block-aligned Qwen3.5 prefix snapshots as content-addressed
  `DiskStore` blocks.
- Activation imports the longest safe memory/SSD hit and skips DFlash requests,
  whose draft-side state is not part of the target KV/GDR snapshot.
- Startup scans existing `.kv` blocks, rejects wrong model fingerprints from the
  small snapshot header, and rebuilds the disk prefix index.

## Benchmark

Status: `pending-remote`

No guidellm TTFT table was captured on this Mac. Run the Qwen3.5 Metal long
prompt scenario with `--kv-disk-dir` enabled and attach first-run vs second-run
TTFT, plus restart-hit TTFT, before closing P0-1A.

## Rule

Disk prefix reuse is opportunistic. Corrupt or stale SSD entries must degrade to
normal prefill instead of failing request activation.
