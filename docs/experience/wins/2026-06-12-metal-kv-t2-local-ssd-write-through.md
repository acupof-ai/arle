# Metal KV T2 local SSD write-through

## Goal

Let the Metal serve path consume `--kv-ssd-path` locally, using the same
engine-core demote/promote tier seam as CUDA instead of a Metal-only scheduler
branch.

## Hypothesis

Metal already publishes page-aligned prompt KV and GDR boundary snapshots into
`MetalPageStore`. If each published page/snapshot is mirrored write-through to a
process-owned SSD namespace, radix demotion can drop RAM mirrors and later
promote the page image back into fresh Metal pages without re-prefilling the
prompt.

## Params

- `infer-metal`: added an opt-in SSD tier under `MetalPageStore`.
- Disk records are versioned binary envelopes containing record kind, logical
  page ids, cache length, array dtype/shape, and raw MLX array bytes.
- Prefix snapshots are keyed by stable logical page ids, not physical page ids,
  so promoted pages can attach even when the host allocator restores them under
  new page ids.
- `publish_slot` writes page blocks and exact-boundary GDR snapshots to SSD
  synchronously. `release_pages` drops RAM mirrors only.
- `demote_prefix_pages` binds engine-local tier keys to already mirrored logical
  pages. `promote_prefix_pages` restores page blocks from SSD into destination
  pages.
- `infer-api` now passes `ServeKvSsdOptions` through the Metal router and
  attaches the tier pre-traffic through `ServeHandle::run_on_executor`.
- `--kv-ssd-max-bytes` is a hard byte budget. When unset, Metal uses the same
  host-tier policy shape as CUDA: statvfs free-disk probe, capped at the proven
  T2 default.

## Env

- Host: local Apple Silicon macOS.
- Model for serve smoke: `mlx-community/Qwen3.5-0.8B-MLX-4bit`.
- Feature set: `--release --no-default-features --features metal,no-cuda,cli`.
- SSD root: `/tmp/arle-metal-t2-smoke`.

## Results

- `cargo test -p infer-metal --release --no-default-features --features metal -- --nocapture`: 23 passed.
- `cargo test -p infer-api --release --no-default-features --features metal,no-cuda kv_ssd -- --nocapture`: 3 passed, adapter 0 tests.
- Local serve smoke:

```bash
/opt/homebrew/bin/timeout 60s cargo run --release \
  --no-default-features --features metal,no-cuda,cli -- serve \
  --backend metal \
  --model-path mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --kv-ssd-path /tmp/arle-metal-t2-smoke \
  --kv-ssd-max-bytes 1073741824 \
  --port 0
```

Observed:

- Metal accepted `--kv-ssd-path` instead of fail-closing.
- Serve reached warmup and printed:
  `KV T2 SSD tier: root=/tmp/arle-metal-t2-smoke/arle-metal-kv-tier-..., budget_bytes=1073741824, capacity_pages=10591`.
- The timeout exit code was 124 by design; the process was stopped after the
  mount was verified.

New regression coverage:

- `ssd_write_through_promotes_released_pages_and_prefix_snapshot`: publishes
  two pages, verifies SSD prefix persistence, demotes, releases RAM mirrors,
  promotes from SSD, materializes the prefix, and checks KV/GDR contents.
- Existing stale-snapshot alias tests now use stable logical page ids and still
  cover physical page reuse.

## Problems

This is a functional local SSD enablement, not a performance license. No
guidellm sweep was run in this tranche, so TTFT/ITL/throughput impact is
deferred. The first implementation is synchronous write-through with bounded
cache records; async prefetch, GDS, and shared cross-process durable cache
identity remain separate work.

## Learnings

Metal SSD tiering must treat GDR state as part of the prefix image. Restoring
only K/V pages is insufficient because the physical page ids can change during
promotion. Stable logical page ids let the runtime keep the existing
engine-core tier seam while preserving the Metal-specific prefix snapshot
invariant.
