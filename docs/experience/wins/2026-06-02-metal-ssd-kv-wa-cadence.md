# Metal SSD KV Write-Amplification Cadence

## Goal

Lower Metal SSD KV write amplification without making recurrent-state restore
unsafe.

## Hypothesis

The segmented store already avoids full snapshot rewrites, but short session
extensions still rewrite Qwen3.6's recurrent GDR state. The production-safe fix
is to keep small tails in memory and persist SSD checkpoints only when a session
extension is large enough to amortize the recurrent state.

## Params

- Backend: Metal Qwen3.6 prefix snapshot path.
- Default cadence: `INFER_METAL_PREFIX_PERSIST_MIN_EXTENSION_TOKENS=64`.
- Store: 64 KiB chunked segment store from
  [`2026-06-02-metal-segmented-ssd-kv.md`](2026-06-02-metal-segmented-ssd-kv.md).
- Smoke binary: `target/release/metal_serve`.
- Smoke model: `mlx-community/Qwen3.6-35B-A3B-4bit`.
- Smoke flags: `--max-running-requests 1 --max-batch-tokens 512 --warmup 0
  --kv-disk-max-bytes 2147483648 --kv-memory-max-bytes 536870912`.
- Smoke trace: `RUST_LOG=info INFER_M_E10_TRACE=1 INFER_M_E13_TRACE=1
  INFER_METAL_PREFIX_READBACK_US_PER_TOKEN=0.001`.

## Results

Implemented:

- Added `INFER_METAL_PREFIX_PERSIST_MIN_EXTENSION_TOKENS`.
- Default is 64 tokens.
- Cold prompt snapshots still use the existing prefill-vs-readback budget gate.
- Session extension snapshots smaller than the cadence stay in the bounded
  in-memory tier and are not written to SSD.
- Disk-imported extensions remain SSD-skipped until the runtime has a semantic
  delta-manifest path. This preserves the measured fix for the byte-drifted
  re-export case.

Verification:

```text
cargo test -p infer --lib --no-default-features --features metal metal_prefix_persist_gate -- --nocapture --test-threads=1
  1 passed

cargo test -p infer --lib --no-default-features --features metal metal_prefix_extension_delta -- --nocapture --test-threads=1
  1 passed

cargo test -p infer --lib --no-default-features --features metal chunked_snapshot -- --nocapture --test-threads=1
  3 passed

cargo test -p infer --lib --no-default-features --features metal disk_snapshot -- --nocapture --test-threads=1
  1 passed

cargo test -p infer --lib --no-default-features --features metal reconciles_persisted_snapshot_headers -- --nocapture --test-threads=1
  1 passed

cargo clippy -p infer --no-default-features --features metal --lib -- -D warnings
  passed

cargo build -p infer --release --no-default-features --features metal --bin metal_serve
  passed
```

Real Qwen3.6 smoke:

```text
request:
  prompt_tokens=26 completion_tokens=8 total_tokens=34

prompt snapshot:
  tokens=26 extension_delta_tokens=None worth_persist=true
  payload_bytes=64923155 chunks_written=1010 chunks_reused=0
  physical_chunk_bytes_written=64921600 manifest_bytes_written=50709
  write_us=66864

completed-session snapshot:
  tokens=33 extension_delta_tokens=Some(7)
  min_extension_tokens=64 worth_persist=false

disk layout:
  manifests=1 segments=1 chunks=0 total=62M
```

The prompt checkpoint still wrote normally at about 927 MiB/s. The 7-token
session tail wrote zero SSD bytes.

Measured Qwen3.6 effect for the short tail:

```text
before:
  extension_delta_tokens=7
  physical_write=1,505,867 bytes
  delta_logical_kv=143,375 bytes
  delta-WA=10.50x

after:
  extension_delta_tokens=7 < 64
  physical_write=0 bytes
  delta-WA=0x for SSD
```

The previous full-snapshot rewrite comparison remains:

```text
old monolithic rewrite: 65,844,884 bytes
segmented write before cadence: 1,505,867 bytes
reduction: 43.7x
cadence write for the 7-token tail: 0 bytes
```

## Problems

GDR delta encoding was intentionally not shipped in this tranche. Qwen3.6 GDR is
recurrent state, not append-only KV. A persisted snapshot at token `N + d`
cannot restore safely from a token `N` GDR checkpoint plus a raw KV tail unless
the runtime also proves and stores the replay or delta contract for the
recurrent state.

The safe production behavior is therefore:

- SSD checkpoint at prompt prefixes and at sufficiently large session
  extensions.
- Keep short generated tails in memory.
- On restart, import the nearest SSD prefix and prefill the tail.

## Learnings

For hybrid KV + recurrent models, SSD write amplification is not just a chunk
store problem. KV is append-like; GDR is whole-state. The storage policy must
respect that difference instead of pretending every byte stream is an append log.
