# Metal Segmented SSD KV Snapshot Store

## Goal

Make Metal SSD prefix KV production-usable: high sequential I/O, bounded disk
usage, low write amplification for prefix extensions, and integrity checks that
are cheap enough for the hot persistence path.

## Hypothesis

Monolithic snapshots make every session extension rewrite tens of MiB. Tiny
standalone chunk files dedupe bytes but create a filesystem storm. A segment
store should get both properties: 64 KiB content chunks for reuse, but new bytes
are appended into one sequential segment per snapshot write. CRC32C catches
corruption cheaply, while the existing BLAKE3 fingerprint remains the content
identity.

## Params

- Host: Apple M4 Pro, macOS 26.3.1, 48 GiB unified memory.
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit`.
- Binary: `target/release/metal_serve`.
- Flags: `--max-running-requests 1 --max-batch-tokens 512 --warmup 0
  --kv-disk-dir /tmp/arle-kv-final-smoke.RJRvf3 --kv-disk-max-bytes
  2147483648 --kv-memory-max-bytes 536870912`.
- Trace: `RUST_LOG=info INFER_M_E13_TRACE=1
  INFER_METAL_PREFIX_READBACK_US_PER_TOKEN=0.001`.

## Results

Implementation:

- Added backend-neutral `ChunkedSnapshotStore` under `infer/src/kv_tier/`.
- Layout is manifest plus segment files:
  `manifests/<id>.manifest` and `segments/<id>.segment`.
- Each logical payload is split into 64 KiB chunks. Existing chunk fingerprints
  are reused across live manifests; new chunks are written sequentially into one
  segment.
- Each chunk stores CRC32C plus a 128-bit BLAKE3 chunk id. Metadata stores
  CRC32C plus BLAKE3. Reads validate length, CRC32C, then chunk id.
- Qwen3.6 Metal snapshots now trim KV arrays to live `cache_len`; unused
  preallocated decode capacity is not written to SSD.

Verification:

```text
cargo test -p infer --lib --no-default-features --features metal chunked -- --nocapture --test-threads=1
  15 passed

cargo test -p infer --lib --no-default-features --features metal crc32c -- --nocapture --test-threads=1
  3 passed

cargo test -p infer --lib --no-default-features --features metal disk_snapshot -- --nocapture --test-threads=1
  1 passed

cargo test -p infer --lib --no-default-features --features metal reconciles_persisted_snapshot_headers -- --nocapture --test-threads=1
  1 passed

cargo test -p infer --lib --no-default-features --features metal -- --test-threads=1
  738 passed, 29 ignored

cargo check -p infer --no-default-features --features metal --bin metal_serve
  passed

cargo build -p infer --release --no-default-features --features metal --bin metal_serve
  passed

cargo clippy -p infer --no-default-features --features metal --lib -- -D warnings
  passed
```

Real Qwen3.6 smoke:

```text
first request, 64-token prompt:
  tokens=64 payload_bytes=65701509 chunks_written=1010 chunks_reused=0
  physical_chunk_bytes_written=65699840 manifest_bytes_written=50856
  write_us=73298

same process, 71-token completed session:
  tokens=71 payload_bytes=65844884 chunks_written=40 chunks_reused=990
  physical_chunk_bytes_written=1454080 manifest_bytes_written=51787
  write_us=55044

disk layout after first server:
  manifests=2 segments=2 chunk_files=0 total=64M

restart:
  indexed entries=2 logical_bytes=131546393
  disk import tokens=64 payload_bytes=65701509 read_us=52095
  decode_us=32770 import_us=36 imported=true
  prompt_tokens=71 resume_prefill_tokens=7 skip_rate=0.901408

after restart request:
  manifests=2 segments=2 chunk_files=0 total=64M
```

The second same-process write extended the session but wrote only about 1.39
MiB of physical chunk data. The restarted request imported the persisted
64-token prefix and did not grow the SSD directory.

## Problems

The disk index currently accounts logical bytes, not physical deduped bytes.
That is conservative for a 20 GiB budget, but it can evict earlier than the
filesystem actually requires.

After an SSD import, the runtime intentionally does not immediately persist the
extended request back to SSD. A measured pre-fix run showed that re-exporting
the C++ session after disk import could produce byte-drifted prefix chunks and
rewrite the full snapshot. Current-process memory still keeps the extension.
The follow-up is a semantic delta-manifest path for imported prefixes.

Segment GC keeps a segment as long as any chunk inside it is referenced. That
avoids unsafe compaction, but partially orphaned segment bytes remain until all
chunks in that segment are dead.

## Learnings

CRC32C is the fast corruption check, not the identity. The content identity
still has to be the chunk fingerprint.

For local SSD KV, the right default shape is not "one huge blob" and not "one
file per chunk". It is manifest plus large sequential segment writes, with
chunk-level references inside the manifest.

Active decode KV remains memory-resident. SSD is a bounded prefix snapshot tier
for admission-time reuse, not a per-token active KV backing store.
