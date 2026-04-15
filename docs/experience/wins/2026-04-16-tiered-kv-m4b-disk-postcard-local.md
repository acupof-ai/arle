# 2026-04-16 · Tiered KV M4b disk postcard local

## Context

M4a landed the real `BlockFingerprint::compute` chain, which means the disk
transport can stop pretending a local `file_id` is a durable identity. M4b
switches `infer/src/kv_tier/transport/disk.rs` from a raw-byte dump plus
sequential filename allocator to a postcard header plus
fingerprint-addressed filename format for T2 session persistence.

Scope stayed narrow:

- added `postcard` as a workspace dependency and pulled it into `infer`
- rewrote `DiskStore`'s block API to use
  `DiskBlockLocation { path, payload_len, fingerprint }`
- kept the keyed `write` / `read` / `remove` API unchanged
- added header validation tests for magic, version, fingerprint, and
  filename stability

## What Worked

- The wire format is minimal and restart-safe: `postcard::to_allocvec` of
  `DiskBlockHeader { magic, version, fingerprint, kv_format_tag, payload_len }`
  followed by the raw payload bytes. No async, no extra framing layer.
- Content-addressed filenames are deterministic and cheap:
  `BlockFingerprint([u8; 16])` becomes `32` lowercase hex chars plus `.kv`.
  Re-writing the same `(fingerprint, payload)` lands on the same path;
  different fingerprints land on different paths.
- `get_block` validates the persisted header before returning bytes:
  wrong magic, wrong version, wrong fingerprint, and payload-length mismatch
  all surface as `io::ErrorKind::InvalidData` instead of silently returning
  garbage.
- The round-trip test now uses a real `KvContentContext` and exercises both
  the positive path and the mismatched-fingerprint rejection path, which is
  the actual M4 requirement for session save/load reconciliation.

## Rule

When a persistence layer crosses a restart boundary, the filename and the file
header have separate jobs and both matter: the filename gives O(1) lookup by
content address, while the header carries enough versioned metadata to reject
stale or malformed bytes before they re-enter the cache. Keep both, validate
both, and keep the format sync + `std::fs` until a real async coordinator
exists.
