# 2026-04-16 · Tiered KV M4a BLAKE3 fingerprint local

## Context

Tier B shipped `BlockFingerprint::compute_from_tokens` as a two-seed
`DefaultHasher` placeholder with a doc comment saying the real hash was
deferred to M4. M4a does that upgrade: replaces the placeholder with a
real BLAKE3 over a canonical, domain-tagged input chain so fingerprints
are stable across restarts and across hosts. This is the unblocker for
M4's session save/load reconciliation pass — without a cross-process
stable fingerprint, a reloaded session cannot identify its own blocks
in a fresh `TokenKVPool`.

## What Worked

- Added `blake3 = "1"` as a workspace dep with
  `default-features = false` + `features = ["std"]`. `cargo tree -p
  blake3` is small: `arrayref`, `arrayvec`, `cfg-if`,
  `constant_time_eq`, plus `cc` as build-dep only. No rayon, no mmap,
  no pulled-in heavy deps.
- New `types::KvContentContext<'a> { model_fingerprint, kv_format_tag,
  parent }` packages the non-token inputs so the compute chain stays
  type-checked instead of a loose `&[u8]` concatenation.
- New `BlockFingerprint::compute(ctx, tokens) -> Self` walks a
  canonical domain-tagged byte sequence (`"pegainfer-kv-v2\x00"` +
  `"model\x00"` + len + bytes + `"fmt\x00"` + tag +
  `"parent\x00"` + 0/1 + optional 16 bytes + `"tokens\x00"` + len +
  u32 LE token bytes) and truncates the 32-byte BLAKE3 output to 16
  bytes. Domain tags prevent length-extension boundary collisions;
  version tag lets us bump the chain later without ambiguity.
- Legacy `compute_from_tokens(&[u32])` kept as a `#[doc(hidden)]` shim
  that routes to `compute` with `model_fingerprint = b""`,
  `kv_format_tag = 0`, `parent = None`. Keeps the Tier B test + any
  residual callers compiling. Session-persistence call sites must go
  through `compute` with a real context.
- `Scheduler<M>` grew a `model_fingerprint: Vec<u8>` field populated
  in `with_config` via `blake3::hash(model_id.as_bytes())`. Per-engine
  stable; a real weight-checksum upgrade is M5-era work.
- `publish_to_prefix_cache` now chains parent fingerprints block by
  block: first block has `parent = None`, each subsequent block uses
  the previous block's fingerprint as its parent. Different radix
  paths with the same tail tokens now hash to different fingerprints
  — what §5.1 of the project doc requires for cross-node reuse.
- `KVFormat::stable_tag() -> u8` locks the wire-level numeric IDs:
  BF16 = 1, INT8 = 3, FP8E4M3 = 4, TurboQuant-{2,3,4} = 10/11/12,
  plus a 32+ fallback for future bit-pair combinations. Unit test
  `stable_tags_are_fixed` guards against silent renumbering breaking
  saved sessions.
- Four new fingerprint tests covering determinism across equivalent
  contexts, model-fingerprint sensitivity, parent-chain sensitivity,
  and non-trivial empty-token hashing.

Verified locally with:
- `cargo test -p infer --release --no-default-features --features no-cuda`
  (271 passed, up from 267)
- `cargo test -p infer --release --no-default-features --features no-cuda fingerprint`
  (9 passed)
- `cargo check -p infer --tests --no-default-features --features cuda,no-cuda`
- `cargo check -p infer --no-default-features --features metal`
- `cargo fmt --all -- --check`
- `cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings`
  (4 pre-existing, 0 new)

## Rule

When replacing a placeholder hash with a cross-process stable one,
(1) pick a hash function with a small transitive-dep footprint —
blake3 without rayon/mmap is tiny; (2) domain-tag every field of the
input chain so length-extension and boundary collisions cannot
happen; (3) keep the old call site compiling via a doc-hidden shim
so the scheduler publish path is not a brittle big-bang migration;
(4) lock wire-level tag values (`KVFormat::stable_tag`) behind a
unit test so a future enum rename cannot silently break saved
sessions.
