# Metal page payloads validate their content key at decode

Date: 2026-09-10 · Lane: lane/d5 · Follow-up to the chunked key-width contract fix (PR #266)

## Context

The tier store's page path (`NS_PAGE`, non-chunked) decoded whatever bytes
sat under the key: `decode_page_payload` took no key and returned arrays
unconditionally. The prefix path already echoed its content key into the
payload and rejected a mismatch (`decode_prefix_payload`). The asymmetry
mattered after PR #266 narrowed content keys to 40 bits: a collision in
the page path — or any stale record under a foreign key — would decode
straight into KV pages served to attention, with no error. This is the
follow-up the #266 wins entry named: with the check in place, a collision
is a reported error and a recompute, not silent corruption.

## What worked

- **Same framing as the prefix path.** `encode_page_payload` prepends the
  content key; `decode_page_payload` reads it back and errors on
  mismatch. Both call sites (`write_page`, `read_tier_page`) already had
  the key in hand, so the change is codec-only. The 8-byte head fits
  inside the existing `PAGE_CODEC_HEADROOM` (64 KiB).
- **The test covers the write path, not just the decode.** An oversized
  insert fails silently — `warn!` + `return false`, the KV layer stops
  caching, and a decode-only test stays green in a world where nothing
  is written. The second test asserts `write_page` returns true and the
  record is readable under `tier_key(NS_PAGE, key)`: a positive control
  on the same principle as "grep == 0 needs a positive control".
- **Format bump.** `MANIFEST_MAGIC` V3→V4, since the page payload layout
  changed; a V3 page record now decodes to a wrong-key error instead of
  arrays.

## Rule

A decode that has the requested key in hand should prove the record
belongs to it — the check is one comparison, and its absence converts a
collision (rare, reportable) into silent wrong data (rare, invisible).
When a test guards a codec, assert the write landed too: a silent refusal
upstream makes every downstream assertion vacuous.

## Net

No baseline: correctness fix, no measurement; test counts and diff stat
are mechanical facts.

2 files, +53/-8. infer-metal 3/3 green (2 new tests); Mac CUDA clippy
gate exit 0.
