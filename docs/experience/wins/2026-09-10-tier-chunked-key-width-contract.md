# The tier store's chunked key width is enforced at the packer

Date: 2026-09-10 · Lane: lane/d4 · Contract fix after the slot-tier move (PR #265)

## Context

`kv_native_sys::chunk_sub` packs a blob key and its chunk index into one
56-bit sub-key: 16 bits for the index, so the key itself must fit 40
bits. The packer had no assertion, and `infer_seam::TIER_KEY_BITS`
declared the key width as 56 — honest for non-chunked keys, wrong for
every chunked user. `prefix_block_content_key` masked to 56 bits and the
Metal NS_PREFIX stores fed those keys into chunked ops, so for uniform
keys the namespace field was contaminated with probability 1 − 2⁻¹⁶ ≈
99.998%: the isolation guarantee was effectively never in effect, and
the key's effective entropy silently dropped to 40 bits. The qwen35
sidecar had the same shape; its call-site mask landed in PR #265, which
is what surfaced the contract.

## What worked

- **The assert sits on the function that applies the constraint.**
  `chunk_sub` now `debug_assert!`s the 40-bit width itself; the old
  failure point was `tier_key`'s namespace assert, which guards a
  different (56-bit) constraint and only fired as a downstream symptom.
- **The constant matches the packing.** `TIER_KEY_BITS` (56,
  non-chunked) is split from `TIER_CHUNKED_KEY_BITS` (40);
  `prefix_block_content_key` masks to the chunked width, fixing the
  Metal stores at the source with zero Metal code change. The sidecar
  hash references the constant instead of a hardcoded 40.
- **The format change is safe by construction.** Narrower keys change
  what is on disk: V2 records keep their 56-bit-derived keys, and a V3
  lookup can collide with an unrelated record. The prefix payload
  carries its key and rejects a mismatch, but the page payload
  (`decode_page_payload`) has no key check — a collision there feeds
  wrong KV pages into attention with no error. `MANIFEST_MAGIC` bumps
  V2→V3, so a store written under the old width cold-starts instead of
  serving mixed-width records.
- **The source mask has a cost, and it is backstopped, not free.**
  `prefix_block_content_key` feeds both the chunked prefix stores and
  the non-chunked page store (`tier_key(NS_PAGE, key)`), so masking at
  the source narrows the page path from 56 to 40 bits too. Birthday
  collisions rise by 2^16: ~0.78 expected colliding pairs in a 131k-page
  Metal store (32 GB budget) versus ~1e-7 at 56 bits; ~0.09% in a 44k-page
  DSv4 1 TB tier. The absolute risk lands on the one path without a key
  check, which is why the page-payload key validation is the immediate
  follow-up: once a wrong key is a reported error instead of a wrong KV
  page, the residual cost of 40 bits is a spurious recompute, not silent
  corruption. (Masking only at the chunked API entry would keep the page
  path at 56 bits, at the cost of masks in five store functions; the
  source mask was the smaller diff and the key validation makes the
  width choice benign.)

## Rule

A bit-packed contract is only as honest as its narrowest packer: the
assert belongs on the function that applies the constraint, the width
constant must match the packing, and every caller of the packer is in
scope of the audit — not just the one that tripped. When the fix changes
persisted key bytes, bump the format magic rather than arguing the
collision probability is small enough: a silent wrong KV page is worse
than a cold start.

## Net

No baseline: the collision figures are a birthday-paradox analysis over the
stated store sizes, not a measurement; the test counts and diff stat are
mechanical facts.

3 files, +27/-8. kv-native-sys 7/7, infer-seam 8/8, infer-kvspace 12/12
green; Mac CUDA clippy gate exit 0.
