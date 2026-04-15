# Tiered KV Tier B fingerprint scaffolding local

## Context

Tier B adds the first local fingerprint plumbing for the tiered KV cache so
M4 persistence can identify full radix blocks by content instead of by
ephemeral pool slot id.

## What Worked

`BlockFingerprint::compute_from_tokens` now exists as a local placeholder
hash, radix inserts can stamp per-block fingerprints without changing
`lookup_or_stage`, serde restore has a post-load reconciliation helper, and
the CUDA publish path computes fingerprints at prompt-block granularity before
inserting into the prefix cache. Disk-store tests now verify that a block's
bytes and derived fingerprint survive a local round trip.

## Rule

Land identity scaffolding before the persistence behavior: keep the local hash
process-stable, thread fingerprints through existing publish/insert paths, and
defer the final cross-process hash choice plus block-index rebuild to the
later M4/Tier D merge.
