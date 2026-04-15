# 2026-04-16 · Tiered KV M4d local session snapshot module

## Context

M4d adds the pure session snapshot/load helpers on top of the M4a/M4b/M4c
building blocks: stable `BlockFingerprint` computation, disk block
round-tripping, and radix reconciliation onto fresh `BlockId`s.

## What Worked

`infer::http_server::sessions` now saves a `RadixCache` snapshot plus a
manifest of persisted disk blocks without wiring any HTTP route. Loading the
snapshot rehydrates payload bytes from `DiskStore`, remaps fingerprints onto
fresh block ids through `RadixCache::reconcile`, and reports missing or
tampered disk files explicitly. The local tests cover the happy path, the
intentional "skip blocks without payload" case, and tampered disk data.

## Rule

Session persistence should treat the radix snapshot and the block manifest as
separate concerns: the radix structure can round-trip through serde, but every
allocator-local `BlockId` must still be re-minted from persisted
`BlockFingerprint`s at load time.
