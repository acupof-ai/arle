# DSv4 Canonical Batched Decode Cleanup -- pending remote, 2026-06-19

## SLO-shape probed?

N. This is a local code/compile gate only; no H20/pod GPU bench was run in this
pass. The remote bench remains required before claiming a throughput win.

## Goal

Delete obsolete DSv4 B>1 decode/spec toggles and leave one canonical MODEL1
batched decode path: B=1 single-row reference, B>1 batched FlashMLA sparse decode.

## Hypothesis

Removing the stale fallback switches reduces path ambiguity without changing the
intended MODEL1 B>1 behavior: topk widens draft candidates only, while verify
rows remain chain-shaped.

## What Changed

- Deleted the `ARLE_DSV4_FLASHMLA_DECODE_BATCHED` runtime gate and its atomic
  override; batched decode now inherits the FlashMLA decode gate.
- Deleted the `ARLE_DSV4_DECODE_COMPRESSOR_BATCH` path gate; compressor/indexer
  projection pre-pass is direct on the MODEL1 B>1 lane.
- Kept V32/GLM off the MODEL1 batched FlashMLA/full-flatten lane by not
  allocating batch scratch for non-512 head dims and by excluding
  `SparseIndexed` from full-flatten.
- Removed the scheduled MTP verify single-token plain-forward fallback; scheduled
  sparse verify now requires the MTP chain shape (`depth + 1 >= 2` rows).
- Locked D2/K2 semantics in unit tests: candidate width does not expand verify
  rows; D2 verifies 3 rows.

## Local Verification

```bash
cargo fmt --check
git diff --check -- crates/infer-cuda/src/attention.rs crates/infer-cuda/src/dsv4.rs crates/infer-cuda/src/executor.rs crates/infer-cuda/src/executor/spec_decode.rs
CUDARC_CUDA_VERSION=12080 cargo test -p infer-cuda d2k2 --release --no-default-features --features cuda,no-cuda
CUDARC_CUDA_VERSION=12080 cargo check -p infer-cuda --release --no-default-features --features cuda,no-cuda --lib
CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
CUDARC_CUDA_VERSION=12080 cargo clippy -p infer-cuda --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
```

All passed locally.

## Pending Remote Bench

Run one controlled DSv4 H20 bench with profiling off and the normal fixed bench
recipe. Required comparisons:

- Normal decode B>1 vs latest baseline.
- MTP D2/T2 with topk=1 and topk=2.
- Confirm no incomplete/errored requests.
- Capture service stats and scheduler/KV counters.

## Rule

Once a DSv4 runtime lane becomes canonical, delete the stale env-gated fallback
and keep model-shape exclusions explicit at the call site.
