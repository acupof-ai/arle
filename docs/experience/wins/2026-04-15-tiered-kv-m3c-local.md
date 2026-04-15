# 2026-04-15 · Tiered KV M3c local cleanup

## Context

After the M2b/M0.3/M3a/M3b local batches, the CUDA tree still carried one
major architectural liability: the old contiguous `model/kv_cache.rs`
CPU-offload path (`k_host/v_host`, `OFFLOAD_BLOCK_SIZE = 64`,
`prefetch_to_gpu`, `offload_if_needed`) coexisted with the paged-pool /
tiered-KV design. That meant two local truths for KV residency even though
only one of them should survive into T1 host-pinned promotion.

The M3c local cleanup tranche retired that legacy path without mixing in any
new scheduler staging behavior.

## What Worked

- Deleted the production contiguous CPU-offload implementation from
  `infer/src/model/kv_cache.rs` and removed all scheduler / model /
  single-request consumers of the old `prefetch/offload` hooks.
- Kept `set_max_gpu_kv` only as a compatibility no-op warning in
  `server_engine` and the CLI surface so external callers do not break in
  the same batch that deletes the old runtime behavior.
- Rewrote `tests/test_kv_cache.py` so it mirrors the resident-only metadata
  that still exists, instead of pinning an obsolete offload state machine.
- Verified locally with:
  - `cargo test -p infer --no-default-features --features no-cuda server_engine`
  - `cargo check -p infer --tests --no-default-features --features cuda,no-cuda`
  - `cargo check -p infer --no-default-features --features metal`
  - `python -m pytest tests/test_kv_cache.py -v`
  - `cargo fmt --all -- --check`

## Rule

When a tiered-KV migration deletes a legacy residency path, keep the batch
strictly subtractive: remove the old producer/consumer hooks first, keep any
public compatibility shim as a no-op, and defer new runtime staging behavior
to a separate commit.
