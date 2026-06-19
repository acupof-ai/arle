# KV Restore Boundary Prefix Blocks -- pending remote, 2026-06-19

## SLO-shape probed?

N. This was a local correctness/refactor gate only. No pod, H20, or guidellm
run was used, so this entry makes no throughput claim.

## Goal

Make prefix reuse a backend restore-boundary contract instead of an API-layer
model-name branch.

## What Changed

- Added `PrefixBlock::{ResidentPage, DemotedKey}` to the backend seam.
- Replaced the old page-only restore hook with `reusable_prefix_blocks`.
- Made the default restore-boundary verdict fail-closed; pages-only backends opt
  in through `pages_only_reusable_prefix_blocks`.
- Core now mchecks prefix blocks before publish, attach, and tier promote.
- Unusable prefix tails are not cached, not promoted, and not attached.
- Attention kernels still consume resident backend page tables only; demoted
  keys are promoted before attach or truncated.
- CUDA dense Qwen checks resident pages and demoted tier keys; Qwen35/DSv4
  return 0 for page-prefix reuse and use their own state/slot paths.
- Metal checks demoted T2 keys before promote by resolving them to logical
  prefix ids.
- `infer-api` no longer disables prefix cache by model name; backend capability
  owns the decision.

## Local Verification

```bash
cargo fmt --check
git diff --check -- crates/infer-core/src crates/infer-seam/src crates/infer-cuda/src crates/infer-metal/src crates/infer-api/src/loaded.rs crates/infer-hip/src/executor.rs crates/infer-vulkan/src/executor.rs docs/research/2026-06-19-kv-system-best-practices-and-refactor-plan.md
cargo test -p infer-core --release
cargo test -p infer-seam --release
CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
cargo check -p infer-api --release --no-default-features --features metal,no-cuda --lib
```

All passed locally.

## Pending Remote Bench

Run the normal fixed guidellm recipe on the affected production backend before
claiming a performance win. This change is primarily a correctness and
architecture cleanup; expected perf delta is workload-dependent.

## Rule

Core owns prefix lifecycle; backend owns restore proof. If required side state
is missing, return a shorter leading prefix or 0.
