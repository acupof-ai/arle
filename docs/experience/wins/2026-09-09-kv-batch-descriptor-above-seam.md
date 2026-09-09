# KvBatchDescriptor above the seam — all backends, 2026-09-09

> Status: Shipped (compile/test matrix); GPU correctness gate not required — no logic moved

## Context

`BackendExecutor::submit` took `&mut dyn KvPool`, so every backend read pool
state (slot epochs, seq lens, page tables, page size, TP shard) directly
against the host pool. Two drift paths followed: the CUDA executor built its
own `KvBatchDescriptor` below the seam (a second copy of the batch view the
engine could have built once), and Metal never built one at all. The pool's
write surface (`KvAllocator`: alloc, truncate, free, detached-page ops) was
also wider than what backends actually do during a submit.

## What Worked

- `KvBatchDescriptor::from_plan` moved above the seam (engine-core builds it
  once per step); `submit` now takes `&KvBatchDescriptor` + a 2-method
  `KvSlotAccounting` (alloc, truncate_slot). The compiler enforces the exit
  gate: no `dyn KvPool` in any submit signature.
- The descriptor gained `page_size`, the TP `shard` spec, and a
  pre-sharded `flat_local_page_ids` with per-row `local_page_range`, so
  backends slice the exact local page count a target length needs
  (`shard.local_page_count(global_pages)`) without pool access.
- `KvSlotAccounting` is a supertrait of `KvPool` (strict subset of
  `KvAllocator`), so `&mut dyn KvPool` upcasts at the engine-core call site
  and the blanket impl covers every pool. Above-seam direct callers
  disambiguate the colliding method names with `KvAllocator::`.
- CUDA (qwen35), Metal, Vulkan, HIP, and the buffered diffusion executor
  all moved to the new signature. Metal's `materialize_slot_from_prefix`
  and `publish_slot` take `page_size` + `slot_pages` directly, so the page
  store no longer needs pool access either.
- Drive-by: fixed a pre-existing compile error in infer-hip/infer-vulkan
  `poll` (match on an already-dereferenced value) that kept both crates
  from building outside CI.

## Rule

The batch view is built once, above the seam; backends receive data, not
pools. A backend that needs a new pool read during submit gets a field added
to `KvBatchDescriptor` — it does not reach for `dyn KvPool`.

## Verification

- `cargo check` on `cpu,no-cuda`, `metal,no-cuda`, and the CUDA lint combo
  (`cuda,no-cuda,nccl,deepep`); the Mac hard gate
  `CUDARC_CUDA_VERSION=12080 cargo clippy -p infer-api --release
  --no-default-features --features cuda,no-cuda,nccl,deepep --lib
  -- -D warnings` passes.
- `cargo test -p infer-seam -p infer-core` and `cargo test -p infer-metal`
  pass. The metal-feature `ssd_write_through_promotes_released_pages_and_prefix_snapshot`
  test fails identically at the parent commit: a `debug_assert` on
  content-key width in `kv-native-sys` tier keys, pre-existing and outside
  this change.
- No logic moved: the descriptor is populated by the same `KvQuery` calls
  the backends used to make inline, so a GPU correctness gate is not
  required for this step.
