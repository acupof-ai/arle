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
- The descriptor gained `page_size` and a pre-sharded `flat_local_page_ids`
  with per-row `local_page_range`, so backends slice the exact local page
  count a target length needs without pool access. The shard count itself
  comes from the executor's own construction-time `kv_shard_spec()` —
  world/rank is executor state, not per-step batch data, so it never lived
  in the descriptor.
- `KvSlotAccounting` is a strict subset of `KvAllocator`, written as the
  subset: `KvAllocator: KvSlotAccounting`, with alloc/truncate_slot defined
  only on the subset trait. The chain `KvPool: KvAllocator: KvSlotAccounting`
  lets `&mut dyn KvPool` upcast at the engine-core call site — no blanket
  impl, no method-name collisions, no disambiguation at the 26 above-seam
  call sites.
- CUDA (qwen35), Metal, Vulkan, HIP, and the buffered diffusion executor
  all moved to the new signature. Metal's `materialize_slot_from_prefix`
  and `publish_slot` take `page_size` + `slot_pages` directly, so the page
  store no longer needs pool access either.
- Drive-by: fixed a pre-existing compile error in infer-hip/infer-vulkan
  `poll` (match on an already-dereferenced value) that kept both crates
  from building. Neither crate is in any gate — no CI lane and no pre-push
  hook entry compiles them — so the breakage landed and sat unnoticed.
  Their test targets had also rotted (stale 2-arg `submit` calls after the
  signature change); this PR repairs those call sites so
  `cargo test -p infer-vulkan -p infer-hip` compiles and runs. Whether the
  two crates deserve a standing `cargo check` gate or deletion is ckl's
  decision — they are on the design-theses frozen list as experimental.

## Rule

The batch view is built once, above the seam; backends receive data, not
pools. A backend that needs a new pool read during submit gets a field added
to `KvBatchDescriptor` — it does not reach for `dyn KvPool`. Construction-time
executor state (world/rank, device ids) stays on the executor and is read
through its own accessors; the descriptor carries only per-step data.

## Verification

- `cargo check` on `cpu,no-cuda`, `metal,no-cuda`, and the CUDA lint combo
  (`cuda,no-cuda,nccl,deepep`); the Mac hard gate
  `CUDARC_CUDA_VERSION=12080 cargo clippy -p infer-api --release
  --no-default-features --features cuda,no-cuda,nccl,deepep --lib
  -- -D warnings` passes.
- `cargo test -p infer-seam -p infer-core`, `cargo test -p infer-metal`, and
  `cargo test -p infer-vulkan -p infer-hip` pass. The metal-feature
  `ssd_write_through_promotes_released_pages_and_prefix_snapshot`
  test fails identically at the parent commit: a `debug_assert` on
  content-key width in `kv-native-sys` tier keys, pre-existing and outside
  this change.
- No logic moved: the descriptor is populated by the same `KvQuery` calls
  the backends used to make inline, so a GPU correctness gate is not
  required for this step.
