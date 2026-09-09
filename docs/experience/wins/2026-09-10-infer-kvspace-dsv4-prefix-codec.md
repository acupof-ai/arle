# infer-kvspace: DSv4 prefix-page codec + capture lifecycle off infer-cuda

Date: 2026-09-10 · Lane: lane/d · Step 2 of the architecture refactor

## Context

The refactor slices the runtime by pipeline stage. `infer-kvspace` is the new
home for KV host logic that needs no GPU to verify — capacity accounting,
content indexing, layout description — while the kernel-operand role (the page
table layout the attention kernel reads) stays in `infer-cuda`. This is the
first batch: the DSv4 per-page entry codec (`Dsv4PrefixPageEntry` +
`Dsv4LayerPageState`, byte framing with a magic that doubles as format version)
and the `PendingPrefixPage` lifecycle (confirm / cancel-provisional / repair
across the radix dedup between capture and publish), plus the two pure
predicates `capture_epoch_matches` and `rekey_target_conflicts`. The types live
in a `dsv4` module inside the crate: the crate boundary is the pipeline stage,
modules inside may follow model families.

The acceptance gate is the dependency graph, not a call audit:
`cargo tree -p infer-kvspace | grep -c cuda-kernels` must be 0. A "zero device
calls" rule is too weak — a pure host state machine hanging off
`impl Dsv4CudaExecutor` passes it. The graph criterion is compiler-enforced.

## What worked

- **Re-export shim, zero call-site churn.** `pub(crate) use infer_kvspace::{...}`
  in `attention/prefix_state.rs` keeps every `crate::attention::Dsv4PrefixPageEntry`
  path valid; the five `dsv4/slot.rs` sites were untouched.
- **Accessors over moved struct literals.** `PendingPrefixPage` gained a
  constructor and read accessors; the two publish loops changed literals and
  field reads to method calls, and the two `cancelled = true` sites became
  `cancel()`. The lifecycle transitions now have exactly one writer each.
- **Mutation negative control on both arms.** Flipping the boundary decode
  (`bytes[8] != 0` → `== 0`) fails the codec round-trip test; flipping
  `confirm`'s `|=` to `&=` fails the lifecycle test. Reverted via reverse sed,
  4/4 green after.
- **Gates wired on both sides.** `pre_push_checks.sh` and the CI test-neutral
  lane carry the same crate list — the hook list drifts from CI otherwise.

## Boundary decisions (from the e2 sync on the qwen35.rs ~300 lines)

- `KvSlotAccounting` stays in `infer-seam`: it is the engine↔backend capacity
  contract the planner already uses. Moving it into infer-kvspace would make
  infer-core's submit path depend on infer-kvspace. infer-kvspace builds on it.
- The tier/sidecar lifecycle splits host policy (which boundaries to snapshot,
  key derivation, chunking, eviction, the page-key map — infer-kvspace) from
  the device adapter (snapshot↔device bytes, the `tp_min_usize` min-reduce —
  backend). Only three of the nine functions touch device. The cut is e2's
  proposal and pending 34's confirmation before the interface solidifies.

## Rule

Crate boundaries follow pipeline stages; modules inside a crate may follow
model families. The gate for a "host-only" crate is `cuda-kernels` absent from
its cargo tree — compiler-enforced, not discipline-enforced.

## Next

`reusable_prefix_blocks` (the prefix-match upper bound over page meta +
frontier tail) moves as a trait with a fake in-memory adapter and the real
infer-cuda adapter, per the refactor's Step 2 contract-test gate. It rides the
same host-policy/device-adapter cut and waits on 34's confirmation.
