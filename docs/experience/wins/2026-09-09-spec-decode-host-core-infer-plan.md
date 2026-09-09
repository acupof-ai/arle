# Spec-decode host core moved to infer-plan

Date: 2026-09-09
Scope: `crates/infer-plan/src/spec.rs` (new), `crates/infer-cuda/src/{dsv4.rs,dsv4/spec_verify.rs,dsv4/mtp.rs,executor/spec_decode.rs}`

## Context

`executor/spec_decode.rs` (369 lines) lived in infer-cuda, so its pure host
state machine — draft-chain accept/rollback — required a GPU build to test.
The move brief asked whether the file's `crate::attention` / `crate::dsv4`
dependencies were sinkable types or real device operations.

## What worked

The file splits in two. The pure core (~180 lines: `SpecKind`, `DecodeRoute`,
`route_decode`, `DraftChain` with `accept_path`/`validate`/`verify_schedule`,
`SpecVerifySchedule`, `MtpDraftRow`, the four spec constants) moved to
`infer-plan::spec`. The executor half (`spec_step`, `draft_chain`) stays in
infer-cuda: it orchestrates real device operations through `self.model` /
`self.slots` — `forward_tokens_verify_scheduled`, `mtp_forward_level`,
`capture_spec_rings`, `truncate_slot`, `restore_spec_ring_tail`,
`commit_accepted_fold`.

The cut is by pipeline stage, not model family: `SpecKind` is referenced by
both the qwen35 and dsv4 executors, so it is step-level scheduling, not a
dsv4 concept. infer-cuda re-exports the moved items (`crate::dsv4::*`,
`executor::spec_decode::*`), so call sites are unchanged.

Tests: 4 unit tests in infer-plan covering route gating, full-path accept,
rollback at first mismatch (top-k sibling hit stops the path at its parent),
and chain validation/verify scheduling. The rollback test goes red when the
accept condition is weakened (`&&` → `||`), verified by mutation.

`cargo test -p infer-plan --profile release-fast` runs on Mac standalone, no
GPU, no infer-cuda.

## The crate::attention verdict

- `set_dsv4_verify_frozen` / `dsv4_verify_frozen` stay in infer-cuda. The flag
  is a host `AtomicBool`, but device code reads it — its consumers are
  silicon-side, so the type belongs with the backend.
- `Dsv4PrefixPageEntry` + its `to_bytes`/`from_bytes` codec (in
  `attention/prefix_state.rs`) are pure host byte math — movable. That half
  of the move targets the new `infer-kvspace` crate and is sequenced after the
  architecture refactor doc (`docs/plans/2026-09-09-architecture-refactor.md`)
  lands; the executor half of `prefix_pool.rs` (fence polling, page capture,
  `mirror_full_band`) stays regardless.

## Rule

The boundary is the dependency graph, not "zero device calls". A pure host
state machine mounted on `impl Dsv4CudaExecutor` passes the weak test — both
move-target files had zero raw cudarc symbols while orchestrating real device
work through `self.model`. The enforceable criterion: `cuda-kernels` must not
appear in the crate's `cargo tree` (`cargo tree -p infer-plan | grep -c
cuda-kernels` = 0), compiler-verified, not discipline-dependent.
