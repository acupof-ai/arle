# KV Tier Zig Substrate Plan

## Context

`infer/src/kv_tier/*` already has the right high-level split:

- Rust policy and orchestration in `lookup.rs`, `coordinator.rs`, `tier.rs`, and `transport.rs`
- a concrete local disk substrate in `transport/disk.rs`
- pinned-host bookkeeping in `host_pool.rs`

The right Zig insertion point is the **low-level persistence / descriptor substrate**, not the scheduler-facing control plane.

## Boundary

Rust keeps:

- `KVTransport`
- `CoordinatorCommand` / `CoordinatorEvent`
- `StagePlanner` / `LookupOutcome`
- `BlockLocation`, `Tier`, `RemoteBlockDesc`
- session persistence and scheduler policy

Zig owns, in phases:

1. local file engine for `DiskStore`
2. object-store primitives
3. mmap segment and descriptor handling
4. WAL append / replay
5. shm-backed local object store
6. future fd-passing / zero-copy IPC hooks

## Phases

Status legend:

- `done`: landed in-tree and locally validated
- `active`: current implementation target
- `planned`: designed, not started

### Phase order

1. Phase 1: file-engine extraction
2. Phase 2: object-store extraction
3. Phase 3: mmap + WAL substrate
4. Phase 4: shm + descriptor substrate
5. Phase 5: coordinator / transport integration

### Phase 1

Status: `done`

Goal: keep the current `DiskStore` API stable while moving low-level file operations into Zig.

Scope:

- add `crates/kv-native-sys`
- compile a Zig static library from Cargo
- expose a minimal C ABI for write / atomic-write / read / remove
- convert `infer/src/kv_tier/transport/disk.rs` into a Rust adapter over that ABI

Non-goals:

- no `KVTransport` implementation change
- no scheduler behavior change
- no `HostPinnedPool` migration yet
- no WAL or mmap in this tranche

Ordered tasks:

1. add `crates/kv-native-sys`
2. compile Zig from Cargo build scripts
3. expose thin file I/O C ABI
4. route `DiskStore::write/read/remove` through the Rust adapter
5. preserve `DiskStore` block API and caller surface
6. validate `no-cuda` and `metal` lanes locally

Completed in-tree:

- `crates/kv-native-sys`
- Zig toolchain invocation from `build.rs`
- `scripts/setup_zig_toolchain.sh`
- `scripts/check_kv_zig.sh`
- `DiskStore` file I/O routed through Zig
- local `infer` no-cuda + metal checks
- local `disk.rs` tests
- local `kv-native-sys` check + clippy

Acceptance:

- `DiskStore` public Rust API is unchanged
- existing `disk.rs` tests still pass
- `http_server/sessions.rs` and scheduler callers do not need code changes
- the workspace builds with the Zig toolchain installed

### Phase 2

Status: `active`

Goal: add a real object-store core under the same crate.

Scope:

- content-addressed object routing
- canonical block naming and path generation
- block read/write/remove ABI keyed by `(root, fingerprint)`
- reduce `disk.rs` to header/policy adapter logic

Ordered tasks:

1. move block filename generation into Zig
2. expose block path/read/write/remove ABI in `kv-native-sys`
3. route `DiskStore::block_path_for` through the native substrate
4. route `put_block/get_block/delete_block` through the native substrate
5. keep `DiskBlockLocation` stable for sessions/coordinator code
6. leave keyed `write/read/remove` API and header validation stable
7. re-run local checks and tests

Current landing target:

- native block naming and object operations move to Zig
- `disk.rs` stops owning production block filename generation
- Rust still owns `DiskBlockHeader`, semantic validation, and path tamper checks
- toolchain/bootstrap and local validation scripts are repository-native

Acceptance:

- `DiskStore` block API stays source-compatible
- `DiskBlockLocation` stays unchanged
- local `kv_tier::transport::disk` tests remain green
- `no-cuda` and `metal` checks remain green
- `scripts/check_kv_zig.sh` runs clean locally

### Phase 3

Status: `planned`

Goal: add mmap and WAL.

Scope:

- append-only WAL segments
- replay on open
- committed object index
- crash-safe commit protocol beyond `rename(.tmp)`

Ordered tasks:

1. define WAL record schema for keyed objects and block objects
2. add mmap segment abstraction in `kv-native-sys`
3. add WAL writer with append + fsync boundaries
4. add WAL replay at object-store open
5. move crash-consistency policy out of `disk.rs`
6. add corruption / partial-write tests

Acceptance:

- crash recovery replays committed records and ignores torn ones
- `DiskStore` keeps the same Rust API while using WAL-backed persistence underneath
- local tests cover replay, truncation, and bad-record handling

### Phase 4

Status: `planned`

Goal: add shm-backed local object store and stable descriptors.

Scope:

- shared-memory segments
- offset/len/generation descriptors
- local zero-copy handoff hooks
- future fd export surface

Ordered tasks:

1. define stable descriptor PODs for shm objects
2. add shared-memory segment allocation and open/close
3. add descriptor-to-mapping resolution
4. add local export/import hooks for same-host consumers
5. thread descriptor types back into the Rust adapter surface

Acceptance:

- shared-memory objects can be created, reopened, and validated by descriptor
- descriptor structs are Rust/FFI-safe and documented
- no scheduler-facing API break yet

### Phase 5

Status: `planned`

Goal: wire the coordinator to the substrate without changing high-level contracts.

Scope:

- `CoordinatorCommand::Spill` / `Rehydrate` drive the Zig substrate
- `DiskStore` remains a compatibility shim for sessions and tests
- transport implementations can start consuming descriptor handles instead of only paths

Ordered tasks:

1. add coordinator-side descriptor/object-store plumbing
2. add spill path from Rust coordinator into Zig object store
3. add rehydrate path back into Rust-controlled staging flow
4. keep `CoordinatorCommand` / `CoordinatorEvent` stable where possible
5. add transport hooks that can graduate from path-based to descriptor-based flows

Acceptance:

- coordinator can spill and rehydrate through the Zig substrate
- session persistence compatibility remains intact
- `KVTransport` evolution is additive, not a flag day

## Follow-up

Immediate next steps, in order:

1. finish the Phase 2 block object-store extraction
2. mark toolchain scripts + env/docs sync complete
3. land Phase 3 WAL schema design before touching coordinator behavior

Remote follow-up:

- run `scripts/bench_guidellm.sh kv-tier-zig-phase1` on the CUDA host
- replace the `pending-remote` stub under `docs/experience/wins/`

## Toolchain and validation

Status: `done` for local lanes, `pending-remote` for CUDA benchmark validation.

Delivered:

1. `scripts/setup_zig_toolchain.sh` validates or installs Zig `0.16.0`
2. `scripts/check_kv_zig.sh` runs the local validation sequence
3. `docs/environment.md` documents `ZIG`
4. `docs/codebase-map.md` records `crates/kv-native-sys`

Local validation completed:

1. `cargo check -p kv-native-sys`
2. `cargo clippy -p kv-native-sys -- -D warnings`
3. `cargo check -p infer --no-default-features --features no-cuda`
4. `cargo check -p infer --no-default-features --features metal`
5. `cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings`
6. `cargo test -p infer --no-default-features --features no-cuda kv_tier::transport::disk -- --nocapture`
