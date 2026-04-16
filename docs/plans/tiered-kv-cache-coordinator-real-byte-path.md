# Tiered KV Cache — coordinator real byte path (M3 runtime completion + T1↔T2 spill)

**Status**: Execution plan. Locally blocked (requires real CUDA to
validate `cudaMemcpyAsync` completion paths); intended to be
executed in the remote L4 window after M4 a/b/c/d remote
acceptance signs off.

**Scope**: replace the local coordinator stub — which today echoes
`StagingCompleted` synchronously for every `Stage` command — with
a real byte-moving transport path. Two tiers are in scope:

1. **T0 ↔ T1** (GPU HBM ↔ host pinned DRAM) via
   `cudaMemcpyAsync` on a dedicated copy stream. This is the M3
   runtime completion half that has been deferred since the
   original M3a/M3b contract tranche.
2. **T1 ↔ T2** (host pinned DRAM ↔ NVMe via `DiskStore`) via the
   `DiskStore::put_block` / `get_block` path that M4b already
   proved byte-faithful, but is not wired to the coordinator.

**Out of scope** for this batch:
- Real NIXL / RDMA cross-node transport (that is M5, trigger-
  only).
- Metal backend tier participation — Metal stays T0-only until
  the MLX wired-memory bindings batch lands (separate doc).
- Cross-slot page aliasing — same safe-same-slot constraint as
  M2b still applies.
- HTTP route wrappers for session save/load — covered by
  `tiered-kv-cache-m4e-session-http-routes.md`.

---

## 1 · What exists today (2026-04-16)

On `main` after the Tier A/B/C + M4 batches:

- `Scheduler::with_config` spawns a `Coordinator` OS thread named
  `infer-tiered-kv-coord` (`infer/src/kv_tier/coordinator.rs`).
  The thread's `run_once` drains its `bounded` command channel and
  emits events back to the scheduler.
- Scheduler admission calls `lookup_or_stage(..., Some(&handle))`;
  a staged hit stores the request in `stage_waiting` and the
  scheduler's `drain_coordinator_events` re-admits on
  `StagingCompleted`.
- The coordinator's **current** Stage handler synchronously
  emits `StagingQueued` **and** `StagingCompleted` for every
  `Stage` command. No byte movement happens. The `u32::MAX` slot
  sentinel in `drain_coordinator_events` flags the flipped
  `BlockLocation::Gpu { slot }` as "ready but not yet owned by a
  real slot" — the next `lookup_or_stage` sees it as ReadyOnGpu.
- `LocalCudaTransport::poll()` is a stub that errors out on
  non-CUDA targets; its CUDA-lane implementation has never
  exercised a real copy.
- `HostPinnedPool` exists structurally but never actually owns a
  `cudaHostAlloc`'d region. `host_pool.rs` has
  `HostPinnedRegion { ptr_addr, len, kind }` but nothing
  populates it.
- `DiskStore::put_block` / `get_block` work end-to-end (M4b),
  but zero coordinator code calls them.

So the **three things that need to actually happen** for this
batch to ship:

1. `HostPinnedPool` allocates real pinned memory.
2. `LocalCudaTransport::poll()` submits `cudaMemcpyAsync` on a
   dedicated copy stream and polls for completion.
3. The coordinator's `Stage` handler submits to the transport
   instead of echoing, and polls each `run_once` tick for
   completed ops to emit `StagingCompleted`.

Plus the T1↔T2 half:

4. A new `CoordinatorCommand::Spill` / `Rehydrate` variant that
   the scheduler's watermark path fires when T1 retained fraction
   exceeds a threshold.
5. The coordinator routes those through `DiskStore::put_block` /
   `get_block` asynchronously (std::thread or a second copy
   stream — see §4 open questions).

---

## 2 · Files to touch

### `infer/src/kv_tier/host_pool.rs`

Real pinned allocation:

- `HostPinnedPool::new(total_bytes: usize, device_ctx: &DeviceContext)`:
  call `cudaHostAlloc(&ptr, total_bytes, cudaHostAllocPortable |
  cudaHostAllocMapped)`. Store the base pointer + a bump
  allocator.
- `fn allocate_region(&self, len: usize) -> Option<HostPinnedRegion>`:
  bump-allocate from the base region. Returns `None` if full.
- `fn free_region(&self, region: HostPinnedRegion)`: for the
  initial cut, implement as a free-list append; do NOT call
  `cudaFreeHost` per-block (the whole pool is a single
  allocation).
- `Drop for HostPinnedPool`: single `cudaFreeHost` on the base
  pointer. Document the "registered memory region must not
  reallocate" invariant from the project doc §4.2.

### `infer/src/kv_tier/transport/local_cuda.rs`

Real `cudaMemcpyAsync`:

- Add a `copy_stream: CudaStream` field populated at
  construction via `cudaStreamCreateWithFlags(&stream,
  cudaStreamNonBlocking)`. Dedicated stream, not the default.
- `submit(ops: Vec<TransferOp>) -> Result<LocalCudaOp>`:
  - For each op, dispatch `cudaMemcpyAsync(dst, src, len,
    direction, copy_stream)` where direction is derived from
    `(op.src.kind, op.dst.kind)`:
    - `(Gpu, HostPinned)` → `cudaMemcpyDeviceToHost`
    - `(HostPinned, Gpu)` → `cudaMemcpyHostToDevice`
    - anything else → error
  - Record a `cudaEventRecord(event, copy_stream)` after the
    last op so we can poll completion later.
  - Return a `LocalCudaOp { ops, state: Pending, event }`.
- `poll(&self, op: &mut LocalCudaOp) -> Poll<Result<()>>`:
  - `cudaEventQuery(op.event)` — if `cudaSuccess`, return
    `Poll::Ready(Ok(()))` and the coordinator moves on.
  - If `cudaErrorNotReady`, return `Poll::Pending`.
  - Any other error → `Poll::Ready(Err(TransportError::Other(..)))`.
- Drop: destroy event + stream.

All of the above lives behind `#[cfg(feature = "cuda")]` on the
method bodies; the no-cuda stubs return deterministic errors so
unit tests don't explode.

### `infer/src/kv_tier/coordinator.rs`

New command variants + real handling:

```rust
pub enum CoordinatorCommand {
    // ... existing: Demote, Promote, Stage, Shutdown ...
    /// T0 → T1 stage (existing). Handler now submits real copies.
    Stage { ticket: StageTicket, requests: Vec<StageRequest> },
    /// T1 → T2 spill: take blocks out of HostPinnedPool and write
    /// them to DiskStore. Coordinator picks its own DiskStore
    /// handle from construction.
    Spill { ticket: StageTicket, blocks: Vec<SpillRequest> },
    /// T2 → T1 rehydrate: read a DiskStore file into HostPinnedPool.
    Rehydrate { ticket: StageTicket, blocks: Vec<RehydrateRequest> },
}

pub struct SpillRequest {
    pub block_id: BlockId,
    pub fingerprint: BlockFingerprint,
    pub kv_format_tag: u8,
    pub host_region: HostPinnedRegion,
}

pub struct RehydrateRequest {
    pub block_id: BlockId,
    pub fingerprint: BlockFingerprint,
    pub disk_location: DiskBlockLocation,
    pub host_region: HostPinnedRegion,
}
```

The `Coordinator` struct grows:

```rust
pub struct Coordinator {
    rx: Receiver<CoordinatorCommand>,
    events: Sender<CoordinatorEvent>,
    transport: Arc<LocalCudaTransport>,
    host_pool: Arc<HostPinnedPool>,
    disk: Arc<DiskStore>,
    in_flight: HashMap<StageTicket, InFlightOp>,
}

enum InFlightOp {
    CudaCopy { ops: LocalCudaOp },
    DiskIo(std::thread::JoinHandle<io::Result<Vec<DiskBlockLocation>>>),
}
```

`run_once` becomes a two-phase loop: (1) try to drain a new
command; (2) for every ticket in `in_flight`, poll its
`InFlightOp` and emit `StagingCompleted` on success / error.

### `infer/src/scheduler/cuda/core.rs`

- `watermark_check_for_t1_spill`: when the host-pinned pool's
  retained fraction crosses an analogous high watermark (start
  with 0.85 → 0.70 as defaults on `SchedulerConfig`; add
  `t1_host_pinned_high_water: f64` + `_low_water` the same way
  Tier C promoted the T0 watermarks), select eviction victims
  and send `CoordinatorCommand::Spill` for each. Schedule this
  in the existing `cleanup()` pass alongside
  `evict_prefix_cache_if_pressured`.
- `publish_to_prefix_cache`: after inserting a radix block,
  mint a HostPinnedRegion for it and record the region on the
  radix node's `tier_location` once the T0→T1 stage fires.
- `drain_coordinator_events`: new arms for
  `CoordinatorEvent::SpillCompleted { ticket, locations }` and
  `CoordinatorEvent::RehydrateCompleted { ticket, host_regions }`.
  Update radix node `tier_location` accordingly.

### `infer/src/scheduler/types.rs`

Extend `SchedulerConfig` with the two new watermarks + one new
keepalive:

```rust
pub t1_host_pinned_high_water: f64,    // default 0.85
pub t1_host_pinned_low_water: f64,     // default 0.70
pub t1_host_pinned_keepalive_ticks: u64, // default 128
```

Add validation checks in `validate()` mirroring the T0 watermark
checks.

---

## 3 · Work items

In dependency order:

1. **Real `HostPinnedPool`**. Must allocate, hand out regions,
   free via Drop. Unit test under `--features cuda`: allocate
   1 MiB, grab a region, return it, drop the pool, assert no
   leak via `cudaMemGetInfo` before/after.
2. **Real `LocalCudaTransport`**. `submit` + `poll` on a
   dedicated stream, event-polled completion. Unit test under
   `--features cuda`: copy a 4 KiB pattern H→D→H, assert round
   trip.
3. **Coordinator `Stage` handler swaps echo → real transport**.
   Test: single-request long-session replay, assert the
   `drain_coordinator_events` path actually waits for the
   `cudaEventQuery` to return `Ready` (not instantaneous).
4. **Scheduler pool-spill watermark trigger**. Test: admit
   enough distinct prefixes to cross the 0.85 T0 watermark,
   assert T1 grows, assert scheduler emits `Spill` commands.
5. **Coordinator `Spill` / `Rehydrate` handlers**. Test:
   force-spill a block, delete it from T1, rehydrate, assert
   byte-equal round trip via `DiskStore::get_block` with
   fingerprint validation.
6. **T1→T2 watermark trigger** (mirror of #4 but on the host
   pinned pool).
7. **Long-session bench**: restart a 30k-token system prompt
   session through `save` / `load` + this coordinator path,
   measure TTFT drop vs first-turn baseline.

---

## 4 · Open questions for the executor

1. **One stream or two for the copy path?** Project doc §4.4
   says "two dedicated copy streams" (one for H→D, one for
   D→H). Start with ONE bidirectional stream for simplicity;
   split only if the bench shows contention.
2. **Disk I/O threading**: run `DiskStore::put_block` /
   `get_block` on a std::thread spawned per-command, or on a
   bounded worker pool? For v1, **per-command thread** is
   fine — disk I/O is not latency-critical and the thread
   cost is amortized over the copy time.
3. **`HostPinnedPool` backpressure**: what happens if every
   Stage command fills the pinned pool? Answer:
   `submit_stage` returns `Err(TransportError::Other("pool
   exhausted"))`, the coordinator emits a failed completion
   event, the scheduler cold-requeues the request
   (analogous to Tier A's disconnect-drain path).
4. **Event pool size**: `cudaEventCreate` isn't free; cache
   events in a `Vec<CudaEvent>` free-list on the transport.
5. **Spill eviction policy**: reuse `SessionBiasedLru` (the
   same policy that drives T0 eviction) — the eviction
   candidate struct already carries `session_id` and
   `soft_pin_until`. Just change the signal source from
   "retained_t0 pages" to "host_pinned retained bytes".
6. **Page-lifecycle drift**: the existing `PageLifecycle`
   state machine (Free → Resident → Demoting → Free) was
   designed to include real demote events. This batch turns
   the sentinel `StageTicket(u64::MAX)` instant-demote path
   from Tier A into real demote transitions. Audit
   `page_lifecycle.rs` once you wire it up.

---

## 5 · Acceptance

```bash
# Unit + module tests
cargo test -p infer --release host_pool
cargo test -p infer --release coordinator
cargo test -p infer --release local_cuda

# Integration — requires CUDA host
CUDA_HOME=/usr/local/cuda cargo test --release --test tier_promotion_end_to_end

# Full matrix
CUDA_HOME=/usr/local/cuda cargo build --release
cargo test --release
INFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e
INFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e_qwen35
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

Sign-off requires:

- [ ] A long-agent-session benchmark (32k+ cumulative tokens,
      num_slots=4) that OOMs on pre-M3 runs **completes** on
      this batch, with the full T0→T1→T2→T1→T0 round trip
      exercised (verify via log counts on `SpillCompleted` /
      `RehydrateCompleted` events).
- [ ] Restart smoke test: save a 30k-token system-prompt
      session, kill the process, restart, reload, measure
      TTFT. Must land within **20%** of the pre-restart
      warm-state baseline.
- [ ] Steady-state decode throughput regression ≤ **3%**
      versus the Tier A/B/C + M4 pre-runtime-path baseline
      (`docs/experience/wins/2026-04-16-tiered-kv-tier-abc-remote.md`).
- [ ] Demote/cancel-on-hit race test: fire a `lookup` against
      a block currently in `PageLifecycleState::Demoting` and
      confirm the lookup either (a) waits and returns the
      block, or (b) cancels the demote and returns the T0
      location. Never returns stale bytes.
- [ ] New remote acceptance doc at
      `docs/plans/tiered-kv-cache-coordinator-real-byte-path-remote-acceptance.md`
      filed with the L4 sweep results.

---

## 6 · Why this is one plan, not two

It is tempting to split T0↔T1 from T1↔T2 into two separate
batches — and that is a valid execution order. The reason this
plan bundles them is that **the coordinator's `InFlightOp` +
polling loop is the same shape for both**, and splitting means
writing the poll loop twice and then merging it. Doing both in
one pass is cleaner even if the T1↔T2 half lands a week later
than the T0↔T1 half on the same branch.

If the executor decides to split anyway:

- **Phase A**: Work items 1–3 (real `HostPinnedPool`, real
  `LocalCudaTransport`, coordinator `Stage` swap). Land +
  sign off independently.
- **Phase B**: Work items 4–7 (watermark triggers, Spill /
  Rehydrate, bench). Land on top of Phase A.

Either way, M5 NIXL remains deferred until a trigger fires.
