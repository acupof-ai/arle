# Tiered KV Swap Substrate - T0/T1/T2 Implementation Contract

Last updated: 2026-05-25

Status: draft implementation contract. No code is implemented by this
document. Use it before touching the memory/GPU/SSD exchange path for tiered
KV.

Related truth surfaces:

- `docs/projects/tiered-kv-cache.md` owns the tiered-KV architecture and
  invariants.
- `docs/plans/tiered-kv-hicache-readmission.md` owns the staged readmission
  queue/backend plan.
- `docs/plans/cpu-gpu-pipeline-sync-stream.md` owns stream/fence semantics.
- `docs/plans/2026-05-03-a8-gpu-sm-kv-io-kernel.md` is the later optional
  SM-assisted copy optimization, gated by benchmark evidence.

## Goal

Build one high-performance exchange substrate for:

```text
T0 GPU HBM <-> T1 host pinned DRAM <-> T2 local NVMe SSD
```

and connect it to the existing tiered-KV abstraction:

```text
RadixCache metadata
  -> ReadmissionPlan / FetchTicket / StoreTicket
  -> Coordinator queues
  -> HostPinnedPool + DiskStore
  -> PagedKVPool GPU pages
```

The implementation must delete the current extra host `Vec<u8>` bounce on
T0<->T1 movement, but must not create a second offload system.

## Non-Goals

- Do not revive the deleted contiguous CPU KV offload path.
- Do not reintroduce `kv_tier/directory.rs` or any separate tier directory.
- Do not parse remote payloads outside the transport that produced them.
- Do not implement the A8 SM-assisted copy kernel in this tranche. This tranche
  should make a correct direct-region DMA path first; A8 can replace the copy
  engine after evidence says copy bandwidth is the binding constraint.
- Do not change paged-KV layout or attention kernels.
- Do not make Metal share the CUDA T1 path. Metal unified memory still skips
  T1 and joins only at T2.

## Current Evidence

The current code already has most of the control plane:

| Existing surface | Current state |
| --- | --- |
| `infer/src/kv_tier/{chunk,io,tier,lookup,readmission}.rs` | KV control/data objects, tier locations, lookup classes, staged readmission plans |
| `infer/src/kv_tier/host_pool.rs` | Allocation-stable host arena, CUDA host registration when `feature=cuda` |
| `infer/src/kv_tier/transport/disk.rs` | Content-addressed T2 block store via `BlockFingerprint` |
| `infer/src/kv_tier/coordinator.rs` | Fetch/store queues, cancellation, events, T1->T2 and T2->T1 |
| `infer/src/scheduler/cuda/runtime/admission.rs` | `lookup_or_stage -> ReadmissionPlan -> WaitingFetch -> promote_fetched_prefix` |
| `infer/src/scheduler/cuda/core.rs` | GPU block publish, demote-to-host, host spill-to-disk |
| `crates/cuda-kernels/src/paged_kv.rs` | T0 page allocator and current GPU<->host payload copies |

The concrete gap is the byte path:

- T0->T1 demotion currently copies GPU pages into a heap `Vec<u8>`, then writes
  that vector into `HostPinnedPool`.
- T1/T2->T0 promotion currently reads a host region as a borrowed slice and
  issues per-layer H2D copies on the compute stream.
- `LocalCudaTransport` exists but is still a structural stub, and it lacks the
  paged-KV page-table context needed to move actual KV pages.

## Target Data Flow

### Demote T0 -> T1

```text
select sealed GPU block
  -> reserve HostPinnedRegion
  -> compute stream fence: GPU writes complete
  -> copy stream D2H: PagedKVPool pages -> HostPinnedRegion
  -> copy completion fence
  -> retag RadixCache block_id to logical T1 block
  -> release old GPU pages
```

Visibility rule: the radix location must not change to `HostPinned` until the
D2H copy has completed successfully.

### Promote T1/T2 -> T0

```text
lookup staged prefix
  -> Coordinator fetches Disk/Remote into HostPinnedRegion when needed
  -> scheduler allocates detached GPU pages
  -> copy stream H2D: HostPinnedRegion -> PagedKVPool pages
  -> compute stream waits on copy fence
  -> retag/insert RadixCache as GPU-ready
  -> release temporary host regions only after copy completion
```

Visibility rule: a request can leave `Phase::WaitingFetch` only after the new
GPU pages are live and the compute stream has been ordered after the H2D copy.

### Spill T1 -> T2

The T1->T2 store path remains coordinator-owned:

```text
HostPinnedRegion -> DiskStore::put_block_with_fsync(false)
```

This path may still allocate a `Vec<u8>` for disk writes in the first tranche.
That is acceptable because the primary hot gap is T0<->T1. A later tranche can
add `DiskStore::put_block_from_slice` or mmap/direct-I/O if T1->T2 becomes the
measured bottleneck.

## Implementation Points

### P0. Keep One Tiered-KV Ownership Model

Touch points:

- `docs/projects/tiered-kv-cache.md`
- `infer/src/kv_tier.rs`
- `infer/src/prefix_cache.rs`

Rules:

- `RadixCache` remains the single source of truth for block tier/location.
- `BlockId` remains an ephemeral pool/logical id, not a content hash.
- `BlockFingerprint` remains the durable T2/T3 key.
- Scheduler decides which blocks should move.
- CUDA scheduler + `PagedKVPool` own local T0<->T1 materialization/demotion
  because they own GPU page allocation, stream fences, and radix retag timing.
- Coordinator owns queued T1<->T2/T3 byte movement and completion events.
- Scheduler code must not issue `TransferOp`s directly.

Failure modes:

- A new map duplicates tier location and diverges from radix metadata.
- A disk location is keyed by `BlockId` instead of fingerprint.
- A demoted GPU page is freed before the D2H copy finishes.

Review verdict: pass only if no new directory/cache-index type is introduced.

### P1. Add Safe Host Region Pointer Access

Touch points:

- `infer/src/kv_tier/host_pool.rs`

Add:

- A checked helper returning the host pointer and length for a live
  `HostPinnedRegion`.
- A short lifetime wrapper or closure API for region views used by CUDA copy
  submission.

Required contract:

- Validate region bounds and live-region membership before returning any
  pointer.
- Do not hold the host-pool mutex across a blocking CUDA stream sync.
- The caller must keep the region live until the returned copy fence is ready.

Preferred API shape:

```rust
pub struct HostRegionPtr {
    ptr: *mut u8,
    len: usize,
}

impl SharedHostPinnedPool {
    pub fn region_ptr(&self, region: HostPinnedRegion) -> Result<HostRegionPtr>;
}
```

If this API cannot express lifetime clearly enough, use a closure:

```rust
pub fn with_region_ptr<R>(
    &self,
    region: HostPinnedRegion,
    f: impl FnOnce(HostRegionPtr) -> R,
) -> Result<R>;
```

`HostRegionPtr` must be non-forgeable outside `host_pool.rs`; expose pointer
and length through methods, not public fields. If the direct copy API accepts
raw pointers instead of `HostRegionPtr`, the copy API itself must be `unsafe`.

Failure modes:

- Returning pointers for released regions.
- Keeping the mutex locked while waiting on CUDA.
- Releasing a region before an in-flight H2D/D2H finishes.

Review verdict: pass only if every unsafe pointer exposure has a local safety
comment and a caller-side fence/lifetime rule.

### P2. Add Direct Region Copy APIs To PagedKVPool

Touch points:

- `crates/cuda-kernels/src/paged_kv.rs`
- `crates/cuda-kernels/src/tensor.rs` if a small copy-stream helper is missing

Add direct APIs:

```rust
pub unsafe fn copy_pages_to_host_region(
    &self,
    ctx: &DeviceContext,
    pages: &[u32],
    dst: *mut u8,
    dst_len: usize,
) -> Result<CudaPipelineFence>;

pub unsafe fn copy_pages_from_host_region(
    &mut self,
    ctx: &DeviceContext,
    pages: &[u32],
    src: *const u8,
    src_len: usize,
) -> Result<CudaPipelineFence>;
```

Safety contract:

- `src` / `dst` must come from a live checked `HostPinnedPool` region, or from a
  non-forgeable `HostRegionPtr` returned by that pool.
- The region must be registered/pinned for the active CUDA context and remain
  live until the returned fence is ready or the copy stream has been
  synchronized. Having the compute stream wait on the fence is not enough to
  release or reuse the CPU host region; it only orders later GPU consumers.
- The pointer range must cover exactly `*_len` bytes and must not alias another
  mutable writer while the copy is in flight.
- Safe callers should prefer a `HostRegionPtr`-taking wrapper if that wrapper
  can encode the lifetime. Raw-pointer entry points are unsafe by construction.

First implementation:

- Use `ctx.copy_waits_for_compute()` before D2H.
- Enqueue layer K/V/scales/norms copies on `ctx.copy_stream`.
- Use raw CUDA async copy calls (`cuMemcpyDtoHAsync_v2` /
  `cuMemcpyHtoDAsync_v2`) against the checked `HostPinnedPool` pointer and
  `CudaSlice` device pointers. `HostPinnedPool` memory is registered with
  `cuMemHostRegister_v2`, but it is not a cudarc `PinnedHostSlice`.
- Return a copy-stream fence.
- D2H demotion must wait for CPU-observed fence readiness, or run the copy
  synchronously, before exposing the host region through radix metadata.
- H2D promotion must make compute wait before GPU use, but host/GPU buffers
  involved in the copy still cannot be released or reused until the copy fence
  is ready or the copy stream has been synchronized.

Performance requirement:

- No heap `Vec<u8>` payload allocation for T0<->T1.
- Copy quantized scales/norms as raw bytes where the on-device layout already
  matches the serialized payload layout. If this is too large for the first
  pass, explicitly mark quantized scale/norm direct-copy optimization as
  deferred and keep BF16 first.

Correctness requirement:

- Payload layout must stay byte-identical to `copy_pages_to_host` /
  `copy_pages_from_host`.
- `expected_len == pages.len() * storage_bytes_per_page()`.
- BF16, FP8/INT8 scales, and TurboQuant norms must either all be implemented
  or the unsupported formats must return a hard error, not silently fall back
  to an incompatible layout.

Failure modes:

- Copy stream races with compute stream writing the same pages.
- Compute stream reads promoted pages before H2D has completed.
- Quantized metadata is copied with the wrong byte order or stride.

Review verdict: pass only if BF16 has byte-equality tests against the old
payload-copy path, and unsupported formats are explicit.

### P3. Replace T0->T1 Demotion Bounce Buffer

Touch points:

- `infer/src/scheduler/cuda/core.rs`

Change `demote_block_to_host`:

1. Read block metadata and page list.
2. Ensure host demote headroom.
3. Reserve `HostPinnedRegion` sized to `metadata.byte_len`.
4. Direct-copy pages into that region.
5. Wait until the copy fence is CPU-observed ready, or make this first pass
   synchronous.
6. Allocate logical T1 `BlockId`.
7. Retag radix and session-slot metadata.
8. Remove `block_to_pages` entry and owner-slot mapping.
9. Release GPU pages.

Rollback rules:

- If reserve fails: leave block on GPU.
- If copy fails: release host region, leave block on GPU.
- If retag fails: release host region, leave block on GPU, release no pages.
- Release GPU pages only after retag has succeeded and D2H is complete.

Failure modes:

- Half-demoted block visible in radix.
- Host region leak on retag failure.
- GPU page released while a request/session still references it.

Review verdict: pass only if all error exits preserve the old GPU-resident
state or release all newly allocated resources.

### P4. Replace T1/T2->T0 Promotion Bounce Buffer

Touch points:

- `infer/src/scheduler/cuda/runtime/admission.rs`

Change `promote_fetched_prefix`:

1. For each staged block, allocate detached GPU pages.
2. Get `HostPinnedRegion` pointer from the fetched block.
3. Direct-copy region into GPU pages.
4. Order compute after the copy fence.
5. Retag session slot block or insert fingerprinted blocks into radix.
6. Record GPU metadata with `record_sealed_gpu_blocks`.
7. Release consumed canonical host regions only after promotion succeeds.
8. Release temporary fetch regions after copy completion.

Rollback rules:

- If any H2D copy fails, first wait/synchronize every previously submitted H2D
  fence in this promotion batch, then release newly allocated GPU pages, release
  any temporary fetch regions that were materialized from Disk/Remote, and keep
  old staged metadata intact.
- If radix insert/retag fails, first wait/synchronize every previously
  submitted H2D fence in this promotion batch, then release new GPU pages,
  release any temporary fetch regions that were materialized from Disk/Remote,
  and keep request on the cold fallback path.
- Multi-block radix/session changes must be transactional. Either preflight the
  whole batch before mutating, add an atomic batch retag/insert helper, or record
  every successful retag/insert and roll it back before releasing promoted GPU
  pages.
- Do not release T1 canonical host regions until the promoted GPU copy has
  completed.
- Temporary fetch regions and canonical T1 regions must have separate ownership
  paths: temporary fetch regions are exact-once cleanup resources, while
  canonical T1 regions remain owned by the tier index until a successful
  promotion policy explicitly consumes them.

Failure modes:

- Direct host hit releases canonical T1 storage too early.
- Multiple waiters race to promote the same staged block.
- Fetched Disk/Remote temporary region leaks after failure.

Review verdict: pass only if deduped waiters still share one fetch result and
unclaimed temporary regions are released exactly once.

### P5. Keep Coordinator T1/T2 Semantics Stable

Touch points:

- `infer/src/kv_tier/coordinator.rs`
- `infer/src/kv_tier/coordinator/{events,builder,types}.rs`

Do not widen coordinator responsibility in the first pass. It should still:

- Fetch T2/T3 payloads into T1 host regions.
- Store T1 host regions to T2/T3.
- Emit `FetchCompleted`/`StoreCompleted` events.
- Track cancellation/backpressure.

Possible small additions:

- Include `byte_len` consistency checks when fetched region length differs from
  metadata.
- Add metrics fields if needed for direct copy latency.

Failure modes:

- Coordinator starts mutating `RadixCache`.
- Coordinator starts owning GPU page allocation.
- Store/fetch event delivery becomes best-effort for critical events.

Review verdict: pass only if scheduler remains the only radix writer and the
coordinator remains byte-orchestration only.

### P6. Observability And Metrics

Touch points:

- `infer/src/metrics.rs`
- `infer/src/http_server/handlers.rs`
- `infer/src/scheduler/cuda/runtime/scheduler_loop.rs`

Minimum metrics:

- T0->T1 D2H copy latency and bytes.
- T1->T0 H2D copy latency and bytes.
- Fetch wait time already exists; keep it.
- Store wait time already exists; keep it.
- Direct-copy fallback/error counters if any format falls back or is rejected.
- HostPinnedPool reservation wait time, reservation attempts, reservation
  failures, and spill/headroom eviction count.

Stats should distinguish:

- queue wait
- disk/remote fetch
- host-pool reservation/backpressure
- host region to GPU copy
- GPU to host region copy

Failure modes:

- One aggregate `tier_wait` hides whether SSD, host pool, or PCIe copy is slow.
- Counter increments before copy completion, overstating success.

Review verdict: pass only if `/v1/stats` can answer: "did TTFT wait on disk,
host pool, H2D, or D2H?"

### P7. Tests And Bench Evidence

Touch points:

- `crates/cuda-kernels/src/paged_kv.rs`
- a CUDA-only scheduler test/bench if end-to-end staged promotion timing is
  needed
- `docs/experience/wins/`

Local tests:

```bash
cargo check -p infer --no-default-features --features cuda,no-cuda
cargo test --release --no-default-features --features no-cuda -p infer --lib kv_tier
```

CUDA tests:

```bash
cargo test -p cuda-kernels --release --features cuda paged_kv
cargo test -p infer --release --features cuda staged_prefix
```

Bench requirement:

- Add or extend an ignored CUDA microbench for T0<->T1 direct region copies.
- Compare old payload-copy path vs direct-region path on the same block size
  and model KV format.
- Add a wins entry under `docs/experience/wins/`.
- If CUDA cannot run locally, create a `pending-remote` wins stub with the exact
  command and target GPU.

Failure modes:

- Unit tests pass but no runtime benchmark exists.
- Benchmark changes multiple variables at once.
- Report quotes narrow copy-window speedup without wall-clock TTFT/ITL context.

Review verdict: pass only if evidence separates correctness, microbench
throughput, and end-to-end scheduler impact.

### P8. Rollout Gate

Default behavior should stay conservative until measured.

Initial acceptable gate:

- New direct-region T0<->T1 path enabled for BF16 paged KV only.
- Other formats return explicit unsupported errors or stay on the old path
  only if the fallback is documented and counted.
- A config/env knob may gate the new path during first validation.

Promotion to default requires:

- Byte-equality correctness.
- No request-state regression in staged readmission tests.
- CUDA microbench win or neutral result.
- `guidellm` or W4/session-resume bench showing no TTFT regression.

Failure modes:

- Default-on path covers only one format but silently corrupts another.
- Old and new paths coexist without a deletion plan.

Review verdict: pass only if default-on scope is explicit and unsupported scope
is noisy.

## Exact File Touch List

Expected implementation files:

| File | Reason |
| --- | --- |
| `infer/src/kv_tier/host_pool.rs` | checked raw pointer / region view for CUDA direct copy |
| `crates/cuda-kernels/src/paged_kv.rs` | direct host-region D2H/H2D page copy APIs |
| `crates/cuda-kernels/src/tensor.rs` | copy-stream helper only if existing fence helpers are insufficient |
| `infer/src/scheduler/cuda/core.rs` | T0->T1 demotion uses direct host region |
| `infer/src/scheduler/cuda/runtime/admission.rs` | T1/T2->T0 promotion uses direct host region |
| `infer/src/metrics.rs` | latency/byte/error counters |
| `infer/src/http_server/handlers.rs` | expose new counters in `/v1/stats` if metrics are added |
| CUDA-only scheduler test/bench | optional end-to-end staged promotion timing if `paged_kv` microbench is insufficient |
| `docs/experience/wins/YYYY-MM-DD-bench-guidellm-*.md` | required runtime evidence |

Files that should not be touched in this tranche:

| File/path | Reason |
| --- | --- |
| `infer/src/kv_tier/directory.rs` | must not be reintroduced |
| `infer/src/kv_tier/coordinator/bench.rs` | T1<->T2 only and compiled in no-cuda tests; do not add CUDA-only T0<->T1 code here |
| `infer/src/model/kv_cache.rs` old contiguous offload path | retired; do not revive |
| `crates/cuda-kernels/csrc/kv/` new A8 kernel | separate gated tranche |
| `infer/src/backend/metal/**` | Metal T1 is a no-op; T2-only later |

If implementation needs more than the expected files, update this document
before editing.

## Implementation Order

1. Add host-region pointer API with unit tests.
2. Add BF16-only direct region copy API in `PagedKVPool`.
3. Add byte-equality test against old `copy_pages_to_host` /
   `copy_pages_from_host`.
4. Replace demote T0->T1.
5. Replace promote T1/T2->T0.
6. Add metrics.
7. Add microbench.
8. Add wins entry or `pending-remote` stub.
9. Run local checks.
10. Run CUDA checks on target hardware.

Do not merge steps 4 and 5 into one opaque diff. They have different rollback
rules and should be reviewable independently.

## Multi-Review Matrix

### Review Pass 1 - Architecture Boundary

| Implementation point | Boundary check | Verdict before coding |
| --- | --- | --- |
| P0 ownership | Radix remains tier/location SSOT | PASS |
| P1 host pointer | Host pool exposes bytes only, not tier metadata | PASS |
| P2 paged copy | Paged pool moves bytes, does not update radix | PASS |
| P3 demote | Scheduler retags after byte movement | PASS |
| P4 promote | Scheduler inserts/retags after byte movement | PASS |
| P5 coordinator | Coordinator remains queue + byte backend owner | PASS |
| P6 metrics | Metrics observe; no control-plane decisions | PASS |
| P7 evidence | Bench/docs only | PASS |
| P8 rollout | Scope gate is explicit | PASS |

Architecture gap: `LocalCudaTransport` remains a stub. This is accepted for
this tranche because it cannot represent paged KV page spans today. Follow-up
should either extend `TransferOp` with paged descriptors or delete the stub if
it remains unused after direct-region APIs land.

### Review Pass 2 - Correctness And Lifetimes

| Implementation point | Main risk | Required proof |
| --- | --- | --- |
| P1 | pointer to released host region | live-region validation test |
| P2 | stream race or early buffer reuse | CPU-observed fence readiness test or explicit sync in first pass |
| P3 | half-demoted radix entry | rollback unit test or local code proof |
| P4 | temporary region double-release or partial radix retag | staged fetch unit test plus transaction/rollback proof |
| P5 | dropped critical event | existing required-send behavior preserved |
| P6 | success counter before completion | counter increments after fence ready |
| P7 | false perf attribution | control run plus one-variable diff |
| P8 | unsupported format corruption | explicit format gate |

Correctness gap: direct async copies into raw host pointers need a precise
region lifetime rule. If the implementation cannot encode that in Rust types,
the first version should synchronously wait for the copy fence before returning
from demote/promote. That sacrifices overlap but keeps correctness SOLID.

### Review Pass 3 - Performance

| Implementation point | Expected performance impact | Measurement |
| --- | --- | --- |
| P1 | no direct win | none beyond no-lock-across-sync audit |
| P2 | removes heap allocation and extra memcpy | T0<->T1 microbench |
| P3 | hypothesized faster demotion and lower host CPU bandwidth | demote latency/bytes |
| P4 | hypothesized faster staged readmission | H2D latency + fetch wait p99 |
| P5 | neutral | existing T1->T2 bench |
| P6 | neutral overhead only | stats overhead check if hot |
| P7 | evidence | old vs new table |
| P8 | prevents regression blast radius | feature/default comparison |

Performance gap: direct-region DMA may be neutral on small blocks if transfer
launch overhead dominates. That does not license A8 automatically; A8 still
needs the gate from `2026-05-03-a8-gpu-sm-kv-io-kernel.md`.

### Review Pass 4 - SOLID Evidence Standard

| Claim | Evidence needed | Status now |
| --- | --- | --- |
| The old path has an avoidable bounce buffer | Code read shows `Vec<u8>` D2H then host write | EVIDENCED |
| Direct-region copy is faster | Microbench old vs new | HYPOTHESIS |
| End-to-end TTFT improves | `guidellm` or W4 resume bench | HYPOTHESIS |
| Correctness is preserved | byte-equality + staged readmission tests | PENDING |
| A8 SM kernel is worthwhile | copy wait dominates wall-clock | DEFERRED |

Do not claim user-visible speedup until the end-to-end bench exists. Before
that, the only allowed claim is: the implementation removes a known extra copy
and creates the measurement surface to decide whether the copy path matters.

## Acceptance Checklist

- [ ] No new tier directory or parallel cache state.
- [ ] Host pointer API validates live regions.
- [ ] T0->T1 demotion has no heap payload bounce.
- [ ] T1/T2->T0 promotion has no heap payload bounce.
- [ ] Radix location flips only after copy completion.
- [ ] Multi-block promotion retag/insert is atomic or explicitly rolled back.
- [ ] Rollback releases no host/GPU buffer until all submitted copy fences are ready.
- [ ] GPU pages release only after successful demote retag.
- [ ] Temporary fetched host regions release exactly once.
- [ ] Unsupported KV formats are explicit.
- [ ] Metrics distinguish queue, disk/remote fetch, D2H, and H2D.
- [ ] BF16 byte-equality test passes.
- [ ] Local non-CUDA check passes.
- [ ] CUDA check passes or `pending-remote` is documented.
- [ ] Bench wins entry exists.

## Commit Shape

Recommended tranche split:

1. `feat(kv-tier): expose checked host region pointer for tier copies`
2. `feat(cuda): copy paged kv directly to host regions`
3. `feat(scheduler): use direct host regions for kv demote`
4. `feat(scheduler): use direct host regions for kv readmission`
5. `docs(kv-tier): record direct swap substrate benchmark`

Each tranche should be small enough for `codex review --uncommitted` or
`codex review --commit <sha>` to give actionable findings.
