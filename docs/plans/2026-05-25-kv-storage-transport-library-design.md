# KV Storage/Transport Library Design

**Status**: design-only, no implementation
**Date**: 2026-05-25
**Scope**: storage organization and byte transport for KV tiers, especially
SSD<->HBM and DRAM<->HBM.
**Evidence level**: local surface inventory is code evidence; bottleneck cost
claims are marked `measured` only when a committed benchmark or runtime metric
exists. Source survey is hypothesis until ARLE measures it.

## 1. Decision

Recommendation: extend `infer/src/kv_tier/transport/` first. Do not create
`crates/kv-transport` yet, and do not move transport policy into
`crates/kv-native-sys`.

Rationale:

- `kv_tier` already owns the invariants that make transfers safe:
  `BlockId` is pool-local, `BlockFingerprint` is content identity, and only the
  coordinator moves blocks between tiers (`infer/src/kv_tier.rs:56-86`).
- `kv-native-sys` is a substrate crate: POSIX file/block I/O, mmap, shm, and a
  host arena (`crates/kv-native-sys/src/lib.rs:1-5`). It should not learn
  scheduler policy, CUDA stream semantics, NIXL metadata exchange, or
  Mooncake/NCCL lifecycle.
- CUDA's live path already routes T1/T2/T3 movement through `Coordinator` and
  `DiskStore`; adding another crate before GDS/NIXL/Mooncake has one measured
  ARLE winner would split the ownership boundary.
- Metal has a runtime-local `DiskStore` adapter today, so the next shared
  abstraction should be a narrow transport/backend boundary under `kv_tier`
  that CUDA and Metal can both call, not a new crate with no live consumer.

Extraction rule: create `crates/kv-transport` only after at least one real
non-local backend passes license-or-kill with measured wall-clock impact and
needs reuse outside `infer`.

## 2. Inventory

| Surface | API shape | Current caller | Status |
|---|---|---|---|
| `infer/src/kv_tier.rs:11-45` | Tier model: T0 GPU HBM, T1 host pinned, T2 disk, T3 remote | module root and scheduler imports | live CUDA, partial Metal |
| `infer/src/kv_tier.rs:70-78` | Coordinator-only byte movement and MR-stable T1 invariant | CUDA scheduler, coordinator | live invariant |
| `infer/src/kv_tier.rs:143-158` | `KvTierAdapter` policy-facing trait | backend adapters | structural only |
| `infer/src/kv_tier/tier.rs:15-43` | `Tier`, `BlockLocation` | prefix cache, coordinator, scheduler | live |
| `infer/src/kv_tier/tier.rs:61-103` | `RemoteBlockDesc`, `TransportId`, `MemKind` | backend/transport descriptors | live descriptors; GDS `Block` reserved |
| `infer/src/kv_tier/transport.rs:43-87` | `TransferOp` with `KVPayloadRef` src/dst | future transport implementors | structural |
| `infer/src/kv_tier/transport.rs:121-195` | `KVTransport`: register, put/get batch, poll, abort | `LocalCudaTransport`, `NixlTransport` stubs | trait locked, not in hot path |
| `infer/src/kv_tier/backend.rs:24-49` | `KVBackend`: object-store store/fetch/delete/exists/poll/abort | `ClusterSharedBackend` | live for shared-fs remote |
| `infer/src/kv_tier/backend.rs:51-190` | `ClusterSharedBackendConfig`/dispatch | coordinator remote target | live shared-fs, NIXL stub-gated |
| `infer/src/kv_tier/io.rs:14-134` | `KVPayload`, `KVBackendStore/Fetch/Delete/Completion` | `DiskStore`, `SharedFsStore`, coordinator | live |
| `infer/src/kv_tier/host_pool.rs:1-7` | T1 host pinned pool backed by `kv-native-sys`, CUDA pins once | CUDA scheduler + coordinator | live |
| `infer/src/kv_tier/host_pool.rs:70-113` | `read_region`, `with_region_slice`, `write_region` | coordinator store/fetch, H2D promote | live; blocking I/O forbidden under slice callback |
| `infer/src/kv_tier/transport/disk.rs:17-42` | Keyed/block DiskStore API, sync I/O, no `KVTransport` impl | coordinator, Metal adapter | live T2 backend |
| `infer/src/kv_tier/transport/disk.rs:492-605` | `put_block_with_fsync`, `get_block` | coordinator store/fetch | live; one allocation and one payload copy on fetch |
| `infer/src/kv_tier/transport/shared_fs.rs:103-249` | `SharedFsStore` wrapping `DiskStore` as remote `KVBackend` | `ClusterSharedBackend` | live minimal T3 |
| `infer/src/kv_tier/transport/local_cuda.rs:1-117` | Local CUDA HBM<->T1 skeleton | none in hot path | stub |
| `infer/src/kv_tier/transport/nixl.rs:28-233` | `NixlTransport` + `KVBackend` stub | feature-gated remote path | stub |
| `infer/src/kv_tier/coordinator.rs:1-11` | owns T1->T2 store and T1/T2/T3 fetch staging | CUDA scheduler | live |
| `infer/src/kv_tier/coordinator.rs:217-399` | `handle_store` host region -> disk/remote | CUDA `spill_host_blocks_if_pressured` | live |
| `infer/src/kv_tier/coordinator.rs:401-602` | `handle_fetch`, disk/remote -> T1 staging | CUDA staged readmission | live |
| `infer/src/kv_tier/coordinator/builder.rs:248-307` | builder wires disk/cluster backend and queues | CUDA construction | live |
| `infer/src/scheduler/cuda/core/construction.rs:234-289` | creates `HostPinnedPool`, `DiskStore`, optional `ClusterSharedBackend`, coordinator | CUDA scheduler init | live |
| `infer/src/scheduler/cuda/core.rs:1379-1392` | `demote_block_to_host`: `copy_pages_to_host` then T1 write | CUDA pressure/retention | live |
| `infer/src/scheduler/cuda/core.rs:1477-1565` | `spill_host_blocks_if_pressured`: T1 -> coordinator store | CUDA scheduler loop | live |
| `infer/src/scheduler/cuda/runtime/admission.rs:760-920` | staged fetch submit and fallback to cold prefill | CUDA admission | live |
| `infer/src/scheduler/cuda/runtime/admission.rs:1022-1100` | fetched T1 payload -> `copy_pages_from_host` -> T0 | CUDA readmission | live |
| `infer/src/scheduler/cuda/runtime/fetch.rs:84-280` | coordinator event handling, fetch/store completion | CUDA scheduler loop | live |
| `crates/cuda-kernels/src/paged_kv.rs:624-808` | page payload serialization D2H and H2D copy | CUDA scheduler demote/promote | live, synchronous at end |
| `crates/kv-native-sys/src/lib.rs:567-690` | host arena create/reserve | `HostPinnedPool` | live substrate |
| `infer/src/backend/metal/runtime.rs:311-389` | `MetalTierAdapter` wraps `DiskStore` | Metal Qwen3.5 prefix runtime | live, local to Metal |
| `infer/src/backend/metal/runtime.rs:467-500` | creates Metal `DiskStore` when disk options are set | Metal backend init | live |
| `infer/src/distributed/nccl.rs:81-170` | NCCL all-reduce/all-gather/reduce-scatter | distributed runtime | live for collectives, not KV |
| `infer/src/distributed/nccl.rs:172-454` | grouped send/recv and self-peer D2D copy | EP dispatch/combine | live NCCL P2P surface, not KV |

## 3. Bottleneck Map

| Path | Current data flow | Cost status | Likely sync/copy points | Verdict |
|---|---|---|---|---|
| T0 GPU hit | radix lookup -> page attach/reuse | hypothesis | no byte movement; metadata only | not a storage-library target |
| T0->T1 demote | `copy_pages_to_host` returns `Vec<u8>`, then `HostPinnedPool::write_region` | instrumented, not measured in SERVE yet | D2H copy, host allocation, T1 memcpy, scheduler-visible latency | candidate only if T4 metrics show wall-clock pressure |
| T1->T0 promote | `with_region_slice` -> `copy_pages_from_host` -> `ctx.sync()` | instrumented, not measured in SERVE yet | H2D copies per page/layer/scale, final device sync | high-priority for async copy/kernel path if readmission is exercised |
| T1->T2 store | `read_region`/payload -> `DiskStore::put_block_with_fsync(false)` | measured code-comment benchmark for fsync choice | T1 read allocation, header+payload allocation, temp write+rename | keep no-fsync cache path; next ROI is fewer copies and async I/O |
| T2->T1 fetch | `DiskStore::get_block` -> `Vec<u8>` -> `stage_into_host_pool` | hypothesis | file read, header decode, payload `to_vec`, T1 memcpy | plausible ROI for borrowed payload/read-into-region API |
| T2->T0 via GDS | not implemented | hypothesis | requires direct file offset -> GPU buffer and aligned payload layout | do not implement until T2 readmission is measured hot |
| T1->T3 shared-fs store | same payload path through `SharedFsStore`/`DiskStore` | hypothesis | same local copies plus filesystem/network latency | useful as correctness bridge only |
| T3->T1 shared-fs fetch | `KVBackend::fetch` -> payload -> T1 staging | hypothesis | same as DiskStore plus remote FS semantics | not enough for production remote tier |
| NIXL/Mooncake remote | stubs only | hypothesis | MR registration, metadata exchange, async completion | license behind PD/multi-node workload |
| HBM<->HBM peer | NCCL P2P/collectives exist outside KV | hypothesis for KV | stream ordering, communicator contention, peer topology | only for multi-GPU KV sharing, not local SSD/DRAM |

Measured local fact: coordinator comments cite a 2026-05-04 benchmark where
small 4 KiB cache writes were 19 MiB/s with fsync and 361 MiB/s without; at
256 MiB, no-fsync remained 1.6x faster (`infer/src/kv_tier/coordinator.rs:268-277`).
Everything else above needs T4b/T2 style wall-clock evidence before it licenses
a transport rewrite.

## 4. Upstream Survey

### 4.1 GPUDirect Storage

NVIDIA GDS provides explicit cuFile APIs for storage<->GPU-memory transfers.
The useful ARLE takeaway is not "replace DiskStore"; it is "make the payload
layout and allocator capable of direct file offset -> GPU page transfer when
T2 readmission is on the critical path."

Constraints from the docs:

- GDS removes CPU bounce-buffer copies only when the transfer is between
  storage and GPU memory; if the CPU must parse or transform the data, the
  benefit shrinks.
- The direct path favors explicit, proactive transfers, coarse enough to
  amortize OS/kernel overhead.
- File-system direct transfer normally wants `O_DIRECT` and aligned buffers,
  sizes, and offsets.
- CUDA stream cuFile APIs can make I/O ordered relative to GPU work.
- BAR1/topology can introduce internal staging, so "GDS enabled" is not proof
  of zero-copy wall-clock benefit.

ARLE implication: current `DiskStore` prepends a postcard header to the payload
and fetch returns a payload `Vec`. A GDS path needs a payload-aligned object
layout or sidecar metadata so the GPU read can target payload bytes directly.

### 4.2 NVLink/NVSwitch and CUDA peer access

CUDA can use peer-to-peer device memory transfers and NVLink when topology
allows peer access. This is a HBM<->HBM primitive, not an SSD<->HBM storage
backend. It belongs behind a multi-GPU KV sharing license, separate from local
T2 disk work.

ARLE implication: use CUDA peer/NCCL only when the source and destination are
both GPU-resident KV pages. Do not force DRAM/HBM staging through this surface.

### 4.3 NCCL DMA paths

ARLE already has NCCL collective and grouped P2P wrappers for distributed
runtime work. NCCL can express scatter/gather/all-to-all style transfers, and
newer NCCL exposes one-sided RMA-style operations with registered memory
windows. That is useful for multi-rank KV fan-out, but it is not a general
object-store API and does not replace `KVBackend`.

ARLE implication: add a KV-specific NCCL path only after a workload shows
duplicate prefill across ranks or peer reuse is material. Otherwise it will
contend with model collectives for no serve-side gain.

### 4.4 SGLang HiCache

SGLang HiCache organizes GPU memory, host memory, and distributed storage
with a radix metadata layer, local matching, L3 prefetch, and write-back
policies. Its design notes call out page-oriented layouts, CPU-to-GPU transfer
optimization, GPU-assisted I/O kernels, and several L3 backends including
Mooncake, NIXL, and file-like storage. It also supports runtime attach/detach
of storage backends only when the service is idle.

ARLE implication: ARLE already matches the high-level split
(`RadixCache + HostPinnedPool + Coordinator + DiskStore`). The missing
transport-library value is narrower:

- page/layout choice for T1/T2 payloads;
- async D2H/H2D engine under coordinator ownership;
- remote backend metadata/registration lifecycle;
- idle-safe backend reconfiguration if runtime attach is added later.

### 4.5 Mooncake Transfer Engine

Mooncake's Transfer Engine has two relevant concepts: registered segments
(DRAM/VRAM/NVMe-oF) and batch transfers over non-contiguous ranges. It supports
TCP, RDMA, NVMe-oF, cuFile/GDS, NVLink/intra-node paths, and async completion
polling.

ARLE implication: this maps well to `KVTransport::{register, put_batch,
get_batch, poll, abort}`. A direct Mooncake binding should not be written
until ARLE has a cluster-shared KV workload; otherwise NIXL may provide the
same plugin path with less local API surface.

### 4.6 NIXL

NIXL targets inference point-to-point data transfer across memory and storage
types: HBM/VRAM, DRAM, local or remote SSD, files, objects, and plugins such as
UCX and GDS. Its metadata model assumes a conductor/control path that exchanges
serialized memory-section metadata once, while data movement is selected by
memory type and available backends.

ARLE implication: `HostPinnedPool` allocation stability is already a good fit.
The missing pieces are a conductor/control-plane owner and exact metadata
lifetime rules. Those belong in `kv_tier/backend` or a future distributed
coordinator, not in `kv-native-sys`.

## 5. Proposed API Shape

Keep two surfaces and make their roles stricter:

1. `KVBackend`: object-store control plane for slower tiers.
   - key/fingerprint existence
   - store/fetch/delete by logical block
   - remote descriptor encoding
   - best fit: `DiskStore`, `SharedFsStore`, future NIXL object/file backend

2. `KVTransport`: registered byte movement engine.
   - memory-region registration
   - batched non-contiguous read/write
   - poll/abort completion model
   - best fit: `LocalCudaTransport`, future GDS, Mooncake, NIXL transfer path

Concrete next-step API additions, still under `infer/src/kv_tier/transport/`:

```rust
pub enum TransferClass {
    LocalDramHbm,
    LocalDiskHbm,
    LocalDiskDram,
    RemoteDramHbm,
    RemoteDiskHbm,
    PeerHbmHbm,
}

pub struct TransportCapabilities {
    pub transfer_class: TransferClass,
    pub supports_cuda_stream_ordering: bool,
    pub supports_non_contiguous_batch: bool,
    pub requires_stable_registration: bool,
    pub preferred_alignment: usize,
    pub min_profitable_bytes: usize,
}

pub trait KVTransport: Send + Sync {
    type Region: Send + Sync;
    type Op: Send;

    fn capabilities(&self) -> TransportCapabilities;

    unsafe fn register(
        &self,
        ptr: *mut u8,
        len: usize,
        kind: MemKind,
    ) -> Result<Self::Region, TransportError>;

    fn put_batch(&self, ops: &[TransferOp]) -> Result<Self::Op, TransportError>;
    fn get_batch(&self, ops: &[TransferOp]) -> Result<Self::Op, TransportError>;
    fn poll(&self, op: &mut Self::Op) -> Poll<Result<(), TransportError>>;
    fn abort(&self, op: &mut Self::Op);
}
```

Do not add this API until one implementation is about to use the capability
fields. The important design point is the direction: capability discovery goes
on `KVTransport`, while object identity stays on `KVBackend`.

Disk layout change for future GDS:

- keep current `DiskBlockHeader` for compatibility;
- add a new version where metadata is sidecar or header-padded so payload starts
  at a configurable alignment;
- store `payload_offset` in `DiskBlockLocation`;
- add a fetch target variant that can read into a registered host region or GPU
  page buffer without returning a `Vec<u8>`.

## 6. ROI and Gates

| Component | ROI hypothesis | License threshold | Kill threshold |
|---|---|---|---|
| Local CUDA async DRAM<->HBM copy | Reduce scheduler-visible promote/demote latency and avoid final sync stalls | T4b workload shows T0->T1 or T1->T0 copy/fetch wait contributes >=10% request wall-clock, and async/kernel path improves p50 or p99 by >=20% without correctness regressions | T4b shows copy path <5% wall-clock or p99 unchanged within noise |
| Disk read-into-region API | Remove T2 fetch `Vec` payload allocation/copy before T1 staging | Synthetic and SERVE T2-readmission tests show >=1.3x T2->T1 throughput or >=10% TTFT improvement on staged disk hits | File I/O dominates and copy removal is <5% wall-clock |
| Disk aligned payload layout | Make future GDS possible and improve direct I/O | Microbench proves no regression for normal DiskStore and unlocks aligned payload offsets for GDS | Compatibility or complexity exceeds measured benefit |
| GDS SSD<->HBM | Remove T2->T1->T0 bounce for long-prefix readmission | Under HBM-pressure SERVE, T2 hit readmission p50 improves >=2x or request TTFT improves >=15%, with lower CPU utilization and no recompute fallback increase | T2 reads are rare, alignment work dominates, or end-to-end gain <10% |
| SharedFS backend | Correctness bridge for T3 remote namespace | Keep only as dev/test backend and if it remains simple | Kill as performance target; it is not a low-latency transport |
| NIXL backend | Unified remote memory/storage transfer with MR metadata | PD or multi-worker workload shows remote KV reuse improves TTFT >=25% or avoids redundant prefill with stable p99 | No conductor/metadata owner or single-node workload only |
| Mooncake direct backend | High-performance segmented batch transfer across RDMA/NVMe-oF/NVLink | Beats NIXL path by >=15% wall-clock on ARLE workload or exposes a feature NIXL cannot | NIXL covers the path or integration surface duplicates NIXL |
| NCCL KV peer path | Share GPU-resident KV across ranks without host/disk bounce | Multi-GPU workload eliminates redundant prefill and improves TTFT >=20% with ITL p99 regression <10% | NCCL contention with model collectives erases request-level gain |
| New `crates/kv-transport` | Reuse transport outside `infer` | At least two in-tree consumers and one measured backend pass | Single consumer or API still changing after first backend |

## 7. Execution Order

1. Finish T4b metrics baseline before any policy or transport rewrite.
2. If T1<->T0 copy is hot, implement real `LocalCudaTransport` under the
   existing trait and measure async/kernel copy against current
   `copy_pages_{to,from}_host`.
3. If T2 fetch is hot, add read-into-region and payload-aligned disk layout
   behind a format version gate.
4. Only then run a GDS spike. Its first deliverable is an end-to-end benchmark,
   not a permanent API.
5. Defer NIXL/Mooncake until a PD/multi-node workload exists and a conductor
   owns metadata exchange.
6. Defer NCCL KV sharing until a multi-GPU duplicate-prefill workload exists.
7. Revisit `crates/kv-transport` extraction after one real transport survives
   the above gates.

## 8. References

- ARLE `kv_tier` module contract:
  `infer/src/kv_tier.rs`, `infer/src/kv_tier/AGENTS.md`.
- ARLE transport/backend surfaces:
  `infer/src/kv_tier/transport.rs`, `infer/src/kv_tier/backend.rs`,
  `infer/src/kv_tier/transport/disk.rs`,
  `infer/src/kv_tier/transport/shared_fs.rs`,
  `infer/src/kv_tier/transport/local_cuda.rs`,
  `infer/src/kv_tier/transport/nixl.rs`.
- ARLE host substrate: `crates/kv-native-sys/src/lib.rs`.
- NVIDIA GPUDirect Storage docs:
  https://docs.nvidia.com/gpudirect-storage/index.html,
  https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html,
  https://docs.nvidia.com/gpudirect-storage/design-guide/index.html.
- NVIDIA CUDA multi-GPU peer access:
  https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html.
- NVIDIA NCCL P2P docs:
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html.
- SGLang HiCache design and best practices:
  https://docs.sglang.io/docs/advanced_features/hicache_design,
  https://docs.sglang.io/docs/advanced_features/hicache_best_practices,
  https://docs.sglang.io/docs/advanced_features/hicache_storage_runtime_attach_detach.
- Mooncake Transfer Engine:
  https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html.
- NIXL design:
  https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md.
