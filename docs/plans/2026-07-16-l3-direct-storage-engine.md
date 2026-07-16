# L3 direct storage engine

> Status: Active — target-NVMe gate pending

## Verdict

Keep one `kv_native_sys::KvTierStore`. Replace only its disk lane: aligned
`O_DIRECT` files submitted through `io_uring` when `ARLE_KV_DISK_IO=direct`, with
the current mmap lane as the default and fallback. Batch inference pages before one
completion wait. Enable RL checkpoint spill only after an explicit host-memory
cap is crossed. GDS stays disabled unless the live file, filesystem, driver and
kernel path all pass; compatibility-mode cuFile is not GDS.

The current pod fails the GDS gate: `libcufile.so` exists, but `nvidia_fs` is not
loaded, `use_pci_p2pdma=false`, cuFile compatibility mode is allowed, and the
container has no mounted local NVMe. This phase must ship the gate and fallback,
not a false GDS result.

## Invariants

1. `KvTierStore` remains the sole L2/L3 key, budget, index and persistence
   authority for CUDA, Metal and optional training spill.
2. Disk mode is selected once at attach. No caller branches on mmap/direct/GDS.
3. Direct I/O uses 4096-byte aligned offsets, lengths, buffers and padded slot
   strides while retaining the logical payload length.
4. A demoted GPU page is freed only after D2H and every accepted disk write
   complete. A promoted page attaches only after disk read and H2D complete.
5. RL defaults to host checkpoint offload. L3 is used only when an explicit host
   cap would be exceeded; fused CE remains compute-only.
6. `Dirty`/checkpoint residency identifies the only current tensor copy. An L3
   tensor cannot masquerade as a host `Vec`.
7. Durable metadata is committed once per batch, never once per page.

## Implementation

### P1 — direct substrate and unified accounting

- `crates/kv-native-sys/Cargo.toml`: add Linux-only `io-uring`.
- `crates/kv-native-sys/src/direct_store.rs`: own the `O_DIRECT` fd, aligned
  buffers, batched SQE submission, CQE validation and runtime probe. One batch
  wait returns all page completions; no per-page thread or runtime.
- `crates/kv-native-sys/src/lib.rs`: keep `KvMmapStore`; expose `DiskIoMode` and
  cumulative `TierIoStats`.
- `crates/kv-native-sys/src/kv_tier.rs:203-854`: replace `DiskTier.store` with
  one internal mmap/direct enum; add `insert_many`/`read_many`; count useful,
  submitted and metadata bytes, operations, failures and latency; write the
  durable manifest once after each batch.

Failure rule: `io_uring_setup`, `O_DIRECT`, alignment, filesystem or first I/O
failure logs one reason and attaches mmap. A failure after accepted writes is an
I/O error, not a silent mode switch that could split one namespace across two
layouts.

### P2 — inference batch prefetch/completion

- `crates/infer-cuda/src/executor/qwen.rs`: collect demotion pages through one
  tier call, call `insert_many`, and use a concatenated batch read for promotion. Direct reads are
  submitted together at queue depth `min(batch, configured_qd)`; one CQ drain
  precedes the existing H2D copy-stream sync. The underlying paged-KV layout
  still issues per-plane CUDA copies.
- `crates/infer-seam/src/lib.rs:298-340`: expose tier I/O stats through the
  existing backend seam; do not leak Linux or io_uring types.
- `crates/infer-core/src/prefix.rs:487-571,599-707`: preserve the current radix
  ownership and fallback-recompute rules; charge disk completion wait separately
  from D2H/H2D copy time.
- `crates/infer-core/src/lib.rs:162-183` and `crates/infer-server/src/{schema.rs,
  metrics.rs,multiproc_relay.rs}`: publish budget, useful/submitted/metadata
  bytes, direct/mmap mode, operation failures and completion-wait time.

This base version overlaps pages within one mget/mset batch. It does not add a
second request scheduler or background thread. Cross-request compute/I/O overlap
requires a measured remaining wall after batched queue-depth parallelism.

### P3 — GDS hardware gate

- `crates/kv-native-sys/src/gds.rs`: validate Linux, direct-mode file, mounted
  block device, cuFile library, `/etc/cufile.json`, `nvidia_fs` or enabled
  PCIe P2PDMA, and reject compatibility-only mode. Return a structured reason.
- Report the gate through tier stats/logs. Do not call cuFile when the gate is
  false. A real SSD-to-HBM path is licensed only on a pod with local NVMe mounted
  into the container and a passing `gdscheck`/runtime transfer probe.

### P4 — RL extreme-pressure spill

- `crates/autograd/Cargo.toml`: reuse `kv-native-sys`; no training-local disk
  implementation.
- `crates/autograd/src/tensor.rs:124-170,346-387,482-514`: give checkpoint
  residency explicit Host/L3 state. Track live host checkpoint bytes. When the
  opt-in host cap would be crossed, write the just-read-back checkpoint through
  `KvTierStore::insert_chunked`, release its host buffer, and restore/remove it
  immediately before backward upload.
- `crates/autograd/src/tape.rs`: restore L3 inputs at the existing checkpoint
  replay boundary. Fused loss and model ops stay unchanged.
- Configuration is off by default and requires root, host cap and disk cap.
  Missing/failed L3 preserves the existing host path; it never drops the only
  copy.

## Measurement contract

### Substrate

Same 2 GiB payload and filesystem for mmap, direct QD1, and io_uring QD1/8/32.
Report GiB/s, page latency p50/p99, CPU time, mode and fallback reason. Cold reads
must evict page cache; warm mmap is a separate result.

### Write amplification

- Application WA = `(submitted payload + metadata bytes) / useful bytes`.
- Block-device WA = sector-write delta × 512 / useful bytes during an exclusive
  quiet window.
- Controller host WA = NVMe `data_units_written` delta × 512000 / useful bytes.
- NAND WA is unavailable without vendor physical-write telemetry and must not be
  inferred from SMART host writes.

### Serving and model load

Run one binary and one shell with L3 mmap versus direct, concurrency 1/4/8/16,
and prove nonzero disk pages/bytes. Report output tok/s, TTFT, ITL, demote/promote
latency and fallback count. Model load is L3-off versus L3-on, three cold runs
per arm, process start to `/v1/models` HTTP 200; expected result is no regression,
not a storage-engine speedup.

## Current baseline

- Historical Qwen3-4B mmap: demote 411 MB/s; H2D 1475 MB/s; pure 33 MB
  promotion 24 ms; end-to-end promotion 241 ms.
- Historical payload-only WA: 1.0×. Durable manifest and device amplification
  were not measured.
- Historical DSv4 local-NVMe process-to-ready: 80.95 s. Current HEAD needs a
  matched rerun.

## Current measurements

- `/host` virtio substrate, 2 GiB: mmap write 1.14 GiB/s and warm read
  2.40 GiB/s; direct QD32 write 0.19 GiB/s and read 0.18 GiB/s. Direct remains
  opt-in on this mount.
- DeepSeek-V4-Flash-FP8 TP=4 mmap L3, 20 requests × 96 output tokens:
  c=1/4/8/16 = 40.35/73.05/109.81/121.82 output tok/s, all 20/20 complete.
  Rank 0 read 11.40 GB and wrote 12.63 GB from L3 with zero failures; the
  multiprocess stats endpoint is rank-0 scoped.
- DSv4's 22,743,881-byte logical slots proved the aligned-stride requirement.
  Direct then reached the real path with 1.00026× payload read/write
  amplification; QD32 saturated virtio and returned transient `EAGAIN`, so the
  lower-QD zero-failure rerun is required before licensing local NVMe.
- Qwen3-4B matched warm process-to-ready median: L3 off 2.446 s, direct enabled
  2.444 s (-0.08%, noise). DSv4 warm process-to-ready is 30.506 s.
- GDS is blocked: cuFile compatibility mode is enabled, `nvidia_fs` is absent,
  P2PDMA is disabled, and no local NVMe filesystem is mounted.

No final before/after number is valid on the current container overlay. Real
direct/GDS A/B waits for a non-destructive bind mount of an already formatted
local NVMe filesystem; raw `/dev/nvme*n1` devices will not be formatted or
mounted by this work.
