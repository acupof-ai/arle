# KvTierStore per-page cold read latency — L2 host DRAM vs L3 NVMe — CUDA, 2026-09-09

> Status: Shipped

## Goal

Single-page cold read latency for `KvTierStore` L2 (host DRAM) and L3 (NVMe),
against the DSv4-Flash decode step budget. This is roadmap 2026-08-24 Goal 2
item 1 — the gate for item 2, per-page paging of active sequences: if a cold
page fault costs more than a decode step, paging active sequences from that
tier is not viable.

## Hypothesis

A decode step stalling on a cold page pays `read(key)` plus consuming the page
bytes. If the p50 fault cost exceeds the decode step budget, per-page paging
from that tier is killed.

## Parameters

- Microbench: `crates/kv-native-sys/examples/cold_page_read.rs`, run as
  `cold_page_read --disk /mnt/data02 --pages 64` (64 pages per arm, release).
- **Page size: 22,830,241 bytes (21.77 MiB)** — the DSv4-Flash 64-token prefix
  entry: sliding-window + compressed KV across 43 layers
  (sliding_window=128, head_dim=512, compress_ratios [0,0,4,128,…],
  index_head_dim=128), per `dsv4_prefix_entry_max_bytes`.
- L2 arm: `KvTierStore::with_budget(64×page, page)`; a 512 MiB sequential pass
  evicts the payloads from CPU cache before the timed reads.
- L3 arm: `with_budget(0, page)` + `set_disk` with `ARLE_KV_DISK_IO=direct`;
  polled to `disk_pages() == 64` before reads so every read is a disk hit.
- Per sample, timed separately: `read(key)`; a `copy_from_slice` of the full
  page into a reusable sink (cold copy); a second copy (warm, bounds the copy
  with the source cached). The fault cost is read + cold copy.
- **Decode step budget: 15.25 ms** = 1000 / 65.56 tok/s, the 2026-08-14
  re-anchored DSv4-Flash-FP8 c=1 DSpark baseline (roadmap Killed section).
  Other c=1 steps for scale: DSv4-Flash NVFP4 22.6 ms (44.4 tok/s),
  Qwen3.8-27B-NVFP4 11.8 ms.

## Environment

- Host: H20 box, pod container; GPUs 5+7 idle (bench is CPU/disk only).
- **L2 path:** host DRAM, `BTreeMap<u64, HostDemotedEntry { payload: Vec<u8> }>`
  (not pinned); `read()` returns `Cow::Borrowed` — zero-copy
  (`crates/kv-native-sys/src/kv_tier.rs:1640`).
- **L3 path:** `/mnt/data02` NVMe, O_DIRECT via io_uring (`DirectStore`,
  `ARLE_KV_DISK_IO=direct`), queue depth 8 default (`ARLE_KV_IO_URING_QD`,
  `direct_store.rs:109`); one slot per page → one bio per read; `read()`
  returns `Cow::Owned` (allocation + direct read, `kv_tier.rs:1673`).
- Binary: lane/b checkout, `cargo build --release -p kv-native-sys
  --example cold_page_read`.

## Results

| tier | read p50 | cold-copy p50 | warm-copy p50 | fault p50 | fault p99 | p50 / 15.25 ms |
|---|---:|---:|---:|---:|---:|---:|
| L2 host DRAM | 0.001 ms | 2.306 ms | 2.156 ms | **2.307 ms** | 9.970 ms | 15.1% |
| L3 NVMe O_DIRECT | 19.198 ms | 1.673 ms | 1.653 ms | **20.848 ms** | 29.399 ms | 136.7% |

`location[0]=Some(Disk) disk_pages=64 host_pages=0` confirms the L3 arm read
every page from disk. L2 effective copy bandwidth: 21.77 MiB ×2 (read+write) /
2.306 ms ≈ 18.9 GB/s. L3 read: 21.77 MiB / 19.198 ms ≈ 1.13 GB/s.

Raw log: pod `/host/runs/b2_bench3.log` (2026-09-09).

## Problems

The first consumer was a byte-sum reduction, which reported L2 fault p50 4.17 ms
and L3 24.1 ms. That version is wrong as a memory-system measurement: its own
decomposition showed cold-touch ≈ warm-touch (4.183 vs 4.108 ms), i.e. the loop
was compute-bound and measured the reduction, not the page read. The memcpy
consumer in this entry is the correct one — it models the host-side cost of
consuming a page (promote/staging copy) — and its L2 number (2.31 ms) is the
trustworthy one. The L3 read itself is 19–21 ms in both versions (run-to-run
variance on the shared NVMe); the L3 fault total differs between versions only
because the consumer changed. The verdict is identical under either consumer:
L3 exceeds the budget on the read alone.

The L2 p99 (9.97 ms) is a single outlier at 64 samples; p50 and the
read/copy split are stable.

## Learnings

**KILL for L3, pass for L2.** A single L3 cold page fault costs 20.8 ms p50 —
1.37× the 15.25 ms DSpark decode budget, 1.77× the 11.8 ms Qwen3.8 budget, and
the read alone (19.2 ms) consumes 85% of the 22.6 ms NVFP4 budget before the
step does any inference work. Per-page paging of active sequences from L3 is
not viable at the DSv4-Flash page size. L2 fits: the read path is ~free
(1 µs borrow) and the fault p50 is 2.31 ms (15% of budget); in production the
consumer is a GPU DMA transfer (~0.9 ms for 21.77 MiB at PCIe gen4 x16
25 GB/s), which overlaps with decode compute rather than stalling it.

Roadmap Goal 2 item 2 (per-page paging of active sequences) was gated on this
measurement and is deleted with this entry; the L2 path remains open for
designs that page from host DRAM only.
