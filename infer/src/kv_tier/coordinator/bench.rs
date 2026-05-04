//! Copy-throughput benchmarks for the T1 (host pinned) ↔ T2 (disk) path.
//!
//! Run:
//! ```text
//! cargo test --release --no-default-features --features no-cuda \
//!   -p infer --lib bench_kv_tier_copy_throughput \
//!   -- --ignored --nocapture
//! ```
//!
//! These numbers feed the CPU↔GPU layer-wise overlap analysis in
//! `docs/plans/2026-05-04-kv-tier-hicache-borrowed-improvements.md`:
//! per-layer KV byte size × measured T1↔T2 bandwidth tells us whether
//! the transfer fits inside one layer's compute budget.

use std::sync::Arc;
use std::time::Instant;

use tempfile::tempdir;

use super::builder::CoordinatorBuilder;
use super::events::{CoordinatorEvent, FetchRequest, StoreRequest, StoreTarget};
use crate::kv_tier::HostPinnedPool;
use crate::kv_tier::host_pool::SharedHostPinnedPool;
use crate::kv_tier::tier::BlockLocation;
use crate::kv_tier::transport::disk::DiskStore;
use crate::types::{BlockFingerprint, BlockId};

const SIZES_BYTES: &[usize] = &[
    4 * 1024,
    64 * 1024,
    1024 * 1024,
    16 * 1024 * 1024,
    256 * 1024 * 1024,
];

fn iters_for(size: usize) -> usize {
    match size {
        s if s <= 4 * 1024 => 5_000,
        s if s <= 64 * 1024 => 2_000,
        s if s <= 1024 * 1024 => 500,
        s if s <= 16 * 1024 * 1024 => 50,
        _ => 10,
    }
}

fn pool_capacity_for(size: usize) -> usize {
    // Headroom for region metadata + at least 4 simultaneous regions.
    (size * 4).max(64 * 1024)
}

fn fmt_bandwidth(bytes_total: u128, secs: f64) -> String {
    let mb = bytes_total as f64 / (1024.0 * 1024.0);
    format!("{:>9.1} MiB/s", mb / secs)
}

fn fmt_avg_us(secs: f64, iters: usize) -> String {
    format!("{:>9.2} µs", (secs / iters as f64) * 1e6)
}

fn fmt_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:>5} MiB", bytes / (1024 * 1024))
    } else if bytes >= 1024 {
        format!("{:>5} KiB", bytes / 1024)
    } else {
        format!("{:>5}   B", bytes)
    }
}

fn bench_t1_self_copy(size: usize) -> (f64, usize) {
    let pool = SharedHostPinnedPool::new(HostPinnedPool::new(pool_capacity_for(size)).unwrap());
    let payload: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();

    // Warm-up
    {
        let region = {
            let mut p = pool.lock().unwrap();
            let r = p.reserve(size).unwrap().unwrap();
            p.as_mut_slice(r).copy_from_slice(&payload);
            r
        };
        let _ = pool.read_region(region).unwrap();
        pool.release_region(region).unwrap();
    }

    let iters = iters_for(size);
    let start = Instant::now();
    for _ in 0..iters {
        let region = {
            let mut p = pool.lock().unwrap();
            let r = p.reserve(size).unwrap().unwrap();
            p.as_mut_slice(r).copy_from_slice(&payload);
            r
        };
        let read = pool.read_region(region).unwrap();
        std::hint::black_box(&read);
        pool.release_region(region).unwrap();
    }
    (start.elapsed().as_secs_f64(), iters)
}

fn bench_t2_disk_roundtrip(size: usize, fsync_each_block: bool) -> (f64, usize) {
    let dir = tempdir().unwrap();
    let store = DiskStore::new(dir.path());
    let payload: Vec<u8> = (0..size).map(|i| ((i * 31) & 0xFF) as u8).collect();

    let warm_fp = BlockFingerprint([0xAA; 16]);
    let warm_loc = store
        .put_block_with_fsync(warm_fp, 1, &payload, fsync_each_block)
        .unwrap();
    let _ = store.get_block(&warm_loc, Some(warm_fp)).unwrap();
    store.delete_block(&warm_loc).unwrap();

    let iters = iters_for(size);
    let start = Instant::now();
    for i in 0..iters {
        let mut fp_bytes = [0u8; 16];
        fp_bytes[0] = (i & 0xFF) as u8;
        fp_bytes[1] = ((i >> 8) & 0xFF) as u8;
        let fp = BlockFingerprint(fp_bytes);
        let loc = store
            .put_block_with_fsync(fp, 1, &payload, fsync_each_block)
            .unwrap();
        let read = store.get_block(&loc, Some(fp)).unwrap();
        std::hint::black_box(&read);
        store.delete_block(&loc).unwrap();
    }
    (start.elapsed().as_secs_f64(), iters)
}

fn bench_coordinator_full_cycle(size: usize) -> (f64, usize) {
    let dir = tempdir().unwrap();
    let disk_store = Arc::new(DiskStore::new(dir.path()));
    let (coordinator, handle, events) = CoordinatorBuilder::new(8).disk_store(disk_store).build();
    let pool = SharedHostPinnedPool::new(HostPinnedPool::new(pool_capacity_for(size)).unwrap());
    let payload: Vec<u8> = (0..size).map(|i| ((i * 17) & 0xFF) as u8).collect();

    let do_one = |i: u64| {
        let region = {
            let mut p = pool.lock().unwrap();
            let r = p.reserve(size).unwrap().unwrap();
            p.as_mut_slice(r).copy_from_slice(&payload);
            r
        };
        let mut fp_bytes = [0u8; 16];
        fp_bytes[..8].copy_from_slice(&i.to_le_bytes());
        let fp = BlockFingerprint(fp_bytes);
        let store_ticket = handle
            .submit_store(vec![StoreRequest {
                block_id: BlockId(1),
                fingerprint: fp,
                kv_format_tag: 1,
                host_pool: pool.clone(),
                host_region: region,
                target: StoreTarget::Disk,
            }])
            .unwrap();
        assert!(coordinator.run_once().unwrap());
        let _ = events.recv().unwrap();
        let payload_len = match events.recv().unwrap() {
            CoordinatorEvent::StoreCompleted { ticket, locations } => {
                assert_eq!(ticket, store_ticket);
                match &locations[0].1 {
                    BlockLocation::Disk { payload_len, .. } => *payload_len,
                    other => panic!("expected disk location, got {other:?}"),
                }
            }
            other => panic!("unexpected event: {other:?}"),
        };
        pool.release_region(region).unwrap();

        let _fetch_ticket = handle
            .submit_fetch(vec![FetchRequest {
                block_id: BlockId(1),
                source: BlockLocation::Disk {
                    fingerprint: fp,
                    payload_len,
                },
                byte_len: usize::try_from(payload_len).unwrap(),
                host_pool: pool.clone(),
            }])
            .unwrap();
        assert!(coordinator.run_once().unwrap());
        let _ = events.recv().unwrap();
        let fetched = match events.recv().unwrap() {
            CoordinatorEvent::FetchCompleted { blocks, .. } => blocks,
            other => panic!("unexpected event: {other:?}"),
        };
        // Note: the production fetch path stops here — the prefill kernel
        // consumes the host region in place via paged_kv. We deliberately
        // skip a verification `read_region` inside the timed loop; the
        // warm-up round below validates correctness once.
        pool.release_region(fetched[0].host_region).unwrap();
    };

    // Warm-up + correctness check: do one full cycle and verify byte-equality.
    do_one(0);
    {
        let region = {
            let mut p = pool.lock().unwrap();
            let r = p.reserve(size).unwrap().unwrap();
            p.as_mut_slice(r).copy_from_slice(&payload);
            r
        };
        let warm_fp = BlockFingerprint([0xFE; 16]);
        let st = handle
            .submit_store(vec![StoreRequest {
                block_id: BlockId(1),
                fingerprint: warm_fp,
                kv_format_tag: 1,
                host_pool: pool.clone(),
                host_region: region,
                target: StoreTarget::Disk,
            }])
            .unwrap();
        assert!(coordinator.run_once().unwrap());
        let _ = events.recv().unwrap();
        let plen = match events.recv().unwrap() {
            CoordinatorEvent::StoreCompleted { ticket, locations } => {
                assert_eq!(ticket, st);
                match &locations[0].1 {
                    BlockLocation::Disk { payload_len, .. } => *payload_len,
                    other => panic!("warm-up: expected disk location, got {other:?}"),
                }
            }
            other => panic!("warm-up: unexpected store event: {other:?}"),
        };
        pool.release_region(region).unwrap();
        let _ = handle
            .submit_fetch(vec![FetchRequest {
                block_id: BlockId(1),
                source: BlockLocation::Disk {
                    fingerprint: warm_fp,
                    payload_len: plen,
                },
                byte_len: usize::try_from(plen).unwrap(),
                host_pool: pool.clone(),
            }])
            .unwrap();
        assert!(coordinator.run_once().unwrap());
        let _ = events.recv().unwrap();
        let fetched = match events.recv().unwrap() {
            CoordinatorEvent::FetchCompleted { blocks, .. } => blocks,
            other => panic!("warm-up: unexpected fetch event: {other:?}"),
        };
        let bytes = pool.read_region(fetched[0].host_region).unwrap();
        assert_eq!(bytes, payload, "warm-up cycle byte-equality failed");
        pool.release_region(fetched[0].host_region).unwrap();
    }

    let iters = iters_for(size);
    let start = Instant::now();
    for i in 0..iters as u64 {
        do_one(i + 1);
    }
    (start.elapsed().as_secs_f64(), iters)
}

#[test]
#[ignore = "perf measurement; run with --ignored --nocapture"]
fn bench_kv_tier_copy_throughput() {
    println!();
    println!("# T1↔T2 copy-throughput micro-benchmark");
    println!();
    println!(
        "Backing: HostPinnedPool (Zig arena, kv-native-sys), DiskStore (tmpfs/disk under tempdir)."
    );
    println!("All paths exercised through actual ARLE types — no mock substrate.");
    println!();
    println!(
        "| {:^11} | {:^7} | {:^21} | {:^21} | {:^21} | {:^21} | {:^21} |",
        "size",
        "iters",
        "T1 self-copy",
        "T2 put+get (fsync)",
        "T2 put+get (no-fsync)",
        "Coord. T1→T2→T1",
        "(coord ops/s)"
    );
    println!(
        "|{:-<13}|{:-<9}|{:-<23}|{:-<23}|{:-<23}|{:-<23}|{:-<23}|",
        "", "", "", "", "", "", ""
    );
    for &size in SIZES_BYTES {
        let (t1_secs, t1_iters) = bench_t1_self_copy(size);
        let (t2_secs, t2_iters) = bench_t2_disk_roundtrip(size, true);
        let (t2_nf_secs, t2_nf_iters) = bench_t2_disk_roundtrip(size, false);
        let (cyc_secs, cyc_iters) = bench_coordinator_full_cycle(size);

        let t1_total = (size as u128) * (t1_iters as u128) * 2; // write + read
        let t2_total = (size as u128) * (t2_iters as u128) * 2; // put + get
        let t2_nf_total = (size as u128) * (t2_nf_iters as u128) * 2;
        let cyc_total = (size as u128) * (cyc_iters as u128) * 2; // store-leg + fetch-leg

        let cyc_ops_per_s = cyc_iters as f64 / cyc_secs;
        println!(
            "| {} | {:>7} | {} | {} | {} | {} | {:>14.2} ops/s |",
            fmt_size(size),
            t1_iters.max(t2_iters).max(cyc_iters),
            fmt_bandwidth(t1_total, t1_secs),
            fmt_bandwidth(t2_total, t2_secs),
            fmt_bandwidth(t2_nf_total, t2_nf_secs),
            fmt_bandwidth(cyc_total, cyc_secs),
            cyc_ops_per_s,
        );
        // Latency view (currently unused but kept for ad-hoc inspection).
        let _ = (
            fmt_avg_us(t1_secs, t1_iters),
            fmt_avg_us(t2_secs, t2_iters),
            fmt_avg_us(t2_nf_secs, t2_nf_iters),
        );
    }
    println!();
    println!("Throughput convention: each row counts BOTH directions of the round-trip");
    println!("(write+read for T1, put+get for T2, store-leg+fetch-leg for coordinator),");
    println!("so the MiB/s value is the byte-volume throughput of the full cycle.");
    println!("T2 (fsync) is `put_block` default (atomic write + fsync data + fsync parent dir);");
    println!("T2 (no-fsync) is `put_block_with_fsync(false)` (rename-only, no fsync).");
}
