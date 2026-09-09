//! Microbench: `KvTierStore` per-page cold read latency (L2 host DRAM, L3 NVMe).
//!
//! Roadmap 2026-08-24 Goal 2 item 1: bound whether a cold page read fits inside
//! one decode step — the gate for per-page paging of active sequences. Each
//! sample times `read(key)` and a full copy of the page bytes separately: the
//! read is a BTreeMap borrow on L2 (microseconds) and an alloc + O_DIRECT bio
//! on L3; the copy is the host-side cost of consuming the bytes (the no-GPU
//! proxy for a promote/staging copy). A second warm copy bounds the copy with
//! the source cached.
//!
//! Default page size is the DSv4-Flash 64-token prefix entry (21.77 MiB:
//! sliding-window + compressed KV across 43 layers), computed by
//! `dsv4_prefix_entry_max_bytes`. Decode step budget 15.25 ms (DSv4-Flash c=1
//! 65.56 tok/s, roadmap 2026-08-14 baseline).
//!
//! The L3 arm uses O_DIRECT (`ARLE_KV_DISK_IO=direct`) so reads are cold by
//! design — no page-cache drop on a shared box. An mmap cold read issues the
//! same NVMe bio on a major fault, so this is the same bound modulo readahead.
//!
//! Usage: `cargo run --release --example cold_page_read -- --disk /mnt/data02`

use std::env;
use std::fs;
use std::time::{Duration, Instant};

use kv_native_sys::KvTierStore;

/// DSv4-Flash 64-token prefix entry: sliding-window + compressed KV, 43 layers.
const DSV4_PAGE_BYTES: usize = 22_830_241;
/// DSv4-Flash c=1 decode step, 65.56 tok/s (roadmap 2026-08-14 baseline).
const DECODE_STEP_BUDGET_MS: f64 = 15.25;

fn main() -> anyhow::Result<()> {
    let mut pages = 64usize;
    let mut page_bytes = DSV4_PAGE_BYTES;
    let mut disk: Option<String> = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pages" => {
                pages = args
                    .next()
                    .ok_or(anyhow::anyhow!("--pages needs a value"))?
                    .parse()?
            }
            "--page-bytes" => {
                page_bytes = args
                    .next()
                    .ok_or(anyhow::anyhow!("--page-bytes needs a value"))?
                    .parse()?
            }
            "--disk" => disk = Some(args.next().ok_or(anyhow::anyhow!("--disk needs a value"))?),
            other => anyhow::bail!("unknown arg {other}"),
        }
    }
    println!(
        "page_bytes={page_bytes} ({:.2} MiB) pages={pages} decode_step_budget_ms={DECODE_STEP_BUDGET_MS}",
        page_bytes as f64 / 1048576.0
    );

    // L2 arm: host-only store; pages are cold in CPU cache after the eviction pass.
    let mut store = KvTierStore::with_budget(pages * page_bytes, page_bytes);
    for k in 0..pages as u64 {
        assert!(
            store.insert(k, pattern(k, page_bytes)),
            "L2 insert {k} refused"
        );
    }
    evict_cpu_cache();
    let mut sink_buf = vec![0u8; page_bytes];
    let mut l2: Vec<[Duration; 3]> = Vec::with_capacity(pages);
    for k in 0..pages as u64 {
        let t = Instant::now();
        let cow = store.read(k)?;
        let read = t.elapsed();
        let t = Instant::now();
        consume(&cow, &mut sink_buf);
        let cold = t.elapsed();
        let t = Instant::now();
        consume(&cow, &mut sink_buf);
        let warm = t.elapsed();
        l2.push([read, cold, warm]);
    }
    report("L2-host-dram", &l2);

    // L3 arm: host budget 0 so every insert lands on disk and every read is a
    // disk hit (no L2 re-insert). O_DIRECT keeps the page cache out of the path.
    if let Some(root) = disk {
        let root = format!("{root}/kv-cold-bench-{}", std::process::id());
        fs::create_dir_all(&root)?;
        // SAFETY: single-threaded bench, no store constructed yet; the store
        // reads this var once inside set_disk.
        unsafe { env::set_var("ARLE_KV_DISK_IO", "direct") };
        let mut store = KvTierStore::with_budget(0, page_bytes);
        assert!(
            store.set_disk(root.clone().into(), pages * page_bytes, page_bytes),
            "set_disk refused (budget below one page?)"
        );
        for k in 0..pages as u64 {
            assert!(
                store.insert(k, pattern(k, page_bytes)),
                "L3 insert {k} refused"
            );
        }
        // Async disk writer: poll until every page has a durable slot.
        while store.disk_pages() < pages {
            std::thread::sleep(Duration::from_millis(20));
        }
        let mut l3: Vec<[Duration; 3]> = Vec::with_capacity(pages);
        for k in 0..pages as u64 {
            let t = Instant::now();
            let cow = store.read(k)?;
            let read = t.elapsed();
            let t = Instant::now();
            consume(&cow, &mut sink_buf);
            let cold = t.elapsed();
            let t = Instant::now();
            consume(&cow, &mut sink_buf);
            let warm = t.elapsed();
            l3.push([read, cold, warm]);
        }
        report("L3-nvme-direct", &l3);
        println!(
            "location[0]={:?} disk_pages={} host_pages={}",
            store.location(0),
            store.disk_pages(),
            store.host_demoted_pages()
        );
        store.persist();
        let _ = fs::remove_dir_all(&root);
    }
    Ok(())
}

/// Deterministic, non-repeating payload so a copy moves real bytes.
fn pattern(key: u64, len: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    let mut x = key.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    while v.len() < len {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        v.extend_from_slice(&x.to_le_bytes());
    }
    v.truncate(len);
    v
}

/// Copy the page into a reusable sink: the host-side cost of consuming the
/// bytes (the no-GPU proxy for a promote/staging copy). A second pass bounds
/// the copy with the source warm in cache.
fn consume(bytes: &[u8], sink: &mut [u8]) {
    sink[..bytes.len()].copy_from_slice(bytes);
    std::hint::black_box(&sink[..bytes.len()]);
}

/// Flush CPU caches: a 512 MiB sequential pass exceeds any L3, so the store's
/// payload pages (written above) are evicted to DRAM before the timed reads.
fn evict_cpu_cache() {
    let v = vec![0u8; 512 << 20];
    let mut s = 0u64;
    for chunk in v.chunks_exact(8) {
        s = s.wrapping_add(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
    std::hint::black_box(s);
}

/// Each sample is [read, cold-copy, warm-copy]; the fault cost a decode step
/// pays is read + cold. Warm bounds the copy with the source cached.
fn report(label: &str, samples: &[[Duration; 3]]) {
    let pct = |idx: usize, p: f64| -> f64 {
        let mut v: Vec<u128> = samples.iter().map(|s| s[idx].as_nanos()).collect();
        v.sort_unstable();
        v[(v.len() as f64 * p) as usize] as f64 / 1e6
    };
    let read = pct(0, 0.5);
    let cold = pct(1, 0.5);
    let warm = pct(2, 0.5);
    let mut totals: Vec<u128> = samples
        .iter()
        .map(|s| s[0].as_nanos() + s[1].as_nanos())
        .collect();
    totals.sort_unstable();
    let total50 = totals[totals.len() / 2] as f64 / 1e6;
    let total99 = totals[(totals.len() as f64 * 0.99) as usize] as f64 / 1e6;
    println!(
        "{label}: read={read:.3}ms cold-copy={cold:.3}ms warm-copy={warm:.3}ms  fault(read+cold) p50={total50:.3}ms p99={total99:.3}ms  p50/budget={:.1}% p99/budget={:.1}%",
        total50 / DECODE_STEP_BUDGET_MS * 100.0,
        total99 / DECODE_STEP_BUDGET_MS * 100.0
    );
}
