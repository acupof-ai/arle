//! R2: can a 47.68 GiB PLE/n-gram table live as a file-backed mmap while the
//! GPU holds ~70 GiB, and what does a per-token gather cost?
//!
//! Qwen3.8-Flash-Next's PLE n-gram embedding is 320,001,536 rows x 160 FP8 =
//! 47.68 GiB, and decode reads exactly `ngram_heads = 16` uniformly-random rows
//! of 160 B per token (2560 B/token). It is a lookup table, not a matmul, so it
//! is the one component that can sit outside device memory — but only if the
//! gather is cheap under real memory pressure.
//!
//! Two things this settles that arithmetic cannot:
//!
//! 1. **Does a device-local Vulkan allocation consume OS-visible RAM?** On this
//!    APU Windows reports 63.6 GB while Vulkan reports a 74.43 GiB DEVICE_LOCAL
//!    heap — the two overlap in ways that decide how much page cache is left for
//!    the table. If device allocations come out of the BIOS carve-out, the OS
//!    keeps its full RAM for cache and the table is mostly resident; if they come
//!    out of OS RAM, the table thrashes.
//! 2. **What does the gather actually cost at queue depth?** A serialized gather
//!    measured ~4.24 ms/token on this NVMe; at QD16 the same work took 0.585 ms.
//!    Prefill is the exposure — 4096 tokens x 16 rows — so the fan-out matters
//!    more than the per-row latency.
//!
//! Opt-in: it allocates tens of GiB and reads from disk.
//!
//! ```text
//! ARLE_PLE_PROBE=1 cargo test -p vulkan-kernels --features vulkan \
//!     --test device_ple_mmap_probe --release -- --nocapture --test-threads=1
//! ```
#![cfg(feature = "vulkan")]

use std::time::Instant;

use vulkan_sys::{DeviceBuffer, VulkanContext};

/// One PLE row: `ple_embed_dim / ngram_heads = 2560 / 16 = 160` FP8 values.
const ROW_BYTES: usize = 160;
/// `ngram_heads = (ngram_size - 1) * heads_per_ngram = 2 * 8`.
const ROWS_PER_TOKEN: usize = 16;

fn free_ram_gib() -> f64 {
    // `wmic` is deprecated but present; CIM via powershell costs ~300 ms/call.
    let out = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory",
        ])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout)
            .trim()
            .parse::<f64>()
            .map(|kb| kb / 1024.0 / 1024.0)
            .unwrap_or(f64::NAN),
        Err(_) => f64::NAN,
    }
}

#[test]
fn ple_table_as_mmap_under_device_pressure() {
    if std::env::var("ARLE_PLE_PROBE").is_err() {
        eprintln!("set ARLE_PLE_PROBE=1 to run the PLE mmap probe; skipping");
        return;
    }
    let dir = std::path::Path::new(r"C:\Users\Asus\models\qwen3.8-flash-next-nvfp4");
    let mut shards: Vec<std::path::PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("model-plefp8-") && n.ends_with(".safetensors"))
            })
            .collect(),
        Err(e) => {
            eprintln!("no checkpoint at {}: {e}; skipping", dir.display());
            return;
        }
    };
    shards.sort();
    if shards.is_empty() {
        eprintln!("no model-plefp8-* shards; skipping");
        return;
    }
    let table_bytes: u64 = shards
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();
    eprintln!(
        "PLE table: {} shards, {:.2} GiB",
        shards.len(),
        table_bytes as f64 / (1u64 << 30) as f64
    );

    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device ({e}); skipping");
            return;
        }
    };

    // ── Q1: does a device-local allocation eat OS RAM? ──────────────────────
    let before = free_ram_gib();
    // Allocate in 2 GiB slabs: `maxMemoryAllocationSize` on this driver is 2 GiB,
    // which is also why the real loader will need a slab suballocator.
    const SLAB: usize = 2 * (1 << 30) - (1 << 20);
    let target_gib = std::env::var("ARLE_PLE_PROBE_DEVICE_GIB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(60);
    let mut slabs: Vec<DeviceBuffer<'_>> = Vec::new();
    let mut committed = 0usize;
    while committed + SLAB <= target_gib * (1 << 30) {
        match DeviceBuffer::alloc_uma(&ctx, SLAB) {
            Ok(b) => {
                committed += SLAB;
                slabs.push(b);
            }
            Err(e) => {
                eprintln!(
                    "  device alloc stopped at {:.1} GiB: {e}",
                    committed as f64 / (1u64 << 30) as f64
                );
                break;
            }
        }
    }
    let after = free_ram_gib();
    eprintln!(
        "\ndevice-local committed {:.1} GiB in {} slabs",
        committed as f64 / (1u64 << 30) as f64,
        slabs.len()
    );
    eprintln!(
        "  OS free RAM  {before:.1} -> {after:.1} GiB   (delta {:.1})",
        before - after
    );
    eprintln!(
        "  -> device allocations {} OS RAM; page cache left for the table is what remains",
        if (before - after) > committed as f64 / (1u64 << 30) as f64 * 0.5 {
            "COME OUT OF"
        } else {
            "DO NOT come out of"
        }
    );

    // ── Q2: gather cost under that pressure ─────────────────────────────────
    let files: Vec<(std::fs::File, u64)> = shards
        .iter()
        .filter_map(|p| {
            let f = std::fs::File::open(p).ok()?;
            let len = f.metadata().ok()?.len();
            Some((f, len))
        })
        .collect();
    eprintln!(
        "\ngather: {ROWS_PER_TOKEN} rows x {ROW_BYTES} B per token, uniform-random over the table"
    );
    for threads in [1usize, 4, 16, 32] {
        let tokens = 200usize;
        let t0 = Instant::now();
        std::thread::scope(|s| {
            let per = tokens.div_ceil(threads);
            for t in 0..threads {
                let files = &files;
                s.spawn(move || {
                    // Deterministic per-thread xorshift; no rand dependency.
                    let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ ((t as u64 + 1) << 32);
                    let mut buf = [0u8; ROW_BYTES];
                    for _ in 0..per * ROWS_PER_TOKEN / threads.max(1) {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        let (f, len) = &files[(x as usize) % files.len()];
                        let off = x % (len - ROW_BYTES as u64);
                        read_at(f, &mut buf, off);
                        std::hint::black_box(&buf);
                    }
                });
            }
        });
        let dt = t0.elapsed().as_secs_f64();
        let per_token = dt / tokens as f64;
        eprintln!(
            "  threads={threads:<3} {:8.3} ms/token   ({:6.0} rows/s)",
            per_token * 1e3,
            tokens as f64 * ROWS_PER_TOKEN as f64 / dt
        );
    }
    drop(slabs);
}

#[cfg(windows)]
fn read_at(f: &std::fs::File, buf: &mut [u8], off: u64) {
    use std::os::windows::fs::FileExt;
    // `seek_read` is positional, so concurrent readers on one handle are safe.
    let _ = f.seek_read(buf, off);
}

#[cfg(not(windows))]
fn read_at(f: &std::fs::File, buf: &mut [u8], off: u64) {
    use std::os::unix::fs::FileExt;
    let _ = f.read_at(buf, off);
}
