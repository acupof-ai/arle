//! How fast can the host READ an `alloc_uma` buffer?
//!
//! The decode path reads `vocab` f32 of logits out of the arena every token
//! (`infer-vulkan/src/forward.rs:1144`), which is ~993 KB for a 248320-token
//! vocab. `alloc_uma` memory is `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`,
//! and on this APU that is write-combined: sequential host WRITES are fast,
//! host READS are not, because WC memory is uncached and every read is a
//! partial-line fetch. A per-token read of that size is therefore a candidate
//! for a large share of the decode step's host-side gap, and it is invisible in
//! a GPU profile because no dispatch is involved.
//!
//! Compares against a plain `alloc` buffer and against host memcpy so the
//! number can be read as a ratio rather than an absolute.
#![cfg(feature = "vulkan")]

use std::time::Instant;

use vulkan_sys::{DeviceBuffer, VulkanContext};

/// The 248320-token Qwen3.5 vocab, the size the decode path actually reads.
const VOCAB: usize = 248_320;

fn time_reads(buf: &DeviceBuffer<'_>, bytes: usize, iters: usize) -> f64 {
    let mut dst = vec![0u8; bytes];
    // Warm: first touch pays page-in.
    buf.copy_to_host(&mut dst[..]).expect("read");
    let t0 = Instant::now();
    for _ in 0..iters {
        buf.copy_to_host(&mut dst[..]).expect("read");
    }
    std::hint::black_box(&dst);
    t0.elapsed().as_secs_f64() / iters as f64
}

#[test]
fn uma_readback_bandwidth_at_logits_size() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device ({e}); skipping");
            return;
        }
    };
    let bytes = VOCAB * 4;
    let iters = 20;

    let uma = DeviceBuffer::alloc_uma(&ctx, bytes).expect("alloc_uma");
    let plain = DeviceBuffer::alloc(&ctx, bytes).expect("alloc");
    let cached = DeviceBuffer::alloc_host_cached(&ctx, bytes).expect("alloc_host_cached");

    let t_uma = time_reads(&uma, bytes, iters);
    let t_plain = time_reads(&plain, bytes, iters);
    let t_cached = time_reads(&cached, bytes, iters);

    // Host-to-host memcpy of the same size, as the "what the CPU can do" anchor.
    let src = vec![7u8; bytes];
    let mut dst = vec![0u8; bytes];
    dst.copy_from_slice(&src);
    let t0 = Instant::now();
    for _ in 0..iters {
        dst.copy_from_slice(&src);
    }
    std::hint::black_box(&dst);
    let t_memcpy = t0.elapsed().as_secs_f64() / iters as f64;

    let gbps = |t: f64| bytes as f64 / t / 1e9;
    eprintln!(
        "read {} KB ({VOCAB} f32 = one token's logits):",
        bytes / 1024
    );
    eprintln!(
        "  alloc_uma (DEVICE_LOCAL|HOST_VISIBLE) {:8.3} ms  {:7.2} GB/s",
        t_uma * 1e3,
        gbps(t_uma)
    );
    eprintln!(
        "  alloc     (HOST_VISIBLE)              {:8.3} ms  {:7.2} GB/s",
        t_plain * 1e3,
        gbps(t_plain)
    );
    eprintln!(
        "  alloc_host_cached (HOST_CACHED)       {:8.3} ms  {:7.2} GB/s",
        t_cached * 1e3,
        gbps(t_cached)
    );
    eprintln!(
        "  host memcpy                           {:8.3} ms  {:7.2} GB/s",
        t_memcpy * 1e3,
        gbps(t_memcpy)
    );
    eprintln!(
        "  -> HOST_CACHED is {:.0}x faster than alloc_uma, saving {:.2} ms per token",
        t_uma / t_cached,
        (t_uma - t_cached) * 1e3
    );
    eprintln!(
        "  -> uma read is {:.1}x slower than memcpy; at 1 read/token that is \
         {:.2} ms of every decode step",
        t_uma / t_memcpy,
        t_uma * 1e3
    );
}
