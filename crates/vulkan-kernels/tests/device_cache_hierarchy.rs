//! Where are this part's cache cliffs, in bytes?
//!
//! Every residency and batching decision on this box turns on one question: how
//! big can a working set be before it falls out of on-chip cache and onto the
//! 256 GB/s LPDDR5X. That number has been quoted from spec sheets ("~32 MB
//! MALL") and never measured here, while the numbers that WERE measured kept
//! contradicting the naive roofline — the gated-delta kernel sustains ~288 GB/s,
//! *above* DRAM peak, because its 3 MB state is cache-resident.
//!
//! The sweep reads a buffer of `S` bytes end-to-end many times and reports
//! achieved bandwidth against `S`. Bandwidth is flat while `S` fits a cache tier
//! and steps down at each cliff, so the knees ARE the tier sizes — no vendor
//! documentation required.
//!
//! Why it matters for `qwen4_exp` specifically: its per-layer working set is
//! unusually small (hidden 2560, `moe_intermediate_size` 640, so ten active
//! experts are ~27 MB at NVFP4), which is the regime where batching — MTP
//! verify, or concurrency — reuses a weight tile several times before it is
//! evicted instead of re-streaming it per token. A dense 27B layer is 235 MB and
//! has no such regime. Sizing that reuse needs the real cliff, not a guess.
//!
//! Opt-in: allocates up to 1 GiB and runs a few seconds.
//!
//! ```text
//! ARLE_CACHE_SWEEP=1 cargo test -p vulkan-kernels --features vulkan \
//!     --test device_cache_hierarchy --release -- --nocapture --test-threads=1
//! ```
#![cfg(feature = "vulkan")]

use std::time::Instant;

use vulkan_kernels::{
    Dispatch, Kernel, KernelCache, record_dispatch, rms_norm_dispatch_rows, rms_norm_params_rows,
};
use vulkan_sys::{CommandRecorder, DescriptorSet, DeviceBuffer, VulkanContext};

/// Repeats per timed run. Enough that a fully cache-resident pass is not
/// dominated by submit latency.
const PASSES: usize = 24;

/// Read `bytes` of `buf` with a row-batched RMSNorm — a pure streaming read of
/// the weight-shaped kind the decode path actually issues, not a synthetic copy.
/// Returns achieved GB/s.
#[allow(clippy::too_many_arguments)]
fn sweep_one<'a>(
    ctx: &'a VulkanContext,
    cache: &mut KernelCache<'a>,
    src: &DeviceBuffer<'a>,
    weight: &DeviceBuffer<'a>,
    dst: &DeviceBuffer<'a>,
    bytes: usize,
) -> f64 {
    // One row per workgroup, 4096 f32 per row: wide enough that the launch is
    // not dominated by per-row overhead, narrow enough that a small working set
    // still spreads over every WGP.
    const NCOLS: usize = 4096;
    let nrows = bytes / (NCOLS * 4);
    if nrows == 0 {
        return f64::NAN;
    }
    let push = rms_norm_params_rows(NCOLS as u32, nrows as u32, NCOLS as u32, 1e-6).to_le_bytes();
    let (pipeline, layout) = cache
        .get(
            ctx,
            Kernel::RmsNorm,
            Kernel::RmsNorm.specialization_u32(),
            push.len() as u32,
            3,
        )
        .expect("rms_norm pipeline");
    let read_bytes = (nrows * NCOLS * 4) as u64;
    let set = DescriptorSet::storage_buffers_ranged(
        ctx,
        layout,
        &[
            (src, 0, read_bytes),
            (weight, 0, (NCOLS * 4) as u64),
            (dst, 0, read_bytes),
        ],
    )
    .expect("bind");
    let d: Dispatch = rms_norm_dispatch_rows(nrows as u32);

    let run = || {
        let mut rec = CommandRecorder::new(ctx).expect("recorder");
        rec.begin().expect("begin");
        for _ in 0..PASSES {
            record_dispatch(&mut rec, pipeline, &set, &push, [d.x, d.y, d.z]);
            rec.barrier();
        }
        let t0 = Instant::now();
        rec.submit_and_wait().expect("submit");
        t0.elapsed().as_secs_f64()
    };
    run(); // warm: first pass pays page-in and pipeline creation
    let secs = run();
    // The kernel reads the row AND writes it, so the link sees 2x.
    (read_bytes as f64 * 2.0 * PASSES as f64) / secs / 1e9
}

#[test]
fn report_cache_cliffs() {
    if std::env::var("ARLE_CACHE_SWEEP").is_err() {
        eprintln!("set ARLE_CACHE_SWEEP=1 to run the cache-hierarchy sweep; skipping");
        return;
    }
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device ({e}); skipping");
            return;
        }
    };
    let mut cache = KernelCache::new();
    const MAX: usize = 1 << 30;
    let src = DeviceBuffer::alloc_uma(&ctx, MAX).expect("alloc src");
    let dst = DeviceBuffer::alloc_uma(&ctx, MAX).expect("alloc dst");
    let weight = {
        let ones: Vec<u8> = (0..4096).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        let mut b = DeviceBuffer::alloc_uma(&ctx, ones.len()).expect("alloc weight");
        b.copy_from_host(&ones).expect("upload weight");
        b
    };

    eprintln!("device: {}", ctx.device_name());
    eprintln!("streaming read+write, achieved GB/s vs working-set size:\n");
    eprintln!("  {:>10}  {:>10}  {:>8}", "working set", "GB/s", "% of 256");
    let mut prev = f64::NAN;
    for mib in [
        1usize, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
    ] {
        let bytes = mib << 20;
        if bytes > MAX {
            break;
        }
        let gbps = sweep_one(&ctx, &mut cache, &src, &weight, &dst, bytes);
        // A cliff is where bandwidth drops materially from the previous size.
        let mark = if prev.is_finite() && gbps < prev * 0.80 {
            "  <- cliff"
        } else {
            ""
        };
        eprintln!(
            "  {mib:>7} MiB  {gbps:>10.1}  {:>7.0}%{mark}",
            100.0 * gbps / 256.0
        );
        prev = gbps;
    }
    eprintln!(
        "\nAnything sustaining >256 GB/s is served on-chip; the last such size is the\n\
         usable tier, and it bounds how much of a layer a batched step can reuse."
    );
}
