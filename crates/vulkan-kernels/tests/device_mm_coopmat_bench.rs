//! COOPMAT `mul_mm` warptile sweep — why the matrix cores lose to `mul_mmq`.
//!
//! Routing prefill through `mul_mm` COOPMAT made it SLOWER, not faster: 0.78x
//! `mul_mmq` end-to-end, and the per-op GPU profile blames the GEMM itself
//! (10.2 s vs 7.3 s over the same 400 dispatches on a 192-token chunk). Two
//! rounds of reading llama.cpp's warptile derivation produced two plausible
//! diagnoses and one measured non-effect (pinning `WARP` to the device's native
//! 64 instead of 32 changed nothing), so this bench stops reasoning about it.
//!
//! What it varies is the only thing that can matter here — the warp tile. In
//! the COOPMAT body each subgroup holds
//!
//! ```text
//! sums[(WM / TM) * (WN / TN)]   coopmat<f32, TM, TN, Accumulator>
//! ```
//!
//! live across the whole K loop, and each accumulator costs `TM * TN / WARP`
//! VGPRs per lane. llama.cpp's `l_warptile_mmq` is `WM = subgroup * 2`,
//! `WN = 64`, which on a 16x16x16 device is `8 * 4 = 32` accumulators — 128
//! VGPRs of the 256 an RDNA3 lane has, before operands. Tiles with the same
//! `BM x BN` footprint but a smaller per-warp share spread that over more
//! subgroups. That is a register-pressure/occupancy question, and the only
//! honest way to answer it is to run all of them.
//!
//! Reports GFLOP/s per (tile, n) against `mul_mmq` on the same weight buffer.
//! Opt-in — it allocates ~200 MB and runs thousands of dispatches:
//!
//! ```text
//! ARLE_MM_CM_BENCH=1 cargo test -p vulkan-kernels --features vulkan \
//!     --test device_mm_coopmat_bench --release -- --nocapture
//! ```
#![cfg(feature = "vulkan")]

use std::time::Instant;

use vulkan_kernels::{
    BLOCK_Q4_K_BYTES, Dispatch, Kernel, KernelCache, MmSpec, MmqSpec, mm_dispatch, mmq_dispatch,
    mmq_params, q8_1_quantize, q8_1_quantize_dispatch, q8_1_quantize_params, record_dispatch,
};
use vulkan_sys::{CommandRecorder, DescriptorSet, DeviceBuffer, VulkanContext};

/// Dispatches per measurement, barrier-separated inside one submit — so this is
/// per-dispatch LATENCY, directly comparable to the in-model GPU timestamps.
const ITERS: usize = 8;

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
    fn next_byte(&mut self) -> u8 {
        (self.next_u32() & 0xFF) as u8
    }
    fn next_unit_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    let mut h = sign | ((exp as u16) << 10) | (mant >> 13) as u16;
    if (mant & 0x1000) != 0 && ((mant & 0x0fff) != 0 || (h & 1) != 0) {
        h += 1;
    }
    h
}

fn build_q4_k_weights(rng: &mut Rng, m: usize, k: usize) -> Vec<u8> {
    let blocks = m * (k / 256);
    let mut bytes = Vec::with_capacity(blocks * BLOCK_Q4_K_BYTES);
    for _ in 0..blocks {
        let d = 0.02 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.04;
        let dmin = 0.01 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.02;
        bytes.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        bytes.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
        for _ in 0..140 {
            bytes.push(rng.next_byte());
        }
    }
    bytes
}

/// The COOPMAT `B` operand: `n` rows of `k` plain f16, row-major.
fn f16_rows(rng: &mut Rng, n: usize, k: usize) -> Vec<u8> {
    (0..n * k)
        .flat_map(|_| f32_to_f16(rng.next_unit_f32()).to_le_bytes())
        .collect()
}

/// The `mul_mmq` `B` operand over the same values: `block_q8_1_x4`, quantized
/// on device so the layout is exactly what the runtime feeds it.
fn q8_1_rows(ctx: &VulkanContext, rng: &mut Rng, n: usize, k: usize) -> Vec<u8> {
    let ne = n * k;
    let input: Vec<u8> = (0..ne)
        .flat_map(|_| rng.next_unit_f32().to_le_bytes())
        .collect();
    let out_len = ne.div_ceil(128) * 4 * vulkan_kernels::BLOCK_Q8_1_BYTES;

    let mut buf_in = DeviceBuffer::alloc(ctx, input.len()).expect("alloc q8_1 input");
    buf_in.copy_from_host(&input).expect("upload q8_1 input");
    let mut buf_out = DeviceBuffer::alloc(ctx, out_len).expect("alloc q8_1 output");
    buf_out
        .copy_from_host(&vec![0u8; out_len])
        .expect("zero q8_1 output");
    q8_1_quantize(
        ctx,
        &[&buf_in, &buf_out],
        q8_1_quantize_dispatch(ne as u32),
        &q8_1_quantize_params(ne as u32),
    )
    .expect("q8_1_quantize");
    let mut got = vec![0u8; out_len];
    buf_out.copy_to_host(&mut got).expect("read back q8_1");
    got
}

#[allow(clippy::too_many_arguments)]
fn time_dispatch<'a>(
    ctx: &'a VulkanContext,
    cache: &mut KernelCache<'a>,
    kernel: Kernel,
    buffers: &[&DeviceBuffer<'_>],
    dispatch: Dispatch,
    push: &[u8],
    spec: &[(u32, u32)],
) -> f64 {
    let (pipeline, layout) = cache
        .get(ctx, kernel, spec, push.len() as u32, buffers.len())
        .expect("build pipeline");
    let set = DescriptorSet::storage_buffers(ctx, layout, buffers).expect("bind descriptor set");

    let run = || {
        let mut recorder = CommandRecorder::new(ctx).expect("recorder");
        recorder.begin().expect("begin");
        for _ in 0..ITERS {
            record_dispatch(
                &mut recorder,
                pipeline,
                &set,
                push,
                [dispatch.x, dispatch.y, dispatch.z],
            );
            recorder.barrier();
        }
        let t0 = Instant::now();
        recorder.submit_and_wait().expect("submit");
        t0.elapsed().as_secs_f64() / ITERS as f64
    };
    run();
    run()
}

fn tflops(m: usize, n: usize, k: usize, secs: f64) -> f64 {
    2.0 * (m * n * k) as f64 / secs / 1e12
}

/// Chunk widths swept. 256 is the runtime's configured prefill chunk; the
/// smaller ones are the tail chunk every real prompt ends on.
const WIDTHS: [usize; 5] = [32, 64, 128, 192, 256];

/// Candidate warptiles as `(label, BM, BN, WM, WN, WARP)`; `BLOCK_SIZE` is
/// derived by [`MmSpec`] as `(BM/WM) * (BN/WN) * WARP`, so every subgroup owns
/// exactly one warp tile.
///
/// The first row is llama.cpp's `l_warptile_mmq` at this device's 64-wide
/// subgroup, kept as the control: it is what [`MmSpec::choose`] used to pick
/// for every prefill chunk, and this sweep is the measurement that retired it
/// (0.57x geomean vs `mul_mmq`). The rows marked `<-` are what `choose` picks
/// now. The rest hold the `BM x BN` footprint roughly constant while shrinking
/// `WM x WN` — the per-warp accumulator count, and therefore the register
/// pressure — by spreading the tile over more subgroups.
const CANDIDATES: &[(&str, u32, u32, u32, u32, u32)] = &[
    //  BM   BN   WM   WN  WARP
    ("l 128x128 w128x64", 128, 128, 128, 64, 64),
    (" 128x64  w32x32", 128, 64, 32, 32, 64), // <- MmSpec::wide
    ("  64x64  w32x32", 64, 64, 32, 32, 64),  // <- MmSpec::medium
    (" 128x32  w32x32", 128, 32, 32, 32, 64), // <- MmSpec::narrow
    ("  32x32  w32x32", 32, 32, 32, 32, 64),  // <- MmSpec::tiny
    ("  64x32  w32x32", 64, 32, 32, 32, 64),
    ("  64x128 w32x32", 64, 128, 32, 32, 64),
    (" 128x128 w32x32", 128, 128, 32, 32, 64),
    (" 128x128 w64x32", 128, 128, 64, 32, 64),
    (" 128x64  w64x32", 128, 64, 64, 32, 64),
    ("  64x64  w64x32", 64, 64, 64, 32, 64),
];

#[test]
fn coopmat_mm_warptile_sweep_on_device() {
    if std::env::var("ARLE_MM_CM_BENCH").is_err() {
        eprintln!("set ARLE_MM_CM_BENCH=1 to run the COOPMAT warptile sweep; skipping");
        return;
    }
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping COOPMAT warptile sweep");
            return;
        }
    };
    let Some(shape) = ctx.coopmat() else {
        eprintln!("device advertises no f16 cooperative-matrix shape; skipping");
        return;
    };
    let max_shared = ctx.max_compute_shared_memory_size();
    let (sg, sg_min, sg_max) = ctx.subgroup_size();
    eprintln!(
        "COOPMAT mul_mm warptile sweep on: {} (coopmat {}x{}x{}, subgroup {sg} in {sg_min}..{sg_max}, \
         maxComputeSharedMemorySize={max_shared})",
        ctx.device_name(),
        shape.m,
        shape.n,
        shape.k,
    );

    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut cache = KernelCache::new();
    // `speedups[c][w]` = this candidate's `vs mmq` at width `WIDTHS[w]`,
    // accumulated across shapes so the summary can rank by geometric mean. A
    // tile is only a policy if it wins on more than one aspect ratio, and the
    // per-cell numbers carry enough throttle noise that eyeballing one table
    // picked the wrong tile twice already.
    let mut speedups: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); WIDTHS.len()]; CANDIDATES.len()];

    // The 27B's three distinct projection aspect ratios: a tall-thin FFN
    // gate/up, its short-wide transpose, and the small KV projection. A tile
    // that only wins on one of them is not a policy.
    for &(shape_label, m, k) in &[
        ("ffn_gate/up", 17408usize, 5120usize),
        ("ffn_down", 5120, 17408),
        ("attn_kv", 1024, 5120),
    ] {
        let weights = build_q4_k_weights(&mut rng, m, k);
        let mut buf_a = DeviceBuffer::alloc(&ctx, weights.len()).expect("alloc weights");
        buf_a.copy_from_host(&weights).expect("upload weights");
        eprintln!(
            "\n######## {shape_label}: m={m} k={k} ({:.1} MiB Q4_K) ########",
            weights.len() as f64 / (1024.0 * 1024.0)
        );

        for (wi, &n) in WIDTHS.iter().enumerate() {
            let b_f16 = f16_rows(&mut rng, n, k);
            let mut buf_b = DeviceBuffer::alloc(&ctx, b_f16.len()).expect("alloc f16 B");
            buf_b.copy_from_host(&b_f16).expect("upload f16 B");
            let b_q8 = q8_1_rows(&ctx, &mut rng, n, k);
            let mut buf_bq = DeviceBuffer::alloc(&ctx, b_q8.len()).expect("alloc q8_1 B");
            buf_bq.copy_from_host(&b_q8).expect("upload q8_1 B");
            let out_len = m * n * 4;
            let mut buf_d = DeviceBuffer::alloc(&ctx, out_len).expect("alloc dst");
            buf_d.copy_from_host(&vec![0u8; out_len]).expect("zero dst");

            let push = mmq_params(m as u32, n as u32, k as u32).to_le_bytes();

            // Baseline first, so every coopmat row has something to be judged by.
            let mmq_spec = MmqSpec::choose(Kernel::MmqQ4K, m as u32, n as u32, max_shared)
                .expect("an mmq tile fits shared memory");
            let mmq_secs = time_dispatch(
                &ctx,
                &mut cache,
                Kernel::MmqQ4K,
                &[&buf_a, &buf_bq, &buf_d],
                mmq_dispatch(m as u32, n as u32, &mmq_spec),
                &push,
                mmq_spec.specialization_u32(),
            );

            eprintln!(
                "\n== n={n} ==\n{:>20} {:>6} {:>8} {:>9} {:>10} {:>8}",
                "tile", "warps", "shmem", "ms", "TFLOP/s", "vs mmq"
            );
            eprintln!(
                "{:>20} {:>6} {:>8} {:>9.2} {:>10.2} {:>8}",
                format!("mul_mmq {}x{}", mmq_spec.bm(), mmq_spec.bn()),
                "-",
                "-",
                mmq_secs * 1e3,
                tflops(m, n, k, mmq_secs),
                "1.00x",
            );

            for (ci, &(label, bm, bn, wm, wn, warp)) in CANDIDATES.iter().enumerate() {
                if warp < sg_min || warp > sg_max {
                    eprintln!(
                        "{label:>20} {:>6} {:>8} {:>9}",
                        "-", "-", "warp unsupported"
                    );
                    continue;
                }
                let Some(spec) = MmSpec::tile(bm, bn, wm, wn, warp, shape, max_shared) else {
                    eprintln!(
                        "{label:>20} {:>6} {:>8} {:>9}",
                        "-", "-", "invalid/over shmem"
                    );
                    continue;
                };
                let secs = time_dispatch(
                    &ctx,
                    &mut cache,
                    Kernel::MmCmQ4K,
                    &[&buf_a, &buf_b, &buf_d],
                    mm_dispatch(m as u32, n as u32, &spec),
                    &push,
                    spec.specialization_u32(),
                );
                speedups[ci][wi].push(mmq_secs / secs);
                eprintln!(
                    "{label:>20} {:>6} {:>8} {:>9.2} {:>10.2} {:>7.2}x",
                    (bm / wm) * (bn / wn),
                    spec.shared_bytes(),
                    secs * 1e3,
                    tflops(m, n, k, secs),
                    mmq_secs / secs,
                );
            }
        }
    }

    // The ranking `MmSpec::choose` should encode: for each width, the tile with
    // the best geometric mean over the shapes. Geometric, not arithmetic — these
    // are ratios, and one 5x outlier on the small `attn_kv` shape should not
    // outvote three shapes' worth of 0.8x on the projections that dominate.
    eprintln!("\n######## geomean speedup vs mul_mmq, over all shapes ########");
    eprint!("{:>20}", "tile");
    for n in WIDTHS {
        eprint!("{:>9}", format!("n={n}"));
    }
    eprintln!("{:>9}", "all n");
    for (ci, &(label, ..)) in CANDIDATES.iter().enumerate() {
        eprint!("{label:>20}");
        let mut all = Vec::new();
        for cells in &speedups[ci] {
            if cells.is_empty() {
                eprint!("{:>9}", "-");
                continue;
            }
            eprint!("{:>8.2}x", geomean(cells));
            all.extend_from_slice(cells);
        }
        if all.is_empty() {
            eprintln!("{:>9}", "-");
        } else {
            eprintln!("{:>8.2}x", geomean(&all));
        }
    }
    for (wi, n) in WIDTHS.iter().enumerate() {
        let best = (0..CANDIDATES.len())
            .filter(|&ci| !speedups[ci][wi].is_empty())
            .max_by(|&a, &b| {
                geomean(&speedups[a][wi])
                    .partial_cmp(&geomean(&speedups[b][wi]))
                    .expect("no NaN speedups")
            });
        if let Some(ci) = best {
            eprintln!(
                "  best at n={n}: {} ({:.2}x)",
                CANDIDATES[ci].0.trim(),
                geomean(&speedups[ci][wi])
            );
        }
    }
}

fn geomean(values: &[f64]) -> f64 {
    let sum_ln: f64 = values.iter().map(|v| v.ln()).sum();
    (sum_ln / values.len() as f64).exp()
}
