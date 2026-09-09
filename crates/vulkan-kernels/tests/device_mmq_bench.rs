//! `mul_mmq` throughput isolation bench — why batched prefill is 9x off llama.cpp.
//!
//! The whole-model GPU profile says `mmq` is 75% of prefill time and, damningly,
//! that its cost barely moves between a 22-token and a 64-token chunk. A GEMM
//! whose runtime is flat in `n` is not doing GEMM work — it is streaming the
//! weights and stalling. This bench isolates that claim from the model:
//!
//!   1. For each real projection shape, run `mul_mmq` and the PROVEN GEMV over
//!      the SAME weight buffer, and report achieved GB/s for both. Same bytes,
//!      same allocation, same device — so any gap is the kernel, not memory
//!      placement (which would slow both equally).
//!   2. Sweep `n` on one shape. Flat time => weight-streaming bound; linear
//!      time => genuinely compute bound.
//!   3. Sweep the small/medium/large tiles, since `MmqSpec::choose` never picks
//!      large at `n = 64` and that choice has never been measured.
//!
//! Opt-in — it allocates ~50 MB weights and runs hundreds of dispatches, so it
//! is not part of the correctness gate:
//!
//! ```text
//! ARLE_MMQ_BENCH=1 cargo test -p vulkan-kernels --features vulkan \
//!     --test device_mmq_bench --release -- --nocapture
//! ```
#![cfg(feature = "vulkan")]

use std::time::Instant;

use vulkan_kernels::{
    BLOCK_Q4_K_BYTES, Dispatch, Kernel, KernelCache, MmqSpec, mmq_dispatch, mmq_params,
    q8_1_quantize, q8_1_quantize_dispatch, q8_1_quantize_params, record_dispatch,
};
use vulkan_sys::{CommandRecorder, DescriptorSet, DeviceBuffer, VulkanContext};

/// Dispatches recorded per measurement, all into ONE command buffer behind one
/// submit.
///
/// The convenience launchers (`mmq_with_params_and_spec` and friends) build a
/// fresh `KernelCache` per call, so timing them times SHADER COMPILATION — for
/// `mul_mmq` that is milliseconds and it swamps the dispatch, which is exactly
/// the trap that made the first version of this bench report a bogus ~2 ms floor
/// on every shape. Everything here shares one cache and one submit.
const ITERS: usize = 16;

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

/// `m * k / 256` valid `block_q4_K`s (144 B each), the resident layout of every
/// projection this bench measures.
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

/// `n` activation rows of `k` f32 each, quantized to `block_q8_1_x4` on device.
fn quantize_rows_x4(ctx: &VulkanContext, rng: &mut Rng, n: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 128, 0, "mul_mmq B rows must be 128-value aligned");
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

/// Seconds per dispatch: record [`ITERS`] copies of one dispatch, separated by
/// barriers so they cannot overlap, behind a single submit. Barriers make this a
/// per-dispatch LATENCY measure, which is what the in-model GPU timestamps
/// reported and therefore what is comparable to them.
///
/// Called twice — the first call warms the pipeline and any lazy allocation, and
/// only the second is returned.
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

fn time_mmq<'a>(
    ctx: &'a VulkanContext,
    cache: &mut KernelCache<'a>,
    buffers: &[&DeviceBuffer<'_>],
    m: usize,
    n: usize,
    k: usize,
    spec: &MmqSpec,
) -> f64 {
    time_dispatch(
        ctx,
        cache,
        Kernel::MmqQ4K,
        buffers,
        mmq_dispatch(m as u32, n as u32, spec),
        &mmq_params(m as u32, n as u32, k as u32).to_le_bytes(),
        spec.specialization_u32(),
    )
}

/// Same weight buffer through the proven single-token GEMV — the reference for
/// what this device CAN stream a Q4_K matrix at.
fn time_gemv<'a>(
    ctx: &'a VulkanContext,
    cache: &mut KernelCache<'a>,
    buf_a: &DeviceBuffer<'_>,
    buf_b: &DeviceBuffer<'_>,
    buf_d: &DeviceBuffer<'_>,
    fuse0: &DeviceBuffer<'_>,
    fuse1: &DeviceBuffer<'_>,
    m: usize,
    k: usize,
) -> f64 {
    time_dispatch(
        ctx,
        cache,
        Kernel::GemvQ4K,
        &[buf_a, buf_b, buf_d, fuse0, fuse1],
        vulkan_kernels::gemv_dispatch(m as u32),
        &vulkan_kernels::gemv_params(k as u32, m as u32).to_le_bytes(),
        vulkan_kernels::Kernel::GemvQ4K.specialization_u32(),
    )
}

fn gbps(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / secs / 1e9
}

/// Every distinct Q4_K projection shape in the Qwen3.8-27B layer stack, plus the
/// tile the runtime actually picks and the two it does not.
#[test]
fn mmq_throughput_vs_gemv_on_device() {
    if std::env::var("ARLE_MMQ_BENCH").is_err() {
        eprintln!("set ARLE_MMQ_BENCH=1 to run the mul_mmq throughput bench; skipping");
        return;
    }
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device available ({e}); skipping mul_mmq bench");
            return;
        }
    };
    let max_shared = ctx.max_compute_shared_memory_size();
    eprintln!(
        "mul_mmq bench on: {} (maxComputeSharedMemorySize={max_shared}, subgroup={:?})",
        ctx.device_name(),
        ctx.subgroup_size()
    );

    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut cache = KernelCache::new();

    // (label, m, k) for the 27B's projections. `k % 256 == 0` throughout.
    let shapes: &[(&str, usize, usize)] = &[
        ("ffn_gate/up", 17408, 5120),
        ("ffn_down", 5120, 17408),
        ("attn_q", 12288, 5120),
        ("attn_kv", 1024, 5120),
        ("attn_out", 5120, 6144),
    ];
    let n = 64usize;

    eprintln!(
        "\n== mul_mmq vs GEMV over the SAME weight buffer (n={n} tokens, Q4_K) ==\n\
         {:<12} {:>8} {:>8} {:>9} {:>9} {:>9} {:>9} {:>7}",
        "shape", "m", "MiB", "mmq ms", "mmq GB/s", "gemv ms", "gemvGB/s", "ratio"
    );
    for &(label, m, k) in shapes {
        let weights = build_q4_k_weights(&mut rng, m, k);
        let bytes = weights.len();
        let mut buf_a = DeviceBuffer::alloc(&ctx, bytes).expect("alloc weights");
        buf_a.copy_from_host(&weights).expect("upload weights");

        let activ = quantize_rows_x4(&ctx, &mut rng, n, k);
        let mut buf_b = DeviceBuffer::alloc(&ctx, activ.len()).expect("alloc activations");
        buf_b.copy_from_host(&activ).expect("upload activations");
        let out_len = m * n * 4;
        let mut buf_d = DeviceBuffer::alloc(&ctx, out_len).expect("alloc dst");
        buf_d.copy_from_host(&vec![0u8; out_len]).expect("zero dst");

        let spec = MmqSpec::choose(Kernel::MmqQ4K, m as u32, n as u32, max_shared)
            .expect("a tile fits shared memory");
        let mmq_s = time_mmq(&ctx, &mut cache, &[&buf_a, &buf_b, &buf_d], m, n, k, &spec);

        // GEMV reads one activation row and writes m floats; the weight traffic
        // is identical, which is the point of the comparison.
        let activ1 = quantize_rows_x4(&ctx, &mut rng, 1, k);
        let mut buf_b1 = DeviceBuffer::alloc(&ctx, activ1.len()).expect("alloc gemv activations");
        buf_b1.copy_from_host(&activ1).expect("upload");
        let mut buf_d1 = DeviceBuffer::alloc(&ctx, m * 4).expect("alloc gemv dst");
        buf_d1.copy_from_host(&vec![0u8; m * 4]).expect("zero");
        let mut fuse0 = DeviceBuffer::alloc(&ctx, 4).expect("alloc fuse0");
        fuse0.copy_from_host(&[0u8; 4]).expect("zero fuse0");
        let mut fuse1 = DeviceBuffer::alloc(&ctx, 4).expect("alloc fuse1");
        fuse1.copy_from_host(&[0u8; 4]).expect("zero fuse1");
        let gemv_s = time_gemv(
            &ctx, &mut cache, &buf_a, &buf_b1, &buf_d1, &fuse0, &fuse1, m, k,
        );

        eprintln!(
            "{label:<12} {m:>8} {:>8.1} {:>9.2} {:>9.1} {:>9.2} {:>9.1} {:>6.2}x",
            bytes as f64 / (1024.0 * 1024.0),
            mmq_s * 1e3,
            gbps(bytes, mmq_s),
            gemv_s * 1e3,
            gbps(bytes, gemv_s),
            gemv_s / mmq_s,
        );
    }

    // Does mmq cost scale with n at all? Flat => weight-streaming bound.
    let (m, k) = (17408usize, 5120usize);
    let weights = build_q4_k_weights(&mut rng, m, k);
    let bytes = weights.len();
    let mut buf_a = DeviceBuffer::alloc(&ctx, bytes).expect("alloc weights");
    buf_a.copy_from_host(&weights).expect("upload weights");
    eprintln!(
        "\n== mul_mmq scaling in n (ffn_gate shape m={m} k={k}) ==\n\
         {:>6} {:>8} {:>10} {:>10} {:>12}",
        "n", "tile", "ms", "GB/s", "TFLOP/s"
    );
    for &n in &[1usize, 8, 16, 32, 64, 128, 256] {
        let activ = quantize_rows_x4(&ctx, &mut rng, n, k);
        let mut buf_b = DeviceBuffer::alloc(&ctx, activ.len()).expect("alloc activations");
        buf_b.copy_from_host(&activ).expect("upload activations");
        let out_len = m * n * 4;
        let mut buf_d = DeviceBuffer::alloc(&ctx, out_len).expect("alloc dst");
        buf_d.copy_from_host(&vec![0u8; out_len]).expect("zero dst");
        let spec = MmqSpec::choose(Kernel::MmqQ4K, m as u32, n as u32, max_shared)
            .expect("a tile fits shared memory");
        let secs = time_mmq(&ctx, &mut cache, &[&buf_a, &buf_b, &buf_d], m, n, k, &spec);
        eprintln!(
            "{n:>6} {:>8} {:>10.2} {:>10.1} {:>12.2}",
            format!("{}x{}", spec.bm(), spec.bn()),
            secs * 1e3,
            gbps(bytes, secs),
            2.0 * (m * n * k) as f64 / secs / 1e12,
        );
    }

    // The tile `choose` never picks at n=64, measured rather than assumed.
    eprintln!(
        "\n== tile sweep at n=64 (ffn_gate shape) ==\n{:>10} {:>8} {:>10} {:>10}",
        "tile", "shmem", "ms", "GB/s"
    );
    let n = 64usize;
    let activ = quantize_rows_x4(&ctx, &mut rng, n, k);
    let mut buf_b = DeviceBuffer::alloc(&ctx, activ.len()).expect("alloc activations");
    buf_b.copy_from_host(&activ).expect("upload activations");
    let out_len = m * n * 4;
    let mut buf_d = DeviceBuffer::alloc(&ctx, out_len).expect("alloc dst");
    buf_d.copy_from_host(&vec![0u8; out_len]).expect("zero dst");
    for (name, spec) in [
        ("small", MmqSpec::k_quant_small()),
        ("medium", MmqSpec::k_quant_medium()),
        ("large", MmqSpec::k_quant_large()),
    ] {
        let shmem = spec.shared_bytes(Kernel::MmqQ4K).unwrap_or(u32::MAX);
        if shmem > max_shared {
            eprintln!("{name:>10} {shmem:>8} {:>10} {:>10}", "-", "over shmem");
            continue;
        }
        let secs = time_mmq(&ctx, &mut cache, &[&buf_a, &buf_b, &buf_d], m, n, k, &spec);
        eprintln!(
            "{name:>10} {shmem:>8} {:>10.2} {:>10.1}",
            secs * 1e3,
            gbps(bytes, secs)
        );
    }
}
