//! At what batch width does the dense tier leave the GEMV and board the
//! matrix cores?
//!
//! Batched verify (MTP / DFlash speculative decode) turns the per-token dense
//! GEMV into a skinny GEMM of k draft tokens. Three ways to run that exist in
//! this crate, and which one wins at which k is a property of THIS part's
//! WMMA-vs-bandwidth balance, so it is measured, not argued:
//!
//! - **loop**: k independent [`Kernel::GemvF16`] dispatches — what a naive
//!   verify would do; weights are re-read k times.
//! - **cols**: one `GemvF16` pipeline at `NUM_COLS = k`
//!   ([`GemvDenseSpec::with_cols`]) — the vendored shader's own batch axis;
//!   weights are read once, `temp[k]` accumulators per thread.
//! - **coopmat**: [`Kernel::MmCmF16`], the f16 weights on the matrix cores.
//!
//! The MoE expert tier deliberately has NO row here: at 512 experts top-10,
//! k=16 draft tokens still put only ~1.16 rows on each activated expert
//! (512·(1−(1−10/512)^k) distinct experts over 10k rows), so the expert path
//! stays GEMV-shaped at any speculative width this model will see. The
//! crossover question is exclusively about the DENSE tier.
//!
//! Correctness runs by default (small shapes, f64 oracle, all three arms).
//! The sweep allocates ~1.3 GiB and runs seconds:
//!
//! ```text
//! ARLE_BATCH_CROSSOVER=1 cargo test -p vulkan-kernels --features vulkan \
//!     --test device_batched_crossover --release -- --nocapture --test-threads=1
//! ```
#![cfg(feature = "vulkan")]

use std::time::Instant;

use vulkan_kernels::{
    GemvDenseSpec, Kernel, KernelCache, MmSpec, gemv_dense_dispatch, gemv_params_f32_b,
    gemv_params_f32_b_cols, mm_dispatch, mmq_params, record_dispatch,
};
use vulkan_sys::{CommandRecorder, DescriptorSet, DeviceBuffer, VulkanContext};

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
    /// Uniform in [-0.5, 0.5): keeps k-long f16 dots O(1) so no arm hides an
    /// error behind saturation.
    fn unit(&mut self) -> f32 {
        (self.next_u32() as f64 / u32::MAX as f64 - 0.5) as f32
    }
}

/// Round-to-nearest-even f32 -> f16, same routine the sibling coopmat bench
/// carries (no `half` dependency in this crate).
fn f32_to_f16_bits(value: f32) -> u16 {
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

fn f16_bits_to_f32(b: u16) -> f32 {
    let sign = u32::from(b & 0x8000) << 16;
    let exp = i32::from((b >> 10) & 0x1f);
    let mant = u32::from(b & 0x3ff);
    let bits = match (exp, mant) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: renormalize into f32.
            let shift = m.leading_zeros() - 21;
            sign | ((127 - 15 - shift as i32 + 1) as u32) << 23 | ((m << (shift + 1)) & 0x3ff) << 13
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7f80_0000 | m << 13,
        (e, m) => sign | ((e - 15 + 127) as u32) << 23 | m << 13,
    };
    f32::from_bits(bits)
}

/// Weights as f16 bytes, row-major `[m][kk]`, plus the f32 view the oracle
/// dots in f64.
fn make_weights(rng: &mut Rng, m: usize, kk: usize) -> (Vec<u8>, Vec<f32>) {
    let vals: Vec<f32> = (0..m * kk)
        .map(|_| f16_bits_to_f32(f32_to_f16_bits(rng.unit())))
        .collect();
    let bytes = vals
        .iter()
        .flat_map(|&v| f32_to_f16_bits(v).to_le_bytes())
        .collect();
    (bytes, vals)
}

fn upload<'a>(ctx: &'a VulkanContext, bytes: &[u8]) -> DeviceBuffer<'a> {
    let mut b = DeviceBuffer::alloc(ctx, bytes.len().max(4)).expect("alloc");
    if !bytes.is_empty() {
        b.copy_from_host(bytes).expect("upload");
    }
    b
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn read_f32(buf: &DeviceBuffer<'_>, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    buf.copy_to_host(&mut bytes).expect("read");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// max |got-want| / max(|want|, rms(want)) — the same floor rationale as the
/// NVFP4 suite: a bounded absolute error over a near-zero element is not a
/// wrong row.
fn max_rel(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let rms = (want
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        / want.len().max(1) as f64)
        .sqrt() as f32;
    let floor = if rms > 0.0 { rms } else { 1e-6 };
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (g - w).abs() / w.abs().max(floor))
        .fold(0.0f32, f32::max)
}

/// One arm's result: the `[k][m]` output and the seconds per step (meaningful
/// only when `passes > 0`; 0 passes = correctness only).
type ArmResult = (Vec<f32>, f64);

/// The three arms over one `[m][kk]` f16 weight and `k` activation columns.
/// Returns (loop, cols, coopmat) outputs, each laid out `[k][m]`.
#[allow(clippy::too_many_arguments)]
fn run_arms<'a>(
    ctx: &'a VulkanContext,
    cache: &mut KernelCache<'a>,
    w_f16: &DeviceBuffer<'a>,
    m: usize,
    kk: usize,
    x_cols: &[Vec<f32>],
    passes: usize,
) -> (ArmResult, ArmResult, Option<ArmResult>) {
    let k = x_cols.len();
    let flat_x: Vec<f32> = x_cols.iter().flatten().copied().collect();
    let buf_x = upload(ctx, &f32_bytes(&flat_x));
    let buf_d = DeviceBuffer::alloc(ctx, k * m * 4).expect("alloc d");
    let dummy = upload(ctx, &[0u8; 4]);

    let time = |rec_fn: &dyn Fn(&mut CommandRecorder<'a>)| -> f64 {
        let run = || {
            let mut rec = CommandRecorder::new(ctx).expect("recorder");
            rec.begin().expect("begin");
            for _ in 0..passes.max(1) {
                rec_fn(&mut rec);
                rec.barrier();
            }
            let t0 = Instant::now();
            rec.submit_and_wait().expect("submit");
            t0.elapsed().as_secs_f64()
        };
        run(); // warm: pipeline + page-in
        run() / passes.max(1) as f64
    };

    // ── loop: k independent GemvF16 dispatches, weights re-read k times ────
    let spec1 = GemvDenseSpec::DEFAULT;
    let push1 = gemv_params_f32_b(kk as u32, m as u32).to_le_bytes();
    let (pipe1, layout1) = cache
        .get(
            ctx,
            Kernel::GemvF16,
            spec1.specialization_u32(),
            push1.len() as u32,
            5,
        )
        .expect("gemv pipeline");
    let sets1: Vec<_> = (0..k)
        .map(|j| {
            DescriptorSet::storage_buffers_ranged(
                ctx,
                layout1,
                &[
                    (w_f16, 0, (m * kk * 2) as u64),
                    (&buf_x, (j * kk * 4) as u64, (kk * 4) as u64),
                    (&buf_d, (j * m * 4) as u64, (m * 4) as u64),
                    (&dummy, 0, 4),
                    (&dummy, 0, 4),
                ],
            )
            .expect("bind loop")
        })
        .collect();
    let d1 = gemv_dense_dispatch(m as u32, &spec1);
    let secs_loop = time(&|rec| {
        for set in &sets1 {
            record_dispatch(rec, pipe1, set, &push1, [d1.x, d1.y, d1.z]);
        }
    });
    let out_loop = read_f32(&buf_d, k * m);

    // ── cols: one pipeline at NUM_COLS = k, weights read once ──────────────
    let spec2 = GemvDenseSpec::with_cols(128, 1, k as u32);
    let push2 = gemv_params_f32_b_cols(kk as u32, m as u32).to_le_bytes();
    let (pipe2, layout2) = cache
        .get(
            ctx,
            Kernel::GemvF16,
            spec2.specialization_u32(),
            push2.len() as u32,
            5,
        )
        .expect("gemv cols pipeline");
    let set2 = DescriptorSet::storage_buffers_ranged(
        ctx,
        layout2,
        &[
            (w_f16, 0, (m * kk * 2) as u64),
            (&buf_x, 0, (k * kk * 4) as u64),
            (&buf_d, 0, (k * m * 4) as u64),
            (&dummy, 0, 4),
            (&dummy, 0, 4),
        ],
    )
    .expect("bind cols");
    let d2 = gemv_dense_dispatch(m as u32, &spec2);
    let secs_cols = time(&|rec| {
        record_dispatch(rec, pipe2, &set2, &push2, [d2.x, d2.y, d2.z]);
    });
    let out_cols = read_f32(&buf_d, k * m);

    // ── coopmat: MmCmF16, B is f16 row-major [k][kk] ───────────────────────
    let coopmat = ctx.coopmat().map(|shape| {
        let spec3 = MmSpec::choose(
            shape,
            ctx.subgroup_size().0,
            k as u32,
            ctx.max_compute_shared_memory_size(),
        )
        .expect("no warptile fits shared memory");
        let b_f16: Vec<u8> = flat_x
            .iter()
            .flat_map(|&v| f32_to_f16_bits(v).to_le_bytes())
            .collect();
        let buf_b16 = upload(ctx, &b_f16);
        let push3 = mmq_params(m as u32, k as u32, kk as u32).to_le_bytes();
        let (pipe3, layout3) = cache
            .get(
                ctx,
                Kernel::MmCmF16,
                spec3.specialization_u32(),
                push3.len() as u32,
                3,
            )
            .expect("coopmat f16 pipeline");
        let set3 = DescriptorSet::storage_buffers_ranged(
            ctx,
            layout3,
            &[
                (w_f16, 0, (m * kk * 2) as u64),
                (&buf_b16, 0, (k * kk * 2) as u64),
                (&buf_d, 0, (k * m * 4) as u64),
            ],
        )
        .expect("bind coopmat");
        let d3 = mm_dispatch(m as u32, k as u32, &spec3);
        let secs = time(&|rec| {
            record_dispatch(rec, pipe3, &set3, &push3, [d3.x, d3.y, d3.z]);
        });
        (read_f32(&buf_d, k * m), secs)
    });

    ((out_loop, secs_loop), (out_cols, secs_cols), coopmat)
}

/// All three arms against an f64 oracle, at every k the sweep uses. This is
/// the gate that lets the sweep below be a pure benchmark: a wrong arm loses
/// here, not silently in a throughput table.
#[test]
fn all_three_arms_match_the_f64_oracle() {
    let ctx = match VulkanContext::create() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Vulkan device ({e}); skipping");
            return;
        }
    };
    let mut cache = KernelCache::new();
    let (m, kk) = (256usize, 512usize);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let (w_bytes, w_vals) = make_weights(&mut rng, m, kk);
    let buf_w = upload(&ctx, &w_bytes);

    for k in [1usize, 2, 4, 8, 16] {
        let x_cols: Vec<Vec<f32>> = (0..k)
            .map(|_| (0..kk).map(|_| rng.unit()).collect())
            .collect();
        // The coopmat arm multiplies f16-ROUNDED activations; give every arm's
        // oracle its own operand so the comparison isolates the GEMM, not the
        // (separately correct) rounding step.
        let want_f32: Vec<f32> = oracle(&w_vals, m, kk, &x_cols, false);
        let want_f16: Vec<f32> = oracle(&w_vals, m, kk, &x_cols, true);

        let ((out_loop, _), (out_cols, _), coop) =
            run_arms(&ctx, &mut cache, &buf_w, m, kk, &x_cols, 0);
        let e_loop = max_rel(&out_loop, &want_f32);
        let e_cols = max_rel(&out_cols, &want_f32);
        assert!(e_loop < 1e-4, "k={k}: loop arm rel {e_loop:.3e}");
        assert!(e_cols < 1e-4, "k={k}: cols arm rel {e_cols:.3e}");
        // loop and cols compute the SAME arithmetic through different
        // pipelines; requiring them to agree with each other (tightly) catches
        // a batch-stride slip that both oracles would tolerate.
        let e_cross = max_rel(&out_cols, &out_loop);
        assert!(e_cross < 1e-5, "k={k}: cols vs loop rel {e_cross:.3e}");
        if let Some((out_cm, _)) = coop {
            let e_cm = max_rel(&out_cm, &want_f16);
            assert!(e_cm < 2e-3, "k={k}: coopmat arm rel {e_cm:.3e}");
        }
        eprintln!("k={k:>2}: loop {e_loop:.2e}  cols {e_cols:.2e}  coopmat ok");
    }
}

fn oracle(w: &[f32], m: usize, kk: usize, x_cols: &[Vec<f32>], round_x_f16: bool) -> Vec<f32> {
    let mut out = vec![0f32; x_cols.len() * m];
    for (j, x) in x_cols.iter().enumerate() {
        for r in 0..m {
            let row = &w[r * kk..(r + 1) * kk];
            out[j * m + r] = row
                .iter()
                .zip(x)
                .map(|(&a, &b)| {
                    let b = if round_x_f16 {
                        f16_bits_to_f32(f32_to_f16_bits(b))
                    } else {
                        b
                    };
                    f64::from(a) * f64::from(b)
                })
                .sum::<f64>() as f32;
        }
    }
    out
}

#[test]
fn crossover_sweep() {
    if std::env::var("ARLE_BATCH_CROSSOVER").is_err() {
        eprintln!("set ARLE_BATCH_CROSSOVER=1 to run the batched-dense crossover sweep; skipping");
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
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    eprintln!("device: {}", ctx.device_name());
    eprintln!(
        "per-STEP time (all k columns), ms; weights f16, B f32 (f16 for coopmat).\n\
         'w GB/s eff' = weight bytes / step — the byte-amortization view.\n"
    );

    // The model's three dense aspect ratios: the fused q projection, its
    // transpose (o_proj), and the one giant row space (lm_head).
    for &(label, m, kk, passes) in &[
        ("q_proj  [6144x2560]", 6144usize, 2560usize, 24usize),
        ("o_proj  [2560x6144]", 2560, 6144, 24),
        ("lm_head [248320x2560]", 248_320, 2560, 4),
    ] {
        let (w_bytes, _) = make_weights(&mut rng, m, kk);
        let buf_w = upload(&ctx, &w_bytes);
        let w_gib = w_bytes.len() as f64 / 1e9;
        eprintln!("── {label}  ({:.1} MB f16) ──", w_bytes.len() as f64 / 1e6);
        eprintln!(
            "  {:>3}  {:>10} {:>10} {:>10}   {:>9}  winner",
            "k", "loop ms", "cols ms", "coopmat ms", "w GB/s eff"
        );
        for k in [1usize, 2, 4, 8, 16] {
            let x_cols: Vec<Vec<f32>> = (0..k)
                .map(|_| (0..kk).map(|_| rng.unit()).collect())
                .collect();
            let ((_, s_loop), (_, s_cols), coop) =
                run_arms(&ctx, &mut cache, &buf_w, m, kk, &x_cols, passes);
            let s_cm = coop.as_ref().map(|(_, s)| *s);
            let best = [
                ("loop", s_loop),
                ("cols", s_cols),
                ("coopmat", s_cm.unwrap_or(f64::INFINITY)),
            ]
            .into_iter()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
            eprintln!(
                "  {k:>3}  {:>10.3} {:>10.3} {:>10}   {:>9.0}  {}",
                s_loop * 1e3,
                s_cols * 1e3,
                s_cm.map_or("      n/a".into(), |s| format!("{:.3}", s * 1e3)),
                w_gib / best.1,
                best.0,
            );
        }
    }
    eprintln!(
        "\nThe k where 'coopmat' first wins is the verify width at which the\n\
         matrix cores pay on this part; below it, 'cols' amortizes weight reads\n\
         without them. Feed the winner into the S8 batched-verify design."
    );
}
