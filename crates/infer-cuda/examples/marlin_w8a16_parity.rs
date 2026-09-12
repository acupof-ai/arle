//! Marlin W8A16 numeric-parity gate.
//!
//! Standalone kernel harness — NO engine, NO serve. Builds a random BF16 weight,
//! quantizes per-group symmetric INT8 (mirrors `scripts/w8a16_quant.py`), then
//! compares three lanes over Qwen3.6-27B dense shapes × an M sweep:
//!   - f32 host reference (ground truth: dequant in f64, matmul)
//!   - Marlin tensor-core GEMM (`marlin_w8a16_gemm_cuda`, via the repacked layout)
//!   - the in-tree dequant→cuBLAS-BF16 fallback (`dequantize_w8a16_to_bf16` + gemm)
//!
//! Anchors on f32 (not bf16-vs-bf16). PASS = both device lanes land within a
//! small multiple of the shared bf16 quant floor of the f32 reference; a Marlin
//! error many× the fallback's error is the silent-wrong-repack / wrong-scale-perm
//! signal this gate exists to catch. Exits non-zero on any violation.
//!
//! `--negative-control` corrupts ONE expectation per comparator family (the
//! Marlin accepted-shape lane and the declined-shape fallback band); each
//! family MUST independently FAIL. The run then prints NEGATIVE CONTROL OK and
//! exits 0.
//!
//! Run on a pod: `INFER_CUDA_DEVICE=<free-gpu> target/release/examples/marlin_w8a16_parity`

fn main() -> anyhow::Result<()> {
    use parity_common::Parsed;
    match parity_common::cli() {
        Parsed::BuildIdPrinted => Ok(()),
        Parsed::Run(cli) => real::run(cli.negative),
    }
}

#[allow(dead_code)] // shared harness; each gate uses only the subset it needs
#[path = "support/parity_common.rs"]
mod parity_common;

#[cfg(not(feature = "cuda"))]
mod real {
    pub(super) fn run(_negative: bool) -> anyhow::Result<()> {
        eprintln!("marlin_w8a16_parity is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::{Result, ensure};
    use cuda_kernels::ffi;
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::tensor::DeviceMatrix;
    use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
    use half::bf16;

    use super::parity_common::Rng;

    const GROUP: usize = 128;
    // (label, N=weight rows / out dim, K=weight cols / contraction) — 27B dense.
    const SHAPES: &[(&str, usize, usize)] = &[
        ("ffn_gate_up", 17408, 5120),
        ("ffn_down", 5120, 17408),
        ("attn_sq", 5120, 5120),
    ];
    const M_SWEEP: &[usize] = &[1, 2, 4, 8, 16, 32];
    /// Three fixed seeds per shape; the gate is shapes × seeds × M.
    const SEEDS: &[u64] = &[
        0x9e37_79b9_7f4a_7c15,
        0x517c_c1cc_9e37_79b9,
        0x2545_f491_4f6c_dd1d,
    ];
    /// Shapes the layout planner must decline: N not %64 with K aligned.
    const DECLINED_SHAPES: &[(&str, usize, usize)] = &[("planner declines n%64", 96, 5120)];
    // Marlin lane must not exceed the fallback lane's f32-anchored error by more
    // than this factor — both share the INT8 group-quant floor, so a correct
    // Marlin lane tracks the fallback closely. A blown ratio = wrong repack/perm.
    const MARLIN_VS_FALLBACK_MAX_RATIO: f64 = 4.0;
    // Absolute ceiling on either lane's rel-L2. The ratio alone passes when
    // BOTH lanes are jointly wrong; this closes that hole. Weight-only floor:
    // per-element INT8 RN error is at most s_g/2 with s_g=amax_g/127, so the
    // worst-case rel-L2 contribution of one GROUP=128 group is
    // (amax_g/σ_w)·√(2G/π)/(2·127); for Gaussian weights with amax≈2.7σ that
    // is ≈0.095, plus the bf16 activation and output-store terms (<0.01). 0.20
    // is ~2x that analytic ceiling so it cannot go red on clean-run noise; it
    // is loose on purpose and tightens in Phase 2 against a measured margin.
    const MAX_REL_L2: f64 = 0.20;

    /// Per-group symmetric INT8 (scale = amax/127), matching w8a16_quant.py.
    /// Returns (int8 weight [n*k], bf16 scales [n * k/GROUP]).
    fn per_group_int8(w: &[f32], n: usize, k: usize) -> (Vec<i8>, Vec<bf16>) {
        let ng = k / GROUP;
        let mut q = vec![0i8; n * k];
        let mut s = vec![bf16::from_f32(0.0); n * ng];
        for row in 0..n {
            for g in 0..ng {
                let base = row * k + g * GROUP;
                let amax = (0..GROUP)
                    .map(|i| w[base + i].abs())
                    .fold(0.0f32, f32::max)
                    .max(1e-8);
                let scale = amax / 127.0;
                s[row * ng + g] = bf16::from_f32(scale);
                for i in 0..GROUP {
                    let v = (w[base + i] / scale).round().clamp(-127.0, 127.0);
                    q[base + i] = v as i8;
                }
            }
        }
        (q, s)
    }

    /// One fail bit per comparator family, so --negative-control can prove
    /// each family independently has teeth.
    #[derive(Clone, Copy)]
    struct Families {
        /// Marlin GEMM vs f64 reference on accepted (tile-aligned) shapes.
        marlin_lane: bool,
        /// dequant→cuBLAS fallback on the shape the repack must decline.
        declined_fallback: bool,
    }
    impl Families {
        fn entries(self) -> [(&'static str, bool); 2] {
            [
                ("marlin lane", self.marlin_lane),
                ("declined fallback", self.declined_fallback),
            ]
        }
    }

    pub(super) fn run(negative: bool) -> Result<()> {
        let ctx = DeviceContext::new()?;
        let cc = ctx.compute_capability();
        eprintln!(
            "[marlin-parity] device={} cc={}.{} build={}{}",
            ctx.ordinal(),
            cc.0,
            cc.1,
            cuda_kernels::KERNEL_BUILD_ID,
            if negative { " NEGATIVE-CONTROL" } else { "" }
        );
        ensure!(
            cc.0 >= 8,
            "Marlin W8A16 needs sm_80+; got sm_{}{}",
            cc.0,
            cc.1
        );

        let mut fams = Families {
            marlin_lane: false,
            declined_fallback: false,
        };
        for &seed in SEEDS {
            // The declined-shape band self-calibrates off the accepted shapes'
            // own fallback floor: same lane, same seed, no magic constant.
            let mut accepted_max_deq = 0f64;
            for &(label, n, k) in SHAPES {
                for &m in M_SWEEP {
                    let (marlin_fail, deq_err) = probe_one(&ctx, label, m, n, k, seed, negative)?;
                    if marlin_fail {
                        fams.marlin_lane = true;
                    }
                    accepted_max_deq = accepted_max_deq.max(deq_err);
                }
            }
            for &(label, n, k) in DECLINED_SHAPES {
                if probe_declined(&ctx, label, n, k, seed, accepted_max_deq, negative)? {
                    fams.declined_fallback = true;
                }
            }
        }

        let mut families = super::parity_common::Families::new();
        for (name, fired) in fams.entries() {
            families.record(name, fired);
        }
        families.finish("marlin-parity", negative)
    }

    /// FNV-mix the fixed seed with the shape identity.
    fn shape_seed(seed: u64, label: &str, n: usize, k: usize) -> u64 {
        let mut mix = seed;
        for b in label.as_bytes() {
            mix ^= *b as u64;
            mix = mix.wrapping_mul(0x0000_0100_0000_01b3);
        }
        mix ^= (n as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        mix ^= (k as u64).wrapping_mul(0x517c_c1cc_9e37_79b9);
        mix
    }

    /// f32 reference: dequant weight (bf16 scale, faithful to device) then
    /// matmul in f64. out[row_m, row_n] = sum_k x[row_m,k] * (q * scale).
    fn host_reference(q: &[i8], s: &[bf16], x_host: &[f32], n: usize, k: usize) -> Vec<f64> {
        let ng = k / GROUP;
        let m = x_host.len() / k;
        let mut ref_out = vec![0f64; m * n];
        for row_n in 0..n {
            for g in 0..ng {
                let scale = f32::from(s[row_n * ng + g]) as f64;
                for i in 0..GROUP {
                    let kk = g * GROUP + i;
                    let w = q[row_n * k + kk] as f64 * scale;
                    for row_m in 0..m {
                        ref_out[row_m * n + row_n] += x_host[row_m * k + kk] as f64 * w;
                    }
                }
            }
        }
        ref_out
    }

    /// Dequantize the INT8 weight to BF16 then cuBLAS GEMM: out[m,n] = x[m,k] · w[n,k]ᵀ.
    fn launch_dequant_lane(
        ctx: &DeviceContext,
        qw: &CudaSlice<i8>,
        qs: &CudaSlice<bf16>,
        x: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        n: usize,
        m: usize,
        k: usize,
    ) -> Result<()> {
        let wbf16 = ctx.stream.alloc_zeros::<bf16>(n * k)?;
        let (qwp, _g0) = qw.device_ptr(&ctx.stream);
        let (qsp, _g1) = qs.device_ptr(&ctx.stream);
        let (wp, _g2) = wbf16.device_ptr(&ctx.stream);
        let (xp, _g3) = x.device_ptr(&ctx.stream);
        let (op, _g4) = out.device_ptr_mut(&ctx.stream);
        // SAFETY: dequant fills [n,k] bf16, then gemm(w[n,k], x[m,k]) -> out[m,n].
        unsafe {
            ffi::dequantize_w8a16_to_bf16_cuda(
                qwp as *const i8,
                qsp as *const ffi::Half,
                wp as *mut ffi::Half,
                n as i32,
                k as i32,
                GROUP as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
            ffi::gemm_cuda(
                wp as *const ffi::Half,
                xp as *const ffi::Half,
                op as *mut ffi::Half,
                n as i32,
                m as i32,
                k as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
        Ok(())
    }

    fn probe_one(
        ctx: &DeviceContext,
        label: &str,
        m: usize,
        n: usize,
        k: usize,
        seed: u64,
        negative: bool,
    ) -> Result<(bool, f64)> {
        let mut seed = shape_seed(seed, label, n, k);
        seed ^= (m as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut rng = Rng::new(seed);

        let x_host: Vec<f32> = (0..m * k).map(|_| rng.normal()).collect();
        let w_host: Vec<f32> = (0..n * k).map(|_| rng.normal() * 0.1).collect();
        let (q, s) = per_group_int8(&w_host, n, k);

        let ref_out = host_reference(&q, &s, &x_host, n, k);

        // Device W8A16 matrix (+ Marlin repack). from_quantized_int8 takes the
        // typed i8 weight + bf16 scale slices directly.
        let mut weight = DeviceMatrix::from_quantized_int8(ctx, &q, &s, n, k, GROUP)?;
        // repack frees qweight/qscales, so upload the dequant-lane copy first.
        let deq_qw = ctx.stream.clone_htod(&q)?;
        let deq_qs = ctx.stream.clone_htod(&s)?;
        weight.repack_for_marlin_w8a16(ctx)?;
        ensure!(
            weight.marlin_packed.is_some(),
            "[{label} m={m}] repack produced no marlin_packed — shape not tile-aligned?"
        );

        let x_bf16: Vec<bf16> = x_host.iter().map(|v| bf16::from_f32(*v)).collect();
        let x = ctx.stream.clone_htod(&x_bf16)?;
        let mut marlin_out = ctx.stream.alloc_zeros::<bf16>(m * n)?;
        let mut deq_out = ctx.stream.alloc_zeros::<bf16>(m * n)?;
        let sms = ctx.sm_count() as i32;
        // SAFETY: size queries only.
        let c_tmp_floats = unsafe { ffi::marlin_c_tmp_floats(m as i32, sms) } as usize;
        // SAFETY: size queries only.
        let ws_ints = unsafe { ffi::marlin_workspace_ints(sms) } as usize;
        let c_tmp = ctx.stream.alloc_zeros::<f32>(c_tmp_floats)?;
        let workspace = ctx.stream.alloc_zeros::<i32>(ws_ints)?;
        ctx.sync()?;

        // Lane 1: Marlin GEMM.
        {
            let (pp, _g0) = weight
                .marlin_packed
                .as_ref()
                .unwrap()
                .device_ptr(&ctx.stream);
            let (sp, _g1) = weight
                .marlin_scales
                .as_ref()
                .unwrap()
                .device_ptr(&ctx.stream);
            let (xp, _g2) = x.device_ptr(&ctx.stream);
            let (op, _g3) = marlin_out.device_ptr_mut(&ctx.stream);
            let (cp, _g4) = c_tmp.device_ptr(&ctx.stream);
            let (wp, _g5) = workspace.device_ptr(&ctx.stream);
            // SAFETY: all ptrs cover the exact m/n/k launch; scratch sized above.
            unsafe {
                ffi::marlin_w8a16_gemm_cuda(
                    xp as *const ffi::Half,
                    pp as *const u32,
                    sp as *const ffi::Half,
                    op as *mut ffi::Half,
                    cp as *mut f32,
                    wp as *mut i32,
                    m as i32,
                    n as i32,
                    k as i32,
                    GROUP as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
        // Lane 2: dequant→cuBLAS-BF16 (in-tree reference).
        launch_dequant_lane(ctx, &deq_qw, &deq_qs, &x, &mut deq_out, n, m, k)?;
        ctx.sync()?;

        let marlin = ctx.stream.clone_dtoh(&marlin_out)?;
        let deq = ctx.stream.clone_dtoh(&deq_out)?;
        // Marlin-lane family corruption: 3x the expectation makes its rel-L2
        // (~2/3) and ratio blow regardless of output size. The fallback lane
        // keeps the clean reference.
        let deq_err = rel_l2(&deq, &ref_out);
        let marlin_err = if negative {
            let bad_ref: Vec<f64> = ref_out.iter().map(|&r| 3.0 * r).collect();
            rel_l2(&marlin, &bad_ref)
        } else {
            rel_l2(&marlin, &ref_out)
        };
        let ratio = marlin_err / deq_err.max(1e-9);
        let pass = ratio <= MARLIN_VS_FALLBACK_MAX_RATIO
            && marlin_err.is_finite()
            && marlin_err <= MAX_REL_L2
            && deq_err <= MAX_REL_L2;
        eprintln!(
            "[{label} m={m:>2} n={n} k={k} seed={seed:#x}] marlin_relL2={marlin_err:.4e} \
             fallback_relL2={deq_err:.4e} ratio={ratio:.2} {}",
            if pass { "PASS" } else { "FAIL" }
        );
        Ok((!pass, deq_err))
    }

    /// A declined shape: planner returns no Marlin transform; the dequant→BF16 fallback alone carries every M against the f32 reference.
    fn probe_declined(
        ctx: &DeviceContext,
        label: &str,
        n: usize,
        k: usize,
        seed: u64,
        accepted_max_deq: f64,
        negative: bool,
    ) -> Result<bool> {
        let mut rng = Rng::new(shape_seed(seed, label, n, k));
        let w_host: Vec<f32> = (0..n * k).map(|_| rng.normal() * 0.1).collect();
        let (q, s) = per_group_int8(&w_host, n, k);

        let weight = DeviceMatrix::from_quantized_int8(ctx, &q, &s, n, k, GROUP)?;

        // Query the same layout decision the loader makes.
        let plan = infer_quant::plan_weight_layout(
            &infer_quant::WeightLayoutQuery::from(&weight),
            &infer_quant::DeviceCaps {
                compute_capability: ctx.compute_capability(),
            },
            &infer_quant::LayoutPolicy::default(),
        )?;
        if plan.transform != infer_quant::RepackTransform::None {
            eprintln!(
                "[{label} n={n} k={k} seed={seed:#x}] expected planner decline, \
                 got transform {:?} — planner W8A16 N%64 boundary regressed",
                plan.transform
            );
            return Ok(true);
        }
        let mut any_fail = false;
        for &m in M_SWEEP {
            let mut seed_m = shape_seed(seed, label, n, k);
            seed_m ^= (m as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let mut rng_m = Rng::new(seed_m);
            let x_host: Vec<f32> = (0..m * k).map(|_| rng_m.normal()).collect();
            let ref_out = host_reference(&q, &s, &x_host, n, k);

            let x_bf16: Vec<bf16> = x_host.iter().map(|v| bf16::from_f32(*v)).collect();
            let x = ctx.stream.clone_htod(&x_bf16)?;
            let mut deq_out = ctx.stream.alloc_zeros::<bf16>(m * n)?;
            launch_dequant_lane(
                ctx,
                weight.qweight.as_ref().unwrap(),
                weight.qscales.as_ref().unwrap(),
                &x,
                &mut deq_out,
                n,
                m,
                k,
            )?;
            ctx.sync()?;

            let deq = ctx.stream.clone_dtoh(&deq_out)?;
            let deq_err = if negative {
                // Declined-fallback family corruption: scale the expectation 3x
                // so the error must exceed the accepted-shape floor cap.
                let bad_ref: Vec<f64> = ref_out.iter().map(|&r| 3.0 * r).collect();
                rel_l2(&deq, &bad_ref)
            } else {
                rel_l2(&deq, &ref_out)
            };
            let cap = accepted_max_deq * MARLIN_VS_FALLBACK_MAX_RATIO;
            let pass = deq_err.is_finite() && deq_err <= cap && deq_err <= MAX_REL_L2;
            any_fail |= !pass;
            eprintln!(
                "[{label} m={m:>2} n={n} k={k} seed={seed:#x}] declined fallback_relL2={deq_err:.4e} \
                 (cap {cap:.4e}) {}",
                if pass { "PASS" } else { "FAIL" }
            );
        }
        Ok(any_fail)
    }

    fn rel_l2(got: &[bf16], reference: &[f64]) -> f64 {
        let mut num = 0f64;
        let mut den = 0f64;
        for (g, r) in got.iter().zip(reference) {
            let d = f32::from(*g) as f64 - r;
            num += d * d;
            den += r * r;
        }
        (num / den.max(1e-12)).sqrt()
    }
}
