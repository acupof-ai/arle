//! Marlin W8A16 numeric-parity gate (Phase 1 of the W8A16 tensor-core port).
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
//! Run on a pod: `INFER_CUDA_DEVICE=<free-gpu> target/release/examples/marlin_w8a16_parity`

fn main() -> anyhow::Result<()> {
    if std::env::args().nth(1).as_deref() == Some("--kernel-build-id") {
        println!("{}", cuda_kernels::KERNEL_BUILD_ID);
        return Ok(());
    }
    real::run()
}

#[cfg(not(feature = "cuda"))]
mod real {
    pub(super) fn run() -> anyhow::Result<()> {
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
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

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
    /// Boundary shapes the repack must decline: N not %64 with K aligned. The
    /// source has to stay resident and the dequant→BF16 fallback carry alone.
    const DECLINED_SHAPES: &[(&str, usize, usize)] = &[("declined n%64", 96, 5120)];
    /// Absolute ceiling for a lane anchored on the f32 reference without a
    /// comparison lane (repack-declined shapes). Expected floor ≈1.1e-3.
    const FALLBACK_MAX_REL_L2: f64 = 8e-3;
    // Marlin lane must not exceed the fallback lane's f32-anchored error by more
    // than this factor — both share the INT8 group-quant floor, so a correct
    // Marlin lane tracks the fallback closely. A blown ratio = wrong repack/perm.
    const MARLIN_VS_FALLBACK_MAX_RATIO: f64 = 4.0;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            // xorshift64*
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn unit(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
        }
        fn normal(&mut self) -> f32 {
            // Box–Muller, one sample.
            let u1 = self.unit().max(1e-7);
            let u2 = self.unit();
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        }
    }

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

    pub(super) fn run() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let cc = ctx.compute_capability();
        eprintln!(
            "[marlin-parity] device={} cc={}.{} build={}",
            ctx.ordinal(),
            cc.0,
            cc.1,
            cuda_kernels::KERNEL_BUILD_ID
        );
        ensure!(
            cc.0 >= 8,
            "Marlin W8A16 needs sm_80+; got sm_{}{}",
            cc.0,
            cc.1
        );

        let mut any_fail = false;
        for &seed in SEEDS {
            for &(label, n, k) in SHAPES {
                for &m in M_SWEEP {
                    let fail = probe_one(&ctx, label, m, n, k, seed)?;
                    any_fail |= fail;
                }
            }
            for &(label, n, k) in DECLINED_SHAPES {
                for &m in M_SWEEP {
                    any_fail |= probe_declined(&ctx, label, m, n, k, seed)?;
                }
            }
        }
        ensure!(
            !any_fail,
            "marlin_w8a16_parity FAILED — see violations above"
        );
        eprintln!("[marlin-parity] ALL PASS");
        Ok(())
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

    fn probe_one(
        ctx: &DeviceContext,
        label: &str,
        m: usize,
        n: usize,
        k: usize,
        seed: u64,
    ) -> Result<bool> {
        let ng = k / GROUP;
        let mut seed = shape_seed(seed, label, n, k);
        seed ^= (m as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut rng = Rng::new(seed);

        let x_host: Vec<f32> = (0..m * k).map(|_| rng.normal()).collect();
        let w_host: Vec<f32> = (0..n * k).map(|_| rng.normal() * 0.1).collect();
        let (q, s) = per_group_int8(&w_host, n, k);

        // f32 reference: dequant weight (bf16 scale, faithful to device) then matmul
        // in f64. out[row_m, row_n] = sum_k x[row_m,k] * (q * scale).
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
        {
            let wbf16 = ctx.stream.alloc_zeros::<bf16>(n * k)?;
            let (qwp, _g0) = deq_qw.device_ptr(&ctx.stream);
            let (qsp, _g1) = deq_qs.device_ptr(&ctx.stream);
            let (wp, _g2) = wbf16.device_ptr(&ctx.stream);
            let (xp, _g3) = x.device_ptr(&ctx.stream);
            let (op, _g4) = deq_out.device_ptr_mut(&ctx.stream);
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
        }
        ctx.sync()?;

        let marlin = ctx.stream.clone_dtoh(&marlin_out)?;
        let deq = ctx.stream.clone_dtoh(&deq_out)?;
        let marlin_err = rel_l2(&marlin, &ref_out);
        let deq_err = rel_l2(&deq, &ref_out);
        let ratio = marlin_err / deq_err.max(1e-9);
        let pass = ratio <= MARLIN_VS_FALLBACK_MAX_RATIO && marlin_err.is_finite();
        eprintln!(
            "[{label} m={m:>2} n={n} k={k} seed={seed:#x}] marlin_relL2={marlin_err:.4e} \
             fallback_relL2={deq_err:.4e} ratio={ratio:.2} {}",
            if pass { "PASS" } else { "FAIL" }
        );
        Ok(!pass)
    }

    /// A shape the repack must decline: the source stays resident and the
    /// dequant→BF16 fallback carries every M against the f32 reference alone.
    fn probe_declined(
        ctx: &DeviceContext,
        label: &str,
        m: usize,
        n: usize,
        k: usize,
        seed: u64,
    ) -> Result<bool> {
        let ng = k / GROUP;
        let mut seed = shape_seed(seed, label, n, k);
        seed ^= (m as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut rng = Rng::new(seed);

        let x_host: Vec<f32> = (0..m * k).map(|_| rng.normal()).collect();
        let w_host: Vec<f32> = (0..n * k).map(|_| rng.normal() * 0.1).collect();
        let (q, s) = per_group_int8(&w_host, n, k);

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

        let mut weight = DeviceMatrix::from_quantized_int8(ctx, &q, &s, n, k, GROUP)?;
        weight.repack_for_marlin_w8a16(ctx)?;
        if weight.marlin_packed.is_some() {
            eprintln!(
                "[{label} n={n} k={k} seed={seed:#x}] expected repack decline, \
                 got a Marlin layout — shape is no longer a boundary"
            );
            return Ok(true);
        }
        ensure!(
            weight.qweight.is_some() && weight.qscales.is_some(),
            "[{label}] repack declined but released the source — no route can serve this shape"
        );

        let x_bf16: Vec<bf16> = x_host.iter().map(|v| bf16::from_f32(*v)).collect();
        let x = ctx.stream.clone_htod(&x_bf16)?;
        let mut deq_out = ctx.stream.alloc_zeros::<bf16>(m * n)?;
        {
            let wbf16 = ctx.stream.alloc_zeros::<bf16>(n * k)?;
            let (qwp, _g0) = weight.qweight.as_ref().unwrap().device_ptr(&ctx.stream);
            let (qsp, _g1) = weight.qscales.as_ref().unwrap().device_ptr(&ctx.stream);
            let (wp, _g2) = wbf16.device_ptr(&ctx.stream);
            let (xp, _g3) = x.device_ptr(&ctx.stream);
            let (op, _g4) = deq_out.device_ptr_mut(&ctx.stream);
            // SAFETY: same dequant→gemm lane as probe_one.
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
        }
        ctx.sync()?;

        let deq = ctx.stream.clone_dtoh(&deq_out)?;
        let deq_err = rel_l2(&deq, &ref_out);
        let pass = deq_err.is_finite() && deq_err <= FALLBACK_MAX_REL_L2;
        eprintln!(
            "[{label} m={m:>2} n={n} k={k} seed={seed:#x}] declined fallback_relL2={deq_err:.4e} {}",
            if pass { "PASS" } else { "FAIL" }
        );
        Ok(!pass)
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
