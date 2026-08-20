//! Marlin FP8 per-channel numeric-parity gate.
//!
//! Standalone kernel harness — NO engine, NO serve. Builds a random BF16 weight
//! over the Qwen3.8-27B per-channel FP8 shapes, quantizes it per output channel
//! to E4M3 (scale = amax/448, the compressed-tensors float-quantized form), then
//! compares three lanes over an M sweep:
//!   - f64 host reference (ground truth: dequant E4M3 in f64, matmul in f64)
//!   - Marlin tensor-core GEMM (`repack_for_marlin_fp8` + `marlin_fp8_gemm_cuda`)
//!   - the in-tree scalar batched GEMV (`gemv_fp8_block_scaled_batch_cuda`)
//!
//! Both device lanes are anchored on f64, never on each other. The reference
//! dequantizes the SAME E4M3 bytes both kernels read, so quantization error is
//! not in the comparison at all — what is left is arithmetic precision, and both
//! lanes should land at the BF16 output-rounding floor (~1.1e-3 rel L2).
//!
//! PASS requires all three of: the Marlin error is finite, it is below an
//! absolute cap, and it is within a small multiple of the GEMV lane's. A Marlin
//! error many× the GEMV's is the silent-wrong-repack / wrong-scale-perm signal
//! this gate exists to catch — the class of defect that made the first NVFP4
//! Marlin wiring return `nonzero 0/256` and that no perf measurement would find.
//! Exits non-zero on any violation.
//!
//! The two failure modes and what they look like in the printout:
//!   - the 2^120 skip-flop factor missing from the per-channel scale: Marlin's
//!     output is `ref * 2^-120`, which underflows BF16 to zero, so
//!     `mean(out/ref)` reads 0.000000 and `rel_l2` reads 1.00.
//!   - the length-64 grouped `scale_perm` used where channelwise needs the
//!     length-32 `scale_perm_single`: output channels get each other's scales,
//!     so `max|err|/rms` is O(1) while `mean(out/ref)` stays near 1.
//!
//! Run on a pod: `INFER_CUDA_DEVICE=<free-gpu> target/release/examples/marlin_fp8_parity`

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
        eprintln!("marlin_fp8_parity is a CUDA harness; rebuild with --features cuda.");
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

    /// (label, N = output dim / weight rows, K = contraction / weight cols).
    ///
    /// The 145 per-channel FP8 GEMMs of `unsloth/Qwen3.8-27B-NVFP4` collapse to
    /// five distinct shapes. hidden 5120, 24 heads, 4 KV heads, head_dim 256,
    /// gated q_proj, 64 layers (48 linear-attn + 16 full-attn), vocab 248320
    /// (`docs/baselines.md`, `docs/experience/wins/2026-08-19-nvfp4-marlin-tensorcore.md`).
    ///
    /// - 12288x5120 covers both the gated `q_proj` (24*256*2, `full_attn_q_proj_dim`
    ///   in `qwen35-spec/src/lib.rs:1031`) and the fused linear-attn `in_proj_qkvz`
    ///   (`qwen35_load.rs:423`); the latter's 12288 assumes Kh=16 / Vh=32
    ///   (`qwen35_load.rs:275`) at head_dim 128, which is inferred, not read off
    ///   the checkpoint.
    /// - 1024x5120 is `k_proj` / `v_proj` (4*256).
    /// - 5120x6144 is `o_proj` (contracting the un-gated 24*256 attention output).
    /// - 5120x4096 is the linear-attn `out_proj` (contracting 32*128 value dims).
    /// - 248320x5120 is `lm_head`.
    const SHAPES: &[(&str, usize, usize)] = &[
        ("q_proj/in_proj_qkvz", 12288, 5120),
        ("k_proj/v_proj", 1024, 5120),
        ("o_proj", 5120, 6144),
        ("linear out_proj", 5120, 4096),
        ("lm_head", 248_320, 5120),
    ];
    const M_SWEEP: &[usize] = &[1, 2, 4, 16, 64, 256];

    /// Three fixed seeds per shape; the gate is shapes × seeds × M.
    const SEEDS: &[u64] = &[
        0x9e37_79b9_7f4a_7c15,
        0x517c_c1cc_9e37_79b9,
        0x2545_f491_4f6c_dd1d,
    ];

    /// Boundary shapes the repack must decline: N not %64 with K aligned. The
    /// source has to stay resident and the GEMV lane carry the shape alone.
    const DECLINED_SHAPES: &[(&str, usize, usize)] = &[("declined n%64", 96, 5120)];

    /// The host reference costs `M_max * K` f64 FMAs per output column, so it is
    /// computed over a sampled column set instead of all N: two contiguous slabs
    /// of this width, at the start and the end of the output dim. 512 columns is
    /// 16 whole 32-column scale-permutation blocks, so every `scale_perm_single`
    /// slot is exercised, and the tail slab covers the last Marlin n-tile.
    const REF_SLAB: usize = 512;

    /// Marlin must not exceed the GEMV lane's f64-anchored error by more than
    /// this. Both lanes read the same E4M3 bytes and accumulate in f32, so a
    /// correct Marlin lane tracks the GEMV closely; a blown ratio is a wrong
    /// repack or a wrong scale permutation.
    const MARLIN_VS_GEMV_MAX_RATIO: f64 = 4.0;

    /// Absolute ceiling on either lane's relative L2. The ratio test alone
    /// passes when BOTH lanes are broken; this catches that. The expected floor
    /// is the BF16 output rounding, rms 2^-9/sqrt(3) ≈ 1.1e-3.
    const MAX_REL_L2: f64 = 8e-3;

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

    /// Round-half-to-even, matching the device `__nv_fp8_e4m3` conversion.
    fn rint_ne(x: f32) -> u8 {
        let fl = x.floor();
        let frac = x - fl;
        let fl_u = fl as u8;
        if frac > 0.5 || (frac == 0.5 && fl_u % 2 == 1) {
            fl_u + 1
        } else {
            fl_u
        }
    }

    /// Encode an f32 to FP8 E4M3 (OCP `fn` variant: no infinities). Clips to
    /// ±448, the max finite value. Never emits the NaN encoding.
    fn f32_to_e4m3(value: f32) -> u8 {
        if !value.is_finite() || value == 0.0 {
            return 0;
        }
        let sign = u8::from(value < 0.0) << 7;
        let abs_val = value.abs();
        if abs_val >= 448.0 {
            return sign | 0x7e;
        }
        let exp_unbiased = abs_val.log2().floor() as i32;
        let exp_rebiased = exp_unbiased + 7; // E4M3 bias = 7
        if exp_rebiased <= 0 {
            // Subnormal: exponent field 0, value = (mant/8) * 2^-6.
            let mant = rint_ne(abs_val * (6.0f32).exp2() * 8.0);
            if mant == 0 {
                return 0;
            }
            if mant >= 8 {
                return sign | (1u8 << 3); // rounded up to the smallest normal
            }
            return sign | mant;
        }
        let mant_frac = abs_val * ((-exp_unbiased) as f32).exp2() - 1.0;
        let mant = rint_ne(mant_frac * 8.0);
        let (exp_final, mant_final) = if mant >= 8 {
            (exp_rebiased + 1, 0u8)
        } else {
            (exp_rebiased, mant)
        };
        if exp_final > 15 {
            return sign | 0x7e;
        }
        sign | ((exp_final as u8) << 3) | mant_final
    }

    /// Decode an FP8 E4M3 byte. Exact in f64 — this is the reference's weight.
    fn e4m3_to_f64(byte: u8) -> f64 {
        let exp = i32::from((byte >> 3) & 0x0f);
        let mant = f64::from(byte & 0x07);
        let mag = if exp == 0 {
            (mant / 8.0) * (-6.0f64).exp2()
        } else {
            (1.0 + mant / 8.0) * f64::from(exp - 7).exp2()
        };
        if byte & 0x80 != 0 { -mag } else { mag }
    }

    /// Per-output-channel E4M3 quantization, `scale = amax / 448`.
    ///
    /// The scale is rounded to BF16 because the checkpoint stores `weight_scale`
    /// as BF16 (`quant_format.rs:192-215` detection arm). That is what makes the
    /// repack's 2^120 fold bit-exact — handing the harness an f32-only scale
    /// would add a per-channel error the production weight never carries and
    /// blunt the ratio test.
    fn quantize_per_channel_e4m3(rng: &mut Rng, n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
        let mut q = vec![0u8; n * k];
        let mut scales = vec![0f32; n];
        let mut row = vec![0f32; k];
        for r in 0..n {
            for v in row.iter_mut() {
                *v = rng.normal() * 0.1;
            }
            let amax = row.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-8);
            let scale = f32::from(bf16::from_f32(amax / 448.0));
            scales[r] = scale;
            for (i, v) in row.iter().enumerate() {
                q[r * k + i] = f32_to_e4m3(v / scale);
            }
        }
        (q, scales)
    }

    /// Output columns the f64 reference covers: see `REF_SLAB`.
    fn reference_columns(n: usize) -> Vec<usize> {
        if n <= 2 * REF_SLAB {
            (0..n).collect()
        } else {
            (0..REF_SLAB).chain(n - REF_SLAB..n).collect()
        }
    }

    /// `ref[j * m_max + row] = sum_k x[row, k] * e4m3(q[col_j, k]) * scale[col_j]`,
    /// column-major over the sampled columns so the inner loop is contiguous in
    /// both operands. `xt` is the `[k][m_max]` f64 transpose of the activations.
    fn host_reference(
        q: &[u8],
        scales: &[f32],
        k: usize,
        cols: &[usize],
        xt: &[f64],
        m_max: usize,
    ) -> Vec<f64> {
        let mut out = vec![0f64; cols.len() * m_max];
        for (j, &col) in cols.iter().enumerate() {
            let scale = f64::from(scales[col]);
            let qrow = &q[col * k..(col + 1) * k];
            let acc = &mut out[j * m_max..(j + 1) * m_max];
            for (kk, &byte) in qrow.iter().enumerate() {
                let w = e4m3_to_f64(byte) * scale;
                if w == 0.0 {
                    continue;
                }
                for (a, x) in acc.iter_mut().zip(&xt[kk * m_max..(kk + 1) * m_max]) {
                    *a += x * w;
                }
            }
        }
        out
    }

    struct LaneStats {
        rel_l2: f64,
        max_over_rms: f64,
        mean_ratio: f64,
    }

    /// Compare the first `m` rows of a `[m, n]` device output against the
    /// column-major f64 reference over `cols`.
    fn lane_stats(
        out: &[bf16],
        n: usize,
        m: usize,
        cols: &[usize],
        reference: &[f64],
        m_max: usize,
    ) -> LaneStats {
        let mut sq_err = 0f64;
        let mut sq_ref = 0f64;
        let mut max_err = 0f64;
        for (j, &col) in cols.iter().enumerate() {
            for row in 0..m {
                let r = reference[j * m_max + row];
                let d = f64::from(f32::from(out[row * n + col])) - r;
                sq_err += d * d;
                sq_ref += r * r;
                if d.abs() > max_err {
                    max_err = d.abs();
                }
            }
        }
        let rms_ref = (sq_ref / (m * cols.len()) as f64).sqrt();
        // out/ref only where the reference is well clear of its own rms: near-zero
        // denominators turn a systematic scale factor into noise.
        let floor = 0.1 * rms_ref;
        let mut ratio_sum = 0f64;
        let mut ratio_n = 0usize;
        for (j, &col) in cols.iter().enumerate() {
            for row in 0..m {
                let r = reference[j * m_max + row];
                if r.abs() <= floor {
                    continue;
                }
                ratio_sum += f64::from(f32::from(out[row * n + col])) / r;
                ratio_n += 1;
            }
        }
        LaneStats {
            rel_l2: (sq_err / sq_ref.max(f64::MIN_POSITIVE)).sqrt(),
            max_over_rms: max_err / rms_ref.max(f64::MIN_POSITIVE),
            mean_ratio: if ratio_n > 0 {
                ratio_sum / ratio_n as f64
            } else {
                f64::NAN
            },
        }
    }

    pub(super) fn run() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let cc = ctx.compute_capability();
        eprintln!(
            "[marlin-fp8-parity] device={} cc={}.{} sms={} build={}",
            ctx.ordinal(),
            cc.0,
            cc.1,
            ctx.sm_count(),
            cuda_kernels::KERNEL_BUILD_ID
        );
        ensure!(cc.0 >= 8, "Marlin needs sm_80+; got sm_{}{}", cc.0, cc.1);
        let m_max = *M_SWEEP
            .iter()
            .max()
            .expect("M_SWEEP must name at least one M");

        let mut any_fail = false;
        for &seed in SEEDS {
            for &(label, n, k) in SHAPES {
                any_fail |= probe_shape(&ctx, label, n, k, m_max, seed)?;
            }
            for &(label, n, k) in DECLINED_SHAPES {
                any_fail |= probe_declined(&ctx, label, n, k, m_max, seed)?;
            }
        }
        ensure!(!any_fail, "marlin_fp8_parity FAILED — see violations above");
        eprintln!("[marlin-fp8-parity] ALL PASS");
        Ok(())
    }

    /// FNV-mix the fixed seed with the shape identity so every (seed, shape)
    /// pair gets a distinct deterministic matrix.
    fn shape_seed(seed: u64, label: &str, n: usize, k: usize) -> u64 {
        let mut mix = seed;
        for byte in label.as_bytes() {
            mix ^= u64::from(*byte);
            mix = mix.wrapping_mul(0x0000_0100_0000_01b3); // FNV prime
        }
        mix ^= (n as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        mix ^= (k as u64).wrapping_mul(0x517c_c1cc_9e37_79b9);
        mix
    }

    /// Sampled columns plus the f64 reference over the `[m_max, k]`
    /// activations. The device reads these same BF16 values, so activation
    /// rounding is not part of the comparison.
    fn build_reference(
        qbytes: &[u8],
        scales: &[f32],
        n: usize,
        k: usize,
        m_max: usize,
        x_bf16: &[bf16],
    ) -> (Vec<usize>, Vec<f64>) {
        let cols = reference_columns(n);
        // [k][m_max] f64 transpose; the reference walks it contiguously.
        let mut xt = vec![0f64; k * m_max];
        for row in 0..m_max {
            for kk in 0..k {
                xt[kk * m_max + row] = f64::from(f32::from(x_bf16[row * k + kk]));
            }
        }
        let reference = host_reference(qbytes, scales, k, &cols, &xt, m_max);
        (cols, reference)
    }

    /// The scalar batched GEMV lane over a source-retaining `DeviceMatrix`.
    fn launch_gemv_lane(
        ctx: &DeviceContext,
        weight: &DeviceMatrix,
        x: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let (qwp, _g0) = weight.qweight_u8.as_ref().unwrap().device_ptr(&ctx.stream);
        let (sfp, _g1) = weight.scale_f32.as_ref().unwrap().device_ptr(&ctx.stream);
        let (xp, _g2) = x.device_ptr(&ctx.stream);
        let (op, _g3) = out.device_ptr_mut(&ctx.stream);
        // SAFETY: weight is [n, k] E4M3 and scales is [n] f32; block_m=1 /
        // block_k=k makes the kernel's index `scales[row]`, matching the
        // per-channel layout. x is the [m, k] prefix, out is exactly m*n.
        unsafe {
            ffi::gemv_fp8_block_scaled_batch_cuda(
                qwp as *const u8,
                sfp as *const f32,
                xp as *const ffi::Half,
                op as *mut ffi::Half,
                m as i32,
                n as i32,
                k as i32,
                n as i32, // scale_rows
                1,        // scale_cols
                1,        // block_m
                k as i32, // block_k
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
        Ok(())
    }

    fn probe_shape(
        ctx: &DeviceContext,
        label: &str,
        n: usize,
        k: usize,
        m_max: usize,
        seed: u64,
    ) -> Result<bool> {
        let mut rng = Rng::new(shape_seed(seed, label, n, k));

        let (qbytes, scales) = quantize_per_channel_e4m3(&mut rng, n, k);
        let x_bf16: Vec<bf16> = (0..m_max * k)
            .map(|_| bf16::from_f32(rng.normal()))
            .collect();

        let (cols, reference) = build_reference(&qbytes, &scales, n, k, m_max, &x_bf16);

        // block_m = 1, block_k = K is the per-channel encoding of a block-scaled
        // FP8 weight — the same one `quant_format.rs` produces for a
        // compressed-tensors `F8_E4M3` + `[N,1]` weight_scale checkpoint.
        let mut weight = DeviceMatrix::from_fp8_block_scaled(ctx, &qbytes, &scales, n, k, 1, k)?;
        // The repack releases the source, so lane 2 gets its own unrepacked copy.
        let gemv_weight = DeviceMatrix::from_fp8_block_scaled(ctx, &qbytes, &scales, n, k, 1, k)?;
        drop(qbytes);
        weight.repack_for_marlin_fp8(ctx)?;
        ensure!(
            weight.marlin_packed.is_some() && weight.marlin_scales.is_some(),
            "[{label} n={n} k={k}] repack_for_marlin_fp8 declined the shape — \
             the Marlin lane is not under test"
        );

        let x = ctx.stream.clone_htod(&x_bf16)?;
        let sms = ctx.sm_count() as i32;
        // SAFETY: size queries, no pointers involved.
        let c_tmp_floats = unsafe { ffi::marlin_c_tmp_floats(m_max as i32, sms) } as usize;
        // SAFETY: size query, no pointers involved.
        let ws_ints = unsafe { ffi::marlin_workspace_ints(sms) } as usize;
        // Sized for the largest M in the sweep and reused: Marlin leaves the
        // locks at zero, so one zero-init covers every call.
        let c_tmp = ctx.stream.alloc_zeros::<f32>(c_tmp_floats)?;
        let workspace = ctx.stream.alloc_zeros::<i32>(ws_ints)?;
        ctx.sync()?;

        let mut any_fail = false;
        for &m in M_SWEEP {
            let mut marlin_out = ctx.stream.alloc_zeros::<bf16>(m * n)?;
            let mut gemv_out = ctx.stream.alloc_zeros::<bf16>(m * n)?;

            // Lane 1: Marlin tensor-core GEMM over the repacked kFE4M3fn weight.
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
                // SAFETY: every pointer comes from a live CudaSlice pinned by its
                // guard. `x` holds m_max*k >= m*k activations and the kernel reads
                // the [m, k] prefix; the output is exactly m*n; the scratch is
                // sized for m_max above. There is no group_size argument — the
                // format is channelwise by construction (group_blocks = -1 is the
                // only kFE4M3fn instantiation this path may reach).
                unsafe {
                    ffi::marlin_fp8_gemm_cuda(
                        xp as *const ffi::Half,
                        pp as *const u32,
                        sp as *const ffi::Half,
                        op as *mut ffi::Half,
                        cp as *mut f32,
                        wp as *mut i32,
                        m as i32,
                        n as i32,
                        k as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
            // Lane 2: the scalar batched GEMV this replaces, reading the same
            // E4M3 bytes and the un-folded f32 per-channel scales.
            launch_gemv_lane(ctx, &gemv_weight, &x, &mut gemv_out, m, n, k)?;
            ctx.sync()?;

            let marlin = ctx.stream.clone_dtoh(&marlin_out)?;
            let gemv = ctx.stream.clone_dtoh(&gemv_out)?;
            let ms = lane_stats(&marlin, n, m, &cols, &reference, m_max);
            let gs = lane_stats(&gemv, n, m, &cols, &reference, m_max);
            let ratio = ms.rel_l2 / gs.rel_l2.max(1e-12);
            let pass = ms.rel_l2.is_finite()
                && ms.rel_l2 <= MAX_REL_L2
                && ratio <= MARLIN_VS_GEMV_MAX_RATIO;
            any_fail |= !pass;
            eprintln!(
                "[{label} m={m:>3} n={n} k={k} seed={seed:#x}] \
                 marlin relL2={:.4e} max/rms={:.4e} mean(out/ref)={:.6} | \
                 gemv relL2={:.4e} max/rms={:.4e} mean(out/ref)={:.6} | \
                 ratio={ratio:.2} {}",
                ms.rel_l2,
                ms.max_over_rms,
                ms.mean_ratio,
                gs.rel_l2,
                gs.max_over_rms,
                gs.mean_ratio,
                if pass { "PASS" } else { "FAIL" }
            );
        }
        Ok(any_fail)
    }

    /// A shape the repack must decline: assert the source stays resident and
    /// the GEMV lane carries every M against the f64 reference alone.
    fn probe_declined(
        ctx: &DeviceContext,
        label: &str,
        n: usize,
        k: usize,
        m_max: usize,
        seed: u64,
    ) -> Result<bool> {
        let mut rng = Rng::new(shape_seed(seed, label, n, k));
        let (qbytes, scales) = quantize_per_channel_e4m3(&mut rng, n, k);
        let x_bf16: Vec<bf16> = (0..m_max * k)
            .map(|_| bf16::from_f32(rng.normal()))
            .collect();

        let (cols, reference) = build_reference(&qbytes, &scales, n, k, m_max, &x_bf16);

        let mut weight = DeviceMatrix::from_fp8_block_scaled(ctx, &qbytes, &scales, n, k, 1, k)?;
        weight.repack_for_marlin_fp8(ctx)?;
        if weight.marlin_packed.is_some() || weight.marlin_scales.is_some() {
            eprintln!(
                "[{label} n={n} k={k} seed={seed:#x}] expected repack decline, \
                 got a Marlin layout — shape is no longer a boundary"
            );
            return Ok(true);
        }
        ensure!(
            weight.qweight_u8.is_some() && weight.scale_f32.is_some(),
            "[{label}] repack declined but released the source — no route can serve this shape"
        );

        let x = ctx.stream.clone_htod(&x_bf16)?;
        let mut any_fail = false;
        for &m in M_SWEEP {
            let mut gemv_out = ctx.stream.alloc_zeros::<bf16>(m * n)?;
            launch_gemv_lane(ctx, &weight, &x, &mut gemv_out, m, n, k)?;
            ctx.sync()?;
            let gemv = ctx.stream.clone_dtoh(&gemv_out)?;
            let gs = lane_stats(&gemv, n, m, &cols, &reference, m_max);
            let pass = gs.rel_l2.is_finite() && gs.rel_l2 <= MAX_REL_L2;
            any_fail |= !pass;
            eprintln!(
                "[{label} m={m:>3} n={n} k={k} seed={seed:#x}] \
                 declined gemv relL2={:.4e} max/rms={:.4e} mean(out/ref)={:.6} {}",
                gs.rel_l2,
                gs.max_over_rms,
                gs.mean_ratio,
                if pass { "PASS" } else { "FAIL" }
            );
        }
        Ok(any_fail)
    }
}
