//! Small-M FP8 dense GEMM probe (Qwen3.6-27B decode/verify shapes).
//!
//! Sweeps M∈{1,2,4,8,16,17,32} over the dense projection shapes and compares
//! the two production FP8 lanes per (shape, M):
//! - scalar warp-per-row GEMV (`gemv_fp8_block_scaled_batch_cuda`, M<MIN_M lane)
//! - DeepGEMM dense (`pack_quantize` + `fp8_gemm_nt`, M>=MIN_M lane), with the
//!   pack/gemm split and host_ms exposed (JIT-cache per-call host overhead).
//!
//! Prints effective weight-read GB/s vs the HBM floor so the crossover M is
//! read directly off the table.
//!
//! Run on a pod: `INFER_CUDA_DEVICE=2 target/release/examples/fp8_smallm_gemm_probe`

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
        eprintln!("fp8_smallm_gemm_probe is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    use anyhow::Result;
    use cuda_kernels::ffi;
    use cuda_kernels::moe;
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::tensor::cache_ptr;
    use cudarc::driver::sys::CUevent_flags;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;
    use serde::Serialize;
    use serde_json::json;

    const FP8_BLOCK: usize = 128;
    const NUMERIC_ABS_TOLERANCE: f64 = 1.0;
    const NUMERIC_REL_TOLERANCE: f64 = 0.02;
    const M_SWEEP: &[usize] = &[1, 2, 4, 8, 16, 17, 32];
    // (label, N=weight rows, K=weight cols) — Qwen3.6-27B dense shapes.
    const SHAPES: &[(&str, usize, usize)] = &[
        ("ffn_gate_up", 17408, 5120),
        ("ffn_down", 5120, 17408),
        ("attn_sq", 5120, 5120),
    ];

    #[derive(Default, Clone)]
    struct Sample {
        cuda_us: Vec<f64>,
    }

    #[derive(Serialize)]
    struct NumericDelta {
        max_abs: f64,
        max_rel: f64,
        rmse: f64,
        max_abs_tolerance: f64,
        max_rel_tolerance: f64,
        tolerance_mode: &'static str,
        max_tolerance_ratio: f64,
        violations: usize,
        passed: bool,
    }

    pub(super) fn run() -> Result<()> {
        let iters: usize = std::env::var("ARLE_SMALLM_PROBE_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200);
        let samples: usize = std::env::var("ARLE_SMALLM_PROBE_SAMPLES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        anyhow::ensure!(samples >= 3, "ARLE_SMALLM_PROBE_SAMPLES must be >= 3");
        let ctx = DeviceContext::new()?;
        let cc = ctx.compute_capability();
        eprintln!(
            "[smallm-probe] device={} cc={}.{} iters={iters} samples={samples}",
            ctx.ordinal(),
            cc.0,
            cc.1
        );
        let (deepgemm, provider) = match moe::dsv4_deepgemm_native_preflight() {
            Ok(report) => {
                eprintln!("[smallm-probe] deepgemm preflight: {report}");
                (true, report)
            }
            Err(err) => {
                eprintln!("[smallm-probe] deepgemm UNAVAILABLE: {err}");
                (false, format!("unavailable:{err}"))
            }
        };

        let mut measurements = Vec::new();
        for &(label, n, k) in SHAPES {
            for &m in M_SWEEP {
                if let Some(measurement) =
                    probe_one(&ctx, label, m, n, k, iters, samples, deepgemm)?
                {
                    measurements.push(measurement);
                }
            }
        }
        let (commit, dirty) = git_source();
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let e2e_status =
            std::env::var("ARLE_SMALLM_E2E_STATUS").unwrap_or_else(|_| "not_run".to_string());
        let e2e_artifact_sha256 = std::env::var("ARLE_SMALLM_E2E_ARTIFACT_SHA256").ok();
        let e2e_model_revision = (e2e_status != "not_run").then(|| {
            json!({
                "kind": env_or_unreported("ARLE_SMALLM_E2E_MODEL_KIND"),
                "id": env_or_unreported("ARLE_SMALLM_E2E_MODEL_REVISION"),
            })
        });
        let run_id = std::env::var("ARLE_SMALLM_RUN_ID")
            .unwrap_or_else(|_| format!("smallm-{}-{now}", &commit[..8.min(commit.len())]));
        let run = json!({
            "schema_version": "arle.operator-evidence/v1",
            "run_id": run_id,
            "source": { "commit": commit, "dirty": dirty },
            "product": {
                "binary_id": env_or_unreported("ARLE_SMALLM_BINARY_ID"),
                "bundle_id": std::env::var("ARLE_SMALLM_BUNDLE_ID").ok(),
                "bundle_id_source": std::env::var("ARLE_SMALLM_BUNDLE_ID_SOURCE")
                    .unwrap_or_else(|_| "unverified".to_string()),
                "bundle_manifest_sha256": std::env::var("ARLE_SMALLM_BUNDLE_MANIFEST_SHA256").ok(),
            },
            "operator_id": "qwen.fp8_dense_projection",
            "model_revision": {
                "kind": env_or_unreported("ARLE_SMALLM_MODEL_KIND"),
                "id": env_or_unreported("ARLE_SMALLM_MODEL_REVISION"),
            },
            "hardware": {
                "gpu": ctx.ctx.name()?,
                "sm_major": cc.0,
                "sm_minor": cc.1,
                "sm_count": ctx.sm_count(),
            },
            "software": {
                "driver": cuda_driver_version(),
                "toolkit": command_last_line("nvcc", &["--version"]),
                "provider": provider,
            },
            "timing": {
                "method": "cuda_event_batched",
                "warmup": 1,
                "iterations_per_sample": iters,
                "samples": samples,
            },
            "e2e_gate": {
                "status": e2e_status,
                "passed": std::env::var("ARLE_SMALLM_E2E_PASS").as_deref() == Ok("1"),
                "artifact_sha256": e2e_artifact_sha256,
                "model_revision": e2e_model_revision,
            },
            "measurements": measurements,
        });
        println!("{}", serde_json::to_string_pretty(&run)?);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn probe_one(
        ctx: &DeviceContext,
        label: &'static str,
        m: usize,
        n: usize,
        k: usize,
        iters: usize,
        samples: usize,
        deepgemm: bool,
    ) -> Result<Option<serde_json::Value>> {
        let scale_rows = n.div_ceil(FP8_BLOCK);
        let scale_cols = k.div_ceil(FP8_BLOCK);
        // Deterministic random data: normal(0, 1) for activations and weight
        // values, uniform(0.5, 2.0) for block scales. This gives well-conditioned
        // block-scaled quantization (unlike repeating patterns where per-block
        // statistics are degenerate).
        // Deterministic seed from label bytes + shape dims (not the &str fat
        // pointer address, which is ASLR-non-deterministic).
        let mut seed: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
        for byte in label.as_bytes() {
            seed ^= *byte as u64;
            seed = seed.wrapping_mul(0x0000_0100_0000_01b3); // FNV prime
        }
        seed ^= m as u64 * 0x9e37_79b9_7f4a_7c15;
        seed ^= n as u64 * 0x7f4a_7c15_517c_c1cc;
        seed ^= k as u64 * 0x517c_c1cc_9e37_79b9;
        let mut rng = Rng::new(seed);
        let x_host: Vec<bf16> = (0..m * k).map(|_| bf16::from_f32(rng.normal())).collect();
        let w_host: Vec<u8> = (0..n * k).map(|_| f32_to_fp8_e4m3(rng.normal())).collect();
        let w_scales_host: Vec<f32> = (0..scale_rows * scale_cols)
            .map(|_| rng.uniform_range(0.5, 2.0))
            .collect();
        // DeepGEMM quantizes activations to FP8 block-scaled (4×128 tiles); GEMV keeps
        // them in BF16. For an apples-to-apples numeric comparison, degrade the GEMV
        // activations through the same FP8 quantization + dequantization so both paths
        // share one precision floor. Performance timing uses the original BF16 x for
        // both (DeepGEMM's pack_quantize times its own quantize; GEMV times raw BF16).
        let x_gemv_host = quantize_dequantize_bf16_fp8(&x_host, m, k);
        let x = ctx.stream.clone_htod(&x_host)?;
        let x_gemv = ctx.stream.clone_htod(&x_gemv_host)?;
        let w = ctx.stream.clone_htod(&w_host)?;
        let w_scales = ctx.stream.clone_htod(&w_scales_host)?;
        let mut out = ctx.stream.alloc_zeros::<bf16>(m * n)?;
        ctx.sync()?;

        // SAFETY: all device allocations below cover the exact M/N/K launch shape.
        let gemv = timed(ctx, iters, samples, || unsafe {
            let (wp, _gw) = w.device_ptr(&ctx.stream);
            let (sp, _gs) = w_scales.device_ptr(&ctx.stream);
            let (xp, _gx) = x_gemv.device_ptr(&ctx.stream);
            let (op, _go) = out.device_ptr_mut(&ctx.stream);
            ffi::gemv_fp8_block_scaled_batch_cuda(
                wp as *const u8,
                sp as *const f32,
                xp as *const ffi::Half,
                op as *mut ffi::Half,
                m as i32,
                n as i32,
                k as i32,
                scale_rows as i32,
                scale_cols as i32,
                FP8_BLOCK as i32,
                FP8_BLOCK as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
            Ok(())
        })?;
        let reference_out = ctx.stream.clone_dtoh(&out)?;

        if !deepgemm {
            return Ok(None);
        }
        let scale_stride_m = m.div_ceil(4) * 4;
        let input_fp8 = ctx.stream.alloc_zeros::<u8>(m * k)?;
        let input_scales = ctx.stream.alloc_zeros::<f32>(scale_stride_m * scale_cols)?;
        let active_experts = ctx.stream.clone_htod(&[0i32])?;
        let active_offsets = ctx.stream.clone_htod(&[0i32])?;
        let active_counts = ctx.stream.clone_htod(&[i32::try_from(m)?])?;
        ctx.sync()?;

        // SAFETY: input, FP8 output, scales, and metadata match the supplied shape.
        let pack = timed(ctx, iters, samples, || unsafe {
            moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                cache_ptr(&x, ctx),
                cache_ptr(&input_fp8, ctx),
                cache_ptr(&input_scales, ctx),
                cache_ptr(&active_experts, ctx),
                cache_ptr(&active_offsets, ctx),
                cache_ptr(&active_counts, ctx),
                1,
                m,
                k,
                scale_stride_m,
                ctx.stream.cu_stream(),
            )
        })?;

        // SAFETY: packed input, weight, scales, and output cover this exact cell.
        let gemm = timed(ctx, iters, samples, || unsafe {
            moe::dsv4_deepgemm_fp8_gemm_nt(
                cache_ptr(&input_fp8, ctx),
                cache_ptr(&input_scales, ctx),
                cache_ptr(&w, ctx),
                cache_ptr(&w_scales, ctx),
                cache_ptr(&out, ctx),
                m,
                n,
                k,
                scale_stride_m,
                ctx.stream.cu_stream(),
            )
        })?;
        let candidate_out = ctx.stream.clone_dtoh(&out)?;
        let numeric = numeric_delta(&reference_out, &candidate_out);
        let candidate_cuda_us: Vec<f64> = pack
            .cuda_us
            .iter()
            .zip(&gemm.cuda_us)
            .map(|(pack, gemm)| pack + gemm)
            .collect();
        let launches = 1 + iters * samples;
        Ok(Some(json!({
            "label": label,
            "position": label,
            "m": m,
            "n": n,
            "k": k,
            "dtype": "bf16xfp8_e4m3_block128->bf16",
            "layout": "row_major_activation+nt_weight",
            "reference": { "id": "cuda.qwen.fp8_gemv", "cuda_us": gemv.cuda_us },
            "candidate": {
                "id": "cuda.qwen.fp8_pack_deepgemm",
                "cuda_us": candidate_cuda_us,
                "components": { "pack": pack.cuda_us, "gemm": gemm.cuda_us },
            },
            "numeric": numeric,
            "engagement": {
                "reference_launches": launches,
                "candidate_pack_launches": launches,
                "candidate_gemm_launches": launches,
            },
        })))
    }

    fn numeric_delta(reference: &[bf16], candidate: &[bf16]) -> NumericDelta {
        assert_eq!(reference.len(), candidate.len());
        let (mut max_abs, mut max_rel, mut squared_error) = (0.0f64, 0.0f64, 0.0f64);
        let mut max_tolerance_ratio = 0.0f64;
        let mut violations = 0usize;
        for (reference, candidate) in reference.iter().zip(candidate) {
            let reference = reference.to_f32() as f64;
            let candidate = candidate.to_f32() as f64;
            let abs = (candidate - reference).abs();
            max_abs = max_abs.max(abs);
            let rel = if reference == 0.0 {
                if abs == 0.0 { 0.0 } else { f64::MAX }
            } else {
                abs / reference.abs()
            };
            max_rel = max_rel.max(rel);
            squared_error += abs * abs;
            let allowed = NUMERIC_ABS_TOLERANCE + NUMERIC_REL_TOLERANCE * reference.abs();
            max_tolerance_ratio = max_tolerance_ratio.max(abs / allowed);
            if abs > allowed {
                violations += 1;
            }
        }
        NumericDelta {
            max_abs,
            max_rel,
            rmse: (squared_error / reference.len() as f64).sqrt(),
            max_abs_tolerance: NUMERIC_ABS_TOLERANCE,
            max_rel_tolerance: NUMERIC_REL_TOLERANCE,
            tolerance_mode: "abs+rel*abs(reference)",
            max_tolerance_ratio,
            violations,
            passed: violations == 0,
        }
    }

    /// xorshift64* — deterministic, no-std, seedable.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            })
        }

        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        /// Uniform f32 in [0, 1).
        fn uniform(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }

        fn uniform_range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + self.uniform() * (hi - lo)
        }

        /// Standard normal via Box-Muller.
        fn normal(&mut self) -> f32 {
            let u1 = self.uniform().max(f32::EPSILON);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        }
    }

    /// Decode FP8 E4M3 byte to f32 (host-side reference).
    fn fp8_e4m3_to_f32(byte: u8) -> f32 {
        let sign = (byte >> 7) & 1;
        let exp = ((byte >> 3) & 0x0F) as i32;
        let mant = byte & 0x07;
        if exp == 0 && mant == 0 {
            return 0.0;
        }
        if exp == 15 {
            // e4m3fn has NO infinities: 0x7F/0xFF are NaN, everything else in
            // this exponent is finite (max 448 — the same bound the encoder
            // below clips to). The canonical decoder is
            // infer-gguf/src/dequant.rs::fp8_e4m3_to_f32.
            if mant == 7 {
                return f32::NAN;
            }
            let val = (1.0 + mant as f32 / 8.0) * 256.0; // 2^(15-7) = 256
            return if sign != 0 { -val } else { val };
        }
        let val = if exp == 0 {
            (mant as f32 / 8.0) * (-6.0f32).exp2()
        } else {
            (1.0 + mant as f32 / 8.0) * ((exp - 7) as f32).exp2()
        };
        if sign != 0 { -val } else { val }
    }

    /// Round-half-to-even (banker's rounding) to match device __nv_fp8_e4m3.
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

    /// Encode an f32 to FP8 E4M3. Clips to ±448 (max finite). Rounds to nearest even.
    fn f32_to_fp8_e4m3(value: f32) -> u8 {
        if !value.is_finite() || value == 0.0 {
            return 0;
        }
        let sign = if value < 0.0 { 1u8 } else { 0u8 };
        let abs_val = value.abs();
        // Max finite E4M3 = 448 = (1+6/8)*2^8
        if abs_val >= 448.0 {
            return if sign != 0 { 0xfe } else { 0x7e };
        }
        let ln2 = abs_val.log2();
        let exp_unbiased = ln2.floor() as i32;
        let exp_rebiased = exp_unbiased + 7; // E4M3 bias = 7
        if exp_rebiased > 15 {
            return if sign != 0 { 0xfe } else { 0x7e };
        }
        if exp_rebiased <= 0 {
            // Subnormal: exp field = 0, mant = round(abs_val / 2^-6 * 8)
            // Smallest subnormal: 2^-6 * 1/8 = 2^-9 ≈ 0.00195
            let scaled = abs_val * (6.0f32).exp2() * 8.0; // abs_val / 2^-6 * 8
            let mant = rint_ne(scaled);
            if mant == 0 {
                return 0;
            }
            if mant >= 8 {
                // Rounds up to smallest normal (exp=1, mant=0)
                let byte = 1u8 << 3;
                return if sign != 0 { byte | 0x80 } else { byte };
            }
            return if sign != 0 { mant | 0x80 } else { mant };
        }
        // Normal: exp in [1, 14]
        let mant_val = abs_val * ((-exp_unbiased) as f32).exp2(); // = abs_val / 2^exp_unbiased
        let mant_frac = mant_val - 1.0; // fractional part, in [0, 1)
        let mant = rint_ne(mant_frac * 8.0);
        let (exp_final, mant_final) = if mant >= 8 {
            (exp_rebiased + 1, 0u8)
        } else {
            (exp_rebiased, mant)
        };
        if exp_final > 15 {
            return if sign != 0 { 0xfe } else { 0x7e };
        }
        let byte = ((exp_final as u8) << 3) | mant_final;
        if sign != 0 { byte | 0x80 } else { byte }
    }

    /// Quantize BF16 activations to FP8 block-scaled then dequantize back to BF16.
    /// Matches DeepGEMM pack_quantize's per-row, per-128-col-block scaling and uses
    /// the FP8 E4M3 grid for rounding (not integer), so the GEMV reference and
    /// DeepGEMM share one activation precision floor.
    fn quantize_dequantize_bf16_fp8(x: &[bf16], m: usize, k: usize) -> Vec<bf16> {
        const MAX_FP8: f32 = 448.0; // max finite E4M3 value
        let mut out = vec![bf16::from_f32(0.0); m * k];
        let tile_cols = FP8_BLOCK;
        for r in 0..m {
            for tile_c in (0..k).step_by(tile_cols) {
                let w = tile_cols.min(k - tile_c);
                let mut max_abs = 0.0f32;
                for dc in 0..w {
                    max_abs = max_abs.max(x[r * k + tile_c + dc].to_f32().abs());
                }
                let scale = if max_abs > 0.0 {
                    max_abs / MAX_FP8
                } else {
                    1.0
                };
                for dc in 0..w {
                    let val = x[r * k + tile_c + dc].to_f32();
                    let fp8_byte = f32_to_fp8_e4m3(val / scale);
                    let dequant = fp8_e4m3_to_f32(fp8_byte) * scale;
                    out[r * k + tile_c + dc] = bf16::from_f32(dequant);
                }
            }
        }
        out
    }

    fn env_or_unreported(name: &str) -> String {
        std::env::var(name).unwrap_or_else(|_| "unreported".to_string())
    }

    fn git_source() -> (String, bool) {
        let commit = command_last_line("git", &["rev-parse", "HEAD"]);
        let dirty = Command::new("git")
            .args(["status", "--porcelain", "--untracked-files=no"])
            .output()
            .map_or(true, |output| {
                !output.status.success() || !output.stdout.is_empty()
            });
        (commit, dirty)
    }

    fn command_last_line(program: &str, args: &[&str]) -> String {
        Command::new(program)
            .args(args)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|output| output.lines().next_back().map(str::to_owned))
            .filter(|line| !line.is_empty())
            .unwrap_or_else(|| "unreported".to_string())
    }

    fn cuda_driver_version() -> String {
        let mut version = 0;
        // SAFETY: the driver writes one integer to this live stack slot.
        let result = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut version) };
        if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            version.to_string()
        } else {
            "unreported".to_string()
        }
    }

    fn timed(
        ctx: &DeviceContext,
        iters: usize,
        samples: usize,
        mut f: impl FnMut() -> Result<()>,
    ) -> Result<Sample> {
        f()?;
        ctx.sync()?;
        let start = ctx.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let stop = ctx.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let mut sample = Sample::default();
        for _ in 0..samples {
            start.record(&ctx.stream)?;
            for _ in 0..iters {
                f()?;
            }
            stop.record(&ctx.stream)?;
            stop.synchronize()?;
            sample
                .cuda_us
                .push(start.elapsed_ms(&stop)? as f64 * 1e3 / iters as f64);
        }
        Ok(sample)
    }
}
