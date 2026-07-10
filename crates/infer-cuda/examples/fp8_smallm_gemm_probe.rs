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
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

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
    const BF16_PATTERN: &[f32] = &[
        1.0, 2.0, 0.5, 3.0, 0.75, 1.5, -1.0, -2.0, -0.5, -3.0, -0.75, -1.5,
    ];
    const FP8_E4M3_PATTERN: &[u8] = &[
        0x38, 0x40, 0x30, 0x48, 0x34, 0x3c, 0xb8, 0xc0, 0xb0, 0xc8, 0xb4, 0xbc,
    ];
    const SCALE_PATTERN: &[f32] = &[0.25, 0.5, 1.0, 2.0, 4.0];
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
        host_us: Vec<f64>,
        cuda_us: Vec<f64>,
    }

    impl Sample {
        fn mean_host_us(&self) -> f64 {
            self.host_us.iter().sum::<f64>() / self.host_us.len() as f64
        }

        fn mean_cuda_us(&self) -> f64 {
            self.cuda_us.iter().sum::<f64>() / self.cuda_us.len() as f64
        }
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
            let weight_gb = (n * k) as f64 / 1e9;
            copy_ceiling(&ctx, label, n, k, iters, samples, weight_gb)?;
            for &m in M_SWEEP {
                if let Some(measurement) =
                    probe_one(&ctx, label, m, n, k, iters, samples, weight_gb, deepgemm)?
                {
                    measurements.push(measurement);
                }
            }
        }
        let (commit, dirty) = git_source();
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let e2e_artifact_sha256 = std::env::var("ARLE_SMALLM_E2E_ARTIFACT")
            .ok()
            .and_then(|path| file_sha256(&path));
        let run_id = std::env::var("ARLE_SMALLM_RUN_ID")
            .unwrap_or_else(|_| format!("smallm-{}-{now}", &commit[..8.min(commit.len())]));
        let run = json!({
            "schema_version": "arle.operator-evidence/v1",
            "run_id": run_id,
            "source": { "commit": commit, "dirty": dirty },
            "product": {
                "binary_id": env_or_unreported("ARLE_SMALLM_BINARY_ID"),
                "bundle_id": env_or_unreported("ARLE_SMALLM_BUNDLE_ID"),
            },
            "operator_id": "qwen.fp8_dense_projection",
            "model_revision": env_or_unreported("ARLE_SMALLM_MODEL_REVISION"),
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
                "passed": e2e_artifact_sha256.is_some()
                    && std::env::var("ARLE_SMALLM_E2E_PASS").as_deref() == Ok("1"),
                "artifact_sha256": e2e_artifact_sha256,
            },
            "measurements": measurements,
        });
        println!("{}", serde_json::to_string_pretty(&run)?);
        Ok(())
    }

    /// DtoD copy of the weight buffer: measured copy rate X GB/s moves 2X
    /// bytes/s through HBM — the achievable-bandwidth ceiling for a
    /// streaming weight pass on this device (floor = weight_gb / (2X)).
    fn copy_ceiling(
        ctx: &DeviceContext,
        label: &'static str,
        n: usize,
        k: usize,
        iters: usize,
        samples: usize,
        weight_gb: f64,
    ) -> Result<()> {
        let src = ctx.stream.alloc_zeros::<u8>(n * k)?;
        let mut dst = ctx.stream.alloc_zeros::<u8>(n * k)?;
        ctx.sync()?;
        let s = timed(ctx, iters, samples, || {
            ctx.stream.memcpy_dtod(&src, &mut dst)?;
            Ok(())
        })?;
        let hbm_gbps = 2.0 * weight_gb / (s.mean_cuda_us() / 1e6);
        eprintln!(
            "[smallm-probe] shape={label} route=dtod_ceiling n={n} k={k} cuda_us={:.1} \
             hbm_gbps={hbm_gbps:.0} read_floor_us={:.1}",
            s.mean_cuda_us(),
            weight_gb / hbm_gbps * 1e6,
        );
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
        weight_gb: f64,
        deepgemm: bool,
    ) -> Result<Option<serde_json::Value>> {
        let scale_rows = n.div_ceil(FP8_BLOCK);
        let scale_cols = k.div_ceil(FP8_BLOCK);
        let x_host: Vec<bf16> = (0..m)
            .flat_map(|row| {
                (0..k).map(move |col| bf16::from_f32(BF16_PATTERN[pattern_index(row, col, 0x51)]))
            })
            .collect();
        let w_host: Vec<u8> = (0..n)
            .flat_map(|row| (0..k).map(move |col| FP8_E4M3_PATTERN[pattern_index(row, col, 0xa7)]))
            .collect();
        let x = ctx.stream.clone_htod(&x_host)?;
        let w = ctx.stream.clone_htod(&w_host)?;
        let w_scales_host: Vec<f32> = (0..scale_rows)
            .flat_map(|row| {
                (0..scale_cols).map(move |col| {
                    SCALE_PATTERN[pattern_index(row, col, 0x3d) % SCALE_PATTERN.len()]
                })
            })
            .collect();
        let w_scales = ctx.stream.clone_htod(&w_scales_host)?;
        let mut out = ctx.stream.alloc_zeros::<bf16>(m * n)?;
        ctx.sync()?;

        // SAFETY: all device allocations below cover the exact M/N/K launch shape.
        let gemv = timed(ctx, iters, samples, || unsafe {
            let (wp, _gw) = w.device_ptr(&ctx.stream);
            let (sp, _gs) = w_scales.device_ptr(&ctx.stream);
            let (xp, _gx) = x.device_ptr(&ctx.stream);
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
        report(label, "gemv", m, n, k, weight_gb, &gemv);
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
        report(label, "dg_pack", m, n, k, weight_gb, &pack);

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
        report(label, "dg_gemm", m, n, k, weight_gb, &gemm);
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

    fn report(label: &str, route: &str, m: usize, n: usize, k: usize, weight_gb: f64, s: &Sample) {
        let cuda_us = s.mean_cuda_us();
        let gbps = weight_gb / (cuda_us / 1e6);
        eprintln!(
            "[smallm-probe] shape={label} route={route} m={m} n={n} k={k} \
             cuda_us={cuda_us:.1} host_us={:.1} weight_gbps={gbps:.0}",
            s.mean_host_us(),
        );
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

    fn pattern_index(row: usize, col: usize, salt: usize) -> usize {
        let mixed = row.wrapping_mul(131)
            ^ col.wrapping_mul(313)
            ^ row.wrapping_mul(col.wrapping_add(17))
            ^ salt;
        (mixed ^ (mixed >> 3) ^ (mixed >> 7)) % BF16_PATTERN.len()
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

    fn file_sha256(path: &str) -> Option<String> {
        [
            ("sha256sum", vec![path]),
            ("shasum", vec!["-a", "256", path]),
        ]
        .into_iter()
        .find_map(|(program, args)| {
            Command::new(program)
                .args(args)
                .output()
                .ok()
                .filter(|output| output.status.success())
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .and_then(|output| output.split_whitespace().next().map(str::to_owned))
                .filter(|digest| digest.len() == 64)
        })
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
            let host_t0 = Instant::now();
            for _ in 0..iters {
                f()?;
            }
            let host_us = host_t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            stop.record(&ctx.stream)?;
            stop.synchronize()?;
            sample.host_us.push(host_us);
            sample
                .cuda_us
                .push(start.elapsed_ms(&stop)? as f64 * 1e3 / iters as f64);
        }
        Ok(sample)
    }
}
