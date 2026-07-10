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
    use std::time::Instant;

    use anyhow::Result;
    use cuda_kernels::ffi;
    use cuda_kernels::moe;
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::tensor::cache_ptr;
    use cudarc::driver::sys::CUevent_flags;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    const FP8_BLOCK: usize = 128;
    const M_SWEEP: &[usize] = &[1, 2, 4, 8, 16, 17, 32];
    // (label, N=weight rows, K=weight cols) — Qwen3.6-27B dense shapes.
    const SHAPES: &[(&str, usize, usize)] = &[
        ("ffn_gate_up", 17408, 5120),
        ("ffn_down", 5120, 17408),
        ("attn_sq", 5120, 5120),
    ];

    #[derive(Default, Clone, Copy)]
    struct Sample {
        host_ms: f64,
        cuda_ms: f64,
    }

    pub(super) fn run() -> Result<()> {
        let iters: usize = std::env::var("ARLE_SMALLM_PROBE_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200);
        let ctx = DeviceContext::new()?;
        let cc = ctx.compute_capability();
        eprintln!(
            "[smallm-probe] device={} cc={}.{} iters={iters}",
            ctx.ordinal(),
            cc.0,
            cc.1
        );
        let deepgemm = match moe::dsv4_deepgemm_native_preflight() {
            Ok(report) => {
                eprintln!("[smallm-probe] deepgemm preflight: {report}");
                true
            }
            Err(err) => {
                eprintln!("[smallm-probe] deepgemm UNAVAILABLE: {err}");
                false
            }
        };

        for &(label, n, k) in SHAPES {
            let weight_gb = (n * k) as f64 / 1e9;
            copy_ceiling(&ctx, label, n, k, iters, weight_gb)?;
            for &m in M_SWEEP {
                probe_one(&ctx, label, m, n, k, iters, weight_gb, deepgemm)?;
            }
        }
        Ok(())
    }

    /// DtoD copy of the weight buffer: measured copy rate X GB/s moves 2X
    /// bytes/s through HBM — the achievable-bandwidth ceiling for a
    /// streaming weight pass on this device (floor = weight_gb / (2X)).
    fn copy_ceiling(
        ctx: &DeviceContext,
        label: &str,
        n: usize,
        k: usize,
        iters: usize,
        weight_gb: f64,
    ) -> Result<()> {
        let src = ctx.stream.alloc_zeros::<u8>(n * k)?;
        let mut dst = ctx.stream.alloc_zeros::<u8>(n * k)?;
        ctx.sync()?;
        let s = timed(ctx, iters, || {
            ctx.stream.memcpy_dtod(&src, &mut dst)?;
            Ok(())
        })?;
        let hbm_gbps = 2.0 * weight_gb / (s.cuda_ms / 1e3);
        eprintln!(
            "[smallm-probe] shape={label} route=dtod_ceiling n={n} k={k} cuda_us={:.1} \
             hbm_gbps={hbm_gbps:.0} read_floor_us={:.1}",
            s.cuda_ms * 1e3,
            weight_gb / hbm_gbps * 1e6,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn probe_one(
        ctx: &DeviceContext,
        label: &str,
        m: usize,
        n: usize,
        k: usize,
        iters: usize,
        weight_gb: f64,
        deepgemm: bool,
    ) -> Result<()> {
        let scale_rows = n.div_ceil(FP8_BLOCK);
        let scale_cols = k.div_ceil(FP8_BLOCK);
        // Deterministic non-zero fill (zeros hide decode/accumulate costs).
        let x_host: Vec<bf16> = (0..m * k)
            .map(|i| bf16::from_f32(((i % 61) as f32 - 30.0) / 61.0))
            .collect();
        // Skip 0x7F/0xFF (e4m3 NaN encodings) so outputs stay finite.
        let w_host: Vec<u8> = (0..n * k)
            .map(|i| {
                let v = ((i * 37 + 13) % 256) as u8;
                if v & 0x7f == 0x7f { v ^ 0x08 } else { v }
            })
            .collect();
        let x = ctx.stream.clone_htod(&x_host)?;
        let w = ctx.stream.clone_htod(&w_host)?;
        let w_scales = ctx
            .stream
            .clone_htod(&vec![1.0f32; scale_rows * scale_cols])?;
        let mut out = ctx.stream.alloc_zeros::<bf16>(m * n)?;
        ctx.sync()?;

        let gemv = timed(ctx, iters, || unsafe {
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
        report(label, "gemv", m, n, k, weight_gb, gemv);

        if m == 1 {
            let mut probe_out = ctx.stream.alloc_zeros::<f32>(n)?;
            for mode in [0i32, 1] {
                let s = timed(ctx, iters, || unsafe {
                    let (wp, _gw) = w.device_ptr(&ctx.stream);
                    let (op, _go) = probe_out.device_ptr_mut(&ctx.stream);
                    ffi::gemv_fp8_wread_probe_cuda(
                        wp as *const u8,
                        op as *mut f32,
                        n as i32,
                        k as i32,
                        mode,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                    Ok(())
                })?;
                let route = if mode == 0 {
                    "wread_raw"
                } else {
                    "wread_decode"
                };
                report(label, route, m, n, k, weight_gb, s);
            }
        }

        if !deepgemm {
            return Ok(());
        }
        let scale_stride_m = m.div_ceil(4) * 4;
        let input_fp8 = ctx.stream.alloc_zeros::<u8>(m * k)?;
        let input_scales = ctx.stream.alloc_zeros::<f32>(scale_stride_m * scale_cols)?;
        let active_experts = ctx.stream.clone_htod(&[0i32])?;
        let active_offsets = ctx.stream.clone_htod(&[0i32])?;
        let active_counts = ctx.stream.clone_htod(&[i32::try_from(m)?])?;
        ctx.sync()?;

        let pack = timed(ctx, iters, || unsafe {
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
        report(label, "dg_pack", m, n, k, weight_gb, pack);

        let gemm = timed(ctx, iters, || unsafe {
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
        report(label, "dg_gemm", m, n, k, weight_gb, gemm);
        Ok(())
    }

    fn report(label: &str, route: &str, m: usize, n: usize, k: usize, weight_gb: f64, s: Sample) {
        let gbps = weight_gb / (s.cuda_ms / 1e3);
        eprintln!(
            "[smallm-probe] shape={label} route={route} m={m} n={n} k={k} \
             cuda_us={:.1} host_us={:.1} weight_gbps={gbps:.0}",
            s.cuda_ms * 1e3,
            s.host_ms * 1e3,
        );
    }

    fn timed(
        ctx: &DeviceContext,
        iters: usize,
        mut f: impl FnMut() -> Result<()>,
    ) -> Result<Sample> {
        f()?;
        ctx.sync()?;
        let start = ctx.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let stop = ctx.ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        start.record(&ctx.stream)?;
        let host_t0 = Instant::now();
        for _ in 0..iters {
            f()?;
        }
        let host_ms_total = host_t0.elapsed().as_secs_f64() * 1000.0;
        stop.record(&ctx.stream)?;
        stop.synchronize()?;
        let cuda_ms_total = start.elapsed_ms(&stop)? as f64;
        Ok(Sample {
            host_ms: host_ms_total / iters as f64,
            cuda_ms: cuda_ms_total / iters as f64,
        })
    }
}
