//! Marlin byte-rate probe: NVFP4 (kFE2M1f, group 16) vs per-channel FP8
//! (kFE4M3fn) at the Qwen3.8-27B dense-MLP shapes, over an M sweep.
//!
//! Standalone kernel harness — NO engine, NO serve, no checkpoint. Synthetic
//! weights, so it starts in a second and an `ncu` run over it profiles only the
//! two Marlin kernels.
//!
//! Why it exists: the c=16 per-op profile has NVFP4 `dense_ffn` moving 10.56 GB
//! in 9.201 ms (1.15 TB/s) while the SAME kernel family on per-channel FP8
//! weights moves in_proj/out_proj/qkv at 1.9-2.0 TB/s. Both are weight-bound
//! reads on one H20, so the byte rate should match; it does not. This isolates
//! that gap from the engine.
//!
//! Run on a pod: `INFER_CUDA_DEVICE=<free-gpu> target/release/examples/marlin_fp4_probe`
//! Under ncu: `ncu --set full -k regex:Marlin -c 8 target/release/examples/marlin_fp4_probe`

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut seed = 0x5eed_1234u64;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--kernel-build-id" => {
                println!("{}", cuda_kernels::KERNEL_BUILD_ID);
                return Ok(());
            }
            "--seed" => {
                seed = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--seed needs a value"))?
                    .parse()?;
            }
            other => anyhow::bail!("unknown arg: {other}"),
        }
    }
    real::run(seed)
}

#[cfg(not(feature = "cuda"))]
mod real {
    pub(super) fn run(_seed: u64) -> anyhow::Result<()> {
        eprintln!("marlin_fp4_probe is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::{Result, anyhow, ensure};
    use cuda_kernels::ffi;
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::tensor::DeviceMatrix;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    /// (label, N = weight rows / output dim, K = weight cols / contraction).
    /// hidden 5120, intermediate 17408; gate and up load row-fused
    /// (`qwen35_load.rs:541`), so the resident matrix is 2x17408 rows.
    const SHAPES: &[(&str, usize, usize)] =
        &[("mlp gate_up", 34816, 5120), ("mlp down", 5120, 17408)];

    /// Decode batch, then the prefill chunk sizes: `chunked_prefill_size` is
    /// 2048 and a 32K prompt walks it 16 times, so M=2048 is where a long-agent
    /// workload spends its time.
    const MS: &[usize] = &[1, 16, 512, 2048];
    const GROUP: usize = 16;
    const ITERS: usize = 20;

    /// Deterministic bytes; a seeded LCG so two runs of the probe compare.
    struct Lcg(u64);
    impl Lcg {
        fn next_u8(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.0 >> 33) as u8
        }
    }

    fn fp4_matrix(ctx: &DeviceContext, n: usize, k: usize, seed: u64) -> Result<DeviceMatrix> {
        let mut rng = Lcg(seed);
        let packed: Vec<u8> = (0..n * k / 2).map(|_| rng.next_u8()).collect();
        // E4M3 group scales: keep the exponent mid-range so no group underflows
        // to zero and none reaches the 0xFF NaN encoding.
        let scales: Vec<u8> = (0..n * k / GROUP)
            .map(|_| 0x38 | (rng.next_u8() & 0x07))
            .collect();
        let mut m =
            DeviceMatrix::from_fp4_e2m1_group(ctx, &packed, &scales, &[1.0f32], None, n, k, GROUP)?;
        m.repack_for_marlin_fp4(ctx)?;
        Ok(m)
    }

    fn fp8_matrix(ctx: &DeviceContext, n: usize, k: usize, seed: u64) -> Result<DeviceMatrix> {
        // seed+1: distinct byte stream from the FP4 matrix built from `seed` in
        // the same run; the decline boundary below uses seed+2.
        let mut rng = Lcg(seed.wrapping_add(1));
        // Same 0xFF-free band as the FP4 group scales above.
        let weight: Vec<u8> = (0..n * k).map(|_| 0x38 | (rng.next_u8() & 0x07)).collect();
        let scale: Vec<f32> = (0..n).map(|_| 1.0 / 448.0).collect();
        let mut m = DeviceMatrix::from_fp8_block_scaled(ctx, &weight, &scale, n, k, 1, k)?;
        m.repack_for_marlin_fp8(ctx)?;
        ensure!(
            m.marlin_packed.is_some() && m.marlin_scales.is_some(),
            "per-channel FP8 repack declined {n}x{k}"
        );
        Ok(m)
    }

    /// Median-of-ITERS device time for one Marlin call, in milliseconds.
    fn time_ms(ctx: &DeviceContext, mut call: impl FnMut() -> Result<()>) -> Result<f64> {
        for _ in 0..10 {
            call()?;
        }
        ctx.stream.synchronize()?;
        let start = ctx
            .ctx
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let stop = ctx
            .ctx
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        start.record(&ctx.stream)?;
        for _ in 0..ITERS {
            call()?;
        }
        stop.record(&ctx.stream)?;
        stop.synchronize()?;
        Ok(start.elapsed_ms(&stop)? as f64 / ITERS as f64)
    }

    pub(super) fn run(seed: u64) -> Result<()> {
        let ctx = DeviceContext::new()?;
        let sms = ctx.sm_count() as i32;
        // SAFETY: pure size queries (arithmetic on sms), no device work.
        let (c_tmp_floats, ws_ints) = unsafe {
            (
                ffi::marlin_c_tmp_floats(64, sms) as usize,
                ffi::marlin_workspace_ints(sms) as usize,
            )
        };
        let c_tmp = ctx.stream.alloc_zeros::<f32>(c_tmp_floats)?;
        let workspace = ctx.stream.alloc_zeros::<i32>(ws_ints)?;
        let (c_tmp_ptr, _gc) = c_tmp.device_ptr(&ctx.stream);
        let (ws_ptr, _gw) = workspace.device_ptr(&ctx.stream);
        let stream = ctx.stream.cu_stream();

        println!(
            "Marlin byte rate, {sms} SMs, {ITERS} iters/point, seed={seed:#x}, build={}\n",
            cuda_kernels::KERNEL_BUILD_ID
        );
        println!(
            "{:<14}{:>7}{:>4}{:>10}{:>10}{:>10}{:>10}{:>10}",
            "shape", "N", "M", "fp4 ms", "fp4 TB/s", "fp8 ms", "fp8 TB/s", "fp4/fp8"
        );

        let mut any_fail = false;
        for &(label, n, k) in SHAPES {
            let w4 = fp4_matrix(&ctx, n, k, seed)?;
            ensure!(
                w4.marlin_packed.is_some() && w4.marlin_scales.is_some(),
                "NVFP4 repack declined {label} {n}x{k}"
            );
            let w8 = fp8_matrix(&ctx, n, k, seed)?;
            // FP4 group scales sit in the tail of the packed allocation
            // (`repack_for_marlin_fp4`); FP8 keeps its BF16 scales separate.
            let (p4, _g1) = w4.marlin_packed.as_ref().unwrap().device_ptr(&ctx.stream);
            let (g4, _g2) = w4.marlin_scales.as_ref().unwrap().device_ptr(&ctx.stream);
            let s4 = p4 + (n * k / 2) as u64;
            let (p8, _g3) = w8.marlin_packed.as_ref().unwrap().device_ptr(&ctx.stream);
            let (s8, _g4) = w8.marlin_scales.as_ref().unwrap().device_ptr(&ctx.stream);
            // 4 bits + one E4M3 scale per 16 vs 8 bits + one BF16 scale per row.
            let bytes4 = (n * k) as f64 * (0.5 + 1.0 / GROUP as f64);
            let bytes8 = (n * k) as f64;

            for &m in MS {
                let x = ctx.stream.alloc_zeros::<bf16>(m * k)?;
                let mut out = ctx.stream.alloc_zeros::<bf16>(m * n)?;
                let (x_ptr, _gx) = x.device_ptr(&ctx.stream);
                let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);

                let ms4 = time_ms(&ctx, || {
                    // SAFETY: every pointer is a live allocation sized above for
                    // exactly these dims.
                    unsafe {
                        ffi::marlin_fp4_gemm_cuda(
                            x_ptr as *const ffi::Half,
                            p4 as *const u32,
                            s4 as *const u8,
                            g4 as *const u16,
                            out_ptr as *mut ffi::Half,
                            c_tmp_ptr as *mut f32,
                            ws_ptr as *mut i32,
                            m as i32,
                            n as i32,
                            k as i32,
                            GROUP as i32,
                            stream,
                        )
                        .result()
                        .map_err(|e| anyhow!("NVFP4 Marlin GEMM failed: {e}"))
                    }
                })?;
                let ms8 = time_ms(&ctx, || {
                    // SAFETY: as above.
                    unsafe {
                        ffi::marlin_fp8_gemm_cuda(
                            x_ptr as *const ffi::Half,
                            p8 as *const u32,
                            s8 as *const ffi::Half,
                            out_ptr as *mut ffi::Half,
                            c_tmp_ptr as *mut f32,
                            ws_ptr as *mut i32,
                            m as i32,
                            n as i32,
                            k as i32,
                            stream,
                        )
                        .result()
                        .map_err(|e| anyhow!("FP8 Marlin GEMM failed: {e}"))
                    }
                })?;
                let tb4 = bytes4 / ms4 / 1e9;
                let tb8 = bytes8 / ms8 / 1e9;
                println!(
                    "{label:<14}{n:>7}{m:>4}{ms4:>10.3}{tb4:>10.2}{ms8:>10.3}{tb8:>10.2}{:>10.2}",
                    tb4 / tb8
                );
            }
        }

        // Repack-decline boundary: N not %64 must leave the source resident.
        let (dn, dk) = (96usize, 5120usize);
        let dm = fp4_matrix(&ctx, dn, dk, seed.wrapping_add(2))?;
        let declined =
            dm.marlin_packed.is_none() && dm.qweight_u8.is_some() && dm.qscale_fp8.is_some();
        any_fail |= !declined;
        println!(
            "declined n%64 {dn}x{dk}: {}",
            if declined { "OK" } else { "FAIL" }
        );

        ensure!(!any_fail, "marlin_fp4_probe FAILED — see violations above");
        Ok(())
    }
}
