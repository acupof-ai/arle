//! Marlin fp4 GEMM correctness: GPU kernel output vs CPU reference at the
//! Qwen3.6-27B dense-MLP shapes, with realistic E2M1/E4M3 distributions.
//!
//! The existing tests (test_cuda_marlin_fp4_share.rs) compare two GPU code
//! paths against each other at 128x128 / 256x512. This example compares the
//! serving GEMM against a CPU ground truth at the model's actual shapes,
//! sampling a subset of output rows to keep the reference tractable.
//!
//! Run on a pod: `INFER_CUDA_DEVICE=<free-gpu> target/release/examples/marlin_fp4_correctness`

fn main() -> anyhow::Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("marlin_fp4_correctness is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
    #[cfg(feature = "cuda")]
    real::run()
}

#[cfg(feature = "cuda")]
mod real {
    use anyhow::{Result, ensure};
    use cuda_kernels::ffi;
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::tensor::DeviceMatrix;
    use cuda_kernels::tensor::{e2m1_to_f32, e4m3_to_f32};
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use half::bf16;

    /// (label, N, K) — the model's actual projection shapes.
    const SHAPES: &[(&str, usize, usize)] = &[
        ("mlp gate_up", 34816, 5120),
        ("mlp down", 5120, 17408),
        ("attn qkv", 14336, 5120),
        ("attn o", 5120, 5120),
    ];

    const MS: &[usize] = &[1, 16, 32, 128, 512];
    const GROUP: usize = 16;
    /// Check this many output rows in the CPU reference.
    const CHECK_ROWS: usize = 64;

    struct Lcg(u64);
    impl Lcg {
        fn next_u8(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.0 >> 33) as u8
        }
    }

    /// CPU reference for a subset of output rows.
    fn cpu_ref_rows(
        packed: &[u8],
        scales: &[u8],
        global_scale: f32,
        x: &[bf16],
        m: usize,
        k: usize,
        rows: &[usize],
    ) -> Vec<f32> {
        let scale_cols = k / GROUP;
        let mut out = Vec::with_capacity(m * rows.len());
        for &ni in rows {
            for mi in 0..m {
                let mut acc = 0f32;
                for ki in 0..k {
                    let byte = packed[ni * (k / 2) + ki / 2];
                    let nibble = if ki % 2 == 0 {
                        byte & 0xF
                    } else {
                        (byte >> 4) & 0xF
                    };
                    let w = e2m1_to_f32(nibble);
                    let s = e4m3_to_f32(scales[ni * scale_cols + ki / GROUP]);
                    acc += x[mi * k + ki].to_f32() * w * s * global_scale;
                }
                out.push(acc);
            }
        }
        out
    }

    pub(super) fn run() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let sms = ctx.sm_count() as i32;
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
            "Marlin fp4 correctness: GPU vs CPU reference, build={}\n",
            cuda_kernels::KERNEL_BUILD_ID
        );
        println!(
            "{:<14}{:>7}{:>6}{:>10}{:>10}{:>8}",
            "shape", "N", "M", "max_rel", "mean_rel", "status"
        );

        let mut any_fail = false;
        for &(label, n, k) in SHAPES {
            let seed = 0x5eed_1234u64.wrapping_add(n as u64);
            let mut rng = Lcg(seed);
            let packed: Vec<u8> = (0..n * k / 2).map(|_| rng.next_u8()).collect();
            let scales: Vec<u8> = (0..n * k / GROUP)
                .map(|_| {
                    let exp = 2 + (rng.next_u8() % 7) as i32;
                    let mant = rng.next_u8() & 0x07;
                    (((exp + 7) as u8) << 3) | mant
                })
                .collect();
            let global_scale = 1.0f32;

            let mut matrix = DeviceMatrix::from_fp4_e2m1_group(
                &ctx,
                &packed,
                &scales,
                &[global_scale],
                None,
                n,
                k,
                GROUP,
            )?;
            matrix.repack_for_marlin_fp4(&ctx)?;
            ensure!(
                matrix.marlin_packed.is_some() && matrix.marlin_scales.is_some(),
                "NVFP4 repack declined {label} {n}x{k}"
            );

            let (p4, _g1) = matrix
                .marlin_packed
                .as_ref()
                .unwrap()
                .device_ptr(&ctx.stream);
            let (g4, _g2) = matrix
                .marlin_scales
                .as_ref()
                .unwrap()
                .device_ptr(&ctx.stream);
            let s4 = p4 + (n * k / 2) as u64;

            // Deterministic subset of output rows, spread across N.
            let check_rows: Vec<usize> = (0..CHECK_ROWS).map(|i| i * (n / CHECK_ROWS)).collect();

            for &m in MS {
                let x: Vec<bf16> = (0..m * k)
                    .map(|i| bf16::from_f32(((i % 13) as f32 - 6.0) * 0.125))
                    .collect();
                let x_dev = ctx.stream.memcpy_stod(&x)?;
                let mut out_dev = ctx.stream.alloc_zeros::<bf16>(m * n)?;
                let (x_ptr, _gx) = x_dev.device_ptr(&ctx.stream);
                {
                    let (out_ptr, _go) = out_dev.device_ptr_mut(&ctx.stream);
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
                        .result()?;
                    }
                }
                let gpu_out: Vec<bf16> = ctx.stream.clone_dtoh(&out_dev)?;
                ctx.sync()?;

                let cpu = cpu_ref_rows(&packed, &scales, global_scale, &x, m, k, &check_rows);

                let mut max_rel = 0f32;
                let mut sum_rel = 0f32;
                let scale = cpu.iter().fold(0f32, |a, &b| a.max(b.abs())).max(1e-6);
                for (idx, &ni) in check_rows.iter().enumerate() {
                    for mi in 0..m {
                        let g = gpu_out[mi * n + ni].to_f32();
                        let c = cpu[idx * m + mi];
                        let rel = (g - c).abs() / scale;
                        max_rel = max_rel.max(rel);
                        sum_rel += rel;
                    }
                }
                let count = (check_rows.len() * m) as f32;
                let mean_rel = sum_rel / count;
                let ok = max_rel < 1e-2;
                if !ok {
                    any_fail = true;
                }
                println!(
                    "{label:<14}{n:>7}{m:>6}{max_rel:>10.4}{mean_rel:>10.6}{:>8}",
                    if ok { "PASS" } else { "FAIL" }
                );
            }
        }

        if any_fail {
            anyhow::bail!("Marlin fp4 correctness check FAILED");
        }
        println!("\nAll shapes passed.");
        Ok(())
    }
}
