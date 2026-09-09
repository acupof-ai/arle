//! `arle kernel <name>`: one CUDA kernel run alone against a CPU f32
//! reference. The registry is explicit — an unknown name lists the
//! registered kernels instead of dispatching.
//!
//! First two entries answer the open M=1 question
//! (`errors/2026-09-09-w4afp8-gemv-killed-item-dsv4-moe-only.md`):
//! `fp4-gemv` is the dense NVFP4 GEMV with no production caller since
//! dispatch convergence, `marlin-fp4-gemm` the serving arm.

use anyhow::{Result, bail, ensure};
use cuda_kernels::prelude::DeviceContext;
use cuda_kernels::quant_linear as cuda_ql;
use cuda_kernels::tensor::DeviceMatrix;
use half::bf16;
use infer_quant::{GROUP, cpu_ref_fp4};

const SEED: u64 = 0x5eed_1234;
const PASS_MAX_REL: f32 = 1e-2;

struct Lcg(u64);
impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 33) as u8
    }
}

/// "--shape M,N,K" -> (m, n, k).
fn parse_shape(s: &str) -> Result<(usize, usize, usize)> {
    let dims: Vec<&str> = s.split(',').collect();
    if dims.len() != 3 {
        bail!("--shape must be M,N,K (got {s:?})");
    }
    let dim = |d: &str| -> Result<usize> {
        d.trim()
            .parse::<usize>()
            .map_err(|_| anyhow::anyhow!("bad dim {d:?} in --shape {s:?}"))
    };
    Ok((dim(dims[0])?, dim(dims[1])?, dim(dims[2])?))
}

/// Synthetic NVFP4 weights + bf16 input, deterministic per (m, n, k) so both
/// kernels in the registry see identical tensors.
fn synth_fp4(m: usize, n: usize, k: usize) -> (Vec<u8>, Vec<u8>, f32, Vec<bf16>) {
    let mut rng = Lcg(SEED.wrapping_add(n as u64).wrapping_add((k as u64) << 32));
    let packed: Vec<u8> = (0..n * k / 2).map(|_| rng.next_u8()).collect();
    // E4M3 band 0x38..0x3f: exponent mid-range, no zero groups, no 0xFF NaN.
    let scales: Vec<u8> = (0..n * k / GROUP)
        .map(|_| 0x38 | (rng.next_u8() & 0x07))
        .collect();
    let global = 1.0f32;
    let x: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32(((i % 13) as f32 - 6.0) * 0.125))
        .collect();
    (packed, scales, global, x)
}

/// Max relative error over all outputs, normalized by the CPU reference's
/// magnitude (same convention as `marlin_fp4_correctness.rs`).
fn max_rel(gpu: &[bf16], cpu: &[f32]) -> f32 {
    let scale = cpu.iter().fold(0f32, |a, &b| a.max(b.abs())).max(1e-6);
    gpu.iter()
        .zip(cpu)
        .map(|(g, c)| (g.to_f32() - c).abs() / scale)
        .fold(0f32, f32::max)
}

struct Report {
    time_ms: f64,
    max_rel: f32,
}

type KernelFn = fn(&DeviceContext, usize, usize, usize, usize) -> Result<Report>;

/// The explicit registry. A kernel with no entry is reported as unknown by
/// `run_kernel_bench`.
const REGISTRY: &[(&str, KernelFn)] = &[
    ("fp4-gemv", run_fp4_gemv),
    ("marlin-fp4-gemm", run_marlin_fp4_gemm),
];

/// Mean device time over `iters` launches after 10 warmups.
fn time_ms(ctx: &DeviceContext, iters: usize, mut call: impl FnMut() -> Result<()>) -> Result<f64> {
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
    for _ in 0..iters {
        call()?;
    }
    stop.record(&ctx.stream)?;
    stop.synchronize()?;
    Ok(start.elapsed_ms(&stop)? as f64 / iters as f64)
}

/// A synthetic NVFP4 matrix on device plus the host tensors the CPU reference
/// needs. `matrix` and the host copies hold the same bytes.
struct SynthFp4 {
    matrix: DeviceMatrix,
    packed: Vec<u8>,
    scales: Vec<u8>,
    global: f32,
    x: Vec<bf16>,
}

fn synth_matrix(ctx: &DeviceContext, m: usize, n: usize, k: usize) -> Result<SynthFp4> {
    let (packed, scales, global, x) = synth_fp4(m, n, k);
    let matrix =
        DeviceMatrix::from_fp4_e2m1_group(ctx, &packed, &scales, &[global], None, n, k, GROUP)?;
    Ok(SynthFp4 {
        matrix,
        packed,
        scales,
        global,
        x,
    })
}

/// The dense NVFP4 M=1 GEMV (`quantized_gemv.cu:1193`): plain packed layout,
/// no Marlin repack. Zero production callers since dispatch convergence.
fn run_fp4_gemv(ctx: &DeviceContext, m: usize, n: usize, k: usize, iters: usize) -> Result<Report> {
    ensure!(
        m == 1,
        "fp4-gemv is M=1 only (got M={m}); use marlin-fp4-gemm for M>1"
    );
    let SynthFp4 {
        matrix,
        packed,
        scales,
        global,
        x,
    } = synth_matrix(ctx, m, n, k)?;
    let x_dev = ctx.stream.clone_htod(&x)?;
    let mut out_dev = ctx.stream.alloc_zeros::<bf16>(n)?;
    let weight = matrix.qweight_u8.as_ref().unwrap();
    let group_scales = matrix.qscale_fp8.as_ref().unwrap();
    let global_dev = matrix.scale_f32.as_ref().unwrap();
    let time_ms = time_ms(ctx, iters, || {
        cuda_ql::gemv_fp4_e2m1_group(
            ctx,
            weight,
            group_scales,
            global_dev,
            &x_dev,
            &mut out_dev,
            n,
            k,
            GROUP,
        )
    })?;
    let gpu: Vec<bf16> = ctx.stream.clone_dtoh(&out_dev)?;
    let cpu = cpu_ref_fp4(&packed, &scales, global, &x, m, n, k);
    Ok(Report {
        time_ms,
        max_rel: max_rel(&gpu, &cpu),
    })
}

/// The serving arm: Marlin tensor-core GEMM on the repacked layout.
fn run_marlin_fp4_gemm(
    ctx: &DeviceContext,
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
) -> Result<Report> {
    let SynthFp4 {
        mut matrix,
        packed,
        scales,
        global,
        x,
    } = synth_matrix(ctx, m, n, k)?;
    matrix.repack_for_marlin_fp4(ctx)?;
    ensure!(
        matrix.marlin_packed.is_some() && matrix.marlin_scales.is_some(),
        "NVFP4 repack declined {n}x{k}"
    );
    let sms = ctx.sm_count();
    let c_tmp = ctx
        .stream
        .alloc_zeros::<f32>(cuda_ql::marlin_c_tmp_floats(64, sms)?)?;
    let workspace = ctx
        .stream
        .alloc_zeros::<i32>(cuda_ql::marlin_workspace_ints(sms)?)?;
    let packed_dev = matrix.marlin_packed.as_ref().unwrap();
    let global_dev = matrix.marlin_scales.as_ref().unwrap();
    let x_dev = ctx.stream.clone_htod(&x)?;
    let mut out_dev = ctx.stream.alloc_zeros::<bf16>(m * n)?;
    let time_ms = time_ms(ctx, iters, || {
        cuda_ql::marlin_fp4_gemm(
            ctx,
            &x_dev,
            packed_dev,
            global_dev,
            &mut out_dev,
            &c_tmp,
            &workspace,
            m,
            n,
            k,
            GROUP,
        )
    })?;
    let gpu: Vec<bf16> = ctx.stream.clone_dtoh(&out_dev)?;
    let cpu = cpu_ref_fp4(&packed, &scales, global, &x, m, n, k);
    Ok(Report {
        time_ms,
        max_rel: max_rel(&gpu, &cpu),
    })
}

/// `arle kernel <name> --shape M,N,K [--ref cpu] [--iters N]`: run one
/// registered kernel and print its device time and max relative error against
/// the CPU f32 reference. Non-zero exit on unknown kernel, bad shape, or
/// correctness failure.
pub fn run_kernel_bench(name: &str, shape: &str, reference: &str, iters: usize) -> Result<()> {
    if reference != "cpu" {
        bail!("unsupported --ref {reference:?}; only `cpu` exists");
    }
    ensure!(iters >= 1, "--iters must be >= 1 (got {iters})");
    let (m, n, k) = parse_shape(shape)?;
    ensure!(
        m >= 1 && n >= 1 && k >= 1,
        "shape dims must be >= 1 (got {m},{n},{k})"
    );
    ensure!(
        k % 2 == 0 && k % GROUP == 0,
        "K must be a multiple of {GROUP} (got {k})"
    );
    let Some((_, run)) = REGISTRY.iter().find(|(registered, _)| *registered == name) else {
        let known = REGISTRY
            .iter()
            .map(|(registered, _)| *registered)
            .collect::<Vec<_>>()
            .join(", ");
        bail!("unknown kernel {name:?}; registered: {known}");
    };
    let ctx = DeviceContext::new()?;
    let report = run(&ctx, m, n, k, iters)?;
    let ok = report.max_rel < PASS_MAX_REL;
    println!(
        "kernel={name} shape={m}x{n}x{k} iters={iters} time_ms={:.4} max_rel={:.4e} status={}",
        report.time_ms,
        report.max_rel,
        if ok { "PASS" } else { "FAIL" }
    );
    if !ok {
        bail!("max_rel {:.4e} exceeds {PASS_MAX_REL}", report.max_rel);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_parses() {
        assert_eq!(parse_shape("1,34816,5120").unwrap(), (1, 34816, 5120));
        assert!(parse_shape("1,2").is_err());
        assert!(parse_shape("1,x,3").is_err());
    }
}
