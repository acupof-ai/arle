//! Isolated single-GPU microbench for the DSv4 single-block prologue kernels.
//!
//! Settles the block-size questions (256 vs 1024 for `dsv4_mhc_params` and the
//! fused `dsv4_mhc_pre_rms_norm`) with CUDA-stream timing on ONE idle GPU —
//! immune to the 8-GPU serve lane's co-tenant contention that poisoned the
//! e2e A/Bs (2026-06-11). Production shapes: H=4096, hc_mult=4, sinkhorn=20.
//!
//! Run (pod): INFER_CUDA_DEVICE=0 target/release/examples/dsv4_microbench

use std::time::Instant;

use anyhow::Result;
use cuda_kernels::ffi;
use cuda_kernels::prelude::DeviceContext;
use cudarc::driver::{DevicePtr, DevicePtrMut};

const HIDDEN: usize = 4096;
const HC_MULT: usize = 4;
const MIX_DIM: usize = HC_MULT * (2 + HC_MULT); // 24
const SINKHORN_ITERS: i32 = 20;
const EPS: f32 = 1e-6;
const ITERS: usize = 2000;
const WARMUP: usize = 50;

fn main() -> Result<()> {
    let ctx = DeviceContext::new()?; // honours INFER_CUDA_DEVICE
    let stream_dim = HIDDEN * HC_MULT;

    // bf16 0x3F80 = 1.0; non-zero data so sinkhorn/softmax run realistic paths.
    let one_bf16: u16 = 0x3F80;
    let residual = ctx.stream.clone_htod(&vec![one_bf16; stream_dim])?;
    let mixes = ctx.stream.clone_htod(&vec![one_bf16; MIX_DIM])?;
    let base = ctx.stream.clone_htod(&vec![one_bf16; MIX_DIM])?;
    let scale = ctx.stream.clone_htod(&vec![one_bf16; 3])?;
    let weight = ctx.stream.clone_htod(&vec![one_bf16; HIDDEN])?;
    let mut pre = ctx.stream.alloc_zeros::<f32>(HC_MULT)?;
    let mut post = ctx.stream.alloc_zeros::<f32>(HC_MULT)?;
    let mut comb = ctx.stream.alloc_zeros::<f32>(HC_MULT * HC_MULT)?;
    let mut normed = ctx.stream.alloc_zeros::<u16>(HIDDEN)?;

    let (res_ptr, _g1) = residual.device_ptr(&ctx.stream);
    let (mix_ptr, _g2) = mixes.device_ptr(&ctx.stream);
    let (base_ptr, _g3) = base.device_ptr(&ctx.stream);
    let (scale_ptr, _g4) = scale.device_ptr(&ctx.stream);
    let (w_ptr, _g5) = weight.device_ptr(&ctx.stream);
    let (pre_ptr, _g6) = pre.device_ptr_mut(&ctx.stream);
    let (post_ptr, _g7) = post.device_ptr_mut(&ctx.stream);
    let (comb_ptr, _g8) = comb.device_ptr_mut(&ctx.stream);
    let (out_ptr, _g9) = normed.device_ptr_mut(&ctx.stream);

    println!("kernel              block   us/inst");
    for &block in &[128i32, 256, 512, 1024] {
        // mhc_params
        let time = |f: &dyn Fn() -> Result<()>| -> Result<f64> {
            for _ in 0..WARMUP {
                f()?;
            }
            ctx.sync()?;
            let t0 = Instant::now();
            for _ in 0..ITERS {
                f()?;
            }
            ctx.sync()?;
            Ok(t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64)
        };

        let us = time(&|| {
            // SAFETY: buffers sized above for the production shapes.
            unsafe {
                ffi::dsv4_mhc_params_bench_cuda(
                    res_ptr as *const ffi::Half,
                    mix_ptr as *const ffi::Half,
                    base_ptr as *const ffi::Half,
                    scale_ptr as *const ffi::Half,
                    pre_ptr as *mut f32,
                    post_ptr as *mut f32,
                    comb_ptr as *mut f32,
                    1,
                    stream_dim as i32,
                    MIX_DIM as i32,
                    HC_MULT as i32,
                    EPS,
                    SINKHORN_ITERS,
                    block,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
            Ok(())
        })?;
        println!("mhc_params          {block:5} {us:9.2}");

        let us = time(&|| {
            // SAFETY: buffers sized above for the production shapes.
            unsafe {
                ffi::dsv4_mhc_pre_rms_norm_bench_cuda(
                    res_ptr as *const ffi::Half,
                    pre_ptr as *const f32,
                    w_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    1,
                    HIDDEN as i32,
                    HC_MULT as i32,
                    EPS,
                    block,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
            Ok(())
        })?;
        println!("hc_pre_rms_norm     {block:5} {us:9.2}");
    }
    Ok(())
}
