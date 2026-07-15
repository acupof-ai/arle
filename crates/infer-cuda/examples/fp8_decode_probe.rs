//! Isolated Qwen3.6 FP8 decode-path probe.
//!
//! This bypasses HTTP, scheduler, tokenizer, and model loading. It constructs
//! the Qwen3.6 decode-band shapes and compares:
//! - BF16 dense B=1 GEMM vs FP8 resident GEMV-batch B=1.
//! - BF16 routed-MoE fused decode kernels vs the legacy FP8 routed-MoE batch
//!   GEMV path, including the separate SwiGLU pass.
//! - The FP8 decode-fused path now used by production for Qwen FP8 128x128
//!   block-scaled weights.
//!
//! Run on a pod:
//! `INFER_CUDA_DEVICE=2 target/release/examples/fp8_decode_probe`

fn main() -> anyhow::Result<()> {
    real::run()
}

#[cfg(not(feature = "cuda"))]
mod real {
    pub(super) fn run() -> anyhow::Result<()> {
        eprintln!("fp8_decode_probe is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use std::time::Instant;

    use anyhow::{Result, anyhow, ensure};
    use cuda_kernels::ffi;
    use cuda_kernels::moe;
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::tensor::cache_ptr;
    use cudarc::driver::sys::CUevent_flags;
    use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
    use half::bf16;

    const DEFAULT_EXPERTS: usize = 256;
    const DEFAULT_TOPK: usize = 8;
    const DEFAULT_HIDDEN: usize = 2048;
    const DEFAULT_INTERMEDIATE: usize = 512;
    const DEFAULT_ITERS: usize = 200;
    const FP8_BLOCK_M: usize = 128;
    const FP8_BLOCK_K: usize = 128;

    #[derive(Clone, Debug)]
    struct Shape {
        experts: usize,
        topk: usize,
        hidden: usize,
        intermediate: usize,
        iters: usize,
    }

    #[derive(Default, Clone, Copy)]
    struct Sample {
        host_ms: f64,
        cuda_ms: f64,
    }

    pub(super) fn run() -> Result<()> {
        let shape = shape_from_env()?;
        validate_shape(&shape)?;
        let ctx = DeviceContext::new()?;
        let cc = ctx.compute_capability();
        eprintln!(
            "[fp8-decode-probe] device={} cc={}.{} experts={} topk={} hidden={} intermediate={} iters={}",
            ctx.ordinal(),
            cc.0,
            cc.1,
            shape.experts,
            shape.topk,
            shape.hidden,
            shape.intermediate,
            shape.iters
        );
        eprintln!(
            "[fp8-decode-probe] production_dispatch fp8_block_scaled => decode_fused when R<=256 and 128x128 scales; legacy_batch=grouped_gemv_pair_batch + silu_mul + grouped_gemv_batch"
        );

        let dense = run_dense_probe(&ctx, &shape)?;
        let moe = run_moe_probe(&ctx, &shape)?;

        eprintln!(
            "[fp8-decode-probe] summary dense_bf16_ms={:.4} dense_fp8_ms={:.4} dense_delta_pct={:.1} moe_bf16_fused_ms={:.4} moe_fp8_legacy_batch_ms={:.4} moe_legacy_batch_delta_pct={:.1} moe_fp8_decode_fused_ms={:.4} moe_decode_fused_delta_pct={:.1}",
            dense.bf16.cuda_ms,
            dense.fp8.cuda_ms,
            pct_delta(dense.fp8.cuda_ms, dense.bf16.cuda_ms),
            moe.bf16_total.cuda_ms,
            moe.fp8_legacy_batch_total.cuda_ms,
            pct_delta(moe.fp8_legacy_batch_total.cuda_ms, moe.bf16_total.cuda_ms),
            moe.fp8_decode_fused_total.cuda_ms,
            pct_delta(moe.fp8_decode_fused_total.cuda_ms, moe.bf16_total.cuda_ms)
        );
        Ok(())
    }

    struct DenseResult {
        bf16: Sample,
        fp8: Sample,
    }

    fn run_dense_probe(ctx: &DeviceContext, shape: &Shape) -> Result<DenseResult> {
        let m = 1usize;
        let n = shape.hidden;
        let k = shape.hidden;
        let x = ctx.stream.alloc_zeros::<bf16>(m * k)?;
        let w_bf16 = ctx.stream.alloc_zeros::<bf16>(n * k)?;
        let mut out_bf16 = ctx.stream.alloc_zeros::<bf16>(m * n)?;
        let w_fp8 = ctx.stream.alloc_zeros::<u8>(n * k)?;
        let scales = ctx.stream.clone_htod(&vec![
            1.0f32;
            n.div_ceil(FP8_BLOCK_M)
                * k.div_ceil(FP8_BLOCK_K)
        ])?;
        let mut out_fp8 = ctx.stream.alloc_zeros::<bf16>(m * n)?;
        ctx.sync()?;

        let bf16 = timed(
            ctx,
            "dense/bf16_gemm_batch_b1",
            shape.iters,
            n,
            k,
            // SAFETY: the buffers and dimensions above match this FFI contract.
            || unsafe {
                let (w, _gw) = w_bf16.device_ptr(&ctx.stream);
                let (x_ptr, _gx) = x.device_ptr(&ctx.stream);
                let (out, _go) = out_bf16.device_ptr_mut(&ctx.stream);
                ffi::gemm_cuda(
                    w as *const ffi::Half,
                    x_ptr as *const ffi::Half,
                    out as *mut ffi::Half,
                    n as i32,
                    m as i32,
                    k as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
                Ok(())
            },
        )?;
        let fp8 = timed(
            ctx,
            "dense/fp8_block_scaled_gemv_batch_b1",
            shape.iters,
            n,
            k,
            // SAFETY: the buffers and dimensions above match this FFI contract.
            || unsafe {
                let (w, _gw) = w_fp8.device_ptr(&ctx.stream);
                let (s, _gs) = scales.device_ptr(&ctx.stream);
                let (x_ptr, _gx) = x.device_ptr(&ctx.stream);
                let (out, _go) = out_fp8.device_ptr_mut(&ctx.stream);
                ffi::gemv_fp8_block_scaled_batch_cuda(
                    w as *const u8,
                    s as *const f32,
                    x_ptr as *const ffi::Half,
                    out as *mut ffi::Half,
                    m as i32,
                    n as i32,
                    k as i32,
                    n.div_ceil(FP8_BLOCK_M) as i32,
                    k.div_ceil(FP8_BLOCK_K) as i32,
                    FP8_BLOCK_M as i32,
                    FP8_BLOCK_K as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
                Ok(())
            },
        )?;
        Ok(DenseResult { bf16, fp8 })
    }

    struct MoeResult {
        bf16_total: Sample,
        fp8_legacy_batch_total: Sample,
        fp8_decode_fused_total: Sample,
    }

    fn run_moe_probe(ctx: &DeviceContext, shape: &Shape) -> Result<MoeResult> {
        let routes = shape.topk;
        let offsets_host: Vec<i32> = (0..shape.experts)
            .map(|expert| i32::try_from(expert.min(routes)).unwrap())
            .collect();
        let counts_host: Vec<i32> = (0..shape.experts)
            .map(|expert| if expert < routes { 1 } else { 0 })
            .collect();
        let expert_indices_host: Vec<i32> = (0..shape.experts)
            .map(i32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let route_slot_host: Vec<i32> = (0..routes)
            .map(i32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let route_weight_host = vec![1.0f32; routes];

        let offsets = ctx.stream.clone_htod(&offsets_host)?;
        let counts = ctx.stream.clone_htod(&counts_host)?;
        let expert_indices = ctx.stream.clone_htod(&expert_indices_host)?;
        let route_slot = ctx.stream.clone_htod(&route_slot_host)?;
        let route_weight = ctx.stream.clone_htod(&route_weight_host)?;

        let packed_hidden = ctx.stream.alloc_zeros::<bf16>(routes * shape.hidden)?;
        let gate_out = ctx
            .stream
            .alloc_zeros::<bf16>(routes * shape.intermediate)?;
        let up_out = ctx
            .stream
            .alloc_zeros::<bf16>(routes * shape.intermediate)?;
        let mut act = ctx
            .stream
            .alloc_zeros::<bf16>(routes * shape.intermediate)?;
        let expert_out = ctx.stream.alloc_zeros::<bf16>(routes * shape.hidden)?;
        let route_out = ctx.stream.alloc_zeros::<bf16>(routes * shape.hidden)?;
        let combined = ctx.stream.alloc_zeros::<bf16>(shape.hidden)?;

        let gate_bf16 = ctx
            .stream
            .alloc_zeros::<bf16>(shape.experts * shape.intermediate * shape.hidden)?;
        let up_bf16 = ctx
            .stream
            .alloc_zeros::<bf16>(shape.experts * shape.intermediate * shape.hidden)?;
        let down_bf16 = ctx
            .stream
            .alloc_zeros::<bf16>(shape.experts * shape.hidden * shape.intermediate)?;
        let gate_bf16_ptrs = ptr_table(
            ctx,
            &gate_bf16,
            shape.experts,
            shape.intermediate * shape.hidden,
        )?;
        let up_bf16_ptrs = ptr_table(
            ctx,
            &up_bf16,
            shape.experts,
            shape.intermediate * shape.hidden,
        )?;
        let down_bf16_ptrs = ptr_table(
            ctx,
            &down_bf16,
            shape.experts,
            shape.hidden * shape.intermediate,
        )?;

        let gate_fp8 = ctx
            .stream
            .alloc_zeros::<u8>(shape.experts * shape.intermediate * shape.hidden)?;
        let up_fp8 = ctx
            .stream
            .alloc_zeros::<u8>(shape.experts * shape.intermediate * shape.hidden)?;
        let down_fp8 = ctx
            .stream
            .alloc_zeros::<u8>(shape.experts * shape.hidden * shape.intermediate)?;
        let gate_scales =
            ctx.stream.clone_htod(&vec![
                1.0f32;
                shape.experts
                    * shape.intermediate.div_ceil(FP8_BLOCK_M)
                    * shape.hidden.div_ceil(FP8_BLOCK_K)
            ])?;
        let up_scales =
            ctx.stream.clone_htod(&vec![
                1.0f32;
                shape.experts
                    * shape.intermediate.div_ceil(FP8_BLOCK_M)
                    * shape.hidden.div_ceil(FP8_BLOCK_K)
            ])?;
        let down_scales =
            ctx.stream.clone_htod(&vec![
                1.0f32;
                shape.experts
                    * shape.hidden.div_ceil(FP8_BLOCK_M)
                    * shape.intermediate.div_ceil(FP8_BLOCK_K)
            ])?;
        let gate_fp8_ptrs = ptr_table(
            ctx,
            &gate_fp8,
            shape.experts,
            shape.intermediate * shape.hidden,
        )?;
        let up_fp8_ptrs = ptr_table(
            ctx,
            &up_fp8,
            shape.experts,
            shape.intermediate * shape.hidden,
        )?;
        let down_fp8_ptrs = ptr_table(
            ctx,
            &down_fp8,
            shape.experts,
            shape.hidden * shape.intermediate,
        )?;
        let gate_scale_ptrs = ptr_table(
            ctx,
            &gate_scales,
            shape.experts,
            shape.intermediate.div_ceil(FP8_BLOCK_M) * shape.hidden.div_ceil(FP8_BLOCK_K),
        )?;
        let up_scale_ptrs = ptr_table(
            ctx,
            &up_scales,
            shape.experts,
            shape.intermediate.div_ceil(FP8_BLOCK_M) * shape.hidden.div_ceil(FP8_BLOCK_K),
        )?;
        let down_scale_ptrs = ptr_table(
            ctx,
            &down_scales,
            shape.experts,
            shape.hidden.div_ceil(FP8_BLOCK_M) * shape.intermediate.div_ceil(FP8_BLOCK_K),
        )?;
        ctx.sync()?;

        let bf16_swiglu = timed(
            ctx,
            "moe/bf16_fused_swiglu_decode",
            shape.iters,
            shape.intermediate,
            shape.hidden,
            // SAFETY: the buffers and dimensions above match this FFI contract.
            || unsafe {
                moe::moe_bf16_grouped_gemm_swiglu_decode(
                    &gate_bf16_ptrs,
                    &up_bf16_ptrs,
                    cache_ptr(&packed_hidden, ctx),
                    cache_ptr(&act, ctx),
                    cache_ptr(&offsets, ctx),
                    cache_ptr(&counts, ctx),
                    cache_ptr(&expert_indices, ctx),
                    shape.experts,
                    routes,
                    shape.intermediate,
                    shape.hidden,
                    ctx,
                    ctx.stream.cu_stream(),
                )
            },
        )?;
        let bf16_down = timed(
            ctx,
            "moe/bf16_fused_down_decode",
            shape.iters,
            shape.hidden,
            shape.intermediate,
            // SAFETY: the buffers and dimensions above match this FFI contract.
            || unsafe {
                moe::moe_bf16_grouped_gemm_decode(
                    &down_bf16_ptrs,
                    cache_ptr(&act, ctx),
                    cache_ptr(&expert_out, ctx),
                    cache_ptr(&offsets, ctx),
                    cache_ptr(&counts, ctx),
                    cache_ptr(&expert_indices, ctx),
                    shape.experts,
                    routes,
                    shape.hidden,
                    shape.intermediate,
                    ctx,
                    ctx.stream.cu_stream(),
                )
            },
        )?;
        let bf16_scatter = timed(
            ctx,
            "moe/common_scatter_combine",
            shape.iters,
            shape.hidden,
            routes,
            // SAFETY: the buffers and dimensions above match this FFI contract.
            || unsafe {
                moe::dsv4_scatter_all_route_slots(
                    cache_ptr(&expert_out, ctx),
                    cache_ptr(&route_out, ctx),
                    cache_ptr(&route_slot, ctx),
                    cache_ptr(&route_weight, ctx),
                    routes,
                    shape.hidden,
                    ctx.stream.cu_stream(),
                )?;
                moe::dsv4_combine_route_slot_outputs(
                    cache_ptr(&route_out, ctx),
                    cache_ptr(&combined, ctx),
                    1,
                    shape.topk,
                    shape.hidden,
                    ctx.stream.cu_stream(),
                )
            },
        )?;
        let bf16_total = sum(&[bf16_swiglu, bf16_down, bf16_scatter]);

        let fp8_pair = timed(
            ctx,
            "moe/fp8_legacy_batch_pair",
            shape.iters,
            shape.intermediate,
            shape.hidden,
            // SAFETY: the buffers and dimensions above match this FFI contract.
            || unsafe {
                moe::moe_fp8_block_scaled_grouped_gemv_pair_batch(
                    &gate_fp8_ptrs,
                    &gate_scale_ptrs,
                    &up_fp8_ptrs,
                    &up_scale_ptrs,
                    cache_ptr(&packed_hidden, ctx),
                    cache_ptr(&gate_out, ctx),
                    cache_ptr(&up_out, ctx),
                    cache_ptr(&offsets, ctx),
                    cache_ptr(&counts, ctx),
                    cache_ptr(&expert_indices, ctx),
                    shape.experts,
                    routes,
                    shape.intermediate,
                    shape.hidden,
                    shape.intermediate.div_ceil(FP8_BLOCK_M),
                    shape.hidden.div_ceil(FP8_BLOCK_K),
                    FP8_BLOCK_M,
                    FP8_BLOCK_K,
                    ctx,
                    ctx.stream.cu_stream(),
                )
            },
        )?;
        let fp8_silu = timed(
            ctx,
            "moe/fp8_legacy_batch_silu_mul",
            shape.iters,
            shape.intermediate,
            routes,
            // SAFETY: the buffers and dimensions above match this FFI contract.
            || unsafe {
                let (gate, _gg) = gate_out.device_ptr(&ctx.stream);
                let (up, _gu) = up_out.device_ptr(&ctx.stream);
                let (act_ptr, _ga) = act.device_ptr_mut(&ctx.stream);
                ffi::silu_mul_cuda(
                    gate as *const ffi::Half,
                    up as *const ffi::Half,
                    act_ptr as *mut ffi::Half,
                    (routes * shape.intermediate) as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
                Ok(())
            },
        )?;
        let fp8_down = timed(
            ctx,
            "moe/fp8_legacy_batch_down",
            shape.iters,
            shape.hidden,
            shape.intermediate,
            // SAFETY: the buffers and dimensions above match this FFI contract.
            || unsafe {
                moe::moe_fp8_block_scaled_grouped_gemv_batch(
                    &down_fp8_ptrs,
                    &down_scale_ptrs,
                    cache_ptr(&act, ctx),
                    cache_ptr(&expert_out, ctx),
                    cache_ptr(&offsets, ctx),
                    cache_ptr(&counts, ctx),
                    cache_ptr(&expert_indices, ctx),
                    shape.experts,
                    routes,
                    shape.hidden,
                    shape.intermediate,
                    shape.hidden.div_ceil(FP8_BLOCK_M),
                    shape.intermediate.div_ceil(FP8_BLOCK_K),
                    FP8_BLOCK_M,
                    FP8_BLOCK_K,
                    ctx,
                    ctx.stream.cu_stream(),
                )
            },
        )?;
        let fp8_scatter = timed(
            ctx,
            "moe/fp8_legacy_batch_scatter_combine",
            shape.iters,
            shape.hidden,
            routes,
            // SAFETY: the buffers and dimensions above match this FFI contract.
            || unsafe {
                moe::dsv4_scatter_all_route_slots(
                    cache_ptr(&expert_out, ctx),
                    cache_ptr(&route_out, ctx),
                    cache_ptr(&route_slot, ctx),
                    cache_ptr(&route_weight, ctx),
                    routes,
                    shape.hidden,
                    ctx.stream.cu_stream(),
                )?;
                moe::dsv4_combine_route_slot_outputs(
                    cache_ptr(&route_out, ctx),
                    cache_ptr(&combined, ctx),
                    1,
                    shape.topk,
                    shape.hidden,
                    ctx.stream.cu_stream(),
                )
            },
        )?;
        let fp8_legacy_batch_total = sum(&[fp8_pair, fp8_silu, fp8_down, fp8_scatter]);

        let fp8_decode_fused_swiglu = timed(
            ctx,
            "moe/fp8_decode_fused_swiglu",
            shape.iters,
            shape.intermediate,
            shape.hidden,
            // SAFETY: the buffers and dimensions above match this FFI contract.
            || unsafe {
                moe::dsv4_fp8_grouped_swiglu_decode(
                    cache_ptr(&gate_fp8_ptrs, ctx),
                    cache_ptr(&gate_scale_ptrs, ctx),
                    cache_ptr(&up_fp8_ptrs, ctx),
                    cache_ptr(&up_scale_ptrs, ctx),
                    cache_ptr(&packed_hidden, ctx),
                    cache_ptr(&act, ctx),
                    cache_ptr(&offsets, ctx),
                    cache_ptr(&counts, ctx),
                    shape.experts,
                    routes,
                    shape.intermediate,
                    shape.hidden,
                    shape.hidden.div_ceil(FP8_BLOCK_K),
                    f32::INFINITY,
                    ctx.stream.cu_stream(),
                )
            },
        )?;
        let fp8_decode_fused_down = timed(
            ctx,
            "moe/fp8_decode_fused_down",
            shape.iters,
            shape.hidden,
            shape.intermediate,
            // SAFETY: the buffers and dimensions above match this FFI contract.
            || unsafe {
                moe::dsv4_fp8_grouped_down_decode(
                    cache_ptr(&down_fp8_ptrs, ctx),
                    cache_ptr(&down_scale_ptrs, ctx),
                    cache_ptr(&act, ctx),
                    cache_ptr(&expert_out, ctx),
                    cache_ptr(&offsets, ctx),
                    cache_ptr(&counts, ctx),
                    shape.experts,
                    routes,
                    shape.hidden,
                    shape.intermediate,
                    shape.intermediate.div_ceil(FP8_BLOCK_K),
                    ctx.stream.cu_stream(),
                )
            },
        )?;
        let fp8_decode_fused_scatter = timed(
            ctx,
            "moe/fp8_decode_fused_scatter_combine",
            shape.iters,
            shape.hidden,
            routes,
            // SAFETY: the buffers and dimensions above match this FFI contract.
            || unsafe {
                moe::dsv4_scatter_all_route_slots(
                    cache_ptr(&expert_out, ctx),
                    cache_ptr(&route_out, ctx),
                    cache_ptr(&route_slot, ctx),
                    cache_ptr(&route_weight, ctx),
                    routes,
                    shape.hidden,
                    ctx.stream.cu_stream(),
                )?;
                moe::dsv4_combine_route_slot_outputs(
                    cache_ptr(&route_out, ctx),
                    cache_ptr(&combined, ctx),
                    1,
                    shape.topk,
                    shape.hidden,
                    ctx.stream.cu_stream(),
                )
            },
        )?;
        let fp8_decode_fused_total = sum(&[
            fp8_decode_fused_swiglu,
            fp8_decode_fused_down,
            fp8_decode_fused_scatter,
        ]);
        eprintln!(
            "[fp8-decode-probe] moe_breakdown bf16_fused_ms={:.4} fp8_legacy_pair_ms={:.4} fp8_legacy_silu_ms={:.4} fp8_legacy_down_ms={:.4} fp8_decode_fused_swiglu_ms={:.4} fp8_decode_fused_down_ms={:.4} scatter_ms={:.4}",
            bf16_swiglu.cuda_ms + bf16_down.cuda_ms,
            fp8_pair.cuda_ms,
            fp8_silu.cuda_ms,
            fp8_down.cuda_ms,
            fp8_decode_fused_swiglu.cuda_ms,
            fp8_decode_fused_down.cuda_ms,
            fp8_scatter.cuda_ms
        );

        Ok(MoeResult {
            bf16_total,
            fp8_legacy_batch_total,
            fp8_decode_fused_total,
        })
    }

    fn ptr_table<T>(
        ctx: &DeviceContext,
        slice: &CudaSlice<T>,
        entries: usize,
        elems_per_entry: usize,
    ) -> Result<CudaSlice<u64>> {
        let (base, _guard) = slice.device_ptr(&ctx.stream);
        let stride = elems_per_entry
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| anyhow!("pointer-table stride overflow"))?;
        let ptrs: Vec<u64> = (0..entries)
            .map(|idx| base + (idx * stride) as u64)
            .collect();
        ctx.stream
            .clone_htod(&ptrs)
            .map_err(|e| anyhow!("pointer table H2D failed: {e}"))
    }

    fn timed(
        ctx: &DeviceContext,
        label: &'static str,
        iters: usize,
        n: usize,
        k: usize,
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
        let sample = Sample {
            host_ms: host_ms_total / iters as f64,
            cuda_ms: cuda_ms_total / iters as f64,
        };
        eprintln!(
            "[fp8-decode-probe] stage={label} n={n} k={k} avg_cuda_ms={:.4} avg_host_ms={:.4}",
            sample.cuda_ms, sample.host_ms
        );
        Ok(sample)
    }

    fn sum(samples: &[Sample]) -> Sample {
        Sample {
            host_ms: samples.iter().map(|s| s.host_ms).sum(),
            cuda_ms: samples.iter().map(|s| s.cuda_ms).sum(),
        }
    }

    fn shape_from_env() -> Result<Shape> {
        Ok(Shape {
            experts: env_usize("ARLE_FP8_DECODE_PROBE_EXPERTS", DEFAULT_EXPERTS)?,
            topk: env_usize("ARLE_FP8_DECODE_PROBE_TOPK", DEFAULT_TOPK)?,
            hidden: env_usize("ARLE_FP8_DECODE_PROBE_HIDDEN", DEFAULT_HIDDEN)?,
            intermediate: env_usize("ARLE_FP8_DECODE_PROBE_INTERMEDIATE", DEFAULT_INTERMEDIATE)?,
            iters: env_usize("ARLE_FP8_DECODE_PROBE_ITERS", DEFAULT_ITERS)?,
        })
    }

    fn validate_shape(shape: &Shape) -> Result<()> {
        ensure!(shape.experts > 0, "experts must be positive");
        ensure!(shape.topk > 0, "topk must be positive");
        ensure!(
            shape.topk <= shape.experts,
            "topk {} must be <= experts {}",
            shape.topk,
            shape.experts
        );
        ensure!(shape.hidden.is_multiple_of(8), "hidden must be 8-aligned");
        ensure!(
            shape.intermediate.is_multiple_of(8),
            "intermediate must be 8-aligned"
        );
        ensure!(
            shape.hidden.is_multiple_of(FP8_BLOCK_K)
                && shape.intermediate.is_multiple_of(FP8_BLOCK_K),
            "probe assumes 128-aligned Qwen FP8 blocks"
        );
        ensure!(shape.iters > 0, "iters must be positive");
        Ok(())
    }

    fn env_usize(name: &str, default: usize) -> Result<usize> {
        match std::env::var(name) {
            Ok(value) => value
                .parse::<usize>()
                .map_err(|e| anyhow!("{name}={value:?} is not usize: {e}")),
            Err(_) => Ok(default),
        }
    }

    fn pct_delta(new: f64, old: f64) -> f64 {
        if old == 0.0 {
            0.0
        } else {
            (new - old) / old * 100.0
        }
    }
}
