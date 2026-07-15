//! Isolated Qwen/DSv4 FP8 grouped-prefill DeepGEMM probe.
//!
//! This bypasses HTTP, model loading, routing, and scheduler state. It calls the
//! same four production kernels used by the large-routed-row FP8 grouped MoE
//! lane: BF16->FP8 pack, grouped w13 GEMM, SwiGLU+requantize, grouped down GEMM.
//!
//! Run on the pod:
//! `INFER_CUDA_DEVICE=2 DG_JIT_CACHE_DIR=/tmp/dg-qwen target/release/examples/fp8_grouped_prefill_probe qwen`
//! `INFER_CUDA_DEVICE=3 DG_JIT_CACHE_DIR=/tmp/dg-dsv4 target/release/examples/fp8_grouped_prefill_probe dsv4`

fn main() -> anyhow::Result<()> {
    real::run()
}

#[cfg(not(feature = "cuda"))]
mod real {
    pub(super) fn run() -> anyhow::Result<()> {
        eprintln!("fp8_grouped_prefill_probe is a CUDA harness; rebuild with --features cuda.");
        Ok(())
    }
}

#[cfg(feature = "cuda")]
mod real {
    use std::time::Instant;

    use anyhow::{Result, anyhow, bail, ensure};
    use cuda_kernels::moe;
    use cuda_kernels::prelude::DeviceContext;
    use cuda_kernels::tensor::cache_ptr;
    use cudarc::driver::sys::CUevent_flags;
    use half::bf16;

    const ALIGN: usize = 128;
    const SCALE_GRAN_K: usize = 128;

    #[derive(Clone, Copy, Debug)]
    enum Kind {
        Qwen,
        Dsv4,
    }

    #[derive(Clone, Debug)]
    struct Shape {
        kind: Kind,
        tokens: usize,
        topk: usize,
        experts: usize,
        hidden: usize,
        intermediate: usize,
        swiglu_limit: f32,
    }

    #[derive(Default)]
    struct StageTimes {
        host_ms: f64,
        cuda_ms: f64,
    }

    pub(super) fn run() -> Result<()> {
        let kind = parse_kind()?;
        let shape = shape_from_env(kind)?;
        validate_shape(&shape)?;
        let routes = shape.tokens * shape.topk;
        let (m_indices, aligned_total, max_count) = build_m_indices(routes, shape.experts, ALIGN)?;
        let rows = deepgemm_contig_rows_cap(routes.max(1), shape.experts, ALIGN);
        ensure!(
            m_indices.len() == rows,
            "internal m_indices length {} != rows {rows}",
            m_indices.len()
        );
        let scale_stride_m = rows.div_ceil(4) * 4;
        let hidden_scale_cols = shape.hidden.div_ceil(SCALE_GRAN_K);
        let inter_scale_cols = shape.intermediate.div_ceil(SCALE_GRAN_K);
        let w13_n = 2 * shape.intermediate;
        let down_n = shape.hidden;

        let ctx = DeviceContext::new()?;
        let cc = ctx.compute_capability();
        let preflight = moe::dsv4_deepgemm_native_preflight()?;
        eprintln!(
            "[fp8-grouped-probe] kind={} dispatch=DeepGEMM_contiguous_fp8 device={} cc={}.{} preflight={}",
            kind.name(),
            ctx.ordinal(),
            cc.0,
            cc.1,
            preflight
        );
        eprintln!(
            "[fp8-grouped-probe] params kind={} tokens={} topk={} total_routes={} experts={} hidden={} intermediate={} rows_cap={} aligned_total={} max_count={} scale_stride_m={} hidden_scale_cols={} inter_scale_cols={} mk_align={} m_indices_valid={} m_indices_pad={}",
            kind.name(),
            shape.tokens,
            shape.topk,
            routes,
            shape.experts,
            shape.hidden,
            shape.intermediate,
            rows,
            aligned_total,
            max_count,
            scale_stride_m,
            hidden_scale_cols,
            inter_scale_cols,
            ALIGN,
            routes,
            rows.saturating_sub(routes)
        );

        // Zero data is intentional: the probe measures dispatch/JIT/memory traffic,
        // not numerical quality. FP8 scales are explicit 1.0 so the GEMM path is
        // realistic and the SwiGLU quantizer handles all-zero output with scale=1.
        let packed_hidden = ctx.stream.alloc_zeros::<bf16>(rows * shape.hidden)?;
        let mut input_fp8 = ctx.stream.alloc_zeros::<u8>(rows * shape.hidden)?;
        let mut input_scales = ctx
            .stream
            .alloc_zeros::<f32>(scale_stride_m * hidden_scale_cols)?;
        let w13_weight = ctx
            .stream
            .alloc_zeros::<u8>(shape.experts * w13_n * shape.hidden)?;
        let w13_scales = ctx.stream.clone_htod(&vec![
            1.0f32;
            shape.experts
                * w13_n.div_ceil(SCALE_GRAN_K)
                * hidden_scale_cols
        ])?;
        let mut w13_out = ctx.stream.alloc_zeros::<bf16>(rows * w13_n)?;
        let mut act_fp8 = ctx.stream.alloc_zeros::<u8>(rows * shape.intermediate)?;
        let mut act_scales = ctx
            .stream
            .alloc_zeros::<f32>(scale_stride_m * inter_scale_cols)?;
        let down_weight = ctx
            .stream
            .alloc_zeros::<u8>(shape.experts * down_n * shape.intermediate)?;
        let down_scales = ctx.stream.clone_htod(&vec![
            1.0f32;
            shape.experts
                * down_n.div_ceil(SCALE_GRAN_K)
                * inter_scale_cols
        ])?;
        let mut expert_out = ctx.stream.alloc_zeros::<bf16>(rows * down_n)?;
        let m_indices = ctx.stream.clone_htod(&m_indices)?;
        let active_experts = ctx.stream.clone_htod(&[0i32])?;
        let active_offsets = ctx.stream.clone_htod(&[0i32])?;
        let active_counts = ctx.stream.clone_htod(&[i32::try_from(rows)?])?;
        ctx.sync()?;

        eprintln!(
            "[fp8-grouped-probe] allocations_mib input_bf16={:.1} input_fp8={:.1} w13_weight={:.1} w13_out={:.1} act_fp8={:.1} down_weight={:.1} out={:.1}",
            mib(rows * shape.hidden * std::mem::size_of::<bf16>()),
            mib(rows * shape.hidden),
            mib(shape.experts * w13_n * shape.hidden),
            mib(rows * w13_n * std::mem::size_of::<bf16>()),
            mib(rows * shape.intermediate),
            mib(shape.experts * down_n * shape.intermediate),
            mib(rows * down_n * std::mem::size_of::<bf16>()),
        );

        let mut cold = StageTimes::default();
        run_phase(
            &ctx,
            &shape,
            rows,
            scale_stride_m,
            &packed_hidden,
            &mut input_fp8,
            &mut input_scales,
            &w13_weight,
            &w13_scales,
            &mut w13_out,
            &mut act_fp8,
            &mut act_scales,
            &down_weight,
            &down_scales,
            &mut expert_out,
            &m_indices,
            &active_experts,
            &active_offsets,
            &active_counts,
            "cold",
            &mut cold,
        )?;
        let mut cached = StageTimes::default();
        run_phase(
            &ctx,
            &shape,
            rows,
            scale_stride_m,
            &packed_hidden,
            &mut input_fp8,
            &mut input_scales,
            &w13_weight,
            &w13_scales,
            &mut w13_out,
            &mut act_fp8,
            &mut act_scales,
            &down_weight,
            &down_scales,
            &mut expert_out,
            &m_indices,
            &active_experts,
            &active_offsets,
            &active_counts,
            "cached",
            &mut cached,
        )?;

        eprintln!(
            "[fp8-grouped-probe] summary kind={} cold_host_ms={:.3} cold_cuda_ms={:.3} cached_host_ms={:.3} cached_cuda_ms={:.3} jit_delta_host_ms={:.3} jit_delta_cuda_ms={:.3}",
            kind.name(),
            cold.host_ms,
            cold.cuda_ms,
            cached.host_ms,
            cached.cuda_ms,
            (cold.host_ms - cached.host_ms).max(0.0),
            (cold.cuda_ms - cached.cuda_ms).max(0.0)
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_phase(
        ctx: &DeviceContext,
        shape: &Shape,
        rows: usize,
        scale_stride_m: usize,
        packed_hidden: &cudarc::driver::CudaSlice<bf16>,
        input_fp8: &mut cudarc::driver::CudaSlice<u8>,
        input_scales: &mut cudarc::driver::CudaSlice<f32>,
        w13_weight: &cudarc::driver::CudaSlice<u8>,
        w13_scales: &cudarc::driver::CudaSlice<f32>,
        w13_out: &mut cudarc::driver::CudaSlice<bf16>,
        act_fp8: &mut cudarc::driver::CudaSlice<u8>,
        act_scales: &mut cudarc::driver::CudaSlice<f32>,
        down_weight: &cudarc::driver::CudaSlice<u8>,
        down_scales: &cudarc::driver::CudaSlice<f32>,
        expert_out: &mut cudarc::driver::CudaSlice<bf16>,
        m_indices: &cudarc::driver::CudaSlice<i32>,
        active_experts: &cudarc::driver::CudaSlice<i32>,
        active_offsets: &cudarc::driver::CudaSlice<i32>,
        active_counts: &cudarc::driver::CudaSlice<i32>,
        phase: &'static str,
        total: &mut StageTimes,
    ) -> Result<()> {
        let stream = ctx.stream.cu_stream();
        timed(
            ctx,
            shape.kind,
            phase,
            "pack_quantize_hidden",
            rows,
            shape.hidden,
            0,
            total,
            // SAFETY: the buffers and dimensions above match this FFI contract.
            || unsafe {
                moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                    cache_ptr(packed_hidden, ctx),
                    cache_ptr(input_fp8, ctx),
                    cache_ptr(input_scales, ctx),
                    cache_ptr(active_experts, ctx),
                    cache_ptr(active_offsets, ctx),
                    cache_ptr(active_counts, ctx),
                    1,
                    rows,
                    shape.hidden,
                    scale_stride_m,
                    stream,
                )
            },
        )?;
        timed(
            ctx,
            shape.kind,
            phase,
            "gemm_w13",
            rows,
            2 * shape.intermediate,
            shape.hidden,
            total,
            // SAFETY: the buffers and dimensions above match this FFI contract.
            || unsafe {
                moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                    cache_ptr(input_fp8, ctx),
                    cache_ptr(input_scales, ctx),
                    cache_ptr(w13_weight, ctx),
                    cache_ptr(w13_scales, ctx),
                    cache_ptr(w13_out, ctx),
                    cache_ptr(m_indices, ctx),
                    shape.experts,
                    rows,
                    2 * shape.intermediate,
                    shape.hidden,
                    scale_stride_m,
                    ALIGN,
                    stream,
                )
            },
        )?;
        timed(
            ctx,
            shape.kind,
            phase,
            "swiglu_quantize",
            rows,
            shape.intermediate,
            0,
            total,
            // SAFETY: the buffers and dimensions above match this FFI contract.
            || unsafe {
                moe::dsv4_deepgemm_swiglu_quantize_w13(
                    cache_ptr(w13_out, ctx),
                    cache_ptr(act_fp8, ctx),
                    cache_ptr(act_scales, ctx),
                    cache_ptr(active_experts, ctx),
                    cache_ptr(active_counts, ctx),
                    1,
                    rows,
                    shape.intermediate,
                    scale_stride_m,
                    shape.swiglu_limit,
                    stream,
                )
            },
        )?;
        timed(
            ctx,
            shape.kind,
            phase,
            "gemm_down",
            rows,
            shape.hidden,
            shape.intermediate,
            total,
            // SAFETY: the buffers and dimensions above match this FFI contract.
            || unsafe {
                moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
                    cache_ptr(act_fp8, ctx),
                    cache_ptr(act_scales, ctx),
                    cache_ptr(down_weight, ctx),
                    cache_ptr(down_scales, ctx),
                    cache_ptr(expert_out, ctx),
                    cache_ptr(m_indices, ctx),
                    shape.experts,
                    rows,
                    shape.hidden,
                    shape.intermediate,
                    scale_stride_m,
                    ALIGN,
                    stream,
                )
            },
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn timed<T>(
        ctx: &DeviceContext,
        kind: Kind,
        phase: &'static str,
        stage: &'static str,
        rows: usize,
        n: usize,
        k: usize,
        total: &mut StageTimes,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let start = ctx
            .ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| anyhow!("create start event for {stage} failed: {e}"))?;
        let stop = ctx
            .ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| anyhow!("create stop event for {stage} failed: {e}"))?;
        start
            .record(&ctx.stream)
            .map_err(|e| anyhow!("record start event for {stage} failed: {e}"))?;
        let host_t0 = Instant::now();
        let result = f();
        let host_ms = host_t0.elapsed().as_secs_f64() * 1000.0;
        stop.record(&ctx.stream)
            .map_err(|e| anyhow!("record stop event for {stage} failed: {e}"))?;
        stop.synchronize()
            .map_err(|e| anyhow!("sync stop event for {stage} failed: {e}"))?;
        let cuda_ms = start
            .elapsed_ms(&stop)
            .map_err(|e| anyhow!("elapsed event for {stage} failed: {e}"))?
            as f64;
        total.host_ms += host_ms;
        total.cuda_ms += cuda_ms;
        eprintln!(
            "[fp8-grouped-probe] kind={} phase={phase} stage={stage} dispatch=DeepGEMM_contiguous_fp8 rows={rows} n={n} k={k} host_ms={host_ms:.3} cuda_ms={cuda_ms:.3}",
            kind.name()
        );
        result
    }

    fn parse_kind() -> Result<Kind> {
        match std::env::args().nth(1).as_deref() {
            None | Some("qwen") | Some("qwen36") | Some("qwen3.6") => Ok(Kind::Qwen),
            Some("dsv4") | Some("deepseek") => Ok(Kind::Dsv4),
            Some(other) => bail!("unknown probe kind {other:?}; expected qwen or dsv4"),
        }
    }

    fn shape_from_env(kind: Kind) -> Result<Shape> {
        let defaults = match kind {
            Kind::Qwen => Shape {
                kind,
                tokens: 4096,
                topk: 8,
                experts: 256,
                hidden: 2048,
                intermediate: 512,
                swiglu_limit: f32::INFINITY,
            },
            Kind::Dsv4 => Shape {
                kind,
                tokens: 4096,
                topk: 6,
                experts: 32,
                hidden: 4096,
                intermediate: 2048,
                swiglu_limit: 10.0,
            },
        };
        Ok(Shape {
            kind,
            tokens: env_usize("ARLE_FP8_PROBE_TOKENS", defaults.tokens)?,
            topk: env_usize("ARLE_FP8_PROBE_TOPK", defaults.topk)?,
            experts: env_usize("ARLE_FP8_PROBE_EXPERTS", defaults.experts)?,
            hidden: env_usize("ARLE_FP8_PROBE_HIDDEN", defaults.hidden)?,
            intermediate: env_usize("ARLE_FP8_PROBE_INTERMEDIATE", defaults.intermediate)?,
            swiglu_limit: env_f32("ARLE_FP8_PROBE_SWIGLU_LIMIT", defaults.swiglu_limit)?,
        })
    }

    fn validate_shape(shape: &Shape) -> Result<()> {
        ensure!(shape.tokens > 0, "tokens must be positive");
        ensure!(shape.topk > 0, "topk must be positive");
        ensure!(shape.experts > 0, "experts must be positive");
        ensure!(
            shape.hidden.is_multiple_of(SCALE_GRAN_K),
            "hidden {} must be {SCALE_GRAN_K}-aligned",
            shape.hidden
        );
        ensure!(
            shape.intermediate.is_multiple_of(SCALE_GRAN_K),
            "intermediate {} must be {SCALE_GRAN_K}-aligned",
            shape.intermediate
        );
        ensure!(
            (2 * shape.intermediate).is_multiple_of(8) && shape.hidden.is_multiple_of(8),
            "DeepGEMM requires n % 8 == 0 for both w13 and down"
        );
        ensure!(
            shape.swiglu_limit > 0.0,
            "swiglu_limit must be positive for the quantize kernel"
        );
        Ok(())
    }

    fn deepgemm_contig_rows_cap(total_routes: usize, local_experts: usize, align: usize) -> usize {
        total_routes.div_ceil(align) * align + align * local_experts.min(total_routes)
    }

    fn build_m_indices(
        routes: usize,
        experts: usize,
        align: usize,
    ) -> Result<(Vec<i32>, usize, usize)> {
        ensure!(routes > 0, "routes must be positive");
        ensure!(experts > 0, "experts must be positive");
        let rows = deepgemm_contig_rows_cap(routes, experts, align);
        let mut out = vec![-1i32; rows];
        let base = routes / experts;
        let remainder = routes % experts;
        let mut cursor = 0usize;
        let mut max_count = 0usize;
        for expert in 0..experts {
            let count = base + usize::from(expert < remainder);
            max_count = max_count.max(count);
            for row in 0..count {
                out[cursor + row] = i32::try_from(expert)?;
            }
            cursor += count.div_ceil(align) * align;
        }
        ensure!(
            cursor <= rows,
            "aligned m_indices total {cursor} exceeds rows cap {rows}"
        );
        Ok((out, cursor, max_count))
    }

    fn env_usize(name: &str, default: usize) -> Result<usize> {
        match std::env::var(name) {
            Ok(value) => value
                .parse::<usize>()
                .map_err(|e| anyhow!("{name} must be usize, got {value:?}: {e}")),
            Err(std::env::VarError::NotPresent) => Ok(default),
            Err(e) => Err(anyhow!("{name} read failed: {e}")),
        }
    }

    fn env_f32(name: &str, default: f32) -> Result<f32> {
        match std::env::var(name) {
            Ok(value) => value
                .parse::<f32>()
                .map_err(|e| anyhow!("{name} must be f32, got {value:?}: {e}")),
            Err(std::env::VarError::NotPresent) => Ok(default),
            Err(e) => Err(anyhow!("{name} read failed: {e}")),
        }
    }

    fn mib(bytes: usize) -> f64 {
        bytes as f64 / 1024.0 / 1024.0
    }

    impl Kind {
        fn name(self) -> &'static str {
            match self {
                Kind::Qwen => "qwen",
                Kind::Dsv4 => "dsv4",
            }
        }
    }
}
