//! DeepSeek MoE routing helper kernels.

use anyhow::{Result, ensure};
use cudarc::driver::sys::CUstream;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi::{self, Half};
use crate::tensor::{DeviceContext, DeviceMatrix, DeviceVec, RawDevicePtr};

// Safe wrappers over the grouped-GEMM + DSv4/Qwen3.6 expert-dispatch FFI: they
// centralize device-pointer extraction + the i32-ABI casts. `RawDevicePtr<T>`
// (from `cache_ptr`) carries buffers; bf16 (Rust) and Half (u16, kernel ABI)
// share a 16-bit layout, so pointers cast directly.

#[derive(Clone, Copy, Debug)]
pub struct Sm90MegaMoeShape {
    pub num_ranks: usize,
    pub num_experts: usize,
    pub requested_max_tokens_per_rank: usize,
    pub num_topk: usize,
    pub hidden: usize,
    pub intermediate_hidden: usize,
}

pub type Sm90MegaMoeWorkspaceLayout = ffi::Sm90MegaMoeWorkspaceLayoutRaw;

/// Copy FP8 gate/up rows into MegaMoE's 8-row interleaved L1 layout.
pub fn interleave_gate_up_fp8_rows(
    ctx: &DeviceContext,
    gate: &impl DevicePtr<u8>,
    up: &impl DevicePtr<u8>,
    output: &mut impl DevicePtrMut<u8>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let half_len = rows
        .checked_mul(cols)
        .ok_or_else(|| anyhow::anyhow!("MegaMoE L1 shape overflow: {rows}x{cols}"))?;
    let total_len = half_len
        .checked_mul(2)
        .ok_or_else(|| anyhow::anyhow!("MegaMoE fused L1 shape overflow: {rows}x{cols}"))?;
    ensure!(
        rows.is_multiple_of(8) && cols.is_multiple_of(16),
        "MegaMoE L1 needs rows divisible by 8 and cols divisible by 16, got {rows}x{cols}"
    );
    ensure!(
        gate.len() == half_len && up.len() == half_len && output.len() == total_len,
        "MegaMoE L1 buffers mismatch: gate={} up={} output={} expected={half_len}/{half_len}/{}",
        gate.len(),
        up.len(),
        output.len(),
        total_len
    );
    let (gate_ptr, _gate_guard) = gate.device_ptr(&ctx.stream);
    let (up_ptr, _up_guard) = up.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    // SAFETY: lengths and alignment are validated above; all slices belong to `ctx.stream`.
    unsafe {
        ffi::dsv4_interleave_gate_up_fp8_rows_cuda(
            gate_ptr as *const u8,
            up_ptr as *const u8,
            output_ptr as *mut u8,
            i32::try_from(rows)?,
            i32::try_from(cols)?,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

/// Returns byte offsets in each rank's symmetric workspace.
pub fn sm90_mega_moe_workspace_layout(
    shape: Sm90MegaMoeShape,
) -> Result<Sm90MegaMoeWorkspaceLayout> {
    let mut layout = Sm90MegaMoeWorkspaceLayout::default();
    // SAFETY: `layout` is a live host output and the FFI reads only scalar inputs.
    unsafe {
        ffi::dsv4_sm90_mega_moe_workspace_layout_cuda(
            i32::try_from(shape.num_ranks)?,
            i32::try_from(shape.num_experts)?,
            i32::try_from(shape.requested_max_tokens_per_rank)?,
            i32::try_from(shape.num_topk)?,
            i32::try_from(shape.hidden)?,
            i32::try_from(shape.intermediate_hidden)?,
            &mut layout,
        )
        .result()?;
    }
    Ok(layout)
}

/// Stages one rank's BF16 tokens and routes into the MegaMoE symmetric input area.
///
/// # Safety
/// The three source pointers must cover contiguous token-major buffers of shapes
/// `[num_tokens, hidden]`, `[num_tokens, topk]`, and `[num_tokens, topk]` on
/// `stream`. `local_workspace` must cover `layout.num_bytes` and remain live
/// until the launch completes. Route weights are copied unchanged.
pub unsafe fn sm90_mega_moe_stage_inputs(
    hidden_states: RawDevicePtr<bf16>,
    route_indices: RawDevicePtr<i32>,
    route_weights: RawDevicePtr<f32>,
    local_workspace: u64,
    layout: &Sm90MegaMoeWorkspaceLayout,
    num_tokens: usize,
    topk: usize,
    hidden: usize,
    stream: CUstream,
) -> Result<()> {
    ensure!(
        local_workspace != 0 && !stream.is_null(),
        "MegaMoE staging needs a workspace and CUDA stream"
    );
    ensure!(
        hidden.is_multiple_of(128) && hidden <= 8192,
        "MegaMoE staging hidden must be a multiple of 128 and <=8192, got {hidden}"
    );
    ensure!(
        topk > 0 && topk <= hidden / 8,
        "MegaMoE staging topk must be in 1..={}, got {topk}",
        hidden / 8
    );
    let padded = usize::try_from(layout.num_max_tokens_per_rank)?;
    ensure!(
        num_tokens <= padded,
        "MegaMoE staging tokens {num_tokens} exceed workspace capacity {padded}"
    );

    let end = |offset: u64, bytes: usize| -> Result<u64> {
        offset
            .checked_add(u64::try_from(bytes)?)
            .ok_or_else(|| anyhow::anyhow!("MegaMoE workspace range overflow"))
    };
    let x_end = end(
        layout.x,
        padded
            .checked_mul(hidden)
            .ok_or_else(|| anyhow::anyhow!("MegaMoE x size overflow"))?,
    )?;
    let x_sf_end = end(
        layout.x_sf,
        padded
            .checked_mul(hidden / 128)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| anyhow::anyhow!("MegaMoE x_sf size overflow"))?,
    )?;
    let topk_slots = padded
        .checked_mul(topk)
        .ok_or_else(|| anyhow::anyhow!("MegaMoE route size overflow"))?;
    let topk_idx_end = end(
        layout.topk_idx,
        topk_slots
            .checked_mul(std::mem::size_of::<i64>())
            .ok_or_else(|| anyhow::anyhow!("MegaMoE route-index size overflow"))?,
    )?;
    let topk_weights_end = end(
        layout.topk_weights,
        topk_slots
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("MegaMoE route-weight size overflow"))?,
    )?;
    ensure!(
        x_end <= layout.x_sf
            && x_sf_end <= layout.topk_idx
            && topk_idx_end <= layout.topk_weights
            && topk_weights_end <= layout.l1_acts
            && layout.l1_acts <= layout.num_bytes,
        "MegaMoE workspace layout does not cover staged inputs"
    );

    let at = |offset: u64| -> Result<u64> {
        local_workspace
            .checked_add(offset)
            .ok_or_else(|| anyhow::anyhow!("MegaMoE workspace pointer overflow"))
    };
    // SAFETY: source and workspace lifetimes are the caller's contract; ranges are checked above.
    unsafe {
        ffi::dsv4_sm90_mega_moe_pre_dispatch_cuda(
            hidden_states.as_ptr().cast(),
            route_indices.as_ptr(),
            route_weights.as_ptr(),
            at(layout.x)? as *mut u8,
            at(layout.x_sf)? as *mut f32,
            at(layout.topk_idx)? as *mut i64,
            at(layout.topk_weights)? as *mut f32,
            i32::try_from(num_tokens)?,
            layout.num_max_tokens_per_rank,
            i32::try_from(hidden)?,
            i32::try_from(topk)?,
            0,
            stream,
        )
        .result()?;
    }
    Ok(())
}

pub struct Sm90MegaMoeLaunch<'a> {
    pub shape: Sm90MegaMoeShape,
    pub workspace: &'a Sm90MegaMoeWorkspaceLayout,
    pub num_tokens: usize,
    pub rank_idx: usize,
    pub y: RawDevicePtr<bf16>,
    pub cumulative_local_expert_recv_stats: Option<RawDevicePtr<i32>>,
    pub peer_buffer_ptrs: &'a [u64],
    pub local_workspace: u64,
    pub activation_clamp: f32,
    pub fast_math: bool,
    pub enable_pdl: bool,
    pub l1_weights: RawDevicePtr<u8>,
    pub l1_weight_stride: usize,
    pub l1_weights_sf: RawDevicePtr<f32>,
    pub l2_weights: RawDevicePtr<u8>,
    pub l2_weight_stride: usize,
    pub l2_weights_sf: RawDevicePtr<f32>,
    pub stream: CUstream,
}

/// Launches the vendored SM90 fused dispatch, L1 SwiGLU, L2, and combine kernel.
///
/// # Safety
/// `peer_buffer_ptrs` must be CUDA symmetric virtual addresses, one per rank, each
/// spanning `workspace.num_bytes`; the kernel mutates its barrier/count metadata,
/// `l1_acts`, `l1_acts_sf`, `l1_topk_weights`, `l2_acts`, `l2_acts_sf`, and
/// `combine` regions. Its `x`, `x_sf`, `topk_idx`, and `topk_weights` regions must
/// contain this step's inputs. `y`, both weight/scale buffers, the local workspace,
/// and optional stats must cover the shape and remain live through `stream`.
/// L1 weights are FP8 K-major `[experts/rank, 2*intermediate, hidden]`; L2 weights
/// are `[experts/rank, hidden, intermediate]`; strides are in FP8 elements.
pub unsafe fn sm90_mega_moe_launch(args: &Sm90MegaMoeLaunch<'_>) -> Result<()> {
    ensure!(
        args.peer_buffer_ptrs.len() == args.shape.num_ranks,
        "SM90 MegaMoE needs one symmetric workspace pointer per rank"
    );
    ensure!(
        args.peer_buffer_ptrs.get(args.rank_idx) == Some(&args.local_workspace),
        "SM90 MegaMoE local workspace must match the rank pointer"
    );
    let stats = args
        .cumulative_local_expert_recv_stats
        .map_or(std::ptr::null_mut(), RawDevicePtr::as_mut_ptr);
    // SAFETY: forwarded from this function's device-buffer and TMA contract.
    unsafe {
        ffi::dsv4_sm90_mega_moe_launch_cuda(
            args.y.as_mut_ptr().cast(),
            stats,
            args.peer_buffer_ptrs.as_ptr(),
            args.local_workspace as *mut u8,
            i32::try_from(args.shape.num_ranks)?,
            i32::try_from(args.rank_idx)?,
            args.workspace.num_max_tokens_per_rank,
            i32::try_from(args.num_tokens)?,
            i32::try_from(args.shape.num_experts)?,
            i32::try_from(args.shape.num_topk)?,
            i32::try_from(args.shape.hidden)?,
            i32::try_from(args.shape.intermediate_hidden)?,
            args.activation_clamp,
            i32::from(args.fast_math),
            i32::from(args.enable_pdl),
            args.l1_weights.as_ptr(),
            i32::try_from(args.l1_weight_stride)?,
            args.l1_weights_sf.as_ptr(),
            args.l2_weights.as_ptr(),
            i32::try_from(args.l2_weight_stride)?,
            args.l2_weights_sf.as_ptr(),
            args.stream,
        )
        .result()?;
    }
    Ok(())
}

/// Build the device-resident per-expert weight-pointer table the grouped-GEMM
/// kernels consume (`*const u64`, one dense `data` pointer per expert).
///
/// # Errors
/// Errors if the host→device upload fails.
pub fn build_expert_weight_ptr_table(
    ctx: &DeviceContext,
    experts: &[&DeviceMatrix],
) -> Result<CudaSlice<u64>> {
    let host: Vec<u64> = experts.iter().map(|m| m.device_ptr(ctx)).collect();
    ctx.stream
        .clone_htod(&host)
        .map_err(|e| anyhow::anyhow!("expert weight-ptr table H2D failed: {e}"))
}

fn build_optional_ptr_table<T>(
    ctx: &DeviceContext,
    experts: &[&DeviceMatrix],
    label: &str,
    get: impl Fn(&DeviceMatrix) -> Option<&CudaSlice<T>>,
) -> Result<CudaSlice<u64>> {
    let host = experts
        .iter()
        .enumerate()
        .map(|(idx, expert)| {
            let slice = get(expert).ok_or_else(|| {
                anyhow::anyhow!(
                    "expert {idx} missing {label} for {} quant table",
                    expert.weight_format
                )
            })?;
            let (ptr, _guard) = slice.device_ptr(&ctx.stream);
            Ok(ptr)
        })
        .collect::<Result<Vec<_>>>()?;
    ctx.stream
        .clone_htod(&host)
        .map_err(|e| anyhow::anyhow!("expert {label} ptr table H2D failed: {e}"))
}

/// Build a per-expert pointer table for ABI-generic uint8 quantized weights.
pub fn build_expert_qweight_u8_ptr_table(
    ctx: &DeviceContext,
    experts: &[&DeviceMatrix],
) -> Result<CudaSlice<u64>> {
    build_optional_ptr_table(ctx, experts, "qweight_u8", |m| m.qweight_u8.as_ref())
}

/// Build a per-expert pointer table for ABI-generic f32 scales.
pub fn build_expert_scale_f32_ptr_table(
    ctx: &DeviceContext,
    experts: &[&DeviceMatrix],
) -> Result<CudaSlice<u64>> {
    build_optional_ptr_table(ctx, experts, "scale_f32", |m| m.scale_f32.as_ref())
}

/// Build a per-expert pointer table for ABI-generic FP8 scale bytes.
pub fn build_expert_qscale_fp8_ptr_table(
    ctx: &DeviceContext,
    experts: &[&DeviceMatrix],
) -> Result<CudaSlice<u64>> {
    build_optional_ptr_table(ctx, experts, "qscale_fp8", |m| m.qscale_fp8.as_ref())
}

/// Build a per-expert pointer table for W4A16 INT4 packed weights (`i8`).
pub fn build_expert_qweight_i8_ptr_table(
    ctx: &DeviceContext,
    experts: &[&DeviceMatrix],
) -> Result<CudaSlice<u64>> {
    build_optional_ptr_table(ctx, experts, "qweight_i8", |m| m.qweight.as_ref())
}

/// Build a per-expert pointer table for W4A16 BF16 per-group scales.
pub fn build_expert_qscale_bf16_ptr_table(
    ctx: &DeviceContext,
    experts: &[&DeviceMatrix],
) -> Result<CudaSlice<u64>> {
    build_optional_ptr_table(ctx, experts, "qscale_bf16", |m| m.qscales.as_ref())
}

/// Single grouped expert GEMM: `output[token] = input[token] @ W_expert^T`,
/// M-grouped by expert. `weight_ptrs` is the [`build_expert_weight_ptr_table`]
/// table; `n`/`k` are one expert's `[n, k]` weight dims.
///
/// # Safety
/// All buffers must be valid on `stream` with lengths matching the shape (checked
/// by the kernel, not here).
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_bf16_grouped_gemm_batch(
    weight_ptrs: &CudaSlice<u64>,
    input: RawDevicePtr<bf16>,
    output: RawDevicePtr<bf16>,
    offsets: RawDevicePtr<i32>,
    counts: RawDevicePtr<i32>,
    expert_indices: RawDevicePtr<i32>,
    num_experts: usize,
    max_count: usize,
    n: usize,
    k: usize,
    ctx: &DeviceContext,
    stream: CUstream,
) -> Result<()> {
    let (wp, _g) = weight_ptrs.device_ptr(&ctx.stream);
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract;
    // the expert-pointer tables are live CudaSlices pinned by the `_g*` guards.
    unsafe {
        ffi::moe_bf16_grouped_gemm_batch_cuda(
            wp as *const u64,
            input.as_ptr() as *const Half,
            output.as_mut_ptr() as *mut Half,
            offsets.as_ptr(),
            counts.as_ptr(),
            expert_indices.as_ptr(),
            i32::try_from(num_experts)?,
            i32::try_from(max_count)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Paired grouped expert GEMM (gate + up in one launch). Wraps
/// [`ffi::moe_bf16_grouped_gemm_pair_batch_cuda`] — the Qwen3.6 SwiGLU
/// `gate_proj` + `up_proj` fused dispatch.
///
/// # Safety
/// See [`moe_bf16_grouped_gemm_batch`]; both weight tables and both output
/// buffers must be valid on `stream` for the given shape.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_bf16_grouped_gemm_pair_batch(
    weight_a_ptrs: &CudaSlice<u64>,
    weight_b_ptrs: &CudaSlice<u64>,
    input: RawDevicePtr<bf16>,
    output_a: RawDevicePtr<bf16>,
    output_b: RawDevicePtr<bf16>,
    offsets: RawDevicePtr<i32>,
    counts: RawDevicePtr<i32>,
    expert_indices: RawDevicePtr<i32>,
    num_experts: usize,
    max_count: usize,
    n: usize,
    k: usize,
    ctx: &DeviceContext,
    stream: CUstream,
) -> Result<()> {
    let (wa, _ga) = weight_a_ptrs.device_ptr(&ctx.stream);
    let (wb, _gb) = weight_b_ptrs.device_ptr(&ctx.stream);
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract;
    // the expert-pointer tables are live CudaSlices pinned by the `_g*` guards.
    unsafe {
        ffi::moe_bf16_grouped_gemm_pair_batch_cuda(
            wa as *const u64,
            wb as *const u64,
            input.as_ptr() as *const Half,
            output_a.as_mut_ptr() as *mut Half,
            output_b.as_mut_ptr() as *mut Half,
            offsets.as_ptr(),
            counts.as_ptr(),
            expert_indices.as_ptr(),
            i32::try_from(num_experts)?,
            i32::try_from(max_count)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Decode-band single grouped expert GEMM (down projection): weight-read-bound
/// — one warp per weight row, 16B-vector loads, each touched expert's `[n, k]`
/// weight read exactly once per ≤8-row activation chunk regardless of the
/// routed-row count. Wraps [`ffi::moe_bf16_grouped_gemm_decode_cuda`].
///
/// # Errors
/// Errors unless `k % 8 == 0` (16-byte vector rows) — callers dispatch to
/// [`moe_bf16_grouped_gemm_batch`] for unaligned shapes.
///
/// # Safety
/// See [`moe_bf16_grouped_gemm_batch`]: all buffers must be valid on `stream`
/// for the shape (input `[num_routes, k]`, output `[num_routes, n]`,
/// offsets/counts/expert_indices `[num_experts]`).
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_bf16_grouped_gemm_decode(
    weight_ptrs: &CudaSlice<u64>,
    input: RawDevicePtr<bf16>,
    output: RawDevicePtr<bf16>,
    offsets: RawDevicePtr<i32>,
    counts: RawDevicePtr<i32>,
    expert_indices: RawDevicePtr<i32>,
    num_experts: usize,
    max_count: usize,
    n: usize,
    k: usize,
    ctx: &DeviceContext,
    stream: CUstream,
) -> Result<()> {
    ensure!(
        k.is_multiple_of(8),
        "decode grouped GEMM needs k % 8 == 0 for 16B vector loads, got n={n} k={k}"
    );
    let (wp, _g) = weight_ptrs.device_ptr(&ctx.stream);
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract;
    // the expert-pointer tables are live CudaSlices pinned by the `_g*` guards.
    unsafe {
        ffi::moe_bf16_grouped_gemm_decode_cuda(
            wp as *const u64,
            input.as_ptr() as *const Half,
            output.as_mut_ptr() as *mut Half,
            offsets.as_ptr(),
            counts.as_ptr(),
            expert_indices.as_ptr(),
            i32::try_from(num_experts)?,
            i32::try_from(max_count)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Decode-band fused gate+up+SwiGLU grouped expert GEMM: reads each touched
/// expert's gate and up `[n, k]` matrices exactly once per ≤8-row activation
/// chunk and writes `act = silu(gate·x) * (up·x)` directly (the separate
/// `silu_mul` pass folds into the epilogue; silu applies to the fp32
/// accumulators before the single bf16 round). Wraps
/// [`ffi::moe_bf16_grouped_gemm_swiglu_decode_cuda`].
///
/// # Errors
/// Errors unless `k % 8 == 0` — see [`moe_bf16_grouped_gemm_decode`].
///
/// # Safety
/// See [`moe_bf16_grouped_gemm_decode`]; both weight tables and the single
/// `act` output `[num_routes, n]` must be valid on `stream` for the shape.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_bf16_grouped_gemm_swiglu_decode(
    weight_gate_ptrs: &CudaSlice<u64>,
    weight_up_ptrs: &CudaSlice<u64>,
    input: RawDevicePtr<bf16>,
    act: RawDevicePtr<bf16>,
    offsets: RawDevicePtr<i32>,
    counts: RawDevicePtr<i32>,
    expert_indices: RawDevicePtr<i32>,
    num_experts: usize,
    max_count: usize,
    n: usize,
    k: usize,
    ctx: &DeviceContext,
    stream: CUstream,
) -> Result<()> {
    ensure!(
        k.is_multiple_of(8),
        "decode swiglu grouped GEMM needs k % 8 == 0 for 16B vector loads, got n={n} k={k}"
    );
    let (wg, _gg) = weight_gate_ptrs.device_ptr(&ctx.stream);
    let (wu, _gu) = weight_up_ptrs.device_ptr(&ctx.stream);
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract;
    // the expert-pointer tables are live CudaSlices pinned by the `_g*` guards.
    unsafe {
        ffi::moe_bf16_grouped_gemm_swiglu_decode_cuda(
            wg as *const u64,
            wu as *const u64,
            input.as_ptr() as *const Half,
            act.as_mut_ptr() as *mut Half,
            offsets.as_ptr(),
            counts.as_ptr(),
            expert_indices.as_ptr(),
            i32::try_from(num_experts)?,
            i32::try_from(max_count)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// ABI-generic FP8 block-scaled grouped expert GEMM.
///
/// # Safety
/// `weight_ptrs` and `scale_ptrs` must contain valid device pointers for every
/// routed expert. `input`, `output`, `offsets`, `counts`, and `expert_indices`
/// must be live on `stream`; each route row selected by the offset/count tables
/// must address `[K]` input elements and `[N]` output elements.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_fp8_block_scaled_grouped_gemv_batch(
    weight_ptrs: &CudaSlice<u64>,
    scale_ptrs: &CudaSlice<u64>,
    input: RawDevicePtr<bf16>,
    output: RawDevicePtr<bf16>,
    offsets: RawDevicePtr<i32>,
    counts: RawDevicePtr<i32>,
    expert_indices: RawDevicePtr<i32>,
    num_experts: usize,
    max_count: usize,
    n: usize,
    k: usize,
    scale_rows: usize,
    scale_cols: usize,
    block_m: usize,
    block_k: usize,
    ctx: &DeviceContext,
    stream: CUstream,
) -> Result<()> {
    let (wp, _g) = weight_ptrs.device_ptr(&ctx.stream);
    let (sp, _sg) = scale_ptrs.device_ptr(&ctx.stream);
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract;
    // the expert-pointer tables are live CudaSlices pinned by the `_g*` guards.
    unsafe {
        ffi::moe_fp8_block_scaled_grouped_gemv_batch_cuda(
            wp as *const u64,
            sp as *const u64,
            input.as_ptr() as *const Half,
            output.as_mut_ptr() as *mut Half,
            offsets.as_ptr(),
            counts.as_ptr(),
            expert_indices.as_ptr(),
            i32::try_from(num_experts)?,
            i32::try_from(max_count)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(scale_rows)?,
            i32::try_from(scale_cols)?,
            i32::try_from(block_m)?,
            i32::try_from(block_k)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// ABI-generic FP8 block-scaled paired grouped expert GEMM.
///
/// # Safety
/// `weight_*_ptrs` and `scale_*_ptrs` must contain valid device pointers for
/// every routed expert. `input`, both outputs, and the route tables must be live
/// on `stream`; each selected route row must address `[K]` input elements and
/// `[N]` elements in both outputs.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_fp8_block_scaled_grouped_gemv_pair_batch(
    weight_a_ptrs: &CudaSlice<u64>,
    scale_a_ptrs: &CudaSlice<u64>,
    weight_b_ptrs: &CudaSlice<u64>,
    scale_b_ptrs: &CudaSlice<u64>,
    input: RawDevicePtr<bf16>,
    output_a: RawDevicePtr<bf16>,
    output_b: RawDevicePtr<bf16>,
    offsets: RawDevicePtr<i32>,
    counts: RawDevicePtr<i32>,
    expert_indices: RawDevicePtr<i32>,
    num_experts: usize,
    max_count: usize,
    n: usize,
    k: usize,
    scale_rows: usize,
    scale_cols: usize,
    block_m: usize,
    block_k: usize,
    ctx: &DeviceContext,
    stream: CUstream,
) -> Result<()> {
    let (wa, _ga) = weight_a_ptrs.device_ptr(&ctx.stream);
    let (sa, _gsa) = scale_a_ptrs.device_ptr(&ctx.stream);
    let (wb, _gb) = weight_b_ptrs.device_ptr(&ctx.stream);
    let (sb, _gsb) = scale_b_ptrs.device_ptr(&ctx.stream);
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract;
    // the expert-pointer tables are live CudaSlices pinned by the `_g*` guards.
    unsafe {
        ffi::moe_fp8_block_scaled_grouped_gemv_pair_batch_cuda(
            wa as *const u64,
            sa as *const u64,
            wb as *const u64,
            sb as *const u64,
            input.as_ptr() as *const Half,
            output_a.as_mut_ptr() as *mut Half,
            output_b.as_mut_ptr() as *mut Half,
            offsets.as_ptr(),
            counts.as_ptr(),
            expert_indices.as_ptr(),
            i32::try_from(num_experts)?,
            i32::try_from(max_count)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(scale_rows)?,
            i32::try_from(scale_cols)?,
            i32::try_from(block_m)?,
            i32::try_from(block_k)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// ABI-generic FP4 E2M1 grouped expert GEMM.
///
/// # Safety
/// `weight_ptrs`, `scale_ptrs`, and `global_ptrs` must contain valid device
/// pointers for every routed expert. `input`, `output`, `offsets`, `counts`, and
/// `expert_indices` must be live on `stream`; each route row selected by the
/// tables must address `[K]` input elements and `[N]` output elements.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_fp4_e2m1_grouped_gemv_batch(
    weight_ptrs: &CudaSlice<u64>,
    scale_ptrs: &CudaSlice<u64>,
    global_ptrs: &CudaSlice<u64>,
    input: RawDevicePtr<bf16>,
    output: RawDevicePtr<bf16>,
    offsets: RawDevicePtr<i32>,
    counts: RawDevicePtr<i32>,
    expert_indices: RawDevicePtr<i32>,
    num_experts: usize,
    max_count: usize,
    n: usize,
    k: usize,
    group_size: usize,
    scale_cols: usize,
    ctx: &DeviceContext,
    stream: CUstream,
) -> Result<()> {
    let (wp, _gw) = weight_ptrs.device_ptr(&ctx.stream);
    let (sp, _gs) = scale_ptrs.device_ptr(&ctx.stream);
    let (gp, _gg) = global_ptrs.device_ptr(&ctx.stream);
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract;
    // the expert-pointer tables are live CudaSlices pinned by the `_g*` guards.
    unsafe {
        ffi::moe_fp4_e2m1_grouped_gemv_batch_cuda(
            wp as *const u64,
            sp as *const u64,
            gp as *const u64,
            input.as_ptr() as *const Half,
            output.as_mut_ptr() as *mut Half,
            offsets.as_ptr(),
            counts.as_ptr(),
            expert_indices.as_ptr(),
            i32::try_from(num_experts)?,
            i32::try_from(max_count)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(group_size)?,
            i32::try_from(scale_cols)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// ABI-generic FP4 E2M1 paired grouped expert GEMM.
///
/// # Safety
/// `weight_*_ptrs`, `scale_*_ptrs`, and `global_*_ptrs` must contain valid
/// device pointers for every routed expert. `input`, both outputs, and the route
/// tables must be live on `stream`; each selected route row must address `[K]`
/// input elements and `[N]` elements in both outputs.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_fp4_e2m1_grouped_gemv_pair_batch(
    weight_a_ptrs: &CudaSlice<u64>,
    scale_a_ptrs: &CudaSlice<u64>,
    global_a_ptrs: &CudaSlice<u64>,
    weight_b_ptrs: &CudaSlice<u64>,
    scale_b_ptrs: &CudaSlice<u64>,
    global_b_ptrs: &CudaSlice<u64>,
    input: RawDevicePtr<bf16>,
    output_a: RawDevicePtr<bf16>,
    output_b: RawDevicePtr<bf16>,
    offsets: RawDevicePtr<i32>,
    counts: RawDevicePtr<i32>,
    expert_indices: RawDevicePtr<i32>,
    num_experts: usize,
    max_count: usize,
    n: usize,
    k: usize,
    group_size: usize,
    scale_cols: usize,
    ctx: &DeviceContext,
    stream: CUstream,
) -> Result<()> {
    let (wa, _gwa) = weight_a_ptrs.device_ptr(&ctx.stream);
    let (sa, _gsa) = scale_a_ptrs.device_ptr(&ctx.stream);
    let (ga, _gga) = global_a_ptrs.device_ptr(&ctx.stream);
    let (wb, _gwb) = weight_b_ptrs.device_ptr(&ctx.stream);
    let (sb, _gsb) = scale_b_ptrs.device_ptr(&ctx.stream);
    let (gb, _ggb) = global_b_ptrs.device_ptr(&ctx.stream);
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract;
    // the expert-pointer tables are live CudaSlices pinned by the `_g*` guards.
    unsafe {
        ffi::moe_fp4_e2m1_grouped_gemv_pair_batch_cuda(
            wa as *const u64,
            sa as *const u64,
            ga as *const u64,
            wb as *const u64,
            sb as *const u64,
            gb as *const u64,
            input.as_ptr() as *const Half,
            output_a.as_mut_ptr() as *mut Half,
            output_b.as_mut_ptr() as *mut Half,
            offsets.as_ptr(),
            counts.as_ptr(),
            expert_indices.as_ptr(),
            i32::try_from(num_experts)?,
            i32::try_from(max_count)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(group_size)?,
            i32::try_from(scale_cols)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// ABI-generic W4A16 grouped expert GEMV (INT4 weights, BF16 per-group scales,
/// zero-point 8). `N` = expert output rows, `K` = input cols, `group_size` = the
/// BF16 scale group size along K.
///
/// # Safety
/// `weight_ptrs` and `scale_ptrs` must contain valid device pointers for every
/// routed expert. `input`, `output`, `offsets`, `counts`, and `expert_indices`
/// must be live on `stream`; each route row selected by the tables must address
/// `[K]` input elements and `[N]` output elements.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_w4a16_grouped_gemv_batch(
    weight_ptrs: &CudaSlice<u64>,
    scale_ptrs: &CudaSlice<u64>,
    input: RawDevicePtr<bf16>,
    output: RawDevicePtr<bf16>,
    offsets: RawDevicePtr<i32>,
    counts: RawDevicePtr<i32>,
    expert_indices: RawDevicePtr<i32>,
    num_experts: usize,
    max_count: usize,
    n: usize,
    k: usize,
    group_size: usize,
    ctx: &DeviceContext,
    stream: CUstream,
) -> Result<()> {
    let (wp, _gw) = weight_ptrs.device_ptr(&ctx.stream);
    let (sp, _gs) = scale_ptrs.device_ptr(&ctx.stream);
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract;
    // the expert-pointer tables are live CudaSlices pinned by the `_g*` guards.
    unsafe {
        ffi::moe_w4a16_grouped_gemv_batch_cuda(
            wp as *const u64,
            sp as *const u64,
            input.as_ptr() as *const Half,
            output.as_mut_ptr() as *mut Half,
            offsets.as_ptr(),
            counts.as_ptr(),
            expert_indices.as_ptr(),
            i32::try_from(num_experts)?,
            i32::try_from(max_count)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(group_size)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// ABI-generic W4A16 paired grouped expert GEMV (gate + up in one launch).
///
/// # Safety
/// `weight_*_ptrs` and `scale_*_ptrs` must contain valid device pointers for
/// every routed expert. `input`, both outputs, and the route tables must be live
/// on `stream`; each selected route row must address `[K]` input elements and
/// `[N]` elements in both outputs.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_w4a16_grouped_gemv_pair_batch(
    weight_a_ptrs: &CudaSlice<u64>,
    scale_a_ptrs: &CudaSlice<u64>,
    weight_b_ptrs: &CudaSlice<u64>,
    scale_b_ptrs: &CudaSlice<u64>,
    input: RawDevicePtr<bf16>,
    output_a: RawDevicePtr<bf16>,
    output_b: RawDevicePtr<bf16>,
    offsets: RawDevicePtr<i32>,
    counts: RawDevicePtr<i32>,
    expert_indices: RawDevicePtr<i32>,
    num_experts: usize,
    max_count: usize,
    n: usize,
    k: usize,
    group_size: usize,
    ctx: &DeviceContext,
    stream: CUstream,
) -> Result<()> {
    let (wa, _gwa) = weight_a_ptrs.device_ptr(&ctx.stream);
    let (sa, _gsa) = scale_a_ptrs.device_ptr(&ctx.stream);
    let (wb, _gwb) = weight_b_ptrs.device_ptr(&ctx.stream);
    let (sb, _gsb) = scale_b_ptrs.device_ptr(&ctx.stream);
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract;
    // the expert-pointer tables are live CudaSlices pinned by the `_g*` guards.
    unsafe {
        ffi::moe_w4a16_grouped_gemv_pair_batch_cuda(
            wa as *const u64,
            sa as *const u64,
            wb as *const u64,
            sb as *const u64,
            input.as_ptr() as *const Half,
            output_a.as_mut_ptr() as *mut Half,
            output_b.as_mut_ptr() as *mut Half,
            offsets.as_ptr(),
            counts.as_ptr(),
            expert_indices.as_ptr(),
            i32::try_from(num_experts)?,
            i32::try_from(max_count)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(group_size)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Count how many routed tokens fall on each local expert.
/// Wraps [`ffi::dsv4_count_local_experts_cuda`].
///
/// # Safety
/// `indices` / `counts` must be valid on `stream` for the given shape.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_count_local_experts(
    indices: RawDevicePtr<i32>,
    counts: RawDevicePtr<i32>,
    num_tokens: usize,
    topk: usize,
    local_expert_start: usize,
    experts_per_rank: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_count_local_experts_cuda(
            indices.as_ptr(),
            counts.as_mut_ptr(),
            i32::try_from(num_tokens)?,
            i32::try_from(topk)?,
            i32::try_from(local_expert_start)?,
            i32::try_from(experts_per_rank)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// DSv4 on-device router: router logits `[num_tokens, n_experts]` to flat
/// token-major `indices` / `weights` (`[num_tokens * topk]`). `routing_kind=0`
/// is hash routing and requires `tid2eid` + `token_ids`; `routing_kind=1` is
/// learned-bias noaux routing and requires `bias`.
///
/// # Safety
/// All pointers must be valid on `stream` for the given shape. Optional
/// pointers are only dereferenced by the matching routing-kind branch.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_route(
    logits: RawDevicePtr<bf16>,
    bias: Option<RawDevicePtr<bf16>>,
    tid2eid: Option<RawDevicePtr<i64>>,
    token_ids: Option<RawDevicePtr<u32>>,
    indices: RawDevicePtr<i32>,
    weights: RawDevicePtr<f32>,
    num_tokens: usize,
    n_experts: usize,
    topk: usize,
    routing_kind: i32,
    scoring_kind: i32,
    routed_scaling_factor: f32,
    stream: CUstream,
) -> Result<()> {
    ensure!(
        routing_kind == 0 || routing_kind == 1,
        "DSv4 route routing_kind must be 0(hash) or 1(learned-bias), got {routing_kind}"
    );
    ensure!(
        scoring_kind == 0 || scoring_kind == 1 || scoring_kind == 2,
        "DSv4 route scoring_kind must be 0/1/2, got {scoring_kind}"
    );
    ensure!(n_experts > 0, "DSv4 route n_experts must be > 0");
    ensure!(topk > 0, "DSv4 route topk must be > 0");
    let bias_ptr = match (routing_kind, bias) {
        (1, Some(ptr)) => ptr.as_ptr() as *const Half,
        (1, None) => anyhow::bail!("DSv4 learned-bias route requires bias pointer"),
        _ => std::ptr::null(),
    };
    let tid2eid_ptr = match (routing_kind, tid2eid) {
        (0, Some(ptr)) => ptr.as_ptr(),
        (0, None) => anyhow::bail!("DSv4 hash route requires tid2eid pointer"),
        _ => std::ptr::null(),
    };
    let token_ids_ptr = match (routing_kind, token_ids) {
        (0, Some(ptr)) => ptr.as_ptr(),
        (0, None) => anyhow::bail!("DSv4 hash route requires token_ids pointer"),
        _ => std::ptr::null(),
    };
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract;
    // the optional pointers were validated per routing_kind above (a null is
    // never dereferenced by the branch that receives it).
    unsafe {
        ffi::dsv4_route_cuda(
            logits.as_ptr() as *const Half,
            bias_ptr,
            tid2eid_ptr,
            token_ids_ptr,
            indices.as_mut_ptr(),
            weights.as_mut_ptr(),
            i32::try_from(num_tokens)?,
            i32::try_from(n_experts)?,
            i32::try_from(topk)?,
            routing_kind,
            scoring_kind,
            routed_scaling_factor,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Cast a flat `i32` device buffer to `i64` on-device.
///
/// # Safety
/// `src` / `dst` must be valid on `stream` for `n` elements.
pub unsafe fn dsv4_cast_i32_to_i64(
    src: RawDevicePtr<i32>,
    dst: RawDevicePtr<i64>,
    n: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_cast_i32_to_i64_cuda(src.as_ptr(), dst.as_mut_ptr(), i32::try_from(n)?, stream)
            .result()?;
    }
    Ok(())
}

/// Cast a flat `i64` device buffer to `i32` on-device.
///
/// # Safety
/// `src` / `dst` must be valid on `stream` for `n` elements. Values must fit
/// in `i32` (expert ids always do).
pub unsafe fn dsv4_cast_i64_to_i32(
    src: RawDevicePtr<i64>,
    dst: RawDevicePtr<i32>,
    n: usize,
    stream: CUstream,
) -> Result<()> {
    unsafe {
        ffi::dsv4_cast_i64_to_i32_cuda(src.as_ptr(), dst.as_mut_ptr(), i32::try_from(n)?, stream)
            .result()?;
    }
    Ok(())
}

/// Exclusive prefix-sum over per-expert counts with each group's span padded
/// up to `alignment` rows — the DeepGEMM m-grouped-contiguous layout builder
/// (group segments must start BLOCK_M=128-aligned because the kernel resolves
/// the per-tile B group from `m_indices[tile_start]`). Wraps
/// [`ffi::moe_exclusive_scan_aligned_i32_cuda`].
///
/// # Safety
/// `counts` / `offsets` / `total` must be valid on `stream`; `offsets` holds
/// `n` entries and `total` one entry.
pub unsafe fn moe_exclusive_scan_aligned_i32(
    counts: RawDevicePtr<i32>,
    offsets: RawDevicePtr<i32>,
    total: RawDevicePtr<i32>,
    n: usize,
    alignment: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::moe_exclusive_scan_aligned_i32_cuda(
            counts.as_ptr(),
            offsets.as_mut_ptr(),
            total.as_mut_ptr(),
            i32::try_from(n)?,
            i32::try_from(alignment)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Exclusive prefix-sum over per-expert counts → offsets (+ total).
/// Wraps [`ffi::dsv4_exclusive_scan_i32_cuda`].
///
/// # Safety
/// `counts` / `offsets` / `total` must be valid on `stream`; `offsets` holds
/// `n` entries and `total` one entry.
pub unsafe fn dsv4_exclusive_scan_i32(
    counts: RawDevicePtr<i32>,
    offsets: RawDevicePtr<i32>,
    total: RawDevicePtr<i32>,
    n: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_exclusive_scan_i32_cuda(
            counts.as_ptr(),
            offsets.as_mut_ptr(),
            total.as_mut_ptr(),
            i32::try_from(n)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Pack routed tokens into per-local-expert contiguous slots (with route slots).
/// Wraps [`ffi::dsv4_pack_local_experts_with_slots_cuda`].
///
/// # Safety
/// All pointers must be valid on `stream` for the given shape; `cursors` is the
/// per-expert write cursor scratch.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_pack_local_experts_with_slots(
    hidden: RawDevicePtr<bf16>,
    indices: RawDevicePtr<i32>,
    weights: RawDevicePtr<f32>,
    offsets: RawDevicePtr<i32>,
    cursors: RawDevicePtr<i32>,
    packed_hidden: RawDevicePtr<bf16>,
    packed_route_slot: RawDevicePtr<i32>,
    packed_weight: RawDevicePtr<f32>,
    num_tokens: usize,
    hidden_dim: usize,
    topk: usize,
    local_expert_start: usize,
    experts_per_rank: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_pack_local_experts_with_slots_cuda(
            hidden.as_ptr() as *const Half,
            indices.as_ptr(),
            weights.as_ptr(),
            offsets.as_ptr(),
            cursors.as_mut_ptr(),
            packed_hidden.as_mut_ptr() as *mut Half,
            packed_route_slot.as_mut_ptr(),
            packed_weight.as_mut_ptr(),
            i32::try_from(num_tokens)?,
            i32::try_from(hidden_dim)?,
            i32::try_from(topk)?,
            i32::try_from(local_expert_start)?,
            i32::try_from(experts_per_rank)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Fill DeepGEMM contiguous `m_indices` from compact per-local-expert counts.
///
/// # Safety
/// `counts`, `offsets`, and `m_indices` must be valid on `stream`;
/// `m_indices` must have at least `row_capacity` rows.
pub unsafe fn dsv4_fill_m_indices_from_counts(
    counts: RawDevicePtr<i32>,
    offsets: RawDevicePtr<i32>,
    m_indices: RawDevicePtr<i32>,
    experts_per_rank: usize,
    row_capacity: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_fill_m_indices_from_counts_cuda(
            counts.as_ptr(),
            offsets.as_ptr(),
            m_indices.as_mut_ptr(),
            i32::try_from(experts_per_rank)?,
            i32::try_from(row_capacity)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Pack routed tokens and emit the contiguous DeepGEMM `m_indices` row→local
/// expert map alongside the compact route-slot metadata.
///
/// # Safety
/// All pointers must be valid on `stream`; `packed_m_indices` has the same row
/// capacity as `packed_hidden`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_pack_local_experts_with_slots_and_indices(
    hidden: RawDevicePtr<bf16>,
    indices: RawDevicePtr<i32>,
    weights: RawDevicePtr<f32>,
    offsets: RawDevicePtr<i32>,
    cursors: RawDevicePtr<i32>,
    packed_hidden: RawDevicePtr<bf16>,
    packed_route_slot: RawDevicePtr<i32>,
    packed_weight: RawDevicePtr<f32>,
    packed_m_indices: RawDevicePtr<i32>,
    num_tokens: usize,
    hidden_dim: usize,
    topk: usize,
    local_expert_start: usize,
    experts_per_rank: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_pack_local_experts_with_slots_and_indices_cuda(
            hidden.as_ptr() as *const Half,
            indices.as_ptr(),
            weights.as_ptr(),
            offsets.as_ptr(),
            cursors.as_mut_ptr(),
            packed_hidden.as_mut_ptr() as *mut Half,
            packed_route_slot.as_mut_ptr(),
            packed_weight.as_mut_ptr(),
            packed_m_indices.as_mut_ptr(),
            i32::try_from(num_tokens)?,
            i32::try_from(hidden_dim)?,
            i32::try_from(topk)?,
            i32::try_from(local_expert_start)?,
            i32::try_from(experts_per_rank)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// SwiGLU with the DSv4 route-clamp over packed per-expert rows.
/// Wraps [`ffi::dsv4_swiglu_clamped_routes_cuda`].
///
/// # Safety
/// `gate` / `up` / `out` / `route_meta` must be valid on `stream` for the shape.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_swiglu_clamped_routes(
    gate: RawDevicePtr<bf16>,
    up: RawDevicePtr<bf16>,
    out: RawDevicePtr<bf16>,
    route_meta: RawDevicePtr<i32>,
    num_routes: usize,
    hidden_dim: usize,
    local_expert_start: usize,
    experts_per_rank: usize,
    limit: f32,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_swiglu_clamped_routes_cuda(
            gate.as_ptr() as *const Half,
            up.as_ptr() as *const Half,
            out.as_mut_ptr() as *mut Half,
            route_meta.as_ptr(),
            i32::try_from(num_routes)?,
            i32::try_from(hidden_dim)?,
            i32::try_from(local_expert_start)?,
            i32::try_from(experts_per_rank)?,
            limit,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Scatter every packed expert output back to its route slot (weighted).
/// Wraps [`ffi::dsv4_scatter_all_route_slots_cuda`].
///
/// # Safety
/// `expert_out` / `route_out` / `expert_route_slot` / `expert_weight` must be
/// valid on `stream` for the shape.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_scatter_all_route_slots(
    expert_out: RawDevicePtr<bf16>,
    route_out: RawDevicePtr<bf16>,
    expert_route_slot: RawDevicePtr<i32>,
    expert_weight: RawDevicePtr<f32>,
    num_routes: usize,
    hidden_dim: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_scatter_all_route_slots_cuda(
            expert_out.as_ptr() as *const Half,
            route_out.as_mut_ptr() as *mut Half,
            expert_route_slot.as_ptr(),
            expert_weight.as_ptr(),
            i32::try_from(num_routes)?,
            i32::try_from(hidden_dim)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Combine per-route-slot outputs into per-token routed output (sum over topk).
/// Wraps [`ffi::dsv4_combine_route_slot_outputs_cuda`].
///
/// # Safety
/// `route_slot_out` / `routed_out` must be valid on `stream` for the shape.
pub unsafe fn dsv4_combine_route_slot_outputs(
    route_slot_out: RawDevicePtr<bf16>,
    routed_out: RawDevicePtr<bf16>,
    num_tokens: usize,
    topk: usize,
    hidden_dim: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_combine_route_slot_outputs_cuda(
            route_slot_out.as_ptr() as *const Half,
            routed_out.as_mut_ptr() as *mut Half,
            i32::try_from(num_tokens)?,
            i32::try_from(topk)?,
            i32::try_from(hidden_dim)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Qwen3.6 shared-expert sigmoid-gated accumulate:
/// `routed[t,:] += sigmoid(gate_logit[t]) * shared_y[t,:]`.
/// Wraps [`ffi::qwen36_add_shared_expert_gated_cuda`].
///
/// # Safety
/// `routed` / `shared_y` / `gate_logit` must be valid on `stream` for the shape.
pub unsafe fn qwen36_add_shared_expert_gated(
    routed: RawDevicePtr<bf16>,
    shared_y: RawDevicePtr<bf16>,
    gate_logit: RawDevicePtr<bf16>,
    num_tokens: usize,
    hidden_dim: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::qwen36_add_shared_expert_gated_cuda(
            routed.as_mut_ptr() as *mut Half,
            shared_y.as_ptr() as *const Half,
            gate_logit.as_ptr() as *const Half,
            i32::try_from(num_tokens)?,
            i32::try_from(hidden_dim)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Qwen3.6 `norm_topk_prob` in-place renorm over `dsv4_route` weights:
/// `weights[t*topk + k] /= Σ_k weights[t*topk + k]` (sum zero-guarded at
/// `1e-20`, matching `infer_moe::route`'s step-5 renorm). Skip the launch
/// entirely when `norm_topk_prob` is false (raw softmax probs are the gate
/// weights). Wraps [`ffi::qwen36_renorm_topk_weights_cuda`].
///
/// # Safety
/// `weights` must be valid on `stream` for `num_tokens * topk` f32 elements.
pub unsafe fn qwen36_renorm_topk_weights(
    weights: RawDevicePtr<f32>,
    num_tokens: usize,
    topk: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::qwen36_renorm_topk_weights_cuda(
            weights.as_mut_ptr(),
            i32::try_from(num_tokens)?,
            i32::try_from(topk)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

// The 5-call native DeepGEMM expert path: pack/quantize the packed grouped
// hidden to FP8 → masked grouped GEMM (w13 fused gate+up) → SwiGLU+requant →
// masked grouped GEMM (w2 down) → unpad the padded grouped output back to
// compact rows. `pack`/`swiglu`/`unpad` index per-group via the compact
// `active_experts` / `active_offsets` / `active_counts` metadata (each length
// `active_count`); the masked GEMM reads the dense per-group `masked_m`.

/// Pack + per-128-block quantize the packed grouped BF16 hidden to FP8 padded
/// `[num_groups * max_m, cols]` for the masked GEMM. Wraps
/// [`ffi::dsv4_deepgemm_pack_quantize_bf16_to_fp8_cuda`].
///
/// # Safety
/// All pointers must be valid on `stream` for the given shape; `output` is the
/// FP8 scratch (`u8`), `scales` the FP32 per-block scale scratch.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_deepgemm_pack_quantize_bf16_to_fp8(
    input: RawDevicePtr<bf16>,
    output: RawDevicePtr<u8>,
    scales: RawDevicePtr<f32>,
    active_experts: RawDevicePtr<i32>,
    active_offsets: RawDevicePtr<i32>,
    active_counts: RawDevicePtr<i32>,
    active_count: usize,
    max_m: usize,
    cols: usize,
    scale_stride_m: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_deepgemm_pack_quantize_bf16_to_fp8_cuda(
            input.as_ptr() as *const Half,
            output.as_mut_ptr(),
            scales.as_mut_ptr(),
            active_experts.as_ptr(),
            active_offsets.as_ptr(),
            active_counts.as_ptr(),
            i32::try_from(active_count)?,
            i32::try_from(max_m)?,
            i32::try_from(cols)?,
            i32::try_from(scale_stride_m)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Masked m-grouped FP8 GEMM (`f8f8bf16`, NT): `d = a @ b^T` per group, with
/// each group's valid row count read from `masked_m`. Wraps
/// [`ffi::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked_cuda`]. `a`/`b` are FP8
/// bytes, `sfa`/`sfb` the FP32 128-block scales, `d` the BF16 output.
///
/// # Safety
/// All pointers must be valid on `stream`; `m` is the padded per-group capacity
/// (`max_m`), `n`/`k` the GEMM output/contraction dims, `num_groups` the local
/// expert count.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked(
    a: RawDevicePtr<u8>,
    sfa: RawDevicePtr<f32>,
    b: RawDevicePtr<u8>,
    sfb: RawDevicePtr<f32>,
    d: RawDevicePtr<bf16>,
    masked_m: RawDevicePtr<i32>,
    num_groups: usize,
    m: usize,
    n: usize,
    k: usize,
    sfa_aligned_m: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked_cuda(
            a.as_ptr(),
            sfa.as_ptr(),
            b.as_ptr(),
            sfb.as_ptr(),
            d.as_mut_ptr() as *mut Half,
            masked_m.as_ptr(),
            i32::try_from(num_groups)?,
            i32::try_from(m)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(sfa_aligned_m)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Contiguous m-grouped FP8 GEMM (`f8f8bf16`, NT): `d = a @ b^T` where each
/// activation row's expert group is read from `m_indices`.
///
/// `mk_align` is the caller's per-group row alignment (64 or 128). The
/// contiguous scheduler resolves the B group once per BLOCK_M output tile, so
/// the bridge caps its block_m candidates at `mk_align`; `m` must be a
/// multiple of it. 128 is the upstream-default alignment; 64 halves the pad
/// rows for the DSv4 decode band.
///
/// # Safety
/// All pointers must be valid on `stream`; `m_indices` must contain one local
/// expert id in `[0, num_groups)` per activation row.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous(
    a: RawDevicePtr<u8>,
    sfa: RawDevicePtr<f32>,
    b: RawDevicePtr<u8>,
    sfb: RawDevicePtr<f32>,
    d: RawDevicePtr<bf16>,
    m_indices: RawDevicePtr<i32>,
    num_groups: usize,
    m: usize,
    n: usize,
    k: usize,
    sfa_aligned_m: usize,
    mk_align: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous_cuda(
            a.as_ptr(),
            sfa.as_ptr(),
            b.as_ptr(),
            sfb.as_ptr(),
            d.as_mut_ptr() as *mut Half,
            m_indices.as_ptr(),
            i32::try_from(num_groups)?,
            i32::try_from(m)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(sfa_aligned_m)?,
            i32::try_from(mk_align)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// D2H a pair of small device `[i32; len]` arrays (raw pointers) to host,
/// stream-ordered then synced with a SINGLE `synchronize`. Lets the sm_120 MoE
/// caller read the shared group offsets/counts once per layer instead of the
/// kernel re-reading + syncing on every GEMM call.
pub fn dtoh_i32_pair(
    ctx: &DeviceContext,
    p0: RawDevicePtr<i32>,
    p1: RawDevicePtr<i32>,
    len: usize,
) -> Result<(Vec<i32>, Vec<i32>)> {
    let mut a = vec![0i32; len];
    let mut b = vec![0i32; len];
    let stream = ctx.stream.cu_stream();
    // SAFETY: p0/p1 are valid device `[i32; len]` finalized on ctx.stream.
    unsafe {
        cudarc::driver::result::memcpy_dtoh_async(&mut a, p0.as_ptr() as usize as u64, stream)?;
        cudarc::driver::result::memcpy_dtoh_async(&mut b, p1.as_ptr() as usize as u64, stream)?;
    }
    ctx.stream.synchronize()?;
    Ok((a, b))
}

/// sm_120 (Blackwell) grouped blockwise-scaled FP8 GEMM (NT): per group `g`,
/// `d[rows(g), :] = a[rows(g), :] @ b[g]^T` with DeepSeek blockwise scaling
/// (per-token A scale 1x128 + 128x128 B block scale). The sm_120 replacement
/// for the Hopper-only [`dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous`] on the
/// SAME grouped FP8 buffers.
///
/// `a`/`b` are E4M3 bytes (`a` contiguous `[total_M, k]`, `b` grouped `[G, n, k]`
/// row-major), `sfa` the per-token f32 scales in DeepGEMM packing (K-block
/// leading dim `scale_stride_m`), `sfb` the f32 128-block weight scales already
/// N-contiguous per group (`n_block + k_block*n_blocks` — the loader transposes
/// the checkpoint layout at load on sm_120). `d` is BF16 out `[total_M, n]`.
/// `host_offsets`/`host_counts` are HOST-resident slices of the 128-aligned
/// per-group row start and the real per-group row count (length `num_groups`);
/// the caller D2H's them once per layer (see [`dtoh_i32_pair`]) and reuses them
/// across the w13 + down GEMMs, so the kernel does no per-call D2H/sync.
///
/// # Safety
/// Device pointers must be valid on `stream` for the shapes above; `n`/`k` are
/// 128-aligned; `host_offsets[g] + host_counts[g] <= total_M`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn arle_fp8_moe_grouped_gemm_nt_sm120(
    a: RawDevicePtr<u8>,
    sfa: RawDevicePtr<f32>,
    b: RawDevicePtr<u8>,
    sfb: RawDevicePtr<f32>,
    d: RawDevicePtr<bf16>,
    host_offsets: &[i32],
    host_counts: &[i32],
    num_groups: usize,
    n: usize,
    k: usize,
    scale_stride_m: usize,
    stream: CUstream,
) -> Result<()> {
    debug_assert_eq!(host_offsets.len(), num_groups);
    debug_assert_eq!(host_counts.len(), num_groups);
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract.
    unsafe {
        ffi::arle_fp8_moe_grouped_gemm_nt_sm120_cuda(
            a.as_ptr(),
            sfa.as_ptr(),
            b.as_ptr(),
            sfb.as_ptr(),
            d.as_mut_ptr() as *mut Half,
            host_offsets.as_ptr(),
            host_counts.as_ptr(),
            i32::try_from(num_groups)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(scale_stride_m)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// SM90 BF16 m-grouped masked GEMM (NT): per group `g`,
/// `d[g, 0..masked_m[g], :] = a[g, 0..masked_m[g], :] @ b[g]^T`. `a` is the
/// per-group padded band `[num_groups, m, k]`, `b` the grouped weights
/// `[num_groups, n, k]`, `d` `[num_groups, m, n]`, all row-major BF16.
/// `expected_m` is a heuristics-only block-size hint (per-group expected
/// valid rows). The native bridge rejects `m % 128 != 0` — the output TMA
/// store writes full BLOCK_M (64/128) tiles, so a smaller band capacity would
/// let a tile cross into the next group's band.
///
/// # Safety
/// All pointers must be valid on `stream` for the shapes above; `masked_m`
/// holds `num_groups` entries, each `<= m`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn deepgemm_m_grouped_bf16_gemm_nt_masked(
    a: RawDevicePtr<bf16>,
    b: RawDevicePtr<bf16>,
    d: RawDevicePtr<bf16>,
    masked_m: RawDevicePtr<i32>,
    num_groups: usize,
    m: usize,
    n: usize,
    k: usize,
    expected_m: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::deepgemm_m_grouped_bf16_gemm_nt_masked_cuda(
            a.as_ptr() as *const Half,
            b.as_ptr() as *const Half,
            d.as_mut_ptr() as *mut Half,
            masked_m.as_ptr(),
            i32::try_from(num_groups)?,
            i32::try_from(m)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(expected_m)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// SM90 BF16 m-grouped contiguous GEMM (NT): `d = a @ b[g]^T` where each
/// activation row's group is `m_indices[row]` (`-1` = padding). `a` is
/// `[m, k]`, `b` `[num_groups, n, k]`, `d` `[m, n]`, all row-major BF16.
///
/// CONTRACT (DeepGEMM scheduler): the kernel reads the B group ONCE per
/// BLOCK_M=128 output tile from `m_indices[tile_start]`, so every group's row
/// segment must start 128-aligned (build the layout with
/// [`moe_exclusive_scan_aligned_i32`]) and pad rows must carry `-1`. Pad-row
/// outputs are computed against group 0 and are garbage — the caller excludes
/// them via the packed route-slot `-1` sentinel (the scatter skips them).
///
/// # Safety
/// All pointers must be valid on `stream` for the shapes above; `m_indices`
/// holds `m` entries in `[-1, num_groups)`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn deepgemm_m_grouped_bf16_gemm_nt_contiguous(
    a: RawDevicePtr<bf16>,
    b: RawDevicePtr<bf16>,
    d: RawDevicePtr<bf16>,
    m_indices: RawDevicePtr<i32>,
    num_groups: usize,
    m: usize,
    n: usize,
    k: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::deepgemm_m_grouped_bf16_gemm_nt_contiguous_cuda(
            a.as_ptr() as *const Half,
            b.as_ptr() as *const Half,
            d.as_mut_ptr() as *mut Half,
            m_indices.as_ptr(),
            i32::try_from(num_groups)?,
            i32::try_from(m)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Dense FP8 DeepGEMM (`f8f8bf16`, NT): `d = a @ b^T`, where `a` is a
/// row-major FP8 activation matrix `[m, k]`, `b` is a row-major FP8 weight
/// cache `[n, k]`, and `sfa` / `sfb` are FP32 128-block scales.
///
/// # Safety
/// All pointers must be valid on `stream`; `sfa_aligned_m` is the TMA-aligned
/// leading dimension of the activation scale matrix.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_deepgemm_fp8_gemm_nt(
    a: RawDevicePtr<u8>,
    sfa: RawDevicePtr<f32>,
    b: RawDevicePtr<u8>,
    sfb: RawDevicePtr<f32>,
    d: RawDevicePtr<bf16>,
    m: usize,
    n: usize,
    k: usize,
    sfa_aligned_m: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_deepgemm_fp8_gemm_nt_cuda(
            a.as_ptr(),
            sfa.as_ptr(),
            b.as_ptr(),
            sfb.as_ptr(),
            d.as_mut_ptr() as *mut Half,
            i32::try_from(m)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(sfa_aligned_m)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// DeepGEMM DSA indexer metadata for the official FP8 paged-MQA logits kernel.
///
/// `context_lens` is `[batch_size, next_n]` i32 and `schedule_metadata` is
/// `[(num_sms + 1), 2]` i32. This is the raw native bridge over vendored
/// DeepGEMM; it does not perform top-k.
///
/// # Safety
/// All pointers must be valid on `stream`; `schedule_metadata` must have room
/// for `(num_sms + 1) * 2` i32 values.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_deepgemm_paged_mqa_logits_metadata(
    context_lens: RawDevicePtr<i32>,
    schedule_metadata: RawDevicePtr<i32>,
    batch_size: usize,
    next_n: usize,
    block_kv: usize,
    num_sms: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_deepgemm_paged_mqa_logits_metadata_cuda(
            context_lens.as_ptr(),
            schedule_metadata.as_mut_ptr(),
            i32::try_from(batch_size)?,
            i32::try_from(next_n)?,
            i32::try_from(block_kv)?,
            i32::try_from(num_sms)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Official DeepGEMM FP8 paged-MQA logits for the DSv4 DSA indexer.
///
/// This computes logits only. Query/cache FP8 packing and top-k are separate
/// official/OSS adoption pieces.
///
/// # Safety
/// All pointers must be valid on `stream`. Layouts must match the DeepGEMM
/// contract: `q=[B,next_n,H,D]` E4M3, `kv_cache=[blocks,64,1,D+4]` byte
/// storage split into FP8 values and FP32 scales, `weights=[B*next_n,H]`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_deepgemm_fp8_paged_mqa_logits(
    q: RawDevicePtr<u8>,
    kv_cache: RawDevicePtr<u8>,
    kv_cache_scales: RawDevicePtr<f32>,
    weights: RawDevicePtr<f32>,
    context_lens: RawDevicePtr<i32>,
    block_table: RawDevicePtr<i32>,
    schedule_meta: RawDevicePtr<i32>,
    logits: RawDevicePtr<f32>,
    batch_size: usize,
    next_n: usize,
    num_heads: usize,
    head_dim: usize,
    num_kv_blocks: usize,
    block_kv: usize,
    max_context_len: usize,
    logits_stride: usize,
    block_table_stride: usize,
    kv_cache_stride_bytes: usize,
    num_sms: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_deepgemm_fp8_paged_mqa_logits_cuda(
            q.as_ptr(),
            kv_cache.as_ptr(),
            kv_cache_scales.as_ptr(),
            weights.as_ptr(),
            context_lens.as_ptr(),
            block_table.as_ptr(),
            schedule_meta.as_ptr(),
            logits.as_mut_ptr(),
            i32::try_from(batch_size)?,
            i32::try_from(next_n)?,
            i32::try_from(num_heads)?,
            i32::try_from(head_dim)?,
            i32::try_from(num_kv_blocks)?,
            i32::try_from(block_kv)?,
            i32::try_from(max_context_len)?,
            i32::try_from(logits_stride)?,
            i32::try_from(block_table_stride)?,
            i32::try_from(kv_cache_stride_bytes)?,
            i32::try_from(num_sms)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Official DeepGEMM FP8 paged-MQA logits over SGLang's fused DSA cache layout:
/// `[page][64][128 FP8 values | 64 FP32 scales]` as one byte buffer.
///
/// # Safety
/// All pointers must be valid on `stream`. `kv_cache_with_scale` must point at
/// a contiguous page buffer with `kv_cache_stride_bytes = 64 * (head_dim + 4)`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_deepgemm_fp8_paged_mqa_logits_fused_cache(
    q: RawDevicePtr<u8>,
    kv_cache_with_scale: RawDevicePtr<u8>,
    weights: RawDevicePtr<f32>,
    context_lens: RawDevicePtr<i32>,
    block_table: RawDevicePtr<i32>,
    schedule_meta: RawDevicePtr<i32>,
    logits: RawDevicePtr<f32>,
    batch_size: usize,
    next_n: usize,
    num_heads: usize,
    head_dim: usize,
    num_kv_blocks: usize,
    block_kv: usize,
    max_context_len: usize,
    logits_stride: usize,
    block_table_stride: usize,
    kv_cache_stride_bytes: usize,
    num_sms: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_deepgemm_fp8_paged_mqa_logits_fused_cache_cuda(
            q.as_ptr(),
            kv_cache_with_scale.as_ptr(),
            weights.as_ptr(),
            context_lens.as_ptr(),
            block_table.as_ptr(),
            schedule_meta.as_ptr(),
            logits.as_mut_ptr(),
            i32::try_from(batch_size)?,
            i32::try_from(next_n)?,
            i32::try_from(num_heads)?,
            i32::try_from(head_dim)?,
            i32::try_from(num_kv_blocks)?,
            i32::try_from(block_kv)?,
            i32::try_from(max_context_len)?,
            i32::try_from(logits_stride)?,
            i32::try_from(block_table_stride)?,
            i32::try_from(kv_cache_stride_bytes)?,
            i32::try_from(num_sms)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Fused clamped-SwiGLU over the `[gate | up]` w13 GEMM output + per-128-block
/// requantize to the FP8 activation the w2 GEMM reads. Wraps
/// [`ffi::dsv4_deepgemm_swiglu_quantize_w13_cuda`]; `limit` is the SwiGLU clamp.
///
/// # Safety
/// All pointers must be valid on `stream` for the given shape; `w13` is the
/// padded BF16 w13 output, `act`/`scales` the FP8 + scale scratch.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_deepgemm_swiglu_quantize_w13(
    w13: RawDevicePtr<bf16>,
    act: RawDevicePtr<u8>,
    scales: RawDevicePtr<f32>,
    active_experts: RawDevicePtr<i32>,
    active_counts: RawDevicePtr<i32>,
    active_count: usize,
    max_m: usize,
    intermediate_dim: usize,
    scale_stride_m: usize,
    limit: f32,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_deepgemm_swiglu_quantize_w13_cuda(
            w13.as_ptr() as *const Half,
            act.as_mut_ptr(),
            scales.as_mut_ptr(),
            active_experts.as_ptr(),
            active_counts.as_ptr(),
            i32::try_from(active_count)?,
            i32::try_from(max_m)?,
            i32::try_from(intermediate_dim)?,
            i32::try_from(scale_stride_m)?,
            limit,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Masked 3-D variant of [`dsv4_deepgemm_swiglu_quantize_w13`] for the
/// `deepep_ll` MoE path: fused clamped-SwiGLU over the `[E, tok_padded,
/// 2*intermediate]` BF16 GEMM1 output + per-128-block FP8 requantize into the
/// `[E, tok_padded, intermediate]` activation the masked w2 GEMM reads. Skips
/// invalid tokens per expert via `masked_m[expert]` (no `active_*` compaction)
/// and emits the column-major SFA scale layout the masked grouped DeepGEMM
/// consumes. Wraps [`ffi::dsv4_deepgemm_silu_mul_masked_quant_cuda`].
///
/// `hidden_dim` is the gate||up width (= `2 * intermediate`); `out_fp8` /
/// `out_scale` are sized `[E, tok_padded, intermediate]` and
/// `[E, scale_stride_m, intermediate/128]` respectively (`scale_stride_m` =
/// `tok_padded` rounded up to a multiple of 4).
///
/// # Safety
/// All pointers must be valid on `stream` for the given shape; `input` is the
/// padded BF16 w13 output, `out_fp8` / `out_scale` the FP8 + scale scratch.
/// `expected_m` is the host-known upper bound on any expert's valid rows
/// (per-expert recv count ≤ the step's global token count); the launch grid
/// covers only `min(expected_m, token_num_padded)` rows per expert. The padded
/// band still backs all memory strides, and the kernel traps loudly if a
/// `masked_m` count ever exceeds the bound.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_deepgemm_silu_mul_masked_quant(
    input: RawDevicePtr<bf16>,
    out_fp8: RawDevicePtr<u8>,
    out_scale: RawDevicePtr<f32>,
    masked_m: RawDevicePtr<i32>,
    expert_num: usize,
    token_num_padded: usize,
    expected_m: usize,
    hidden_dim: usize,
    swiglu_limit: f32,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_deepgemm_silu_mul_masked_quant_cuda(
            input.as_ptr() as *const Half,
            out_fp8.as_mut_ptr(),
            out_scale.as_mut_ptr(),
            masked_m.as_ptr(),
            i32::try_from(expert_num)?,
            i32::try_from(token_num_padded)?,
            i32::try_from(expected_m)?,
            i32::try_from(hidden_dim)?,
            swiglu_limit,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Unpad the padded `[num_groups * max_m, hidden]` grouped GEMM output back to
/// the compact `[total_routes, hidden]` row layout (per-group `active_offsets`).
/// Wraps [`ffi::dsv4_deepgemm_unpad_grouped_bf16_cuda`].
///
/// # Safety
/// All pointers must be valid on `stream` for the given shape.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_deepgemm_unpad_grouped_bf16(
    grouped: RawDevicePtr<bf16>,
    compact: RawDevicePtr<bf16>,
    active_experts: RawDevicePtr<i32>,
    active_offsets: RawDevicePtr<i32>,
    active_counts: RawDevicePtr<i32>,
    active_count: usize,
    max_m: usize,
    hidden_dim: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_deepgemm_unpad_grouped_bf16_cuda(
            grouped.as_ptr() as *const Half,
            compact.as_mut_ptr() as *mut Half,
            active_experts.as_ptr(),
            active_offsets.as_ptr(),
            active_counts.as_ptr(),
            i32::try_from(active_count)?,
            i32::try_from(max_m)?,
            i32::try_from(hidden_dim)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Fail-loud preflight: confirm the native DeepGEMM bridge is compiled in (a
/// stub binary returns `CUDA_ERROR_NOT_SUPPORTED`). Returns the device report
/// string on success. Wraps [`ffi::dsv4_deepgemm_native_preflight_cuda`].
pub fn dsv4_deepgemm_native_preflight() -> Result<String> {
    let mut report = vec![0 as std::ffi::c_char; 4096];
    // SAFETY: `report` is a live 4096-byte zeroed buffer; the FFI writes a
    // NUL-terminated report within `report.len()` bytes.
    let result =
        unsafe { ffi::dsv4_deepgemm_native_preflight_cuda(report.as_mut_ptr(), report.len()) };
    // SAFETY: the zero-initialized buffer guarantees NUL termination within its
    // 4096 bytes even if the FFI wrote nothing, so CStr::from_ptr stays in
    // bounds; `report` outlives the borrow.
    let report = unsafe { std::ffi::CStr::from_ptr(report.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    result
        .result()
        .map_err(|err| anyhow::anyhow!("DSv4 DeepGEMM native preflight failed: {err}; {report}"))?;
    Ok(report)
}

/// Dense clamped SwiGLU over a `[seq_len, intermediate]` batch:
/// `out = silu(min(gate, limit)) * clamp(up, -limit, limit)`. Used by the DSv4
/// shared expert (not the routed grouped path). Wraps
/// [`ffi::dsv4_swiglu_clamped_cuda`].
///
/// # Safety
/// `gate` / `up` / `out` must be valid on `stream` for `n` elements.
pub unsafe fn dsv4_swiglu_clamped_batch(
    gate: RawDevicePtr<bf16>,
    up: RawDevicePtr<bf16>,
    out: RawDevicePtr<bf16>,
    n: usize,
    limit: f32,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_swiglu_clamped_cuda(
            gate.as_ptr() as *const Half,
            up.as_ptr() as *const Half,
            out.as_mut_ptr() as *mut Half,
            i32::try_from(n)?,
            limit,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// DSv4 FP8 decode-band fused gate+up+SwiGLU grouped GEMM (w8a16, f32 block
/// scales). Compact (work scales with real routed rows), 16-byte vectorized
/// FP8 weight loads, per-route correct (one warp per output row, no tile
/// contract). `N` = moe intermediate, `K` = hidden; `scale_cols` = K/128.
/// Writes `act[route, :] = silu(gate·x) * (up·x)`.
///
/// # Safety
/// All pointers/tables valid on `stream`; tables hold `num_experts` entries;
/// `K % 16 == 0`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_fp8_grouped_swiglu_decode(
    weight_gate_ptrs: RawDevicePtr<u64>,
    scale_gate_ptrs: RawDevicePtr<u64>,
    weight_up_ptrs: RawDevicePtr<u64>,
    scale_up_ptrs: RawDevicePtr<u64>,
    input: RawDevicePtr<bf16>,
    act: RawDevicePtr<bf16>,
    offsets: RawDevicePtr<i32>,
    counts: RawDevicePtr<i32>,
    num_experts: usize,
    max_count: usize,
    n: usize,
    k: usize,
    scale_cols: usize,
    limit: f32,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_fp8_grouped_swiglu_decode_cuda(
            weight_gate_ptrs.as_ptr(),
            scale_gate_ptrs.as_ptr(),
            weight_up_ptrs.as_ptr(),
            scale_up_ptrs.as_ptr(),
            input.as_ptr() as *const Half,
            act.as_mut_ptr() as *mut Half,
            offsets.as_ptr(),
            counts.as_ptr(),
            std::ptr::null(),
            i32::try_from(num_experts)?,
            i32::try_from(max_count)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(scale_cols)?,
            limit,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// DSv4 FP8 decode-band down (w2) grouped GEMM (w8a16, f32 block scales).
/// `N` = hidden, `K` = moe intermediate; `scale_cols` = K/128.
///
/// # Safety
/// See [`dsv4_fp8_grouped_swiglu_decode`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_fp8_grouped_down_decode(
    weight_ptrs: RawDevicePtr<u64>,
    scale_ptrs: RawDevicePtr<u64>,
    input: RawDevicePtr<bf16>,
    output: RawDevicePtr<bf16>,
    offsets: RawDevicePtr<i32>,
    counts: RawDevicePtr<i32>,
    num_experts: usize,
    max_count: usize,
    n: usize,
    k: usize,
    scale_cols: usize,
    stream: CUstream,
) -> Result<()> {
    // SAFETY: forwarded — the caller upholds this fn's `# Safety` contract
    // (all raw pointers valid on `stream` for the shape); i32 casts are checked.
    unsafe {
        ffi::dsv4_fp8_grouped_down_decode_cuda(
            weight_ptrs.as_ptr(),
            scale_ptrs.as_ptr(),
            input.as_ptr() as *const Half,
            output.as_mut_ptr() as *mut Half,
            offsets.as_ptr(),
            counts.as_ptr(),
            std::ptr::null(),
            i32::try_from(num_experts)?,
            i32::try_from(max_count)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(scale_cols)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Convenience: build a [`RawDevicePtr`] over a [`DeviceVec`]'s dense data.
#[must_use]
pub fn device_vec_ptr(vec: &DeviceVec, ctx: &DeviceContext) -> RawDevicePtr<bf16> {
    crate::tensor::cache_ptr(&vec.data, ctx)
}

// W4A16 (INT4) MoE grouped GEMV — numerical correctness vs a dequantized
// BF16 reference. GPU-gated: builds the pointer tables, runs
// `moe_w4a16_grouped_gemv_batch`, dequantizes the same weights to BF16 and
// runs `moe_bf16_grouped_gemm_batch` as the reference, then compares the
// outputs. No HF dependency — synthetic weights only.
#[cfg(all(test, feature = "cuda"))]
mod w4a16_tests {
    use super::*;
    use crate::tensor::{cache_ptr, null_raw_ptr};
    use half::bf16;

    // Tiny but shape-valid: 2 experts, N=64 output rows, K=256 input cols,
    // group_size=128 (2 groups per row). K must be even and K % group_size == 0.
    const N: usize = 64;
    const K: usize = 256;
    const GROUP_SIZE: usize = 128;
    const NUM_EXPERTS: usize = 2;
    const NUM_TOKENS: usize = 4; // 2 routed to each expert.

    // Deterministic INT4 nibble pattern (values 0..=15) covering the range.
    const NIBBLES: [u8; 8] = [0, 8, 15, 13, 4, 2, 7, 11];

    fn packed_int4_and_scales() -> (Vec<u8>, Vec<bf16>) {
        let num_groups = K / GROUP_SIZE;
        let mut packed = vec![0u8; N * K / 2];
        let mut scales = vec![bf16::from_f32(0.0); N * num_groups];
        for row in 0..N {
            for k in 0..K {
                let nibble = NIBBLES[(row + k) % NIBBLES.len()];
                let byte = &mut packed[row * (K / 2) + k / 2];
                if k % 2 == 0 {
                    *byte = nibble;
                } else {
                    *byte |= nibble << 4;
                }
            }
            for g in 0..num_groups {
                // Small, non-trivial per-group scale (different per row/group).
                let v = 0.01 + (row as f32) * 0.001 + (g as f32) * 0.005;
                scales[row * num_groups + g] = bf16::from_f32(v);
            }
        }
        (packed, scales)
    }

    // Dequantize W4A16 -> BF16 on host: (nibble - zero_point 8) * per-group scale.
    fn dequantize(packed: &[u8], scales: &[bf16]) -> Vec<bf16> {
        let num_groups = K / GROUP_SIZE;
        let mut out = vec![bf16::from_f32(0.0); N * K];
        for row in 0..N {
            for k in 0..K {
                let byte = packed[row * (K / 2) + k / 2];
                let nibble = if k % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                let scale = scales[row * num_groups + k / GROUP_SIZE].to_f32();
                let v = (nibble as f32 - 8.0) * scale;
                out[row * K + k] = bf16::from_f32(v);
            }
        }
        out
    }

    #[test]
    fn w4a16_grouped_gemv_matches_dequantized_bf16() {
        let ctx = DeviceContext::new().expect("cuda context");
        let (packed, scales) = packed_int4_and_scales();

        // Build the per-expert W4A16 matrices (same weights both experts).
        let w4a16 = (0..NUM_EXPERTS)
            .map(|_| DeviceMatrix::from_quantized_int4(&ctx, &packed, &scales, N, K, GROUP_SIZE))
            .collect::<Result<Vec<_>>>()
            .expect("build w4a16 matrices");
        let w4a16_refs: Vec<&DeviceMatrix> = w4a16.iter().collect();
        let weight_ptrs =
            build_expert_qweight_i8_ptr_table(&ctx, &w4a16_refs).expect("qweight ptrs");
        let scale_ptrs =
            build_expert_qscale_bf16_ptr_table(&ctx, &w4a16_refs).expect("qscale ptrs");

        // Reference: dequantize -> BF16 matrices -> dense grouped GEMM.
        let dequant = dequantize(&packed, &scales);
        let bf16_mats = (0..NUM_EXPERTS)
            .map(|_| DeviceMatrix::from_host(&ctx, &dequant, N, K))
            .collect::<Result<Vec<_>>>()
            .expect("build bf16 matrices");
        let bf16_refs: Vec<&DeviceMatrix> = bf16_mats.iter().collect();
        let bf16_weight_ptrs =
            build_expert_weight_ptr_table(&ctx, &bf16_refs).expect("bf16 weight ptrs");

        // Input activations (deterministic, signed).
        let input_host: Vec<bf16> = (0..NUM_TOKENS * K)
            .map(|i| bf16::from_f32(((i as f32 % 13.0) - 6.0) * 0.01))
            .collect();
        let input_dev = ctx.stream.clone_htod(&input_host).expect("input h2d");

        // Route tables: tokens [0,1] -> expert 0, tokens [2,3] -> expert 1.
        // compact index == expert index (expert_indices = null below).
        let offsets = ctx.stream.clone_htod(&[0i32, 2i32]).expect("offsets h2d");
        let counts = ctx.stream.clone_htod(&[2i32, 2i32]).expect("counts h2d");
        let max_count = 2;

        let w4a16_out = ctx
            .stream
            .alloc_zeros::<bf16>(NUM_TOKENS * N)
            .expect("w4a16 out");
        let bf16_out = ctx
            .stream
            .alloc_zeros::<bf16>(NUM_TOKENS * N)
            .expect("bf16 out");

        let input_p = cache_ptr(&input_dev, &ctx);
        let offsets_p = cache_ptr(&offsets, &ctx).cast::<i32>();
        let counts_p = cache_ptr(&counts, &ctx).cast::<i32>();
        let w4a16_out_p = cache_ptr(&w4a16_out, &ctx);
        let bf16_out_p = cache_ptr(&bf16_out, &ctx);
        let stream = ctx.stream.cu_stream();

        // SAFETY: all buffers live for the duration, shapes match, route tables
        // address only valid tokens/experts (test-only, single stream).
        unsafe {
            moe_w4a16_grouped_gemv_batch(
                &weight_ptrs,
                &scale_ptrs,
                input_p,
                w4a16_out_p,
                offsets_p,
                counts_p,
                null_raw_ptr(),
                NUM_EXPERTS,
                max_count,
                N,
                K,
                GROUP_SIZE,
                &ctx,
                stream,
            )
            .expect("w4a16 grouped gemv");

            moe_bf16_grouped_gemm_batch(
                &bf16_weight_ptrs,
                input_p,
                bf16_out_p,
                offsets_p,
                counts_p,
                null_raw_ptr(),
                NUM_EXPERTS,
                max_count,
                N,
                K,
                &ctx,
                stream,
            )
            .expect("bf16 grouped gemm");
        }
        ctx.sync().expect("sync");

        let w4a16_host = ctx.stream.clone_dtoh(&w4a16_out).expect("w4a16 d2h");
        let bf16_host = ctx.stream.clone_dtoh(&bf16_out).expect("bf16 d2h");

        let mut max_err = 0.0f32;
        let mut sum_err = 0.0f32;
        for (a, b) in w4a16_host.iter().zip(bf16_host.iter()) {
            let e = (a.to_f32() - b.to_f32()).abs();
            max_err = max_err.max(e);
            sum_err += e;
        }
        let mean_err = sum_err / (w4a16_host.len() as f32);
        assert!(
            max_err < 0.05 && mean_err < 0.01,
            "W4A16 vs dequantized-BF16 mismatch: max_err={max_err} mean_err={mean_err}"
        );
    }
}
