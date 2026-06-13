//! DeepSeek MoE routing helper kernels.

use anyhow::{Result, ensure};
use cudarc::driver::sys::CUstream;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use crate::ffi::{self, Half};
use crate::tensor::{DeviceContext, DeviceMatrix, DeviceVec, RawDevicePtr};

#[allow(clippy::too_many_arguments)]
pub fn dsv4_mask_indices_by_ep_i64(
    ctx: &DeviceContext,
    indices: &CudaSlice<i64>,
    masked_indices: &mut CudaSlice<i64>,
    num_tokens: usize,
    num_topk: usize,
    experts_per_ep_rank: usize,
    experts_per_moe_dp_group: usize,
    num_tp_ranks: usize,
    tp_rank: usize,
) -> Result<()> {
    ensure_mask_args(
        indices.len(),
        masked_indices.len(),
        num_tokens,
        num_topk,
        experts_per_ep_rank,
        experts_per_moe_dp_group,
        num_tp_ranks,
        tp_rank,
    )?;

    let (indices_ptr, _g0) = indices.device_ptr(&ctx.stream);
    let (masked_ptr, _g1) = masked_indices.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::dsv4_mask_indices_by_ep_i64_cuda(
            indices_ptr as *const i64,
            masked_ptr as *mut i64,
            num_tokens as i32,
            num_topk as i32,
            experts_per_ep_rank as i32,
            experts_per_moe_dp_group as i32,
            num_tp_ranks as i32,
            tp_rank as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn dsv4_mask_indices_by_ep_i32(
    ctx: &DeviceContext,
    indices: &CudaSlice<i32>,
    masked_indices: &mut CudaSlice<i32>,
    num_tokens: usize,
    num_topk: usize,
    experts_per_ep_rank: usize,
    experts_per_moe_dp_group: usize,
    num_tp_ranks: usize,
    tp_rank: usize,
) -> Result<()> {
    ensure_mask_args(
        indices.len(),
        masked_indices.len(),
        num_tokens,
        num_topk,
        experts_per_ep_rank,
        experts_per_moe_dp_group,
        num_tp_ranks,
        tp_rank,
    )?;

    let (indices_ptr, _g0) = indices.device_ptr(&ctx.stream);
    let (masked_ptr, _g1) = masked_indices.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::dsv4_mask_indices_by_ep_i32_cuda(
            indices_ptr as *const i32,
            masked_ptr as *mut i32,
            num_tokens as i32,
            num_topk as i32,
            experts_per_ep_rank as i32,
            experts_per_moe_dp_group as i32,
            num_tp_ranks as i32,
            tp_rank as i32,
            ctx.stream.cu_stream(),
        )
        .result()?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn ensure_mask_args(
    input_len: usize,
    output_len: usize,
    num_tokens: usize,
    num_topk: usize,
    experts_per_ep_rank: usize,
    experts_per_moe_dp_group: usize,
    num_tp_ranks: usize,
    tp_rank: usize,
) -> Result<()> {
    let expected = num_tokens
        .checked_mul(num_topk)
        .ok_or_else(|| anyhow::anyhow!("num_tokens * num_topk overflows"))?;
    ensure!(
        input_len >= expected && output_len >= expected,
        "mask_indices_by_ep buffers too small: input={} output={} expected={}",
        input_len,
        output_len,
        expected
    );
    ensure!(experts_per_ep_rank > 0, "experts_per_ep_rank must be > 0");
    ensure!(
        experts_per_moe_dp_group >= experts_per_ep_rank,
        "experts_per_moe_dp_group ({experts_per_moe_dp_group}) must be >= experts_per_ep_rank ({experts_per_ep_rank})"
    );
    ensure!(num_tp_ranks > 0, "num_tp_ranks must be > 0");
    ensure!(
        tp_rank < num_tp_ranks,
        "tp_rank {tp_rank} must be < num_tp_ranks {num_tp_ranks}"
    );
    ensure!(expected <= i32::MAX as usize, "mask input too large");
    ensure!(
        experts_per_ep_rank <= i32::MAX as usize
            && experts_per_moe_dp_group <= i32::MAX as usize
            && num_tp_ranks <= i32::MAX as usize
            && tp_rank <= i32::MAX as usize,
        "mask parameter exceeds i32 kernel ABI"
    );
    Ok(())
}

// Safe wrappers over the grouped-GEMM + DSv4/Qwen3.6 expert-dispatch FFI: they
// centralize device-pointer extraction + the i32-ABI casts. `RawDevicePtr<T>`
// (from `cache_ptr`) carries buffers; bf16 (Rust) and Half (u16, kernel ABI)
// share a 16-bit layout, so pointers cast directly.

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
    unsafe {
        ffi::dsv4_cast_i32_to_i64_cuda(src.as_ptr(), dst.as_mut_ptr(), i32::try_from(n)?, stream)
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

// ── DSv4 FP8 DeepGEMM MoE pipeline (f8f8bf16, 128-block scale) ─────────────
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
    let result =
        unsafe { ffi::dsv4_deepgemm_native_preflight_cuda(report.as_mut_ptr(), report.len()) };
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

/// Grouped FP8 w8a16 GEMV over per-expert weight/scale pointer tables:
/// for packed row `offsets[e] + i` (i < counts[e], device-resident), computes
/// `out[row] = dequant(weights[e]) @ input[row]`. Scales are f32 over
/// `[scale_rows, scale_cols]` blocks of the `[n, k]` weight. `expert_indices`
/// NULL ⇒ identity (grid z spans all `num_experts`; inactive groups exit on
/// `counts`). Wraps [`ffi::dsv4_fp8_grouped_gemv_batch_f32s_cuda`].
///
/// # Safety
/// All pointers/tables valid on `stream`; tables hold `num_experts` entries.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_fp8_grouped_gemv_batch(
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
    scale_rows: usize,
    scale_cols: usize,
    stream: CUstream,
) -> Result<()> {
    unsafe {
        ffi::dsv4_fp8_grouped_gemv_batch_f32s_cuda(
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
            i32::try_from(scale_rows)?,
            i32::try_from(scale_cols)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Pair variant of [`dsv4_fp8_grouped_gemv_batch`]: two weight/scale tables
/// (gate + up halves of the fused w13) computed from ONE pass over the input
/// row — halves the activation reads for the SwiGLU pair.
///
/// # Safety
/// See [`dsv4_fp8_grouped_gemv_batch`]; both tables share `[n, k]` geometry.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_fp8_grouped_gemv_pair_batch(
    weight_a_ptrs: RawDevicePtr<u64>,
    scale_a_ptrs: RawDevicePtr<u64>,
    weight_b_ptrs: RawDevicePtr<u64>,
    scale_b_ptrs: RawDevicePtr<u64>,
    input: RawDevicePtr<bf16>,
    output_a: RawDevicePtr<bf16>,
    output_b: RawDevicePtr<bf16>,
    offsets: RawDevicePtr<i32>,
    counts: RawDevicePtr<i32>,
    num_experts: usize,
    max_count: usize,
    n: usize,
    k: usize,
    scale_rows: usize,
    scale_cols: usize,
    stream: CUstream,
) -> Result<()> {
    unsafe {
        ffi::dsv4_fp8_grouped_gemv_pair_batch_f32s_cuda(
            weight_a_ptrs.as_ptr(),
            scale_a_ptrs.as_ptr(),
            weight_b_ptrs.as_ptr(),
            scale_b_ptrs.as_ptr(),
            input.as_ptr() as *const Half,
            output_a.as_mut_ptr() as *mut Half,
            output_b.as_mut_ptr() as *mut Half,
            offsets.as_ptr(),
            counts.as_ptr(),
            std::ptr::null(),
            i32::try_from(num_experts)?,
            i32::try_from(max_count)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(scale_rows)?,
            i32::try_from(scale_cols)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

// ============================================================================
// SGLang `fused_moe` decode lane (U3, opt-in `ARLE_QWEN35_MOE_FUSED_SGLANG`).
// Native `moe_align_block_size` + the four triton AOT cubins (GEMM1 / act /
// GEMM2 / sum). All take raw device pointers (the `moe_forward_into` call
// convention) and `.result()?` the CUresult. The triton wrappers are
// Hopper-only AOT: a non-sm90 / no-triton build returns
// `CUDA_ERROR_NOT_SUPPORTED`, which the Rust caller maps to the loud opt-in
// failure. The block size MUST equal the triton GEMM `BLOCK_SIZE_M` (64) the
// cubins were compiled with, so `sorted_token_ids` / `expert_ids` group
// boundaries line up with the GEMM tile rows.
// ============================================================================

/// SGLang `fused_moe` align block: `block_size = BLOCK_SIZE_M = 64`.
pub const FUSED_MOE_BLOCK_SIZE_M: usize = 64;
/// Triton `fused_moe_kernel` GEMM tile N (cubins built with `BLOCK_SIZE_N=64`).
pub const FUSED_MOE_BLOCK_SIZE_N: usize = 64;
/// `act_and_mul_kernel` block (cubin built with `BLOCK_SIZE=512`).
pub const FUSED_MOE_ACT_BLOCK_SIZE: usize = 512;
/// `_moe_sum_reduce_kernel` hidden tile (cubin built with `BLOCK_DIM=2048`).
pub const FUSED_MOE_SUM_BLOCK_DIM: usize = 2048;

#[inline]
fn cdiv(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// `max_num_tokens_padded` for [`moe_align_block_size`]: every expert group can
/// pad up to `block_size - 1` extra rows, so the worst case is
/// `numel + num_experts * (block_size - 1)`. SGLang rounds this up to a
/// `block_size` multiple for the vectorized pad fill (`int4` stores).
#[must_use]
pub fn fused_moe_max_num_tokens_padded(
    numel: usize,
    num_experts: usize,
    block_size: usize,
) -> usize {
    let raw = numel + num_experts * (block_size.saturating_sub(1));
    cdiv(raw, block_size) * block_size
}

/// SGLang `moe_align_block_size` (FULL path). `topk_ids` is token-major
/// `[numel]` (i32 expert ids in `[0, num_experts)`); fills `sorted_token_ids`
/// `[max_num_tokens_padded]`, `expert_ids` `[num_blocks]`,
/// `total_tokens_post_pad` `[1]`. `cumsum_buffer` `[num_experts + 1]` is
/// caller scratch. `cumsum_buffer` AND `total_tokens_post_pad` must be
/// device-zeroed before the call (atomicAdd accumulators); the caller's
/// `get_zeroed` slots cover this.
///
/// # Safety
/// All pointers must be valid on `stream` for the documented shapes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_align_block_size(
    topk_ids: RawDevicePtr<i32>,
    sorted_token_ids: RawDevicePtr<i32>,
    expert_ids: RawDevicePtr<i32>,
    total_tokens_post_pad: RawDevicePtr<i32>,
    cumsum_buffer: RawDevicePtr<i32>,
    num_experts: usize,
    block_size: usize,
    numel: usize,
    max_num_tokens_padded: usize,
    stream: CUstream,
) -> Result<()> {
    ensure!(num_experts > 0, "moe_align num_experts must be > 0");
    ensure!(block_size > 0, "moe_align block_size must be > 0");
    unsafe {
        ffi::arle_moe_align_block_size_cuda(
            topk_ids.as_ptr(),
            sorted_token_ids.as_mut_ptr(),
            expert_ids.as_mut_ptr(),
            total_tokens_post_pad.as_mut_ptr(),
            cumsum_buffer.as_mut_ptr(),
            i32::try_from(num_experts)?,
            i32::try_from(block_size)?,
            i32::try_from(numel)?,
            i32::try_from(max_num_tokens_padded)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Triton `fused_moe_kernel` GEMM1 (gate||up): `a [M, K]` × `w1 [E, 2N, K]` →
/// `c [M*top_k, 2N]`. `em` is the padded sorted-id length (drives grid X with
/// `n` = 2N). Strides are in ELEMENTS. `top_k`/`MUL_ROUTED_WEIGHT` are baked in
/// the cubin (8 / 0).
///
/// # Safety
/// All pointers valid on `stream`; shapes match the align outputs.
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_moe_gemm1(
    a: RawDevicePtr<bf16>,
    w1: RawDevicePtr<bf16>,
    c: RawDevicePtr<bf16>,
    topk_weights: RawDevicePtr<f32>,
    sorted_token_ids: RawDevicePtr<i32>,
    expert_ids: RawDevicePtr<i32>,
    num_tokens_post_padded: RawDevicePtr<i32>,
    n: usize,
    k: usize,
    em: usize,
    num_valid_tokens: usize,
    stride_am: usize,
    stride_ak: usize,
    stride_be: usize,
    stride_bk: usize,
    stride_bn: usize,
    stride_cm: usize,
    stride_cn: usize,
    stream: CUstream,
) -> Result<()> {
    let grid = cdiv(em, FUSED_MOE_BLOCK_SIZE_M) * cdiv(n, FUSED_MOE_BLOCK_SIZE_N);
    unsafe {
        ffi::arle_fused_moe_gemm1_cuda(
            u32::try_from(grid)?,
            1,
            1,
            a.as_ptr() as *const Half,
            w1.as_ptr() as *const Half,
            c.as_mut_ptr() as *mut Half,
            topk_weights.as_ptr(),
            sorted_token_ids.as_ptr(),
            expert_ids.as_ptr(),
            num_tokens_post_padded.as_ptr(),
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(em)?,
            i32::try_from(num_valid_tokens)?,
            i32::try_from(stride_am)?,
            i32::try_from(stride_ak)?,
            i32::try_from(stride_be)?,
            i32::try_from(stride_bk)?,
            i32::try_from(stride_bn)?,
            i32::try_from(stride_cm)?,
            i32::try_from(stride_cn)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Triton `fused_moe_kernel` GEMM2 (down): `a [M*top_k, N]` × `w2 [E, K, N]` →
/// `c [M*top_k, K]`. Same grid formula as GEMM1 with `n` = K (hidden).
/// `top_k`/`MUL_ROUTED_WEIGHT` baked (1 / 1 — applies topk_weights here).
///
/// # Safety
/// All pointers valid on `stream`; shapes match the align outputs.
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_moe_gemm2(
    a: RawDevicePtr<bf16>,
    w2: RawDevicePtr<bf16>,
    c: RawDevicePtr<bf16>,
    topk_weights: RawDevicePtr<f32>,
    sorted_token_ids: RawDevicePtr<i32>,
    expert_ids: RawDevicePtr<i32>,
    num_tokens_post_padded: RawDevicePtr<i32>,
    n: usize,
    k: usize,
    em: usize,
    num_valid_tokens: usize,
    stride_am: usize,
    stride_ak: usize,
    stride_be: usize,
    stride_bk: usize,
    stride_bn: usize,
    stride_cm: usize,
    stride_cn: usize,
    stream: CUstream,
) -> Result<()> {
    let grid = cdiv(em, FUSED_MOE_BLOCK_SIZE_M) * cdiv(n, FUSED_MOE_BLOCK_SIZE_N);
    unsafe {
        ffi::arle_fused_moe_gemm2_cuda(
            u32::try_from(grid)?,
            1,
            1,
            a.as_ptr() as *const Half,
            w2.as_ptr() as *const Half,
            c.as_mut_ptr() as *mut Half,
            topk_weights.as_ptr(),
            sorted_token_ids.as_ptr(),
            expert_ids.as_ptr(),
            num_tokens_post_padded.as_ptr(),
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::try_from(em)?,
            i32::try_from(num_valid_tokens)?,
            i32::try_from(stride_am)?,
            i32::try_from(stride_ak)?,
            i32::try_from(stride_be)?,
            i32::try_from(stride_bk)?,
            i32::try_from(stride_bn)?,
            i32::try_from(stride_cm)?,
            i32::try_from(stride_cn)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Triton `act_and_mul_kernel` (silu): `gateup [num_rows, 2N]` → `down_input
/// [num_rows, N]`, `down_input = silu(gate) * up`. `hidden_size` = 2N (the
/// kernel halves it). `expert_ids` is the token-major topk_ids row buffer
/// (pad-row skip; never fires on the dense decode path). Grid = (num_rows,).
///
/// # Safety
/// `gateup` / `down_input` / `expert_ids` valid on `stream` for `num_rows`.
pub unsafe fn fused_moe_act_and_mul(
    gateup: RawDevicePtr<bf16>,
    down_input: RawDevicePtr<bf16>,
    expert_ids: RawDevicePtr<i32>,
    num_rows: usize,
    gate_up_hidden: usize,
    stream: CUstream,
) -> Result<()> {
    unsafe {
        ffi::arle_fused_moe_act_cuda(
            u32::try_from(num_rows)?,
            1,
            1,
            gateup.as_ptr() as *const Half,
            down_input.as_mut_ptr() as *mut Half,
            i32::try_from(gate_up_hidden)?,
            expert_ids.as_ptr(),
            stream,
        )
        .result()?;
    }
    Ok(())
}

/// Triton `_moe_sum_reduce_kernel`: sum `input [M, top_k, K]` over `top_k` and
/// scale by the baked `routed_scaling_factor` into `output [M, K]`. Strides in
/// ELEMENTS. Grid = (M, cdiv(K, 2048)). BLOCK_M=1, so grid X = M.
///
/// # Safety
/// `input` / `output` valid on `stream` for the documented shapes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_moe_sum_reduce(
    input: RawDevicePtr<bf16>,
    input_stride_0: usize,
    input_stride_1: usize,
    input_stride_2: usize,
    output: RawDevicePtr<bf16>,
    output_stride_0: usize,
    output_stride_1: usize,
    token_num: usize,
    topk_num: usize,
    hidden_dim: usize,
    stream: CUstream,
) -> Result<()> {
    let grid_x = token_num; // BLOCK_M = 1
    let grid_y = cdiv(hidden_dim, FUSED_MOE_SUM_BLOCK_DIM);
    unsafe {
        ffi::arle_fused_moe_sum_cuda(
            u32::try_from(grid_x)?,
            u32::try_from(grid_y)?,
            1,
            input.as_ptr() as *const Half,
            i32::try_from(input_stride_0)?,
            i32::try_from(input_stride_1)?,
            i32::try_from(input_stride_2)?,
            output.as_mut_ptr() as *mut Half,
            i32::try_from(output_stride_0)?,
            i32::try_from(output_stride_1)?,
            i32::try_from(token_num)?,
            i32::try_from(topk_num)?,
            i32::try_from(hidden_dim)?,
            stream,
        )
        .result()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(
        indices: &[i64],
        experts_per_ep_rank: i64,
        experts_per_moe_dp_group: i64,
        num_tp_ranks: i64,
        tp_rank: i64,
    ) -> Vec<i64> {
        indices
            .iter()
            .map(|&raw| {
                if raw < 0 || ((raw / experts_per_ep_rank) % num_tp_ranks) != tp_rank {
                    return -1;
                }
                let mut value = raw - tp_rank * experts_per_ep_rank;
                let dp_rank = value / experts_per_moe_dp_group;
                value -= dp_rank * (experts_per_moe_dp_group - experts_per_ep_rank);
                if value < 0 { -1 } else { value }
            })
            .collect()
    }

    #[test]
    fn dsv4_mask_indices_by_ep_i64_matches_tilekernels_formula() {
        let ctx = DeviceContext::new().expect("CUDA context");
        let host = vec![-1, 0, 1, 3, 4, 7, 8, 12, 15, 16, 19, 23, 24, 31];
        let input = ctx.stream.clone_htod(&host).expect("H2D input");
        let mut output = ctx
            .stream
            .alloc_zeros::<i64>(host.len())
            .expect("alloc output");
        dsv4_mask_indices_by_ep_i64(&ctx, &input, &mut output, 2, 7, 4, 8, 2, 1)
            .expect("mask_indices_by_ep");
        ctx.sync().expect("sync");
        let got = ctx.stream.clone_dtoh(&output).expect("D2H output");
        let expected = reference(&host, 4, 8, 2, 1);
        assert_eq!(got, expected);
    }
}
