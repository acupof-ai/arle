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

/// Convenience: build a [`RawDevicePtr`] over a [`DeviceVec`]'s dense data.
#[must_use]
pub fn device_vec_ptr(vec: &DeviceVec, ctx: &DeviceContext) -> RawDevicePtr<bf16> {
    crate::tensor::cache_ptr(&vec.data, ctx)
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
