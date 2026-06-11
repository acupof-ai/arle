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
