//! Single-GPU BF16 MoE forward — Qwen3.5/3.6 SparseMoeBlock (all experts local).
//!
//! Routing runs on DEVICE: `dsv4_route` with a zero selection bias is exactly
//! greedy top-k. `--qwen35-gpu-router false` selects the host `infer_moe::route`
//! reference, which costs a full-stream `ctx.sync` + logits D2H + 2×H2D per layer
//! per step. DSv4 uses the device route kernel unconditionally.

use infer_moe::RoutingDecision;

/// Flatten per-token routing into the token-major flat buffers the `dsv4_*`
/// kernels read at `route = token * topk + k`, each of length `num_tokens *
/// topk`. Each decision must carry exactly `topk` experts; order is preserved.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn flatten_routing(
    decisions: &[RoutingDecision],
    topk: usize,
) -> anyhow::Result<(Vec<i32>, Vec<f32>)> {
    let total = decisions.len() * topk;
    let mut indices = Vec::with_capacity(total);
    let mut weights = Vec::with_capacity(total);
    for (token, decision) in decisions.iter().enumerate() {
        anyhow::ensure!(
            decision.experts.len() == topk,
            "routing decision for token {token} has {} experts, expected topk {topk}",
            decision.experts.len()
        );
        for ew in &decision.experts {
            indices.push(
                i32::try_from(ew.expert).map_err(|_| {
                    anyhow::anyhow!("expert id {} exceeds i32 kernel ABI", ew.expert)
                })?,
            );
            weights.push(ew.weight);
        }
    }
    Ok((indices, weights))
}

/// DeepGEMM m-grouped-contiguous row alignment on SM90: the kernel resolves the
/// B-side expert group ONCE per BLOCK_M output tile from `m_indices[tile_start]`
/// and BLOCK_M is pinned to 128 upstream — so every group's row segment must
/// start at a multiple of 128, with `-1` on every pad row.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const DEEPGEMM_CONTIG_ALIGN: usize = 128;

/// DSv4 decode-band per-group row alignment. 64 is the smallest legal value:
/// the SM90 warpgroup MMA grants only block_m ∈ {64, 128}, and the native bridge
/// caps block_m at the alignment the caller packed with. At decode shapes
/// (R = topk·B ≤ 128) the row-linear kernels dominate, so halving pad rows halves
/// them (128-alignment inflated the lane 149×, −23% e2e B=1).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const DSV4_DECODE_CONTIG_ALIGN: usize = 64;

/// Routed-row ceiling for the 64-aligned decode band: `topk(8) × B_max(16)`.
/// Above it the GEMM tail dominates and block_m=128 wins — use
/// [`DEEPGEMM_CONTIG_ALIGN`].
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const DSV4_DECODE_CONTIG_MAX_ROUTES: usize = 128;

/// Host upper bound on the `align`-aligned packed row count for the DeepGEMM
/// contiguous grouped layout. The true total is device-resident while the GEMM's
/// `m` and the TMA descriptors are host values, so the launch uses this cap
/// instead of a per-layer D2H sync:
/// `Σ_g align(c_g, a) ≤ align(R, a) + a·min(R, G)`. Tiles past the true aligned
/// total carry `m_indices = -1`. Always a multiple of `align`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn deepgemm_contig_rows_cap(
    total_routes: usize,
    local_experts: usize,
    align: usize,
) -> usize {
    total_routes.div_ceil(align) * align + align * local_experts.min(total_routes)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
/// Routed-row floor below which the DeepGEMM grouped path loses to the hand
/// CUDA-core kernels (decode R=8 hand +8%, prefill R=16384 DeepGEMM -33% needle
/// wall; 1024 = 128-token chunk x top-8). The live value is
/// `--qwen35-deepgemm-min-routes`.
pub(crate) const QWEN35_DEEPGEMM_MIN_ROUTES: usize = 1024;

/// Routed-row ceiling for the decode-specialized weight-read-bound grouped
/// kernels: `256 = top_k(8) × B_max(32)`, the batched-decode envelope where each
/// expert receives ≤ B rows so weight traffic dominates (the batch kernels burn
/// 487 µs/layer at R=8 on scalar 2B loads, ≈3% of HBM bandwidth, vs a 25-60
/// µs/layer target).
///
/// Must stay `< QWEN35_DEEPGEMM_MIN_ROUTES` so the decode band never shadows the
/// DeepGEMM prefill dispatch.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const QWEN35_MOE_DECODE_MAX_ROUTES: usize = 256;

/// Allocate an i32 buffer pre-filled with `-1` ON DEVICE (memset 0xFF) — the
/// "route slot not packed on this rank" sentinel.
///
/// A `clone_htod(&vec![-1; n])` would record a CUDA-graph memcpy node whose HOST
/// source dies with this call, so replay reads freed memory.
#[cfg(feature = "cuda")]
fn alloc_neg1_i32(
    ctx: &cuda_kernels::prelude::DeviceContext,
    len: usize,
) -> anyhow::Result<cudarc::driver::CudaSlice<i32>> {
    use cudarc::driver::DevicePtrMut;

    let mut buf = ctx
        .stream
        .alloc_zeros::<i32>(len)
        .map_err(|e| anyhow::anyhow!("MoE -1 sentinel alloc failed: {e}"))?;
    let n_bytes = len * std::mem::size_of::<i32>();
    let (ptr, _guard) = buf.device_ptr_mut(&ctx.stream);
    // SAFETY: `ptr` is a live device allocation of `n_bytes` on this stream.
    unsafe {
        cudarc::driver::result::memset_d8_async(ptr, 0xFF, n_bytes, ctx.stream.cu_stream())
            .map_err(|e| anyhow::anyhow!("MoE -1 sentinel memset failed: {e}"))?;
    }
    drop(_guard);
    Ok(buf)
}

#[cfg(feature = "cuda")]
#[path = "moe/qwen.rs"]
mod qwen;
// `dead_code` marks pending-consumer infra (the `model.rs` DSv4 branch), not cruft.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
#[path = "moe/dsv4.rs"]
mod dsv4;
#[cfg(feature = "deepep")]
#[path = "moe/dsv4_deepep.rs"]
mod dsv4_deepep;
#[cfg(feature = "cuda")]
#[allow(unused_imports)] // consumed by the model.rs DSv4 branch
pub(crate) use dsv4::{
    Dsv4GemvTables, Dsv4MoeTailScratch, Dsv4SharedDecodeScratch, Dsv4W4A16GemvTables, GroupedCache,
    GroupedWeightLayout, W4Afp8ExpertWeights, build_grouped_cache, dsv4_moe_forward,
    dsv4_shared_expert_forward, dsv4_shared_expert_forward_decode_scratch,
};
#[cfg(feature = "deepep")]
pub(crate) use dsv4_deepep::{dsv4_moe_forward_deepep, dsv4_moe_forward_deepep_ll};
#[cfg(feature = "cuda")]
pub(crate) use qwen::{
    MoeForwardScratch, moe_forward, moe_forward_into, qwen35_decode_moe_graph_capturable,
};
