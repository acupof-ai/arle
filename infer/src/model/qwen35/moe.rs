//! Qwen3.5-MoE / Qwen3.6 single-GPU SOTA-grouped MoE forward (BF16,
//! correctness-first).
//!
//! Architecture (NOT a per-expert loop — grouped permute → grouped GEMM →
//! combine), reusing the DeepSeek-V4 grouped-MoE kernel pipeline on a single
//! GPU (no expert-parallel all-to-all; all experts are local):
//!
//!   route (plain softmax + top-k)
//!     → optional `norm_topk_prob` renorm
//!     → permute/pack tokens grouped-by-expert (route → token, weight)
//!     → grouped expert GEMM (gate + up paired, BF16)
//!     → SwiGLU
//!     → grouped down GEMM (BF16)
//!     → scale-by-route-weight + scatter-accumulate back to token rows
//!     → + shared expert (dense SwiGLU) * sigmoid(x @ shared_expert_gate)
//!
//! Kernels reused from the DSv4 pipeline (dtype-agnostic on BF16 activations):
//!   - `dsv4_route_cuda` (scoring_kind=0 softmax, routing_kind=1 block-argmax
//!     top-k, bias=null) — the router top-k + weight build.
//!   - `dsv4_count_local_experts_cuda` / `dsv4_exclusive_scan_i32_cuda` —
//!     per-expert route counts → group offsets.
//!   - `dsv4_pack_local_experts_cuda` — permute hidden grouped-by-expert,
//!     emitting `packed_token` + `packed_weight` for the combine.
//!   - `dsv4_scatter_packed_expert_cuda` — weighted scatter-accumulate of the
//!     packed expert outputs back into the per-token routed output.
//!
//! New BF16 kernels (this change):
//!   - `moe_bf16_grouped_gemm_pair_batch_cuda` (gate+up), `..._batch_cuda`
//!     (down) — BF16 mirror of the DSv4 FP8 grouped GEMM structure.
//!   - `qwen36_renorm_topk_weights_cuda` — `norm_topk_prob` renorm.
//!   - `qwen36_add_shared_expert_gated_cuda` — shared-expert sigmoid gate.
//!
//! W4 grouped GEMM + perf tuning is an explicit follow-up; this path is
//! correctness-first and runs on `tiny-random/qwen3.5-moe` (BF16 safetensors).

use std::collections::HashMap;

use anyhow::{Result, ensure};
use cudarc::driver::{CudaSlice, DevicePtr};
use safetensors::SafeTensors;

#[cfg(feature = "cuda")]
use cudarc::driver::DevicePtrMut;

#[cfg(feature = "cuda")]
use super::config::Config35;
#[cfg(feature = "cuda")]
use crate::ops;
#[cfg(feature = "cuda")]
use cuda_kernels::ffi;
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::HiddenStates;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix};

/// MoE block weights for one Qwen3.6 decoder layer (single-GPU; all experts
/// local). Mirrors the Metal `mlx_qwen35_moe_block.cpp` weight set.
///
/// Fields are consumed by the CUDA grouped forward; under non-CUDA builds the
/// struct still type-checks (it is loaded by the shared weight loader) but the
/// forward is `#[cfg(feature = "cuda")]`, hence the dead-code allowance.
#[allow(dead_code)]
pub(super) struct MoeMlp {
    /// Router: `[num_experts, hidden]`. `gemm(gate, hidden)` → per-token logits.
    pub(super) gate: DeviceMatrix,
    /// Per-expert gate/up/down projections (BF16). Length == `num_experts`.
    pub(super) expert_gate: Vec<DeviceMatrix>,
    pub(super) expert_up: Vec<DeviceMatrix>,
    pub(super) expert_down: Vec<DeviceMatrix>,
    /// Dense shared expert (always runs, every token).
    pub(super) shared_gate: DeviceMatrix,
    pub(super) shared_up: DeviceMatrix,
    pub(super) shared_down: DeviceMatrix,
    /// Scalar shared-expert router: `[1, hidden]`. `sigmoid(gemm(.,x))` gates
    /// the shared-expert contribution per token.
    pub(super) shared_gate_router: DeviceMatrix,

    pub(super) num_experts: usize,
    pub(super) top_k: usize,
    pub(super) norm_topk_prob: bool,

    /// Device-resident `[num_experts]` arrays of the per-expert weight base
    /// pointers, consumed by the grouped GEMM kernels (one launch over all
    /// experts). Built once at load.
    expert_gate_ptrs: CudaSlice<u64>,
    expert_up_ptrs: CudaSlice<u64>,
    expert_down_ptrs: CudaSlice<u64>,

    /// Zeroed `[num_experts]` BF16 router bias. `dsv4_route_cuda` with
    /// `routing_kind=1` (block-argmax top-k) unconditionally reads `bias[e]`;
    /// Qwen3.6 has no router bias, so we pass an all-zero buffer (additive
    /// identity) rather than null. Allocated once at load.
    zero_bias: CudaSlice<u16>,
}

/// MLP variant on a Qwen3.5 transformer block: classic dense SwiGLU, or the
/// Qwen3.6 sparse MoE block. Dense remains the canonical Qwen3.5 path and is
/// bit-identical to before this change.
#[allow(clippy::large_enum_variant)]
pub(super) enum Mlp {
    Dense(crate::model::common::MLP),
    Moe(MoeMlp),
}

impl Mlp {
    /// Dense weights for the existing gemm-into call sites. Panics if called
    /// on a MoE layer — call sites must branch on `as_moe()` first.
    pub(super) fn dense(&self) -> &crate::model::common::MLP {
        match self {
            Mlp::Dense(mlp) => mlp,
            Mlp::Moe(_) => {
                panic!("Mlp::dense() called on a MoE layer; branch on as_moe() first")
            }
        }
    }

    pub(super) fn as_moe(&self) -> Option<&MoeMlp> {
        match self {
            Mlp::Moe(moe) => Some(moe),
            Mlp::Dense(_) => None,
        }
    }

    /// Whether any dense projection uses the Marlin W4A8 quant path. MoE
    /// layers are always plain BF16 in this (correctness-first) path, so they
    /// never report Marlin.
    pub(super) fn is_marlin_w4a8(&self) -> bool {
        match self {
            Mlp::Dense(mlp) => {
                mlp.gate_proj.is_marlin_w4a8()
                    || mlp.up_proj.is_marlin_w4a8()
                    || mlp.down_proj.is_marlin_w4a8()
            }
            Mlp::Moe(_) => false,
        }
    }
}

/// Resolve a tensor name that may or may not carry the `.weight` suffix
/// (HF safetensors ship `.weight`; some exports drop it).
fn resolve_name(
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    base: &str,
) -> Result<String> {
    let with_weight = format!("{base}.weight");
    if crate::weight_loader::tensor_exists(shards, weight_map, &with_weight) {
        return Ok(with_weight);
    }
    if crate::weight_loader::tensor_exists(shards, weight_map, base) {
        return Ok(base.to_string());
    }
    Err(anyhow::anyhow!(
        "Qwen3.6 MoE tensor not found: neither `{with_weight}` nor `{base}`"
    ))
}

/// Load one Qwen3.6 MoE block's weights from safetensors.
///
/// `prefix` is e.g. `model.language_model.layers.3` (the HF convention the
/// CUDA safetensors loader uses). Tensor names:
///   router : `{prefix}.mlp.gate(.weight)`
///   experts: `{prefix}.mlp.experts.{i}.{gate_proj,up_proj,down_proj}(.weight)`
///   shared : `{prefix}.mlp.shared_expert.{gate_proj,up_proj,down_proj}(.weight)`
///   s-gate : `{prefix}.mlp.shared_expert_gate(.weight)`
///
/// The stacked `switch_mlp.{gate,up,down}_proj` convention (one [E, ...] tensor
/// per projection) is detected and rejected with a clear error — slicing the
/// expert axis into per-expert `DeviceMatrix`es is an explicit follow-up
/// (the BF16 smoke checkpoint `tiny-random/qwen3.5-moe` uses the per-expert
/// `experts.{i}.*` layout).
pub(super) fn load_moe_mlp(
    ctx: &DeviceContext,
    shards: &[SafeTensors],
    weight_map: &HashMap<String, usize>,
    prefix: &str,
    config: &super::config::Config35,
) -> Result<MoeMlp> {
    use crate::weight_loader::{load_tensor_2d, tensor_exists};

    let mlp_prefix = format!("{prefix}.mlp");
    let num_experts = config.num_experts;
    ensure!(
        num_experts > 0 && config.num_experts_per_tok > 0,
        "Qwen3.6 MoE layer requires num_experts>0 and num_experts_per_tok>0 (got {} / {})",
        num_experts,
        config.num_experts_per_tok
    );

    // Routed experts ship in one of two layouts:
    //   • per-expert    : `experts.{i}.{gate,up,down}_proj` (tiny-random / some
    //     HF exports) — loaded as separate matrices.
    //   • stacked+fused : `experts.gate_up_proj` `[E, 2*moe_inter, hidden]`
    //     (gate‖up on the output axis) + `experts.down_proj`
    //     `[E, hidden, moe_inter]` (production Qwen3.6-35B-A3B) — sliced
    //     per-expert into the same `DeviceMatrix` set below (bit-identical).
    // The legacy stacked `switch_mlp.*` convention is NOT supported.
    let per_expert = tensor_exists(
        shards,
        weight_map,
        &format!("{mlp_prefix}.experts.0.gate_proj.weight"),
    ) || tensor_exists(
        shards,
        weight_map,
        &format!("{mlp_prefix}.experts.0.gate_proj"),
    );
    let stacked_fused = tensor_exists(
        shards,
        weight_map,
        &format!("{mlp_prefix}.experts.gate_up_proj.weight"),
    ) || tensor_exists(
        shards,
        weight_map,
        &format!("{mlp_prefix}.experts.gate_up_proj"),
    );
    let switch_stacked = tensor_exists(
        shards,
        weight_map,
        &format!("{mlp_prefix}.switch_mlp.gate_proj.weight"),
    ) || tensor_exists(
        shards,
        weight_map,
        &format!("{mlp_prefix}.switch_mlp.gate_proj"),
    );

    let load = |base: &str| -> Result<DeviceMatrix> {
        let name = resolve_name(shards, weight_map, base)?;
        load_tensor_2d(ctx, shards, weight_map, &name)
    };

    let gate = load(&format!("{mlp_prefix}.gate"))?;

    let mut expert_gate = Vec::with_capacity(num_experts);
    let mut expert_up = Vec::with_capacity(num_experts);
    let mut expert_down = Vec::with_capacity(num_experts);
    if per_expert {
        for i in 0..num_experts {
            let ep = format!("{mlp_prefix}.experts.{i}");
            expert_gate.push(load(&format!("{ep}.gate_proj"))?);
            expert_up.push(load(&format!("{ep}.up_proj"))?);
            expert_down.push(load(&format!("{ep}.down_proj"))?);
        }
    } else if stacked_fused {
        use crate::weight_loader::load_stacked_expert_2d;
        let hidden = config.hidden_size;
        let mi = config.moe_intermediate_size;
        let gate_up = resolve_name(
            shards,
            weight_map,
            &format!("{mlp_prefix}.experts.gate_up_proj"),
        )?;
        let down = resolve_name(
            shards,
            weight_map,
            &format!("{mlp_prefix}.experts.down_proj"),
        )?;
        for i in 0..num_experts {
            // gate_up_proj [E, 2*mi, hidden]: gate = rows [0, mi), up = rows [mi, 2*mi).
            expert_gate.push(load_stacked_expert_2d(
                ctx,
                shards,
                weight_map,
                &gate_up,
                i,
                num_experts,
                2 * mi,
                0,
                mi,
                hidden,
            )?);
            expert_up.push(load_stacked_expert_2d(
                ctx,
                shards,
                weight_map,
                &gate_up,
                i,
                num_experts,
                2 * mi,
                mi,
                mi,
                hidden,
            )?);
            // down_proj [E, hidden, mi].
            expert_down.push(load_stacked_expert_2d(
                ctx,
                shards,
                weight_map,
                &down,
                i,
                num_experts,
                hidden,
                0,
                hidden,
                mi,
            )?);
        }
    } else {
        anyhow::bail!(
            "Qwen3.6 MoE layer `{mlp_prefix}`: no recognized expert layout — need per-expert \
             `experts.{{i}}.gate_proj` or stacked+fused `experts.gate_up_proj`+`experts.down_proj`{}.",
            if switch_stacked {
                " (found unsupported `switch_mlp.*`)"
            } else {
                ""
            }
        );
    }

    let shared_prefix = format!("{mlp_prefix}.shared_expert");
    let shared_gate = load(&format!("{shared_prefix}.gate_proj"))?;
    let shared_up = load(&format!("{shared_prefix}.up_proj"))?;
    let shared_down = load(&format!("{shared_prefix}.down_proj"))?;
    let shared_gate_router = load(&format!("{mlp_prefix}.shared_expert_gate"))?;

    MoeMlp::finalize_ptrs(
        ctx,
        gate,
        expert_gate,
        expert_up,
        expert_down,
        shared_gate,
        shared_up,
        shared_down,
        shared_gate_router,
        num_experts,
        config.num_experts_per_tok,
        config.norm_topk_prob,
    )
}

impl MoeMlp {
    /// Build the device-resident per-expert weight-pointer arrays after the
    /// per-expert `DeviceMatrix`es are loaded.
    pub(super) fn finalize_ptrs(
        ctx: &DeviceContext,
        gate: DeviceMatrix,
        expert_gate: Vec<DeviceMatrix>,
        expert_up: Vec<DeviceMatrix>,
        expert_down: Vec<DeviceMatrix>,
        shared_gate: DeviceMatrix,
        shared_up: DeviceMatrix,
        shared_down: DeviceMatrix,
        shared_gate_router: DeviceMatrix,
        num_experts: usize,
        top_k: usize,
        norm_topk_prob: bool,
    ) -> Result<Self> {
        ensure!(
            expert_gate.len() == num_experts
                && expert_up.len() == num_experts
                && expert_down.len() == num_experts,
            "Qwen3.6 MoE expert count mismatch: gate={} up={} down={} num_experts={}",
            expert_gate.len(),
            expert_up.len(),
            expert_down.len(),
            num_experts
        );
        let ptr_vec = |mats: &[DeviceMatrix]| -> Vec<u64> {
            mats.iter()
                .map(|m| {
                    let (p, _g) = m.data.device_ptr(&ctx.stream);
                    p
                })
                .collect()
        };
        let expert_gate_ptrs = ctx
            .stream
            .clone_htod(&ptr_vec(&expert_gate))
            .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE expert gate ptr H2D failed: {e}"))?;
        let expert_up_ptrs = ctx
            .stream
            .clone_htod(&ptr_vec(&expert_up))
            .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE expert up ptr H2D failed: {e}"))?;
        let expert_down_ptrs = ctx
            .stream
            .clone_htod(&ptr_vec(&expert_down))
            .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE expert down ptr H2D failed: {e}"))?;
        let zero_bias = ctx
            .stream
            .alloc_zeros::<u16>(num_experts)
            .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE zero-bias alloc failed: {e}"))?;
        Ok(Self {
            gate,
            expert_gate,
            expert_up,
            expert_down,
            shared_gate,
            shared_up,
            shared_down,
            shared_gate_router,
            num_experts,
            top_k,
            norm_topk_prob,
            expert_gate_ptrs,
            expert_up_ptrs,
            expert_down_ptrs,
            zero_bias,
        })
    }
}

#[cfg(feature = "cuda")]
impl MoeMlp {
    /// Grouped MoE forward for a packed `[seq_len, hidden]` row block.
    ///
    /// `normed` is the post-attention-layernorm hidden (token-major
    /// `[seq_len, hidden]`). Returns the MoE output (routed experts + gated
    /// shared expert) as a fresh `[seq_len, hidden]` `HiddenStates`.
    pub(super) fn forward(
        &self,
        ctx: &DeviceContext,
        config: &Config35,
        normed: &HiddenStates,
    ) -> Result<HiddenStates> {
        let num_tokens = normed.seq_len;
        let hidden_dim = normed.hidden_dim;
        ensure!(
            self.gate.cols == hidden_dim && self.gate.rows == self.num_experts,
            "Qwen3.6 MoE router shape mismatch: gate={}x{} hidden_dim={} num_experts={}",
            self.gate.rows,
            self.gate.cols,
            hidden_dim,
            self.num_experts
        );

        // ── 1. Router: logits [num_tokens, num_experts] (token-major). ──────
        let logits = ops::gemm(ctx, &self.gate, normed)?;

        let total_routes = num_tokens * self.top_k;
        let mut route_indices = ctx
            .stream
            .alloc_zeros::<i32>(total_routes)
            .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE route index alloc failed: {e}"))?;
        let mut route_weights = ctx
            .stream
            .alloc_zeros::<f32>(total_routes)
            .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE route weight alloc failed: {e}"))?;

        // dsv4_route_cuda with scoring_kind=0 (plain softmax), routing_kind=1
        // (block-argmax top-k), zeroed bias (Qwen3.6 has no router bias; the
        // kernel reads bias[e] unconditionally under routing_kind=1, so we pass
        // an all-zero buffer rather than null), routed_scaling_factor=1.0.
        // token_ids is only read for routing_kind==0 (hash) — pass null.
        {
            let (logits_ptr, _lg) = logits.data.device_ptr(&ctx.stream);
            let (bias_ptr, _bg) = self.zero_bias.device_ptr(&ctx.stream);
            let (idx_ptr, _ig) = route_indices.device_ptr_mut(&ctx.stream);
            let (w_ptr, _wg) = route_weights.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::dsv4_route_cuda(
                    logits_ptr as *const ffi::Half,
                    bias_ptr as *const ffi::Half,
                    std::ptr::null(),
                    std::ptr::null(),
                    idx_ptr as *mut i32,
                    w_ptr as *mut f32,
                    num_tokens as i32,
                    self.num_experts as i32,
                    self.top_k as i32,
                    1, // routing_kind = learned/argmax top-k
                    0, // scoring_kind = softmax
                    1.0,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE router failed: {e}"))?;
            }
        }

        // ── 2. Optional norm_topk_prob renorm over the route weights. ───────
        if self.norm_topk_prob {
            let (w_ptr, _wg) = route_weights.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::qwen36_renorm_topk_weights_cuda(
                    w_ptr as *mut f32,
                    num_tokens as i32,
                    self.top_k as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE topk renorm failed: {e}"))?;
            }
        }

        // ── 3. Per-expert route counts → group offsets. ─────────────────────
        let mut counts = ctx
            .stream
            .alloc_zeros::<i32>(self.num_experts)
            .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE count alloc failed: {e}"))?;
        let mut offsets = ctx
            .stream
            .alloc_zeros::<i32>(self.num_experts)
            .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE offset alloc failed: {e}"))?;
        {
            let (idx_ptr, _ig) = route_indices.device_ptr(&ctx.stream);
            let (count_ptr, _cg) = counts.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::dsv4_count_local_experts_cuda(
                    idx_ptr as *const i32,
                    count_ptr as *mut i32,
                    num_tokens as i32,
                    self.top_k as i32,
                    0, // local_expert_start (single GPU → 0)
                    self.num_experts as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE count failed: {e}"))?;
            }
        }
        {
            let (count_ptr, _cg) = counts.device_ptr(&ctx.stream);
            let (off_ptr, _og) = offsets.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::dsv4_exclusive_scan_i32_cuda(
                    count_ptr as *const i32,
                    off_ptr as *mut i32,
                    std::ptr::null_mut(),
                    self.num_experts as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE offset scan failed: {e}"))?;
            }
        }

        // ── 4. Permute/pack hidden grouped-by-expert. ───────────────────────
        // packed_hidden: [total_routes, hidden]; packed_token/packed_weight:
        // [total_routes]. cursors scratch is zeroed per expert.
        let mut packed_hidden = HiddenStates::zeros(ctx, hidden_dim, total_routes)?;
        let mut packed_token = ctx
            .stream
            .alloc_zeros::<i32>(total_routes)
            .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE packed_token alloc failed: {e}"))?;
        let mut packed_weight = ctx
            .stream
            .alloc_zeros::<f32>(total_routes)
            .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE packed_weight alloc failed: {e}"))?;
        let mut cursors = ctx
            .stream
            .alloc_zeros::<i32>(self.num_experts)
            .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE cursors alloc failed: {e}"))?;
        {
            let (h_ptr, _hg) = normed.data.device_ptr(&ctx.stream);
            let (idx_ptr, _ig) = route_indices.device_ptr(&ctx.stream);
            let (rw_ptr, _rwg) = route_weights.device_ptr(&ctx.stream);
            let (off_ptr, _og) = offsets.device_ptr(&ctx.stream);
            let (cur_ptr, _cg) = cursors.device_ptr_mut(&ctx.stream);
            let (ph_ptr, _phg) = packed_hidden.data.device_ptr_mut(&ctx.stream);
            let (pt_ptr, _ptg) = packed_token.device_ptr_mut(&ctx.stream);
            let (pw_ptr, _pwg) = packed_weight.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::dsv4_pack_local_experts_cuda(
                    h_ptr as *const ffi::Half,
                    idx_ptr as *const i32,
                    rw_ptr as *const f32,
                    off_ptr as *const i32,
                    cur_ptr as *mut i32,
                    ph_ptr as *mut ffi::Half,
                    pt_ptr as *mut i32,
                    pw_ptr as *mut f32,
                    num_tokens as i32,
                    hidden_dim as i32,
                    self.top_k as i32,
                    0,
                    self.num_experts as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE pack failed: {e}"))?;
            }
        }

        // ── 5. Grouped expert GEMM (gate + up paired). ──────────────────────
        let moe_inter = self.expert_gate[0].rows;
        ensure!(
            self.expert_gate[0].cols == hidden_dim && self.expert_up[0].cols == hidden_dim,
            "Qwen3.6 MoE expert gate/up cols {} / {} != hidden_dim {}",
            self.expert_gate[0].cols,
            self.expert_up[0].cols,
            hidden_dim
        );
        let mut gate_out = HiddenStates::zeros(ctx, moe_inter, total_routes)?;
        let mut up_out = HiddenStates::zeros(ctx, moe_inter, total_routes)?;
        // `max_count` only sizes grid Y; total_routes is a safe upper bound.
        let max_count = total_routes.max(1);
        {
            let (wg_ptr, _wgg) = self.expert_gate_ptrs.device_ptr(&ctx.stream);
            let (wu_ptr, _wug) = self.expert_up_ptrs.device_ptr(&ctx.stream);
            let (x_ptr, _xg) = packed_hidden.data.device_ptr(&ctx.stream);
            let (off_ptr, _og) = offsets.device_ptr(&ctx.stream);
            let (count_ptr, _cg) = counts.device_ptr(&ctx.stream);
            let (ga_ptr, _gag) = gate_out.data.device_ptr_mut(&ctx.stream);
            let (ua_ptr, _uag) = up_out.data.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::moe_bf16_grouped_gemm_pair_batch_cuda(
                    wg_ptr as *const u64,
                    wu_ptr as *const u64,
                    x_ptr as *const ffi::Half,
                    ga_ptr as *mut ffi::Half,
                    ua_ptr as *mut ffi::Half,
                    off_ptr as *const i32,
                    count_ptr as *const i32,
                    std::ptr::null(),
                    self.num_experts as i32,
                    max_count as i32,
                    moe_inter as i32,
                    hidden_dim as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE grouped gate/up GEMM failed: {e}"))?;
            }
        }

        // ── 6. SwiGLU (unclamped). ──────────────────────────────────────────
        let act = ops::silu_mul_batch(ctx, &gate_out, &up_out)?;

        // ── 7. Grouped down GEMM. ───────────────────────────────────────────
        ensure!(
            self.expert_down[0].cols == moe_inter && self.expert_down[0].rows == hidden_dim,
            "Qwen3.6 MoE expert down shape {}x{} != hidden_dim {} / moe_inter {}",
            self.expert_down[0].rows,
            self.expert_down[0].cols,
            hidden_dim,
            moe_inter
        );
        let mut expert_out = HiddenStates::zeros(ctx, hidden_dim, total_routes)?;
        {
            let (wd_ptr, _wdg) = self.expert_down_ptrs.device_ptr(&ctx.stream);
            let (a_ptr, _ag) = act.data.device_ptr(&ctx.stream);
            let (off_ptr, _og) = offsets.device_ptr(&ctx.stream);
            let (count_ptr, _cg) = counts.device_ptr(&ctx.stream);
            let (eo_ptr, _eog) = expert_out.data.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::moe_bf16_grouped_gemm_batch_cuda(
                    wd_ptr as *const u64,
                    a_ptr as *const ffi::Half,
                    eo_ptr as *mut ffi::Half,
                    off_ptr as *const i32,
                    count_ptr as *const i32,
                    std::ptr::null(),
                    self.num_experts as i32,
                    max_count as i32,
                    hidden_dim as i32,
                    moe_inter as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE grouped down GEMM failed: {e}"))?;
            }
        }

        // ── 8. Scale-by-route-weight + scatter-accumulate to token rows. ────
        let mut routed = HiddenStates::zeros(ctx, hidden_dim, num_tokens)?;
        {
            let (eo_ptr, _eog) = expert_out.data.device_ptr(&ctx.stream);
            let (out_ptr, _og) = routed.data.device_ptr_mut(&ctx.stream);
            let (pt_ptr, _ptg) = packed_token.device_ptr(&ctx.stream);
            let (pw_ptr, _pwg) = packed_weight.device_ptr(&ctx.stream);
            unsafe {
                ffi::dsv4_scatter_packed_expert_cuda(
                    eo_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    pt_ptr as *const i32,
                    pw_ptr as *const f32,
                    0,
                    total_routes as i32,
                    hidden_dim as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE scatter-combine failed: {e}"))?;
            }
        }

        // ── 9. Shared expert: dense SwiGLU * sigmoid(x @ shared_gate_router).
        let shared = self.shared_expert_forward(ctx, normed)?;
        let gate_logit = ops::gemm(ctx, &self.shared_gate_router, normed)?;
        {
            let (out_ptr, _og) = routed.data.device_ptr_mut(&ctx.stream);
            let (sh_ptr, _shg) = shared.data.device_ptr(&ctx.stream);
            let (gl_ptr, _glg) = gate_logit.data.device_ptr(&ctx.stream);
            unsafe {
                ffi::qwen36_add_shared_expert_gated_cuda(
                    out_ptr as *mut ffi::Half,
                    sh_ptr as *const ffi::Half,
                    gl_ptr as *const ffi::Half,
                    num_tokens as i32,
                    hidden_dim as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow::anyhow!("Qwen3.6 MoE shared-expert gate failed: {e}"))?;
            }
        }
        let _ = config;
        Ok(routed)
    }

    /// Dense shared-expert SwiGLU: `down(silu(gate(x)) * up(x))`.
    fn shared_expert_forward(
        &self,
        ctx: &DeviceContext,
        normed: &HiddenStates,
    ) -> Result<HiddenStates> {
        let gate = ops::gemm(ctx, &self.shared_gate, normed)?;
        let up = ops::gemm(ctx, &self.shared_up, normed)?;
        let act = ops::silu_mul_batch(ctx, &gate, &up)?;
        ops::gemm(ctx, &self.shared_down, &act)
    }
}
