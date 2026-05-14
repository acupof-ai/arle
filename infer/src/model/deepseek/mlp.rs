//! DeepSeek V4 MoE FFN scaffold.
//!
//! The local V4 1B checkpoint uses routed experts plus a shared expert on each
//! layer. Phase 0.5 records the correct tensor shape; Phase 1 supplies the
//! shared CUDA MoE primitive, and Phase 2A wires this block into forward.

#[cfg(feature = "cuda")]
use anyhow::{Result, bail, ensure};
#[cfg(feature = "cuda")]
use cuda_kernels::{
    ffi,
    prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates},
};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
#[cfg(feature = "cuda")]
use deepseek_spec::{DeepSeekV4Config, DeepSeekV4MoeRoutingKind};
#[cfg(feature = "cuda")]
use log::info;
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use crate::distributed::expert_state::ExpertGroup;
#[cfg(test)]
use crate::distributed::expert_state::{ExpertRoute, ExpertRoutingWeights, LocalExpertRouting};
#[cfg(feature = "cuda")]
use crate::ops;

/// One SwiGLU expert: `w2(silu(w1(x)) * w3(x))`.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // populated once the Phase 2A loader allocates tensors
pub(super) struct DeepseekV4Expert {
    pub(super) w1: DeviceMatrix,
    pub(super) w2: DeviceMatrix,
    pub(super) w3: DeviceMatrix,
}

#[cfg(feature = "cuda")]
impl DeepseekV4Expert {
    /// Run one DeepSeek V4 SwiGLU expert on a packed `[tokens, hidden]` row block.
    pub(super) fn forward(
        &self,
        ctx: &DeviceContext,
        hidden: &HiddenStates,
        swiglu_limit: f32,
    ) -> Result<HiddenStates> {
        ensure!(
            self.w1.cols == hidden.hidden_dim && self.w3.cols == hidden.hidden_dim,
            "DeepSeek V4 expert input width mismatch: hidden_dim={} w1.cols={} w3.cols={}",
            hidden.hidden_dim,
            self.w1.cols,
            self.w3.cols
        );
        ensure!(
            self.w1.rows == self.w3.rows && self.w2.cols == self.w1.rows,
            "DeepSeek V4 expert intermediate mismatch: w1.rows={} w3.rows={} w2.cols={}",
            self.w1.rows,
            self.w3.rows,
            self.w2.cols
        );

        let phase = if hidden.seq_len > 1 {
            ops::LinearDispatchPhase::Prefill
        } else {
            ops::LinearDispatchPhase::Decode
        };
        let mut gate = HiddenStates::zeros(ctx, self.w1.rows, hidden.seq_len)?;
        ops::try_gemm_with_phase_into(ctx, &self.w1, hidden, &mut gate, phase)?;
        let mut up = HiddenStates::zeros(ctx, self.w3.rows, hidden.seq_len)?;
        ops::try_gemm_with_phase_into(ctx, &self.w3, hidden, &mut up, phase)?;
        let mut act = HiddenStates::zeros(ctx, self.w1.rows, hidden.seq_len)?;
        ops::dsv4_swiglu_clamped_batch_into(ctx, &gate, &up, &mut act, swiglu_limit)?;
        let mut out = HiddenStates::zeros(ctx, self.w2.rows, hidden.seq_len)?;
        ops::try_gemm_with_phase_into(ctx, &self.w2, &act, &mut out, phase)?;
        Ok(out)
    }
}

/// V4 routed MoE block plus optional shared expert.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // populated once the Phase 2A loader allocates tensors
pub(super) struct DeepseekV4MoeBlock {
    pub(super) gate_weight: DeviceMatrix,
    pub(super) gate_bias: Option<DeviceVec>,
    /// Hash-router table for early layers. The exact integer storage type is
    /// finalized with the Phase 2A loader; Phase 0.5 only validates the tensor
    /// name and keeps the field explicit.
    pub(super) gate_tid2eid: Option<CudaSlice<i64>>,
    pub(super) experts: Vec<DeepseekV4Expert>,
    pub(super) shared_experts: Option<DeepseekV4Expert>,
}

#[cfg(feature = "cuda")]
#[allow(dead_code)] // method called from forward.rs once MoE kernels land
impl DeepseekV4MoeBlock {
    /// Run routed V4 MoE for a packed `[tokens, hidden]` row block.
    pub(super) fn forward(
        &self,
        ctx: &DeviceContext,
        hidden: &HiddenStates,
        swiglu_limit: f32,
    ) -> Result<HiddenStates> {
        ensure!(
            self.experts.is_empty(),
            "DeepSeek V4 routed MoE combine is not wired yet; local experts loaded={}",
            self.experts.len()
        );
        let shared = self
            .shared_experts
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 MoE block has no shared expert"))?;
        shared.forward(ctx, hidden, swiglu_limit)
    }

    /// Run the EP-local portion of routed V4 MoE and add the shared expert.
    ///
    /// `routing` must already be localized for this EP rank. The returned
    /// tensor is this rank's partial MoE output; callers that run multiple EP
    /// ranks still need the cross-rank reduction step.
    #[cfg(test)]
    pub(super) fn forward_local_routes(
        &self,
        ctx: &DeviceContext,
        hidden: &HiddenStates,
        routing: &LocalExpertRouting,
        swiglu_limit: f32,
    ) -> Result<HiddenStates> {
        let routed = self.forward_local_routed_only(ctx, hidden, routing, swiglu_limit)?;
        self.add_shared_expert(ctx, hidden, routed, swiglu_limit)
    }

    /// Run only this EP rank's routed expert contribution.
    ///
    /// The shared expert is intentionally excluded so callers can all-reduce
    /// routed expert outputs across EP ranks and then add the shared expert
    /// exactly once per rank.
    #[cfg(test)]
    pub(super) fn forward_local_routed_only(
        &self,
        ctx: &DeviceContext,
        hidden: &HiddenStates,
        routing: &LocalExpertRouting,
        swiglu_limit: f32,
    ) -> Result<HiddenStates> {
        ensure!(
            routing.experts_per_rank == self.experts.len(),
            "DeepSeek V4 routing expects {} local experts but block loaded {}",
            routing.experts_per_rank,
            self.experts.len()
        );

        let mut out = HiddenStates::zeros(ctx, hidden.hidden_dim, hidden.seq_len)?;

        for route in &routing.routes {
            ensure!(
                route.token_idx < hidden.seq_len,
                "DeepSeek V4 route token {} out of range for seq_len {}",
                route.token_idx,
                hidden.seq_len
            );
            let expert = self.experts.get(route.local_expert_idx).ok_or_else(|| {
                anyhow::anyhow!(
                    "DeepSeek V4 local expert {} out of range for {} local experts",
                    route.local_expert_idx,
                    self.experts.len()
                )
            })?;
            let token_hidden = hidden_token(ctx, hidden, route.token_idx)?;
            let expert_out = expert.forward(ctx, &token_hidden, swiglu_limit)?;
            ops::add_scaled_row_into(ctx, &expert_out, &mut out, route.token_idx, route.weight)?;
        }

        Ok(out)
    }

    pub(super) fn add_shared_expert(
        &self,
        ctx: &DeviceContext,
        hidden: &HiddenStates,
        routed: HiddenStates,
        swiglu_limit: f32,
    ) -> Result<HiddenStates> {
        let Some(shared) = &self.shared_experts else {
            return Ok(routed);
        };
        let shared = shared.forward(ctx, hidden, swiglu_limit)?;
        ops::add_batch(ctx, &routed, &shared)
    }

    /// Route tokens with the loaded gate tensors, localize routes to this EP
    /// rank, and run the local MoE contribution.
    #[cfg(test)]
    pub(super) fn forward_routed(
        &self,
        ctx: &DeviceContext,
        layer_idx: usize,
        config: &DeepSeekV4Config,
        ep: &ExpertGroup,
        hidden: &HiddenStates,
        token_ids: &[u32],
    ) -> Result<HiddenStates> {
        let routing = self.route_local(ctx, layer_idx, config, ep, hidden, token_ids)?;
        self.forward_local_routes(ctx, hidden, &routing, config.swiglu_limit)
    }

    #[cfg(test)]
    pub(super) fn route_local_for_layer(
        &self,
        ctx: &DeviceContext,
        layer_idx: usize,
        config: &DeepSeekV4Config,
        ep: &ExpertGroup,
        hidden: &HiddenStates,
        token_ids: &[u32],
    ) -> Result<LocalExpertRouting> {
        self.route_local(ctx, layer_idx, config, ep, hidden, token_ids)
    }

    pub(super) fn forward_local_routed_gpu(
        &self,
        ctx: &DeviceContext,
        layer_idx: usize,
        config: &DeepSeekV4Config,
        ep: &ExpertGroup,
        hidden: &HiddenStates,
        token_ids: &[u32],
    ) -> Result<HiddenStates> {
        ensure!(
            token_ids.len() == hidden.seq_len,
            "DeepSeek V4 GPU route token count {} does not match hidden seq_len {}",
            token_ids.len(),
            hidden.seq_len
        );
        ensure!(
            self.gate_weight.rows == config.n_routed_experts
                && self.gate_weight.cols == hidden.hidden_dim,
            "DeepSeek V4 GPU gate shape mismatch: gate={}x{} hidden_dim={} n_routed_experts={}",
            self.gate_weight.rows,
            self.gate_weight.cols,
            hidden.hidden_dim,
            config.n_routed_experts
        );
        ensure!(
            ep.experts_per_rank == self.experts.len(),
            "DeepSeek V4 GPU route expects {} local experts but block loaded {}",
            ep.experts_per_rank,
            self.experts.len()
        );

        let trace = dsv4_moe_trace_begin(ctx)?;
        let logits = ops::gemm(ctx, &self.gate_weight, hidden)?;
        dsv4_moe_trace_end(ctx, "ffn_route_logits", layer_idx, hidden.seq_len, trace)?;

        let trace = dsv4_moe_trace_begin(ctx)?;
        let token_ids_gpu = ctx
            .stream
            .clone_htod(token_ids)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 token ids H2D failed: {err}"))?;
        let mut route_indices = ctx
            .stream
            .alloc_zeros::<i32>(hidden.seq_len * config.num_experts_per_tok)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 route index alloc failed: {err}"))?;
        let mut route_weights = ctx
            .stream
            .alloc_zeros::<f32>(hidden.seq_len * config.num_experts_per_tok)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 route weight alloc failed: {err}"))?;
        dsv4_moe_trace_end(ctx, "ffn_route_setup", layer_idx, hidden.seq_len, trace)?;

        let routing_kind = match config.moe_routing_kind(layer_idx) {
            DeepSeekV4MoeRoutingKind::Hash => 0,
            DeepSeekV4MoeRoutingKind::LearnedBias => 1,
        };
        let scoring_kind = match config.scoring_func.as_str() {
            "softmax" => 0,
            "sigmoid" => 1,
            "sqrtsoftplus" => 2,
            other => bail!("unsupported DSV4 GPU router scoring_func `{other}`"),
        };
        if routing_kind == 0 {
            ensure!(
                self.gate_tid2eid.is_some(),
                "hash-routed DeepSeek V4 MoE layer missing tid2eid"
            );
        } else {
            ensure!(
                self.gate_bias.is_some(),
                "bias-routed DeepSeek V4 MoE layer missing gate bias"
            );
        }

        let trace = dsv4_moe_trace_begin(ctx)?;
        {
            let (logits_ptr, _logits_guard) = logits.data.device_ptr(&ctx.stream);
            let bias_guard;
            let bias_ptr = if let Some(bias) = self.gate_bias.as_ref() {
                let (ptr, guard) = bias.data.device_ptr(&ctx.stream);
                bias_guard = Some(guard);
                ptr as *const ffi::Half
            } else {
                bias_guard = None;
                std::ptr::null()
            };
            let tid_guard;
            let tid_ptr = if let Some(tid2eid) = self.gate_tid2eid.as_ref() {
                let (ptr, guard) = tid2eid.device_ptr(&ctx.stream);
                tid_guard = Some(guard);
                ptr as *const i64
            } else {
                tid_guard = None;
                std::ptr::null()
            };
            let (token_ptr, _token_guard) = token_ids_gpu.device_ptr(&ctx.stream);
            let (idx_ptr, _idx_guard) = route_indices.device_ptr_mut(&ctx.stream);
            let (weight_ptr, _weight_guard) = route_weights.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::dsv4_route_cuda(
                    logits_ptr as *const ffi::Half,
                    bias_ptr,
                    tid_ptr,
                    token_ptr as *const u32,
                    idx_ptr as *mut i32,
                    weight_ptr as *mut f32,
                    hidden.seq_len as i32,
                    config.n_routed_experts as i32,
                    config.num_experts_per_tok as i32,
                    routing_kind,
                    scoring_kind,
                    config.routed_scaling_factor,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 GPU router failed: {err}"))?;
            }
            drop(bias_guard);
            drop(tid_guard);
        }
        dsv4_moe_trace_end(ctx, "ffn_route_select", layer_idx, hidden.seq_len, trace)?;

        let local_expert_start = ep.local_expert_range().start;
        let local_expert_start_i32 = i32::try_from(local_expert_start)
            .map_err(|_| anyhow::anyhow!("DeepSeek V4 local expert start overflows i32"))?;
        let experts_per_rank_i32 = i32::try_from(ep.experts_per_rank)
            .map_err(|_| anyhow::anyhow!("DeepSeek V4 experts_per_rank overflows i32"))?;
        let mut local_counts = ctx
            .stream
            .alloc_zeros::<i32>(ep.experts_per_rank)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 local route count alloc failed: {err}"))?;
        let trace = dsv4_moe_trace_begin(ctx)?;
        {
            let (idx_ptr, _idx_guard) = route_indices.device_ptr(&ctx.stream);
            let (count_ptr, _count_guard) = local_counts.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::dsv4_count_local_experts_cuda(
                    idx_ptr as *const i32,
                    count_ptr as *mut i32,
                    hidden.seq_len as i32,
                    config.num_experts_per_tok as i32,
                    local_expert_start_i32,
                    experts_per_rank_i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 local route count failed: {err}"))?;
            }
        }
        dsv4_moe_trace_end(
            ctx,
            "ffn_route_count_kernel",
            layer_idx,
            hidden.seq_len,
            trace,
        )?;
        let trace = dsv4_moe_trace_begin(ctx)?;
        let counts_host = ctx
            .stream
            .clone_dtoh(&local_counts)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 local route count D2H failed: {err}"))?;
        dsv4_moe_trace_end(ctx, "ffn_route_count_d2h", layer_idx, hidden.seq_len, trace)?;
        let mut offsets_host = Vec::with_capacity(ep.experts_per_rank);
        let mut total_local_routes = 0usize;
        for &count in &counts_host {
            ensure!(
                count >= 0,
                "DeepSeek V4 local route count kernel returned negative count {count}"
            );
            offsets_host.push(
                i32::try_from(total_local_routes).map_err(|_| {
                    anyhow::anyhow!("DeepSeek V4 packed route offset overflows i32")
                })?,
            );
            total_local_routes += usize::try_from(count)
                .map_err(|_| anyhow::anyhow!("DeepSeek V4 local route count overflows usize"))?;
        }

        let mut out = HiddenStates::zeros(ctx, hidden.hidden_dim, hidden.seq_len)?;
        if total_local_routes == 0 {
            return Ok(out);
        }

        let trace = dsv4_moe_trace_begin(ctx)?;
        let offsets_gpu = ctx
            .stream
            .clone_htod(&offsets_host)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 local route offsets H2D failed: {err}"))?;
        let mut pack_cursors = ctx
            .stream
            .alloc_zeros::<i32>(ep.experts_per_rank)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 local route cursor alloc failed: {err}"))?;
        let mut packed_hidden = HiddenStates::zeros(ctx, hidden.hidden_dim, total_local_routes)?;
        let mut packed_token = ctx
            .stream
            .alloc_zeros::<i32>(total_local_routes)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 packed token alloc failed: {err}"))?;
        let mut packed_weight = ctx
            .stream
            .alloc_zeros::<f32>(total_local_routes)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 packed weight alloc failed: {err}"))?;
        dsv4_moe_trace_end(
            ctx,
            "ffn_route_pack_setup",
            layer_idx,
            hidden.seq_len,
            trace,
        )?;
        let trace = dsv4_moe_trace_begin(ctx)?;
        {
            let (hidden_ptr, _hidden_guard) = hidden.data.device_ptr(&ctx.stream);
            let (idx_ptr, _idx_guard) = route_indices.device_ptr(&ctx.stream);
            let (weight_ptr, _weight_guard) = route_weights.device_ptr(&ctx.stream);
            let (offset_ptr, _offset_guard) = offsets_gpu.device_ptr(&ctx.stream);
            let (cursor_ptr, _cursor_guard) = pack_cursors.device_ptr_mut(&ctx.stream);
            let (packed_hidden_ptr, _packed_hidden_guard) =
                packed_hidden.data.device_ptr_mut(&ctx.stream);
            let (packed_token_ptr, _packed_token_guard) = packed_token.device_ptr_mut(&ctx.stream);
            let (packed_weight_ptr, _packed_weight_guard) =
                packed_weight.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::dsv4_pack_local_experts_cuda(
                    hidden_ptr as *const ffi::Half,
                    idx_ptr as *const i32,
                    weight_ptr as *const f32,
                    offset_ptr as *const i32,
                    cursor_ptr as *mut i32,
                    packed_hidden_ptr as *mut ffi::Half,
                    packed_token_ptr as *mut i32,
                    packed_weight_ptr as *mut f32,
                    hidden.seq_len as i32,
                    hidden.hidden_dim as i32,
                    config.num_experts_per_tok as i32,
                    local_expert_start_i32,
                    experts_per_rank_i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 local expert pack failed: {err}"))?;
            }
        }
        dsv4_moe_trace_end(
            ctx,
            "ffn_route_pack_kernel",
            layer_idx,
            hidden.seq_len,
            trace,
        )?;

        let trace = dsv4_moe_trace_begin(ctx)?;
        for (local_expert_idx, expert) in self.experts.iter().enumerate() {
            let count = usize::try_from(counts_host[local_expert_idx])
                .map_err(|_| anyhow::anyhow!("DeepSeek V4 local route count overflows usize"))?;
            if count == 0 {
                continue;
            }
            let offset = usize::try_from(offsets_host[local_expert_idx])
                .map_err(|_| anyhow::anyhow!("DeepSeek V4 packed route offset overflows usize"))?;
            let elem_start = offset * hidden.hidden_dim;
            let elem_end = elem_start + count * hidden.hidden_dim;
            let mut expert_input = HiddenStates::zeros(ctx, hidden.hidden_dim, count)?;
            {
                let src = packed_hidden.data.slice(elem_start..elem_end);
                ctx.stream
                    .memcpy_dtod(&src, &mut expert_input.data)
                    .map_err(|err| anyhow::anyhow!("DeepSeek V4 expert input D2D failed: {err}"))?;
            }

            let expert_out = expert.forward(ctx, &expert_input, config.swiglu_limit)?;
            let (expert_ptr, _expert_guard) = expert_out.data.device_ptr(&ctx.stream);
            let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
            let (token_ptr, _token_guard) = packed_token.device_ptr(&ctx.stream);
            let (weight_ptr, _weight_guard) = packed_weight.device_ptr(&ctx.stream);
            unsafe {
                ffi::dsv4_scatter_packed_expert_cuda(
                    expert_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    token_ptr as *const i32,
                    weight_ptr as *const f32,
                    offsets_host[local_expert_idx],
                    counts_host[local_expert_idx],
                    hidden.hidden_dim as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| {
                    anyhow::anyhow!("DeepSeek V4 packed expert scatter failed: {err}")
                })?;
            }
        }
        dsv4_moe_trace_end(ctx, "ffn_expert_loop", layer_idx, hidden.seq_len, trace)?;
        Ok(out)
    }

    #[cfg(test)]
    fn route_local(
        &self,
        ctx: &DeviceContext,
        layer_idx: usize,
        config: &DeepSeekV4Config,
        ep: &ExpertGroup,
        hidden: &HiddenStates,
        token_ids: &[u32],
    ) -> Result<LocalExpertRouting> {
        ensure!(
            token_ids.len() == hidden.seq_len,
            "DeepSeek V4 route token count {} does not match hidden seq_len {}",
            token_ids.len(),
            hidden.seq_len
        );
        ensure!(
            self.gate_weight.rows == config.n_routed_experts
                && self.gate_weight.cols == hidden.hidden_dim,
            "DeepSeek V4 gate shape mismatch: gate={}x{} hidden_dim={} n_routed_experts={}",
            self.gate_weight.rows,
            self.gate_weight.cols,
            hidden.hidden_dim,
            config.n_routed_experts
        );
        if let Some(bias) = &self.gate_bias {
            ensure!(
                bias.len == config.n_routed_experts,
                "DeepSeek V4 gate bias len {} does not match n_routed_experts {}",
                bias.len,
                config.n_routed_experts
            );
        }

        let logits = ops::gemm(ctx, &self.gate_weight, hidden)?;
        let logits_host = ctx.stream.clone_dtoh(&logits.data)?;
        let bias_host = self
            .gate_bias
            .as_ref()
            .map(|bias| ctx.stream.clone_dtoh(&bias.data))
            .transpose()?
            .map(|bias| {
                bias.into_iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>()
            });
        let mut routes = Vec::with_capacity(hidden.seq_len * config.num_experts_per_tok);

        for token_idx in 0..hidden.seq_len {
            let start = token_idx * logits.hidden_dim;
            let token_logits = logits_host[start..start + logits.hidden_dim]
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>();
            let scores = config.router_scores_from_logits(&token_logits)?;
            let hash_experts = match config.moe_routing_kind(layer_idx) {
                DeepSeekV4MoeRoutingKind::Hash => {
                    Some(self.hash_experts_for_token(ctx, config, token_ids[token_idx])?)
                }
                DeepSeekV4MoeRoutingKind::LearnedBias => None,
            };
            let token_routes = config.moe_routes_from_scores(
                layer_idx,
                token_idx,
                &scores,
                bias_host.as_deref(),
                hash_experts.as_deref(),
            )?;
            routes.extend(token_routes.into_iter().map(|route| ExpertRoute {
                token_idx: route.token_idx,
                expert_idx: route.expert_idx,
                weight: route.weight,
            }));
        }

        ep.localize_routing(&ExpertRoutingWeights::new(config.n_routed_experts, routes))
    }

    #[cfg(test)]
    fn hash_experts_for_token(
        &self,
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        token_id: u32,
    ) -> Result<Vec<usize>> {
        let table = self
            .gate_tid2eid
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("hash-routed DeepSeek V4 MoE layer missing tid2eid"))?;
        ensure!(
            (token_id as usize) < config.vocab_size,
            "DeepSeek V4 token id {token_id} exceeds vocab_size {}",
            config.vocab_size
        );
        let start = token_id as usize * config.num_experts_per_tok;
        let end = start + config.num_experts_per_tok;
        ensure!(
            end <= table.len(),
            "DeepSeek V4 tid2eid table too short: need {} entries for token {}, have {}",
            end,
            token_id,
            table.len()
        );
        let experts_i64 = ctx.stream.clone_dtoh(&table.slice(start..end))?;
        experts_i64
            .into_iter()
            .map(|expert_idx| {
                ensure!(
                    expert_idx >= 0,
                    "DeepSeek V4 tid2eid contains negative expert id"
                );
                usize::try_from(expert_idx)
                    .map_err(|_| anyhow::anyhow!("DeepSeek V4 tid2eid expert id overflow"))
            })
            .collect()
    }
}

#[cfg(feature = "cuda")]
fn dsv4_moe_trace_enabled() -> bool {
    std::env::var("ARLE_DSV4_TRACE_LAYER")
        .ok()
        .is_some_and(|raw| !matches!(raw.as_str(), "0" | "false" | "FALSE" | "off" | "OFF"))
}

#[cfg(feature = "cuda")]
fn dsv4_moe_trace_begin(ctx: &DeviceContext) -> Result<Instant> {
    if dsv4_moe_trace_enabled() {
        ctx.stream
            .synchronize()
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 MoE trace pre-sync failed: {err}"))?;
    }
    Ok(Instant::now())
}

#[cfg(feature = "cuda")]
fn dsv4_moe_trace_end(
    ctx: &DeviceContext,
    phase: &str,
    layer_idx: usize,
    tokens: usize,
    started: Instant,
) -> Result<()> {
    if !dsv4_moe_trace_enabled() {
        return Ok(());
    }
    ctx.stream
        .synchronize()
        .map_err(|err| anyhow::anyhow!("DeepSeek V4 MoE trace post-sync failed: {err}"))?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    info!(
        "dsv4_trace layer={} phase={} tokens={} elapsed_ms={:.3}",
        layer_idx, phase, tokens, elapsed_ms
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[cfg(test)]
fn hidden_token(
    ctx: &DeviceContext,
    hidden: &HiddenStates,
    token_idx: usize,
) -> Result<HiddenStates> {
    let token = ops::extract_vec(ctx, hidden, token_idx)?;
    Ok(HiddenStates {
        data: token.data,
        hidden_dim: hidden.hidden_dim,
        seq_len: 1,
    })
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::distributed::expert_state::{ExpertGroup, LocalExpertRoute};
    use half::bf16;

    fn bf16_vec(values: &[f32]) -> Vec<bf16> {
        values.iter().map(|&value| bf16::from_f32(value)).collect()
    }

    fn silu(value: f32) -> f32 {
        value / (1.0 + (-value).exp())
    }

    fn tiny_config() -> DeepSeekV4Config {
        DeepSeekV4Config::from_json_str(
            r#"{
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "torch_dtype": "bfloat16",
            "vocab_size": 16,
            "hidden_size": 2,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 1,
            "hidden_act": "silu",
            "swiglu_limit": 10.0,
            "q_lora_rank": 1,
            "o_lora_rank": 1,
            "o_groups": 1,
            "qk_rope_head_dim": 1,
            "n_routed_experts": 4,
            "n_shared_experts": 0,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 1,
            "routed_scaling_factor": 1.0,
            "norm_topk_prob": false,
            "scoring_func": "softmax",
            "topk_method": "noaux_tc",
            "index_n_heads": 1,
            "index_head_dim": 1,
            "index_topk": 1,
            "num_hash_layers": 0,
            "sliding_window": 4,
            "compress_ratios": [0],
            "compress_rope_theta": 160000.0,
            "hc_mult": 1,
            "hc_sinkhorn_iters": 1,
            "hc_eps": 1.0e-6,
            "num_nextn_predict_layers": 0,
            "max_position_embeddings": 16,
            "rope_theta": 10000.0,
            "rope_scaling": {
                "type": "yarn",
                "factor": 1.0,
                "original_max_position_embeddings": 16,
                "beta_fast": 32.0,
                "beta_slow": 1.0
            },
            "rms_norm_eps": 1.0e-6,
            "initializer_range": 0.02,
            "tie_word_embeddings": false,
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "eos_token_id": 1
        }"#,
        )
        .unwrap()
    }

    #[test]
    fn expert_forward_runs_clamped_swiglu_on_gpu() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let hidden = HiddenStates {
            data: ctx.stream.clone_htod(&bf16_vec(&[1.0, -2.0, 0.5, 3.0]))?,
            hidden_dim: 2,
            seq_len: 2,
        };
        let expert = DeepseekV4Expert {
            w1: DeviceMatrix::from_host(
                &ctx,
                &bf16_vec(&[
                    1.0, 0.0, //
                    0.0, 1.0, //
                    1.0, 1.0,
                ]),
                3,
                2,
            )?,
            w2: DeviceMatrix::from_host(
                &ctx,
                &bf16_vec(&[
                    1.0, 0.0, 0.5, //
                    0.0, 1.0, -1.0,
                ]),
                2,
                3,
            )?,
            w3: DeviceMatrix::from_host(
                &ctx,
                &bf16_vec(&[
                    0.5, 0.0, //
                    0.0, -1.0, //
                    1.0, -1.0,
                ]),
                3,
                2,
            )?,
        };

        let out = expert.forward(&ctx, &hidden, 2.0)?;
        let out_host = ctx.stream.clone_dtoh(&out.data)?;
        ctx.sync()?;

        let inputs = [[1.0_f32, -2.0_f32], [0.5_f32, 3.0_f32]];
        let mut expected = Vec::new();
        for x in inputs {
            let gate = [x[0], x[1], x[0] + x[1]];
            let up = [0.5 * x[0], -x[1], x[0] - x[1]];
            let act = [
                silu(gate[0].min(2.0)) * up[0].clamp(-2.0, 2.0),
                silu(gate[1].min(2.0)) * up[1].clamp(-2.0, 2.0),
                silu(gate[2].min(2.0)) * up[2].clamp(-2.0, 2.0),
            ];
            expected.push(act[0] + 0.5 * act[2]);
            expected.push(act[1] - act[2]);
        }

        for (idx, got) in out_host.iter().enumerate() {
            assert!(
                (got.to_f32() - expected[idx]).abs() < 0.05,
                "idx={idx} expected={} got={}",
                expected[idx],
                got.to_f32()
            );
        }
        Ok(())
    }

    #[test]
    fn moe_forward_local_routes_accumulates_ep_local_experts() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let hidden = HiddenStates {
            data: ctx.stream.clone_htod(&bf16_vec(&[1.0, 2.0, 3.0, -1.0]))?,
            hidden_dim: 2,
            seq_len: 2,
        };
        let expert0 = DeepseekV4Expert {
            w1: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 0.0]), 1, 2)?,
            w2: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 2.0]), 2, 1)?,
            w3: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 0.0]), 1, 2)?,
        };
        let expert1 = DeepseekV4Expert {
            w1: DeviceMatrix::from_host(&ctx, &bf16_vec(&[0.0, 1.0]), 1, 2)?,
            w2: DeviceMatrix::from_host(&ctx, &bf16_vec(&[-1.0, 0.5]), 2, 1)?,
            w3: DeviceMatrix::from_host(&ctx, &bf16_vec(&[0.0, 1.0]), 1, 2)?,
        };
        let block = DeepseekV4MoeBlock {
            gate_weight: DeviceMatrix::from_host(&ctx, &bf16_vec(&[0.0, 0.0]), 1, 2)?,
            gate_bias: None,
            gate_tid2eid: None,
            experts: vec![expert0, expert1],
            shared_experts: None,
        };
        let routing = LocalExpertRouting {
            num_global_experts: 4,
            experts_per_rank: 2,
            routes: vec![
                LocalExpertRoute {
                    token_idx: 0,
                    global_expert_idx: 0,
                    local_expert_idx: 0,
                    weight: 0.25,
                },
                LocalExpertRoute {
                    token_idx: 0,
                    global_expert_idx: 1,
                    local_expert_idx: 1,
                    weight: 0.5,
                },
                LocalExpertRoute {
                    token_idx: 1,
                    global_expert_idx: 1,
                    local_expert_idx: 1,
                    weight: 1.0,
                },
            ],
        };

        let out = block.forward_local_routes(&ctx, &hidden, &routing, 10.0)?;
        let out_host = ctx.stream.clone_dtoh(&out.data)?;
        ctx.sync()?;

        let e0_t0 = silu(1.0) * 1.0;
        let e1_t0 = silu(2.0) * 2.0;
        let e1_t1 = silu(-1.0) * -1.0;
        let expected = [
            0.25 * e0_t0 - 0.5 * e1_t0,
            0.25 * (2.0 * e0_t0) + 0.5 * (0.5 * e1_t0),
            -e1_t1,
            0.5 * e1_t1,
        ];

        for (idx, got) in out_host.iter().enumerate() {
            assert!(
                (got.to_f32() - expected[idx]).abs() < 0.05,
                "idx={idx} expected={} got={}",
                expected[idx],
                got.to_f32()
            );
        }
        Ok(())
    }

    #[test]
    fn moe_forward_routed_computes_gate_routes_and_localizes_ep() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let config = tiny_config();
        let ep = ExpertGroup::new(0, 2, config.n_routed_experts)?;
        let hidden = HiddenStates {
            data: ctx.stream.clone_htod(&bf16_vec(&[1.0, 0.0, 0.0, 2.0]))?,
            hidden_dim: 2,
            seq_len: 2,
        };
        let expert0 = DeepseekV4Expert {
            w1: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 0.0]), 1, 2)?,
            w2: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 2.0]), 2, 1)?,
            w3: DeviceMatrix::from_host(&ctx, &bf16_vec(&[1.0, 0.0]), 1, 2)?,
        };
        let expert1 = DeepseekV4Expert {
            w1: DeviceMatrix::from_host(&ctx, &bf16_vec(&[0.0, 1.0]), 1, 2)?,
            w2: DeviceMatrix::from_host(&ctx, &bf16_vec(&[-1.0, 0.5]), 2, 1)?,
            w3: DeviceMatrix::from_host(&ctx, &bf16_vec(&[0.0, 1.0]), 1, 2)?,
        };
        let block = DeepseekV4MoeBlock {
            gate_weight: DeviceMatrix::from_host(
                &ctx,
                &bf16_vec(&[
                    1.0, 0.0, //
                    0.0, 1.0, //
                    -1.0, 0.0, //
                    0.0, -1.0,
                ]),
                4,
                2,
            )?,
            gate_bias: Some(DeviceVec::from_host(
                &ctx,
                &bf16_vec(&[0.0, 0.0, 0.0, 0.0]),
            )?),
            gate_tid2eid: None,
            experts: vec![expert0, expert1],
            shared_experts: None,
        };

        let out = block.forward_routed(&ctx, 0, &config, &ep, &hidden, &[3, 4])?;
        let out_host = ctx.stream.clone_dtoh(&out.data)?;
        ctx.sync()?;

        let token0_scores = config.router_scores_from_logits(&[1.0, 0.0, -1.0, 0.0])?;
        let token1_scores = config.router_scores_from_logits(&[0.0, 2.0, 0.0, -2.0])?;
        let e0_t0 = silu(1.0) * 1.0;
        let e1_t1 = silu(2.0) * 2.0;
        let expected = [
            token0_scores[0] * e0_t0,
            token0_scores[0] * (2.0 * e0_t0),
            -token1_scores[1] * e1_t1,
            token1_scores[1] * (0.5 * e1_t1),
        ];

        for (idx, got) in out_host.iter().enumerate() {
            assert!(
                (got.to_f32() - expected[idx]).abs() < 0.05,
                "idx={idx} expected={} got={}",
                expected[idx],
                got.to_f32()
            );
        }
        Ok(())
    }
}
