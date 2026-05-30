//! Per-request mutable state for the DeepSeek V4 model scaffold.
//!
//! Mirrors `Qwen3State`: wraps `GenerationStateBase` and forwards every
//! `GenerationState` method through it. V4-specific scratch (hybrid attention,
//! compressor/indexer metadata, and MoE route buffers) will live alongside
//! `decode_bufs` once Phase 2A exposes the kernel surfaces.

#[cfg(all(test, feature = "cuda"))]
use std::collections::VecDeque;

use anyhow::Result;

use crate::model::GenerationState;
#[cfg(feature = "cuda")]
use crate::model::generation_state::GenerationStateBase;
#[cfg(feature = "cuda")]
use crate::model::kv_cache::KVCacheDtype;
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates, PagedKVPool};
#[cfg(feature = "cuda")]
use cuda_kernels::tensor::CudaAllocTraceExt;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
#[cfg(feature = "cuda")]
use half::bf16;

/// Per-request DeepSeek mutable state.
///
/// Currently a thin wrapper over `GenerationStateBase` so the trait surface
/// matches `Qwen3State`. V4 attention/MoE scratch lands here when the kernel
/// surface is real.
pub struct DeepseekState {
    #[cfg(feature = "cuda")]
    pub(crate) base: GenerationStateBase,
    #[cfg(feature = "cuda")]
    pub(crate) decode_logits: DeviceVec,
    #[cfg(feature = "cuda")]
    pub(crate) sample_probs: CudaSlice<f32>,
    #[cfg(feature = "cuda")]
    pub(crate) sample_out: CudaSlice<i32>,
    #[cfg(feature = "cuda")]
    pub(crate) reference_tokens: Vec<u32>,
    #[cfg(feature = "cuda")]
    pub(crate) incremental: DeepseekIncrementalState,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
pub(crate) struct DeepseekIncrementalState {
    pub(crate) processed_tokens: usize,
    pub(crate) layers: Vec<DeepseekLayerRuntimeCache>,
    pub(crate) stream_recycle: Option<DeepseekHiddenRuntimeScratch>,
}

#[cfg(feature = "cuda")]
impl DeepseekIncrementalState {
    pub(crate) fn clear(&mut self) {
        self.processed_tokens = 0;
        self.layers.clear();
        self.stream_recycle = None;
    }

    pub(crate) fn ensure_layers(&mut self, layers: usize) {
        if self.layers.len() < layers {
            self.layers
                .resize_with(layers, DeepseekLayerRuntimeCache::default);
        }
    }

    pub(crate) fn trim_prefill_scratch(&mut self) {
        self.stream_recycle = None;
        for layer in &mut self.layers {
            layer.trim_prefill_scratch();
        }
    }
}

#[cfg(feature = "cuda")]
impl DeepseekLayerRuntimeCache {
    pub(crate) fn trim_prefill_scratch(&mut self) {
        self.stream_recycle = None;
        self.attn_mhc = None;
        self.ffn_mhc = None;
        self.attn_pre = None;
        self.attn_normed = None;
        self.attn_post = None;
        self.ffn_pre = None;
        self.ffn_normed = None;
        self.moe = DeepseekMoeRuntimeCache::default();
        self.attention.trim_prefill_scratch();
    }
}

#[cfg(feature = "cuda")]
#[derive(Default)]
pub(crate) struct DeepseekLayerRuntimeCache {
    pub(crate) attention: DeepseekAttentionRuntimeCache,
    pub(crate) moe: DeepseekMoeRuntimeCache,
    pub(crate) stream_recycle: Option<DeepseekHiddenRuntimeScratch>,
    pub(crate) attn_mhc: Option<DeepseekMhcRuntimeScratch>,
    pub(crate) ffn_mhc: Option<DeepseekMhcRuntimeScratch>,
    pub(crate) attn_pre: Option<DeepseekHiddenRuntimeScratch>,
    pub(crate) attn_normed: Option<DeepseekHiddenRuntimeScratch>,
    pub(crate) attn_post: Option<DeepseekHiddenRuntimeScratch>,
    pub(crate) ffn_pre: Option<DeepseekHiddenRuntimeScratch>,
    pub(crate) ffn_normed: Option<DeepseekHiddenRuntimeScratch>,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
pub(crate) struct DeepseekMoeRuntimeCache {
    pub(crate) route_logits: Option<DeepseekRouteLogitsRuntimeScratch>,
    pub(crate) dispatch: Option<DeepseekDispatchRuntimeScratch>,
    pub(crate) dispatch_payload: Option<DeepseekDispatchPayloadRuntimeScratch>,
    pub(crate) send_route: Option<DeepseekSendRouteRuntimeScratch>,
    pub(crate) recv_route: Option<DeepseekRecvRouteRuntimeScratch>,
    pub(crate) local_route: Option<DeepseekLocalRouteRuntimeScratch>,
    pub(crate) expert: Option<DeepseekExpertRuntimeScratch>,
    pub(crate) shared_expert: Option<DeepseekExpertRuntimeScratch>,
    pub(crate) grouped: Option<DeepseekGroupedExpertRuntimeScratch>,
    pub(crate) route_combine: Option<DeepseekRouteCombineRuntimeScratch>,
    pub(crate) native_deepep: Option<DeepseekNativeDeepEpRuntimeScratch>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekNativeDeepEpRuntimeScratch {
    pub(crate) capacity_tokens: usize,
    pub(crate) capacity_recv: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) topk: usize,
    pub(crate) ep_world: usize,
    pub(crate) num_experts: usize,
    pub(crate) num_channels: usize,
    pub(crate) topk_idx_i64: CudaSlice<i64>,
    pub(crate) recv_x: HiddenStates,
    pub(crate) recv_src_idx: CudaSlice<i32>,
    pub(crate) recv_topk_idx: CudaSlice<i64>,
    pub(crate) recv_topk_w: CudaSlice<f32>,
    pub(crate) rank_prefix: CudaSlice<i32>,
    pub(crate) recv_channel_prefix: CudaSlice<i32>,
    pub(crate) send_head: CudaSlice<i32>,
    pub(crate) num_tokens_per_rank: CudaSlice<i32>,
    pub(crate) num_tokens_per_expert: CudaSlice<i32>,
    pub(crate) is_token_in_rank: CudaSlice<u8>,
    pub(crate) channel_prefix_matrix: CudaSlice<i32>,
    pub(crate) combined_x: HiddenStates,
    pub(crate) combined_topk_w: CudaSlice<f32>,
    pub(crate) expert_out: HiddenStates,
    pub(crate) experts_per_rank: usize,
    pub(crate) recv_topk_idx_i32: CudaSlice<i32>,
    pub(crate) local_counts: CudaSlice<i32>,
    pub(crate) local_offsets: CudaSlice<i32>,
    pub(crate) local_cursors: CudaSlice<i32>,
    pub(crate) packed_x: HiddenStates,
    pub(crate) packed_token: CudaSlice<i32>,
    pub(crate) packed_weight: CudaSlice<f32>,
    /// Grouped DeepGEMM expert scratch — populated by
    /// `forward_native_deepep_routed_gpu` when EXPERT_BACKEND=deepgemm
    /// is active. Taken into a temporary cache for
    /// `forward_deepgemm_all_dsv4_experts_gpu`, then restored.
    pub(crate) grouped: Option<DeepseekGroupedExpertRuntimeScratch>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekRouteLogitsRuntimeScratch {
    pub(crate) capacity_tokens: usize,
    pub(crate) n_experts: usize,
    pub(crate) logits: HiddenStates,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekDispatchRuntimeScratch {
    pub(crate) capacity_tokens: usize,
    pub(crate) capacity_routes: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) topk: usize,
    pub(crate) ep_world: usize,
    pub(crate) experts_per_rank: usize,
    pub(crate) token_ids: CudaSlice<u32>,
    pub(crate) route_indices: CudaSlice<i32>,
    pub(crate) route_weights: CudaSlice<f32>,
    pub(crate) send_rank_counts: CudaSlice<i32>,
    pub(crate) send_rank_offsets: CudaSlice<i32>,
    pub(crate) rank_cursors: CudaSlice<i32>,
    pub(crate) send_hidden: HiddenStates,
    pub(crate) send_meta: CudaSlice<i32>,
    pub(crate) all_rank_counts: CudaSlice<i32>,
    pub(crate) recv_rank_counts: CudaSlice<i32>,
    pub(crate) local_counts: CudaSlice<i32>,
    pub(crate) local_offsets: CudaSlice<i32>,
    pub(crate) local_cursors: CudaSlice<i32>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekDispatchPayloadRuntimeScratch {
    pub(crate) capacity_routes: usize,
    pub(crate) stride_elems: usize,
    pub(crate) send_payload: CudaSlice<bf16>,
    pub(crate) recv_payload: CudaSlice<bf16>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekSendRouteRuntimeScratch {
    pub(crate) capacity_routes: usize,
    pub(crate) send_token: CudaSlice<i32>,
    pub(crate) send_route_slot: CudaSlice<i32>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekRecvRouteRuntimeScratch {
    pub(crate) capacity_routes: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) recv_hidden: HiddenStates,
    pub(crate) recv_meta: CudaSlice<i32>,
    pub(crate) route_out: HiddenStates,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekLocalRouteRuntimeScratch {
    pub(crate) capacity_routes: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) expert_hidden: HiddenStates,
    pub(crate) expert_weight: CudaSlice<f32>,
    pub(crate) expert_route_slot: CudaSlice<i32>,
    pub(crate) route_out: HiddenStates,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekMhcRuntimeScratch {
    pub(crate) capacity_tokens: usize,
    pub(crate) stream_hidden_dim: usize,
    pub(crate) mix_dim: usize,
    pub(crate) hc_mult: usize,
    pub(crate) mixes: HiddenStates,
    pub(crate) pre: CudaSlice<f32>,
    pub(crate) post: CudaSlice<f32>,
    pub(crate) comb: CudaSlice<f32>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekHiddenRuntimeScratch {
    pub(crate) capacity_tokens: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) hidden: HiddenStates,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekExpertRuntimeScratch {
    pub(crate) capacity_tokens: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) intermediate_dim: usize,
    pub(crate) output_dim: usize,
    pub(crate) input: HiddenStates,
    pub(crate) gate: HiddenStates,
    pub(crate) up: HiddenStates,
    pub(crate) act: HiddenStates,
    pub(crate) out: HiddenStates,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekGroupedExpertRuntimeScratch {
    pub(crate) capacity_routes: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) intermediate_dim: usize,
    pub(crate) gate: HiddenStates,
    pub(crate) up: HiddenStates,
    pub(crate) act: HiddenStates,
    pub(crate) out: HiddenStates,
    pub(crate) w1_ptrs: Option<DeepseekGroupedExpertWeightPtrCache>,
    pub(crate) w3_ptrs: Option<DeepseekGroupedExpertWeightPtrCache>,
    pub(crate) w2_ptrs: Option<DeepseekGroupedExpertWeightPtrCache>,
    pub(crate) active: Option<DeepseekGroupedExpertActiveScratch>,
    pub(crate) deepgemm: Option<DeepseekDeepGemmExpertRuntimeScratch>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekGroupedExpertActiveScratch {
    pub(crate) capacity_experts: usize,
    pub(crate) indices: CudaSlice<i32>,
    pub(crate) offsets: CudaSlice<i32>,
    pub(crate) counts: CudaSlice<i32>,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekDeepGemmExpertRuntimeScratch {
    pub(crate) capacity_experts: usize,
    pub(crate) capacity_m: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) intermediate_dim: usize,
    pub(crate) scale_stride_m: usize,
    pub(crate) input_fp8: CudaSlice<u8>,
    pub(crate) input_scales: CudaSlice<f32>,
    pub(crate) w13_out: HiddenStates,
    pub(crate) act_fp8: CudaSlice<u8>,
    pub(crate) act_scales: CudaSlice<f32>,
    pub(crate) out_padded: HiddenStates,
    pub(crate) out_compact: HiddenStates,
    pub(crate) masked_m: CudaSlice<i32>,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DeepseekDsv4GroupedBlockFormat {
    Fp8,
    Fp4,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekGroupedExpertWeightPtrCache {
    pub(crate) weight_ptrs: CudaSlice<u64>,
    pub(crate) scale_ptrs: CudaSlice<u64>,
    pub(crate) format: DeepseekDsv4GroupedBlockFormat,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) scale_rows: usize,
    pub(crate) scale_cols: usize,
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekRouteCombineRuntimeScratch {
    pub(crate) capacity_routes: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) combine_recv: HiddenStates,
    pub(crate) route_slot_out: HiddenStates,
    pub(crate) combine_fp8_send: CudaSlice<u8>,
    pub(crate) combine_fp8_recv: CudaSlice<u8>,
    pub(crate) combine_scale_send: CudaSlice<f32>,
    pub(crate) combine_scale_recv: CudaSlice<f32>,
}

#[cfg(feature = "cuda")]
impl DeepseekMoeRuntimeCache {
    pub(crate) fn ensure_dispatch_scratch(
        &mut self,
        ctx: &DeviceContext,
        hidden_dim: usize,
        capacity_tokens: usize,
        topk: usize,
        ep_world: usize,
        experts_per_rank: usize,
    ) -> Result<&mut DeepseekDispatchRuntimeScratch> {
        let capacity_tokens = capacity_tokens.max(1);
        let capacity_routes = capacity_tokens.saturating_mul(topk).max(1);
        let needs_alloc = self
            .dispatch
            .as_ref()
            .map(|scratch| {
                scratch.capacity_tokens < capacity_tokens
                    || scratch.capacity_routes < capacity_routes
                    || scratch.hidden_dim != hidden_dim
                    || scratch.topk != topk
                    || scratch.ep_world != ep_world
                    || scratch.experts_per_rank != experts_per_rank
            })
            .unwrap_or(true);
        if needs_alloc {
            self.dispatch = Some(DeepseekDispatchRuntimeScratch {
                capacity_tokens,
                capacity_routes,
                hidden_dim,
                topk,
                ep_world,
                experts_per_rank,
                token_ids: unsafe { ctx.stream.alloc_zeros_traced::<u32>(capacity_tokens)? },
                route_indices: unsafe { ctx.stream.alloc_zeros_traced::<i32>(capacity_routes)? },
                route_weights: unsafe { ctx.stream.alloc_zeros_traced::<f32>(capacity_routes)? },
                send_rank_counts: unsafe { ctx.stream.alloc_zeros_traced::<i32>(ep_world)? },
                send_rank_offsets: unsafe { ctx.stream.alloc_zeros_traced::<i32>(ep_world)? },
                rank_cursors: unsafe { ctx.stream.alloc_zeros_traced::<i32>(ep_world)? },
                send_hidden: unsafe { HiddenStates::uninit(ctx, hidden_dim, capacity_routes)? },
                send_meta: unsafe { ctx.stream.alloc_zeros_traced::<i32>(capacity_routes * 3)? },
                all_rank_counts: unsafe {
                    ctx.stream.alloc_zeros_traced::<i32>(ep_world * ep_world)?
                },
                recv_rank_counts: unsafe { ctx.stream.alloc_zeros_traced::<i32>(ep_world)? },
                local_counts: unsafe { ctx.stream.alloc_zeros_traced::<i32>(experts_per_rank)? },
                local_offsets: unsafe { ctx.stream.alloc_zeros_traced::<i32>(experts_per_rank)? },
                local_cursors: unsafe { ctx.stream.alloc_zeros_traced::<i32>(experts_per_rank)? },
            });
        }
        Ok(self
            .dispatch
            .as_mut()
            .expect("DeepSeek V4 dispatch scratch allocated"))
    }

    pub(crate) fn ensure_expert_scratch(
        &mut self,
        ctx: &DeviceContext,
        hidden_dim: usize,
        intermediate_dim: usize,
        output_dim: usize,
        capacity_tokens: usize,
    ) -> Result<&mut DeepseekExpertRuntimeScratch> {
        let capacity_tokens = capacity_tokens.max(1);
        let needs_alloc = self
            .expert
            .as_ref()
            .map(|scratch| {
                scratch.capacity_tokens < capacity_tokens
                    || scratch.hidden_dim != hidden_dim
                    || scratch.intermediate_dim != intermediate_dim
                    || scratch.output_dim != output_dim
            })
            .unwrap_or(true);
        if needs_alloc {
            self.expert = Some(DeepseekExpertRuntimeScratch {
                capacity_tokens,
                hidden_dim,
                intermediate_dim,
                output_dim,
                input: unsafe { HiddenStates::uninit(ctx, hidden_dim, capacity_tokens)? },
                gate: unsafe { HiddenStates::uninit(ctx, intermediate_dim, capacity_tokens)? },
                up: unsafe { HiddenStates::uninit(ctx, intermediate_dim, capacity_tokens)? },
                act: unsafe { HiddenStates::uninit(ctx, intermediate_dim, capacity_tokens)? },
                out: unsafe { HiddenStates::uninit(ctx, output_dim, capacity_tokens)? },
            });
        }
        Ok(self
            .expert
            .as_mut()
            .expect("DeepSeek V4 expert scratch allocated"))
    }

    pub(crate) fn ensure_shared_expert_scratch(
        &mut self,
        ctx: &DeviceContext,
        hidden_dim: usize,
        intermediate_dim: usize,
        output_dim: usize,
        capacity_tokens: usize,
    ) -> Result<&mut DeepseekExpertRuntimeScratch> {
        let capacity_tokens = capacity_tokens.max(1);
        let needs_alloc = self
            .shared_expert
            .as_ref()
            .map(|scratch| {
                scratch.capacity_tokens < capacity_tokens
                    || scratch.hidden_dim != hidden_dim
                    || scratch.intermediate_dim != intermediate_dim
                    || scratch.output_dim != output_dim
            })
            .unwrap_or(true);
        if needs_alloc {
            self.shared_expert = Some(DeepseekExpertRuntimeScratch {
                capacity_tokens,
                hidden_dim,
                intermediate_dim,
                output_dim,
                input: unsafe { HiddenStates::uninit(ctx, hidden_dim, capacity_tokens)? },
                gate: unsafe { HiddenStates::uninit(ctx, intermediate_dim, capacity_tokens)? },
                up: unsafe { HiddenStates::uninit(ctx, intermediate_dim, capacity_tokens)? },
                act: unsafe { HiddenStates::uninit(ctx, intermediate_dim, capacity_tokens)? },
                out: unsafe { HiddenStates::uninit(ctx, output_dim, capacity_tokens)? },
            });
        }
        Ok(self
            .shared_expert
            .as_mut()
            .expect("DeepSeek V4 shared expert scratch allocated"))
    }

    pub(crate) fn ensure_grouped_expert_scratch(
        &mut self,
        ctx: &DeviceContext,
        hidden_dim: usize,
        intermediate_dim: usize,
        capacity_routes: usize,
    ) -> Result<&mut DeepseekGroupedExpertRuntimeScratch> {
        let capacity_routes = capacity_routes.max(1);
        let needs_alloc = self
            .grouped
            .as_ref()
            .map(|scratch| {
                scratch.capacity_routes < capacity_routes
                    || scratch.hidden_dim != hidden_dim
                    || scratch.intermediate_dim != intermediate_dim
            })
            .unwrap_or(true);
        if needs_alloc {
            self.grouped = Some(DeepseekGroupedExpertRuntimeScratch {
                capacity_routes,
                hidden_dim,
                intermediate_dim,
                gate: unsafe { HiddenStates::uninit(ctx, intermediate_dim, capacity_routes)? },
                up: unsafe { HiddenStates::uninit(ctx, intermediate_dim, capacity_routes)? },
                act: unsafe { HiddenStates::uninit(ctx, intermediate_dim, capacity_routes)? },
                out: unsafe { HiddenStates::uninit(ctx, hidden_dim, capacity_routes)? },
                w1_ptrs: None,
                w3_ptrs: None,
                w2_ptrs: None,
                active: None,
                deepgemm: None,
            });
        }
        Ok(self
            .grouped
            .as_mut()
            .expect("DeepSeek V4 grouped expert scratch allocated"))
    }

    pub(crate) fn ensure_route_combine_scratch(
        &mut self,
        ctx: &DeviceContext,
        hidden_dim: usize,
        capacity_routes: usize,
    ) -> Result<&mut DeepseekRouteCombineRuntimeScratch> {
        let capacity_routes = capacity_routes.max(1);
        let needs_alloc = self
            .route_combine
            .as_ref()
            .map(|scratch| {
                scratch.capacity_routes < capacity_routes || scratch.hidden_dim != hidden_dim
            })
            .unwrap_or(true);
        if needs_alloc {
            self.route_combine = Some(DeepseekRouteCombineRuntimeScratch {
                capacity_routes,
                hidden_dim,
                combine_recv: unsafe { HiddenStates::uninit(ctx, hidden_dim, capacity_routes)? },
                route_slot_out: unsafe { HiddenStates::uninit(ctx, hidden_dim, capacity_routes)? },
                combine_fp8_send: unsafe {
                    ctx.stream
                        .alloc_zeros_traced::<u8>(hidden_dim * capacity_routes)?
                },
                combine_fp8_recv: unsafe {
                    ctx.stream
                        .alloc_zeros_traced::<u8>(hidden_dim * capacity_routes)?
                },
                combine_scale_send: unsafe {
                    ctx.stream.alloc_zeros_traced::<f32>(capacity_routes)?
                },
                combine_scale_recv: unsafe {
                    ctx.stream.alloc_zeros_traced::<f32>(capacity_routes)?
                },
            });
        }
        Ok(self
            .route_combine
            .as_mut()
            .expect("DeepSeek V4 route combine scratch allocated"))
    }

    pub(crate) fn ensure_native_deepep_scratch(
        &mut self,
        ctx: &DeviceContext,
        hidden_dim: usize,
        capacity_tokens: usize,
        topk: usize,
        ep_world: usize,
        num_experts: usize,
        num_channels: usize,
        experts_per_rank: usize,
    ) -> Result<&mut DeepseekNativeDeepEpRuntimeScratch> {
        let capacity_tokens = capacity_tokens.max(1);
        // Worst-case received tokens: every input could be broadcast to every
        // rank's experts. Match DeepEP's upper bound (num_tokens * world).
        let capacity_recv = capacity_tokens.saturating_mul(ep_world).max(1);
        let topk = topk.max(1);
        let ep_world = ep_world.max(1);
        let num_experts = num_experts.max(1);
        let num_channels = num_channels.max(1);
        let experts_per_rank = experts_per_rank.max(1);
        let capacity_local_routes = capacity_recv.saturating_mul(topk).max(1);
        let needs_alloc = self
            .native_deepep
            .as_ref()
            .map(|scratch| {
                scratch.capacity_tokens < capacity_tokens
                    || scratch.capacity_recv < capacity_recv
                    || scratch.hidden_dim != hidden_dim
                    || scratch.topk != topk
                    || scratch.ep_world != ep_world
                    || scratch.num_experts != num_experts
                    || scratch.num_channels != num_channels
                    || scratch.experts_per_rank != experts_per_rank
            })
            .unwrap_or(true);
        if needs_alloc {
            self.native_deepep = Some(DeepseekNativeDeepEpRuntimeScratch {
                capacity_tokens,
                capacity_recv,
                hidden_dim,
                topk,
                ep_world,
                num_experts,
                num_channels,
                experts_per_rank,
                topk_idx_i64: unsafe {
                    ctx.stream
                        .alloc_zeros_traced::<i64>(capacity_tokens.saturating_mul(topk))?
                },
                recv_x: unsafe { HiddenStates::uninit(ctx, hidden_dim, capacity_recv)? },
                // Zero-init the dispatch-output metadata the combine indexes
                // with: the dispatch populates only the valid range, leaving an
                // uninitialized tail that the DeepEP combine kernel reads ->
                // garbage index -> illegal memory access. compute-sanitizer
                // (which zero-fills allocations) masked the IMA, proving the
                // root cause is an uninitialized read. Zero send_head /
                // rank_prefix / recv_channel_prefix / channel_prefix_matrix /
                // recv_src_idx so the unused tail is a safe 0.
                recv_src_idx: unsafe { ctx.stream.alloc_zeros_traced::<i32>(capacity_recv)? },
                recv_topk_idx: unsafe {
                    ctx.stream
                        .alloc_zeros_traced::<i64>(capacity_recv.saturating_mul(topk))?
                },
                recv_topk_w: unsafe {
                    ctx.stream
                        .alloc_zeros_traced::<f32>(capacity_recv.saturating_mul(topk))?
                },
                rank_prefix: unsafe {
                    ctx.stream
                        .alloc_zeros_traced::<i32>(ep_world.saturating_mul(ep_world))?
                },
                recv_channel_prefix: unsafe {
                    ctx.stream
                        .alloc_zeros_traced::<i32>(ep_world.saturating_mul(num_channels))?
                },
                send_head: unsafe {
                    ctx.stream
                        .alloc_zeros_traced::<i32>(capacity_tokens.saturating_mul(ep_world))?
                },
                num_tokens_per_rank: unsafe { ctx.stream.alloc_zeros_traced::<i32>(ep_world)? },
                num_tokens_per_expert: unsafe {
                    ctx.stream.alloc_zeros_traced::<i32>(num_experts)?
                },
                is_token_in_rank: unsafe {
                    ctx.stream
                        .alloc_zeros_traced::<u8>(capacity_tokens.saturating_mul(ep_world))?
                },
                channel_prefix_matrix: unsafe {
                    ctx.stream
                        .alloc_zeros_traced::<i32>(ep_world.saturating_mul(num_channels))?
                },
                combined_x: unsafe { HiddenStates::uninit(ctx, hidden_dim, capacity_tokens)? },
                combined_topk_w: unsafe {
                    ctx.stream
                        .alloc_zeros_traced::<f32>(capacity_tokens.saturating_mul(topk))?
                },
                expert_out: unsafe { HiddenStates::uninit(ctx, hidden_dim, capacity_recv)? },
                recv_topk_idx_i32: unsafe {
                    ctx.stream
                        .alloc_zeros_traced::<i32>(capacity_recv.saturating_mul(topk))?
                },
                local_counts: unsafe { ctx.stream.alloc_zeros_traced::<i32>(experts_per_rank)? },
                local_offsets: unsafe { ctx.stream.alloc_zeros_traced::<i32>(experts_per_rank)? },
                local_cursors: unsafe { ctx.stream.alloc_zeros_traced::<i32>(experts_per_rank)? },
                packed_x: unsafe { HiddenStates::uninit(ctx, hidden_dim, capacity_local_routes)? },
                packed_token: unsafe {
                    ctx.stream
                        .alloc_zeros_traced::<i32>(capacity_local_routes)?
                },
                packed_weight: unsafe {
                    ctx.stream
                        .alloc_zeros_traced::<f32>(capacity_local_routes)?
                },
                grouped: None,
            });
        }
        Ok(self
            .native_deepep
            .as_mut()
            .expect("DeepSeek V4 native-deepep scratch allocated"))
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn ensure_route_logits_scratch<'a>(
    slot: &'a mut Option<DeepseekRouteLogitsRuntimeScratch>,
    ctx: &DeviceContext,
    n_experts: usize,
    capacity_tokens: usize,
) -> Result<&'a mut DeepseekRouteLogitsRuntimeScratch> {
    let capacity_tokens = capacity_tokens.max(1);
    let needs_alloc = slot
        .as_ref()
        .map(|scratch| scratch.capacity_tokens < capacity_tokens || scratch.n_experts != n_experts)
        .unwrap_or(true);
    if needs_alloc {
        *slot = Some(DeepseekRouteLogitsRuntimeScratch {
            capacity_tokens,
            n_experts,
            logits: unsafe { HiddenStates::uninit(ctx, n_experts, capacity_tokens)? },
        });
    }
    Ok(slot
        .as_mut()
        .expect("DeepSeek V4 route logits scratch allocated"))
}

#[cfg(feature = "cuda")]
pub(crate) fn ensure_send_route_scratch<'a>(
    slot: &'a mut Option<DeepseekSendRouteRuntimeScratch>,
    ctx: &DeviceContext,
    capacity_routes: usize,
) -> Result<&'a mut DeepseekSendRouteRuntimeScratch> {
    let capacity_routes = capacity_routes.max(1);
    let needs_alloc = slot
        .as_ref()
        .map(|scratch| scratch.capacity_routes < capacity_routes)
        .unwrap_or(true);
    if needs_alloc {
        *slot = Some(DeepseekSendRouteRuntimeScratch {
            capacity_routes,
            send_token: unsafe { ctx.stream.alloc_zeros_traced::<i32>(capacity_routes)? },
            send_route_slot: unsafe { ctx.stream.alloc_zeros_traced::<i32>(capacity_routes)? },
        });
    }
    Ok(slot
        .as_mut()
        .expect("DeepSeek V4 send-route scratch allocated"))
}

#[cfg(feature = "cuda")]
pub(crate) fn ensure_dispatch_payload_scratch<'a>(
    slot: &'a mut Option<DeepseekDispatchPayloadRuntimeScratch>,
    ctx: &DeviceContext,
    capacity_routes: usize,
    stride_elems: usize,
) -> Result<&'a mut DeepseekDispatchPayloadRuntimeScratch> {
    let capacity_routes = capacity_routes.max(1);
    let stride_elems = stride_elems.max(1);
    let needs_alloc = slot
        .as_ref()
        .map(|scratch| {
            scratch.capacity_routes < capacity_routes || scratch.stride_elems != stride_elems
        })
        .unwrap_or(true);
    if needs_alloc {
        let elems = capacity_routes.saturating_mul(stride_elems);
        *slot = Some(DeepseekDispatchPayloadRuntimeScratch {
            capacity_routes,
            stride_elems,
            send_payload: unsafe { ctx.stream.alloc_zeros_traced::<bf16>(elems)? },
            recv_payload: unsafe { ctx.stream.alloc_zeros_traced::<bf16>(elems)? },
        });
    }
    Ok(slot
        .as_mut()
        .expect("DeepSeek V4 dispatch payload scratch allocated"))
}

#[cfg(feature = "cuda")]
pub(crate) fn ensure_recv_route_scratch<'a>(
    slot: &'a mut Option<DeepseekRecvRouteRuntimeScratch>,
    ctx: &DeviceContext,
    hidden_dim: usize,
    capacity_routes: usize,
) -> Result<&'a mut DeepseekRecvRouteRuntimeScratch> {
    let capacity_routes = capacity_routes.max(1);
    let needs_alloc = slot
        .as_ref()
        .map(|scratch| {
            scratch.capacity_routes < capacity_routes || scratch.hidden_dim != hidden_dim
        })
        .unwrap_or(true);
    if needs_alloc {
        *slot = Some(DeepseekRecvRouteRuntimeScratch {
            capacity_routes,
            hidden_dim,
            recv_hidden: unsafe { HiddenStates::uninit(ctx, hidden_dim, capacity_routes)? },
            recv_meta: unsafe { ctx.stream.alloc_zeros_traced::<i32>(capacity_routes * 3)? },
            route_out: unsafe { HiddenStates::uninit(ctx, hidden_dim, capacity_routes)? },
        });
    }
    Ok(slot
        .as_mut()
        .expect("DeepSeek V4 recv-route scratch allocated"))
}

#[cfg(feature = "cuda")]
pub(crate) fn ensure_local_route_scratch<'a>(
    slot: &'a mut Option<DeepseekLocalRouteRuntimeScratch>,
    ctx: &DeviceContext,
    hidden_dim: usize,
    capacity_routes: usize,
) -> Result<&'a mut DeepseekLocalRouteRuntimeScratch> {
    let capacity_routes = capacity_routes.max(1);
    let needs_alloc = slot
        .as_ref()
        .map(|scratch| {
            scratch.capacity_routes < capacity_routes || scratch.hidden_dim != hidden_dim
        })
        .unwrap_or(true);
    if needs_alloc {
        *slot = Some(DeepseekLocalRouteRuntimeScratch {
            capacity_routes,
            hidden_dim,
            expert_hidden: unsafe { HiddenStates::uninit(ctx, hidden_dim, capacity_routes)? },
            expert_weight: unsafe { ctx.stream.alloc_zeros_traced::<f32>(capacity_routes)? },
            expert_route_slot: unsafe { ctx.stream.alloc_zeros_traced::<i32>(capacity_routes)? },
            route_out: unsafe { HiddenStates::uninit(ctx, hidden_dim, capacity_routes)? },
        });
    }
    Ok(slot
        .as_mut()
        .expect("DeepSeek V4 local-route scratch allocated"))
}

#[cfg(feature = "cuda")]
impl DeepseekGroupedExpertRuntimeScratch {
    pub(crate) fn ensure_active_scratch(
        &mut self,
        ctx: &DeviceContext,
        capacity_experts: usize,
    ) -> Result<&mut DeepseekGroupedExpertActiveScratch> {
        let capacity_experts = capacity_experts.max(1);
        let needs_alloc = self
            .active
            .as_ref()
            .map(|scratch| scratch.capacity_experts < capacity_experts)
            .unwrap_or(true);
        if needs_alloc {
            self.active = Some(DeepseekGroupedExpertActiveScratch {
                capacity_experts,
                indices: unsafe { ctx.stream.alloc_zeros_traced::<i32>(capacity_experts)? },
                offsets: unsafe { ctx.stream.alloc_zeros_traced::<i32>(capacity_experts)? },
                counts: unsafe { ctx.stream.alloc_zeros_traced::<i32>(capacity_experts)? },
            });
        }
        Ok(self
            .active
            .as_mut()
            .expect("DeepSeek V4 grouped expert active scratch allocated"))
    }

    pub(crate) fn ensure_deepgemm_scratch(
        &mut self,
        ctx: &DeviceContext,
        capacity_experts: usize,
        capacity_m: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Result<&mut DeepseekDeepGemmExpertRuntimeScratch> {
        let capacity_experts = capacity_experts.max(1);
        let capacity_m = capacity_m.max(1);
        let scale_stride_m = capacity_m
            .div_ceil(4)
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 DeepGEMM scale stride overflows usize"))?;
        let hidden_scale_cols = hidden_dim.div_ceil(128);
        let intermediate_scale_cols = intermediate_dim.div_ceil(128);
        let rows = capacity_experts.checked_mul(capacity_m).ok_or_else(|| {
            anyhow::anyhow!(
                "DeepSeek V4 DeepGEMM scratch row count overflow: experts={} capacity_m={}",
                capacity_experts,
                capacity_m
            )
        })?;
        let input_elems = rows.checked_mul(hidden_dim).ok_or_else(|| {
            anyhow::anyhow!(
                "DeepSeek V4 DeepGEMM input scratch overflow: rows={} hidden_dim={}",
                rows,
                hidden_dim
            )
        })?;
        let hidden_scale_elems = capacity_experts
            .checked_mul(scale_stride_m)
            .and_then(|value| value.checked_mul(hidden_scale_cols))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "DeepSeek V4 DeepGEMM input scale scratch overflow: experts={} stride={} cols={}",
                    capacity_experts,
                    scale_stride_m,
                    hidden_scale_cols
                )
            })?;
        let w13_dim = intermediate_dim.checked_mul(2).ok_or_else(|| {
            anyhow::anyhow!(
                "DeepSeek V4 DeepGEMM w13 scratch width overflow: intermediate_dim={}",
                intermediate_dim
            )
        })?;
        let act_elems = rows.checked_mul(intermediate_dim).ok_or_else(|| {
            anyhow::anyhow!(
                "DeepSeek V4 DeepGEMM activation scratch overflow: rows={} intermediate_dim={}",
                rows,
                intermediate_dim
            )
        })?;
        let intermediate_scale_elems = capacity_experts
            .checked_mul(scale_stride_m)
            .and_then(|value| value.checked_mul(intermediate_scale_cols))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "DeepSeek V4 DeepGEMM activation scale scratch overflow: experts={} stride={} cols={}",
                    capacity_experts,
                    scale_stride_m,
                    intermediate_scale_cols
                )
            })?;
        let needs_alloc = self
            .deepgemm
            .as_ref()
            .map(|scratch| {
                scratch.capacity_experts < capacity_experts
                    || scratch.capacity_m < capacity_m
                    || scratch.hidden_dim != hidden_dim
                    || scratch.intermediate_dim != intermediate_dim
            })
            .unwrap_or(true);
        if needs_alloc {
            self.deepgemm = Some(DeepseekDeepGemmExpertRuntimeScratch {
                capacity_experts,
                capacity_m,
                hidden_dim,
                intermediate_dim,
                scale_stride_m,
                input_fp8: unsafe { ctx.stream.alloc_zeros_traced::<u8>(input_elems)? },
                input_scales: unsafe { ctx.stream.alloc_zeros_traced::<f32>(hidden_scale_elems)? },
                w13_out: unsafe { HiddenStates::uninit(ctx, w13_dim, rows)? },
                act_fp8: unsafe { ctx.stream.alloc_zeros_traced::<u8>(act_elems)? },
                act_scales: unsafe {
                    ctx.stream
                        .alloc_zeros_traced::<f32>(intermediate_scale_elems)?
                },
                out_padded: unsafe { HiddenStates::uninit(ctx, hidden_dim, rows)? },
                out_compact: unsafe { HiddenStates::uninit(ctx, hidden_dim, rows)? },
                masked_m: unsafe { ctx.stream.alloc_zeros_traced::<i32>(capacity_experts)? },
            });
        }
        let scratch = self
            .deepgemm
            .as_mut()
            .expect("DeepSeek V4 DeepGEMM expert scratch allocated");
        scratch.w13_out.seq_len = rows;
        scratch.out_padded.seq_len = rows;
        scratch.out_compact.seq_len = rows;
        Ok(scratch)
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn ensure_mhc_scratch<'a>(
    slot: &'a mut Option<DeepseekMhcRuntimeScratch>,
    ctx: &DeviceContext,
    stream_hidden_dim: usize,
    mix_dim: usize,
    hc_mult: usize,
    capacity_tokens: usize,
) -> Result<&'a mut DeepseekMhcRuntimeScratch> {
    let capacity_tokens = capacity_tokens.max(1);
    let needs_alloc = slot
        .as_ref()
        .map(|scratch| {
            scratch.capacity_tokens < capacity_tokens
                || scratch.stream_hidden_dim != stream_hidden_dim
                || scratch.mix_dim != mix_dim
                || scratch.hc_mult != hc_mult
        })
        .unwrap_or(true);
    if needs_alloc {
        *slot = Some(DeepseekMhcRuntimeScratch {
            capacity_tokens,
            stream_hidden_dim,
            mix_dim,
            hc_mult,
            mixes: unsafe { HiddenStates::uninit(ctx, mix_dim, capacity_tokens)? },
            pre: unsafe {
                ctx.stream
                    .alloc_zeros_traced::<f32>(capacity_tokens * hc_mult)?
            },
            post: unsafe {
                ctx.stream
                    .alloc_zeros_traced::<f32>(capacity_tokens * hc_mult)?
            },
            comb: unsafe {
                ctx.stream
                    .alloc_zeros_traced::<f32>(capacity_tokens * hc_mult * hc_mult)?
            },
        });
    }
    Ok(slot.as_mut().expect("DeepSeek V4 MHC scratch allocated"))
}

#[cfg(feature = "cuda")]
pub(crate) fn ensure_hidden_scratch<'a>(
    slot: &'a mut Option<DeepseekHiddenRuntimeScratch>,
    ctx: &DeviceContext,
    hidden_dim: usize,
    seq_len: usize,
) -> Result<&'a mut HiddenStates> {
    let capacity_tokens = seq_len.max(1);
    let needs_alloc = slot
        .as_ref()
        .map(|scratch| {
            scratch.capacity_tokens < capacity_tokens || scratch.hidden_dim != hidden_dim
        })
        .unwrap_or(true);
    if needs_alloc {
        *slot = Some(DeepseekHiddenRuntimeScratch {
            capacity_tokens,
            hidden_dim,
            hidden: unsafe { HiddenStates::uninit(ctx, hidden_dim, capacity_tokens)? },
        });
    }
    let scratch = slot.as_mut().expect("DeepSeek V4 hidden scratch allocated");
    scratch.hidden.seq_len = seq_len;
    Ok(&mut scratch.hidden)
}

#[cfg(feature = "cuda")]
pub(crate) fn take_hidden_scratch(
    slot: &mut Option<DeepseekHiddenRuntimeScratch>,
    ctx: &DeviceContext,
    hidden_dim: usize,
    seq_len: usize,
) -> Result<DeepseekHiddenRuntimeScratch> {
    let capacity_tokens = seq_len.max(1);
    let scratch = match slot.take() {
        Some(mut scratch)
            if scratch.capacity_tokens >= capacity_tokens && scratch.hidden_dim == hidden_dim =>
        {
            scratch.hidden.seq_len = seq_len;
            scratch
        }
        _ => DeepseekHiddenRuntimeScratch {
            capacity_tokens,
            hidden_dim,
            hidden: unsafe { HiddenStates::uninit(ctx, hidden_dim, capacity_tokens)? },
        },
    };
    Ok(scratch)
}

#[cfg(feature = "cuda")]
pub(crate) fn put_hidden_scratch(
    slot: &mut Option<DeepseekHiddenRuntimeScratch>,
    scratch: DeepseekHiddenRuntimeScratch,
) {
    *slot = Some(scratch);
}

#[cfg(feature = "cuda")]
#[derive(Default)]
pub(crate) struct DeepseekAttentionRuntimeCache {
    #[cfg(test)]
    pub(crate) window: VecDeque<DeepseekKvRow>,
    pub(crate) window_gpu: Option<CudaSlice<bf16>>,
    pub(crate) window_gpu_len: usize,
    pub(crate) c_q: Option<DeepseekHiddenRuntimeScratch>,
    pub(crate) c_q_normed: Option<DeepseekHiddenRuntimeScratch>,
    pub(crate) q_raw: Option<DeepseekHiddenRuntimeScratch>,
    pub(crate) kv_raw: Option<DeepseekHiddenRuntimeScratch>,
    pub(crate) kv_normed: Option<DeepseekHiddenRuntimeScratch>,
    pub(crate) q_prepared: Option<DeepseekHiddenRuntimeScratch>,
    pub(crate) k_prepared: Option<DeepseekHiddenRuntimeScratch>,
    pub(crate) local_attn: Option<DeepseekHiddenRuntimeScratch>,
    pub(crate) output_latent: Option<DeepseekHiddenRuntimeScratch>,
    #[cfg(test)]
    pub(crate) compressed: Option<DeepseekCompressorRuntimeCache>,
    #[cfg(test)]
    pub(crate) indexer: Option<DeepseekCompressorRuntimeCache>,
    pub(crate) compressed_gpu: Option<DeepseekGpuCompressorRuntimeCache>,
    pub(crate) indexer_gpu: Option<DeepseekGpuCompressorRuntimeCache>,

    // ----------------------------------------------------------------
    // Phase D-4 — FlashMLA FP8 sparse-decode KV pool (block-paged).
    //
    // Lazy-allocated only when `ARLE_DSV4_FLASHMLA_DECODE` is enabled
    // AND a decode step on this layer first packs a token. Sized to
    // `(sw_blocks + compressed_blocks) * page_block_size * bytes_per_token`
    // bytes, where the layout mirrors upstream FlashMLA MODEL1 (584
    // B/token, AoS [NoPE 448 | RoPE 128] + block-tail e8m0 scales).
    // See `docs/plans/2026-05-28-dsv4-flashmla-decode-integration.md`
    // Phase D-3' for the byte contract.
    //
    // Two contiguous sub-pools:
    //   blocks [0, sw_blocks)                      → per-token K stream
    //                                                (packed on SW
    //                                                 window update)
    //   blocks [sw_blocks, sw_blocks + comp_blocks) → compressor output
    //                                                (packed on
    //                                                 compressor update)
    //
    // Freed alongside the bf16 buffers at session end (drop-on-reset
    // pattern via `Option::take`).
    //
    // `ARLE_DSV4_SHARED_KV_POOL` OFF (default): this owns the per-(slot, layer)
    // pool, lazy-allocated by `ensure_dsv4_flashmla_fp8_kv_pool`. The bound-view
    // fields below stay zero/unused.
    //
    // `ARLE_DSV4_SHARED_KV_POOL` ON: this stays `None`; the pool lives once in
    // the scheduler-owned `DeepseekBatchDecodeBuffers` decode context, and the
    // bound-view fields below (`fp8_kv_pool_ptr` + `fp8_kv_pool_view_bytes`)
    // carry this (slot, layer)'s sub-range, refreshed at the per-step bind site.
    // The per-row pack/decode logic is byte-identical across both modes — only
    // the base pointer moves from an owned buffer's byte 0 to the slot's
    // sub-range start.
    pub(crate) fp8_kv_pool: Option<CudaSlice<u8>>,
    pub(crate) fp8_kv_pool_bytes: usize,
    /// Shared-pool ON only: device base pointer of this (slot, layer)'s bound
    /// sub-range. `0` means unbound (env knob off, or not yet bound this
    /// session).
    pub(crate) fp8_kv_pool_ptr: u64,
    /// Shared-pool ON only: byte length of the bound sub-range.
    pub(crate) fp8_kv_pool_view_bytes: usize,
    pub(crate) fp8_kv_sw_blocks: usize,
    pub(crate) fp8_kv_comp_blocks: usize,
    pub(crate) fp8_kv_page_block_size: usize,
    pub(crate) fp8_kv_bytes_per_token: usize,
    // Phase D-4 — set true after the prefill→decode SW bootstrap bulk-pack
    // has run for this layer. Used to gate the one-shot bf16-SW-ring →
    // FP8-sub-pool copy that runs the first time a decode step (`token_count
    // == 1`) executes against this cache.
    pub(crate) fp8_kv_sw_bootstrapped: bool,
    // Phase D-4 — tracks how many compressor rows have been packed into the
    // FP8 compressed sub-pool so the per-step hook only packs new rows.
    pub(crate) fp8_kv_comp_packed_rows: usize,
    // Phase D-4 — per-layer i32 scratches used by the strided FP8 KV pack
    // hooks (block_ids + in-block rows). Lazy-allocated on first decode
    // step when the env knob is on.
    //   `fp8_kv_sw_bulk_bids` / `fp8_kv_sw_bulk_rows` — size = sliding_window
    //     (one entry per SW ring slot) for the prefill→decode bootstrap.
    //   `fp8_kv_one_token_scratch` — [1]-element scratches for the per-step
    //     SW pack and (single-row) compressor pack.
    //   `fp8_kv_comp_scratch` — sized to max compressor rows per step
    //     (worst case = prefill compressor batch).
    pub(crate) fp8_kv_sw_bulk_bids: Option<CudaSlice<i32>>,
    pub(crate) fp8_kv_sw_bulk_rows: Option<CudaSlice<i32>>,
    pub(crate) fp8_kv_one_token_scratch: Option<(CudaSlice<i32>, CudaSlice<i32>)>,
    pub(crate) fp8_kv_comp_scratch: Option<(CudaSlice<i32>, CudaSlice<i32>)>,

    // Phase D-4 step 2 — amortized FlashMLA decode scratch arena.
    //
    // Allocated once on first decode step when the FlashMLA decode env
    // knob is on. Lifetime = session; reused every step (no per-step
    // alloc). Sized for worst-case `num_sm_parts` (H20 SM90 → 132 SMs /
    // s_q=1 / (h_q/64=1) = 132 at h_q=64, capped to a 256-headroom max)
    // and worst-case `topk_unified` (sliding_window + index_topk rounded
    // up to 128; e.g. 128+512 = 640).
    //
    // Per upstream `vendor/flashmla/csrc/api/sparse_decode.h:189-194`:
    //   tile_scheduler_metadata: [num_sm_parts, DecodingSchedMetaSize/4]
    //   num_splits:             [b+1] = [2] for b=1
    //   lse_accum:              [num_splits, s_q=1, h_q]
    //   o_accum:                [num_splits, s_q=1, h_q, d_v=512]
    //   indices:                [s_q=1, topk_unified]
    //
    // `DecodingSchedMetaSize == 32 B == 8 × int32` per
    // `vendor/flashmla/csrc/params.h:10-17`. Worst-case `num_splits` is
    // bounded by `num_sm_parts + 1`, so split-axis sizing uses
    // `num_sm_parts_max + 1`.
    //
    // `fm_decode_scratch_num_sm_parts` records the capacity these
    // buffers were sized for; if a future dispatch needs more (config
    // drift) the arena grows in place.
    pub(crate) fm_decode_lse_accum: Option<CudaSlice<f32>>,
    pub(crate) fm_decode_o_accum: Option<CudaSlice<f32>>,
    pub(crate) fm_decode_sched_meta: Option<CudaSlice<i32>>,
    pub(crate) fm_decode_num_splits: Option<CudaSlice<i32>>,
    pub(crate) fm_decode_indices: Option<CudaSlice<i32>>,
    pub(crate) fm_decode_scratch_num_sm_parts: usize,
    pub(crate) fm_decode_scratch_topk_unified: usize,
    pub(crate) fm_decode_scratch_h_q: usize,
}

#[cfg(feature = "cuda")]
impl DeepseekAttentionRuntimeCache {
    fn trim_prefill_scratch(&mut self) {
        self.c_q = None;
        self.c_q_normed = None;
        self.q_raw = None;
        self.kv_raw = None;
        self.kv_normed = None;
        self.q_prepared = None;
        self.k_prepared = None;
        self.local_attn = None;
        self.output_latent = None;
        if let Some(cache) = &mut self.compressed_gpu {
            cache.trim_prefill_scratch();
        }
        if let Some(cache) = &mut self.indexer_gpu {
            cache.trim_prefill_scratch();
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) struct DeepseekKvRow {
    pub(crate) pos: usize,
    pub(crate) values: Vec<f32>,
}

#[cfg(all(test, feature = "cuda"))]
pub(crate) struct DeepseekCompressedRow {
    pub(crate) end_pos: usize,
    pub(crate) values: Vec<f32>,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
pub(crate) struct DeepseekGpuCompressorRuntimeCache {
    pub(crate) kv_raw: Option<DeepseekHiddenRuntimeScratch>,
    pub(crate) score_raw: Option<DeepseekHiddenRuntimeScratch>,
    pub(crate) pending_kv: Option<CudaSlice<bf16>>,
    pub(crate) pending_score: Option<CudaSlice<bf16>>,
    pub(crate) prev_overlap_kv: Option<CudaSlice<bf16>>,
    pub(crate) prev_overlap_score: Option<CudaSlice<bf16>>,
    pub(crate) compressed: Option<CudaSlice<bf16>>,
    pub(crate) pending_len: usize,
    pub(crate) compressed_rows: usize,
    pub(crate) compressed_capacity: usize,
    pub(crate) pending_width: usize,
    pub(crate) head_dim: usize,
}

#[cfg(feature = "cuda")]
impl DeepseekGpuCompressorRuntimeCache {
    fn trim_prefill_scratch(&mut self) {
        self.kv_raw = None;
        self.score_raw = None;
    }
}

#[cfg(all(test, feature = "cuda"))]
#[derive(Default)]
pub(crate) struct DeepseekCompressorRuntimeCache {
    pub(crate) pending_kv: Vec<f32>,
    pub(crate) pending_score: Vec<f32>,
    pub(crate) prev_overlap_kv: Vec<f32>,
    pub(crate) prev_overlap_score: Vec<f32>,
    pub(crate) compressed: Vec<DeepseekCompressedRow>,
}

// SAFETY: identical invariant to `Qwen3State` — every `DeepseekState` is owned
// by exactly one scheduler slot, accessed only from the single inference
// thread that runs `Scheduler::run()`. CUDA resources held inside
// `GenerationStateBase` are not shared across threads.
#[cfg(feature = "cuda")]
unsafe impl Send for DeepseekState {}

#[cfg(feature = "cuda")]
impl GenerationState for DeepseekState {
    fn logits(&self) -> &DeviceVec {
        self.base.logits_or(&self.decode_logits)
    }

    fn reset(&mut self) -> Result<()> {
        self.reference_tokens.clear();
        self.incremental.clear();
        self.base.reset()
    }

    fn reset_for_warmup_clear(&mut self) -> Result<()> {
        self.reference_tokens.clear();
        self.incremental.clear();
        self.base.reset()
    }

    fn truncate_to(&mut self, len: usize) -> Result<()> {
        self.reference_tokens.truncate(len);
        if self.incremental.processed_tokens > len {
            self.incremental.clear();
        }
        self.base.truncate_to(len)
    }

    fn supports_partial_prefix(&self) -> bool {
        false
    }

    fn set_max_seq_len(&mut self, max_seq: usize) {
        self.base.set_max_seq_len(max_seq);
    }

    fn set_kv_dtype(&mut self, dtype: KVCacheDtype) {
        self.base.set_kv_dtype(dtype);
    }

    fn migrate_kv_to_paged(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot: usize,
    ) -> Result<()> {
        self.base.migrate_kv_to_paged(ctx, pool, slot)
    }

    fn migrate_kv_range_to_paged(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot: usize,
        start_pos: usize,
        token_count: usize,
    ) -> Result<()> {
        self.base
            .migrate_kv_range_to_paged(ctx, pool, slot, start_pos, token_count)
    }
}
