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
    }

    pub(crate) fn ensure_layers(&mut self, layers: usize) {
        if self.layers.len() < layers {
            self.layers
                .resize_with(layers, DeepseekLayerRuntimeCache::default);
        }
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
}

#[cfg(feature = "cuda")]
pub(crate) struct DeepseekGroupedExpertActiveScratch {
    pub(crate) capacity_experts: usize,
    pub(crate) indices: CudaSlice<i32>,
    pub(crate) offsets: CudaSlice<i32>,
    pub(crate) counts: CudaSlice<i32>,
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
                token_ids: unsafe { ctx.stream.alloc::<u32>(capacity_tokens)? },
                route_indices: unsafe { ctx.stream.alloc::<i32>(capacity_routes)? },
                route_weights: unsafe { ctx.stream.alloc::<f32>(capacity_routes)? },
                send_rank_counts: unsafe { ctx.stream.alloc::<i32>(ep_world)? },
                send_rank_offsets: unsafe { ctx.stream.alloc::<i32>(ep_world)? },
                rank_cursors: unsafe { ctx.stream.alloc::<i32>(ep_world)? },
                send_hidden: unsafe { HiddenStates::uninit(ctx, hidden_dim, capacity_routes)? },
                send_meta: unsafe { ctx.stream.alloc::<i32>(capacity_routes * 3)? },
                all_rank_counts: unsafe { ctx.stream.alloc::<i32>(ep_world * ep_world)? },
                recv_rank_counts: unsafe { ctx.stream.alloc::<i32>(ep_world)? },
                local_counts: unsafe { ctx.stream.alloc::<i32>(experts_per_rank)? },
                local_offsets: unsafe { ctx.stream.alloc::<i32>(experts_per_rank)? },
                local_cursors: unsafe { ctx.stream.alloc::<i32>(experts_per_rank)? },
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
                combine_fp8_send: unsafe { ctx.stream.alloc::<u8>(hidden_dim * capacity_routes)? },
                combine_fp8_recv: unsafe { ctx.stream.alloc::<u8>(hidden_dim * capacity_routes)? },
                combine_scale_send: unsafe { ctx.stream.alloc::<f32>(capacity_routes)? },
                combine_scale_recv: unsafe { ctx.stream.alloc::<f32>(capacity_routes)? },
            });
        }
        Ok(self
            .route_combine
            .as_mut()
            .expect("DeepSeek V4 route combine scratch allocated"))
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
            send_token: unsafe { ctx.stream.alloc::<i32>(capacity_routes)? },
            send_route_slot: unsafe { ctx.stream.alloc::<i32>(capacity_routes)? },
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
            send_payload: unsafe { ctx.stream.alloc::<bf16>(elems)? },
            recv_payload: unsafe { ctx.stream.alloc::<bf16>(elems)? },
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
            recv_meta: unsafe { ctx.stream.alloc::<i32>(capacity_routes * 3)? },
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
            expert_weight: unsafe { ctx.stream.alloc::<f32>(capacity_routes)? },
            expert_route_slot: unsafe { ctx.stream.alloc::<i32>(capacity_routes)? },
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
                indices: unsafe { ctx.stream.alloc::<i32>(capacity_experts)? },
                offsets: unsafe { ctx.stream.alloc::<i32>(capacity_experts)? },
                counts: unsafe { ctx.stream.alloc::<i32>(capacity_experts)? },
            });
        }
        Ok(self
            .active
            .as_mut()
            .expect("DeepSeek V4 grouped expert active scratch allocated"))
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
            pre: unsafe { ctx.stream.alloc::<f32>(capacity_tokens * hc_mult)? },
            post: unsafe { ctx.stream.alloc::<f32>(capacity_tokens * hc_mult)? },
            comb: unsafe {
                ctx.stream
                    .alloc::<f32>(capacity_tokens * hc_mult * hc_mult)?
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
