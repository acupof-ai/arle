//! `ModelForward` impl for the DeepSeek V4 scaffold.
//!
//! Phase 2A starts with a CUDA-backed, SW-only one-token decode smoke. It is
//! intentionally shape/finite only: real attention, MoE, and parity work remain
//! separate tranches.

#[cfg(feature = "cuda")]
use anyhow::{Result, ensure};
#[cfg(feature = "cuda")]
use rand::{RngExt, rngs::StdRng};

#[cfg(feature = "cuda")]
use super::batch_decode::DeepseekBatchDecodeBuffers;
#[cfg(feature = "cuda")]
use super::prefill::DeepseekPrefillContext;
#[cfg(feature = "cuda")]
use super::state::DeepseekState;
#[cfg(feature = "cuda")]
use super::weights::{
    DeepseekModel, dsv4_flashmla_decode_enabled, dsv4_flashmla_prefill_enabled,
    dsv4_incremental_kv_enabled, dsv4_shared_kv_pool_enabled,
};
#[cfg(feature = "cuda")]
use crate::model::generation_state::GenerationStateBase;
#[cfg(feature = "cuda")]
use crate::model::kv_cache::{KVCacheDtype, KVFormat};
#[cfg(feature = "cuda")]
use crate::model::{
    CudaGraphDecodeSupport, MixedBatchFallbackReason, MixedBatchOutcome, MixedBatchRequest,
    ModelForward, PrefillBatchRequest, prepare_paged_prefill_batch,
};
#[cfg(feature = "cuda")]
use crate::model_arch::ModelArchInfo;
#[cfg(feature = "cuda")]
use crate::model_registry::ModelArch;
#[cfg(feature = "cuda")]
use crate::ops;
#[cfg(feature = "cuda")]
use crate::sampler::SamplingParams;
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceContext, DeviceVec, PagedKVPool};
#[cfg(feature = "cuda")]
use cuda_kernels::tensor::CudaAllocTraceExt;

#[cfg(feature = "cuda")]
impl ModelForward for DeepseekModel {
    type State = DeepseekState;
    type DecodeContext = DeepseekBatchDecodeBuffers;
    type PrefillContext = DeepseekPrefillContext;

    fn create_state(&self) -> Result<Self::State> {
        Ok(DeepseekState {
            base: GenerationStateBase::new(
                self.config.num_hidden_layers,
                self.config.num_key_value_heads,
            ),
            decode_logits: cuda_kernels::prelude::DeviceVec::zeros(
                &self.ctx,
                self.config.vocab_size,
            )?
            .with_label("dsv4_phase2a0_decode_logits"),
            sample_probs: self
                .ctx
                .stream
                .alloc_zeros_traced(self.config.vocab_size)
                .map_err(|e| anyhow::anyhow!("Alloc DeepSeek V4 sample_probs failed: {e}"))?,
            sample_out: self
                .ctx
                .stream
                .alloc_zeros_traced(1)
                .map_err(|e| anyhow::anyhow!("Alloc DeepSeek V4 sample_out failed: {e}"))?,
            reference_tokens: Vec::new(),
            incremental: super::state::DeepseekIncrementalState::default(),
        })
    }

    fn create_decode_context(
        &self,
        max_batch_size: usize,
        max_seq_len: Option<usize>,
        pool: &PagedKVPool,
    ) -> Result<Self::DecodeContext> {
        let mut ctx =
            DeepseekBatchDecodeBuffers::new(&self.ctx, max_batch_size, pool.max_total_pages)?;
        if let Some(head_hc) = self.head_hc.as_ref()
            && self.embed_tokens.is_some()
            && self.norm.is_some()
            && self.lm_head.is_some()
        {
            ctx.ensure_batched_scratch(
                &self.ctx,
                self.config.hidden_size,
                self.config.hidden_size * self.config.hc_mult,
                head_hc.mix_fn.rows,
                self.config.vocab_size,
                1,
            )?;
        }
        // Phase D-4 (shared-pool, `ARLE_DSV4_SHARED_KV_POOL` ON only): allocate
        // the shared persistent FP8 KV pool once, sized for
        // `num_slots × layers × slot_blocks`, when the FlashMLA decode env knob
        // is on and the layer weights are loaded. This replaces the per-state
        // lazy allocation (which OOMed at c≥8) and is accounted in the static
        // budget via `scheduler_runtime_workspace_bytes`. When the knob is OFF
        // (default), `ctx.fp8_kv_pool` stays `None` and decode uses the
        // per-state path, byte-identical to `main`.
        if dsv4_shared_kv_pool_enabled()?
            && dsv4_flashmla_decode_enabled()?
            && self.loaded_layer_count() > 0
        {
            let max_seq_len = max_seq_len.unwrap_or(self.config.max_position_embeddings);
            let (sw_blocks, comp_blocks) = self.dsv4_flashmla_pool_slot_blocks(max_seq_len);
            ctx.ensure_fp8_kv_pool(
                &self.ctx,
                max_batch_size,
                self.loaded_layer_count(),
                sw_blocks + comp_blocks,
            )?;
            ctx.set_fp8_kv_max_seq_len(max_seq_len);
        }
        Ok(ctx)
    }

    fn create_prefill_context(
        &self,
        _max_batch_size: usize,
        _prefill_budget_tokens: usize,
        _pool: &PagedKVPool,
    ) -> Result<Self::PrefillContext> {
        Ok(DeepseekPrefillContext::new())
    }

    fn forward_prefill(&self, tokens: &[u32], state: &mut Self::State) -> Result<()> {
        self.prefill_one(tokens, state)
    }

    fn forward_prefill_batch(
        &self,
        requests: &[PrefillBatchRequest<'_>],
        states: &mut [Self::State],
        paged_kv_pool: Option<&mut PagedKVPool>,
    ) -> Result<()> {
        if let Some(pool) = paged_kv_pool
            && pool.is_active()
            && !prepare_paged_prefill_batch(self.device_context(), requests, pool)?
        {
            return Ok(());
        }
        self.prefill_batch_chunks(requests, states)
    }

    fn prefill_uses_paged_pool(&self) -> bool {
        true
    }

    fn supports_cross_slot_prefix_attach(&self) -> bool {
        false
    }

    fn forward_decode(&self, token: u32, state: &mut Self::State) -> Result<()> {
        self.validate_phase0_sw_decode_scope()?;
        ensure!(
            (token as usize) < self.config.vocab_size,
            "DeepSeek V4 token id {token} exceeds vocab_size {}",
            self.config.vocab_size
        );

        if let Some(logits) = self.compute_reference_logits_after_decode(token, state)? {
            state.decode_logits = logits;
            state.base.prefill_logits = None;
            state.base.kv_cache.advance_seq_len(1);
            return Ok(());
        }

        // Phase 2A.1 uses the loaded top-level tensors for non-zero logits when
        // available. Real contextual attention and shared-expert compute land
        // in later, separately gated tranches.
        if let Some(logits) = self.compute_gpu_logits_after_decode(token, state)? {
            state.decode_logits = logits;
        }
        state.base.prefill_logits = None;
        state.base.kv_cache.advance_seq_len(1);
        Ok(())
    }

    fn forward_decode_batch(
        &self,
        tokens: &[u32],
        states: &mut [Self::State],
        slot_indices: &[usize],
        _paged_kv_pool: Option<&mut PagedKVPool>,
        decode_ctx: &mut Self::DecodeContext,
        _skip_logit_scatter: bool,
    ) -> Result<()> {
        ensure!(
            tokens.len() == slot_indices.len(),
            "DeepSeek V4 decode token/slot mismatch: tokens={} slots={}",
            tokens.len(),
            slot_indices.len()
        );
        if tokens.is_empty() {
            return Ok(());
        }

        // Phase D-4 (shared-pool, `ARLE_DSV4_SHARED_KV_POOL` ON only): bind every
        // active (slot, layer) attention cache to its fixed sub-range in the
        // shared FP8 KV pool BEFORE any decode hook runs. This is the single
        // site that owns both the decode context AND the slot identity, so both
        // the N≥2 batched path and the N==1 per-row fallback below read
        // pre-bound views — no slot/ctx threading through the attention chain,
        // and no bind on the prefill path (prefill never reaches here).
        //
        // No-op when the shared pool is off: `fp8_kv_max_seq_len()` returns
        // `None` (the pool was never allocated), so the loop is skipped and the
        // per-state lazy pool path runs unchanged.
        if let Some(max_seq_len) = decode_ctx.fp8_kv_max_seq_len() {
            let num_layers = self.loaded_layer_count();
            for &slot_idx in slot_indices {
                ensure!(
                    slot_idx < states.len(),
                    "DeepSeek V4 decode slot {slot_idx} out of range for {} states",
                    states.len()
                );
                let state = &mut states[slot_idx];
                state.incremental.ensure_layers(num_layers);
                for layer_idx in 0..num_layers {
                    let layer_cache = state
                        .incremental
                        .layers
                        .get_mut(layer_idx)
                        .expect("incremental cache layer initialized");
                    self.bind_fp8_kv_pool_view(
                        decode_ctx,
                        &mut layer_cache.attention,
                        slot_idx,
                        layer_idx,
                        max_seq_len,
                    )?;
                }
            }
        }

        // TRUE batched decode: process all N decode tokens as ONE forward (the
        // routed-MoE FFN half + NCCL all-reduce amortize over the batch; the
        // per-sequence attention core still loops per row). Eligibility is
        // gated by `try_decode_batch`; on any unsupported config it returns
        // `false` and we fall through to the per-row loop, which stays the
        // correctness reference + fallback and is NEVER deleted.
        if self.try_decode_batch(tokens, states, slot_indices, decode_ctx)? {
            return Ok(());
        }
        for (&token, &slot_idx) in tokens.iter().zip(slot_indices) {
            ensure!(
                slot_idx < states.len(),
                "DeepSeek V4 decode slot {slot_idx} out of range for {} states",
                states.len()
            );
            self.forward_decode(token, &mut states[slot_idx])?;
        }
        Ok(())
    }

    fn forward_mixed_batch(
        &self,
        _batch: MixedBatchRequest<'_>,
        _states: &mut [Self::State],
        _paged_kv_pool: Option<&mut PagedKVPool>,
        _decode_ctx: &mut Self::DecodeContext,
    ) -> Result<MixedBatchOutcome> {
        // No mixed-batch support until the V4 prefill + decode kernels share a
        // single varlen launch path. Mirrors qwen3 default.
        Ok(MixedBatchOutcome::Fallback(
            MixedBatchFallbackReason::UnsupportedModel,
        ))
    }

    fn select_token(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<u32> {
        ensure!(
            !params.has_penalties() && params.min_p <= 0.0,
            "DeepSeek V4 sampler supports greedy and temperature/top_k/top_p sampling; \
             penalties and min_p are not implemented yet"
        );
        let random_val: f32 = rng.random();
        let DeepseekState {
            base,
            decode_logits,
            sample_probs,
            sample_out,
            ..
        } = state;
        let logits = base.logits_or(decode_logits);
        let selected = ops::gpu_sample_into(
            &self.ctx,
            logits,
            sample_probs,
            sample_out,
            params,
            random_val,
        )?;
        log_dsv4_sampler_topk(&self.ctx, logits, selected, random_val)?;
        Ok(selected)
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        // DeepSeek V4 generation stops on EOS; BOS is a valid emitted special
        // token and the CPU reference path intentionally does not stop on it.
        self.config.eos_token_id == Some(token_id)
    }

    fn device_context(&self) -> &DeviceContext {
        &self.ctx
    }

    #[cfg(feature = "nccl")]
    fn ep_nccl(&self) -> Option<std::sync::Arc<crate::distributed::nccl::NcclGroup>> {
        self.layer_communicator.ep_nccl()
    }

    #[cfg(feature = "nccl")]
    fn request_token_sync_nccl(
        &self,
    ) -> Option<std::sync::Arc<crate::distributed::nccl::NcclGroup>> {
        self.layer_communicator.request_token_sync_nccl()
    }

    fn supports_decode_warmup(&self) -> bool {
        self.config.tp.world_size == 1 && self.config.ep.world_size == 1
    }

    fn cuda_graph_decode_support(&self) -> CudaGraphDecodeSupport {
        if !self.config.enable_cuda_graph {
            return CudaGraphDecodeSupport::unsupported(
                "CUDA Graph disabled by runtime configuration",
            );
        }
        if super::trace::dsv4_operator_trace_enabled() {
            return CudaGraphDecodeSupport::unsupported(
                "DSv4 operator trace synchronizes CUDA streams for phase timing; disable \
                 ARLE_DSV4_OPERATOR_TRACE / ARLE_DSV4_TRACE_LAYER before graph capture",
            );
        }
        if self.config.tp.world_size > 1 || self.config.ep.world_size > 1 {
            return CudaGraphDecodeSupport::unsupported(
                "DSv4 TP/EP path still launches NCCL collectives and replicated-token \
                 all-reduce/transport outside a graph-safe collective contract",
            );
        }
        CudaGraphDecodeSupport::unsupported(
            "DSv4 decode is not graph-safe yet: attention/compressor metadata still uses \
             host launch params and host-updated cache counters (start_pos, pending_len, \
             compressed_rows, FP8 pack high-water marks); top-level batched scratch now uses \
            stable decode-context buffers",
        )
    }

    fn validate_scheduler_contract(
        &self,
        kv_cache_dtype: KVCacheDtype,
        kv_pool_format: KVFormat,
        cuda_graph_max_bs: usize,
    ) -> Result<()> {
        let profile = self.config.performance_profile()?;
        let graph_support = self.cuda_graph_decode_support();
        let moe_backend = dsv4_moe_backend_label()?;
        let expert_backend = super::mlp::dsv4_expert_backend_label()?;
        let flashmla_prefill = dsv4_flashmla_prefill_enabled()?;
        let flashmla_decode = dsv4_flashmla_decode_enabled()?;
        let shared_kv_pool = dsv4_shared_kv_pool_enabled()?;
        let incremental_kv = dsv4_incremental_kv_enabled()?;
        let fallback_lane = if profile.requires_best_practice() {
            "forbidden"
        } else {
            "allowed-debug-only"
        };

        log::info!(
            "DeepSeek V4 startup contract: profile={} fallback_lane={} tp={}/{} ep={}/{} axes={} coord={:?} kv_cache_dtype={:?} kv_pool_format={:?} cuda_graph_max_bs={} cuda_graph_supported={} cuda_graph_mode={} cuda_graph_reason=\"{}\" moe_backend={} expert_backend={} flashmla_prefill={} flashmla_decode={} shared_kv_pool={} incremental_kv={}",
            profile.as_str(),
            fallback_lane,
            self.config.tp.rank,
            self.config.tp.world_size,
            self.config.ep.rank,
            self.config.ep.world_size,
            self.config.axes.summary(),
            self.config.rank_coord,
            kv_cache_dtype,
            kv_pool_format,
            cuda_graph_max_bs,
            graph_support.supported(),
            graph_support.mode_label(),
            graph_support.reason,
            moe_backend,
            expert_backend,
            flashmla_prefill,
            flashmla_decode,
            shared_kv_pool,
            incremental_kv,
        );

        if !profile.requires_best_practice() {
            return Ok(());
        }

        let mut missing = Vec::new();
        if kv_pool_format != KVFormat::FP8E4M3 {
            missing.push(format!(
                "kv_pool_format must be FP8E4M3 for DSv4 best-practice FP8 KV, got {kv_pool_format:?}"
            ));
        }
        if cuda_graph_max_bs == 0 {
            missing.push("cuda_graph_max_bs must be > 0".to_string());
        }
        if !graph_support.supported() {
            missing.push(format!(
                "CUDA graph decode must be supported, got mode={} reason={}",
                graph_support.mode_label(),
                graph_support.reason
            ));
        }
        if moe_backend != "native-deepep" {
            missing.push(format!(
                "ARLE_DSV4_MOE_BACKEND must be native-deepep for the current ARLE DeepEP target, got {moe_backend}"
            ));
        }
        if expert_backend != "deepgemm" {
            missing.push(format!(
                "ARLE_DSV4_EXPERT_BACKEND must be deepgemm required mode, got {expert_backend}"
            ));
        }
        if !flashmla_prefill {
            missing.push("ARLE_DSV4_FLASHMLA_PREFILL must be enabled".to_string());
        }
        if !flashmla_decode {
            missing.push("ARLE_DSV4_FLASHMLA_DECODE must be enabled".to_string());
        }
        if !shared_kv_pool {
            missing.push(
                "ARLE_DSV4_SHARED_KV_POOL=1 is required for persistent DSv4 FP8 KV".to_string(),
            );
        }
        if !incremental_kv {
            missing.push("ARLE_DSV4_INCREMENTAL_KV must be enabled".to_string());
        }
        missing.push(
            "DSv4 graph-captured SWA/C4/C128 metadata replay is not implemented in this executable route"
                .to_string(),
        );

        anyhow::bail!(
            "DeepSeek V4 profile `{}` requested, but this binary is not on the SGLang best-practice path:\n - {}\nSet ARLE_DSV4_PERFORMANCE_PROFILE=debug-fallback only for correctness/debug runs.",
            profile.as_str(),
            missing.join("\n - ")
        );
    }

    fn supports_prefill_warmup(&self) -> bool {
        false
    }

    fn scheduler_runtime_workspace_bytes(
        &self,
        budget: crate::model::SchedulerRuntimeWorkspaceBudget,
    ) -> usize {
        // Phase D-4 (shared-pool, `ARLE_DSV4_SHARED_KV_POOL` ON only): reserve
        // the shared FP8 KV pool in the static budget so the KV-pool sizing
        // leaves headroom for it. Sized for
        // `num_slots × layers × slot_blocks × 37376 B`, bounded by the served
        // `max_seq_len` (not `max_position_embeddings`). Zero when the shared
        // pool is off (default — per-state path, no static reservation), the
        // FlashMLA decode env knob is off, or no layers are loaded.
        if !dsv4_shared_kv_pool_enabled().unwrap_or(false)
            || !dsv4_flashmla_decode_enabled().unwrap_or(false)
            || self.loaded_layer_count() == 0
        {
            return 0;
        }
        let max_seq_len = budget
            .max_seq_len
            .unwrap_or(self.config.max_position_embeddings);
        let (sw_blocks, comp_blocks) = self.dsv4_flashmla_pool_slot_blocks(max_seq_len);
        DeepseekBatchDecodeBuffers::fp8_kv_pool_bytes(
            budget.max_batch_size,
            self.loaded_layer_count(),
            sw_blocks + comp_blocks,
        )
    }
}

#[cfg(feature = "cuda")]
fn log_dsv4_sampler_topk(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    selected: u32,
    random_val: f32,
) -> Result<()> {
    let Some(k) = std::env::var("ARLE_DSV4_LOG_TOPK")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&value| value > 0)
    else {
        return Ok(());
    };
    let host = ctx.stream.clone_dtoh(&logits.data)?;
    let mut top = Vec::<(u32, f32)>::with_capacity(k);
    let mut selected_logit = None;
    for (idx, value) in host.iter().enumerate() {
        let value = value.to_f32();
        if idx == selected as usize {
            selected_logit = Some(value);
        }
        if !value.is_finite() {
            continue;
        }
        let insert_at = top
            .iter()
            .position(|&(_, existing)| value > existing)
            .unwrap_or(top.len());
        if insert_at < k {
            top.insert(insert_at, (idx as u32, value));
            top.truncate(k);
        }
    }
    let top = top
        .into_iter()
        .map(|(token_id, value)| format!("{token_id}:{value:.4}"))
        .collect::<Vec<_>>()
        .join(",");
    log::info!(
        "DeepSeek V4 sampler selected={} selected_logit={:.4} random={:.6} top{}=[{}]",
        selected,
        selected_logit.unwrap_or(f32::NAN),
        random_val,
        k,
        top
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn dsv4_moe_backend_label() -> Result<&'static str> {
    let Some(raw) = std::env::var("ARLE_DSV4_MOE_BACKEND").ok() else {
        return Ok("allreduce");
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "allreduce" | "all_reduce" | "legacy" | "0" | "false" | "off" => Ok("allreduce"),
        "deepep" | "dispatch" | "dispatch_combine" => Ok("deepep-style"),
        "native-deepep" | "native_deepep" => Ok("native-deepep"),
        other => anyhow::bail!("invalid ARLE_DSV4_MOE_BACKEND value `{other}`"),
    }
}

#[cfg(feature = "cuda")]
impl DeepseekModel {
    fn scheduler_c128_cache_layers(&self) -> usize {
        self.config
            .compress_ratios
            .iter()
            .copied()
            .filter(|&ratio| {
                self.config.spec.attention_mode_for_compress_ratio(ratio)
                    == deepseek_spec::DeepSeekV4AttentionMode::HybridCompressed
            })
            .count()
            .max(1)
    }

    fn scheduler_c128_cache_head_dim(&self) -> usize {
        let c128_ratio = self
            .config
            .compress_ratios
            .iter()
            .copied()
            .filter(|&ratio| {
                self.config.spec.attention_mode_for_compress_ratio(ratio)
                    == deepseek_spec::DeepSeekV4AttentionMode::HybridCompressed
            })
            .min()
            .unwrap_or(128)
            .max(1);
        self.config.head_dim.div_ceil(c128_ratio).max(1)
    }
}

#[cfg(feature = "cuda")]
impl ModelArchInfo for DeepseekModel {
    fn arch_kind(&self) -> ModelArch {
        ModelArch::DeepSeekV4
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn num_hidden_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn num_kv_layers(&self) -> usize {
        self.scheduler_c128_cache_layers()
    }

    fn num_kv_heads(&self) -> usize {
        self.config.num_key_value_heads
    }

    fn num_q_heads(&self) -> usize {
        self.config.num_attention_heads
    }

    fn head_dim(&self) -> usize {
        self.scheduler_c128_cache_head_dim()
    }

    fn kv_cache_bytes_per_token(&self) -> usize {
        // Scheduler-visible DSv4 cache profile:
        // - C128/HCA summaries stay hot in the GPU/host-visible TokenKVPool.
        // - C4/CSA entries are sparse and tiered through the offload path.
        // - SWA uses the 128-token local window and is not charged to the
        //   long-context pool.
        //
        // This keeps admission/page accounting aligned with DSv4's compact
        // cache shape instead of the generic expanded MHA K/V envelope.
        2 * self.scheduler_c128_cache_layers()
            * self.config.num_key_value_heads
            * self.scheduler_c128_cache_head_dim()
            * 2
    }
}
