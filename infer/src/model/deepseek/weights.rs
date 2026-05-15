//! DeepSeek V4 model weights.
//!
//! The runtime target is the local `DeepseekV4ForCausalLM` checkpoint at
//! `infer/models/dsv4-mini-1B-init/`. Infer-side DeepSeek wiring uses
//! [`deepseek_spec::DeepSeekV4Config`] and its HF tensor-name contract only.

use std::path::Path;
#[cfg(all(feature = "cuda", feature = "nccl"))]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::time::Instant;

use anyhow::{Result, bail, ensure};
use half::bf16;
use log::info;
use safetensors::Dtype;

use super::config::DeepseekRuntimeConfig;
#[cfg(feature = "cuda")]
use super::load::load_dsv4_matrix_raw;
#[cfg(feature = "cuda")]
use super::load::{load_dsv4_matrix_raw_sharded, load_dsv4_vec_bf16};
#[cfg(feature = "cuda")]
use super::mla::{DeepseekV4Attention, DeepseekV4Compressor, DeepseekV4Indexer};
#[cfg(feature = "cuda")]
use super::mlp::{
    DeepseekRoutedMoeOutput, DeepseekV4Expert, DeepseekV4MoeBlock,
    dsv4_try_build_grouped_weight_ptrs,
};
#[cfg(feature = "cuda")]
use super::state::{
    DeepseekAttentionRuntimeCache, DeepseekGpuCompressorRuntimeCache, DeepseekHiddenRuntimeScratch,
    DeepseekMhcRuntimeScratch, ensure_hidden_scratch, ensure_mhc_scratch, put_hidden_scratch,
    take_hidden_scratch,
};
#[cfg(all(test, feature = "cuda"))]
use super::state::{DeepseekCompressedRow, DeepseekCompressorRuntimeCache};
#[cfg(feature = "cuda")]
use cuda_kernels::{
    ffi,
    prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates},
};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use deepseek_spec::DeepSeekV4Config;

use crate::deepseek_v4_manifest::{
    DeepseekV4CheckpointManifest, validate_deepseek_v4_checkpoint_manifest,
};
#[cfg(feature = "cuda")]
use crate::deepseek_v4_reference::DeepseekV4ReferenceModel;
#[cfg(feature = "cuda")]
use crate::model::common;
#[cfg(feature = "cuda")]
use crate::model::layer_communicator::LayerCommunicator;
#[cfg(feature = "cuda")]
use crate::ops;
#[cfg(feature = "cuda")]
use crate::tp::TpLoadContext;
#[cfg(feature = "cuda")]
use crate::weight_loader::load_tensor_1d;

/// Hyper-connection tensors used by the V4 layer/head mixers.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // populated once the Phase 2A loader allocates tensors
pub(super) struct DeepseekV4HyperConnection {
    pub(super) base: DeviceVec,
    pub(super) mix_fn: DeviceMatrix,
    pub(super) scale: DeviceVec,
}

/// One DeepSeek V4 transformer layer.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // fields populated by the safetensors loader once kernels land
pub(super) struct DeepseekLayer {
    pub(super) attn_norm: DeviceVec,
    pub(super) hc_attn: DeepseekV4HyperConnection,
    pub(super) attention: DeepseekV4Attention,
    pub(super) ffn_norm: DeviceVec,
    pub(super) hc_ffn: DeepseekV4HyperConnection,
    pub(super) ffn: DeepseekV4MoeBlock,
}

/// DeepSeek V4 model: immutable weights plus runtime config. Mutable per-slot
/// state lives in [`super::state::DeepseekState`].
#[allow(dead_code)] // fields populated by the safetensors loader once kernels land
pub struct DeepseekModel {
    pub(super) config: DeepseekRuntimeConfig,
    #[cfg(feature = "cuda")]
    pub(super) ctx: DeviceContext,
    #[cfg(feature = "cuda")]
    pub(super) embed_tokens: Option<DeviceMatrix>,
    #[cfg(feature = "cuda")]
    pub(super) lm_head: Option<DeviceMatrix>,
    #[cfg(feature = "cuda")]
    pub(super) norm: Option<DeviceVec>,
    #[cfg(feature = "cuda")]
    pub(super) head_hc: Option<DeepseekV4HyperConnection>,
    #[cfg(feature = "cuda")]
    pub(super) layers: Vec<DeepseekLayer>,
    #[cfg(feature = "cuda")]
    pub(super) layer_communicator: LayerCommunicator,
    #[cfg(feature = "cuda")]
    pub(super) reference: Option<DeepseekV4ReferenceModel>,
}

impl DeepseekModel {
    /// Read-only view of the runtime config.
    pub fn config(&self) -> &DeepseekRuntimeConfig {
        &self.config
    }

    /// Read-only view of the underlying DeepSeek V4 spec config.
    pub fn spec(&self) -> &DeepSeekV4Config {
        &self.config.spec
    }

    /// Every layer in the local V4 1B checkpoint has a routed MoE FFN plus
    /// shared expert. The old dense/nano runtime path is no longer the serving
    /// target.
    pub fn is_dense_layer(&self, _idx: usize) -> bool {
        false
    }

    /// Parse the safetensors manifest and verify every tensor required by the
    /// DeepSeek V4 spec is present. This is a cold-path truth gate and performs
    /// no GPU allocation.
    pub fn validate_checkpoint_manifest(
        model_path: impl AsRef<Path>,
        config: &DeepSeekV4Config,
    ) -> Result<DeepseekV4CheckpointManifest> {
        validate_deepseek_v4_checkpoint_manifest(model_path, config)
    }

    pub(super) fn validate_phase0_sw_decode_scope(&self) -> Result<()> {
        let summary = self.config.spec.attention_operator_summary();
        ensure!(
            summary.sliding_window_layers > 0,
            "DeepSeek V4 Phase 0 requires at least one SlidingWindow attention layer; \
             found csa_layers={} hca_layers={}",
            summary.csa_layers,
            summary.hca_layers
        );
        ensure!(
            self.config.vocab_size > 0,
            "DeepSeek V4 Phase 0 requires a non-empty vocab"
        );
        ensure!(
            self.config.ep.num_experts == self.config.n_routed_experts,
            "DeepSeek V4 EP layout has {} experts but config declares {} routed experts",
            self.config.ep.num_experts,
            self.config.n_routed_experts
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl DeepseekModel {
    /// Allocate a model from a spec config without loading weights.
    ///
    /// Phase 0.5 intentionally stops before GPU allocation; return an error
    /// instead of panicking so loader tests can distinguish "parsed V4 config"
    /// from "kernels not implemented yet".
    pub fn from_config(config: DeepseekRuntimeConfig) -> Result<Self> {
        let ctx = DeviceContext::new()?;
        let layer_communicator = Self::layer_communicator_from_config(&ctx, &config)?;
        let model = Self {
            config,
            ctx,
            embed_tokens: None,
            lm_head: None,
            norm: None,
            head_hc: None,
            layers: Vec::new(),
            layer_communicator,
            reference: None,
        };
        model.validate_phase0_sw_decode_scope()?;
        Ok(model)
    }

    fn layer_communicator_from_config(
        ctx: &DeviceContext,
        config: &DeepseekRuntimeConfig,
    ) -> Result<LayerCommunicator> {
        let mut comm = LayerCommunicator::new_with_ep(
            config.tp.rank,
            config.tp.world_size,
            0,
            1,
            0,
            1,
            config.ep.rank,
            config.ep.world_size,
        )?;

        #[cfg(feature = "nccl")]
        {
            use crate::distributed::nccl::{NcclGroup, NcclInitMethod};

            let mut tp_nccl = None;
            if config.tp.world_size > 1 {
                let group = Arc::new(NcclGroup::new_on_stream(
                    config.tp.rank,
                    config.tp.world_size,
                    NcclInitMethod::EnvBootstrap,
                    ctx.stream.clone(),
                )?);
                comm = comm.with_tp_nccl(Arc::clone(&group))?;
                tp_nccl = Some(group);
            }
            if config.ep.world_size > 1 {
                let group = if config.ep.world_size == config.tp.world_size
                    && config.ep.rank == config.tp.rank
                    && tp_nccl.is_some()
                {
                    tp_nccl.expect("checked is_some")
                } else {
                    Arc::new(NcclGroup::new_on_stream(
                        config.ep.rank,
                        config.ep.world_size,
                        NcclInitMethod::EnvBootstrap,
                        ctx.stream.clone(),
                    )?)
                };
                comm = comm.with_ep_nccl(group)?;

                if dsv4_combine_overlap_enabled() {
                    let overlap_group = Arc::new(NcclGroup::new_on_stream(
                        config.ep.rank,
                        config.ep.world_size,
                        dsv4_nccl_env_bootstrap_with_port_offset(1)?,
                        ctx.comm_stream.clone(),
                    )?);
                    comm = comm.with_ep_overlap_nccl(overlap_group)?;
                }
            }
        }

        #[cfg(not(feature = "nccl"))]
        {
            if config.tp.world_size > 1 || config.ep.world_size > 1 {
                bail!(
                    "DeepSeek V4 TP/EP world_size > 1 requires building infer with --features nccl"
                );
            }
        }

        Ok(comm)
    }

    /// Load a V4 checkpoint by safetensors path.
    ///
    /// Phase 2A.1 validates config + tensor-name truth, loads the top-level
    /// embedding/final-norm/LM-head tensors, and brings up a CUDA logits smoke.
    /// Full per-layer weight allocation remains deferred until attention/MoE
    /// kernels graduate to numerical parity.
    pub fn from_safetensors(path: &str, config: DeepseekRuntimeConfig) -> Result<Self> {
        let _manifest = Self::validate_checkpoint_manifest(path, &config.spec)?;
        let mut model = Self::from_config(config)?;
        let real_reference = infer_real_reference_enabled()?;
        if real_reference {
            if load_layer_weights_enabled()? {
                let (mmaps, weight_map) = common::load_safetensors(path, false)?;
                let shards = common::deserialize_shards(&mmaps)?;
                model.load_layer_weights(&shards, &weight_map)?;
            }
            model.reference = Some(DeepseekV4ReferenceModel::load(path)?);
            let summary = model.config.spec.attention_operator_summary();
            info!(
                "DeepSeek V4 real-reference logits enabled: skipping top-level CUDA smoke \
                 weights, sliding_window_layers={} csa_layers={} hca_layers={} vocab_size={} \
                 hidden_size={} tp_rank={}/{} ep_rank={}/{} experts_per_rank={}",
                summary.sliding_window_layers,
                summary.csa_layers,
                summary.hca_layers,
                model.config.vocab_size,
                model.config.hidden_size,
                model.config.tp.rank,
                model.config.tp.world_size,
                model.config.ep.rank,
                model.config.ep.world_size,
                model.config.ep.experts_per_rank,
            );
            return Ok(model);
        }

        let (mmaps, weight_map) = common::load_safetensors(path, false)?;
        let shards = common::deserialize_shards(&mmaps)?;
        let names = model.config.spec.tensor_names();
        let vocab_size = model.config.vocab_size;
        let hidden_size = model.config.hidden_size;

        let embed_tokens =
            load_dsv4_matrix_raw(&model.ctx, &shards, &weight_map, names.embed_tokens())?;
        ensure!(
            embed_tokens.rows == vocab_size && embed_tokens.cols == hidden_size,
            "DeepSeek V4 embed.weight shape [{}, {}] does not match vocab_size={} hidden_size={}",
            embed_tokens.rows,
            embed_tokens.cols,
            vocab_size,
            hidden_size
        );
        let lm_head = load_dsv4_matrix_raw(&model.ctx, &shards, &weight_map, names.lm_head())?;
        ensure!(
            lm_head.rows == vocab_size && lm_head.cols == hidden_size,
            "DeepSeek V4 head.weight shape [{}, {}] does not match vocab_size={} hidden_size={}",
            lm_head.rows,
            lm_head.cols,
            vocab_size,
            hidden_size
        );
        let norm = load_tensor_1d(&model.ctx, &shards, &weight_map, names.norm())?;
        ensure!(
            norm.len == hidden_size,
            "DeepSeek V4 norm.weight len {} does not match hidden_size={}",
            norm.len,
            hidden_size
        );

        model.embed_tokens = Some(embed_tokens);
        model.lm_head = Some(lm_head);
        model.norm = Some(norm);
        if load_layer_weights_enabled()? {
            model.load_layer_weights(&shards, &weight_map)?;
        }

        let summary = model.config.spec.attention_operator_summary();
        info!(
            "DeepSeek V4 Phase 2A.1 CUDA top-level logits smoke loaded: sliding_window_layers={} \
             csa_layers={} hca_layers={} vocab_size={} hidden_size={} tp_rank={}/{} ep_rank={}/{} experts_per_rank={} real_reference={}",
            summary.sliding_window_layers,
            summary.csa_layers,
            summary.hca_layers,
            model.config.vocab_size,
            model.config.hidden_size,
            model.config.tp.rank,
            model.config.tp.world_size,
            model.config.ep.rank,
            model.config.ep.world_size,
            model.config.ep.experts_per_rank,
            real_reference,
        );
        Ok(model)
    }

    pub(super) fn compute_top_level_logits(&self, tokens: &[u32]) -> Result<Option<DeviceVec>> {
        let gpu_ffn_layers = dsv4_gpu_ffn_layer_limit()?;
        let gpu_full_layers = dsv4_gpu_full_layer_limit()?;
        self.compute_top_level_logits_with_layer_limits(tokens, gpu_ffn_layers, gpu_full_layers)
    }

    #[allow(dead_code)] // exercised by CUDA unit tests to avoid mutating process env
    fn compute_top_level_logits_with_ffn_layer_limit(
        &self,
        tokens: &[u32],
        gpu_ffn_layers: usize,
    ) -> Result<Option<DeviceVec>> {
        self.compute_top_level_logits_with_layer_limits(tokens, gpu_ffn_layers, 0)
    }

    fn compute_top_level_logits_with_layer_limits(
        &self,
        tokens: &[u32],
        gpu_ffn_layers: usize,
        gpu_full_layers: usize,
    ) -> Result<Option<DeviceVec>> {
        ensure!(
            !tokens.is_empty(),
            "DeepSeek V4 top-level logits require at least one token"
        );
        ensure!(
            gpu_ffn_layers == 0 || gpu_full_layers == 0,
            "DeepSeek V4 GPU FFN-only layers and full layers are mutually exclusive"
        );
        let (Some(embed_tokens), Some(norm), Some(lm_head)) = (
            self.embed_tokens.as_ref(),
            self.norm.as_ref(),
            self.lm_head.as_ref(),
        ) else {
            return Ok(None);
        };
        let embeddings =
            common::get_embeddings_batch(&self.ctx, embed_tokens, tokens, self.config.hidden_size)?;
        let hidden = if let Some(head_hc) = &self.head_hc {
            ensure!(
                gpu_ffn_layers.max(gpu_full_layers) <= self.layers.len(),
                "DeepSeek V4 requested {} GPU layers but only {} layers are loaded",
                gpu_ffn_layers.max(gpu_full_layers),
                self.layers.len()
            );
            ensure!(
                gpu_full_layers == 0 || self.config.tp.world_size == self.config.o_groups,
                "DeepSeek V4 GPU attention currently maps TP ranks to O-LoRA groups; tp_world={} o_groups={}",
                self.config.tp.world_size,
                self.config.o_groups
            );
            ensure!(
                gpu_full_layers == 0 || self.config.tp.rank < self.config.o_groups,
                "DeepSeek V4 GPU attention tp_rank={} out of O-LoRA group range {}",
                self.config.tp.rank,
                self.config.o_groups
            );
            let mut stream = initial_hc_stream_from_embeddings(
                &self.ctx,
                &embeddings,
                self.config.hidden_size,
                self.config.hc_mult,
            )?;
            for layer_idx in 0..gpu_full_layers {
                stream = self.forward_transformer_layer_stream(layer_idx, &stream, tokens)?;
            }
            for layer_idx in 0..gpu_ffn_layers {
                stream = self.forward_ffn_layer_stream(layer_idx, &stream, tokens)?;
            }
            head_hidden_from_stream(
                &self.ctx,
                head_hc,
                &stream,
                tokens.len() - 1,
                self.config.hidden_size,
                self.config.hc_mult,
                self.config.hc_eps,
            )?
        } else {
            ensure!(
                gpu_ffn_layers == 0 && gpu_full_layers == 0,
                "DeepSeek V4 GPU layer path requires loaded HC/layer weights"
            );
            embeddings
        };
        let logits = common::compute_logits_batch(
            &self.ctx,
            &hidden,
            norm,
            lm_head,
            self.config.rms_norm_eps,
            false,
        )?;
        Ok(Some(logits.with_label("dsv4_phase2a1_top_level_logits")))
    }

    fn compute_top_level_logits_incremental(
        &self,
        tokens: &[u32],
        state: &mut super::state::DeepseekState,
        emit_logits: bool,
    ) -> Result<Option<DeviceVec>> {
        ensure!(
            !tokens.is_empty(),
            "DeepSeek V4 incremental logits require at least one token"
        );
        let (Some(embed_tokens), Some(head_hc), Some(norm), Some(lm_head)) = (
            self.embed_tokens.as_ref(),
            self.head_hc.as_ref(),
            self.norm.as_ref(),
            self.lm_head.as_ref(),
        ) else {
            return Ok(None);
        };
        ensure!(
            !self.layers.is_empty(),
            "DeepSeek V4 incremental KV path requires loaded layer weights"
        );

        let start_pos = state.incremental.processed_tokens;
        ensure!(
            start_pos == state.base.kv_cache.len(),
            "DeepSeek V4 incremental state length {} does not match scheduler KV length {}",
            start_pos,
            state.base.kv_cache.len()
        );
        state.incremental.ensure_layers(self.layers.len());

        let embeddings =
            common::get_embeddings_batch(&self.ctx, embed_tokens, tokens, self.config.hidden_size)?;
        let stream_hidden_dim = self.config.hidden_size * self.config.hc_mult;
        let mut stream = take_hidden_scratch(
            &mut state.incremental.stream_recycle,
            &self.ctx,
            stream_hidden_dim,
            tokens.len(),
        )?;
        initial_hc_stream_from_embeddings_into(
            &self.ctx,
            &embeddings,
            self.config.hidden_size,
            self.config.hc_mult,
            &mut stream.hidden,
        )?;
        for layer_idx in 0..self.layers.len() {
            let layer_cache = state
                .incremental
                .layers
                .get_mut(layer_idx)
                .expect("incremental cache layer initialized");
            let mut next_stream = take_hidden_scratch(
                &mut layer_cache.stream_recycle,
                &self.ctx,
                stream_hidden_dim,
                tokens.len(),
            )?;
            self.forward_transformer_layer_stream_incremental_into(
                layer_idx,
                &stream.hidden,
                tokens,
                start_pos,
                layer_cache,
                &mut next_stream.hidden,
            )?;
            put_hidden_scratch(&mut layer_cache.stream_recycle, stream);
            stream = next_stream;
        }
        state.incremental.processed_tokens += tokens.len();

        if !emit_logits {
            put_hidden_scratch(&mut state.incremental.stream_recycle, stream);
            return Ok(None);
        }

        let hidden = head_hidden_from_stream(
            &self.ctx,
            head_hc,
            &stream.hidden,
            tokens.len() - 1,
            self.config.hidden_size,
            self.config.hc_mult,
            self.config.hc_eps,
        )?;
        let logits = common::compute_logits_batch(
            &self.ctx,
            &hidden,
            norm,
            lm_head,
            self.config.rms_norm_eps,
            false,
        )?;
        put_hidden_scratch(&mut state.incremental.stream_recycle, stream);
        Ok(Some(logits.with_label("dsv4_incremental_top_level_logits")))
    }

    fn forward_transformer_layer_stream(
        &self,
        layer_idx: usize,
        stream: &HiddenStates,
        tokens: &[u32],
    ) -> Result<HiddenStates> {
        ensure!(
            tokens.len() == stream.seq_len,
            "DeepSeek V4 full layer token count {} does not match stream seq_len {}",
            tokens.len(),
            stream.seq_len
        );
        ensure!(
            stream.hidden_dim == self.config.hidden_size * self.config.hc_mult,
            "DeepSeek V4 full layer stream dim {} does not match hidden_size {} * hc_mult {}",
            stream.hidden_dim,
            self.config.hidden_size,
            self.config.hc_mult
        );
        let layer = self.layers.get(layer_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "DeepSeek V4 GPU full layer {} out of range for {} loaded layers",
                layer_idx,
                self.layers.len()
            )
        })?;
        let trace = dsv4_trace_begin(&self.ctx)?;
        let mhc = gen_mhc_params(
            &self.ctx,
            &layer.hc_attn,
            stream,
            self.config.hc_mult,
            self.config.hc_eps,
            self.config.hc_sinkhorn_iters,
        )?;
        dsv4_trace_end(&self.ctx, "attn_mhc", layer_idx, stream.seq_len, trace)?;
        let trace = dsv4_trace_begin(&self.ctx)?;
        let attn_in = hc_pre_from_stream(
            &self.ctx,
            stream,
            &mhc.pre,
            self.config.hidden_size,
            self.config.hc_mult,
        )?;
        let mut normed =
            unsafe { HiddenStates::uninit(&self.ctx, self.config.hidden_size, stream.seq_len)? };
        ops::rms_norm_batch_into(
            &self.ctx,
            &attn_in,
            &layer.attn_norm,
            self.config.rms_norm_eps,
            &mut normed,
        );
        dsv4_trace_end(&self.ctx, "attn_pre_norm", layer_idx, stream.seq_len, trace)?;
        let trace = dsv4_trace_begin(&self.ctx)?;
        let attn_out =
            self.forward_sliding_window_attention(layer_idx, &layer.attention, &normed)?;
        dsv4_trace_end(&self.ctx, "attn_total", layer_idx, stream.seq_len, trace)?;
        let trace = dsv4_trace_begin(&self.ctx)?;
        let stream = hc_post_to_stream(
            &self.ctx,
            &attn_out,
            stream,
            &mhc.post,
            &mhc.comb,
            self.config.hidden_size,
            self.config.hc_mult,
        )?;
        dsv4_trace_end(&self.ctx, "attn_post", layer_idx, stream.seq_len, trace)?;
        let trace = dsv4_trace_begin(&self.ctx)?;
        let stream = self.forward_ffn_layer_stream(layer_idx, &stream, tokens)?;
        dsv4_trace_end(&self.ctx, "ffn_total", layer_idx, stream.seq_len, trace)?;
        Ok(stream)
    }

    fn forward_transformer_layer_stream_incremental_into(
        &self,
        layer_idx: usize,
        stream: &HiddenStates,
        tokens: &[u32],
        start_pos: usize,
        cache: &mut super::state::DeepseekLayerRuntimeCache,
        out: &mut HiddenStates,
    ) -> Result<()> {
        ensure!(
            tokens.len() == stream.seq_len,
            "DeepSeek V4 incremental full layer token count {} does not match stream seq_len {}",
            tokens.len(),
            stream.seq_len
        );
        ensure!(
            stream.hidden_dim == self.config.hidden_size * self.config.hc_mult,
            "DeepSeek V4 incremental full layer stream dim {} does not match hidden_size {} * hc_mult {}",
            stream.hidden_dim,
            self.config.hidden_size,
            self.config.hc_mult
        );
        ensure!(
            out.hidden_dim == self.config.hidden_size * self.config.hc_mult
                && out.seq_len == stream.seq_len,
            "DeepSeek V4 incremental full layer output shape mismatch: out={}x{} expected={}x{}",
            out.seq_len,
            out.hidden_dim,
            stream.seq_len,
            self.config.hidden_size * self.config.hc_mult
        );
        let layer = self.layers.get(layer_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "DeepSeek V4 GPU incremental layer {} out of range for {} loaded layers",
                layer_idx,
                self.layers.len()
            )
        })?;
        let trace = dsv4_trace_begin(&self.ctx)?;
        let mhc_scratch = ensure_mhc_scratch(
            &mut cache.attn_mhc,
            &self.ctx,
            stream.hidden_dim,
            layer.hc_attn.mix_fn.rows,
            self.config.hc_mult,
            stream.seq_len,
        )?;
        let mhc = MhcParamsView::Cached(gen_mhc_params_cached(
            &self.ctx,
            &layer.hc_attn,
            stream,
            self.config.hc_mult,
            self.config.hc_eps,
            self.config.hc_sinkhorn_iters,
            mhc_scratch,
        )?);
        dsv4_trace_end(&self.ctx, "attn_mhc", layer_idx, stream.seq_len, trace)?;
        let trace = dsv4_trace_begin(&self.ctx)?;
        let attn_in = ensure_hidden_scratch(
            &mut cache.attn_pre,
            &self.ctx,
            self.config.hidden_size,
            stream.seq_len,
        )?;
        hc_pre_from_stream_into(
            &self.ctx,
            stream,
            mhc.pre(),
            self.config.hidden_size,
            self.config.hc_mult,
            attn_in,
        )?;
        let normed = ensure_hidden_scratch(
            &mut cache.attn_normed,
            &self.ctx,
            self.config.hidden_size,
            stream.seq_len,
        )?;
        ops::rms_norm_batch_into(
            &self.ctx,
            attn_in,
            &layer.attn_norm,
            self.config.rms_norm_eps,
            normed,
        );
        dsv4_trace_end(&self.ctx, "attn_pre_norm", layer_idx, stream.seq_len, trace)?;
        let trace = dsv4_trace_begin(&self.ctx)?;
        let attn_out = self.forward_sliding_window_attention_incremental(
            layer_idx,
            &layer.attention,
            normed,
            start_pos,
            &mut cache.attention,
        )?;
        dsv4_trace_end(&self.ctx, "attn_total", layer_idx, stream.seq_len, trace)?;
        let trace = dsv4_trace_begin(&self.ctx)?;
        let attn_stream = ensure_hidden_scratch(
            &mut cache.attn_post,
            &self.ctx,
            self.config.hidden_size * self.config.hc_mult,
            stream.seq_len,
        )?;
        hc_post_to_stream_into(
            &self.ctx,
            &attn_out,
            stream,
            mhc.post(),
            mhc.comb(),
            self.config.hidden_size,
            self.config.hc_mult,
            attn_stream,
        )?;
        dsv4_trace_end(
            &self.ctx,
            "attn_post",
            layer_idx,
            attn_stream.seq_len,
            trace,
        )?;
        let trace = dsv4_trace_begin(&self.ctx)?;
        let ffn_mhc_scratch = ensure_mhc_scratch(
            &mut cache.ffn_mhc,
            &self.ctx,
            attn_stream.hidden_dim,
            layer.hc_ffn.mix_fn.rows,
            self.config.hc_mult,
            attn_stream.seq_len,
        )?;
        self.forward_ffn_layer_stream_with_scratch_into(
            layer_idx,
            attn_stream,
            tokens,
            Some(&mut cache.moe),
            Some(ffn_mhc_scratch),
            Some(&mut cache.ffn_pre),
            Some(&mut cache.ffn_normed),
            out,
        )?;
        dsv4_trace_end(&self.ctx, "ffn_total", layer_idx, out.seq_len, trace)?;
        Ok(())
    }

    fn forward_sliding_window_attention(
        &self,
        layer_idx: usize,
        attention: &DeepseekV4Attention,
        hidden: &HiddenStates,
    ) -> Result<HiddenStates> {
        let compress_ratio = *self.config.compress_ratios.get(layer_idx).ok_or_else(|| {
            anyhow::anyhow!("DeepSeek V4 layer {layer_idx} missing compress_ratio")
        })?;
        let mode = self
            .config
            .attention_mode_for_compress_ratio(compress_ratio);
        ensure!(
            hidden.hidden_dim == self.config.hidden_size,
            "DeepSeek V4 attention hidden dim {} does not match hidden_size {}",
            hidden.hidden_dim,
            self.config.hidden_size
        );
        let head_dim = self.config.head_dim;
        ensure!(
            head_dim > 0,
            "DeepSeek V4 attention head_dim must be non-zero"
        );
        let local_width = attention.wq_b.rows;
        ensure!(
            local_width.is_multiple_of(head_dim),
            "DeepSeek V4 local q width {} is not divisible by head_dim {}",
            local_width,
            head_dim
        );
        let local_heads = local_width / head_dim;
        ensure!(
            local_heads > 0,
            "DeepSeek V4 attention requires at least one local head"
        );
        ensure!(
            attention.wkv.rows == head_dim,
            "DeepSeek V4 attention wkv rows {} does not match head_dim {}",
            attention.wkv.rows,
            head_dim
        );
        ensure!(
            attention.wo_a.cols == local_width,
            "DeepSeek V4 attention wo_a cols {} does not match local attention width {}",
            attention.wo_a.cols,
            local_width
        );

        let c_q = ops::gemm(&self.ctx, &attention.wq_a, hidden)?;
        let mut c_q_normed =
            unsafe { HiddenStates::uninit(&self.ctx, c_q.hidden_dim, c_q.seq_len)? };
        ops::rms_norm_batch_into(
            &self.ctx,
            &c_q,
            &attention.q_norm,
            self.config.rms_norm_eps,
            &mut c_q_normed,
        );
        let q_raw = ops::gemm(&self.ctx, &attention.wq_b, &c_q_normed)?;
        let kv_raw = ops::gemm(&self.ctx, &attention.wkv, hidden)?;
        let mut kv_normed =
            unsafe { HiddenStates::uninit(&self.ctx, kv_raw.hidden_dim, kv_raw.seq_len)? };
        ops::rms_norm_batch_into(
            &self.ctx,
            &kv_raw,
            &attention.kv_norm,
            self.config.rms_norm_eps,
            &mut kv_normed,
        );

        self.forward_attention_gpu(
            layer_idx,
            attention,
            hidden,
            &c_q_normed,
            &q_raw,
            &kv_normed,
            hidden.seq_len,
            0,
            local_heads,
            local_width,
            head_dim,
            compress_ratio,
            mode,
            None,
        )
    }

    fn forward_sliding_window_attention_incremental(
        &self,
        layer_idx: usize,
        attention: &DeepseekV4Attention,
        hidden: &HiddenStates,
        start_pos: usize,
        cache: &mut DeepseekAttentionRuntimeCache,
    ) -> Result<HiddenStates> {
        let compress_ratio = *self.config.compress_ratios.get(layer_idx).ok_or_else(|| {
            anyhow::anyhow!("DeepSeek V4 layer {layer_idx} missing compress_ratio")
        })?;
        let mode = self
            .config
            .attention_mode_for_compress_ratio(compress_ratio);
        ensure!(
            hidden.hidden_dim == self.config.hidden_size,
            "DeepSeek V4 incremental attention hidden dim {} does not match hidden_size {}",
            hidden.hidden_dim,
            self.config.hidden_size
        );
        let head_dim = self.config.head_dim;
        let local_width = attention.wq_b.rows;
        ensure!(
            local_width.is_multiple_of(head_dim),
            "DeepSeek V4 incremental local q width {} is not divisible by head_dim {}",
            local_width,
            head_dim
        );
        let local_heads = local_width / head_dim;
        ensure!(
            local_heads > 0,
            "DeepSeek V4 incremental attention requires at least one local head"
        );

        let trace = dsv4_trace_begin(&self.ctx)?;
        let mut c_q_scratch = take_hidden_scratch(
            &mut cache.c_q,
            &self.ctx,
            attention.wq_a.rows,
            hidden.seq_len,
        )?;
        let mut c_q_normed_scratch = take_hidden_scratch(
            &mut cache.c_q_normed,
            &self.ctx,
            attention.wq_a.rows,
            hidden.seq_len,
        )?;
        let mut q_raw_scratch =
            take_hidden_scratch(&mut cache.q_raw, &self.ctx, local_width, hidden.seq_len)?;
        let mut kv_raw_scratch =
            take_hidden_scratch(&mut cache.kv_raw, &self.ctx, head_dim, hidden.seq_len)?;
        let mut kv_normed_scratch =
            take_hidden_scratch(&mut cache.kv_normed, &self.ctx, head_dim, hidden.seq_len)?;
        let c_q = &mut c_q_scratch.hidden;
        ops::try_gemm_with_phase_into(
            &self.ctx,
            &attention.wq_a,
            hidden,
            c_q,
            ops::LinearDispatchPhase::Decode,
        )?;
        let c_q_normed = &mut c_q_normed_scratch.hidden;
        ops::rms_norm_batch_into(
            &self.ctx,
            c_q,
            &attention.q_norm,
            self.config.rms_norm_eps,
            c_q_normed,
        );
        let q_raw = &mut q_raw_scratch.hidden;
        ops::try_gemm_with_phase_into(
            &self.ctx,
            &attention.wq_b,
            c_q_normed,
            q_raw,
            ops::LinearDispatchPhase::Decode,
        )?;
        let kv_raw = &mut kv_raw_scratch.hidden;
        ops::try_gemm_with_phase_into(
            &self.ctx,
            &attention.wkv,
            hidden,
            kv_raw,
            ops::LinearDispatchPhase::Decode,
        )?;
        let kv_normed = &mut kv_normed_scratch.hidden;
        ops::rms_norm_batch_into(
            &self.ctx,
            kv_raw,
            &attention.kv_norm,
            self.config.rms_norm_eps,
            kv_normed,
        );
        dsv4_trace_end(&self.ctx, "attn_proj", layer_idx, hidden.seq_len, trace)?;

        let trace = dsv4_trace_begin(&self.ctx)?;
        let result = self.forward_attention_gpu(
            layer_idx,
            attention,
            hidden,
            c_q_normed,
            q_raw,
            kv_normed,
            hidden.seq_len,
            start_pos,
            local_heads,
            local_width,
            head_dim,
            compress_ratio,
            mode,
            Some(&mut *cache),
        );
        put_hidden_scratch(&mut cache.c_q, c_q_scratch);
        put_hidden_scratch(&mut cache.c_q_normed, c_q_normed_scratch);
        put_hidden_scratch(&mut cache.q_raw, q_raw_scratch);
        put_hidden_scratch(&mut cache.kv_raw, kv_raw_scratch);
        put_hidden_scratch(&mut cache.kv_normed, kv_normed_scratch);
        result.and_then(|out| {
            dsv4_trace_end(&self.ctx, "attn_core", layer_idx, hidden.seq_len, trace)?;
            Ok(out)
        })
    }

    fn forward_swa_attention_gpu(
        &self,
        layer_idx: usize,
        attention: &DeepseekV4Attention,
        q_raw: &HiddenStates,
        kv_normed: &HiddenStates,
        token_count: usize,
        start_pos: usize,
        local_heads: usize,
        local_width: usize,
        head_dim: usize,
        mut cache: Option<&mut DeepseekAttentionRuntimeCache>,
    ) -> Result<HiddenStates> {
        ensure!(
            q_raw.hidden_dim == local_width && q_raw.seq_len == token_count,
            "DeepSeek V4 GPU SWA q shape mismatch: got {}x{} expected {}x{}",
            q_raw.hidden_dim,
            q_raw.seq_len,
            local_width,
            token_count
        );
        ensure!(
            kv_normed.hidden_dim == head_dim && kv_normed.seq_len == token_count,
            "DeepSeek V4 GPU SWA kv shape mismatch: got {}x{} expected {}x{}",
            kv_normed.hidden_dim,
            kv_normed.seq_len,
            head_dim,
            token_count
        );
        ensure!(
            self.config.sliding_window > 0,
            "DeepSeek V4 GPU SWA requires non-zero sliding_window"
        );
        ensure!(
            self.config.qk_rope_head_dim <= head_dim,
            "DeepSeek V4 GPU SWA rope dim {} exceeds head_dim {}",
            self.config.qk_rope_head_dim,
            head_dim
        );
        ensure!(
            attention.attn_sink.len >= self.config.tp.rank * local_heads + local_heads,
            "DeepSeek V4 GPU SWA attn_sink len {} cannot cover local heads {} at rank {}",
            attention.attn_sink.len,
            local_heads,
            self.config.tp.rank
        );

        let rope_params = &self.config.rope_parameters;
        let rope_base = self.config.rope_theta;
        let original_seq_len = 0;
        let trace = dsv4_trace_begin(&self.ctx)?;
        let mut q_prepared = unsafe { HiddenStates::uninit(&self.ctx, local_width, token_count)? };
        let mut k_prepared = unsafe { HiddenStates::uninit(&self.ctx, head_dim, token_count)? };
        {
            let (q_raw_ptr, _q_raw_guard) = q_raw.data.device_ptr(&self.ctx.stream);
            let (k_raw_ptr, _k_raw_guard) = kv_normed.data.device_ptr(&self.ctx.stream);
            let (q_out_ptr, _q_out_guard) = q_prepared.data.device_ptr_mut(&self.ctx.stream);
            let (k_out_ptr, _k_out_guard) = k_prepared.data.device_ptr_mut(&self.ctx.stream);
            unsafe {
                ffi::dsv4_prepare_qk_cuda(
                    q_raw_ptr as *const ffi::Half,
                    k_raw_ptr as *const ffi::Half,
                    q_out_ptr as *mut ffi::Half,
                    k_out_ptr as *mut ffi::Half,
                    token_count as i32,
                    local_heads as i32,
                    head_dim as i32,
                    self.config.qk_rope_head_dim as i32,
                    start_pos as i32,
                    self.config.rms_norm_eps,
                    rope_base,
                    original_seq_len,
                    rope_params.factor,
                    rope_params.beta_fast,
                    rope_params.beta_slow,
                    self.ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 GPU SWA q/k prep failed: {err}"))?;
            }
        }
        dsv4_trace_end(
            &self.ctx,
            "attn_swa_prepare_qk",
            layer_idx,
            token_count,
            trace,
        )?;

        let trace = dsv4_trace_begin(&self.ctx)?;
        let cache_len = self.config.sliding_window * head_dim;
        let mut scratch_window;
        let update_window_cache = cache.is_some();
        let window_cache = if let Some(cache) = cache.as_deref_mut() {
            ensure_swa_window_cache(&self.ctx, cache, cache_len)?
        } else {
            scratch_window = self
                .ctx
                .stream
                .alloc_zeros::<bf16>(cache_len)
                .map_err(|err| {
                    anyhow::anyhow!("DeepSeek V4 GPU SWA scratch alloc failed: {err}")
                })?;
            &mut scratch_window
        };
        dsv4_trace_end(
            &self.ctx,
            "attn_swa_window_alloc",
            layer_idx,
            token_count,
            trace,
        )?;

        let trace = dsv4_trace_begin(&self.ctx)?;
        let mut local_attn = unsafe { HiddenStates::uninit(&self.ctx, local_width, token_count)? };
        {
            let (q_ptr, _q_guard) = q_prepared.data.device_ptr(&self.ctx.stream);
            let (k_ptr, _k_guard) = k_prepared.data.device_ptr(&self.ctx.stream);
            let (window_ptr, _window_guard) = window_cache.device_ptr(&self.ctx.stream);
            let (sink_ptr, _sink_guard) = attention.attn_sink.data.device_ptr(&self.ctx.stream);
            let (out_ptr, _out_guard) = local_attn.data.device_ptr_mut(&self.ctx.stream);
            unsafe {
                ffi::dsv4_swa_attention_cuda(
                    q_ptr as *const ffi::Half,
                    k_ptr as *const ffi::Half,
                    window_ptr as *const ffi::Half,
                    sink_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    token_count as i32,
                    local_heads as i32,
                    head_dim as i32,
                    self.config.sliding_window as i32,
                    start_pos as i32,
                    (self.config.tp.rank * local_heads) as i32,
                    1.0 / (head_dim as f32).sqrt(),
                    self.config.qk_rope_head_dim as i32,
                    rope_base,
                    original_seq_len,
                    rope_params.factor,
                    rope_params.beta_fast,
                    rope_params.beta_slow,
                    self.ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 GPU SWA attention failed: {err}"))?;
            }
        }
        dsv4_trace_end(&self.ctx, "attn_swa_kernel", layer_idx, token_count, trace)?;

        if update_window_cache {
            let trace = dsv4_trace_begin(&self.ctx)?;
            let (k_ptr, _k_guard) = k_prepared.data.device_ptr(&self.ctx.stream);
            let (window_ptr, _window_guard) = window_cache.device_ptr_mut(&self.ctx.stream);
            unsafe {
                ffi::dsv4_update_window_cache_cuda(
                    k_ptr as *const ffi::Half,
                    window_ptr as *mut ffi::Half,
                    token_count as i32,
                    start_pos as i32,
                    self.config.sliding_window as i32,
                    head_dim as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 GPU SWA cache update failed: {err}"))?;
            }
            dsv4_trace_end(
                &self.ctx,
                "attn_swa_window_update",
                layer_idx,
                token_count,
                trace,
            )?;
        }

        let trace = dsv4_trace_begin(&self.ctx)?;
        let latent = ops::gemm(&self.ctx, &attention.wo_a, &local_attn)?;
        let mut out = ops::gemm(&self.ctx, &attention.wo_b, &latent)?;
        dsv4_trace_end(
            &self.ctx,
            "attn_swa_output_proj",
            layer_idx,
            token_count,
            trace,
        )?;
        let trace = dsv4_trace_begin(&self.ctx)?;
        self.layer_communicator
            .post_attn_all_reduce_hidden_states(&mut out)?;
        dsv4_trace_end(
            &self.ctx,
            "attn_swa_all_reduce",
            layer_idx,
            token_count,
            trace,
        )?;
        Ok(out)
    }

    fn forward_attention_gpu(
        &self,
        layer_idx: usize,
        attention: &DeepseekV4Attention,
        hidden: &HiddenStates,
        c_q_normed: &HiddenStates,
        q_raw: &HiddenStates,
        kv_normed: &HiddenStates,
        token_count: usize,
        start_pos: usize,
        local_heads: usize,
        local_width: usize,
        head_dim: usize,
        compress_ratio: usize,
        mode: deepseek_spec::DeepSeekV4AttentionMode,
        cache: Option<&mut DeepseekAttentionRuntimeCache>,
    ) -> Result<HiddenStates> {
        if compress_ratio == 0 {
            return self.forward_swa_attention_gpu(
                layer_idx,
                attention,
                q_raw,
                kv_normed,
                token_count,
                start_pos,
                local_heads,
                local_width,
                head_dim,
                cache,
            );
        }
        match cache {
            Some(cache) => self.forward_attention_gpu_cached(
                layer_idx,
                attention,
                hidden,
                c_q_normed,
                q_raw,
                kv_normed,
                token_count,
                start_pos,
                local_heads,
                local_width,
                head_dim,
                compress_ratio,
                mode,
                cache,
            ),
            None => self.forward_attention_gpu_uncached(
                layer_idx,
                attention,
                hidden,
                c_q_normed,
                q_raw,
                kv_normed,
                token_count,
                start_pos,
                local_heads,
                local_width,
                head_dim,
                compress_ratio,
                mode,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attention_gpu_uncached(
        &self,
        layer_idx: usize,
        attention: &DeepseekV4Attention,
        hidden: &HiddenStates,
        c_q_normed: &HiddenStates,
        q_raw: &HiddenStates,
        kv_normed: &HiddenStates,
        token_count: usize,
        start_pos: usize,
        local_heads: usize,
        local_width: usize,
        head_dim: usize,
        compress_ratio: usize,
        mode: deepseek_spec::DeepSeekV4AttentionMode,
    ) -> Result<HiddenStates> {
        let trace = dsv4_trace_begin(&self.ctx)?;
        let compressed = self.compressor_forward_gpu_temp(
            attention.compressor.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "DeepSeek V4 layer {} has compress_ratio {} but no compressor weights",
                    layer_idx,
                    compress_ratio
                )
            })?,
            hidden,
            head_dim,
            compress_ratio,
            compress_ratio < 16,
            start_pos,
            true,
        )?;
        dsv4_trace_end(&self.ctx, "attn_compressor", layer_idx, token_count, trace)?;
        let selected = if matches!(
            mode,
            deepseek_spec::DeepSeekV4AttentionMode::CompressedSparse
        ) {
            let indexer = attention.indexer.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "DeepSeek V4 layer {} has CSA compress_ratio {} but no indexer weights",
                    layer_idx,
                    compress_ratio
                )
            })?;
            let trace = dsv4_trace_begin(&self.ctx)?;
            let index_keys = self.compressor_forward_gpu_temp(
                &indexer.compressor,
                hidden,
                self.config.index_head_dim,
                compress_ratio,
                true,
                start_pos,
                false,
            )?;
            dsv4_trace_end(
                &self.ctx,
                "attn_indexer_compressor",
                layer_idx,
                token_count,
                trace,
            )?;
            Some(self.csa_selected_blocks_gpu(
                layer_idx,
                indexer,
                hidden,
                c_q_normed,
                &index_keys.data,
                index_keys.seq_len,
                start_pos,
                compress_ratio,
            )?)
        } else {
            None
        };
        self.finish_attention_gpu(
            layer_idx,
            attention,
            q_raw,
            kv_normed,
            Some((&compressed.data, compressed.seq_len)),
            selected.as_ref(),
            token_count,
            start_pos,
            local_heads,
            local_width,
            head_dim,
            compress_ratio,
            mode,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_attention_gpu_cached(
        &self,
        layer_idx: usize,
        attention: &DeepseekV4Attention,
        hidden: &HiddenStates,
        c_q_normed: &HiddenStates,
        q_raw: &HiddenStates,
        kv_normed: &HiddenStates,
        token_count: usize,
        start_pos: usize,
        local_heads: usize,
        local_width: usize,
        head_dim: usize,
        compress_ratio: usize,
        mode: deepseek_spec::DeepSeekV4AttentionMode,
        cache: &mut DeepseekAttentionRuntimeCache,
    ) -> Result<HiddenStates> {
        let compressor = attention.compressor.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "DeepSeek V4 layer {} has compress_ratio {} but no compressor weights",
                layer_idx,
                compress_ratio
            )
        })?;
        let trace = dsv4_trace_begin(&self.ctx)?;
        self.update_compressor_gpu_cache(
            compressor,
            hidden,
            head_dim,
            compress_ratio,
            compress_ratio < 16,
            start_pos,
            true,
            self.config.max_position_embeddings.div_ceil(compress_ratio),
            cache
                .compressed_gpu
                .get_or_insert_with(DeepseekGpuCompressorRuntimeCache::default),
        )?;
        dsv4_trace_end(
            &self.ctx,
            "attn_compressor_update",
            layer_idx,
            token_count,
            trace,
        )?;

        let selected = if matches!(
            mode,
            deepseek_spec::DeepSeekV4AttentionMode::CompressedSparse
        ) {
            let indexer = attention.indexer.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "DeepSeek V4 layer {} has CSA compress_ratio {} but no indexer weights",
                    layer_idx,
                    compress_ratio
                )
            })?;
            let trace = dsv4_trace_begin(&self.ctx)?;
            self.update_compressor_gpu_cache(
                &indexer.compressor,
                hidden,
                self.config.index_head_dim,
                compress_ratio,
                true,
                start_pos,
                false,
                self.config.max_position_embeddings.div_ceil(compress_ratio),
                cache
                    .indexer_gpu
                    .get_or_insert_with(DeepseekGpuCompressorRuntimeCache::default),
            )?;
            dsv4_trace_end(
                &self.ctx,
                "attn_indexer_update",
                layer_idx,
                token_count,
                trace,
            )?;
            let index_cache = cache
                .indexer_gpu
                .as_ref()
                .expect("indexer cache initialized");
            let keys = index_cache
                .compressed
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 indexer GPU cache missing rows"))?;
            Some(self.csa_selected_blocks_gpu(
                layer_idx,
                indexer,
                hidden,
                c_q_normed,
                keys,
                index_cache.compressed_rows,
                start_pos,
                compress_ratio,
            )?)
        } else {
            None
        };

        let compressed_rows = cache
            .compressed_gpu
            .as_ref()
            .expect("compressed cache initialized")
            .compressed_rows;
        let compressed_buf = cache
            .compressed_gpu
            .as_mut()
            .expect("compressed cache initialized")
            .compressed
            .take()
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 compressed GPU cache missing rows"))?;
        let result = self.finish_attention_gpu(
            layer_idx,
            attention,
            q_raw,
            kv_normed,
            Some((&compressed_buf, compressed_rows)),
            selected.as_ref(),
            token_count,
            start_pos,
            local_heads,
            local_width,
            head_dim,
            compress_ratio,
            mode,
            Some(cache),
        );
        cache
            .compressed_gpu
            .as_mut()
            .expect("compressed cache initialized")
            .compressed = Some(compressed_buf);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_attention_gpu(
        &self,
        layer_idx: usize,
        attention: &DeepseekV4Attention,
        q_raw: &HiddenStates,
        kv_normed: &HiddenStates,
        compressed: Option<(&CudaSlice<bf16>, usize)>,
        selected: Option<&CudaSlice<i32>>,
        token_count: usize,
        start_pos: usize,
        local_heads: usize,
        local_width: usize,
        head_dim: usize,
        compress_ratio: usize,
        mode: deepseek_spec::DeepSeekV4AttentionMode,
        mut cache: Option<&mut DeepseekAttentionRuntimeCache>,
    ) -> Result<HiddenStates> {
        ensure!(
            q_raw.hidden_dim == local_width && q_raw.seq_len == token_count,
            "DeepSeek V4 GPU attention q shape mismatch: got {}x{} expected {}x{}",
            q_raw.hidden_dim,
            q_raw.seq_len,
            local_width,
            token_count
        );
        ensure!(
            kv_normed.hidden_dim == head_dim && kv_normed.seq_len == token_count,
            "DeepSeek V4 GPU attention kv shape mismatch: got {}x{} expected {}x{}",
            kv_normed.hidden_dim,
            kv_normed.seq_len,
            head_dim,
            token_count
        );
        let rope_params = &self.config.rope_parameters;
        let (rope_base, original_seq_len) = if compress_ratio > 0 {
            (
                self.config.compress_rope_theta,
                rope_params.original_max_position_embeddings,
            )
        } else {
            (self.config.rope_theta, 0)
        };
        let trace = dsv4_trace_begin(&self.ctx)?;
        let reuse_decode_scratch = token_count == 1;
        let mut q_prepared_scratch = if reuse_decode_scratch {
            if let Some(cache) = cache.as_deref_mut() {
                Some(take_hidden_scratch(
                    &mut cache.q_prepared,
                    &self.ctx,
                    local_width,
                    token_count,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        let mut k_prepared_scratch = if reuse_decode_scratch {
            if let Some(cache) = cache.as_deref_mut() {
                Some(take_hidden_scratch(
                    &mut cache.k_prepared,
                    &self.ctx,
                    head_dim,
                    token_count,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        let mut local_attn_scratch = if reuse_decode_scratch {
            if let Some(cache) = cache.as_deref_mut() {
                Some(take_hidden_scratch(
                    &mut cache.local_attn,
                    &self.ctx,
                    local_width,
                    token_count,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        let mut q_prepared_owned;
        let mut k_prepared_owned;
        let q_prepared = if let Some(scratch) = q_prepared_scratch.as_mut() {
            &mut scratch.hidden
        } else {
            q_prepared_owned =
                unsafe { HiddenStates::uninit(&self.ctx, local_width, token_count)? };
            &mut q_prepared_owned
        };
        let k_prepared = if let Some(scratch) = k_prepared_scratch.as_mut() {
            &mut scratch.hidden
        } else {
            k_prepared_owned = unsafe { HiddenStates::uninit(&self.ctx, head_dim, token_count)? };
            &mut k_prepared_owned
        };
        {
            let (q_raw_ptr, _q_raw_guard) = q_raw.data.device_ptr(&self.ctx.stream);
            let (k_raw_ptr, _k_raw_guard) = kv_normed.data.device_ptr(&self.ctx.stream);
            let (q_out_ptr, _q_out_guard) = q_prepared.data.device_ptr_mut(&self.ctx.stream);
            let (k_out_ptr, _k_out_guard) = k_prepared.data.device_ptr_mut(&self.ctx.stream);
            unsafe {
                ffi::dsv4_prepare_qk_cuda(
                    q_raw_ptr as *const ffi::Half,
                    k_raw_ptr as *const ffi::Half,
                    q_out_ptr as *mut ffi::Half,
                    k_out_ptr as *mut ffi::Half,
                    token_count as i32,
                    local_heads as i32,
                    head_dim as i32,
                    self.config.qk_rope_head_dim as i32,
                    start_pos as i32,
                    self.config.rms_norm_eps,
                    rope_base,
                    original_seq_len as i32,
                    rope_params.factor,
                    rope_params.beta_fast,
                    rope_params.beta_slow,
                    self.ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 GPU q/k prep failed: {err}"))?;
            }
        }
        dsv4_trace_end(&self.ctx, "attn_prepare_qk", layer_idx, token_count, trace)?;

        let trace = dsv4_trace_begin(&self.ctx)?;
        let cache_len = self.config.sliding_window * head_dim;
        let mut scratch_window;
        let update_window_cache = cache.is_some();
        let window_cache = if let Some(cache) = cache.as_deref_mut() {
            ensure_swa_window_cache(&self.ctx, cache, cache_len)?
        } else {
            scratch_window = self
                .ctx
                .stream
                .alloc_zeros::<bf16>(cache_len)
                .map_err(|err| {
                    anyhow::anyhow!("DeepSeek V4 GPU attention scratch alloc failed: {err}")
                })?;
            &mut scratch_window
        };
        dsv4_trace_end(
            &self.ctx,
            "attn_window_alloc",
            layer_idx,
            token_count,
            trace,
        )?;

        let trace = dsv4_trace_begin(&self.ctx)?;
        let mut local_attn_owned;
        let local_attn = if let Some(scratch) = local_attn_scratch.as_mut() {
            &mut scratch.hidden
        } else {
            local_attn_owned =
                unsafe { HiddenStates::uninit(&self.ctx, local_width, token_count)? };
            &mut local_attn_owned
        };
        {
            let (q_ptr, _q_guard) = q_prepared.data.device_ptr(&self.ctx.stream);
            let (k_ptr, _k_guard) = k_prepared.data.device_ptr(&self.ctx.stream);
            let (window_ptr, _window_guard) = window_cache.device_ptr(&self.ctx.stream);
            let compressed_guard;
            let (compressed_ptr, compressed_count) = if let Some((compressed, count)) = compressed {
                let (ptr, guard) = compressed.device_ptr(&self.ctx.stream);
                compressed_guard = Some(guard);
                (ptr as *const ffi::Half, count)
            } else {
                compressed_guard = None;
                (std::ptr::null(), 0)
            };
            let selected_guard;
            let selected_ptr = if let Some(selected) = selected {
                let (ptr, guard) = selected.device_ptr(&self.ctx.stream);
                selected_guard = Some(guard);
                ptr as *const i32
            } else {
                selected_guard = None;
                std::ptr::null()
            };
            let (sink_ptr, _sink_guard) = attention.attn_sink.data.device_ptr(&self.ctx.stream);
            let (out_ptr, _out_guard) = local_attn.data.device_ptr_mut(&self.ctx.stream);
            let mode_int = match mode {
                deepseek_spec::DeepSeekV4AttentionMode::SlidingWindow => 0,
                deepseek_spec::DeepSeekV4AttentionMode::CompressedSparse => 1,
                deepseek_spec::DeepSeekV4AttentionMode::HybridCompressed => 2,
            };
            unsafe {
                ffi::dsv4_hybrid_attention_cuda(
                    q_ptr as *const ffi::Half,
                    k_ptr as *const ffi::Half,
                    window_ptr as *const ffi::Half,
                    compressed_ptr,
                    selected_ptr,
                    sink_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    token_count as i32,
                    local_heads as i32,
                    head_dim as i32,
                    self.config.sliding_window as i32,
                    start_pos as i32,
                    (self.config.tp.rank * local_heads) as i32,
                    1.0 / (head_dim as f32).sqrt(),
                    self.config.qk_rope_head_dim as i32,
                    rope_base,
                    original_seq_len as i32,
                    rope_params.factor,
                    rope_params.beta_fast,
                    rope_params.beta_slow,
                    mode_int,
                    compress_ratio as i32,
                    compressed_count as i32,
                    self.config.index_topk as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 GPU attention failed: {err}"))?;
            }
            drop(compressed_guard);
            drop(selected_guard);
        }
        dsv4_trace_end(
            &self.ctx,
            "attn_hybrid_kernel",
            layer_idx,
            token_count,
            trace,
        )?;

        if update_window_cache {
            let trace = dsv4_trace_begin(&self.ctx)?;
            let (k_ptr, _k_guard) = k_prepared.data.device_ptr(&self.ctx.stream);
            let (window_ptr, _window_guard) = window_cache.device_ptr_mut(&self.ctx.stream);
            unsafe {
                ffi::dsv4_update_window_cache_cuda(
                    k_ptr as *const ffi::Half,
                    window_ptr as *mut ffi::Half,
                    token_count as i32,
                    start_pos as i32,
                    self.config.sliding_window as i32,
                    head_dim as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 GPU cache update failed: {err}"))?;
            }
            dsv4_trace_end(
                &self.ctx,
                "attn_window_update",
                layer_idx,
                token_count,
                trace,
            )?;
        }

        let trace = dsv4_trace_begin(&self.ctx)?;
        let mut latent_scratch = if reuse_decode_scratch {
            if let Some(cache) = cache.as_deref_mut() {
                Some(take_hidden_scratch(
                    &mut cache.output_latent,
                    &self.ctx,
                    attention.wo_a.rows,
                    token_count,
                )?)
            } else {
                None
            }
        } else {
            None
        };
        let mut latent_owned;
        let latent = if let Some(scratch) = latent_scratch.as_mut() {
            &mut scratch.hidden
        } else {
            latent_owned =
                unsafe { HiddenStates::uninit(&self.ctx, attention.wo_a.rows, token_count)? };
            &mut latent_owned
        };
        ops::try_gemm_with_phase_into(
            &self.ctx,
            &attention.wo_a,
            &*local_attn,
            latent,
            if token_count > 1 {
                ops::LinearDispatchPhase::Prefill
            } else {
                ops::LinearDispatchPhase::Decode
            },
        )?;
        let mut out = unsafe { HiddenStates::uninit(&self.ctx, attention.wo_b.rows, token_count)? };
        ops::try_gemm_with_phase_into(
            &self.ctx,
            &attention.wo_b,
            &*latent,
            &mut out,
            if token_count > 1 {
                ops::LinearDispatchPhase::Prefill
            } else {
                ops::LinearDispatchPhase::Decode
            },
        )?;
        dsv4_trace_end(&self.ctx, "attn_output_proj", layer_idx, token_count, trace)?;
        if let Some(cache) = cache.as_deref_mut() {
            if let Some(scratch) = q_prepared_scratch.take() {
                put_hidden_scratch(&mut cache.q_prepared, scratch);
            }
            if let Some(scratch) = k_prepared_scratch.take() {
                put_hidden_scratch(&mut cache.k_prepared, scratch);
            }
            if let Some(scratch) = local_attn_scratch.take() {
                put_hidden_scratch(&mut cache.local_attn, scratch);
            }
            if let Some(scratch) = latent_scratch.take() {
                put_hidden_scratch(&mut cache.output_latent, scratch);
            }
        }
        let trace = dsv4_trace_begin(&self.ctx)?;
        self.layer_communicator
            .post_attn_all_reduce_hidden_states(&mut out)?;
        dsv4_trace_end(&self.ctx, "attn_all_reduce", layer_idx, token_count, trace)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn compressor_forward_gpu_temp(
        &self,
        compressor: &DeepseekV4Compressor,
        hidden: &HiddenStates,
        head_dim: usize,
        ratio: usize,
        overlap: bool,
        start_pos: usize,
        apply_rope: bool,
    ) -> Result<HiddenStates> {
        let rows = hidden.seq_len / ratio;
        if rows == 0 {
            return HiddenStates::zeros(&self.ctx, head_dim, 0);
        }
        let width = if overlap { 2 * head_dim } else { head_dim };
        let mut cache = DeepseekGpuCompressorRuntimeCache {
            compressed_capacity: rows,
            pending_width: width,
            head_dim,
            ..Default::default()
        };
        self.update_compressor_gpu_cache(
            compressor, hidden, head_dim, ratio, overlap, start_pos, apply_rope, rows, &mut cache,
        )?;
        let data = cache
            .compressed
            .take()
            .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 temp compressor output missing"))?;
        Ok(HiddenStates {
            data,
            hidden_dim: head_dim,
            seq_len: cache.compressed_rows,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn update_compressor_gpu_cache(
        &self,
        compressor: &DeepseekV4Compressor,
        hidden: &HiddenStates,
        head_dim: usize,
        ratio: usize,
        overlap: bool,
        start_pos: usize,
        apply_rope: bool,
        capacity_rows: usize,
        cache: &mut DeepseekGpuCompressorRuntimeCache,
    ) -> Result<()> {
        ensure!(ratio > 0, "DeepSeek V4 compressor ratio must be non-zero");
        let width = if overlap { 2 * head_dim } else { head_dim };
        ensure!(
            compressor.wkv.rows == width && compressor.wgate.rows == width,
            "DeepSeek V4 GPU compressor rows mismatch: wkv={} wgate={} expected_width={}",
            compressor.wkv.rows,
            compressor.wgate.rows,
            width
        );
        ensure_gpu_compressor_cache(&self.ctx, cache, capacity_rows, ratio, width, head_dim)?;
        let kv_raw = ensure_hidden_scratch(&mut cache.kv_raw, &self.ctx, width, hidden.seq_len)?;
        ops::gemm_into(&self.ctx, &compressor.wkv, hidden, kv_raw);
        let score_raw =
            ensure_hidden_scratch(&mut cache.score_raw, &self.ctx, width, hidden.seq_len)?;
        ops::gemm_into(&self.ctx, &compressor.wgate, hidden, score_raw);
        let completed = (cache.pending_len + hidden.seq_len) / ratio;
        ensure!(
            cache.compressed_rows + completed <= cache.compressed_capacity,
            "DeepSeek V4 GPU compressor capacity exceeded: rows={} completed={} capacity={}",
            cache.compressed_rows,
            completed,
            cache.compressed_capacity
        );
        let rope_params = &self.config.rope_parameters;
        let (rope_dim, rope_base, original_seq_len) = if apply_rope {
            (
                self.config.qk_rope_head_dim,
                self.config.compress_rope_theta,
                rope_params.original_max_position_embeddings,
            )
        } else {
            (0, self.config.compress_rope_theta, 0)
        };
        {
            let (kv_ptr, _kv_guard) = kv_raw.data.device_ptr(&self.ctx.stream);
            let (score_ptr, _score_guard) = score_raw.data.device_ptr(&self.ctx.stream);
            let (ape_ptr, _ape_guard) = compressor.ape.data.device_ptr(&self.ctx.stream);
            let (norm_ptr, _norm_guard) = compressor.norm.data.device_ptr(&self.ctx.stream);
            let (pending_kv_ptr, _pending_kv_guard) = cache
                .pending_kv
                .as_mut()
                .expect("pending kv allocated")
                .device_ptr_mut(&self.ctx.stream);
            let (pending_score_ptr, _pending_score_guard) = cache
                .pending_score
                .as_mut()
                .expect("pending score allocated")
                .device_ptr_mut(&self.ctx.stream);
            let (prev_kv_ptr, _prev_kv_guard) = cache
                .prev_overlap_kv
                .as_mut()
                .expect("prev kv allocated")
                .device_ptr_mut(&self.ctx.stream);
            let (prev_score_ptr, _prev_score_guard) = cache
                .prev_overlap_score
                .as_mut()
                .expect("prev score allocated")
                .device_ptr_mut(&self.ctx.stream);
            let (compressed_ptr, _compressed_guard) = cache
                .compressed
                .as_mut()
                .expect("compressed rows allocated")
                .device_ptr_mut(&self.ctx.stream);
            unsafe {
                ffi::dsv4_compressor_update_cuda(
                    kv_ptr as *const ffi::Half,
                    score_ptr as *const ffi::Half,
                    ape_ptr as *const ffi::Half,
                    norm_ptr as *const ffi::Half,
                    pending_kv_ptr as *mut ffi::Half,
                    pending_score_ptr as *mut ffi::Half,
                    prev_kv_ptr as *mut ffi::Half,
                    prev_score_ptr as *mut ffi::Half,
                    compressed_ptr as *mut ffi::Half,
                    hidden.seq_len as i32,
                    start_pos as i32,
                    cache.pending_len as i32,
                    cache.compressed_rows as i32,
                    head_dim as i32,
                    ratio as i32,
                    width as i32,
                    i32::from(overlap),
                    i32::from(cache.compressed_rows > 0),
                    self.config.rms_norm_eps,
                    rope_dim as i32,
                    rope_base,
                    original_seq_len as i32,
                    rope_params.factor,
                    rope_params.beta_fast,
                    rope_params.beta_slow,
                    self.ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 GPU compressor failed: {err}"))?;
            }
        }
        cache.compressed_rows += completed;
        cache.pending_len = (cache.pending_len + hidden.seq_len) % ratio;
        Ok(())
    }

    fn csa_selected_blocks_gpu(
        &self,
        layer_idx: usize,
        indexer: &DeepseekV4Indexer,
        hidden: &HiddenStates,
        c_q: &HiddenStates,
        keys: &CudaSlice<bf16>,
        key_count: usize,
        start_pos: usize,
        ratio: usize,
    ) -> Result<CudaSlice<i32>> {
        let trace = dsv4_trace_begin(&self.ctx)?;
        let q_i = ops::gemm(&self.ctx, &indexer.wq_b, c_q)?;
        let weights = ops::gemm(&self.ctx, &indexer.weights_proj, hidden)?;
        dsv4_trace_end(
            &self.ctx,
            "attn_csa_project",
            layer_idx,
            hidden.seq_len,
            trace,
        )?;
        ensure!(
            q_i.hidden_dim.is_multiple_of(self.config.index_head_dim),
            "DeepSeek V4 GPU indexer q width {} is not divisible by index_head_dim {}",
            q_i.hidden_dim,
            self.config.index_head_dim
        );
        let local_index_heads = q_i.hidden_dim / self.config.index_head_dim;
        ensure!(
            weights.hidden_dim == local_index_heads,
            "DeepSeek V4 GPU indexer weights width {} does not match local heads {}",
            weights.hidden_dim,
            local_index_heads
        );
        let mut selected = self
            .ctx
            .stream
            .alloc_zeros::<i32>(hidden.seq_len * self.config.index_topk)
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 CSA selected alloc failed: {err}"))?;
        let score_scale = (self.config.index_head_dim as f32).powf(-0.5)
            * (self.config.index_n_heads as f32).powf(-0.5);
        let trace = dsv4_trace_begin(&self.ctx)?;
        {
            let (q_ptr, _q_guard) = q_i.data.device_ptr(&self.ctx.stream);
            let (weights_ptr, _weights_guard) = weights.data.device_ptr(&self.ctx.stream);
            let (keys_ptr, _keys_guard) = keys.device_ptr(&self.ctx.stream);
            let (selected_ptr, _selected_guard) = selected.device_ptr_mut(&self.ctx.stream);
            unsafe {
                ffi::dsv4_csa_select_cuda(
                    q_ptr as *const ffi::Half,
                    weights_ptr as *const ffi::Half,
                    keys_ptr as *const ffi::Half,
                    selected_ptr as *mut i32,
                    hidden.seq_len as i32,
                    q_i.hidden_dim as i32,
                    local_index_heads as i32,
                    self.config.index_head_dim as i32,
                    key_count as i32,
                    ratio as i32,
                    self.config.index_topk as i32,
                    score_scale,
                    start_pos as i32,
                    self.ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 GPU CSA select failed: {err}"))?;
            }
        }
        dsv4_trace_end(
            &self.ctx,
            "attn_csa_select_kernel",
            layer_idx,
            hidden.seq_len,
            trace,
        )?;
        Ok(selected)
    }

    fn forward_ffn_layer_stream(
        &self,
        layer_idx: usize,
        stream: &HiddenStates,
        tokens: &[u32],
    ) -> Result<HiddenStates> {
        self.forward_ffn_layer_stream_with_scratch(
            layer_idx, stream, tokens, None, None, None, None,
        )
    }

    fn forward_ffn_layer_stream_with_scratch(
        &self,
        layer_idx: usize,
        stream: &HiddenStates,
        tokens: &[u32],
        moe_scratch: Option<&mut super::state::DeepseekMoeRuntimeCache>,
        mhc_scratch: Option<&mut DeepseekMhcRuntimeScratch>,
        ffn_pre_scratch: Option<&mut Option<DeepseekHiddenRuntimeScratch>>,
        ffn_normed_scratch: Option<&mut Option<DeepseekHiddenRuntimeScratch>>,
    ) -> Result<HiddenStates> {
        self.forward_ffn_layer_stream_with_scratch_impl(
            layer_idx,
            stream,
            tokens,
            moe_scratch,
            mhc_scratch,
            ffn_pre_scratch,
            ffn_normed_scratch,
            None,
        )?
        .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 FFN owned output missing"))
    }

    fn forward_ffn_layer_stream_with_scratch_into(
        &self,
        layer_idx: usize,
        stream: &HiddenStates,
        tokens: &[u32],
        moe_scratch: Option<&mut super::state::DeepseekMoeRuntimeCache>,
        mhc_scratch: Option<&mut DeepseekMhcRuntimeScratch>,
        ffn_pre_scratch: Option<&mut Option<DeepseekHiddenRuntimeScratch>>,
        ffn_normed_scratch: Option<&mut Option<DeepseekHiddenRuntimeScratch>>,
        out: &mut HiddenStates,
    ) -> Result<()> {
        self.forward_ffn_layer_stream_with_scratch_impl(
            layer_idx,
            stream,
            tokens,
            moe_scratch,
            mhc_scratch,
            ffn_pre_scratch,
            ffn_normed_scratch,
            Some(out),
        )?;
        Ok(())
    }

    fn forward_ffn_layer_stream_with_scratch_impl(
        &self,
        layer_idx: usize,
        stream: &HiddenStates,
        tokens: &[u32],
        mut moe_scratch: Option<&mut super::state::DeepseekMoeRuntimeCache>,
        mhc_scratch: Option<&mut DeepseekMhcRuntimeScratch>,
        ffn_pre_scratch: Option<&mut Option<DeepseekHiddenRuntimeScratch>>,
        ffn_normed_scratch: Option<&mut Option<DeepseekHiddenRuntimeScratch>>,
        post_out: Option<&mut HiddenStates>,
    ) -> Result<Option<HiddenStates>> {
        ensure!(
            tokens.len() == stream.seq_len,
            "DeepSeek V4 FFN layer token count {} does not match stream seq_len {}",
            tokens.len(),
            stream.seq_len
        );
        ensure!(
            stream.hidden_dim == self.config.hidden_size * self.config.hc_mult,
            "DeepSeek V4 FFN layer stream dim {} does not match hidden_size {} * hc_mult {}",
            stream.hidden_dim,
            self.config.hidden_size,
            self.config.hc_mult
        );
        let layer = self.layers.get(layer_idx).ok_or_else(|| {
            anyhow::anyhow!(
                "DeepSeek V4 GPU FFN layer {} out of range for {} loaded layers",
                layer_idx,
                self.layers.len()
            )
        })?;

        let trace = dsv4_trace_begin(&self.ctx)?;
        let mhc = match mhc_scratch {
            Some(scratch) => MhcParamsView::Cached(gen_mhc_params_cached(
                &self.ctx,
                &layer.hc_ffn,
                stream,
                self.config.hc_mult,
                self.config.hc_eps,
                self.config.hc_sinkhorn_iters,
                scratch,
            )?),
            None => MhcParamsView::Owned(gen_mhc_params(
                &self.ctx,
                &layer.hc_ffn,
                stream,
                self.config.hc_mult,
                self.config.hc_eps,
                self.config.hc_sinkhorn_iters,
            )?),
        };
        dsv4_trace_end(&self.ctx, "ffn_mhc", layer_idx, stream.seq_len, trace)?;
        let trace = dsv4_trace_begin(&self.ctx)?;
        let sub_in_owned;
        let sub_in = if let Some(slot) = ffn_pre_scratch {
            let scratch =
                ensure_hidden_scratch(slot, &self.ctx, self.config.hidden_size, stream.seq_len)?;
            hc_pre_from_stream_into(
                &self.ctx,
                stream,
                mhc.pre(),
                self.config.hidden_size,
                self.config.hc_mult,
                scratch,
            )?;
            &*scratch
        } else {
            sub_in_owned = hc_pre_from_stream(
                &self.ctx,
                stream,
                mhc.pre(),
                self.config.hidden_size,
                self.config.hc_mult,
            )?;
            &sub_in_owned
        };
        let mut normed_owned;
        let normed: &HiddenStates = if let Some(slot) = ffn_normed_scratch {
            let scratch =
                ensure_hidden_scratch(slot, &self.ctx, self.config.hidden_size, stream.seq_len)?;
            ops::rms_norm_batch_into(
                &self.ctx,
                sub_in,
                &layer.ffn_norm,
                self.config.rms_norm_eps,
                scratch,
            );
            &*scratch
        } else {
            normed_owned = unsafe {
                HiddenStates::uninit(&self.ctx, self.config.hidden_size, stream.seq_len)?
            };
            ops::rms_norm_batch_into(
                &self.ctx,
                sub_in,
                &layer.ffn_norm,
                self.config.rms_norm_eps,
                &mut normed_owned,
            );
            &normed_owned
        };
        dsv4_trace_end(&self.ctx, "ffn_pre_norm", layer_idx, stream.seq_len, trace)?;
        let routed = if dsv4_moe_deepep_enabled()? && self.config.ep.world_size > 1 {
            #[cfg(feature = "nccl")]
            {
                let trace = dsv4_trace_begin(&self.ctx)?;
                let routed = layer.ffn.forward_deepep_routed_gpu(
                    &self.ctx,
                    &self.layer_communicator,
                    layer_idx,
                    &self.config.spec,
                    &self.config.ep,
                    &normed,
                    tokens,
                    moe_scratch.as_deref_mut(),
                )?;
                dsv4_trace_end(
                    &self.ctx,
                    "ffn_deepep_dispatch_combine",
                    layer_idx,
                    stream.seq_len,
                    trace,
                )?;
                routed
            }
            #[cfg(not(feature = "nccl"))]
            {
                bail!(
                    "DeepSeek V4 ARLE_DSV4_MOE_BACKEND=deepep requires building infer with --features nccl"
                );
            }
        } else {
            let trace = dsv4_trace_begin(&self.ctx)?;
            let mut routed = layer.ffn.forward_local_routed_gpu(
                &self.ctx,
                layer_idx,
                &self.config.spec,
                &self.config.ep,
                &normed,
                tokens,
            )?;
            dsv4_trace_end(
                &self.ctx,
                "ffn_routed_local",
                layer_idx,
                stream.seq_len,
                trace,
            )?;
            let trace = dsv4_trace_begin(&self.ctx)?;
            self.layer_communicator
                .post_moe_expert_all_reduce_hidden_states(&mut routed)?;
            dsv4_trace_end(
                &self.ctx,
                "ffn_all_reduce",
                layer_idx,
                stream.seq_len,
                trace,
            )?;
            DeepseekRoutedMoeOutput {
                hidden: routed,
                ready: None,
            }
        };
        let trace = dsv4_trace_begin(&self.ctx)?;
        let ffn_out = if normed.seq_len == 1 {
            if let Some(scratch) = moe_scratch.as_deref_mut() {
                layer.ffn.add_shared_expert_with_scratch(
                    &self.ctx,
                    &normed,
                    routed.hidden,
                    routed.ready,
                    self.config.swiglu_limit,
                    scratch,
                )?
            } else {
                layer.ffn.add_shared_expert(
                    &self.ctx,
                    &normed,
                    routed.hidden,
                    routed.ready,
                    self.config.swiglu_limit,
                )?
            }
        } else {
            layer.ffn.add_shared_expert(
                &self.ctx,
                &normed,
                routed.hidden,
                routed.ready,
                self.config.swiglu_limit,
            )?
        };
        dsv4_trace_end(&self.ctx, "ffn_shared", layer_idx, stream.seq_len, trace)?;
        let trace = dsv4_trace_begin(&self.ctx)?;
        let owned_out = if let Some(out) = post_out {
            hc_post_to_stream_into(
                &self.ctx,
                &ffn_out,
                stream,
                mhc.post(),
                mhc.comb(),
                self.config.hidden_size,
                self.config.hc_mult,
                out,
            )?;
            None
        } else {
            Some(hc_post_to_stream(
                &self.ctx,
                &ffn_out,
                stream,
                mhc.post(),
                mhc.comb(),
                self.config.hidden_size,
                self.config.hc_mult,
            )?)
        };
        dsv4_trace_end(&self.ctx, "ffn_post", layer_idx, stream.seq_len, trace)?;
        Ok(owned_out)
    }

    pub(super) fn compute_reference_logits_after_prefill(
        &self,
        tokens: &[u32],
        state: &mut super::state::DeepseekState,
    ) -> Result<Option<DeviceVec>> {
        let Some(reference) = self.reference.as_ref() else {
            return Ok(None);
        };
        state.reference_tokens.extend_from_slice(tokens);
        let logits = reference.forward_last_logits(&state.reference_tokens)?;
        Ok(Some(self.reference_logits_to_device(logits)?))
    }

    pub(super) fn compute_gpu_logits_after_prefill(
        &self,
        tokens: &[u32],
        state: &mut super::state::DeepseekState,
    ) -> Result<Option<DeviceVec>> {
        state.reference_tokens.extend_from_slice(tokens);
        if dsv4_incremental_kv_enabled()? {
            return self.compute_top_level_logits_incremental(tokens, state, true);
        }
        if dsv4_gpu_contextual_logits_enabled()? {
            self.compute_top_level_logits(&state.reference_tokens)
        } else {
            self.compute_top_level_logits(&[tokens[tokens.len() - 1]])
        }
    }

    pub(super) fn compute_gpu_incremental_prefill_chunk(
        &self,
        tokens: &[u32],
        state: &mut super::state::DeepseekState,
        emit_logits: bool,
    ) -> Result<Option<DeviceVec>> {
        state.reference_tokens.extend_from_slice(tokens);
        self.compute_top_level_logits_incremental(tokens, state, emit_logits)
    }

    fn load_layer_weights(
        &mut self,
        shards: &[safetensors::SafeTensors],
        weight_map: &std::collections::HashMap<String, usize>,
    ) -> Result<()> {
        if !self.layers.is_empty() {
            return Ok(());
        }
        let mut layers = Vec::with_capacity(self.config.num_hidden_layers);
        self.head_hc = Some(self.load_hyper_connection(
            shards,
            weight_map,
            &self.config.spec.tensor_names().head_hc(),
        )?);
        for layer_idx in 0..self.config.num_hidden_layers {
            let names = self.config.spec.layer_tensor_names(layer_idx);
            layers.push(DeepseekLayer {
                attn_norm: load_dsv4_vec_bf16(&self.ctx, shards, weight_map, &names.attn_norm)?,
                hc_attn: self.load_hyper_connection(shards, weight_map, &names.hc_attn)?,
                attention: self.load_attention(shards, weight_map, &names.attn)?,
                ffn_norm: load_dsv4_vec_bf16(&self.ctx, shards, weight_map, &names.ffn_norm)?,
                hc_ffn: self.load_hyper_connection(shards, weight_map, &names.hc_ffn)?,
                ffn: self.load_moe_block(shards, weight_map, &names.ffn)?,
            });
        }
        info!(
            "DeepSeek V4 loaded GPU-resident layer weights: layers={} local_experts_per_layer={} tp_rank={}/{} ep_rank={}/{}",
            layers.len(),
            self.config.ep.experts_per_rank,
            self.config.tp.rank,
            self.config.tp.world_size,
            self.config.ep.rank,
            self.config.ep.world_size,
        );
        self.layers = layers;
        Ok(())
    }

    fn load_hyper_connection(
        &self,
        shards: &[safetensors::SafeTensors],
        weight_map: &std::collections::HashMap<String, usize>,
        names: &deepseek_spec::DeepSeekV4HyperConnectionTensorNames,
    ) -> Result<DeepseekV4HyperConnection> {
        Ok(DeepseekV4HyperConnection {
            base: load_dsv4_vec_bf16(&self.ctx, shards, weight_map, &names.base)?,
            mix_fn: load_dsv4_matrix_raw(&self.ctx, shards, weight_map, &names.mix_fn)?,
            scale: load_dsv4_vec_bf16(&self.ctx, shards, weight_map, &names.scale)?,
        })
    }

    fn load_attention(
        &self,
        shards: &[safetensors::SafeTensors],
        weight_map: &std::collections::HashMap<String, usize>,
        names: &deepseek_spec::DeepSeekV4AttentionTensorNames,
    ) -> Result<DeepseekV4Attention> {
        Ok(DeepseekV4Attention {
            wq_a: load_dsv4_matrix_raw(&self.ctx, shards, weight_map, &names.wq_a)?,
            q_norm: load_dsv4_vec_bf16(&self.ctx, shards, weight_map, &names.q_norm)?,
            wq_b: self.load_tp_column_matrix(shards, weight_map, &names.wq_b)?,
            wkv: load_dsv4_matrix_raw(&self.ctx, shards, weight_map, &names.wkv)?,
            kv_norm: load_dsv4_vec_bf16(&self.ctx, shards, weight_map, &names.kv_norm)?,
            wo_a: self.load_tp_column_matrix(shards, weight_map, &names.wo_a)?,
            wo_b: self.load_tp_row_matrix(shards, weight_map, &names.wo_b)?,
            attn_sink: load_dsv4_vec_bf16(&self.ctx, shards, weight_map, &names.attn_sink)?,
            compressor: names
                .compressor
                .as_ref()
                .map(|compressor| self.load_compressor(shards, weight_map, compressor))
                .transpose()?,
            indexer: names
                .indexer
                .as_ref()
                .map(|indexer| self.load_indexer(shards, weight_map, indexer))
                .transpose()?,
        })
    }

    fn load_compressor(
        &self,
        shards: &[safetensors::SafeTensors],
        weight_map: &std::collections::HashMap<String, usize>,
        names: &deepseek_spec::DeepSeekV4CompressorTensorNames,
    ) -> Result<DeepseekV4Compressor> {
        Ok(DeepseekV4Compressor {
            wkv: load_dsv4_matrix_raw(&self.ctx, shards, weight_map, &names.wkv)?,
            wgate: load_dsv4_matrix_raw(&self.ctx, shards, weight_map, &names.wgate)?,
            ape: load_dsv4_matrix_raw(&self.ctx, shards, weight_map, &names.ape)?,
            norm: load_dsv4_vec_bf16(&self.ctx, shards, weight_map, &names.norm)?,
        })
    }

    fn load_indexer(
        &self,
        shards: &[safetensors::SafeTensors],
        weight_map: &std::collections::HashMap<String, usize>,
        names: &deepseek_spec::DeepSeekV4IndexerTensorNames,
    ) -> Result<DeepseekV4Indexer> {
        Ok(DeepseekV4Indexer {
            wq_b: load_dsv4_matrix_raw(&self.ctx, shards, weight_map, &names.wq_b)?,
            weights_proj: load_dsv4_matrix_raw(&self.ctx, shards, weight_map, &names.weights_proj)?,
            compressor: self.load_compressor(shards, weight_map, &names.compressor)?,
        })
    }

    fn load_moe_block(
        &self,
        shards: &[safetensors::SafeTensors],
        weight_map: &std::collections::HashMap<String, usize>,
        names: &deepseek_spec::DeepSeekV4MoeTensorNames,
    ) -> Result<DeepseekV4MoeBlock> {
        let mut experts = Vec::with_capacity(self.config.ep.experts_per_rank);
        for expert_idx in self.config.ep.local_expert_range() {
            let expert = names.expert(expert_idx);
            experts.push(self.load_expert(shards, weight_map, &expert)?);
        }
        let grouped_w1_ptrs =
            dsv4_try_build_grouped_weight_ptrs(&self.ctx, &experts, |expert| &expert.w1)?;
        let grouped_w3_ptrs =
            dsv4_try_build_grouped_weight_ptrs(&self.ctx, &experts, |expert| &expert.w3)?;
        let grouped_w2_ptrs =
            dsv4_try_build_grouped_weight_ptrs(&self.ctx, &experts, |expert| &expert.w2)?;
        Ok(DeepseekV4MoeBlock {
            gate_weight: load_dsv4_matrix_raw(&self.ctx, shards, weight_map, &names.gate_weight)?,
            gate_bias: names
                .gate_bias
                .as_deref()
                .map(|name| load_dsv4_vec_bf16(&self.ctx, shards, weight_map, name))
                .transpose()?,
            gate_tid2eid: names
                .gate_tid2eid
                .as_deref()
                .map(|name| self.load_i64_tensor(shards, weight_map, name))
                .transpose()?,
            experts,
            grouped_w1_ptrs,
            grouped_w3_ptrs,
            grouped_w2_ptrs,
            shared_experts: names
                .shared_experts
                .as_ref()
                .map(|shared| self.load_expert(shards, weight_map, shared))
                .transpose()?,
        })
    }

    fn load_expert(
        &self,
        shards: &[safetensors::SafeTensors],
        weight_map: &std::collections::HashMap<String, usize>,
        names: &deepseek_spec::DeepSeekV4ExpertTensorNames,
    ) -> Result<DeepseekV4Expert> {
        Ok(DeepseekV4Expert {
            w1: load_dsv4_matrix_raw(&self.ctx, shards, weight_map, &names.w1)?,
            w2: load_dsv4_matrix_raw(&self.ctx, shards, weight_map, &names.w2)?,
            w3: load_dsv4_matrix_raw(&self.ctx, shards, weight_map, &names.w3)?,
        })
    }

    fn load_tp_column_matrix(
        &self,
        shards: &[safetensors::SafeTensors],
        weight_map: &std::collections::HashMap<String, usize>,
        name: &str,
    ) -> Result<DeviceMatrix> {
        if self.config.tp.is_single() {
            return load_dsv4_matrix_raw(&self.ctx, shards, weight_map, name);
        }
        let rows = self.matrix_rows(shards, weight_map, name)?;
        let tp = TpLoadContext::column(self.config.tp.rank, self.config.tp.world_size, rows)?;
        load_dsv4_matrix_raw_sharded(&self.ctx, shards, weight_map, name, Some(&tp))
    }

    fn load_tp_row_matrix(
        &self,
        shards: &[safetensors::SafeTensors],
        weight_map: &std::collections::HashMap<String, usize>,
        name: &str,
    ) -> Result<DeviceMatrix> {
        if self.config.tp.is_single() {
            return load_dsv4_matrix_raw(&self.ctx, shards, weight_map, name);
        }
        let cols = self.matrix_logical_cols(shards, weight_map, name)?;
        let tp = TpLoadContext::row(self.config.tp.rank, self.config.tp.world_size, cols)?;
        load_dsv4_matrix_raw_sharded(&self.ctx, shards, weight_map, name, Some(&tp))
    }

    fn matrix_rows(
        &self,
        shards: &[safetensors::SafeTensors],
        weight_map: &std::collections::HashMap<String, usize>,
        name: &str,
    ) -> Result<usize> {
        let tensor = deepseek_find_tensor(shards, weight_map, name)?;
        ensure!(
            tensor.shape().len() == 2,
            "{name}: expected 2D tensor, got {:?}",
            tensor.shape()
        );
        Ok(tensor.shape()[0])
    }

    fn matrix_logical_cols(
        &self,
        shards: &[safetensors::SafeTensors],
        weight_map: &std::collections::HashMap<String, usize>,
        name: &str,
    ) -> Result<usize> {
        let tensor = deepseek_find_tensor(shards, weight_map, name)?;
        ensure!(
            tensor.shape().len() == 2,
            "{name}: expected 2D tensor, got {:?}",
            tensor.shape()
        );
        let physical_cols = tensor.shape()[1];
        Ok(if tensor.dtype() == safetensors::Dtype::I8 {
            physical_cols * 2
        } else {
            physical_cols
        })
    }

    fn load_i64_tensor(
        &self,
        shards: &[safetensors::SafeTensors],
        weight_map: &std::collections::HashMap<String, usize>,
        name: &str,
    ) -> Result<cudarc::driver::CudaSlice<i64>> {
        let tensor = deepseek_find_tensor(shards, weight_map, name)?;
        ensure!(
            tensor.dtype() == Dtype::I64,
            "{name}: expected I64 tensor, got {:?}",
            tensor.dtype()
        );
        ensure!(
            tensor
                .data()
                .len()
                .is_multiple_of(std::mem::size_of::<i64>()),
            "{name}: I64 tensor has unaligned byte length {}",
            tensor.data().len()
        );
        let mut host = Vec::with_capacity(tensor.data().len() / std::mem::size_of::<i64>());
        for chunk in tensor.data().chunks_exact(std::mem::size_of::<i64>()) {
            let mut bytes = [0_u8; 8];
            bytes.copy_from_slice(chunk);
            host.push(i64::from_le_bytes(bytes));
        }
        self.ctx
            .stream
            .clone_htod(&host)
            .map_err(|err| anyhow::anyhow!("uploading DeepSeek V4 I64 tensor {name}: {err}"))
    }

    pub(super) fn compute_reference_logits_after_decode(
        &self,
        token: u32,
        state: &mut super::state::DeepseekState,
    ) -> Result<Option<DeviceVec>> {
        let Some(reference) = self.reference.as_ref() else {
            return Ok(None);
        };
        state.reference_tokens.push(token);
        let logits = reference.forward_last_logits(&state.reference_tokens)?;
        Ok(Some(self.reference_logits_to_device(logits)?))
    }

    pub(super) fn compute_gpu_logits_after_decode(
        &self,
        token: u32,
        state: &mut super::state::DeepseekState,
    ) -> Result<Option<DeviceVec>> {
        state.reference_tokens.push(token);
        if dsv4_incremental_kv_enabled()? {
            return self.compute_top_level_logits_incremental(&[token], state, true);
        }
        if dsv4_gpu_contextual_logits_enabled()? {
            self.compute_top_level_logits(&state.reference_tokens)
        } else {
            self.compute_top_level_logits(&[token])
        }
    }

    fn reference_logits_to_device(&self, logits: Vec<f32>) -> Result<DeviceVec> {
        ensure!(
            logits.len() == self.config.vocab_size,
            "DeepSeek V4 reference logits len {} does not match vocab_size {}",
            logits.len(),
            self.config.vocab_size
        );
        let host = logits.into_iter().map(bf16::from_f32).collect::<Vec<_>>();
        DeviceVec::from_host(&self.ctx, &host).map(|v| v.with_label("dsv4_real_reference_logits"))
    }
}

#[cfg(feature = "cuda")]
fn initial_hc_stream_from_embeddings(
    ctx: &DeviceContext,
    embeddings: &HiddenStates,
    hidden_size: usize,
    hc_mult: usize,
) -> Result<HiddenStates> {
    let stream_hidden = hidden_size * hc_mult;
    let mut stream = unsafe { HiddenStates::uninit(ctx, stream_hidden, embeddings.seq_len)? };
    initial_hc_stream_from_embeddings_into(ctx, embeddings, hidden_size, hc_mult, &mut stream)?;
    Ok(stream)
}

#[cfg(feature = "cuda")]
fn initial_hc_stream_from_embeddings_into(
    ctx: &DeviceContext,
    embeddings: &HiddenStates,
    hidden_size: usize,
    hc_mult: usize,
    stream: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        embeddings.hidden_dim == hidden_size,
        "DeepSeek V4 embedding hidden dim {} does not match hidden_size {}",
        embeddings.hidden_dim,
        hidden_size
    );
    ensure!(hc_mult > 0, "DeepSeek V4 hc_mult must be non-zero");
    let stream_hidden = hidden_size * hc_mult;
    ensure!(
        stream.hidden_dim == stream_hidden && stream.seq_len == embeddings.seq_len,
        "DeepSeek V4 initial HC stream output shape mismatch: out={}x{} expected={}x{}",
        stream.seq_len,
        stream.hidden_dim,
        embeddings.seq_len,
        stream_hidden
    );
    {
        let (emb_ptr, _emb_guard) = embeddings.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = stream.data.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::dsv4_mhc_expand_cuda(
                emb_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                embeddings.seq_len as i32,
                hidden_size as i32,
                hc_mult as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 initial HC expand CUDA failed: {err}"))?;
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
fn hidden_states_from_f32(
    ctx: &DeviceContext,
    values: &[f32],
    hidden_dim: usize,
    seq_len: usize,
) -> Result<HiddenStates> {
    ensure!(
        values.len() == hidden_dim * seq_len,
        "DeepSeek V4 host hidden state len {} does not match hidden_dim {} * seq_len {}",
        values.len(),
        hidden_dim,
        seq_len
    );
    Ok(HiddenStates {
        data: ctx
            .stream
            .clone_htod(
                &values
                    .iter()
                    .map(|&value| bf16::from_f32(value))
                    .collect::<Vec<_>>(),
            )
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 host hidden H2D copy: {err}"))?,
        hidden_dim,
        seq_len,
    })
}

#[cfg(feature = "cuda")]
fn ensure_swa_window_cache<'a>(
    ctx: &DeviceContext,
    cache: &'a mut DeepseekAttentionRuntimeCache,
    len: usize,
) -> Result<&'a mut CudaSlice<bf16>> {
    if cache.window_gpu_len != len || cache.window_gpu.is_none() {
        cache.window_gpu =
            Some(ctx.stream.alloc_zeros::<bf16>(len).map_err(|err| {
                anyhow::anyhow!("DeepSeek V4 SWA window cache alloc failed: {err}")
            })?);
        cache.window_gpu_len = len;
    }
    cache
        .window_gpu
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 SWA window cache allocation missing"))
}

#[cfg(feature = "cuda")]
fn ensure_gpu_compressor_cache(
    ctx: &DeviceContext,
    cache: &mut DeepseekGpuCompressorRuntimeCache,
    capacity_rows: usize,
    ratio: usize,
    width: usize,
    head_dim: usize,
) -> Result<()> {
    let pending_len = ratio
        .checked_mul(width)
        .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 compressor pending size overflow"))?;
    let compressed_len = capacity_rows
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow::anyhow!("DeepSeek V4 compressor compressed size overflow"))?;
    if cache.pending_width != width
        || cache.head_dim != head_dim
        || cache
            .pending_kv
            .as_ref()
            .is_none_or(|buf| buf.len() < pending_len)
    {
        cache.pending_kv = Some(
            ctx.stream
                .alloc_zeros::<bf16>(pending_len)
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 pending kv alloc failed: {err}"))?,
        );
        cache.pending_score = Some(
            ctx.stream
                .alloc_zeros::<bf16>(pending_len)
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 pending score alloc failed: {err}"))?,
        );
        cache.prev_overlap_kv = Some(
            ctx.stream
                .alloc_zeros::<bf16>(ratio * head_dim)
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 prev kv alloc failed: {err}"))?,
        );
        cache.prev_overlap_score = Some(
            ctx.stream
                .alloc_zeros::<bf16>(ratio * head_dim)
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 prev score alloc failed: {err}"))?,
        );
        cache.pending_len = 0;
        cache.compressed_rows = 0;
        cache.pending_width = width;
        cache.head_dim = head_dim;
    }
    if cache.compressed_capacity < capacity_rows
        || cache
            .compressed
            .as_ref()
            .is_none_or(|buf| buf.len() < compressed_len)
    {
        cache.compressed = Some(
            ctx.stream
                .alloc_zeros::<bf16>(compressed_len)
                .map_err(|err| {
                    anyhow::anyhow!("DeepSeek V4 compressed cache alloc failed: {err}")
                })?,
        );
        cache.compressed_capacity = capacity_rows;
        cache.compressed_rows = 0;
        cache.pending_len = 0;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
struct MhcParams {
    pre: CudaSlice<f32>,
    post: CudaSlice<f32>,
    comb: CudaSlice<f32>,
}

#[cfg(feature = "cuda")]
struct MhcParamsRef<'a> {
    pre: &'a CudaSlice<f32>,
    post: &'a CudaSlice<f32>,
    comb: &'a CudaSlice<f32>,
}

#[cfg(feature = "cuda")]
enum MhcParamsView<'a> {
    Owned(MhcParams),
    Cached(MhcParamsRef<'a>),
}

#[cfg(feature = "cuda")]
impl<'a> MhcParamsView<'a> {
    fn pre(&self) -> &CudaSlice<f32> {
        match self {
            Self::Owned(params) => &params.pre,
            Self::Cached(params) => params.pre,
        }
    }

    fn post(&self) -> &CudaSlice<f32> {
        match self {
            Self::Owned(params) => &params.post,
            Self::Cached(params) => params.post,
        }
    }

    fn comb(&self) -> &CudaSlice<f32> {
        match self {
            Self::Owned(params) => &params.comb,
            Self::Cached(params) => params.comb,
        }
    }
}

#[cfg(feature = "cuda")]
fn gen_mhc_params(
    ctx: &DeviceContext,
    hc: &DeepseekV4HyperConnection,
    stream: &HiddenStates,
    hc_mult: usize,
    hc_eps: f32,
    hc_sinkhorn_iters: usize,
) -> Result<MhcParams> {
    ensure!(
        hc_mult > 0,
        "DeepSeek V4 MHC generation requires non-zero hc_mult"
    );
    let mix_dim = (2 + hc_mult) * hc_mult;
    ensure!(
        hc.mix_fn.cols == stream.hidden_dim && hc.mix_fn.rows >= mix_dim,
        "DeepSeek V4 HC mix shape {}x{} cannot produce {} weights from stream dim {}",
        hc.mix_fn.rows,
        hc.mix_fn.cols,
        mix_dim,
        stream.hidden_dim
    );
    ensure!(
        hc.base.len >= mix_dim && hc.scale.len >= 3,
        "DeepSeek V4 HC base/scale too short: base={} scale={} required_base={} required_scale=3",
        hc.base.len,
        hc.scale.len,
        mix_dim
    );
    ensure!(
        hc.base.len >= mix_dim && hc.scale.len >= 3,
        "DeepSeek V4 HC base/scale too short: base={} scale={} required_base={} required_scale=3",
        hc.base.len,
        hc.scale.len,
        mix_dim
    );

    let mixes = ops::gemm(ctx, &hc.mix_fn, stream)?;
    let mut pre = ctx
        .stream
        .alloc_zeros::<f32>(stream.seq_len * hc_mult)
        .map_err(|err| anyhow::anyhow!("DeepSeek V4 HC pre alloc failed: {err}"))?;
    let mut post = ctx
        .stream
        .alloc_zeros::<f32>(stream.seq_len * hc_mult)
        .map_err(|err| anyhow::anyhow!("DeepSeek V4 HC post alloc failed: {err}"))?;
    let mut comb = ctx
        .stream
        .alloc_zeros::<f32>(stream.seq_len * hc_mult * hc_mult)
        .map_err(|err| anyhow::anyhow!("DeepSeek V4 HC comb alloc failed: {err}"))?;

    {
        let (stream_ptr, _stream_guard) = stream.data.device_ptr(&ctx.stream);
        let (mixes_ptr, _mixes_guard) = mixes.data.device_ptr(&ctx.stream);
        let (base_ptr, _base_guard) = hc.base.data.device_ptr(&ctx.stream);
        let (scale_ptr, _scale_guard) = hc.scale.data.device_ptr(&ctx.stream);
        let (pre_ptr, _pre_guard) = pre.device_ptr_mut(&ctx.stream);
        let (post_ptr, _post_guard) = post.device_ptr_mut(&ctx.stream);
        let (comb_ptr, _comb_guard) = comb.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::dsv4_mhc_params_cuda(
                stream_ptr as *const ffi::Half,
                mixes_ptr as *const ffi::Half,
                base_ptr as *const ffi::Half,
                scale_ptr as *const ffi::Half,
                pre_ptr as *mut f32,
                post_ptr as *mut f32,
                comb_ptr as *mut f32,
                stream.seq_len as i32,
                stream.hidden_dim as i32,
                mixes.hidden_dim as i32,
                hc_mult as i32,
                hc_eps,
                hc_sinkhorn_iters as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 HC params CUDA failed: {err}"))?;
        }
    }

    Ok(MhcParams { pre, post, comb })
}

#[cfg(feature = "cuda")]
fn gen_mhc_params_cached<'a>(
    ctx: &DeviceContext,
    hc: &DeepseekV4HyperConnection,
    stream: &HiddenStates,
    hc_mult: usize,
    hc_eps: f32,
    hc_sinkhorn_iters: usize,
    scratch: &'a mut DeepseekMhcRuntimeScratch,
) -> Result<MhcParamsRef<'a>> {
    ensure!(
        hc_mult > 0,
        "DeepSeek V4 MHC generation requires non-zero hc_mult"
    );
    let mix_dim = (2 + hc_mult) * hc_mult;
    ensure!(
        hc.mix_fn.cols == stream.hidden_dim && hc.mix_fn.rows >= mix_dim,
        "DeepSeek V4 HC mix shape {}x{} cannot produce {} weights from stream dim {}",
        hc.mix_fn.rows,
        hc.mix_fn.cols,
        mix_dim,
        stream.hidden_dim
    );
    ensure!(
        scratch.capacity_tokens >= stream.seq_len
            && scratch.stream_hidden_dim == stream.hidden_dim
            && scratch.mix_dim == hc.mix_fn.rows
            && scratch.hc_mult == hc_mult,
        "DeepSeek V4 MHC scratch shape mismatch"
    );
    scratch.mixes.seq_len = stream.seq_len;
    ops::try_gemm_with_phase_into(
        ctx,
        &hc.mix_fn,
        stream,
        &mut scratch.mixes,
        if stream.seq_len > 1 {
            ops::LinearDispatchPhase::Prefill
        } else {
            ops::LinearDispatchPhase::Decode
        },
    )?;

    {
        let (stream_ptr, _stream_guard) = stream.data.device_ptr(&ctx.stream);
        let (mixes_ptr, _mixes_guard) = scratch.mixes.data.device_ptr(&ctx.stream);
        let (base_ptr, _base_guard) = hc.base.data.device_ptr(&ctx.stream);
        let (scale_ptr, _scale_guard) = hc.scale.data.device_ptr(&ctx.stream);
        let (pre_ptr, _pre_guard) = scratch.pre.device_ptr_mut(&ctx.stream);
        let (post_ptr, _post_guard) = scratch.post.device_ptr_mut(&ctx.stream);
        let (comb_ptr, _comb_guard) = scratch.comb.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::dsv4_mhc_params_cuda(
                stream_ptr as *const ffi::Half,
                mixes_ptr as *const ffi::Half,
                base_ptr as *const ffi::Half,
                scale_ptr as *const ffi::Half,
                pre_ptr as *mut f32,
                post_ptr as *mut f32,
                comb_ptr as *mut f32,
                stream.seq_len as i32,
                stream.hidden_dim as i32,
                scratch.mixes.hidden_dim as i32,
                hc_mult as i32,
                hc_eps,
                hc_sinkhorn_iters as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 HC params CUDA failed: {err}"))?;
        }
    }

    Ok(MhcParamsRef {
        pre: &scratch.pre,
        post: &scratch.post,
        comb: &scratch.comb,
    })
}

#[cfg(feature = "cuda")]
fn hc_pre_from_stream(
    ctx: &DeviceContext,
    stream: &HiddenStates,
    pre: &CudaSlice<f32>,
    hidden_size: usize,
    hc_mult: usize,
) -> Result<HiddenStates> {
    let mut out = unsafe { HiddenStates::uninit(ctx, hidden_size, stream.seq_len)? };
    hc_pre_from_stream_into(ctx, stream, pre, hidden_size, hc_mult, &mut out)?;
    Ok(out)
}

#[cfg(feature = "cuda")]
fn hc_pre_from_stream_into(
    ctx: &DeviceContext,
    stream: &HiddenStates,
    pre: &CudaSlice<f32>,
    hidden_size: usize,
    hc_mult: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        stream.hidden_dim == hidden_size * hc_mult,
        "DeepSeek V4 HC pre stream dim {} does not match hidden_size {} * hc_mult {}",
        stream.hidden_dim,
        hidden_size,
        hc_mult
    );
    ensure!(
        pre.len() >= stream.seq_len * hc_mult,
        "DeepSeek V4 HC pre len {} is smaller than seq_len {} * hc_mult {}",
        pre.len(),
        stream.seq_len,
        hc_mult
    );
    ensure!(
        out.hidden_dim == hidden_size && out.seq_len == stream.seq_len,
        "DeepSeek V4 HC pre output shape mismatch: out={}x{} expected={}x{}",
        out.seq_len,
        out.hidden_dim,
        stream.seq_len,
        hidden_size
    );
    {
        let (stream_ptr, _stream_guard) = stream.data.device_ptr(&ctx.stream);
        let (pre_ptr, _pre_guard) = pre.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::dsv4_mhc_pre_cuda(
                stream_ptr as *const ffi::Half,
                pre_ptr as *const f32,
                out_ptr as *mut ffi::Half,
                stream.seq_len as i32,
                hidden_size as i32,
                hc_mult as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 HC pre CUDA failed: {err}"))?;
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn hc_post_to_stream(
    ctx: &DeviceContext,
    new_x: &HiddenStates,
    residual: &HiddenStates,
    post: &CudaSlice<f32>,
    comb: &CudaSlice<f32>,
    hidden_size: usize,
    hc_mult: usize,
) -> Result<HiddenStates> {
    let mut out = unsafe { HiddenStates::uninit(ctx, hidden_size * hc_mult, residual.seq_len)? };
    hc_post_to_stream_into(
        ctx,
        new_x,
        residual,
        post,
        comb,
        hidden_size,
        hc_mult,
        &mut out,
    )?;
    Ok(out)
}

#[cfg(feature = "cuda")]
fn hc_post_to_stream_into(
    ctx: &DeviceContext,
    new_x: &HiddenStates,
    residual: &HiddenStates,
    post: &CudaSlice<f32>,
    comb: &CudaSlice<f32>,
    hidden_size: usize,
    hc_mult: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        new_x.hidden_dim == hidden_size && residual.hidden_dim == hidden_size * hc_mult,
        "DeepSeek V4 HC post dim mismatch: new_x={} residual={} hidden_size={} hc_mult={}",
        new_x.hidden_dim,
        residual.hidden_dim,
        hidden_size,
        hc_mult
    );
    ensure!(
        new_x.seq_len == residual.seq_len,
        "DeepSeek V4 HC post seq mismatch: new_x={} residual={}",
        new_x.seq_len,
        residual.seq_len
    );
    ensure!(
        post.len() >= residual.seq_len * hc_mult
            && comb.len() >= residual.seq_len * hc_mult * hc_mult,
        "DeepSeek V4 HC post weights too small: post={} comb={} seq_len={} hc_mult={}",
        post.len(),
        comb.len(),
        residual.seq_len,
        hc_mult
    );
    ensure!(
        out.hidden_dim == hidden_size * hc_mult && out.seq_len == residual.seq_len,
        "DeepSeek V4 HC post output shape mismatch: out={}x{} expected={}x{}",
        out.seq_len,
        out.hidden_dim,
        residual.seq_len,
        hidden_size * hc_mult
    );

    {
        let (new_ptr, _new_guard) = new_x.data.device_ptr(&ctx.stream);
        let (residual_ptr, _residual_guard) = residual.data.device_ptr(&ctx.stream);
        let (post_ptr, _post_guard) = post.device_ptr(&ctx.stream);
        let (comb_ptr, _comb_guard) = comb.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::dsv4_mhc_post_cuda(
                new_ptr as *const ffi::Half,
                residual_ptr as *const ffi::Half,
                post_ptr as *const f32,
                comb_ptr as *const f32,
                out_ptr as *mut ffi::Half,
                residual.seq_len as i32,
                hidden_size as i32,
                hc_mult as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 HC post CUDA failed: {err}"))?;
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
fn compressor_forward(
    ctx: &DeviceContext,
    compressor: &DeepseekV4Compressor,
    x: &HiddenStates,
    head_dim: usize,
    ratio: usize,
    overlap: bool,
    eps: f32,
) -> Result<Vec<f32>> {
    ensure!(ratio > 0, "DeepSeek V4 compressor ratio must be non-zero");
    let coeff = if overlap { 2 } else { 1 };
    let width = coeff * head_dim;
    ensure!(
        compressor.wkv.rows == width && compressor.wgate.rows == width,
        "DeepSeek V4 compressor rows mismatch: wkv={} wgate={} expected_width={}",
        compressor.wkv.rows,
        compressor.wgate.rows,
        width
    );
    ensure!(
        compressor.ape.rows >= ratio && compressor.ape.cols == width,
        "DeepSeek V4 compressor APE shape {}x{} does not cover ratio {} width {}",
        compressor.ape.rows,
        compressor.ape.cols,
        ratio,
        width
    );
    ensure!(
        compressor.norm.len == head_dim,
        "DeepSeek V4 compressor norm len {} does not match head_dim {}",
        compressor.norm.len,
        head_dim
    );

    let kv_raw = ops::gemm(ctx, &compressor.wkv, x)?;
    let score_raw = ops::gemm(ctx, &compressor.wgate, x)?;
    let kv_raw = ctx
        .stream
        .clone_dtoh(&kv_raw.data)?
        .into_iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let score_raw = ctx
        .stream
        .clone_dtoh(&score_raw.data)?
        .into_iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let ape = matrix_host_f32(ctx, &compressor.ape)?;
    let norm = ctx
        .stream
        .clone_dtoh(&compressor.norm.data)?
        .into_iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();

    let cutoff = x.seq_len - (x.seq_len % ratio);
    let nb = cutoff / ratio;
    if nb == 0 {
        return Ok(Vec::new());
    }
    let mut kv = vec![0.0_f32; cutoff * width];
    let mut score = vec![0.0_f32; cutoff * width];
    for token_idx in 0..cutoff {
        let dst = token_idx * width;
        kv[dst..dst + width].copy_from_slice(&kv_raw[dst..dst + width]);
        score[dst..dst + width].copy_from_slice(&score_raw[dst..dst + width]);
    }
    for token_idx in 0..cutoff {
        let pos = token_idx % ratio;
        for col in 0..width {
            score[token_idx * width + col] += ape[pos * width + col];
        }
    }

    let mut out = vec![0.0_f32; nb * head_dim];
    for block_idx in 0..nb {
        for col in 0..head_dim {
            let mut logits = Vec::with_capacity(if overlap { 2 * ratio } else { ratio });
            let mut values = Vec::with_capacity(logits.capacity());
            if overlap {
                for pos in 0..ratio {
                    if block_idx == 0 {
                        logits.push(f32::NEG_INFINITY);
                        values.push(0.0);
                    } else {
                        let token_idx = (block_idx - 1) * ratio + pos;
                        logits.push(score[token_idx * width + col]);
                        values.push(kv[token_idx * width + col]);
                    }
                }
                for pos in 0..ratio {
                    let token_idx = block_idx * ratio + pos;
                    logits.push(score[token_idx * width + head_dim + col]);
                    values.push(kv[token_idx * width + head_dim + col]);
                }
            } else {
                for pos in 0..ratio {
                    let token_idx = block_idx * ratio + pos;
                    logits.push(score[token_idx * width + col]);
                    values.push(kv[token_idx * width + col]);
                }
            }
            let probs = softmax(&logits);
            out[block_idx * head_dim + col] = probs
                .iter()
                .zip(values)
                .map(|(prob, value)| prob * value)
                .sum::<f32>();
        }
    }
    for block_idx in 0..nb {
        let row = &mut out[block_idx * head_dim..(block_idx + 1) * head_dim];
        let mean_square = row.iter().map(|value| value.powi(2)).sum::<f32>() / head_dim as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        for col in 0..head_dim {
            row[col] *= scale * norm[col];
        }
    }
    Ok(out)
}

#[cfg(all(test, feature = "cuda"))]
fn update_compressor_runtime_cache(
    ctx: &DeviceContext,
    compressor: &DeepseekV4Compressor,
    x: &HiddenStates,
    head_dim: usize,
    ratio: usize,
    overlap: bool,
    eps: f32,
    start_pos: usize,
    rope: Option<(&[f32], &[f32], usize)>,
    cache: &mut DeepseekCompressorRuntimeCache,
) -> Result<()> {
    ensure!(ratio > 0, "DeepSeek V4 compressor ratio must be non-zero");
    let coeff = if overlap { 2 } else { 1 };
    let width = coeff * head_dim;
    ensure!(
        compressor.wkv.rows == width && compressor.wgate.rows == width,
        "DeepSeek V4 compressor rows mismatch: wkv={} wgate={} expected_width={}",
        compressor.wkv.rows,
        compressor.wgate.rows,
        width
    );
    let kv_raw = ops::gemm(ctx, &compressor.wkv, x)?;
    let score_raw = ops::gemm(ctx, &compressor.wgate, x)?;
    let kv_raw = ctx
        .stream
        .clone_dtoh(&kv_raw.data)?
        .into_iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let score_raw = ctx
        .stream
        .clone_dtoh(&score_raw.data)?
        .into_iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let ape = matrix_host_f32(ctx, &compressor.ape)?;
    let norm = ctx
        .stream
        .clone_dtoh(&compressor.norm.data)?
        .into_iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();

    for token_idx in 0..x.seq_len {
        let abs_pos = start_pos + token_idx;
        let pos_in_block = abs_pos % ratio;
        let src = token_idx * width;
        for col in 0..width {
            cache.pending_kv.push(kv_raw[src + col]);
            cache
                .pending_score
                .push(score_raw[src + col] + ape[pos_in_block * width + col]);
        }
        if pos_in_block + 1 != ratio {
            continue;
        }

        let mut row = vec![0.0_f32; head_dim];
        for col in 0..head_dim {
            let mut logits = Vec::with_capacity(if overlap { 2 * ratio } else { ratio });
            let mut values = Vec::with_capacity(logits.capacity());
            if overlap {
                for pos in 0..ratio {
                    if cache.prev_overlap_kv.is_empty() {
                        logits.push(f32::NEG_INFINITY);
                        values.push(0.0);
                    } else {
                        logits.push(cache.prev_overlap_score[pos * head_dim + col]);
                        values.push(cache.prev_overlap_kv[pos * head_dim + col]);
                    }
                }
                for pos in 0..ratio {
                    logits.push(cache.pending_score[pos * width + head_dim + col]);
                    values.push(cache.pending_kv[pos * width + head_dim + col]);
                }
            } else {
                for pos in 0..ratio {
                    logits.push(cache.pending_score[pos * width + col]);
                    values.push(cache.pending_kv[pos * width + col]);
                }
            }
            let probs = softmax(&logits);
            row[col] = probs
                .iter()
                .zip(values)
                .map(|(prob, value)| prob * value)
                .sum::<f32>();
        }
        let mean_square = row.iter().map(|value| value.powi(2)).sum::<f32>() / head_dim as f32;
        let norm_scale = 1.0 / (mean_square + eps).sqrt();
        for col in 0..head_dim {
            row[col] *= norm_scale * norm[col];
        }
        if let Some((rope_cos, rope_sin, rope_dim)) = rope {
            let local_idx = token_idx;
            apply_partial_rope(
                &mut row,
                &rope_cos[local_idx * rope_dim..(local_idx + 1) * rope_dim],
                &rope_sin[local_idx * rope_dim..(local_idx + 1) * rope_dim],
                rope_dim,
                1.0,
            );
        }
        cache.compressed.push(DeepseekCompressedRow {
            end_pos: abs_pos,
            values: row,
        });

        if overlap {
            cache.prev_overlap_kv.clear();
            cache.prev_overlap_score.clear();
            cache.prev_overlap_kv.reserve(ratio * head_dim);
            cache.prev_overlap_score.reserve(ratio * head_dim);
            for pos in 0..ratio {
                for col in 0..head_dim {
                    cache
                        .prev_overlap_kv
                        .push(cache.pending_kv[pos * width + col]);
                    cache
                        .prev_overlap_score
                        .push(cache.pending_score[pos * width + col]);
                }
            }
        }
        cache.pending_kv.clear();
        cache.pending_score.clear();
    }

    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
fn csa_selected_blocks(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    indexer: &DeepseekV4Indexer,
    x: &HiddenStates,
    c_q: &HiddenStates,
    ratio: usize,
) -> Result<Vec<Vec<usize>>> {
    let keys = compressor_forward(
        ctx,
        &indexer.compressor,
        x,
        config.index_head_dim,
        ratio,
        true,
        config.rms_norm_eps,
    )?;
    let nb = keys.len() / config.index_head_dim;
    let q_i = ops::gemm(ctx, &indexer.wq_b, c_q)?;
    let weights = ops::gemm(ctx, &indexer.weights_proj, x)?;
    ensure!(
        q_i.hidden_dim.is_multiple_of(config.index_head_dim),
        "DeepSeek V4 indexer q width {} is not divisible by index_head_dim {}",
        q_i.hidden_dim,
        config.index_head_dim
    );
    let local_index_heads = q_i.hidden_dim / config.index_head_dim;
    ensure!(
        weights.hidden_dim == local_index_heads,
        "DeepSeek V4 indexer weights width {} does not match local heads {}",
        weights.hidden_dim,
        local_index_heads
    );
    let q_host = ctx
        .stream
        .clone_dtoh(&q_i.data)?
        .into_iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let weights_host = ctx
        .stream
        .clone_dtoh(&weights.data)?
        .into_iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let score_scale =
        (config.index_head_dim as f32).powf(-0.5) * (config.index_n_heads as f32).powf(-0.5);
    let mut out = vec![Vec::new(); x.seq_len];
    for token_idx in 0..x.seq_len {
        let mut scored = Vec::new();
        for block_idx in 0..nb {
            if block_idx >= (token_idx + 1) / ratio {
                continue;
            }
            let key =
                &keys[block_idx * config.index_head_dim..(block_idx + 1) * config.index_head_dim];
            let mut score = 0.0_f32;
            for head_idx in 0..local_index_heads {
                let q_start = token_idx * q_i.hidden_dim + head_idx * config.index_head_dim;
                let qh = &q_host[q_start..q_start + config.index_head_dim];
                let weight = weights_host[token_idx * weights.hidden_dim + head_idx] * score_scale;
                score += weight * dot(qh, key).max(0.0);
            }
            if score.is_finite() {
                scored.push((score, block_idx));
            }
        }
        scored.sort_by(|lhs, rhs| rhs.0.total_cmp(&lhs.0).then_with(|| lhs.1.cmp(&rhs.1)));
        scored.truncate(config.index_topk.min(scored.len()));
        out[token_idx] = scored.into_iter().map(|(_, block_idx)| block_idx).collect();
    }
    Ok(out)
}

#[cfg(all(test, feature = "cuda"))]
fn csa_selected_blocks_cached(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    indexer: &DeepseekV4Indexer,
    x: &HiddenStates,
    c_q: &HiddenStates,
    start_pos: usize,
    ratio: usize,
    cache: &mut DeepseekCompressorRuntimeCache,
) -> Result<Vec<Vec<usize>>> {
    update_compressor_runtime_cache(
        ctx,
        &indexer.compressor,
        x,
        config.index_head_dim,
        ratio,
        true,
        config.rms_norm_eps,
        start_pos,
        None,
        cache,
    )?;
    let q_i = ops::gemm(ctx, &indexer.wq_b, c_q)?;
    let weights = ops::gemm(ctx, &indexer.weights_proj, x)?;
    ensure!(
        q_i.hidden_dim.is_multiple_of(config.index_head_dim),
        "DeepSeek V4 indexer q width {} is not divisible by index_head_dim {}",
        q_i.hidden_dim,
        config.index_head_dim
    );
    let local_index_heads = q_i.hidden_dim / config.index_head_dim;
    ensure!(
        weights.hidden_dim == local_index_heads,
        "DeepSeek V4 indexer weights width {} does not match local heads {}",
        weights.hidden_dim,
        local_index_heads
    );
    let q_host = ctx
        .stream
        .clone_dtoh(&q_i.data)?
        .into_iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let weights_host = ctx
        .stream
        .clone_dtoh(&weights.data)?
        .into_iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let score_scale =
        (config.index_head_dim as f32).powf(-0.5) * (config.index_n_heads as f32).powf(-0.5);
    let mut out = vec![Vec::new(); x.seq_len];
    for token_idx in 0..x.seq_len {
        let abs_pos = start_pos + token_idx;
        let mut scored = Vec::new();
        for (block_idx, key) in cache.compressed.iter().enumerate() {
            if key.end_pos > abs_pos {
                continue;
            }
            let mut score = 0.0_f32;
            for head_idx in 0..local_index_heads {
                let q_start = token_idx * q_i.hidden_dim + head_idx * config.index_head_dim;
                let qh = &q_host[q_start..q_start + config.index_head_dim];
                let weight = weights_host[token_idx * weights.hidden_dim + head_idx] * score_scale;
                score += weight * dot(qh, &key.values).max(0.0);
            }
            if score.is_finite() {
                scored.push((score, block_idx));
            }
        }
        scored.sort_by(|lhs, rhs| rhs.0.total_cmp(&lhs.0).then_with(|| lhs.1.cmp(&rhs.1)));
        scored.truncate(config.index_topk.min(scored.len()));
        out[token_idx] = scored.into_iter().map(|(_, block_idx)| block_idx).collect();
    }
    Ok(out)
}

#[cfg(all(test, feature = "cuda"))]
fn matrix_host_f32(ctx: &DeviceContext, matrix: &DeviceMatrix) -> Result<Vec<f32>> {
    Ok(ctx
        .stream
        .clone_dtoh(&matrix.data)?
        .into_iter()
        .map(|value| value.to_f32())
        .collect())
}

#[cfg(feature = "cuda")]
fn head_hidden_from_stream(
    ctx: &DeviceContext,
    head_hc: &DeepseekV4HyperConnection,
    stream: &HiddenStates,
    token_idx: usize,
    hidden_size: usize,
    hc_mult: usize,
    hc_eps: f32,
) -> Result<HiddenStates> {
    ensure!(
        token_idx < stream.seq_len,
        "DeepSeek V4 head token {} out of range for stream seq_len {}",
        token_idx,
        stream.seq_len
    );
    ensure!(
        stream.hidden_dim == hidden_size * hc_mult,
        "DeepSeek V4 head stream dim {} does not match hidden_size {} * hc_mult {}",
        stream.hidden_dim,
        hidden_size,
        hc_mult
    );
    ensure!(
        head_hc.mix_fn.cols == stream.hidden_dim && head_hc.mix_fn.rows >= hc_mult,
        "DeepSeek V4 head HC mix shape {}x{} cannot produce {} pre weights from stream dim {}",
        head_hc.mix_fn.rows,
        head_hc.mix_fn.cols,
        hc_mult,
        stream.hidden_dim
    );
    ensure!(
        head_hc.base.len >= hc_mult && head_hc.scale.len >= 1,
        "DeepSeek V4 head HC base/scale too short: base={} scale={} hc_mult={}",
        head_hc.base.len,
        head_hc.scale.len,
        hc_mult
    );

    let stream_row = extract_hidden_token_with_width(ctx, stream, token_idx, stream.hidden_dim)?;
    let mixes = ops::gemm(ctx, &head_hc.mix_fn, &stream_row)?;
    let mut out = unsafe { HiddenStates::uninit(ctx, hidden_size, 1)? };
    {
        let (row_ptr, _row_guard) = stream_row.data.device_ptr(&ctx.stream);
        let (mixes_ptr, _mixes_guard) = mixes.data.device_ptr(&ctx.stream);
        let (base_ptr, _base_guard) = head_hc.base.data.device_ptr(&ctx.stream);
        let (scale_ptr, _scale_guard) = head_hc.scale.data.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::dsv4_mhc_head_pre_cuda(
                row_ptr as *const ffi::Half,
                mixes_ptr as *const ffi::Half,
                base_ptr as *const ffi::Half,
                scale_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                stream.hidden_dim as i32,
                hidden_size as i32,
                hc_mult as i32,
                hc_eps,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 head HC pre CUDA failed: {err}"))?;
        }
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
fn extract_hidden_token_with_width(
    ctx: &DeviceContext,
    hidden: &HiddenStates,
    token_idx: usize,
    width: usize,
) -> Result<HiddenStates> {
    ensure!(
        token_idx < hidden.seq_len,
        "DeepSeek V4 token {} out of range for seq_len {}",
        token_idx,
        hidden.seq_len
    );
    ensure!(
        hidden.hidden_dim == width,
        "DeepSeek V4 token extract width {} does not match hidden dim {}",
        width,
        hidden.hidden_dim
    );
    let mut out = unsafe { HiddenStates::uninit(ctx, width, 1)? };
    let start = token_idx * width;
    let src = hidden.data.slice(start..start + width);
    ctx.stream
        .memcpy_dtod(&src, &mut out.data)
        .map_err(|err| anyhow::anyhow!("DeepSeek V4 token extract copy: {err}"))?;
    Ok(out)
}

#[cfg(all(test, feature = "cuda"))]
fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[cfg(all(test, feature = "cuda"))]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(lhs, rhs)| lhs * rhs).sum()
}

#[cfg(all(test, feature = "cuda"))]
fn sink_softmax(logits: &[f32], sink: f32) -> Vec<f32> {
    let max = logits.iter().copied().fold(sink, f32::max);
    let denom = logits.iter().map(|value| (*value - max).exp()).sum::<f32>() + (sink - max).exp();
    logits
        .iter()
        .map(|value| (*value - max).exp() / denom)
        .collect()
}

#[cfg(all(test, feature = "cuda"))]
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return vec![0.0; logits.len()];
    }
    let exp = logits
        .iter()
        .map(|value| (*value - max).exp())
        .collect::<Vec<_>>();
    let denom = exp.iter().sum::<f32>();
    exp.into_iter().map(|value| value / denom).collect()
}

#[cfg(all(test, feature = "cuda"))]
fn fixed_rms_norm_in_place(values: &mut [f32], eps: f32) {
    let mean_square = values.iter().map(|value| value.powi(2)).sum::<f32>() / values.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    for value in values {
        *value *= scale;
    }
}

#[cfg(all(test, feature = "cuda"))]
fn build_rope_cache(
    seq: usize,
    dim: usize,
    base: f32,
    original_seq_len: usize,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> (Vec<f32>, Vec<f32>) {
    if dim == 0 {
        return (Vec::new(), Vec::new());
    }
    let half = dim / 2;
    let mut inv_freq = (0..half)
        .map(|i| 1.0_f32 / base.powf((2 * i) as f32 / dim as f32))
        .collect::<Vec<_>>();
    if original_seq_len > 0 {
        let low = yarn_correction_dim(beta_fast, dim, base, original_seq_len as f32)
            .floor()
            .max(0.0) as usize;
        let high = yarn_correction_dim(beta_slow, dim, base, original_seq_len as f32)
            .ceil()
            .max(0.0)
            .min((dim.saturating_sub(1)) as f32) as usize;
        let denom = if low == high {
            0.001_f32
        } else {
            (high - low) as f32
        };
        for (i, freq) in inv_freq.iter_mut().enumerate() {
            let ramp = ((i as f32 - low as f32) / denom).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *freq = *freq / factor * (1.0 - smooth) + *freq * smooth;
        }
    }
    let mut cos = vec![0.0_f32; seq * dim];
    let mut sin = vec![0.0_f32; seq * dim];
    for pos in 0..seq {
        for i in 0..half {
            let value = pos as f32 * inv_freq[i];
            let c = value.cos();
            let s = value.sin();
            let col = 2 * i;
            cos[pos * dim + col] = c;
            cos[pos * dim + col + 1] = c;
            sin[pos * dim + col] = s;
            sin[pos * dim + col + 1] = s;
        }
    }
    (cos, sin)
}

#[cfg(all(test, feature = "cuda"))]
fn build_rope_cache_range(
    start_pos: usize,
    seq: usize,
    dim: usize,
    base: f32,
    original_seq_len: usize,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> (Vec<f32>, Vec<f32>) {
    if dim == 0 {
        return (Vec::new(), Vec::new());
    }
    let half = dim / 2;
    let mut inv_freq = (0..half)
        .map(|i| 1.0_f32 / base.powf((2 * i) as f32 / dim as f32))
        .collect::<Vec<_>>();
    if original_seq_len > 0 {
        let low = yarn_correction_dim(beta_fast, dim, base, original_seq_len as f32)
            .floor()
            .max(0.0) as usize;
        let high = yarn_correction_dim(beta_slow, dim, base, original_seq_len as f32)
            .ceil()
            .max(0.0)
            .min((dim.saturating_sub(1)) as f32) as usize;
        let denom = if low == high {
            0.001_f32
        } else {
            (high - low) as f32
        };
        for (i, freq) in inv_freq.iter_mut().enumerate() {
            let ramp = ((i as f32 - low as f32) / denom).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *freq = *freq / factor * (1.0 - smooth) + *freq * smooth;
        }
    }
    let mut cos = vec![0.0_f32; seq * dim];
    let mut sin = vec![0.0_f32; seq * dim];
    for local_pos in 0..seq {
        let abs_pos = start_pos + local_pos;
        for i in 0..half {
            let value = abs_pos as f32 * inv_freq[i];
            let c = value.cos();
            let s = value.sin();
            let col = 2 * i;
            cos[local_pos * dim + col] = c;
            cos[local_pos * dim + col + 1] = c;
            sin[local_pos * dim + col] = s;
            sin[local_pos * dim + col + 1] = s;
        }
    }
    (cos, sin)
}

#[cfg(all(test, feature = "cuda"))]
fn yarn_correction_dim(num_rotations: f32, dim: usize, base: f32, max_seq_len: f32) -> f32 {
    dim as f32 * (max_seq_len / (num_rotations * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * base.ln())
}

#[cfg(all(test, feature = "cuda"))]
fn apply_partial_rope(row: &mut [f32], cos: &[f32], sin: &[f32], rope_dim: usize, sign: f32) {
    if rope_dim == 0 {
        return;
    }
    debug_assert!(row.len() >= rope_dim);
    debug_assert!(cos.len() >= rope_dim && sin.len() >= rope_dim);
    let start = row.len() - rope_dim;
    let half = rope_dim / 2;
    for i in 0..half {
        let idx = start + 2 * i;
        let a = row[idx];
        let b = row[idx + 1];
        let c = cos[2 * i];
        let s = sign * sin[2 * i];
        row[idx] = a * c - b * s;
        row[idx + 1] = b * c + a * s;
    }
}

fn infer_real_reference_enabled() -> Result<bool> {
    let Some(raw) = std::env::var("ARLE_DSV4_INFER_REAL_REFERENCE").ok() else {
        return Ok(false);
    };
    match raw.as_str() {
        "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON" => Ok(true),
        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF" => Ok(false),
        _ => bail!("invalid ARLE_DSV4_INFER_REAL_REFERENCE value `{raw}`"),
    }
}

fn load_layer_weights_enabled() -> Result<bool> {
    let Some(raw) = std::env::var("ARLE_DSV4_LOAD_LAYER_WEIGHTS").ok() else {
        return Ok(false);
    };
    match raw.as_str() {
        "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON" => Ok(true),
        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF" => Ok(false),
        _ => bail!("invalid ARLE_DSV4_LOAD_LAYER_WEIGHTS value `{raw}`"),
    }
}

fn dsv4_gpu_ffn_layer_limit() -> Result<usize> {
    let Some(raw) = std::env::var("ARLE_DSV4_GPU_FFN_LAYERS").ok() else {
        return Ok(0);
    };
    raw.parse::<usize>()
        .map_err(|err| anyhow::anyhow!("invalid ARLE_DSV4_GPU_FFN_LAYERS value `{raw}`: {err}"))
}

fn dsv4_gpu_full_layer_limit() -> Result<usize> {
    let Some(raw) = std::env::var("ARLE_DSV4_GPU_FULL_LAYERS").ok() else {
        return Ok(0);
    };
    raw.parse::<usize>()
        .map_err(|err| anyhow::anyhow!("invalid ARLE_DSV4_GPU_FULL_LAYERS value `{raw}`: {err}"))
}

fn dsv4_gpu_contextual_logits_enabled() -> Result<bool> {
    let Some(raw) = std::env::var("ARLE_DSV4_GPU_CONTEXT_TOKENS").ok() else {
        return Ok(dsv4_gpu_full_layer_limit()? > 0);
    };
    match raw.as_str() {
        "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON" => Ok(true),
        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF" => Ok(false),
        _ => bail!("invalid ARLE_DSV4_GPU_CONTEXT_TOKENS value `{raw}`"),
    }
}

#[cfg(feature = "cuda")]
fn dsv4_moe_deepep_enabled() -> Result<bool> {
    let Some(raw) = std::env::var("ARLE_DSV4_MOE_BACKEND").ok() else {
        return Ok(false);
    };
    match raw.as_str() {
        "deepep" | "DeepEP" | "dispatch" | "dispatch_combine" => Ok(true),
        "allreduce" | "all_reduce" | "legacy" | "0" | "false" | "FALSE" | "off" | "OFF" => {
            Ok(false)
        }
        _ => bail!("invalid ARLE_DSV4_MOE_BACKEND value `{raw}`"),
    }
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn dsv4_combine_overlap_enabled() -> bool {
    std::env::var("ARLE_DSV4_COMBINE_OVERLAP")
        .map(|raw| !matches!(raw.as_str(), "0" | "false" | "FALSE" | "off" | "OFF"))
        .unwrap_or(false)
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
fn dsv4_nccl_env_bootstrap_with_port_offset(
    offset: u16,
) -> Result<crate::distributed::nccl::NcclInitMethod> {
    use std::net::ToSocketAddrs;

    let host = std::env::var("MASTER_ADDR").unwrap_or_else(|_| "127.0.0.1".to_string());
    let raw_port = std::env::var("MASTER_PORT")
        .map_err(|err| anyhow::anyhow!("NCCL overlap EnvBootstrap requires MASTER_PORT: {err}"))?;
    let port = raw_port.parse::<u16>().map_err(|err| {
        anyhow::anyhow!("invalid MASTER_PORT for NCCL overlap: {raw_port}: {err}")
    })?;
    let port = port.checked_add(offset).ok_or_else(|| {
        anyhow::anyhow!("MASTER_PORT {port} plus NCCL overlap offset {offset} overflows u16")
    })?;
    let addr = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(|err| anyhow::anyhow!("failed to resolve NCCL overlap addr {host}:{port}: {err}"))?
        .next()
        .ok_or_else(|| anyhow::anyhow!("NCCL overlap addr {host}:{port} resolved to zero addrs"))?;
    Ok(crate::distributed::nccl::NcclInitMethod::TcpStore(addr))
}

#[cfg(feature = "cuda")]
fn dsv4_trace_layer_enabled() -> bool {
    std::env::var("ARLE_DSV4_TRACE_LAYER")
        .ok()
        .is_some_and(|raw| !matches!(raw.as_str(), "0" | "false" | "FALSE" | "off" | "OFF"))
}

#[cfg(feature = "cuda")]
fn dsv4_trace_begin(ctx: &DeviceContext) -> Result<Instant> {
    if dsv4_trace_layer_enabled() {
        ctx.stream
            .synchronize()
            .map_err(|err| anyhow::anyhow!("DeepSeek V4 trace pre-sync failed: {err}"))?;
    }
    Ok(Instant::now())
}

#[cfg(feature = "cuda")]
fn dsv4_trace_end(
    ctx: &DeviceContext,
    phase: &str,
    layer_idx: usize,
    tokens: usize,
    started: Instant,
) -> Result<()> {
    if !dsv4_trace_layer_enabled() {
        return Ok(());
    }
    ctx.stream
        .synchronize()
        .map_err(|err| anyhow::anyhow!("DeepSeek V4 trace post-sync failed: {err}"))?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
    info!(
        "dsv4_trace layer={} phase={} tokens={} elapsed_ms={:.3}",
        layer_idx, phase, tokens, elapsed_ms
    );
    Ok(())
}

#[cfg(feature = "cuda")]
pub(super) fn dsv4_incremental_kv_enabled() -> Result<bool> {
    let Some(raw) = std::env::var("ARLE_DSV4_INCREMENTAL_KV").ok() else {
        return Ok(false);
    };
    match raw.as_str() {
        "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON" => Ok(true),
        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF" => Ok(false),
        _ => bail!("invalid ARLE_DSV4_INCREMENTAL_KV value `{raw}`"),
    }
}

fn deepseek_find_tensor<'data>(
    shards: &[safetensors::SafeTensors<'data>],
    weight_map: &std::collections::HashMap<String, usize>,
    name: &str,
) -> Result<safetensors::tensor::TensorView<'data>> {
    let shard_idx = *weight_map
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))?;
    let shard = shards
        .get(shard_idx)
        .ok_or_else(|| anyhow::anyhow!("tensor {name} points to missing shard {shard_idx}"))?;
    shard
        .tensor(name)
        .map_err(|err| anyhow::anyhow!("loading tensor {name}: {err}"))
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::distributed::expert_state::ExpertGroup;
    use half::bf16;

    fn bf16_vec(values: &[f32]) -> Vec<bf16> {
        values.iter().map(|&value| bf16::from_f32(value)).collect()
    }

    fn tiny_config() -> DeepSeekV4Config {
        DeepSeekV4Config::from_json_str(
            r#"{
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "torch_dtype": "bfloat16",
            "vocab_size": 4,
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
            "n_routed_experts": 1,
            "n_shared_experts": 0,
            "num_experts_per_tok": 1,
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

    fn matrix(
        ctx: &DeviceContext,
        values: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<DeviceMatrix> {
        DeviceMatrix::from_host(ctx, &bf16_vec(values), rows, cols)
    }

    fn vec(ctx: &DeviceContext, values: &[f32]) -> Result<DeviceVec> {
        DeviceVec::from_host(ctx, &bf16_vec(values))
    }

    fn dummy_attention(ctx: &DeviceContext) -> Result<DeepseekV4Attention> {
        Ok(DeepseekV4Attention {
            wq_a: matrix(ctx, &[0.0, 0.0], 1, 2)?,
            q_norm: vec(ctx, &[1.0])?,
            wq_b: matrix(ctx, &[0.0], 1, 1)?,
            wkv: matrix(ctx, &[0.0, 0.0], 1, 2)?,
            kv_norm: vec(ctx, &[1.0])?,
            wo_a: matrix(ctx, &[0.0], 1, 1)?,
            wo_b: matrix(ctx, &[0.0, 0.0], 2, 1)?,
            attn_sink: vec(ctx, &[0.0])?,
            compressor: None,
            indexer: None,
        })
    }

    fn assert_close(got: f32, expected: f32, tol: f32) {
        assert!(
            (got - expected).abs() <= tol,
            "expected {expected}, got {got}, tol {tol}"
        );
    }

    #[test]
    fn partial_rope_rotates_tail_adjacent_pairs() {
        let mut row = vec![10.0, 20.0, 1.0, 0.0, 0.0, 1.0];
        let cos = vec![0.0, 0.0, 0.0, 0.0];
        let sin = vec![1.0, 1.0, 1.0, 1.0];

        apply_partial_rope(&mut row, &cos, &sin, 4, 1.0);

        assert_eq!(row[0], 10.0);
        assert_eq!(row[1], 20.0);
        assert_close(row[2], 0.0, 1.0e-6);
        assert_close(row[3], 1.0, 1.0e-6);
        assert_close(row[4], -1.0, 1.0e-6);
        assert_close(row[5], 0.0, 1.0e-6);
    }

    #[test]
    fn initial_hc_stream_repeats_embedding_rows() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let embeddings = HiddenStates {
            data: ctx.stream.clone_htod(&bf16_vec(&[1.0, 2.0, 3.0, 4.0]))?,
            hidden_dim: 2,
            seq_len: 2,
        };

        let stream = initial_hc_stream_from_embeddings(&ctx, &embeddings, 2, 3)?;
        let host = ctx.stream.clone_dtoh(&stream.data)?;
        ctx.sync()?;
        let got = host.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        assert_eq!(
            got,
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
        );
        Ok(())
    }

    #[test]
    fn head_hidden_from_stream_combines_hc_lanes() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let stream = HiddenStates {
            data: ctx.stream.clone_htod(&bf16_vec(&[1.0, 2.0, 3.0, 5.0]))?,
            hidden_dim: 4,
            seq_len: 1,
        };
        let head_hc = DeepseekV4HyperConnection {
            base: DeviceVec::from_host(&ctx, &bf16_vec(&[0.0, 0.0]))?,
            mix_fn: DeviceMatrix::from_host(
                &ctx,
                &bf16_vec(&[
                    1.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0,
                ]),
                2,
                4,
            )?,
            scale: DeviceVec::from_host(&ctx, &bf16_vec(&[1.0]))?,
        };

        let hidden = head_hidden_from_stream(&ctx, &head_hc, &stream, 0, 2, 2, 0.0)?;
        let host = ctx.stream.clone_dtoh(&hidden.data)?;
        ctx.sync()?;
        let got = host.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        let rsqrt = 1.0_f32 / ((1.0_f32 + 4.0 + 9.0 + 25.0) / 4.0).sqrt();
        let pre0 = sigmoid(rsqrt);
        let pre1 = 0.5_f32;
        let expected = [pre0 * 1.0 + pre1 * 3.0, pre0 * 2.0 + pre1 * 5.0];
        for (idx, value) in got.iter().enumerate() {
            assert!(
                (*value - expected[idx]).abs() < 0.03,
                "idx={idx} expected={} got={value}",
                expected[idx]
            );
        }
        Ok(())
    }

    #[test]
    fn gen_mhc_params_uses_rms_scaled_mixes() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let stream = HiddenStates {
            data: ctx.stream.clone_htod(&bf16_vec(&[1.0, 2.0, 3.0, 5.0]))?,
            hidden_dim: 4,
            seq_len: 1,
        };
        let hc = DeepseekV4HyperConnection {
            base: vec(&ctx, &[0.0; 8])?,
            mix_fn: matrix(
                &ctx,
                &[
                    1.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0,
                ],
                8,
                4,
            )?,
            scale: vec(&ctx, &[1.0, 1.0, 1.0])?,
        };

        let mhc = gen_mhc_params(&ctx, &hc, &stream, 2, 1.0e-6, 2)?;
        let pre = ctx.stream.clone_dtoh(&mhc.pre)?;
        let post = ctx.stream.clone_dtoh(&mhc.post)?;
        let comb = ctx.stream.clone_dtoh(&mhc.comb)?;
        ctx.sync()?;
        let rsqrt = 1.0_f32 / ((1.0_f32 + 4.0 + 9.0 + 25.0) / 4.0 + 1.0e-6).sqrt();
        assert_close(pre[0], sigmoid(rsqrt) + 1.0e-6, 0.003);
        assert_close(pre[1], 0.5 + 1.0e-6, 0.003);
        assert_close(post[0], 1.0, 0.003);
        assert_close(post[1], 2.0 * sigmoid(2.0 * rsqrt), 0.003);
        for col in 0..2 {
            let sum = comb[col] + comb[2 + col];
            assert_close(sum, 1.0, 0.01);
        }
        Ok(())
    }

    #[test]
    fn hc_pre_and_post_move_rows_through_segments() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let stream = HiddenStates {
            data: ctx.stream.clone_htod(&bf16_vec(&[1.0, 2.0, 3.0, 5.0]))?,
            hidden_dim: 4,
            seq_len: 1,
        };

        let pre_weights = ctx.stream.clone_htod(&[0.25_f32, 0.5])?;
        let pre = hc_pre_from_stream(&ctx, &stream, &pre_weights, 2, 2)?;
        let pre_host = ctx.stream.clone_dtoh(&pre.data)?;
        ctx.sync()?;
        let pre_got = pre_host
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        assert_close(pre_got[0], 1.75, 0.01);
        assert_close(pre_got[1], 3.0, 0.01);

        let new_x = HiddenStates {
            data: ctx.stream.clone_htod(&bf16_vec(&[10.0, 20.0]))?,
            hidden_dim: 2,
            seq_len: 1,
        };
        let out = hc_post_to_stream(
            &ctx,
            &new_x,
            &stream,
            &ctx.stream.clone_htod(&[0.1_f32, 0.2])?,
            &ctx.stream.clone_htod(&[1.0_f32, 0.0, 0.25, 0.75])?,
            2,
            2,
        )?;
        let host = ctx.stream.clone_dtoh(&out.data)?;
        ctx.sync()?;
        let got = host.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        assert_close(got[0], 2.0, 0.02);
        assert_close(got[1], 4.0, 0.02);
        assert_close(got[2], 4.5, 0.03);
        assert_close(got[3], 8.25, 0.04);
        Ok(())
    }

    #[test]
    fn top_level_logits_can_run_one_gpu_ffn_layer() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let mut config = DeepseekRuntimeConfig::from_spec(tiny_config());
        config.ep = ExpertGroup::new(0, 1, 1)?;
        let model = DeepseekModel {
            embed_tokens: Some(matrix(
                &ctx,
                &[
                    1.0, 0.0, //
                    0.0, 1.0, //
                    1.0, 1.0, //
                    -1.0, 1.0,
                ],
                4,
                2,
            )?),
            lm_head: Some(matrix(
                &ctx,
                &[
                    1.0, 0.0, //
                    0.0, 1.0, //
                    1.0, 1.0, //
                    -1.0, 1.0,
                ],
                4,
                2,
            )?),
            norm: Some(DeviceVec::ones(&ctx, 2)?),
            head_hc: Some(DeepseekV4HyperConnection {
                base: vec(&ctx, &[0.0])?,
                mix_fn: matrix(&ctx, &[0.0, 0.0], 1, 2)?,
                scale: vec(&ctx, &[0.0])?,
            }),
            layers: vec![DeepseekLayer {
                attn_norm: DeviceVec::ones(&ctx, 2)?,
                hc_attn: DeepseekV4HyperConnection {
                    base: vec(&ctx, &[0.0, 0.0, 0.0])?,
                    mix_fn: matrix(&ctx, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3, 2)?,
                    scale: vec(&ctx, &[0.0, 0.0, 0.0])?,
                },
                attention: dummy_attention(&ctx)?,
                ffn_norm: DeviceVec::ones(&ctx, 2)?,
                hc_ffn: DeepseekV4HyperConnection {
                    base: vec(&ctx, &[0.0, 0.0, 0.0])?,
                    mix_fn: matrix(&ctx, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3, 2)?,
                    scale: vec(&ctx, &[0.0, 0.0, 0.0])?,
                },
                ffn: DeepseekV4MoeBlock {
                    gate_weight: matrix(&ctx, &[1.0, 0.0], 1, 2)?,
                    gate_bias: Some(vec(&ctx, &[0.0])?),
                    gate_tid2eid: None,
                    experts: vec![DeepseekV4Expert {
                        w1: matrix(&ctx, &[1.0, 0.0], 1, 2)?,
                        w2: matrix(&ctx, &[1.0, 1.0], 2, 1)?,
                        w3: matrix(&ctx, &[0.0, 1.0], 1, 2)?,
                    }],
                    grouped_w1_ptrs: None,
                    grouped_w3_ptrs: None,
                    grouped_w2_ptrs: None,
                    shared_experts: None,
                },
            }],
            config,
            ctx,
            layer_communicator: LayerCommunicator::single(),
            reference: None,
        };

        let logits = model
            .compute_top_level_logits_with_ffn_layer_limit(&[0], 1)?
            .expect("logits");
        assert_eq!(logits.len, 4);
        let host = model.ctx.stream.clone_dtoh(&logits.data)?;
        model.ctx.sync()?;
        assert!(host.iter().all(|value| value.to_f32().is_finite()));
        Ok(())
    }

    #[test]
    fn sliding_window_attention_runs_gpu_projection_path() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let hidden = HiddenStates {
            data: ctx.stream.clone_htod(&bf16_vec(&[1.0, 2.0]))?,
            hidden_dim: 2,
            seq_len: 1,
        };
        let attention = DeepseekV4Attention {
            wq_a: matrix(&ctx, &[1.0, 0.0], 1, 2)?,
            q_norm: vec(&ctx, &[1.0])?,
            wq_b: matrix(&ctx, &[1.0], 1, 1)?,
            wkv: matrix(&ctx, &[0.0, 1.0], 1, 2)?,
            kv_norm: vec(&ctx, &[1.0])?,
            wo_a: matrix(&ctx, &[1.0], 1, 1)?,
            wo_b: matrix(&ctx, &[1.0, 1.0], 2, 1)?,
            attn_sink: vec(&ctx, &[0.0])?,
            compressor: None,
            indexer: None,
        };
        let mut config = DeepseekRuntimeConfig::from_spec(tiny_config());
        config.ep = ExpertGroup::new(0, 1, 1)?;
        let model = DeepseekModel {
            config,
            ctx,
            embed_tokens: None,
            lm_head: None,
            norm: None,
            head_hc: None,
            layers: Vec::new(),
            layer_communicator: LayerCommunicator::single(),
            reference: None,
        };

        let out = model.forward_sliding_window_attention(0, &attention, &hidden)?;
        let host = model.ctx.stream.clone_dtoh(&out.data)?;
        model.ctx.sync()?;
        let got = host.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        let expected = 1.0_f32.exp() / (1.0_f32.exp() + 1.0);
        assert_close(got[0], expected, 0.01);
        assert_close(got[1], expected, 0.01);
        Ok(())
    }

    #[test]
    fn compressed_attention_short_sequence_uses_local_window_only() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let hidden = HiddenStates {
            data: ctx.stream.clone_htod(&bf16_vec(&[1.0, 2.0]))?,
            hidden_dim: 2,
            seq_len: 1,
        };
        let attention = DeepseekV4Attention {
            wq_a: matrix(&ctx, &[1.0, 0.0], 1, 2)?,
            q_norm: vec(&ctx, &[1.0])?,
            wq_b: matrix(&ctx, &[1.0], 1, 1)?,
            wkv: matrix(&ctx, &[0.0, 1.0], 1, 2)?,
            kv_norm: vec(&ctx, &[1.0])?,
            wo_a: matrix(&ctx, &[1.0], 1, 1)?,
            wo_b: matrix(&ctx, &[1.0, 1.0], 2, 1)?,
            attn_sink: vec(&ctx, &[0.0])?,
            compressor: None,
            indexer: None,
        };
        let mut config = DeepseekRuntimeConfig::from_spec(tiny_config());
        config.spec.compress_ratios[0] = 4;
        config.ep = ExpertGroup::new(0, 1, 1)?;
        let model = DeepseekModel {
            config,
            ctx,
            embed_tokens: None,
            lm_head: None,
            norm: None,
            head_hc: None,
            layers: Vec::new(),
            layer_communicator: LayerCommunicator::single(),
            reference: None,
        };

        let out = model.forward_sliding_window_attention(0, &attention, &hidden)?;
        let host = model.ctx.stream.clone_dtoh(&out.data)?;
        model.ctx.sync()?;
        let got = host.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
        let expected = 1.0_f32.exp() / (1.0_f32.exp() + 1.0);
        assert_close(got[0], expected, 0.01);
        assert_close(got[1], expected, 0.01);
        Ok(())
    }

    #[test]
    fn compressor_forward_uses_only_complete_blocks() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let hidden = HiddenStates {
            data: ctx
                .stream
                .clone_htod(&bf16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))?,
            hidden_dim: 2,
            seq_len: 3,
        };
        let compressor = DeepseekV4Compressor {
            wkv: matrix(&ctx, &[0.0, 1.0], 1, 2)?,
            wgate: matrix(&ctx, &[1.0, 0.0], 1, 2)?,
            ape: matrix(&ctx, &[0.0, 0.0], 2, 1)?,
            norm: vec(&ctx, &[1.0])?,
        };

        let out = compressor_forward(&ctx, &compressor, &hidden, 1, 2, false, 1.0e-6)?;
        assert_eq!(out.len(), 1);
        assert!(out.iter().all(|value| value.is_finite()));
        assert_close(out[0], 1.0, 0.01);
        Ok(())
    }
}
