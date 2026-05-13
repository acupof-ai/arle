//! DeepSeek V4 model weights.
//!
//! The runtime target is the local `DeepseekV4ForCausalLM` checkpoint at
//! `infer/models/dsv4-mini-1B-init/`. Infer-side DeepSeek wiring uses
//! [`deepseek_spec::DeepSeekV4Config`] and its HF tensor-name contract only.

use std::path::Path;

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
use super::mlp::{DeepseekV4Expert, DeepseekV4MoeBlock};
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use deepseek_spec::DeepSeekV4Config;

use crate::deepseek_v4_manifest::{
    DeepseekV4CheckpointManifest, validate_deepseek_v4_checkpoint_manifest,
};
#[cfg(feature = "cuda")]
use crate::deepseek_v4_reference::DeepseekV4ReferenceModel;
#[cfg(feature = "cuda")]
use crate::model::common;
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
        let model = Self {
            config,
            ctx,
            embed_tokens: None,
            lm_head: None,
            norm: None,
            head_hc: None,
            layers: Vec::new(),
            reference: None,
        };
        model.validate_phase0_sw_decode_scope()?;
        Ok(model)
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
        let mhc = gen_mhc_params(
            &self.ctx,
            &layer.hc_attn,
            stream,
            self.config.hc_mult,
            self.config.hc_eps,
            self.config.hc_sinkhorn_iters,
        )?;
        let attn_in = hc_pre_from_stream(
            &self.ctx,
            stream,
            &mhc.pre,
            self.config.hidden_size,
            self.config.hc_mult,
        )?;
        let mut normed = HiddenStates::zeros(&self.ctx, self.config.hidden_size, stream.seq_len)?;
        ops::rms_norm_batch_into(
            &self.ctx,
            &attn_in,
            &layer.attn_norm,
            self.config.rms_norm_eps,
            &mut normed,
        );
        let attn_out =
            self.forward_sliding_window_attention(layer_idx, &layer.attention, &normed)?;
        let stream = hc_post_to_stream(
            &self.ctx,
            &attn_out,
            stream,
            &mhc.post,
            &mhc.comb,
            self.config.hidden_size,
            self.config.hc_mult,
        )?;
        self.forward_ffn_layer_stream(layer_idx, &stream, tokens)
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
        if compress_ratio > 0 {
            ensure!(
                hidden.seq_len <= compress_ratio,
                "DeepSeek V4 GPU attention layer {} needs compressed blocks for seq_len {} > compress_ratio {}; compressed CSA/HCA blocks are not wired yet",
                layer_idx,
                hidden.seq_len,
                compress_ratio
            );
        }
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
        let mut c_q_normed = HiddenStates::zeros(&self.ctx, c_q.hidden_dim, c_q.seq_len)?;
        ops::rms_norm_batch_into(
            &self.ctx,
            &c_q,
            &attention.q_norm,
            self.config.rms_norm_eps,
            &mut c_q_normed,
        );
        let q_raw = ops::gemm(&self.ctx, &attention.wq_b, &c_q_normed)?;
        let kv_raw = ops::gemm(&self.ctx, &attention.wkv, hidden)?;
        let mut kv_normed = HiddenStates::zeros(&self.ctx, kv_raw.hidden_dim, kv_raw.seq_len)?;
        ops::rms_norm_batch_into(
            &self.ctx,
            &kv_raw,
            &attention.kv_norm,
            self.config.rms_norm_eps,
            &mut kv_normed,
        );

        let mut q_host = self
            .ctx
            .stream
            .clone_dtoh(&q_raw.data)?
            .into_iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let mut kv_host = self
            .ctx
            .stream
            .clone_dtoh(&kv_normed.data)?
            .into_iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let sink = self
            .ctx
            .stream
            .clone_dtoh(&attention.attn_sink.data)?
            .into_iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let sink_offset = self.config.tp.rank * local_heads;
        ensure!(
            sink_offset + local_heads <= sink.len(),
            "DeepSeek V4 attn_sink len {} cannot cover local heads {} at offset {}",
            sink.len(),
            local_heads,
            sink_offset
        );
        let (rope_cos, rope_sin) = build_rope_cache(
            hidden.seq_len,
            self.config.qk_rope_head_dim,
            self.config.rope_theta,
        );
        for token_idx in 0..hidden.seq_len {
            let kv_start = token_idx * head_dim;
            apply_partial_rope(
                &mut kv_host[kv_start..kv_start + head_dim],
                &rope_cos[token_idx * self.config.qk_rope_head_dim
                    ..(token_idx + 1) * self.config.qk_rope_head_dim],
                &rope_sin[token_idx * self.config.qk_rope_head_dim
                    ..(token_idx + 1) * self.config.qk_rope_head_dim],
                self.config.qk_rope_head_dim,
                1.0,
            );
            for head_idx in 0..local_heads {
                let q_start = token_idx * local_width + head_idx * head_dim;
                let qh = &mut q_host[q_start..q_start + head_dim];
                fixed_rms_norm_in_place(qh, self.config.rms_norm_eps);
                apply_partial_rope(
                    qh,
                    &rope_cos[token_idx * self.config.qk_rope_head_dim
                        ..(token_idx + 1) * self.config.qk_rope_head_dim],
                    &rope_sin[token_idx * self.config.qk_rope_head_dim
                        ..(token_idx + 1) * self.config.qk_rope_head_dim],
                    self.config.qk_rope_head_dim,
                    1.0,
                );
            }
        }

        let mut attn_out = vec![0.0_f32; hidden.seq_len * local_width];
        let scale = 1.0 / (head_dim as f32).sqrt();
        for token_idx in 0..hidden.seq_len {
            let sw_start = (token_idx + 1).saturating_sub(self.config.sliding_window);
            for head_idx in 0..local_heads {
                let q_start = token_idx * local_width + head_idx * head_dim;
                let qh = &q_host[q_start..q_start + head_dim];
                let mut logits = Vec::with_capacity(token_idx + 1 - sw_start);
                for key_idx in sw_start..=token_idx {
                    let key = &kv_host[key_idx * head_dim..(key_idx + 1) * head_dim];
                    logits.push(dot(qh, key) * scale);
                }
                let probs = sink_softmax(&logits, sink[sink_offset + head_idx]);
                let dst_start = token_idx * local_width + head_idx * head_dim;
                let dst = &mut attn_out[dst_start..dst_start + head_dim];
                for (offset, prob) in probs.iter().enumerate() {
                    let key_idx = sw_start + offset;
                    let value = &kv_host[key_idx * head_dim..(key_idx + 1) * head_dim];
                    for col in 0..head_dim {
                        dst[col] += prob * value[col];
                    }
                }
                apply_partial_rope(
                    dst,
                    &rope_cos[token_idx * self.config.qk_rope_head_dim
                        ..(token_idx + 1) * self.config.qk_rope_head_dim],
                    &rope_sin[token_idx * self.config.qk_rope_head_dim
                        ..(token_idx + 1) * self.config.qk_rope_head_dim],
                    self.config.qk_rope_head_dim,
                    -1.0,
                );
            }
        }

        let local_attn = hidden_states_from_f32(&self.ctx, &attn_out, local_width, hidden.seq_len)?;
        let latent = ops::gemm(&self.ctx, &attention.wo_a, &local_attn)?;
        ops::gemm(&self.ctx, &attention.wo_b, &latent)
    }

    fn forward_ffn_layer_stream(
        &self,
        layer_idx: usize,
        stream: &HiddenStates,
        tokens: &[u32],
    ) -> Result<HiddenStates> {
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

        let mhc = gen_mhc_params(
            &self.ctx,
            &layer.hc_ffn,
            stream,
            self.config.hc_mult,
            self.config.hc_eps,
            self.config.hc_sinkhorn_iters,
        )?;
        let sub_in = hc_pre_from_stream(
            &self.ctx,
            stream,
            &mhc.pre,
            self.config.hidden_size,
            self.config.hc_mult,
        )?;
        let mut normed = HiddenStates::zeros(&self.ctx, self.config.hidden_size, stream.seq_len)?;
        ops::rms_norm_batch_into(
            &self.ctx,
            &sub_in,
            &layer.ffn_norm,
            self.config.rms_norm_eps,
            &mut normed,
        );
        let ffn_out = layer.ffn.forward_routed(
            &self.ctx,
            layer_idx,
            &self.config.spec,
            &self.config.ep,
            &normed,
            tokens,
        )?;
        hc_post_to_stream(
            &self.ctx,
            &ffn_out,
            stream,
            &mhc.post,
            &mhc.comb,
            self.config.hidden_size,
            self.config.hc_mult,
        )
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
        if dsv4_gpu_contextual_logits_enabled()? {
            self.compute_top_level_logits(&state.reference_tokens)
        } else {
            self.compute_top_level_logits(&[tokens[tokens.len() - 1]])
        }
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
            wkv: self.load_tp_column_matrix(shards, weight_map, &names.wkv)?,
            wgate: self.load_tp_column_matrix(shards, weight_map, &names.wgate)?,
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
            wq_b: self.load_tp_column_matrix(shards, weight_map, &names.wq_b)?,
            weights_proj: self.load_tp_column_matrix(shards, weight_map, &names.weights_proj)?,
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
    ensure!(
        embeddings.hidden_dim == hidden_size,
        "DeepSeek V4 embedding hidden dim {} does not match hidden_size {}",
        embeddings.hidden_dim,
        hidden_size
    );
    ensure!(hc_mult > 0, "DeepSeek V4 hc_mult must be non-zero");
    let stream_hidden = hidden_size * hc_mult;
    let mut stream = HiddenStates::zeros(ctx, stream_hidden, embeddings.seq_len)?;
    for token_idx in 0..embeddings.seq_len {
        let src_start = token_idx * hidden_size;
        let src = embeddings.data.slice(src_start..src_start + hidden_size);
        for hc_idx in 0..hc_mult {
            let dst_start = token_idx * stream_hidden + hc_idx * hidden_size;
            let mut dst = stream.data.slice_mut(dst_start..dst_start + hidden_size);
            ctx.stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|err| anyhow::anyhow!("DeepSeek V4 initial HC stream copy: {err}"))?;
        }
    }
    Ok(stream)
}

#[cfg(feature = "cuda")]
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
struct MhcParams {
    pre: Vec<f32>,
    post: Vec<f32>,
    comb: Vec<f32>,
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

    let mixes = ops::gemm(ctx, &hc.mix_fn, stream)?;
    let stream_host = ctx.stream.clone_dtoh(&stream.data)?;
    let mixes_host = ctx.stream.clone_dtoh(&mixes.data)?;
    let base_host = ctx.stream.clone_dtoh(&hc.base.data)?;
    let scale_host = ctx.stream.clone_dtoh(&hc.scale.data)?;
    let mut pre = vec![0.0_f32; stream.seq_len * hc_mult];
    let mut post = vec![0.0_f32; stream.seq_len * hc_mult];
    let mut comb = vec![0.0_f32; stream.seq_len * hc_mult * hc_mult];

    for token_idx in 0..stream.seq_len {
        let stream_start = token_idx * stream.hidden_dim;
        let row = &stream_host[stream_start..stream_start + stream.hidden_dim];
        let rsqrt = rms_rsqrt_bf16(row, hc_eps);
        let mix_start = token_idx * mixes.hidden_dim;
        let token_mixes = &mixes_host[mix_start..mix_start + mixes.hidden_dim];

        for hc_idx in 0..hc_mult {
            let pre_mix = token_mixes[hc_idx].to_f32() * rsqrt;
            let post_mix = token_mixes[hc_mult + hc_idx].to_f32() * rsqrt;
            pre[token_idx * hc_mult + hc_idx] =
                sigmoid(scale_host[0].to_f32() * pre_mix + base_host[hc_idx].to_f32()) + hc_eps;
            post[token_idx * hc_mult + hc_idx] = 2.0
                * sigmoid(scale_host[1].to_f32() * post_mix + base_host[hc_mult + hc_idx].to_f32());
        }

        let mut raw = vec![0.0_f32; hc_mult * hc_mult];
        for row_idx in 0..hc_mult {
            for col_idx in 0..hc_mult {
                let idx = row_idx * hc_mult + col_idx;
                let mix = token_mixes[2 * hc_mult + idx].to_f32() * rsqrt;
                raw[idx] = scale_host[2].to_f32() * mix + base_host[2 * hc_mult + idx].to_f32();
            }
        }
        row_softmax_plus_eps(&mut raw, hc_mult, hc_eps);
        column_normalize(&mut raw, hc_mult, hc_eps);
        for _ in 1..hc_sinkhorn_iters {
            row_normalize(&mut raw, hc_mult, hc_eps);
            column_normalize(&mut raw, hc_mult, hc_eps);
        }
        let dst = token_idx * hc_mult * hc_mult;
        comb[dst..dst + hc_mult * hc_mult].copy_from_slice(&raw);
    }

    Ok(MhcParams { pre, post, comb })
}

#[cfg(feature = "cuda")]
fn hc_pre_from_stream(
    ctx: &DeviceContext,
    stream: &HiddenStates,
    pre: &[f32],
    hidden_size: usize,
    hc_mult: usize,
) -> Result<HiddenStates> {
    ensure!(
        stream.hidden_dim == hidden_size * hc_mult,
        "DeepSeek V4 HC pre stream dim {} does not match hidden_size {} * hc_mult {}",
        stream.hidden_dim,
        hidden_size,
        hc_mult
    );
    ensure!(
        pre.len() == stream.seq_len * hc_mult,
        "DeepSeek V4 HC pre len {} does not match seq_len {} * hc_mult {}",
        pre.len(),
        stream.seq_len,
        hc_mult
    );
    let mut out = HiddenStates::zeros(ctx, hidden_size, stream.seq_len)?;
    for token_idx in 0..stream.seq_len {
        for hc_idx in 0..hc_mult {
            let lane = extract_hc_lane(ctx, stream, token_idx, hc_idx, hidden_size)?;
            ops::add_scaled_row_into(
                ctx,
                &lane,
                &mut out,
                token_idx,
                pre[token_idx * hc_mult + hc_idx],
            )?;
        }
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
fn hc_post_to_stream(
    ctx: &DeviceContext,
    new_x: &HiddenStates,
    residual: &HiddenStates,
    post: &[f32],
    comb: &[f32],
    hidden_size: usize,
    hc_mult: usize,
) -> Result<HiddenStates> {
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
        post.len() == residual.seq_len * hc_mult
            && comb.len() == residual.seq_len * hc_mult * hc_mult,
        "DeepSeek V4 HC post weights mismatch: post={} comb={} seq_len={} hc_mult={}",
        post.len(),
        comb.len(),
        residual.seq_len,
        hc_mult
    );

    let mut out = HiddenStates::zeros(ctx, hidden_size * hc_mult, residual.seq_len)?;
    for token_idx in 0..residual.seq_len {
        let token_new = extract_hidden_token_with_width(ctx, new_x, token_idx, hidden_size)?;
        for dst_hc in 0..hc_mult {
            let segment_offset = dst_hc * hidden_size;
            ops::add_scaled_row_segment_into(
                ctx,
                &token_new,
                &mut out,
                token_idx,
                segment_offset,
                post[token_idx * hc_mult + dst_hc],
            )?;
            for src_hc in 0..hc_mult {
                let residual_lane = extract_hc_lane(ctx, residual, token_idx, src_hc, hidden_size)?;
                ops::add_scaled_row_segment_into(
                    ctx,
                    &residual_lane,
                    &mut out,
                    token_idx,
                    segment_offset,
                    comb[(token_idx * hc_mult + dst_hc) * hc_mult + src_hc],
                )?;
            }
        }
    }
    Ok(out)
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
    let stream_row_host = ctx.stream.clone_dtoh(&stream_row.data)?;
    let rsqrt = rms_rsqrt_bf16(&stream_row_host, hc_eps);
    let mixes_host = ctx.stream.clone_dtoh(&mixes.data)?;
    let base_host = ctx.stream.clone_dtoh(&head_hc.base.data)?;
    let scale_host = ctx.stream.clone_dtoh(&head_hc.scale.data)?;
    let scale = scale_host[0].to_f32();
    let pre = (0..hc_mult)
        .map(|idx| {
            sigmoid(scale * mixes_host[idx].to_f32() * rsqrt + base_host[idx].to_f32()) + hc_eps
        })
        .collect::<Vec<_>>();

    let mut out = HiddenStates::zeros(ctx, hidden_size, 1)?;
    for (hc_idx, weight) in pre.into_iter().enumerate() {
        let lane = extract_hc_lane(ctx, stream, token_idx, hc_idx, hidden_size)?;
        ops::add_scaled_row_into(ctx, &lane, &mut out, 0, weight)?;
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
    let mut out = HiddenStates::zeros(ctx, width, 1)?;
    let start = token_idx * width;
    let src = hidden.data.slice(start..start + width);
    ctx.stream
        .memcpy_dtod(&src, &mut out.data)
        .map_err(|err| anyhow::anyhow!("DeepSeek V4 token extract copy: {err}"))?;
    Ok(out)
}

#[cfg(feature = "cuda")]
fn extract_hc_lane(
    ctx: &DeviceContext,
    stream: &HiddenStates,
    token_idx: usize,
    hc_idx: usize,
    hidden_size: usize,
) -> Result<HiddenStates> {
    let start = token_idx * stream.hidden_dim + hc_idx * hidden_size;
    let mut out = HiddenStates::zeros(ctx, hidden_size, 1)?;
    let src = stream.data.slice(start..start + hidden_size);
    ctx.stream
        .memcpy_dtod(&src, &mut out.data)
        .map_err(|err| anyhow::anyhow!("DeepSeek V4 HC lane extract copy: {err}"))?;
    Ok(out)
}

#[cfg(feature = "cuda")]
fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[cfg(feature = "cuda")]
fn rms_rsqrt_bf16(values: &[bf16], eps: f32) -> f32 {
    let mean_square = values
        .iter()
        .map(|value| value.to_f32().powi(2))
        .sum::<f32>()
        / values.len().max(1) as f32;
    1.0 / (mean_square + eps).sqrt()
}

#[cfg(feature = "cuda")]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(lhs, rhs)| lhs * rhs).sum()
}

#[cfg(feature = "cuda")]
fn sink_softmax(logits: &[f32], sink: f32) -> Vec<f32> {
    let max = logits.iter().copied().fold(sink, f32::max);
    let denom = logits.iter().map(|value| (*value - max).exp()).sum::<f32>() + (sink - max).exp();
    logits
        .iter()
        .map(|value| (*value - max).exp() / denom)
        .collect()
}

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
fn row_softmax_plus_eps(raw: &mut [f32], n: usize, eps: f32) {
    for row in 0..n {
        let start = row * n;
        let probs = softmax(&raw[start..start + n]);
        for col in 0..n {
            raw[start + col] = probs[col] + eps;
        }
    }
}

#[cfg(feature = "cuda")]
fn row_normalize(raw: &mut [f32], n: usize, eps: f32) {
    for row in 0..n {
        let start = row * n;
        let sum = raw[start..start + n].iter().sum::<f32>() + eps;
        for col in 0..n {
            raw[start + col] /= sum;
        }
    }
}

#[cfg(feature = "cuda")]
fn column_normalize(raw: &mut [f32], n: usize, eps: f32) {
    for col in 0..n {
        let mut sum = eps;
        for row in 0..n {
            sum += raw[row * n + col];
        }
        for row in 0..n {
            raw[row * n + col] /= sum;
        }
    }
}

#[cfg(feature = "cuda")]
fn fixed_rms_norm_in_place(values: &mut [f32], eps: f32) {
    let mean_square = values.iter().map(|value| value.powi(2)).sum::<f32>() / values.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    for value in values {
        *value *= scale;
    }
}

#[cfg(feature = "cuda")]
fn build_rope_cache(seq: usize, dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    if dim == 0 {
        return (Vec::new(), Vec::new());
    }
    let half = dim / 2;
    let inv_freq = (0..half)
        .map(|i| 1.0_f32 / base.powf((2 * i) as f32 / dim as f32))
        .collect::<Vec<_>>();
    let mut cos = vec![0.0_f32; seq * dim];
    let mut sin = vec![0.0_f32; seq * dim];
    for pos in 0..seq {
        for i in 0..half {
            let value = pos as f32 * inv_freq[i];
            let c = value.cos();
            let s = value.sin();
            cos[pos * dim + i] = c;
            cos[pos * dim + i + half] = c;
            sin[pos * dim + i] = s;
            sin[pos * dim + i + half] = s;
        }
    }
    (cos, sin)
}

#[cfg(feature = "cuda")]
fn apply_partial_rope(row: &mut [f32], cos: &[f32], sin: &[f32], rope_dim: usize, sign: f32) {
    let half = rope_dim / 2;
    for i in 0..half {
        let a = row[i];
        let b = row[i + half];
        let s = sign * sin[i];
        row[i] = a * cos[i] - b * s;
        row[i + half] = b * cos[i] + a * s;
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
        let rsqrt = 1.0_f32 / ((1.0_f32 + 4.0 + 9.0 + 25.0) / 4.0 + 1.0e-6).sqrt();
        assert_close(mhc.pre[0], sigmoid(rsqrt) + 1.0e-6, 0.003);
        assert_close(mhc.pre[1], 0.5 + 1.0e-6, 0.003);
        assert_close(mhc.post[0], 1.0, 0.003);
        assert_close(mhc.post[1], 2.0 * sigmoid(2.0 * rsqrt), 0.003);
        for col in 0..2 {
            let sum = mhc.comb[col] + mhc.comb[2 + col];
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

        let pre = hc_pre_from_stream(&ctx, &stream, &[0.25, 0.5], 2, 2)?;
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
            &[0.1, 0.2],
            &[1.0, 0.0, 0.25, 0.75],
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
                    shared_experts: None,
                },
            }],
            config,
            ctx,
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
}
