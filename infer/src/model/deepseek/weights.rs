//! DeepSeek V4 model weights.
//!
//! The runtime target is the local `DeepseekV4ForCausalLM` checkpoint at
//! `infer/models/dsv4-mini-1B-init/`. Infer-side DeepSeek wiring uses
//! [`deepseek_spec::DeepSeekV4Config`] and its HF tensor-name contract only.

use std::path::Path;

use anyhow::{Result, ensure};
use log::info;

use super::config::DeepseekRuntimeConfig;
#[cfg(feature = "cuda")]
use super::mla::DeepseekV4Attention;
#[cfg(feature = "cuda")]
use super::mlp::DeepseekV4MoeBlock;
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec};
use deepseek_spec::DeepSeekV4Config;

use crate::deepseek_v4_manifest::{
    DeepseekV4CheckpointManifest, validate_deepseek_v4_checkpoint_manifest,
};
#[cfg(feature = "cuda")]
use crate::model::common;
#[cfg(feature = "cuda")]
use crate::weight_loader::{load_tensor_1d, load_tensor_2d};

/// Hyper-connection tensors used by the V4 layer/head mixers.
#[cfg(feature = "cuda")]
#[allow(dead_code)] // populated once the Phase 2A loader allocates tensors
pub(super) struct DeepseekV4HyperConnection {
    pub(super) base: DeviceVec,
    pub(super) mix_fn: DeviceVec,
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
        let (mmaps, weight_map) = common::load_safetensors(path, false)?;
        let shards = common::deserialize_shards(&mmaps)?;
        let names = model.config.spec.tensor_names();
        let vocab_size = model.config.vocab_size;
        let hidden_size = model.config.hidden_size;

        let embed_tokens = load_tensor_2d(&model.ctx, &shards, &weight_map, names.embed_tokens())?;
        ensure!(
            embed_tokens.rows == vocab_size && embed_tokens.cols == hidden_size,
            "DeepSeek V4 embed.weight shape [{}, {}] does not match vocab_size={} hidden_size={}",
            embed_tokens.rows,
            embed_tokens.cols,
            vocab_size,
            hidden_size
        );
        let lm_head = load_tensor_2d(&model.ctx, &shards, &weight_map, names.lm_head())?;
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

        let summary = model.config.spec.attention_operator_summary();
        info!(
            "DeepSeek V4 Phase 2A.1 CUDA top-level logits smoke loaded: sliding_window_layers={} \
             csa_layers={} hca_layers={} vocab_size={} hidden_size={} ep_rank={}/{} experts_per_rank={}",
            summary.sliding_window_layers,
            summary.csa_layers,
            summary.hca_layers,
            model.config.vocab_size,
            model.config.hidden_size,
            model.config.ep.rank,
            model.config.ep.world_size,
            model.config.ep.experts_per_rank
        );
        Ok(model)
    }

    pub(super) fn compute_top_level_logits(&self, tokens: &[u32]) -> Result<Option<DeviceVec>> {
        let (Some(embed_tokens), Some(norm), Some(lm_head)) = (
            self.embed_tokens.as_ref(),
            self.norm.as_ref(),
            self.lm_head.as_ref(),
        ) else {
            return Ok(None);
        };
        let hidden =
            common::get_embeddings_batch(&self.ctx, embed_tokens, tokens, self.config.hidden_size)?;
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
}
