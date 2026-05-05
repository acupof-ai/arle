//! DeepSeek model weights.
//!
//! Mirrors the `Qwen3Model` shape: pool-shared immutable weights with per-
//! request mutable state living in [`super::state::DeepseekState`]. The
//! tensor name layout follows
//! [`deepseek_spec::DeepSeekConfig::layer_tensor_names`] and
//! [`deepseek_spec::DeepSeekConfig::shard_for_global_tensor`].

use anyhow::Result;

use super::config::DeepseekRuntimeConfig;
#[cfg(feature = "cuda")]
use super::mla::MlaAttention;
#[cfg(feature = "cuda")]
use super::mlp::DenseMlp;
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec};
use deepseek_spec::DeepSeekConfig;

/// One DeepSeek transformer layer: input norm → MLA → post norm → dense MLP.
///
/// MoE wiring (the `DeepSeekMoeTensorNames` branch in
/// `DeepSeekConfig::layer_tensor_names`) is deferred until SKU-B; the nano
/// fixture stays on the dense MLP path on every layer.
#[cfg(feature = "cuda")]
pub(super) struct DeepseekLayer {
    pub(super) input_layernorm: DeviceVec,
    pub(super) attention: MlaAttention,
    pub(super) post_attention_layernorm: DeviceVec,
    pub(super) mlp: DenseMlp,
}

/// DeepSeek model — weights and config only. Mutable per-slot state lives in
/// [`super::state::DeepseekState`].
pub struct DeepseekModel {
    pub(super) config: DeepseekRuntimeConfig,
    #[cfg(feature = "cuda")]
    pub(super) ctx: DeviceContext,
    #[cfg(feature = "cuda")]
    pub(super) embed_tokens: DeviceMatrix,
    /// `None` when `tie_word_embeddings == true` (nano / SKU-A / SKU-B all
    /// tie); the sampler reads through `embed_tokens` in that case.
    #[cfg(feature = "cuda")]
    pub(super) lm_head: Option<DeviceMatrix>,
    #[cfg(feature = "cuda")]
    pub(super) norm: DeviceVec,
    #[cfg(feature = "cuda")]
    pub(super) layers: Vec<DeepseekLayer>,
}

impl DeepseekModel {
    /// Read-only view of the runtime config (architecture + serving knobs).
    pub fn config(&self) -> &DeepseekRuntimeConfig {
        &self.config
    }

    /// Read-only view of the underlying spec config.
    pub fn spec(&self) -> &DeepSeekConfig {
        &self.config.spec
    }

    /// Whether layer `idx` uses the dense SwiGLU MLP path (true for the nano
    /// fixture and SKU-A on every layer; true for SKU-B only on the
    /// `first_k_dense_replace` prefix). Inverse of
    /// `DeepSeekConfig::is_moe_layer`.
    pub fn is_dense_layer(&self, idx: usize) -> bool {
        !self.config.spec.is_moe_layer(idx)
    }
}

#[cfg(feature = "cuda")]
impl DeepseekModel {
    /// Allocate a model from a spec config without loading any weights.
    ///
    /// Stub: returns `todo!()` until safetensors loading and the MLA + dense
    /// MLP weight allocation paths land. The intent is to mirror
    /// `Qwen3Model::from_safetensors_with_runtime` once kernels exist.
    pub fn from_config(_config: DeepseekRuntimeConfig) -> Result<Self> {
        todo!("DeepSeek weight allocation — wires alongside MLA kernel")
    }

    /// Load a checkpoint by safetensors path, validating tensor names against
    /// `DeepSeekConfig::layer_tensor_names` / `mtp_tensor_names` /
    /// `shard_for_global_tensor`.
    pub fn from_safetensors(_path: &str, _config: DeepseekRuntimeConfig) -> Result<Self> {
        todo!("DeepSeek safetensors loader — wires alongside MLA kernel")
    }
}

/// Walk every tensor name produced by [`DeepSeekConfig`]'s spec helpers for the
/// given config, asserting each one is recognized by exactly one shard rule
/// (per-layer, MoE expert, MTP, or global). Substrate plan §5 names this as a
/// build-time sanity check; we run it as a unit test so a mis-prefixed tensor
/// in the spec (or a missing branch in `shard_for`) fails fast at scaffold
/// build time rather than at safetensors load time.
#[cfg(test)]
fn validate_tensor_name_coverage(config: &DeepSeekConfig) -> std::result::Result<(), String> {
    use deepseek_spec::DeepSeekMlpTensorNames;

    let mut tensors: Vec<String> = Vec::new();
    let mut shard_lookup: Vec<Box<dyn Fn(&str) -> bool>> = Vec::new();

    tensors.push(config.embed_tokens_tensor_name().to_string());
    tensors.push(config.norm_tensor_name().to_string());
    if !config.tie_word_embeddings {
        tensors.push("lm_head.weight".to_string());
    }
    let global_cfg = config.clone();
    shard_lookup.push(Box::new(move |name| {
        global_cfg.shard_for_global_tensor(name).is_some()
    }));

    for layer_idx in 0..config.num_hidden_layers {
        let layer_names = config.layer_tensor_names(layer_idx);
        tensors.push(layer_names.input_layernorm.clone());
        tensors.push(layer_names.post_attention_layernorm.clone());
        tensors.push(layer_names.attention.kv_a_proj_with_mqa.clone());
        tensors.push(layer_names.attention.kv_a_layernorm.clone());
        tensors.push(layer_names.attention.kv_b_proj.clone());
        tensors.push(layer_names.attention.o_proj.clone());
        if config.q_lora_rank.is_some() {
            tensors.push(layer_names.attention.q_a_proj.clone());
            tensors.push(layer_names.attention.q_a_layernorm.clone());
            tensors.push(layer_names.attention.q_b_proj.clone());
        } else {
            tensors.push(layer_names.attention.q_proj.clone());
        }
        match &layer_names.mlp {
            DeepSeekMlpTensorNames::Dense(dense) => {
                tensors.push(dense.gate_proj.clone());
                tensors.push(dense.up_proj.clone());
                tensors.push(dense.down_proj.clone());
            }
            DeepSeekMlpTensorNames::Moe(moe) => {
                tensors.push(moe.router.clone());
                if let Some(shared) = &moe.shared_experts {
                    tensors.push(shared.gate_proj.clone());
                    tensors.push(shared.up_proj.clone());
                    tensors.push(shared.down_proj.clone());
                }
                for expert_idx in 0..config.num_experts {
                    let expert = moe.expert(expert_idx);
                    tensors.push(expert.gate_proj.clone());
                    tensors.push(expert.up_proj.clone());
                    tensors.push(expert.down_proj.clone());
                }
            }
        }
        let layer_cfg = config.clone();
        shard_lookup.push(Box::new(move |name| {
            layer_cfg
                .layer_tensor_names(layer_idx)
                .shard_for(name)
                .is_some()
        }));
    }

    if config.has_mtp() {
        for mtp_idx in 0..config.num_nextn_predict_layers {
            let mtp = config.mtp_tensor_names(mtp_idx);
            tensors.push(mtp.embed_tokens);
            tensors.push(mtp.enorm);
            tensors.push(mtp.hnorm);
            tensors.push(mtp.eh_proj);
            tensors.push(mtp.shared_head_norm);
            tensors.push(mtp.lm_head);
            // The inner MTP layer reuses the per-layer shard helpers via
            // `mtp.layer.shard_for`, so add one closure per MTP block.
            let mtp_cfg = config.clone();
            shard_lookup.push(Box::new(move |name| {
                mtp_cfg
                    .mtp_tensor_names(mtp_idx)
                    .layer
                    .shard_for(name)
                    .is_some()
            }));
        }
    }

    for tensor in &tensors {
        if !shard_lookup.iter().any(|f| f(tensor)) {
            return Err(format!(
                "tensor `{tensor}` is not covered by any shard rule"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nano_tensor_names_fully_covered() {
        let cfg = DeepSeekConfig::nano();
        validate_tensor_name_coverage(&cfg).expect("nano tensor coverage");
    }

    #[test]
    fn nano_runtime_config_round_trips() {
        let runtime = DeepseekRuntimeConfig::from_spec(DeepSeekConfig::nano());
        assert_eq!(runtime.num_hidden_layers, 2);
        assert_eq!(runtime.kv_lora_rank, 64);
        assert_eq!(runtime.qk_rope_head_dim, 16);
        assert!(runtime.tie_word_embeddings);
        assert!(runtime.enable_cuda_graph);
    }
}
