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
            let stream = initial_hc_stream_from_embeddings(
                &self.ctx,
                &embeddings,
                self.config.hidden_size,
                self.config.hc_mult,
            )?;
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
    use half::bf16;

    fn bf16_vec(values: &[f32]) -> Vec<bf16> {
        values.iter().map(|&value| bf16::from_f32(value)).collect()
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
}
