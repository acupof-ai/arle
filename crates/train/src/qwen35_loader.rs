//! HuggingFace-format safetensors loader for [`Qwen35Model`].
//!
//! ## Schema coverage
//!
//! - **Qwen3.5 / Qwen3.6** (`model.language_model.*`, gated `q_proj`): natively
//!   supported.
//! - **Vanilla Qwen3** (`model.*`, plain `q_proj`): partially supported — the
//!   loader remaps `model.*` → `model.language_model.*` and synthesizes
//!   `linear_*` fields. A plain `q_proj` checkpoint fails the gated-attention
//!   shape check; see [`load_qwen35_from_hf_dir`].
//!
//! Quantized checkpoint support is deliberately narrow: CUDA LoRA-student loads
//! accept frozen FP8 E4M3 block-scaled base weights with a matching
//! `*.weight_scale_inv` side tensor. Teacher, trainable-base, and CPU loads
//! reject quantized weights.
//!
//! ## Independence from the `infer` crate
//!
//! Train must not depend on `infer` at runtime (OPD-only pivot contract), so
//! this file re-implements shard discovery + BF16/F16 widening; safetensors
//! parsing goes through the workspace `safetensors` crate.

use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
};

use autograd::{Device, Tensor, TensorId, TensorStore};
use half::{bf16, f16};
use memmap2::Mmap;
use qwen35_spec::{LayerType, Qwen35Config, Qwen35ConfigError};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::Deserialize;
use thiserror::Error;

use crate::{
    lora::{LoraConfig, LoraTargetSet},
    qwen35::{Qwen35Error, Qwen35Model},
};

/// One frozen base projection's resident FP8 block-scaled device pointers,
/// borrowed from a co-resident infer-cuda engine (`--share-frozen-base`).
///
/// Backend-agnostic: train must not depend on `infer` (OPD-only pivot), so
/// `train_cli` maps infer-api's `SharedFp8BaseProjection` into this struct.
#[derive(Debug, Clone)]
pub struct SharedFrozenBaseEntry {
    pub layer_idx: usize,
    pub proj_suffix: String,
    pub weight_ptr: u64,
    pub scale_ptr: u64,
    pub rows: usize,
    pub cols: usize,
    pub block_m: usize,
    pub block_k: usize,
}

impl SharedFrozenBaseEntry {
    fn matches(&self, train_name: &str) -> bool {
        train_name.ends_with(&format!(
            ".layers.{}.{}.weight",
            self.layer_idx, self.proj_suffix
        ))
    }
}

/// `None`/empty = no sharing — every frozen FP8 base uploads its own copy.
pub type SharedFrozenBaseTable<'a> = &'a [SharedFrozenBaseEntry];

#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error(
        "failed to read {path}: {source}. Hint: verify the OPD checkpoint directory contains the expected file and is readable."
    )]
    ReadFile {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(
        "failed to open safetensors shard {path}: {source}. Hint: verify model.safetensors or every shard listed in model.safetensors.index.json exists and is readable."
    )]
    OpenShard {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(
        "failed to memory-map safetensors shard {path}: {source}. Hint: verify the checkpoint file is local, complete, and not being modified while OPD loads it."
    )]
    MmapShard {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(
        "json: {0}. Hint: validate config.json or model.safetensors.index.json \
         is valid JSON in the OPD checkpoint directory."
    )]
    Json(#[from] serde_json::Error),
    #[error(
        "safetensors: {0}. Hint: verify each shard is a complete local \
         safetensors file and matches model.safetensors.index.json."
    )]
    Safetensors(String),
    #[error(
        "config: {0}. Hint: verify config.json uses a supported Qwen3/Qwen3.5 \
         schema and matches the checkpoint tensors."
    )]
    Config(#[from] Qwen35ConfigError),
    #[error(
        "model: {0}. Hint: verify config.json is compatible with the train-side \
         Qwen35Model schema before running OPD."
    )]
    Model(#[from] Qwen35Error),
    #[error("shape mismatch for {name}: model expects {expected:?}, safetensors has {got:?}{hint}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
        hint: String,
    },
    #[error(
        "missing tensor {0} in safetensors (and no fallback rule applies). \
         Hint: verify the checkpoint is complete for its config, \
         model.safetensors.index.json points at every shard, and the directory \
         uses HF-compatible Qwen3.5/Qwen3.6 tensor names."
    )]
    MissingTensor(String),
    #[error(
        "unsupported dtype {0:?} for {1}. Hint: OPD loader accepts F32/BF16/F16, plus CUDA LoRA-student frozen FP8 E4M3 weights with matching *.weight_scale_inv side tensors."
    )]
    UnsupportedDtype(Dtype, String),
    #[error(
        "autograd: {0}. Hint: report this with the checkpoint path, config.json, \
         and OPD loader follow-up tranche context."
    )]
    Autograd(#[from] autograd::AutogradError),
    #[error("loader: {0}")]
    Custom(String),
}

pub type Result<T> = std::result::Result<T, LoaderError>;

/// Union of vanilla Qwen3 and Qwen3.5/3.6 nested `text_config` fields.
/// `linear_*` and `layer_types` optional (vanilla Qwen3 omits them).
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Qwen35HfConfig {
    pub hidden_size: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(alias = "num_kv_heads")]
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub layer_types: Option<Vec<LayerType>>,
    #[serde(default)]
    pub linear_conv_kernel_dim: Option<usize>,
    #[serde(default)]
    pub linear_key_head_dim: Option<usize>,
    #[serde(default)]
    pub linear_num_key_heads: Option<usize>,
    #[serde(default)]
    pub linear_num_value_heads: Option<usize>,
    #[serde(default)]
    pub linear_value_head_dim: Option<usize>,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_decoder_sparse_step")]
    pub decoder_sparse_step: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default)]
    moe_config: Option<Qwen35HfMoeConfig>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
struct Qwen35HfMoeConfig {
    #[serde(default)]
    num_experts: usize,
    #[serde(default)]
    num_experts_per_tok: usize,
    #[serde(default = "default_decoder_sparse_step")]
    decoder_sparse_step: usize,
    #[serde(default)]
    moe_intermediate_size: usize,
    #[serde(default)]
    shared_expert_intermediate_size: usize,
    #[serde(default = "default_norm_topk_prob")]
    norm_topk_prob: bool,
    #[serde(default)]
    mlp_only_layers: Vec<usize>,
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

fn default_tie_word_embeddings() -> bool {
    false
}

fn default_decoder_sparse_step() -> usize {
    1
}

fn default_norm_topk_prob() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfSchema {
    /// `model.layers.N.*`, plain `q_proj`. Qwen3-0.6B/1.7B/4B.
    Qwen3,
    /// `model.language_model.layers.N.*`, gated `q_proj`. Qwen3.5/3.6.
    Qwen35,
}

impl Qwen35HfConfig {
    pub fn from_json_str(content: &str) -> Result<(Self, HfSchema)> {
        let value: serde_json::Value = serde_json::from_str(content)?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &serde_json::Value) -> Result<(Self, HfSchema)> {
        let (text, schema) = match value.get("text_config") {
            Some(text) => (text.clone(), HfSchema::Qwen35),
            None => (value.clone(), HfSchema::Qwen3),
        };
        let text = if schema == HfSchema::Qwen35 {
            merge_token_ids(text, value)
        } else {
            text
        };

        let mut config: Qwen35HfConfig = serde_json::from_value(text.clone())?;
        config.merge_nested_moe_config();

        if let Some(rope) = text.get("rope_parameters") {
            if let Some(theta) = rope.get("rope_theta").and_then(serde_json::Value::as_f64) {
                config.rope_theta = theta as f32;
            }
            if let Some(prf) = rope
                .get("partial_rotary_factor")
                .and_then(serde_json::Value::as_f64)
            {
                config.partial_rotary_factor = prf as f32;
            }
        }

        Ok((config, schema))
    }

    fn merge_nested_moe_config(&mut self) {
        let Some(nested) = self.moe_config.take() else {
            return;
        };
        if nested.num_experts != 0 {
            self.num_experts = nested.num_experts;
        }
        if nested.num_experts_per_tok != 0 {
            self.num_experts_per_tok = nested.num_experts_per_tok;
        }
        if nested.decoder_sparse_step != default_decoder_sparse_step() {
            self.decoder_sparse_step = nested.decoder_sparse_step;
        }
        if nested.moe_intermediate_size != 0 {
            self.moe_intermediate_size = nested.moe_intermediate_size;
        }
        if nested.shared_expert_intermediate_size != 0 {
            self.shared_expert_intermediate_size = nested.shared_expert_intermediate_size;
        }
        if !nested.norm_topk_prob {
            self.norm_topk_prob = nested.norm_topk_prob;
        }
        if !nested.mlp_only_layers.is_empty() {
            self.mlp_only_layers = nested.mlp_only_layers;
        }
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<(Self, HfSchema)> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(|source| LoaderError::ReadFile {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_json_str(&content)
    }

    /// Missing `linear_*` fields default to the dense attention shape — only
    /// consulted when a layer has `LayerType::LinearAttention`.
    pub fn to_qwen35_config(&self) -> Result<Qwen35Config> {
        let eos = self.eos_token_id.unwrap_or(0);
        let num_layers = self.num_hidden_layers;
        let layer_types = match self.layer_types.clone() {
            Some(types) if types.len() == num_layers => types,
            Some(types) => {
                return Err(LoaderError::Custom(format!(
                    "layer_types length {} != num_hidden_layers {num_layers}. \
                     Hint: fix config.json text_config.layer_types so it has \
                     exactly one entry per decoder layer.",
                    types.len()
                )));
            }
            None => vec![LayerType::FullAttention; num_layers],
        };
        let head_dim = self.head_dim;
        let partial_rotary_factor = self.partial_rotary_factor;
        let rotary_dim = (head_dim as f32 * partial_rotary_factor) as usize;

        let cfg = Qwen35Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: num_layers,
            vocab_size: self.vocab_size,
            rms_norm_eps: self.rms_norm_eps,
            stop_token_ids: vec![eos],
            bos_token_id: self.bos_token_id,
            eos_token_id: eos,
            tie_word_embeddings: self.tie_word_embeddings,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim,
            linear_num_key_heads: self
                .linear_num_key_heads
                .unwrap_or(self.num_attention_heads),
            linear_key_head_dim: self.linear_key_head_dim.unwrap_or(head_dim),
            linear_num_value_heads: self
                .linear_num_value_heads
                .unwrap_or(self.num_attention_heads),
            linear_value_head_dim: self.linear_value_head_dim.unwrap_or(head_dim),
            linear_conv_kernel_dim: self.linear_conv_kernel_dim.unwrap_or(4),
            rope_theta: self.rope_theta,
            rope_scaling: None,
            partial_rotary_factor,
            rotary_dim,
            rope_cache_len_hint: Some(self.max_position_embeddings.unwrap_or(32_768)),
            layer_types,
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            decoder_sparse_step: self.decoder_sparse_step,
            moe_intermediate_size: self.moe_intermediate_size,
            shared_expert_intermediate_size: self.shared_expert_intermediate_size,
            norm_topk_prob: self.norm_topk_prob,
            mlp_only_layers: self.mlp_only_layers.clone(),
            full_attn_gated: true,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

fn merge_token_ids(mut text: serde_json::Value, parent: &serde_json::Value) -> serde_json::Value {
    if let Some(obj) = text.as_object_mut() {
        for key in ["eos_token_id", "bos_token_id"] {
            if obj.get(key).is_none()
                && let Some(v) = parent.get(key)
            {
                obj.insert(key.to_string(), v.clone());
            }
        }
    }
    text
}

struct ShardFile {
    mmap: Mmap,
}

impl ShardFile {
    fn open(path: &Path) -> Result<Self> {
        let file = fs::File::open(path).map_err(|source| LoaderError::OpenShard {
            path: path.to_path_buf(),
            source,
        })?;
        // SAFETY: weights file is not mutated during loading.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|source| LoaderError::MmapShard {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Self { mmap })
    }

    fn safetensors(&self) -> Result<SafeTensors<'_>> {
        SafeTensors::deserialize(&self.mmap[..])
            .map_err(|err| LoaderError::Safetensors(err.to_string()))
    }
}

fn discover_shards(dir: &Path) -> Result<Vec<PathBuf>> {
    let single = dir.join("model.safetensors");
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        let content = fs::read_to_string(&index).map_err(|source| LoaderError::ReadFile {
            path: index.clone(),
            source,
        })?;
        let manifest: serde_json::Value = serde_json::from_str(&content)?;
        let weight_map = manifest
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                LoaderError::Custom(format!(
                    "{} missing weight_map object. Hint: regenerate \
                     model.safetensors.index.json or provide a single \
                     model.safetensors shard in the checkpoint directory.",
                    index.display()
                ))
            })?;
        let mut files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect();
        files.sort();
        files.dedup();
        return Ok(files.into_iter().map(|name| dir.join(name)).collect());
    }
    if single.is_file() {
        return Ok(vec![single]);
    }
    Err(LoaderError::Custom(format!(
        "no safetensors shards found under {}. Hint: pass a local HF/ModelScope \
         checkpoint directory containing model.safetensors or \
         model.safetensors.index.json.",
        dir.display()
    )))
}

/// For `HfSchema::Qwen35` this is identity. For `HfSchema::Qwen3` we strip
/// the `language_model.` segment. The lm_head case is handled by
/// [`hf_lm_head_candidates`].
fn train_name_to_hf(train_name: &str, schema: HfSchema) -> String {
    match schema {
        HfSchema::Qwen35 => train_name.to_owned(),
        HfSchema::Qwen3 => {
            const PREFIX: &str = "model.language_model.";
            if let Some(rest) = train_name.strip_prefix(PREFIX) {
                format!("model.{rest}")
            } else {
                train_name.to_owned()
            }
        }
    }
}

/// LM head fallback list. Vanilla Qwen3 ships `lm_head.weight` (not under
/// `model.`). When `tie_word_embeddings` is true the embedding row is reused
/// and the LM head tensor may be absent.
fn hf_lm_head_candidates(schema: HfSchema) -> &'static [&'static str] {
    match schema {
        HfSchema::Qwen35 => &["lm_head.weight", "model.language_model.lm_head.weight"],
        HfSchema::Qwen3 => &["lm_head.weight", "model.lm_head.weight"],
    }
}

fn hf_candidates_for_train_name(train_name: &str, schema: HfSchema) -> Vec<String> {
    if train_name.ends_with("lm_head.weight") {
        hf_lm_head_candidates(schema)
            .iter()
            .map(|s| (*s).to_owned())
            .collect()
    } else {
        vec![train_name_to_hf(train_name, schema)]
    }
}

fn dtype_to_f32(view: &TensorView<'_>, name: &str) -> Result<Vec<f32>> {
    let bytes = view.data();
    match view.dtype() {
        Dtype::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        Dtype::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        Dtype::F16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        other => Err(LoaderError::UnsupportedDtype(other, name.to_owned())),
    }
}

fn dtype_to_bf16_bits(view: &TensorView<'_>, name: &str) -> Result<Vec<u16>> {
    if view.dtype() != Dtype::BF16 {
        return Err(LoaderError::UnsupportedDtype(view.dtype(), name.to_owned()));
    }
    let bytes = view.data();
    if !bytes.len().is_multiple_of(2) {
        return Err(LoaderError::Custom(format!(
            "{name} BF16 byte length {} is not divisible by 2",
            bytes.len()
        )));
    }
    // The old `chunks_exact(2).map(u16::from_le_bytes).collect()` ran one scalar
    // write per element — single-threaded over the giant frozen-base tensors
    // (embed_tokens + lm_head each ~1.27 B u16 on 27B), so the student load
    // spun minutes on one core. Bulk-copy the bytes into the `Vec<u16>` (one
    // memcpy), then byte-swap only on big-endian hosts.
    let mut out = vec![0u16; bytes.len() / 2];
    {
        // SAFETY: `out` owns `out.len()` u16 = `bytes.len()` contiguous bytes;
        // a `[u8]` view over its buffer is valid for that exact length,
        // properly aligned (u16 ⊇ u8), and non-overlapping with `bytes`.
        let out_bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), bytes.len()) };
        out_bytes.copy_from_slice(bytes);
    }
    if cfg!(target_endian = "big") {
        for v in &mut out {
            *v = v.swap_bytes();
        }
    }
    Ok(out)
}

/// Loads a frozen eval model (no LoRA, no `requires_grad`). On error, rolls
/// `store` back to its entry state.
pub fn load_qwen35_from_hf_dir(dir: &Path, store: &mut TensorStore) -> Result<Qwen35Model> {
    let rollback = TensorStoreRollback::capture(store);
    match load_qwen35_from_hf_dir_inner(dir, store, LoadMode::FrozenEval, None) {
        Ok(model) => Ok(model),
        Err(err) => {
            rollback.restore(store);
            Err(err)
        }
    }
}

/// Like [`load_qwen35_from_hf_dir`] but keeps `requires_grad = true` on
/// trainable slots. Use the frozen loader for teachers.
pub fn load_qwen35_trainable_from_hf_dir(
    dir: &Path,
    store: &mut TensorStore,
) -> Result<Qwen35Model> {
    let rollback = TensorStoreRollback::capture(store);
    match load_qwen35_from_hf_dir_inner(dir, store, LoadMode::TrainableStudent, None) {
        Ok(model) => Ok(model),
        Err(err) => {
            rollback.restore(store);
            Err(err)
        }
    }
}

/// Cross-size OPD student: teacher and student checkpoints can differ, the
/// student base stays frozen, only adapter weights receive gradients.
pub fn load_qwen35_lora_from_hf_dir(
    dir: &Path,
    lora: LoraConfig,
    target_set: LoraTargetSet,
    store: &mut TensorStore,
) -> Result<Qwen35Model> {
    load_qwen35_lora_from_hf_dir_with_layer_start(dir, lora, target_set, None, store)
}

pub fn load_qwen35_lora_from_hf_dir_with_layer_start(
    dir: &Path,
    lora: LoraConfig,
    target_set: LoraTargetSet,
    lora_layer_start: Option<usize>,
    store: &mut TensorStore,
) -> Result<Qwen35Model> {
    load_qwen35_lora_from_hf_dir_with_shared_base(
        dir,
        lora,
        target_set,
        lora_layer_start,
        false,
        None,
        store,
    )
}

/// LoRA-student load with optional train-infer FP8 base sharing
/// (`--share-frozen-base`).
///
/// When `shared_base` is `Some(table)`, every frozen FP8 block-scaled base
/// projection whose `train_name` matches a table entry imports a NON-OWNING
/// device view over the co-resident infer engine's resident base weight
/// (zero-copy) instead of uploading its own ~27 GB copy. Unmatched tensors
/// and the `None` default take the existing upload path.
pub fn load_qwen35_lora_from_hf_dir_with_shared_base(
    dir: &Path,
    lora: LoraConfig,
    target_set: LoraTargetSet,
    lora_layer_start: Option<usize>,
    lora_skip_experts: bool,
    shared_base: Option<SharedFrozenBaseTable<'_>>,
    store: &mut TensorStore,
) -> Result<Qwen35Model> {
    let rollback = TensorStoreRollback::capture(store);
    match load_qwen35_from_hf_dir_inner(
        dir,
        store,
        LoadMode::LoraStudent {
            lora,
            target_set,
            lora_layer_start,
            lora_skip_experts,
        },
        shared_base,
    ) {
        Ok(model) => Ok(model),
        Err(err) => {
            rollback.restore(store);
            Err(err)
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum LoadMode {
    FrozenEval,
    TrainableStudent,
    LoraStudent {
        lora: LoraConfig,
        target_set: LoraTargetSet,
        lora_layer_start: Option<usize>,
        lora_skip_experts: bool,
    },
}

fn load_qwen35_from_hf_dir_inner(
    dir: &Path,
    store: &mut TensorStore,
    mode: LoadMode,
    shared_base: Option<SharedFrozenBaseTable<'_>>,
) -> Result<Qwen35Model> {
    if !dir.is_dir() {
        return Err(LoaderError::Custom(format!(
            "{} is not a directory. Hint: pass a local HF/ModelScope checkpoint \
             directory containing config.json and model.safetensors.",
            dir.display()
        )));
    }

    let (hf_cfg, schema) = Qwen35HfConfig::from_json_file(dir.join("config.json"))?;
    let mut cfg = hf_cfg.to_qwen35_config()?;
    // Vanilla Qwen3 ships un-gated q_proj; Qwen3.5/3.6 ships gated q_proj.
    if matches!(schema, HfSchema::Qwen3) {
        cfg.full_attn_gated = false;
    }

    let shard_paths = discover_shards(dir)?;
    let shards: Vec<ShardFile> = shard_paths
        .iter()
        .map(|p| ShardFile::open(p))
        .collect::<Result<_>>()?;
    let safetensors_views: Vec<SafeTensors<'_>> = shards
        .iter()
        .map(ShardFile::safetensors)
        .collect::<Result<_>>()?;
    let mut hf_name_to_shard: HashMap<String, usize> = HashMap::new();
    for (idx, view) in safetensors_views.iter().enumerate() {
        for name in view.names() {
            hf_name_to_shard.entry(name.to_string()).or_insert(idx);
        }
    }

    let load_trace = std::env::var("ARLE_OPD_LOAD_TRACE").is_ok();
    if load_trace {
        eprintln!(
            "[opd-load-trace] pre model-construct (shared_base={})",
            shared_base.is_some()
        );
    }
    let mut model = match mode {
        LoadMode::FrozenEval => Qwen35Model::new_for_checkpoint_load(&cfg, store)?,
        LoadMode::TrainableStudent => Qwen35Model::new(&cfg, store)?,
        LoadMode::LoraStudent {
            lora,
            target_set,
            lora_layer_start,
            lora_skip_experts,
        } => {
            if let Some(start) = lora_layer_start {
                Qwen35Model::new_with_lora_targets_for_checkpoint_load_layer_start(
                    &cfg,
                    lora,
                    target_set,
                    Some(start),
                    lora_skip_experts,
                    store,
                )?
            } else {
                Qwen35Model::new_with_lora_targets_for_checkpoint_load(
                    &cfg,
                    lora,
                    target_set,
                    lora_skip_experts,
                    store,
                )?
            }
        }
    };
    if load_trace {
        eprintln!("[opd-load-trace] post model-construct OK");
    }
    if !matches!(mode, LoadMode::FrozenEval) && crate::runtime_flags::gradient_checkpointing() {
        model.set_gradient_checkpointing(true);
    }
    let param_map = model.param_name_map();

    let load_plan = plan_tensor_loads(
        &param_map,
        &cfg,
        schema,
        mode,
        &hf_name_to_shard,
        &safetensors_views,
        store,
    )?;

    for (i, planned) in load_plan.iter().enumerate() {
        if load_trace {
            eprintln!(
                "[opd-load-trace] materialize[{i}/{}] {}",
                load_plan.len(),
                planned.train_name
            );
        }
        load_planned_tensor_into_slot(planned, &safetensors_views, shared_base, store)?;
    }
    if load_trace {
        eprintln!(
            "[opd-load-trace] all {} tensors materialized",
            load_plan.len()
        );
    }

    Ok(model)
}

struct TensorStoreRollback {
    tensors_len: usize,
    free_ids: Vec<TensorId>,
}

impl TensorStoreRollback {
    fn capture(store: &TensorStore) -> Self {
        Self {
            tensors_len: store.tensors.len(),
            free_ids: store.free_ids.clone(),
        }
    }

    fn restore(self, store: &mut TensorStore) {
        store.tensors.truncate(self.tensors_len);
        for &id in &self.free_ids {
            if id < store.tensors.len() {
                store.tensors[id] = None;
            }
        }
        store.free_ids = self.free_ids;
    }
}

struct PlannedTensorLoad {
    hf_name: String,
    train_name: String,
    id: TensorId,
    expected_shape: Vec<usize>,
    requires_grad: bool,
    shard_idx: usize,
    bf16_cuda_frozen_base: bool,
    fp8_cuda_frozen_base: Option<PlannedFp8BlockScaled>,
}

struct PlannedFp8BlockScaled {
    scale_name: String,
    scale_shard_idx: usize,
    block_m: usize,
    block_k: usize,
}

fn plan_tensor_loads(
    param_map: &HashMap<&'static str, TensorId>,
    cfg: &Qwen35Config,
    schema: HfSchema,
    mode: LoadMode,
    hf_name_to_shard: &HashMap<String, usize>,
    safetensors_views: &[SafeTensors<'_>],
    store: &TensorStore,
) -> Result<Vec<PlannedTensorLoad>> {
    // `param_name_map()` returns the same `TensorId` for the embedding row
    // twice (embed_tokens + lm_head when `tie_word_embeddings`). Dedupe here
    // to avoid writing the same slot twice.
    let mut planned_ids: std::collections::HashSet<TensorId> = std::collections::HashSet::new();
    let mut plan = Vec::new();
    for (&train_name, &id) in param_map {
        if planned_ids.contains(&id) {
            continue;
        }
        let candidates = hf_candidates_for_train_name(train_name, schema);
        let mut last_err: Option<LoaderError> = None;
        let mut planned = None;
        for candidate in &candidates {
            match plan_tensor_load(
                candidate,
                train_name,
                id,
                mode,
                hf_name_to_shard,
                safetensors_views,
                store,
            ) {
                Ok(tensor_load) => {
                    planned = Some(tensor_load);
                    break;
                }
                Err(LoaderError::MissingTensor(_)) => continue,
                Err(err) => {
                    last_err = Some(err);
                    break;
                }
            }
        }
        if let Some(tensor_load) = planned {
            planned_ids.insert(id);
            plan.push(tensor_load);
            continue;
        } else {
            if train_name.ends_with("lm_head.weight") && cfg.tie_word_embeddings {
                continue;
            }
            return Err(last_err.unwrap_or_else(|| {
                LoaderError::MissingTensor(format!("{train_name} (tried HF names: {candidates:?})"))
            }));
        }
    }

    Ok(plan)
}

fn plan_tensor_load(
    hf_name: &str,
    train_name: &str,
    id: TensorId,
    mode: LoadMode,
    hf_name_to_shard: &HashMap<String, usize>,
    safetensors_views: &[SafeTensors<'_>],
    store: &TensorStore,
) -> Result<PlannedTensorLoad> {
    let shard_idx = match hf_name_to_shard.get(hf_name) {
        Some(idx) => *idx,
        None => return Err(LoaderError::MissingTensor(hf_name.to_owned())),
    };
    let view = safetensors_views[shard_idx]
        .tensor(hf_name)
        .map_err(|err| LoaderError::Safetensors(format!("{hf_name}: {err}")))?;
    let got_shape: Vec<usize> = view.shape().to_vec();

    let slot = store.get(id).ok_or_else(|| {
        LoaderError::Custom(format!(
            "missing slot for {train_name}. Hint: this indicates a \
             Qwen35Model::param_name_map/config mismatch; report it with \
             the checkpoint config.json and OPD loader follow-up tranche."
        ))
    })?;
    let expected_shape = slot.shape.clone();
    let requires_grad = slot.requires_grad;
    let shape_compatible = expected_shape == got_shape
        || can_squeeze_linear_conv1d_weight(train_name, &expected_shape, &got_shape);
    if !shape_compatible {
        let hint = shape_mismatch_hint(hf_name, train_name, &expected_shape, &got_shape);
        return Err(LoaderError::ShapeMismatch {
            name: train_name.to_owned(),
            expected: expected_shape,
            got: got_shape,
            hint,
        });
    }

    let fp8_cuda_frozen_base = plan_fp8_cuda_frozen_base(
        mode,
        hf_name,
        train_name,
        &expected_shape,
        requires_grad,
        view.dtype(),
        hf_name_to_shard,
        safetensors_views,
        store,
    )?;
    if fp8_cuda_frozen_base.is_none() {
        validate_supported_dtype(&view, hf_name)?;
    }

    let bf16_cuda_frozen_base = should_load_bf16_cuda_frozen_base(
        mode,
        train_name,
        &expected_shape,
        requires_grad,
        view.dtype(),
        store,
    );

    Ok(PlannedTensorLoad {
        hf_name: hf_name.to_owned(),
        train_name: train_name.to_owned(),
        id,
        expected_shape,
        requires_grad,
        shard_idx,
        bf16_cuda_frozen_base,
        fp8_cuda_frozen_base,
    })
}

fn can_squeeze_linear_conv1d_weight(train_name: &str, expected: &[usize], got: &[usize]) -> bool {
    train_name.ends_with(".linear_attn.conv1d.weight")
        && expected.len() == 2
        && got.len() == 3
        && got[1] == 1
        && expected[0] == got[0]
        && expected[1] == got[2]
}

fn should_load_bf16_cuda_frozen_base(
    mode: LoadMode,
    train_name: &str,
    expected_shape: &[usize],
    requires_grad: bool,
    dtype: Dtype,
    store: &TensorStore,
) -> bool {
    // ARLE_NO_BF16_CUDA_FROZEN_BASE forces CPU residency for frozen bf16 base
    // weights. Used by w2s when the student base + aux post-RL would otherwise
    // OOM the GPU during loading (e.g. 27B pair: 54 GB + 55 GB > 97 GB H20).
    if std::env::var("ARLE_NO_BF16_CUDA_FROZEN_BASE").is_ok() {
        return false;
    }
    matches!(mode, LoadMode::LoraStudent { .. } | LoadMode::FrozenEval)
        && !requires_grad
        && dtype == Dtype::BF16
        && store.backend().device() == Device::Cuda
        && is_bf16_cuda_frozen_base_tensor(train_name, expected_shape)
}

fn is_bf16_cuda_frozen_base_tensor(train_name: &str, expected_shape: &[usize]) -> bool {
    if expected_shape.len() != 2 {
        return false;
    }
    train_name.ends_with("embed_tokens.weight")
        || train_name.ends_with("lm_head.weight")
        || train_name.ends_with(".self_attn.q_proj.weight")
        || train_name.ends_with(".self_attn.k_proj.weight")
        || train_name.ends_with(".self_attn.v_proj.weight")
        || train_name.ends_with(".self_attn.o_proj.weight")
        || train_name.ends_with(".linear_attn.in_proj_qkv.weight")
        || train_name.ends_with(".linear_attn.in_proj_z.weight")
        || train_name.ends_with(".linear_attn.in_proj_b.weight")
        || train_name.ends_with(".linear_attn.in_proj_a.weight")
        || train_name.ends_with(".linear_attn.out_proj.weight")
        || train_name.ends_with(".mlp.gate_proj.weight")
        || train_name.ends_with(".mlp.up_proj.weight")
        || train_name.ends_with(".mlp.down_proj.weight")
        || train_name.ends_with(".mlp.gate.weight")
        || train_name.ends_with(".mlp.shared_expert.gate_proj.weight")
        || train_name.ends_with(".mlp.shared_expert.up_proj.weight")
        || train_name.ends_with(".mlp.shared_expert.down_proj.weight")
        || train_name.ends_with(".mlp.shared_expert_gate.weight")
        || is_qwen36_per_expert_projection(train_name)
}

fn plan_fp8_cuda_frozen_base(
    mode: LoadMode,
    hf_name: &str,
    train_name: &str,
    expected_shape: &[usize],
    requires_grad: bool,
    dtype: Dtype,
    hf_name_to_shard: &HashMap<String, usize>,
    safetensors_views: &[SafeTensors<'_>],
    store: &TensorStore,
) -> Result<Option<PlannedFp8BlockScaled>> {
    const QWEN36_FP8_BLOCK_M: usize = 128;
    const QWEN36_FP8_BLOCK_K: usize = 128;

    if !(matches!(mode, LoadMode::LoraStudent { .. } | LoadMode::FrozenEval)
        && !requires_grad
        && dtype == Dtype::F8_E4M3
        && store.backend().device() == Device::Cuda
        && is_fp8_cuda_frozen_base_tensor(train_name, expected_shape))
    {
        return Ok(None);
    }

    let scale_name = fp8_scale_tensor_name(hf_name)?;
    let scale_shard_idx = *hf_name_to_shard
        .get(&scale_name)
        .ok_or_else(|| LoaderError::MissingTensor(scale_name.clone()))?;
    let scale_view = safetensors_views[scale_shard_idx]
        .tensor(&scale_name)
        .map_err(|err| LoaderError::Safetensors(format!("{scale_name}: {err}")))?;
    validate_fp8_scale_view(
        &scale_name,
        &scale_view,
        expected_shape,
        QWEN36_FP8_BLOCK_M,
        QWEN36_FP8_BLOCK_K,
    )?;

    Ok(Some(PlannedFp8BlockScaled {
        scale_name,
        scale_shard_idx,
        block_m: QWEN36_FP8_BLOCK_M,
        block_k: QWEN36_FP8_BLOCK_K,
    }))
}

fn fp8_scale_tensor_name(hf_name: &str) -> Result<String> {
    let base = hf_name.strip_suffix(".weight").ok_or_else(|| {
        LoaderError::Custom(format!(
            "FP8 tensor {hf_name} is missing the .weight suffix. Hint: \
             Qwen3.6 FP8 block-scaled weights must have a matching \
             *.weight_scale_inv side tensor."
        ))
    })?;
    Ok(format!("{base}.weight_scale_inv"))
}

fn validate_fp8_scale_view(
    scale_name: &str,
    scale_view: &impl safetensors::View,
    expected_weight_shape: &[usize],
    block_m: usize,
    block_k: usize,
) -> Result<()> {
    match scale_view.dtype() {
        Dtype::BF16 | Dtype::F32 | Dtype::F16 => {}
        other => return Err(LoaderError::UnsupportedDtype(other, scale_name.to_owned())),
    }
    let expected = fp8_scale_shape(expected_weight_shape, block_m, block_k)?;
    let got = scale_view.shape().to_vec();
    if got != expected {
        return Err(LoaderError::ShapeMismatch {
            name: scale_name.to_owned(),
            expected,
            got,
            hint: ". Hint: Qwen3.6 FP8 block-scaled weights require scale shape \
                   [ceil(rows/128), ceil(cols/128)] for each *.weight_scale_inv tensor."
                .to_owned(),
        });
    }
    Ok(())
}

fn fp8_scale_shape(shape: &[usize], block_m: usize, block_k: usize) -> Result<Vec<usize>> {
    if shape.len() != 2 {
        return Err(LoaderError::Custom(format!(
            "FP8 frozen-base tensor must be rank-2, got shape {shape:?}"
        )));
    }
    Ok(vec![shape[0].div_ceil(block_m), shape[1].div_ceil(block_k)])
}

fn is_fp8_cuda_frozen_base_tensor(train_name: &str, expected_shape: &[usize]) -> bool {
    expected_shape.len() == 2
        && (train_name.ends_with(".self_attn.q_proj.weight")
            || train_name.ends_with(".self_attn.k_proj.weight")
            || train_name.ends_with(".self_attn.v_proj.weight")
            || train_name.ends_with(".self_attn.o_proj.weight")
            || train_name.ends_with(".linear_attn.in_proj_qkv.weight")
            || train_name.ends_with(".linear_attn.in_proj_z.weight")
            || train_name.ends_with(".linear_attn.in_proj_b.weight")
            || train_name.ends_with(".linear_attn.in_proj_a.weight")
            || train_name.ends_with(".linear_attn.out_proj.weight")
            // Dense MLP (non-MoE students, e.g. Qwen3.x-27B dense).
            || train_name.ends_with(".mlp.gate_proj.weight")
            || train_name.ends_with(".mlp.up_proj.weight")
            || train_name.ends_with(".mlp.down_proj.weight")
            || train_name.ends_with(".mlp.shared_expert.gate_proj.weight")
            || train_name.ends_with(".mlp.shared_expert.up_proj.weight")
            || train_name.ends_with(".mlp.shared_expert.down_proj.weight")
            || is_qwen36_per_expert_projection(train_name))
}

fn is_qwen36_per_expert_projection(train_name: &str) -> bool {
    train_name.contains(".mlp.experts.")
        && (train_name.ends_with(".gate_proj.weight")
            || train_name.ends_with(".up_proj.weight")
            || train_name.ends_with(".down_proj.weight"))
}

fn validate_supported_dtype(view: &impl safetensors::View, name: &str) -> Result<()> {
    match view.dtype() {
        Dtype::F32 | Dtype::BF16 | Dtype::F16 => Ok(()),
        other => Err(LoaderError::UnsupportedDtype(other, name.to_owned())),
    }
}

fn load_planned_tensor_into_slot(
    planned: &PlannedTensorLoad,
    safetensors_views: &[SafeTensors<'_>],
    shared_base: Option<SharedFrozenBaseTable<'_>>,
    store: &mut TensorStore,
) -> Result<()> {
    let view = safetensors_views[planned.shard_idx]
        .tensor(&planned.hf_name)
        .map_err(|err| LoaderError::Safetensors(format!("{}: {err}", planned.hf_name)))?;
    if let Some(fp8) = &planned.fp8_cuda_frozen_base {
        // Train-infer weight sharing (`--share-frozen-base`): if a co-resident
        // infer engine exposes the resident FP8 base for this projection and
        // the dims match, import a NON-OWNING device view (zero-copy) instead
        // of uploading a private ~27 GB copy.
        if let Some(entry) = shared_base.and_then(|table| {
            table.iter().find(|e| {
                e.matches(&planned.train_name)
                    && e.rows == planned.expected_shape[0]
                    && e.cols == planned.expected_shape[1]
                    && e.block_m == fp8.block_m
                    && e.block_k == fp8.block_k
            })
        }) {
            let handle = store
                .backend()
                .import_fp8_block_scaled_device_ptr(
                    entry.weight_ptr,
                    entry.scale_ptr,
                    &planned.expected_shape,
                    fp8.block_m,
                    fp8.block_k,
                )
                .map_err(LoaderError::Autograd)?;
            store
                .replace_device_handle(planned.id, handle)
                .map_err(LoaderError::Autograd)?;
            store
                .set_requires_grad(planned.id, false)
                .map_err(LoaderError::Autograd)?;
            return Ok(());
        }
        let scale_view = safetensors_views[fp8.scale_shard_idx]
            .tensor(&fp8.scale_name)
            .map_err(|err| LoaderError::Safetensors(format!("{}: {err}", fp8.scale_name)))?;
        let scales = dtype_to_f32(&scale_view, &fp8.scale_name)?;
        let weight = view.data();
        let handle = store
            .backend()
            .upload_fp8_block_scaled(
                weight.as_ref(),
                &scales,
                &planned.expected_shape,
                fp8.block_m,
                fp8.block_k,
            )
            .map_err(LoaderError::Autograd)?;
        store
            .replace_device_handle(planned.id, handle)
            .map_err(LoaderError::Autograd)?;
        store
            .set_requires_grad(planned.id, false)
            .map_err(LoaderError::Autograd)?;
        return Ok(());
    }
    if planned.bf16_cuda_frozen_base {
        let data = dtype_to_bf16_bits(&view, &planned.hf_name)?;
        let handle = store
            .backend()
            .upload_bf16_bits(&data, &planned.expected_shape)
            .map_err(LoaderError::Autograd)?;
        store
            .replace_device_handle(planned.id, handle)
            .map_err(LoaderError::Autograd)?;
        return Ok(());
    }

    let data = dtype_to_f32(&view, &planned.hf_name)?;

    let tensor = Tensor::new(data, planned.expected_shape.clone(), planned.requires_grad).map_err(
        |err| {
            LoaderError::Custom(format!(
                "failed to materialize {} from {}: {err}. Hint: verify the safetensors \
                 data length matches the validated checkpoint shape.",
                planned.train_name, planned.hf_name
            ))
        },
    )?;
    store.tensors[planned.id] = Some(tensor);
    Ok(())
}

fn shape_mismatch_hint(
    hf_name: &str,
    train_name: &str,
    expected: &[usize],
    got: &[usize],
) -> String {
    let q_proj_hint = q_proj_gate_hint(train_name, expected, got);
    if !q_proj_hint.is_empty() {
        return q_proj_hint;
    }
    format!(
        ". Hint: verify config.json matches the safetensors checkpoint and \
         that HF tensor `{hf_name}` belongs to the same Qwen3.5/Qwen3.6 model \
         family as `{train_name}`."
    )
}

/// Train side is Qwen3.5-shaped (`q_proj` includes the per-head output gate);
/// vanilla Qwen3 ships `q_proj` without that gate.
fn q_proj_gate_hint(train_name: &str, expected: &[usize], got: &[usize]) -> String {
    if !train_name.ends_with(".self_attn.q_proj.weight") {
        return String::new();
    }
    if expected.len() != 2 || got.len() != 2 {
        return String::new();
    }
    if expected[1] != got[1] {
        return String::new();
    }
    if expected[0] != got[0] * 2 {
        return String::new();
    }
    " — vanilla Qwen3 ships an un-gated q_proj; train::Qwen35Model expects \
     the Qwen3.5/3.6 gated layout where out_features = num_heads * head_dim * 2. \
     Loading a non-gated checkpoint into the gated model requires either (a) a \
     plain-Qwen3 model variant in `crates/train/` or (b) a documented gate-\
     synthesis hook on Qwen35Model. Neither is in scope of this loader."
        .to_owned()
}
