//! HuggingFace-format safetensors loader for [`Qwen35Model`].
//!
//! Reads a HF-style model directory (a `config.json` plus one or more
//! `model*.safetensors` shards, optionally with a `model.safetensors.index.json`
//! manifest) and materializes a live `Qwen35Model` whose `TensorStore` slots
//! are populated from the on-disk weights.
//!
//! ## Schema coverage
//!
//! - **Qwen3.5 / Qwen3.6 layout** (nested `text_config`, tensor names rooted at
//!   `model.language_model.*`, `q_proj` includes the output gate so its
//!   `out_features == num_attention_heads * head_dim * 2`): natively supported.
//!   The HF config is consumed via [`Qwen35Config::from_json_str`] which handles
//!   both nested and flat layouts.
//!
//! - **Vanilla Qwen3 layout** (flat HF config, tensor names rooted at
//!   `model.*`, plain `q_proj` of shape `[num_heads * head_dim, hidden]` and
//!   no `linear_attention` layers): partially supported. The loader maps the
//!   `model.*` prefix to the `model.language_model.*` namespace the train
//!   model uses internally, synthesizes the missing `linear_*` config fields
//!   from the standard full-attention sizes, and reports a clear error if
//!   `q_proj`'s on-disk shape does not match the gated-attention shape the
//!   train-side `Qwen35Model` was built for. See [`load_qwen35_from_hf_dir`]
//!   for the exact failure mode and the follow-up tranche needed to land a
//!   non-gated full-attention variant of `Qwen35Model`.
//!
//! ## What the loader does not do
//!
//! - It does not download anything. It expects an already-materialized
//!   directory on disk (the canonical entry point is
//!   `~/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B/` for the OPD-only pivot
//!   smoke path).
//! - It does not touch the tokenizer or generation config. Those live in
//!   the same directory but are read elsewhere (e.g. `train::tokenizer`).
//! - Quantized checkpoint support is deliberately narrow: CUDA LoRA-student
//!   loads accept frozen FP8 E4M3 block-scaled base linear weights when the
//!   checkpoint provides the matching `*.weight_scale_inv` side tensor. Teacher,
//!   trainable-base, and CPU loads still reject quantized weights.
//!
//! ## Independence from the `infer` crate
//!
//! Train must not depend on `infer` at runtime per the OPD-only pivot
//! contract. This file therefore re-implements the small amount of shard
//! discovery + BF16/F16 widening needed; the heavy lifting (safetensors
//! parsing) goes through the workspace `safetensors` crate directly.

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
/// borrowed read-only from a co-resident infer-cuda engine for train-infer
/// weight sharing (`--share-frozen-base`).
///
/// Backend-agnostic by design: the loader must not depend on `infer` (OPD-only
/// pivot contract), so `train_cli` maps the infer-api `SharedFp8BaseProjection`
/// into this plain struct. `layer_idx` + `proj_suffix` (e.g.
/// `"self_attn.q_proj"`) form the key the loader matches against a planned
/// tensor's `train_name` (`*.layers.{layer_idx}.{proj_suffix}.weight`).
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
    /// True iff this entry is the shared base for the given autograd
    /// `train_name`. Matches `*.layers.{layer_idx}.{proj_suffix}.weight`, robust
    /// to both the `model.language_model.*` (Qwen3.5/3.6) and `model.*`
    /// (vanilla Qwen3) name prefixes.
    fn matches(&self, train_name: &str) -> bool {
        train_name.ends_with(&format!(
            ".layers.{}.{}.weight",
            self.layer_idx, self.proj_suffix
        ))
    }
}

/// Lookup table of shared frozen base projections, keyed by `train_name` match.
/// `None`/empty = the default (no sharing) path — every frozen FP8 base tensor
/// uploads its own copy, byte-identical to today.
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

// ─────────────────────────── HF config schema ────────────────────────────────

/// Minimal serde mirror of a HuggingFace Qwen3 / Qwen3.5 `config.json`.
///
/// Field set is the union of vanilla Qwen3 (0.6B / 1.7B / 4B) and the
/// Qwen3.5 / Qwen3.6 nested `text_config` layout. We accept either by
/// reading both shapes via [`serde_json::Value`] inside
/// [`Qwen35HfConfig::from_value`] before binding fields, rather than relying
/// on a tagged enum that complicates downstream consumers.
///
/// All `linear_*` fields are optional because vanilla Qwen3 omits them
/// entirely. `layer_types` is also optional — when missing we treat every
/// layer as `FullAttention`.
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

    // Optional Qwen3.5-style fields (absent on vanilla Qwen3 0.6B/1.7B/4B).
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

/// What kind of HF schema this directory exposes — controls name remapping
/// and downstream contract checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfSchema {
    /// `model.layers.N.*` prefix, plain (un-gated) `q_proj`. Examples:
    /// `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`.
    Qwen3,
    /// `model.language_model.layers.N.*` prefix, gated `q_proj` (out_features
    /// includes the per-head output gate). Examples: `Qwen/Qwen3.5-*`,
    /// `Qwen/Qwen3.6-*`.
    Qwen35,
}

impl Qwen35HfConfig {
    /// Parse a HuggingFace `config.json`. Accepts both the flat (Qwen3) and
    /// nested-`text_config` (Qwen3.5 / Qwen3.6) layouts; the nested form is
    /// unwrapped before field binding.
    pub fn from_json_str(content: &str) -> Result<(Self, HfSchema)> {
        let value: serde_json::Value = serde_json::from_str(content)?;
        Self::from_value(&value)
    }

    pub fn from_value(value: &serde_json::Value) -> Result<(Self, HfSchema)> {
        let (text, schema) = match value.get("text_config") {
            Some(text) => (text.clone(), HfSchema::Qwen35),
            None => (value.clone(), HfSchema::Qwen3),
        };
        // Fold the model-level `eos_token_id` / `bos_token_id` from the outer
        // object onto the text block when the nested block doesn't carry them
        // (Qwen3.5 typical layout).
        let text = if schema == HfSchema::Qwen35 {
            merge_token_ids(text, value)
        } else {
            text
        };

        let mut config: Qwen35HfConfig = serde_json::from_value(text.clone())?;
        config.merge_nested_moe_config();

        // Qwen3.5 / Qwen3.6 stash rope under a `rope_parameters` block.
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

    /// Convert into the train-side [`Qwen35Config`]. Missing `linear_*`
    /// fields are filled with defaults derived from the dense attention
    /// shape — the train model only consults them when a layer has
    /// `LayerType::LinearAttention`, so for vanilla full-attention Qwen3
    /// the synthesized values are inert.
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
            // Defaults derived from the dense attention shape — only consulted
            // if a layer is `LinearAttention`.
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
            output_gate_type: "sigmoid".to_string(),
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

// ─────────────────────────── shard discovery ─────────────────────────────────

/// One memory-mapped safetensors shard plus its (lazy) deserialized index.
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

/// Discover shards. Returns either a single `model.safetensors` shard
/// (when the index manifest is absent) or one shard per file referenced
/// in `model.safetensors.index.json`.
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

// ─────────────────────────── name remapping ──────────────────────────────────

/// Map a train-side tensor name (rooted under `model.language_model.*`) to
/// the HF tensor name for the supplied schema.
///
/// For `HfSchema::Qwen35` this is a no-op (the train side uses the Qwen3.5
/// canonical naming). For `HfSchema::Qwen3` we strip the `language_model.`
/// segment so e.g. `model.language_model.layers.0.self_attn.q_proj.weight`
/// becomes `model.layers.0.self_attn.q_proj.weight`. The lm_head case is
/// handled by [`hf_lm_head_candidates`].
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
/// and the LM head tensor may be absent; the train-side tied case maps to
/// `embed_tokens.weight`, so only explicit untied `*.lm_head.weight` names
/// should route through these fallback candidates.
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

// ─────────────────────────── dtype widening ──────────────────────────────────

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
    // The safetensors BF16 payload is little-endian u16. The old
    // `chunks_exact(2).map(u16::from_le_bytes).collect()` ran one scalar
    // `core::ptr::write::<u16>` per element — single-threaded over the giant
    // frozen-base tensors (`embed_tokens` + `lm_head` are each [248320, 5120] =
    // 1.27 B u16), so the 27B student load spun ~minutes on one core (GPU idle,
    // RSS flat) and the over-long load got reaped by the box watchdog before it
    // finished. Bulk-copy the bytes into the `Vec<u16>` instead (one memcpy),
    // then byte-swap only on a big-endian host. x86-64/aarch64 are LE, so this
    // is a straight memcpy there — the per-element loop is fully eliminated.
    let mut out = vec![0u16; bytes.len() / 2];
    {
        // SAFETY: `out` owns `out.len()` u16 = `bytes.len()` contiguous bytes; a
        // `[u8]` view over its buffer is valid for that exact length, properly
        // sized/aligned (u16 alignment ⊇ u8), and non-overlapping with `bytes`
        // (fresh alloc). `copy_from_slice` asserts equal length.
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

// ─────────────────────────── public entry point ──────────────────────────────

/// Load a HF-format Qwen3 / Qwen3.5 checkpoint into a fresh frozen
/// [`Qwen35Model`].
///
/// The model is initialized via [`Qwen35Model::new_for_eval`] (frozen, no
/// LoRA, no `requires_grad`) and every parameter slot is overwritten with
/// the data read from the safetensors shards in `dir`.
///
/// Returns the constructed model. On any error the function rolls `store`
/// back to its entry state, so callers do not need to discard the store after
/// a failed OPD checkpoint load.
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

/// Load a HF-format Qwen3 / Qwen3.5 checkpoint into a fresh trainable
/// [`Qwen35Model`] suitable for OPD student optimization.
///
/// This keeps the same shard discovery, shape validation, dtype widening, and
/// rollback semantics as [`load_qwen35_from_hf_dir`], but initializes the model
/// with [`Qwen35Model::new`] so loaded trainable parameter slots keep
/// `requires_grad = true`. Use the frozen loader for teachers.
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

/// Load a HF-format Qwen3 / Qwen3.5 checkpoint as a frozen base plus
/// trainable LoRA adapters. This is the cross-size OPD student path: teacher
/// and student checkpoints can differ, while the student base stays frozen and
/// only adapter weights receive gradients.
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
/// (zero-copy) instead of uploading its own ~27 GB copy. Unmatched frozen FP8
/// tensors, and all tensors when `shared_base` is `None`, take the existing
/// `upload_fp8_block_scaled` path — so the default (`None`) is byte-identical
/// to [`load_qwen35_lora_from_hf_dir_with_layer_start`].
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

fn opd_gradient_checkpointing_enabled() -> bool {
    crate::runtime_flags::gradient_checkpointing()
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

    // 1) HF config → Qwen35Config.
    let (hf_cfg, schema) = Qwen35HfConfig::from_json_file(dir.join("config.json"))?;
    let mut cfg = hf_cfg.to_qwen35_config()?;
    // Vanilla Qwen3 (flat-schema HF config) ships un-gated q_proj. Qwen3.5 /
    // Qwen3.6 (nested `text_config`) ships gated q_proj — the default that
    // `to_qwen35_config` writes.
    if matches!(schema, HfSchema::Qwen3) {
        cfg.full_attn_gated = false;
    }

    // 2) Open every shard once and build a `hf_name -> shard_idx` lookup before
    //    allocating model tensors in the caller's store. Missing checkpoint
    //    files should fail without leaving a half-constructed eval model behind.
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

    // 3) Qwen35Config → fresh model. The load mode controls whether loaded
    //    slots remain frozen for teachers/eval or trainable for OPD students.
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
    if !matches!(mode, LoadMode::FrozenEval) && opd_gradient_checkpointing_enabled() {
        model.set_gradient_checkpointing(true);
    }
    let param_map = model.param_name_map();

    // 4) Preflight every tensor before writing any checkpoint data into the
    //    store. This keeps missing/mismatched later tensors from leaving
    //    partially materialized checkpoint weights behind.
    let load_plan = plan_tensor_loads(
        &param_map,
        &cfg,
        schema,
        mode,
        &hf_name_to_shard,
        &safetensors_views,
        store,
    )?;

    // 5) Materialize each train parameter from the safetensors.
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
    //
    // The `param_name_map()` contract returns the same `TensorId` for the
    // embedding row twice (once under the embed_tokens name, once under
    // `lm_head` when `tie_word_embeddings == true`). Deduplicating here keeps
    // us from writing the same slot twice and lets us report a clean
    // "missing lm_head" error only when the model genuinely needs a separate
    // head tensor.
    let mut planned_ids: std::collections::HashSet<TensorId> = std::collections::HashSet::new();
    let mut plan = Vec::new();
    for (&train_name, &id) in param_map {
        if planned_ids.contains(&id) {
            // Already filled (tied lm_head case).
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
            // Tied-embedding fallback: if this slot is the lm_head and the
            // tied embedding slot was already planned, we're done. If this
            // lm_head name appears before embed_tokens in the HashMap order,
            // leave the id unplanned so the embedding name can still load it.
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
    matches!(mode, LoadMode::LoraStudent { .. })
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

    if !(matches!(mode, LoadMode::LoraStudent { .. })
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
            // Dense MLP (non-MoE students, e.g. Qwen3.x-27B dense). The whitelist
            // predates dense FP8 students (only MoE/DSv4 were loaded), so the plain
            // mlp.{gate,up,down}_proj had no FP8 path and fell to "unsupported dtype".
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
        // infer engine exposes the resident FP8 base for THIS frozen projection
        // and the dims match, import a NON-OWNING device view (zero-copy)
        // instead of uploading a private ~27 GB copy. Default (`None`) and any
        // unmatched tensor fall through to the byte-identical upload below.
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

/// Detect the specific "vanilla Qwen3 q_proj has half the rows the train
/// model expects" mismatch and surface a precise, actionable hint. The
/// train side is Qwen3.5-shaped (`q_proj` includes the per-head output
/// gate); vanilla Qwen3 ships `q_proj` without that gate.
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

// ─────────────────────────── unit tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use safetensors::{Dtype, serialize_to_file};

    use super::*;

    /// Canonical Qwen3-0.6B `config.json` (HF flat layout). Used to verify
    /// the HF-config → `Qwen35Config` conversion without needing the
    /// safetensors file on disk.
    const QWEN3_0_6B_CONFIG_JSON: &str = r#"{
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 40960,
        "max_window_layers": 28,
        "model_type": "qwen3",
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": null,
        "rope_theta": 1000000,
        "sliding_window": null,
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.0",
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 151936
    }"#;

    #[test]
    fn bf16_cuda_frozen_base_predicate_includes_large_linear_tables() {
        let shape = [151_936, 1024];
        assert!(is_bf16_cuda_frozen_base_tensor(
            "model.language_model.embed_tokens.weight",
            &shape
        ));
        assert!(is_bf16_cuda_frozen_base_tensor(
            "model.language_model.lm_head.weight",
            &shape
        ));

        let linear_shape = [3072, 1024];
        for name in [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.self_attn.k_proj.weight",
            "model.language_model.layers.0.self_attn.v_proj.weight",
            "model.language_model.layers.0.self_attn.o_proj.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.up_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.layers.0.mlp.gate.weight",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.language_model.layers.0.mlp.experts.7.up_proj.weight",
            "model.language_model.layers.0.mlp.experts.255.down_proj.weight",
        ] {
            assert!(
                is_bf16_cuda_frozen_base_tensor(name, &linear_shape),
                "{name}"
            );
        }
    }

    #[test]
    fn dtype_to_bf16_bits_bulk_matches_scalar_reference() {
        // Mirror the load-time conversion of the giant frozen-base tables: build
        // a BF16 safetensors view over a known little-endian payload and assert
        // the bulk-copy path produces EXACTLY the old per-element
        // `chunks_exact(2).map(u16::from_le_bytes)` result (the fix must be
        // byte-identical, only faster). Covers odd values, 0x0000, 0xFFFF, and a
        // non-zero low/high byte so any endian/stride bug surfaces.
        let words: Vec<u16> = (0u16..4096).chain([0, 0xFFFF, 0x1234, 0xABCD]).collect();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let shape = vec![words.len()];
        let view = safetensors::tensor::TensorView::new(Dtype::BF16, shape, &bytes)
            .expect("construct BF16 view");
        let got = dtype_to_bf16_bits(&view, "test.weight").expect("convert");
        let reference: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(got, reference, "bulk bf16-bits conversion diverged");
        assert_eq!(
            got, words,
            "bf16 bits must equal the source little-endian u16"
        );
    }

    #[test]
    fn bf16_cuda_frozen_base_predicate_excludes_non_linear_tables() {
        assert!(!is_bf16_cuda_frozen_base_tensor(
            "model.language_model.layers.0.input_layernorm.weight",
            &[1024]
        ));
        assert!(!is_bf16_cuda_frozen_base_tensor(
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            &[1024, 1, 4]
        ));
        assert!(!is_bf16_cuda_frozen_base_tensor(
            "model.language_model.layers.0.linear_attn.dt_bias",
            &[1024]
        ));
        assert!(!is_bf16_cuda_frozen_base_tensor(
            "model.language_model.layers.0.linear_attn.a_log",
            &[1024]
        ));
        assert!(!is_bf16_cuda_frozen_base_tensor(
            "model.language_model.layers.0.linear_attn.q_norm.weight",
            &[1024]
        ));
        assert!(!is_bf16_cuda_frozen_base_tensor(
            "model.language_model.layers.0.unrelated.weight",
            &[1024, 1024]
        ));
    }

    #[test]
    fn parses_qwen3_0_6b_flat_config() {
        let (cfg, schema) = Qwen35HfConfig::from_json_str(QWEN3_0_6B_CONFIG_JSON).unwrap();
        assert_eq!(schema, HfSchema::Qwen3);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 151_936);
        assert_eq!(cfg.rope_theta, 1_000_000.0);
        assert_eq!(cfg.partial_rotary_factor, 1.0);
        assert_eq!(cfg.max_position_embeddings, Some(40_960));
        assert_eq!(cfg.eos_token_id, Some(151_645));
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.layer_types, None);
    }

    #[test]
    fn converts_qwen3_0_6b_to_qwen35_config() {
        let (hf, _schema) = Qwen35HfConfig::from_json_str(QWEN3_0_6B_CONFIG_JSON).unwrap();
        let cfg = hf.to_qwen35_config().expect("convert");
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.intermediate_size, 3072);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab_size, 151_936);
        assert_eq!(cfg.rope_theta, 1_000_000.0);
        assert_eq!(cfg.rotary_dim, 128); // partial=1.0
        assert_eq!(cfg.rope_cache_len_hint, Some(40_960));
        assert_eq!(cfg.eos_token_id, 151_645);
        assert_eq!(cfg.bos_token_id, Some(151_643));
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.layer_types.len(), 28);
        assert!(
            cfg.layer_types
                .iter()
                .all(|lt| *lt == LayerType::FullAttention)
        );
        // Synthesized linear_* fields are inert (no LinearAttention layers).
        assert_eq!(cfg.linear_num_key_heads, 16);
        assert_eq!(cfg.linear_key_head_dim, 128);
        assert_eq!(cfg.linear_conv_kernel_dim, 4);
    }

    /// Nested-layout Qwen3.5/Qwen3.6 style config — verifies the schema
    /// detection picks `Qwen35` and the rope_parameters block parses.
    const QWEN35_NESTED_CONFIG_JSON: &str = r#"{
        "architectures": ["Qwen3_5_NextForCausalLM"],
        "eos_token_id": 248044,
        "text_config": {
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 8192,
            "rms_norm_eps": 1e-6,
            "layer_types": ["full_attention", "full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "rope_parameters": {
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.5
            },
            "max_position_embeddings": 32768,
            "tie_word_embeddings": true
        }
    }"#;

    const QWEN36_MOE_CONFIG_JSON: &str = r#"{
        "architectures": ["Qwen3_5_MoeForCausalLM"],
        "eos_token_id": 248044,
        "text_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "rope_parameters": {
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25
            },
            "max_position_embeddings": 32768,
            "tie_word_embeddings": false,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "decoder_sparse_step": 1,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512
        }
    }"#;

    const QWEN36_NESTED_MOE_CONFIG_JSON: &str = r#"{
        "architectures": ["Qwen3_5_MoeForCausalLM"],
        "eos_token_id": 248044,
        "text_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 248320,
            "rms_norm_eps": 1e-6,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_value_head_dim": 128,
            "rope_parameters": {
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25
            },
            "max_position_embeddings": 32768,
            "tie_word_embeddings": false,
            "moe_config": {
                "num_experts": 128,
                "num_experts_per_tok": 4,
                "decoder_sparse_step": 2,
                "moe_intermediate_size": 1024,
                "shared_expert_intermediate_size": 1024,
                "norm_topk_prob": false,
                "mlp_only_layers": [0]
            }
        }
    }"#;

    const TINY_QWEN35_CONFIG_JSON: &str = r#"{
        "architectures": ["Qwen3_5_NextForCausalLM"],
        "eos_token_id": 7,
        "text_config": {
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 8,
            "rms_norm_eps": 1e-6,
            "layer_types": ["full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 4,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_value_head_dim": 4,
            "rope_parameters": {
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0
            },
            "max_position_embeddings": 8,
            "tie_word_embeddings": true
        }
    }"#;

    struct TestTensorView {
        dtype: Dtype,
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    impl TestTensorView {
        fn from_f32(shape: Vec<usize>, values: &[f32]) -> Self {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            Self {
                dtype: Dtype::F32,
                shape,
                bytes,
            }
        }

        fn from_dtype(dtype: Dtype, shape: Vec<usize>, bytes: Vec<u8>) -> Self {
            Self {
                dtype,
                shape,
                bytes,
            }
        }
    }

    impl safetensors::View for TestTensorView {
        fn dtype(&self) -> Dtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(self.bytes.as_slice())
        }

        fn data_len(&self) -> usize {
            self.bytes.len()
        }
    }

    fn write_partial_tiny_qwen35_checkpoint(dir: &Path, sentinel: f32) {
        std::fs::write(dir.join("config.json"), TINY_QWEN35_CONFIG_JSON).expect("write config");
        let embed_values = [sentinel; 8 * 4];
        let embed = TestTensorView::from_f32(vec![8, 4], &embed_values);
        serialize_to_file(
            vec![(
                "model.language_model.embed_tokens.weight".to_string(),
                embed,
            )],
            None,
            &dir.join("model.safetensors"),
        )
        .expect("write partial safetensors");
    }

    #[test]
    fn parses_qwen35_nested_text_config() {
        let (cfg, schema) = Qwen35HfConfig::from_json_str(QWEN35_NESTED_CONFIG_JSON).unwrap();
        assert_eq!(schema, HfSchema::Qwen35);
        // rope_parameters: rope_theta is taken from the nested block, not the root.
        assert_eq!(cfg.rope_theta, 1_000_000.0);
        assert_eq!(cfg.partial_rotary_factor, 0.5);
        // eos_token_id is on the root, not the text_config block.
        assert_eq!(cfg.eos_token_id, Some(248_044));
        assert_eq!(cfg.hidden_size, 2560);
        let layer_types = cfg.layer_types.as_ref().expect("layer_types present");
        assert_eq!(layer_types.len(), 2);
    }

    #[test]
    fn parses_qwen36_moe_config_without_intermediate_size() {
        let (hf, schema) = Qwen35HfConfig::from_json_str(QWEN36_MOE_CONFIG_JSON).unwrap();
        assert_eq!(schema, HfSchema::Qwen35);
        assert_eq!(hf.intermediate_size, 0);
        assert_eq!(hf.num_experts, 256);
        assert_eq!(hf.num_experts_per_tok, 8);
        assert_eq!(hf.moe_intermediate_size, 512);
        assert_eq!(hf.shared_expert_intermediate_size, 512);

        let cfg = hf.to_qwen35_config().expect("convert qwen36 moe");
        assert!(cfg.is_moe());
        assert_eq!(cfg.intermediate_size, 0);
        assert_eq!(cfg.num_experts, 256);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.decoder_sparse_step, 1);
        assert_eq!(cfg.moe_intermediate_size, 512);
        assert_eq!(cfg.shared_expert_intermediate_size, 512);
        assert!(cfg.norm_topk_prob);
        assert!(!cfg.tie_word_embeddings);
        assert_eq!(cfg.lm_head_tensor_name(), "lm_head.weight");
        for layer_idx in 0..cfg.num_hidden_layers {
            assert!(cfg.is_moe_layer(layer_idx), "layer {layer_idx}");
        }
    }

    #[test]
    fn parses_qwen36_nested_moe_config() {
        let (hf, _schema) = Qwen35HfConfig::from_json_str(QWEN36_NESTED_MOE_CONFIG_JSON).unwrap();
        let cfg = hf.to_qwen35_config().expect("convert nested moe");
        assert!(cfg.is_moe());
        assert_eq!(cfg.num_experts, 128);
        assert_eq!(cfg.num_experts_per_tok, 4);
        assert_eq!(cfg.decoder_sparse_step, 2);
        assert_eq!(cfg.moe_intermediate_size, 1024);
        assert_eq!(cfg.shared_expert_intermediate_size, 1024);
        assert!(!cfg.norm_topk_prob);
        assert_eq!(cfg.mlp_only_layers, vec![0]);
        assert!(!cfg.is_moe_layer(0));
        assert!(cfg.is_moe_layer(1));
        assert!(!cfg.is_moe_layer(2));
        assert!(cfg.is_moe_layer(3));
    }

    #[test]
    fn train_name_to_hf_qwen3_strips_language_model_segment() {
        assert_eq!(
            train_name_to_hf(
                "model.language_model.layers.7.self_attn.q_proj.weight",
                HfSchema::Qwen3
            ),
            "model.layers.7.self_attn.q_proj.weight"
        );
        assert_eq!(
            train_name_to_hf("model.language_model.embed_tokens.weight", HfSchema::Qwen3),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            train_name_to_hf("model.language_model.norm.weight", HfSchema::Qwen3),
            "model.norm.weight"
        );
    }

    #[test]
    fn train_name_to_hf_qwen35_is_identity() {
        assert_eq!(
            train_name_to_hf(
                "model.language_model.layers.0.self_attn.q_proj.weight",
                HfSchema::Qwen35
            ),
            "model.language_model.layers.0.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn tied_embedding_uses_embed_tokens_candidate_not_lm_head_fallback() {
        let (hf, schema) = Qwen35HfConfig::from_json_str(TINY_QWEN35_CONFIG_JSON).unwrap();
        let cfg = hf.to_qwen35_config().expect("convert");
        assert!(cfg.tie_word_embeddings);

        let candidates = hf_candidates_for_train_name(cfg.embed_tokens_tensor_name(), schema);

        assert_eq!(
            candidates,
            vec!["model.language_model.embed_tokens.weight".to_string()]
        );
        assert!(
            !candidates.iter().any(|name| name.contains("lm_head")),
            "tied embedding must load the embedding tensor, not lm_head fallback candidates"
        );
    }

    #[test]
    fn q_proj_gate_hint_detects_gated_vs_plain_mismatch() {
        let hint = q_proj_gate_hint(
            "model.language_model.layers.0.self_attn.q_proj.weight",
            &[4096, 1024],
            &[2048, 1024],
        );
        assert!(
            hint.contains("vanilla Qwen3 ships an un-gated q_proj"),
            "hint missing diagnostic: {hint}"
        );
        // unrelated tensor → no hint
        let unrelated = q_proj_gate_hint(
            "model.language_model.layers.0.input_layernorm.weight",
            &[1024],
            &[2048],
        );
        assert!(unrelated.is_empty());
        // matching shapes → no hint
        let matching = q_proj_gate_hint(
            "model.language_model.layers.0.self_attn.q_proj.weight",
            &[2048, 1024],
            &[2048, 1024],
        );
        assert!(matching.is_empty());
    }

    #[test]
    fn shape_mismatch_hint_falls_back_to_checkpoint_hint() {
        let hint = shape_mismatch_hint(
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight",
            &[16, 8],
            &[8, 8],
        );

        assert!(hint.contains("Hint: verify config.json"));
        assert!(hint.contains("HF tensor"));
        assert!(hint.contains("Qwen3.5/Qwen3.6"));
    }

    #[test]
    fn unsupported_dtype_error_includes_conversion_hint() {
        let view = TestTensorView::from_dtype(Dtype::F8_E4M3, vec![2, 2], vec![0_u8; 4]);

        let err = validate_supported_dtype(&view, "model.language_model.embed_tokens.weight")
            .expect_err("quantized dtype must be rejected");

        let message = err.to_string();
        assert!(message.contains("unsupported dtype F8_E4M3"));
        assert!(message.contains("F32/BF16/F16"));
        assert!(message.contains("CUDA LoRA-student frozen FP8 E4M3"));
        assert!(message.contains("weight_scale_inv"));
    }

    #[test]
    fn fp8_scale_name_and_shape_match_qwen36_block_contract() {
        assert_eq!(
            fp8_scale_tensor_name("model.language_model.layers.0.mlp.experts.7.up_proj.weight")
                .unwrap(),
            "model.language_model.layers.0.mlp.experts.7.up_proj.weight_scale_inv"
        );
        assert_eq!(
            fp8_scale_shape(&[512, 2048], 128, 128).unwrap(),
            vec![4, 16]
        );
        assert_eq!(
            fp8_scale_shape(&[8192, 2048], 128, 128).unwrap(),
            vec![64, 16]
        );
    }

    #[test]
    fn fp8_cuda_frozen_base_predicate_matches_linear_weight_only_contract() {
        assert!(is_fp8_cuda_frozen_base_tensor(
            "model.language_model.layers.0.mlp.experts.0.up_proj.weight",
            &[512, 2048]
        ));
        assert!(is_fp8_cuda_frozen_base_tensor(
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            &[2048, 512]
        ));
        assert!(is_fp8_cuda_frozen_base_tensor(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            &[8192, 2048]
        ));
        assert!(!is_fp8_cuda_frozen_base_tensor(
            "model.language_model.embed_tokens.weight",
            &[248_320, 2048]
        ));
        assert!(!is_fp8_cuda_frozen_base_tensor(
            "model.language_model.layers.0.input_layernorm.weight",
            &[2048]
        ));
    }

    #[test]
    fn fp8_scale_view_rejects_wrong_shape() {
        let scale = TestTensorView::from_dtype(Dtype::BF16, vec![3, 16], vec![0_u8; 3 * 16 * 2]);
        let err = validate_fp8_scale_view(
            "model.language_model.layers.0.mlp.experts.0.up_proj.weight_scale_inv",
            &scale,
            &[512, 2048],
            128,
            128,
        )
        .expect_err("wrong scale shape should fail");
        let message = err.to_string();
        assert!(message.contains("shape mismatch"));
        assert!(message.contains("ceil(rows/128)"));
    }

    #[test]
    fn missing_config_file_error_includes_path_and_hint() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.json");
        let err = Qwen35HfConfig::from_json_file(&path).expect_err("missing config should fail");
        let message = err.to_string();
        assert!(message.contains(&path.display().to_string()));
        assert!(message.contains("OPD checkpoint directory"));
        assert!(message.contains("readable"));
    }

    #[test]
    fn load_non_directory_error_includes_hint_and_leaves_store_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("missing-model-dir");
        let mut store = TensorStore::default();

        let err = match load_qwen35_from_hf_dir(&missing, &mut store) {
            Ok(_) => panic!("non-directory load should fail"),
            Err(err) => err,
        };

        let message = err.to_string();
        assert!(message.contains(&missing.display().to_string()));
        assert!(message.contains("not a directory"));
        assert!(message.contains("config.json"));
        assert!(message.contains("model.safetensors"));
        assert!(
            store.tensors.is_empty(),
            "non-directory failure must not allocate model tensors"
        );
    }

    #[test]
    fn load_missing_safetensors_error_includes_hint_and_leaves_store_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.json"), QWEN35_NESTED_CONFIG_JSON)
            .expect("write config");
        let mut store = TensorStore::default();

        let err = match load_qwen35_from_hf_dir(dir.path(), &mut store) {
            Ok(_) => panic!("missing safetensors load should fail"),
            Err(err) => err,
        };

        let message = err.to_string();
        assert!(message.contains("no safetensors shards found"));
        assert!(message.contains("model.safetensors"));
        assert!(message.contains("model.safetensors.index.json"));
        assert!(
            store.tensors.is_empty(),
            "missing-shard failure must not allocate model tensors"
        );
    }

    #[test]
    fn load_missing_weight_map_error_includes_hint_and_leaves_store_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.json"), QWEN35_NESTED_CONFIG_JSON)
            .expect("write config");
        std::fs::write(dir.path().join("model.safetensors.index.json"), "{}").expect("write index");
        let mut store = TensorStore::default();

        let err = match load_qwen35_from_hf_dir(dir.path(), &mut store) {
            Ok(_) => panic!("index without weight_map should fail"),
            Err(err) => err,
        };

        let message = err.to_string();
        assert!(message.contains("missing weight_map object"));
        assert!(message.contains("regenerate"));
        assert!(message.contains("model.safetensors"));
        assert!(
            store.tensors.is_empty(),
            "invalid-index failure must not allocate model tensors"
        );
    }

    #[test]
    fn load_missing_tensor_rolls_back_store_and_checkpoint_writes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sentinel = 42.0_f32;
        write_partial_tiny_qwen35_checkpoint(dir.path(), sentinel);

        let mut store = TensorStore::default();
        let err = match load_qwen35_from_hf_dir(dir.path(), &mut store) {
            Ok(_) => panic!("partial checkpoint should fail on missing tensors"),
            Err(err) => err,
        };

        let message = err.to_string();
        assert!(message.contains("missing tensor"));
        assert!(message.contains("Hint: verify"));
        assert!(message.contains("model.safetensors.index.json"));
        assert!(message.contains("HF-compatible"));
        assert!(
            store.tensors.is_empty(),
            "failed checkpoint load must roll back scratch eval model allocation"
        );
    }

    #[test]
    fn load_failure_preserves_existing_store_tensors_and_free_list() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_partial_tiny_qwen35_checkpoint(dir.path(), 42.0);
        let mut store = TensorStore::default();
        let keep_id = store.alloc(Tensor::new(vec![1.25], vec![1], false).expect("keep tensor"));
        let free_id = store.alloc(Tensor::new(vec![9.0], vec![1], false).expect("free tensor"));
        store.free(free_id).expect("free scratch slot");
        let before_free_ids = store.free_ids.clone();
        let before_len = store.tensors.len();

        let err = load_qwen35_from_hf_dir(dir.path(), &mut store)
            .expect_err("partial checkpoint should fail on missing tensors");

        assert!(err.to_string().contains("missing tensor"));
        assert_eq!(
            store.tensors.len(),
            before_len,
            "rollback must restore TensorStore length"
        );
        assert_eq!(
            store.free_ids, before_free_ids,
            "rollback must restore the free-list exactly"
        );
        assert_eq!(
            store
                .get(keep_id)
                .expect("kept tensor survives rollback")
                .data,
            vec![1.25]
        );
        assert!(
            store.get(free_id).is_none(),
            "model allocation that reused a pre-existing free slot must be cleared"
        );
    }
}
