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
//! - It does not (yet) support quantized checkpoints. Float and BF16/F16
//!   safetensors only; quantized weights surface as a `LoaderError::UnsupportedDtype`.
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

use autograd::{Tensor, TensorId, TensorStore};
use half::{bf16, f16};
use memmap2::Mmap;
use qwen35_spec::{LayerType, Qwen35Config, Qwen35ConfigError};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::Deserialize;
use thiserror::Error;

use crate::qwen35::{Qwen35Error, Qwen35Model};

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
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("safetensors: {0}")]
    Safetensors(String),
    #[error("config: {0}")]
    Config(#[from] Qwen35ConfigError),
    #[error("model: {0}")]
    Model(#[from] Qwen35Error),
    #[error("shape mismatch for {name}: model expects {expected:?}, safetensors has {got:?}{hint}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
        hint: String,
    },
    #[error("missing tensor {0} in safetensors (and no fallback rule applies)")]
    MissingTensor(String),
    #[error(
        "unsupported dtype {0:?} for {1}. Hint: OPD loader currently accepts F32, BF16, and F16 safetensors only; convert quantized checkpoints before loading."
    )]
    UnsupportedDtype(Dtype, String),
    #[error("autograd: {0}")]
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
                    "layer_types length {} != num_hidden_layers {num_layers}",
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
            num_experts: 0,
            num_experts_per_tok: 0,
            decoder_sparse_step: 1,
            moe_intermediate_size: 0,
            shared_expert_intermediate_size: 0,
            norm_topk_prob: true,
            mlp_only_layers: Vec::new(),
            full_attn_gated: true,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

fn merge_token_ids(mut text: serde_json::Value, parent: &serde_json::Value) -> serde_json::Value {
    if let Some(obj) = text.as_object_mut() {
        for key in ["eos_token_id", "bos_token_id"] {
            if obj.get(key).is_none() {
                if let Some(v) = parent.get(key) {
                    obj.insert(key.to_string(), v.clone());
                }
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
                LoaderError::Custom(format!("{} missing weight_map object", index.display()))
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

// ─────────────────────────── public entry point ──────────────────────────────

/// Load a HF-format Qwen3 / Qwen3.5 checkpoint into a fresh [`Qwen35Model`].
///
/// The model is initialized via [`Qwen35Model::new_for_eval`] (frozen, no
/// LoRA, no `requires_grad`) and every parameter slot is overwritten with
/// the data read from the safetensors shards in `dir`.
///
/// Returns the constructed model. On any name/shape mismatch the function
/// returns a [`LoaderError`] before committing any checkpoint tensor
/// overwrites. Directory/config/shard-discovery failures leave `store`
/// untouched; tensor-level failures can leave the scratch eval model
/// allocation in `store`, so callers should discard the store after any
/// returned `Err`.
pub fn load_qwen35_from_hf_dir(dir: &Path, store: &mut TensorStore) -> Result<Qwen35Model> {
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

    // 3) Qwen35Config → eval-init Qwen35Model.
    let model = Qwen35Model::new_for_eval(&cfg, store)?;
    let param_map = model.param_name_map();

    // 4) Preflight every tensor before writing any checkpoint data into the
    //    store. This keeps missing/mismatched later tensors from leaving
    //    partially materialized checkpoint weights behind.
    let load_plan = plan_tensor_loads(
        &param_map,
        &cfg,
        schema,
        &hf_name_to_shard,
        &safetensors_views,
        store,
    )?;

    // 5) Materialize each train parameter from the safetensors.
    for planned in &load_plan {
        load_planned_tensor_into_slot(planned, &safetensors_views, store)?;
    }

    Ok(model)
}

struct PlannedTensorLoad {
    hf_name: String,
    train_name: String,
    id: TensorId,
    expected_shape: Vec<usize>,
    shard_idx: usize,
}

fn plan_tensor_loads(
    param_map: &HashMap<&'static str, TensorId>,
    cfg: &Qwen35Config,
    schema: HfSchema,
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
    validate_supported_dtype(&view, hf_name)?;

    let expected_shape = store
        .get(id)
        .map(|t| t.shape.clone())
        .ok_or_else(|| LoaderError::Custom(format!("missing slot for {train_name}")))?;
    if expected_shape != got_shape {
        let hint = q_proj_gate_hint(train_name, &expected_shape, &got_shape);
        return Err(LoaderError::ShapeMismatch {
            name: train_name.to_owned(),
            expected: expected_shape,
            got: got_shape,
            hint,
        });
    }

    Ok(PlannedTensorLoad {
        hf_name: hf_name.to_owned(),
        train_name: train_name.to_owned(),
        id,
        expected_shape,
        shard_idx,
    })
}

fn validate_supported_dtype(view: &TensorView<'_>, name: &str) -> Result<()> {
    match view.dtype() {
        Dtype::F32 | Dtype::BF16 | Dtype::F16 => Ok(()),
        other => Err(LoaderError::UnsupportedDtype(other, name.to_owned())),
    }
}

fn load_planned_tensor_into_slot(
    planned: &PlannedTensorLoad,
    safetensors_views: &[SafeTensors<'_>],
    store: &mut TensorStore,
) -> Result<()> {
    let view = safetensors_views[planned.shard_idx]
        .tensor(&planned.hf_name)
        .map_err(|err| LoaderError::Safetensors(format!("{}: {err}", planned.hf_name)))?;
    let data = dtype_to_f32(&view, &planned.hf_name)?;

    let tensor = Tensor::new(data, planned.expected_shape.clone(), false).map_err(|err| {
        LoaderError::Custom(format!(
            "failed to materialize {} from {}: {err}. Hint: verify the safetensors \
             data length matches the validated checkpoint shape.",
            planned.train_name, planned.hf_name
        ))
    })?;
    store.tensors[planned.id] = Some(tensor);
    Ok(())
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
        shape: Vec<usize>,
        bytes: Vec<u8>,
    }

    impl TestTensorView {
        fn from_f32(shape: Vec<usize>, values: &[f32]) -> Self {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Self { shape, bytes }
        }
    }

    impl safetensors::View for TestTensorView {
        fn dtype(&self) -> Dtype {
            Dtype::F32
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
    fn load_missing_tensor_preflights_before_checkpoint_weight_writes() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.json"), TINY_QWEN35_CONFIG_JSON)
            .expect("write config");
        let sentinel = 42.0_f32;
        let embed_values = [sentinel; 8 * 4];
        let embed = TestTensorView::from_f32(vec![8, 4], &embed_values);
        serialize_to_file(
            vec![(
                "model.language_model.embed_tokens.weight".to_string(),
                embed,
            )],
            None,
            &dir.path().join("model.safetensors"),
        )
        .expect("write partial safetensors");

        let mut store = TensorStore::default();
        let err = match load_qwen35_from_hf_dir(dir.path(), &mut store) {
            Ok(_) => panic!("partial checkpoint should fail on missing tensors"),
            Err(err) => err,
        };

        let message = err.to_string();
        assert!(message.contains("missing tensor"));
        assert!(
            !store.tensors.is_empty(),
            "tensor-level failure happens after eval model allocation"
        );
        let wrote_sentinel = store
            .tensors
            .iter()
            .filter_map(|slot| slot.as_ref())
            .flat_map(|tensor| tensor.data.iter())
            .any(|value| value.to_bits() == sentinel.to_bits());
        assert!(
            !wrote_sentinel,
            "checkpoint tensor data must not be written before the whole load plan validates"
        );
    }
}
