//! Cold path: safetensors loading, paging metadata, and config validation.
//!
//! Stable once correct — consolidated per the v2 churn-weighted module design.
//! Holds the BF16 safetensors loader (`SafetensorLoader`/`SafetensorIndex`/
//! `OwnedTensor`), `CudaModel::from_safetensors` weight upload, the per-step
//! paging metadata (`PageMeta`/`for_slot`), and the clean-BF16 config gate.
//! Pure relocation from `model.rs` — identical numerics.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, ensure};
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, PagedKVPool};
use cudarc::driver::CudaSlice;
use qwen3_spec::Qwen3Config;
use safetensors::{SafeTensors, tensor::Dtype};

use crate::model::{Attention, CudaModel, Mlp, TransformerBlock};
use crate::ops::{precompute_rope, upload_i32};

const DEFAULT_ROPE_CACHE_LEN: usize = 32_768;

impl CudaModel {
    pub(crate) fn from_safetensors(model_path: &Path) -> Result<Self> {
        let config = Qwen3Config::from_json_file(model_path.join("config.json"))
            .with_context(|| format!("load Qwen3 config from {}", model_path.display()))?;
        validate_clean_bf16_config(&config)?;

        let ctx = DeviceContext::new()?;
        let loader = SafetensorLoader::new(model_path)?;

        let embed_tokens = loader.load_matrix(&ctx, config.embed_tokens_tensor_name())?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(loader.load_matrix(&ctx, config.lm_head_tensor_name())?)
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let names = config.layer_tensor_names(layer_idx);
            layers.push(TransformerBlock {
                input_layernorm: loader.load_vec(&ctx, &names.input_layernorm)?,
                attention: Attention {
                    q_proj: loader.load_matrix(&ctx, &names.q_proj)?,
                    k_proj: loader.load_matrix(&ctx, &names.k_proj)?,
                    v_proj: loader.load_matrix(&ctx, &names.v_proj)?,
                    o_proj: loader.load_matrix(&ctx, &names.o_proj)?,
                    q_norm: loader.load_vec(&ctx, &names.q_norm)?,
                    k_norm: loader.load_vec(&ctx, &names.k_norm)?,
                },
                post_attention_layernorm: loader.load_vec(&ctx, &names.post_attention_layernorm)?,
                mlp: Mlp {
                    gate_proj: loader.load_matrix(&ctx, &names.mlp_gate_proj)?,
                    up_proj: loader.load_matrix(&ctx, &names.mlp_up_proj)?,
                    down_proj: loader.load_matrix(&ctx, &names.mlp_down_proj)?,
                },
            });
        }
        let norm = loader.load_vec(&ctx, config.norm_tensor_name())?;

        let rope_len = config
            .rope_cache_len_hint()
            .unwrap_or(DEFAULT_ROPE_CACHE_LEN)
            .max(DEFAULT_ROPE_CACHE_LEN);
        let (cos_cache, sin_cache) = precompute_rope(
            &ctx,
            config.head_dim,
            rope_len,
            config.rope_theta,
            config.rope_scaling.as_ref(),
        )?;
        ctx.sync()?;

        Ok(Self {
            ctx,
            config,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
        })
    }
}

pub(crate) fn validate_clean_bf16_config(config: &Qwen3Config) -> Result<()> {
    ensure!(
        config.hidden_size == config.num_attention_heads * config.head_dim,
        "Qwen3 hidden_size {} must equal num_attention_heads {} * head_dim {}",
        config.hidden_size,
        config.num_attention_heads,
        config.head_dim
    );
    ensure!(
        config
            .num_attention_heads
            .is_multiple_of(config.num_key_value_heads),
        "Qwen3 num_attention_heads must be divisible by num_key_value_heads"
    );
    Ok(())
}

#[derive(Debug)]
pub(crate) struct PageMeta {
    pub(crate) q_indptr: CudaSlice<i32>,
    pub(crate) kv_indptr: CudaSlice<i32>,
    pub(crate) kv_indices: CudaSlice<i32>,
    pub(crate) kv_last_page_len: CudaSlice<i32>,
    pub(crate) page_table_offsets: CudaSlice<i32>,
    pub(crate) start_positions: CudaSlice<i32>,
    pub(crate) positions: CudaSlice<i32>,
    pub(crate) seq_len: usize,
    pub(crate) num_pages: usize,
}

impl PageMeta {
    pub(crate) fn for_slot(
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let total_len = start_pos + seq_len;
        ensure!(
            pool.seq_len(slot) == total_len,
            "PagedKVPool seq_len {} != materialized total_len {} for slot {}",
            pool.seq_len(slot),
            total_len,
            slot
        );
        let num_pages = total_len.div_ceil(pool.page_size);
        let pages = pool.page_indices(slot);
        ensure!(
            pages.len() >= num_pages,
            "slot {} has {} pages, expected at least {}",
            slot,
            pages.len(),
            num_pages
        );
        let last_page_len = total_len % pool.page_size;
        let last_page_len = if last_page_len == 0 {
            pool.page_size
        } else {
            last_page_len
        };
        let page_ids = pages[..num_pages]
            .iter()
            .map(|&page| page as i32)
            .collect::<Vec<_>>();
        Ok(Self {
            q_indptr: upload_i32(ctx, &[0, seq_len as i32])?,
            kv_indptr: upload_i32(ctx, &[0, num_pages as i32])?,
            kv_indices: upload_i32(ctx, &page_ids)?,
            kv_last_page_len: upload_i32(ctx, &[last_page_len as i32])?,
            page_table_offsets: upload_i32(ctx, &[0])?,
            start_positions: upload_i32(ctx, &[start_pos as i32])?,
            positions: upload_i32(ctx, &[(total_len - 1) as i32])?,
            seq_len,
            num_pages,
        })
    }
}

struct SafetensorLoader {
    base: PathBuf,
    shards: Vec<PathBuf>,
    weight_map: HashMap<String, usize>,
    /// Read-once cache of shard bytes. Without this, loading N tensors re-reads +
    /// re-deserializes the whole shard file N times — O(N × file_size) I/O, which
    /// stalls model load for minutes on a multi-hundred-tensor model.
    shard_cache: std::cell::RefCell<HashMap<usize, Vec<u8>>>,
}

impl SafetensorLoader {
    fn new(base: &Path) -> Result<Self> {
        let index_path = base.join("model.safetensors.index.json");
        if index_path.exists() {
            let content = fs::read_to_string(&index_path)
                .with_context(|| format!("read {}", index_path.display()))?;
            let index: SafetensorIndex = serde_json::from_str(&content)
                .with_context(|| format!("parse {}", index_path.display()))?;
            let mut shards = Vec::<PathBuf>::new();
            let mut file_to_idx = HashMap::<String, usize>::new();
            let mut weight_map = HashMap::new();
            for (name, file) in index.weight_map {
                let idx = match file_to_idx.get(&file) {
                    Some(&idx) => idx,
                    None => {
                        let idx = shards.len();
                        shards.push(base.join(&file));
                        file_to_idx.insert(file, idx);
                        idx
                    }
                };
                weight_map.insert(name, idx);
            }
            return Ok(Self {
                base: base.to_path_buf(),
                shards,
                weight_map,
                shard_cache: std::cell::RefCell::new(HashMap::new()),
            });
        }

        let single = base.join("model.safetensors");
        if single.exists() {
            return Ok(Self {
                base: base.to_path_buf(),
                shards: vec![single],
                weight_map: HashMap::new(),
                shard_cache: std::cell::RefCell::new(HashMap::new()),
            });
        }

        let mut shards = fs::read_dir(base)
            .with_context(|| format!("scan {}", base.display()))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
            .collect::<Vec<_>>();
        shards.sort();
        ensure!(
            !shards.is_empty(),
            "no safetensors shards found under {}",
            base.display()
        );
        Ok(Self {
            base: base.to_path_buf(),
            shards,
            weight_map: HashMap::new(),
            shard_cache: std::cell::RefCell::new(HashMap::new()),
        })
    }

    fn load_vec(&self, ctx: &DeviceContext, name: &str) -> Result<DeviceVec> {
        let tensor = self.load_tensor(name)?;
        ensure!(
            tensor.shape.len() == 1,
            "{name}: expected 1D BF16 tensor, got shape {:?}",
            tensor.shape
        );
        DeviceVec::from_safetensors(ctx, &tensor.bytes)
            .with_context(|| format!("upload tensor {name}"))
    }

    fn load_matrix(&self, ctx: &DeviceContext, name: &str) -> Result<DeviceMatrix> {
        let tensor = self.load_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D BF16 tensor, got shape {:?}",
            tensor.shape
        );
        DeviceMatrix::from_safetensors(ctx, &tensor.bytes, tensor.shape[0], tensor.shape[1])
            .with_context(|| format!("upload tensor {name}"))
    }

    fn load_tensor(&self, name: &str) -> Result<OwnedTensor> {
        if let Some(&idx) = self.weight_map.get(name) {
            return self.load_tensor_from_shard(idx, name);
        }
        for idx in 0..self.shards.len() {
            if let Ok(tensor) = self.load_tensor_from_shard(idx, name) {
                return Ok(tensor);
            }
        }
        Err(anyhow!(
            "tensor {name} not found in safetensors under {}",
            self.base.display()
        ))
    }

    fn load_tensor_from_shard(&self, idx: usize, name: &str) -> Result<OwnedTensor> {
        let path = self
            .shards
            .get(idx)
            .ok_or_else(|| anyhow!("shard index {idx} out of range"))?;
        // Read each shard at most once (the data views below are zero-copy over
        // these bytes; re-reading per tensor was O(tensors × file_size)).
        if !self.shard_cache.borrow().contains_key(&idx) {
            let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
            self.shard_cache.borrow_mut().insert(idx, bytes);
        }
        let cache = self.shard_cache.borrow();
        let bytes = cache.get(&idx).expect("shard bytes just cached");
        let tensors = SafeTensors::deserialize(bytes)
            .with_context(|| format!("deserialize {}", path.display()))?;
        let view = tensors
            .tensor(name)
            .with_context(|| format!("find tensor {name} in {}", path.display()))?;
        ensure!(
            view.dtype() == Dtype::BF16,
            "{name}: R6 clean CUDA path accepts BF16 only, got {:?}",
            view.dtype()
        );
        Ok(OwnedTensor {
            shape: view.shape().to_vec(),
            bytes: view.data().to_vec(),
        })
    }
}

#[derive(serde::Deserialize)]
struct SafetensorIndex {
    weight_map: HashMap<String, String>,
}

struct OwnedTensor {
    shape: Vec<usize>,
    bytes: Vec<u8>,
}
