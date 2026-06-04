//! Cold path: safetensors loading, paging metadata, and config validation.
//!
//! Holds the BF16 safetensors loader, `CudaModel::from_safetensors` weight
//! upload, the per-step paging metadata (`PageMeta`), and the BF16 config gate.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, PagedKVPool};
use cudarc::driver::CudaSlice;
use deepseek_spec::Shard;
use infer_topo::{ShardingSpec, TpConfig};
use qwen3_spec::Qwen3Config;
use safetensors::{SafeTensors, tensor::Dtype};

use crate::model::{Attention, CudaModel, Mlp, TransformerBlock};
use crate::ops::{precompute_rope, upload_i32};

const DEFAULT_ROPE_CACHE_LEN: usize = 32_768;

impl CudaModel {
    pub(crate) fn from_safetensors(model_path: &Path) -> Result<Self> {
        let tp = build_tp_runtime()?;
        Self::from_safetensors_with_tp(model_path, tp)
    }

    /// Load with an explicit [`crate::tp::TpRuntime`] (tests inject a single-GPU
    /// runtime).
    pub(crate) fn from_safetensors_with_tp(
        model_path: &Path,
        tp: crate::tp::TpRuntime,
    ) -> Result<Self> {
        let config = Qwen3Config::from_json_file(model_path.join("config.json"))
            .with_context(|| format!("load Qwen3 config from {}", model_path.display()))?;
        validate_clean_bf16_config(&config)?;

        let tp_cfg = *tp.config();
        // Per-rank GQA head counts. `head_shard` requires both counts divide the
        // world size, keeping every rank's attention shape uniform — the kv8
        // TileLang kernels and the all-reduce both rely on it.
        let (local_q_heads, local_kv_heads) = if tp_cfg.is_single() {
            (config.num_attention_heads, config.num_key_value_heads)
        } else {
            infer_topo::head_shard(
                config.num_attention_heads,
                config.num_key_value_heads,
                &tp_cfg,
            )
            .map_err(|e| anyhow!("TP head shard failed: {e}"))?
        };

        let ctx = DeviceContext::new()?;
        let loader = SafetensorLoader::new(model_path)?;

        // lm_head / embed_tokens stay replicated across ranks (avoids an
        // all-gather of logits); only per-layer Q/K/V/O + MLP are sharded.
        let embed_tokens = loader.load_matrix(&ctx, config.embed_tokens_tensor_name())?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(loader.load_matrix(&ctx, config.lm_head_tensor_name())?)
        };

        let head_dim = config.head_dim;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let names = config.layer_tensor_names(layer_idx);
            let (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj) =
                if tp_cfg.is_single() {
                    // Single GPU: full tensors.
                    (
                        loader.load_matrix(&ctx, &names.q_proj)?,
                        loader.load_matrix(&ctx, &names.k_proj)?,
                        loader.load_matrix(&ctx, &names.v_proj)?,
                        loader.load_matrix(&ctx, &names.o_proj)?,
                        loader.load_matrix(&ctx, &names.mlp_gate_proj)?,
                        loader.load_matrix(&ctx, &names.mlp_up_proj)?,
                        loader.load_matrix(&ctx, &names.mlp_down_proj)?,
                    )
                } else {
                    // TP: Q/K/V column-parallel on whole-head boundaries (so the
                    // o_proj input shard and head count agree); gate/up plain
                    // column-parallel; o_proj/down_proj row-parallel.
                    (
                        loader.load_qkv_head_sharded(
                            &ctx,
                            &names.q_proj,
                            local_q_heads,
                            head_dim,
                            &tp_cfg,
                        )?,
                        loader.load_qkv_head_sharded(
                            &ctx,
                            &names.k_proj,
                            local_kv_heads,
                            head_dim,
                            &tp_cfg,
                        )?,
                        loader.load_qkv_head_sharded(
                            &ctx,
                            &names.v_proj,
                            local_kv_heads,
                            head_dim,
                            &tp_cfg,
                        )?,
                        loader.load_matrix_sharded(
                            &ctx,
                            &names.o_proj,
                            infer_topo::ParallelLinearKind::Row,
                            &tp_cfg,
                        )?,
                        loader.load_matrix_sharded(
                            &ctx,
                            &names.mlp_gate_proj,
                            infer_topo::ParallelLinearKind::Column,
                            &tp_cfg,
                        )?,
                        loader.load_matrix_sharded(
                            &ctx,
                            &names.mlp_up_proj,
                            infer_topo::ParallelLinearKind::Column,
                            &tp_cfg,
                        )?,
                        loader.load_matrix_sharded(
                            &ctx,
                            &names.mlp_down_proj,
                            infer_topo::ParallelLinearKind::Row,
                            &tp_cfg,
                        )?,
                    )
                };
            layers.push(TransformerBlock {
                input_layernorm: loader.load_vec(&ctx, &names.input_layernorm)?,
                attention: Attention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm: loader.load_vec(&ctx, &names.q_norm)?,
                    k_norm: loader.load_vec(&ctx, &names.k_norm)?,
                },
                post_attention_layernorm: loader.load_vec(&ctx, &names.post_attention_layernorm)?,
                mlp: Some(Mlp {
                    gate_proj,
                    up_proj,
                    down_proj,
                }),
                moe: None,
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
            tp,
            local_q_heads,
            local_kv_heads,
            moe_config: None,
        })
    }
}

/// Build the tensor-parallel runtime for model load. Multi-rank `nccl` builds
/// take the NCCL `unique_id` from `INFER_NCCL_UNIQUE_ID`; otherwise the no-op
/// single runtime.
fn build_tp_runtime() -> Result<crate::tp::TpRuntime> {
    #[cfg(feature = "nccl")]
    {
        let cfg = crate::tp::resolve_tp_config_from_env().map_err(|e| anyhow!("{e}"))?;
        if !cfg.is_single() {
            let unique_id = nccl_unique_id_from_env()?;
            return crate::tp::TpRuntime::from_env_with_nccl(unique_id);
        }
    }
    crate::tp::TpRuntime::from_env().map_err(|e| anyhow!("{e}"))
}

/// Decode the NCCL `unique_id` from `INFER_NCCL_UNIQUE_ID` (128 hex bytes = 256
/// chars).
#[cfg(feature = "nccl")]
pub(crate) fn nccl_unique_id_from_env() -> Result<cuda_kernels::ffi::nccl::ncclUniqueId> {
    let hex = std::env::var("INFER_NCCL_UNIQUE_ID").map_err(|_| {
        anyhow!(
            "multi-rank TP requires INFER_NCCL_UNIQUE_ID (128 hex-encoded bytes \
             from the launcher's ncclGetUniqueId broadcast)"
        )
    })?;
    let hex = hex.trim();
    ensure!(
        hex.len() == 256,
        "INFER_NCCL_UNIQUE_ID must be 256 hex chars (128 bytes), got {}",
        hex.len()
    );
    let mut internal = [0i8; 128];
    for (i, slot) in internal.iter_mut().enumerate() {
        let byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
            .with_context(|| format!("INFER_NCCL_UNIQUE_ID bad hex at byte {i}"))?;
        *slot = byte as i8;
    }
    Ok(cuda_kernels::ffi::nccl::ncclUniqueId { internal })
}

pub(crate) fn validate_clean_bf16_config(config: &Qwen3Config) -> Result<()> {
    // Qwen3 decouples head_dim from hidden_size/num_heads (e.g. Qwen3-0.6B:
    // hidden 1024, heads 16, head_dim 128), so `hidden_size == heads*head_dim`
    // is NOT an invariant. Real constraints: head_dim==128 (checked in model.rs)
    // + GQA divisibility below.
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

pub(crate) struct SafetensorLoader {
    base: PathBuf,
    shards: Vec<PathBuf>,
    weight_map: HashMap<String, usize>,
    /// Read-once cache of shard bytes: without it, loading N tensors re-reads the
    /// whole shard N times (O(N × file_size) I/O).
    shard_cache: std::cell::RefCell<HashMap<usize, Vec<u8>>>,
}

impl SafetensorLoader {
    pub(crate) fn new(base: &Path) -> Result<Self> {
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

    pub(crate) fn load_vec(&self, ctx: &DeviceContext, name: &str) -> Result<DeviceVec> {
        let tensor = self.load_tensor(name)?;
        ensure!(
            tensor.shape.len() == 1,
            "{name}: expected 1D BF16 tensor, got shape {:?}",
            tensor.shape
        );
        DeviceVec::from_safetensors(ctx, &tensor.bytes)
            .with_context(|| format!("upload tensor {name}"))
    }

    /// Load a 1D vector that may be BF16 or F32 on disk (Qwen3.5 `dt_bias` ships
    /// BF16; this normalizes F32→BF16 so the recurrent kernel's bf16 input ABI
    /// holds), uploaded as a [`DeviceVec`].
    pub(crate) fn load_vec_any(&self, ctx: &DeviceContext, name: &str) -> Result<DeviceVec> {
        let tensor = self.load_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 1,
            "{name}: expected 1D tensor, got shape {:?}",
            tensor.shape
        );
        DeviceVec::from_safetensors(ctx, Self::dsv4_bytes_to_bf16(name, &tensor)?.as_ref())
            .with_context(|| format!("upload vec {name}"))
    }

    /// Load a 1D F32 tensor (Qwen3.5 `A_log` / gated-norm scale) directly into a
    /// device `f32` slice — the recurrent + gated-RMSNorm kernels read these as
    /// `*const f32`. Accepts F32 (passthrough) or BF16 (widened to F32).
    pub(crate) fn load_f32_vec(&self, ctx: &DeviceContext, name: &str) -> Result<CudaSlice<f32>> {
        let tensor = self.load_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 1,
            "{name}: expected 1D tensor, got shape {:?}",
            tensor.shape
        );
        let host: Vec<f32> = match tensor.dtype {
            Dtype::F32 => tensor
                .bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::BF16 => tensor
                .bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            other => bail!("{name}: expected F32/BF16 1D tensor, got {other:?}"),
        };
        ctx.stream
            .clone_htod(&host)
            .map_err(|e| anyhow!("upload f32 vec {name}: {e}"))
    }

    /// Load a Qwen3.5 depthwise conv1d weight (`[qkv_dim, 1, kernel]` BF16) as a
    /// flat `[qkv_dim*kernel]` [`DeviceVec`] (the conv1d kernel's channel-major
    /// ABI). The singleton middle dim is squeezed by the flat byte upload.
    pub(crate) fn load_conv1d_vec(&self, ctx: &DeviceContext, name: &str) -> Result<DeviceVec> {
        let tensor = self.load_tensor(name)?;
        ensure!(
            !tensor.shape.is_empty(),
            "{name}: expected conv1d tensor, got rank-0"
        );
        DeviceVec::from_safetensors(ctx, &tensor.bytes)
            .with_context(|| format!("upload conv1d {name}"))
    }

    pub(crate) fn load_matrix(&self, ctx: &DeviceContext, name: &str) -> Result<DeviceMatrix> {
        let tensor = self.load_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D BF16 tensor, got shape {:?}",
            tensor.shape
        );
        DeviceMatrix::from_safetensors(ctx, &tensor.bytes, tensor.shape[0], tensor.shape[1])
            .with_context(|| format!("upload tensor {name}"))
    }

    /// Load a 2D BF16 weight, slice it to this TP rank via [`crate::shard_slice`],
    /// and upload the shard. Column kind (`q/k/v/gate/up`) slices rows; row kind
    /// (`o/down`) slices cols. Single-GPU is the identity slice.
    fn load_matrix_sharded(
        &self,
        ctx: &DeviceContext,
        name: &str,
        kind: infer_topo::ParallelLinearKind,
        tp: &infer_topo::TpConfig,
    ) -> Result<DeviceMatrix> {
        const BF16_ELEM_SIZE: usize = 2;
        let tensor = self.load_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D BF16 tensor, got shape {:?}",
            tensor.shape
        );
        let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
        let sharded = match kind {
            infer_topo::ParallelLinearKind::Column => {
                let spec = infer_topo::column_shard(rows, tp);
                crate::shard_slice::shard_column_parallel(
                    &tensor.bytes,
                    rows,
                    cols,
                    BF16_ELEM_SIZE,
                    &spec,
                )?
            }
            infer_topo::ParallelLinearKind::Row => {
                let spec = infer_topo::row_shard(cols, tp);
                crate::shard_slice::shard_row_parallel(
                    &tensor.bytes,
                    rows,
                    cols,
                    BF16_ELEM_SIZE,
                    &spec,
                )?
            }
        };
        DeviceMatrix::from_safetensors(ctx, &sharded.bytes, sharded.rows, sharded.cols)
            .with_context(|| format!("upload sharded tensor {name}"))
    }

    /// Load a head-aligned column-parallel Q/K/V projection for this TP rank.
    ///
    /// The split MUST land on whole-head boundaries (so head count, o_proj input
    /// shard, and RoPE/RMSNorm agree); a plain `column_shard` on the raw output
    /// dim would split a head on the last rank. `local_heads` (from `head_shard`,
    /// which requires global heads divide world size) gives a contiguous shard at
    /// offset `rank * local_heads * head_dim`.
    fn load_qkv_head_sharded(
        &self,
        ctx: &DeviceContext,
        name: &str,
        local_heads: usize,
        head_dim: usize,
        tp: &infer_topo::TpConfig,
    ) -> Result<DeviceMatrix> {
        const BF16_ELEM_SIZE: usize = 2;
        let tensor = self.load_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D BF16 tensor, got shape {:?}",
            tensor.shape
        );
        let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
        let total_rows = rows;
        let local_rows = local_heads * head_dim;
        let offset = tp.rank * local_rows;
        ensure!(
            offset + local_rows <= total_rows,
            "{name}: head shard [{offset}, {}) exceeds rows {total_rows} \
             (local_heads={local_heads}, head_dim={head_dim}, rank={})",
            offset + local_rows,
            tp.rank
        );
        let spec = infer_topo::ShardingSpec {
            offset,
            size: local_rows,
            total: total_rows,
        };
        let sharded = crate::shard_slice::shard_column_parallel(
            &tensor.bytes,
            rows,
            cols,
            BF16_ELEM_SIZE,
            &spec,
        )?;
        DeviceMatrix::from_safetensors(ctx, &sharded.bytes, sharded.rows, sharded.cols)
            .with_context(|| format!("upload head-sharded tensor {name}"))
    }

    /// Load this EP rank's per-expert MoE weights for one layer (gate/up/down +
    /// router gate + shared expert) and build the per-expert weight-pointer
    /// tables. Naming follows the mlx-lm `qwen3_5_moe` HF convention. Only the
    /// experts in `split.local_expert_start..local_expert_end()` are loaded
    /// (single-GPU loads all).
    pub(crate) fn load_moe_layer_experts(
        &self,
        ctx: &DeviceContext,
        layer_prefix: &str,
        split: &crate::moe_config::ExpertSplit,
    ) -> Result<MoeLayerWeights> {
        let mut gate = Vec::with_capacity(split.experts_per_rank);
        let mut up = Vec::with_capacity(split.experts_per_rank);
        let mut down = Vec::with_capacity(split.experts_per_rank);
        for e in split.local_expert_start..split.local_expert_end() {
            let base = format!("{layer_prefix}.mlp.experts.{e}");
            gate.push(self.load_matrix(ctx, &format!("{base}.gate_proj.weight"))?);
            up.push(self.load_matrix(ctx, &format!("{base}.up_proj.weight"))?);
            down.push(self.load_matrix(ctx, &format!("{base}.down_proj.weight"))?);
        }
        let router_gate = self.load_matrix(ctx, &format!("{layer_prefix}.mlp.gate.weight"))?;
        let shared_prefix = format!("{layer_prefix}.mlp.shared_expert");
        let shared_gate = self.load_matrix(ctx, &format!("{shared_prefix}.gate_proj.weight"))?;
        let shared_up = self.load_matrix(ctx, &format!("{shared_prefix}.up_proj.weight"))?;
        let shared_down = self.load_matrix(ctx, &format!("{shared_prefix}.down_proj.weight"))?;
        let shared_gate_router = self.load_matrix(
            ctx,
            &format!("{layer_prefix}.mlp.shared_expert_gate.weight"),
        )?;

        // Per-expert weight-pointer tables (one device pointer per owned expert).
        let gate_refs: Vec<&DeviceMatrix> = gate.iter().collect();
        let up_refs: Vec<&DeviceMatrix> = up.iter().collect();
        let down_refs: Vec<&DeviceMatrix> = down.iter().collect();
        let gate_ptrs = cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &gate_refs)?;
        let up_ptrs = cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &up_refs)?;
        let down_ptrs = cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &down_refs)?;

        Ok(MoeLayerWeights {
            gate,
            up,
            down,
            gate_ptrs,
            up_ptrs,
            down_ptrs,
            router_gate,
            shared_gate,
            shared_up,
            shared_down,
            shared_gate_router,
        })
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
        let tensor = self.load_raw_from_shard(idx, name)?;
        ensure!(
            tensor.dtype == Dtype::BF16,
            "{name}: R6 clean CUDA path accepts BF16 only, got {:?}",
            tensor.dtype
        );
        Ok(tensor)
    }

    /// Dtype-agnostic shard read (DSv4 FP8/FP4/E8M0). Same read-once cache as the
    /// BF16 path; the typed gate lives in the callers.
    fn load_raw_from_shard(&self, idx: usize, name: &str) -> Result<OwnedTensor> {
        let path = self
            .shards
            .get(idx)
            .ok_or_else(|| anyhow!("shard index {idx} out of range"))?;
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
        Ok(OwnedTensor {
            shape: view.shape().to_vec(),
            bytes: view.data().to_vec(),
            dtype: view.dtype(),
        })
    }
}

// DSv4 FP8/FP4 + E8M0 loaders. Loader-only milestone: reachable from
// `Dsv4Model::from_dsv4_fp8_safetensors`, which the executor enum branch wires
// with the Piece 2/3 forward (see `feedback_necessity_not_callers`).
#[allow(dead_code)]
impl SafetensorLoader {
    fn load_raw_tensor(&self, name: &str) -> Result<OwnedTensor> {
        if let Some(&idx) = self.weight_map.get(name) {
            return self.load_raw_from_shard(idx, name);
        }
        for idx in 0..self.shards.len() {
            if let Ok(tensor) = self.load_raw_from_shard(idx, name) {
                return Ok(tensor);
            }
        }
        Err(anyhow!(
            "tensor {name} not found in safetensors under {}",
            self.base.display()
        ))
    }

    /// Normalize a DSv4 small 1D/2D tensor to BF16 bytes — these ship as BF16
    /// (norms) or F32 (attn_sink, router gate) depending on the checkpoint.
    fn dsv4_bytes_to_bf16<'a>(
        name: &str,
        tensor: &'a OwnedTensor,
    ) -> Result<std::borrow::Cow<'a, [u8]>> {
        match tensor.dtype {
            Dtype::BF16 => Ok(std::borrow::Cow::Borrowed(tensor.bytes.as_slice())),
            Dtype::F32 => Ok(std::borrow::Cow::Owned(
                tensor
                    .bytes
                    .chunks_exact(4)
                    .flat_map(|c| {
                        half::bf16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .to_le_bytes()
                    })
                    .collect(),
            )),
            other => anyhow::bail!("{name}: DSv4 tensor expected BF16/F32, got {other:?}"),
        }
    }

    /// Load a DSv4 1D norm/bias vector (q_norm, kv_norm, attn_sink, layer norms,
    /// gate bias) — BF16 or F32 in the checkpoint, normalized to BF16.
    pub(crate) fn load_dsv4_vec(&self, ctx: &DeviceContext, name: &str) -> Result<DeviceVec> {
        let tensor = self.load_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 1,
            "{name}: expected 1D tensor, got shape {:?}",
            tensor.shape
        );
        DeviceVec::from_safetensors(ctx, Self::dsv4_bytes_to_bf16(name, &tensor)?.as_ref())
            .with_context(|| format!("upload DSv4 vec {name}"))
    }

    /// Load a DSv4 2D router gate (the only non-FP8 2D weight) — BF16 or F32 in
    /// the checkpoint, normalized to BF16.
    pub(crate) fn load_dsv4_bf16_matrix(
        &self,
        ctx: &DeviceContext,
        name: &str,
    ) -> Result<DeviceMatrix> {
        let tensor = self.load_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D tensor, got shape {:?}",
            tensor.shape
        );
        DeviceMatrix::from_safetensors(
            ctx,
            Self::dsv4_bytes_to_bf16(name, &tensor)?.as_ref(),
            tensor.shape[0],
            tensor.shape[1],
        )
        .with_context(|| format!("upload DSv4 gate {name}"))
    }

    /// Load a DSv4 block-scaled FP8 (`F8_E4M3`) or packed FP4 (`I8`) `<name>` plus
    /// its sibling `<prefix>.scale` (`F8_E8M0`) into a [`DeviceMatrix`]. The whole
    /// tensor is loaded (EP rank selection happens by expert index in the caller;
    /// TP weight sharding is Piece 4). Returns the block-scaled `DeviceMatrix` the
    /// shared `Dsv4Fp8DeepGemmWeightCache` consumes.
    pub(crate) fn load_dsv4_block_scaled(
        &self,
        ctx: &DeviceContext,
        name: &str,
    ) -> Result<DeviceMatrix> {
        let tensor = self.load_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D quantized tensor, got shape {:?}",
            tensor.shape
        );
        let scale_name = name
            .strip_suffix(".weight")
            .map(|prefix| format!("{prefix}.scale"))
            .ok_or_else(|| anyhow!("{name}: quantized DSv4 tensor must end with .weight"))?;
        let scale = self.load_raw_tensor(&scale_name)?;
        ensure!(
            scale.dtype == Dtype::F8_E8M0,
            "{scale_name}: expected F8_E8M0 block scale, got {:?}",
            scale.dtype
        );
        ensure!(
            scale.shape.len() == 2,
            "{scale_name}: expected 2D scale, got shape {:?}",
            scale.shape
        );
        let (scale_rows, scale_cols) = (scale.shape[0], scale.shape[1]);

        match tensor.dtype {
            Dtype::F8_E4M3 => {
                let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
                DeviceMatrix::from_dsv4_fp8_block_scaled(
                    ctx,
                    &tensor.bytes,
                    &scale.bytes,
                    rows,
                    cols,
                    scale_rows,
                    scale_cols,
                )
                .with_context(|| format!("upload DSv4 FP8 matrix {name}"))
            }
            // FP4 E2M1 is row-major, 2 nibbles per byte → logical_cols = 2 * bytes.
            Dtype::I8 => {
                let (rows, packed_cols) = (tensor.shape[0], tensor.shape[1]);
                let logical_cols = packed_cols * 2;
                DeviceMatrix::from_dsv4_fp4_block_scaled(
                    ctx,
                    &tensor.bytes,
                    &scale.bytes,
                    rows,
                    logical_cols,
                    scale_rows,
                    scale_cols,
                )
                .with_context(|| format!("upload DSv4 FP4 matrix {name}"))
            }
            other => bail!("{name}: unsupported DSv4 block-scaled dtype {other:?}"),
        }
    }

    fn dsv4_scale_shard_for_value_shard(
        name: &str,
        value: &ShardingSpec,
        scale_total: usize,
        block: usize,
    ) -> Result<ShardingSpec> {
        ensure!(block > 0, "{name}: FP8 scale block must be non-zero");
        ensure!(
            value.total.div_ceil(block) == scale_total,
            "{name}: scale total {scale_total} does not match ceil({}/{block})",
            value.total
        );
        ensure!(
            value.offset.is_multiple_of(block) && value.size.is_multiple_of(block),
            "{name}: TP shard {:?} is not aligned to FP8 block size {block}",
            value.range()
        );
        Ok(ShardingSpec {
            offset: value.offset / block,
            size: value.size / block,
            total: scale_total,
        })
    }

    /// Load a DSv4 block-scaled FP8 matrix and apply a TP shard before upload.
    /// The FP8 payload and E8M0 block scales must be sliced together; otherwise
    /// the shard reads valid FP8 bytes with the wrong scale blocks.
    pub(crate) fn load_dsv4_block_scaled_sharded(
        &self,
        ctx: &DeviceContext,
        name: &str,
        shard: Shard,
        tp: &TpConfig,
    ) -> Result<DeviceMatrix> {
        if tp.is_single() || shard == Shard::Replicated {
            return self.load_dsv4_block_scaled(ctx, name);
        }

        let tensor = self.load_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D quantized tensor, got shape {:?}",
            tensor.shape
        );
        let scale_name = name
            .strip_suffix(".weight")
            .map(|prefix| format!("{prefix}.scale"))
            .ok_or_else(|| anyhow!("{name}: quantized DSv4 tensor must end with .weight"))?;
        let scale = self.load_raw_tensor(&scale_name)?;
        ensure!(
            scale.dtype == Dtype::F8_E8M0,
            "{scale_name}: expected F8_E8M0 block scale, got {:?}",
            scale.dtype
        );
        ensure!(
            scale.shape.len() == 2,
            "{scale_name}: expected 2D scale, got shape {:?}",
            scale.shape
        );

        match tensor.dtype {
            Dtype::F8_E4M3 => {
                let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
                let (scale_rows, scale_cols) = (scale.shape[0], scale.shape[1]);
                let (weight, scales) = match shard {
                    Shard::Column { dim: 0 } => {
                        let spec = infer_topo::column_shard(rows, tp);
                        let weight = crate::shard_slice::shard_column_parallel(
                            &tensor.bytes,
                            rows,
                            cols,
                            1,
                            &spec,
                        )?;
                        let scale_spec =
                            Self::dsv4_scale_shard_for_value_shard(name, &spec, scale_rows, 128)?;
                        let scales = crate::shard_slice::shard_column_parallel(
                            &scale.bytes,
                            scale_rows,
                            scale_cols,
                            1,
                            &scale_spec,
                        )?;
                        (weight, scales)
                    }
                    Shard::Row { dim: 1 } => {
                        let spec = infer_topo::row_shard(cols, tp);
                        let weight = crate::shard_slice::shard_row_parallel(
                            &tensor.bytes,
                            rows,
                            cols,
                            1,
                            &spec,
                        )?;
                        let scale_spec =
                            Self::dsv4_scale_shard_for_value_shard(name, &spec, scale_cols, 128)?;
                        let scales = crate::shard_slice::shard_row_parallel(
                            &scale.bytes,
                            scale_rows,
                            scale_cols,
                            1,
                            &scale_spec,
                        )?;
                        (weight, scales)
                    }
                    Shard::Replicated => unreachable!("replicated handled above"),
                    other => bail!("{name}: unsupported DSv4 FP8 TP shard policy {other:?}"),
                };
                DeviceMatrix::from_dsv4_fp8_block_scaled(
                    ctx,
                    &weight.bytes,
                    &scales.bytes,
                    weight.rows,
                    weight.cols,
                    scales.rows,
                    scales.cols,
                )
                .with_context(|| format!("upload sharded DSv4 FP8 matrix {name}"))
            }
            Dtype::I8 => bail!("{name}: non-replicated DSv4 FP4 TP sharding is not implemented"),
            other => bail!("{name}: unsupported DSv4 block-scaled dtype {other:?}"),
        }
    }

    /// Build the per-rank DSv4 MoE layer (FP8 DeepGEMM expert caches + router).
    /// Bias-routed layers load `gate.bias`; hash-routed layers load the host
    /// `gate.tid2eid` table instead (and skip the bias).
    pub(crate) fn load_dsv4_moe_layer(
        &self,
        ctx: &DeviceContext,
        names: &deepseek_spec::DeepSeekV4MoeTensorNames,
        split: &crate::moe_config::ExpertSplit,
        routing_kind: deepseek_spec::DeepSeekV4MoeRoutingKind,
    ) -> Result<crate::dsv4::Dsv4MoeLayer> {
        use cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache;
        use deepseek_spec::DeepSeekV4MoeRoutingKind;

        let mut w13 = Vec::with_capacity(split.experts_per_rank);
        let mut w2 = Vec::with_capacity(split.experts_per_rank);
        for e in split.local_expert_start..split.local_expert_end() {
            let expert = names.expert(e);
            // w1 (gate) over w3 (up), row-stacked into one fused FP8 cache so the
            // masked grouped GEMM produces [gate | up] in a single launch.
            let w1 = self.load_dsv4_block_scaled(ctx, &expert.w1)?;
            let w3 = self.load_dsv4_block_scaled(ctx, &expert.w3)?;
            w13.push(Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight_pair_rows(
                ctx, &w1, &w3,
            )?);
            let down = self.load_dsv4_block_scaled(ctx, &expert.w2)?;
            w2.push(Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight(ctx, &down)?);
        }
        let first_w13 = w13
            .first()
            .ok_or_else(|| anyhow!("DSv4 MoE layer has no local experts"))?;
        let first_w2 = w2
            .first()
            .ok_or_else(|| anyhow!("DSv4 MoE layer has no local down experts"))?;
        let hidden_dim = first_w13.cols;
        let intermediate = first_w2.cols;
        ensure!(
            first_w13.rows == 2 * intermediate,
            "DSv4 grouped w13 rows {} != 2*intermediate {}",
            first_w13.rows,
            2 * intermediate
        );
        ensure!(
            first_w2.rows == hidden_dim,
            "DSv4 grouped w2 rows {} != hidden_dim {hidden_dim}",
            first_w2.rows
        );
        let w13_grouped =
            crate::moe::build_grouped_cache(ctx, w13.as_slice(), 2 * intermediate, hidden_dim)?;
        let w2_grouped =
            crate::moe::build_grouped_cache(ctx, w2.as_slice(), hidden_dim, intermediate)?;
        let num_groups = w13_grouped.groups;
        ensure!(
            num_groups == split.experts_per_rank && w2_grouped.groups == num_groups,
            "DSv4 grouped expert count mismatch: w13={} w2={} expected {}",
            w13_grouped.groups,
            w2_grouped.groups,
            split.experts_per_rank
        );

        let gate = self.load_dsv4_bf16_matrix(ctx, &names.gate_weight)?;
        let (gate_bias, hash_tid2eid, hash_tid2eid_device) = match routing_kind {
            DeepSeekV4MoeRoutingKind::LearnedBias => {
                let bias_name = names
                    .gate_bias
                    .as_ref()
                    .ok_or_else(|| anyhow!("DSv4 bias-routed MoE layer missing gate.bias"))?;
                (Some(self.load_dsv4_vec(ctx, bias_name)?), None, None)
            }
            DeepSeekV4MoeRoutingKind::Hash => {
                let tid_name = names
                    .gate_tid2eid
                    .as_ref()
                    .ok_or_else(|| anyhow!("DSv4 hash-routed MoE layer missing gate.tid2eid"))?;
                let table = self.load_dsv4_i64_host(tid_name)?;
                let device = ctx
                    .stream
                    .clone_htod(&table)
                    .map_err(|e| anyhow!("DSv4 tid2eid H2D failed for {tid_name}: {e}"))?;
                (None, Some(table), Some(device))
            }
        };

        let shared = names
            .shared_experts
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 expects an always-on shared expert"))?;
        let shared_w1 = self.load_dsv4_block_scaled(ctx, &shared.w1)?;
        let shared_w3 = self.load_dsv4_block_scaled(ctx, &shared.w3)?;
        let shared_w13 =
            Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight_pair_rows(ctx, &shared_w1, &shared_w3)?;
        let shared_down = self.load_dsv4_block_scaled(ctx, &shared.w2)?;
        let shared_w2 = Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight(ctx, &shared_down)?;

        Ok(crate::dsv4::Dsv4MoeLayer {
            w13_grouped,
            w2_grouped,
            num_groups,
            hidden_dim,
            intermediate,
            gate,
            gate_bias,
            hash_tid2eid,
            hash_tid2eid_device,
            routing_kind,
            shared_w13,
            shared_w2,
        })
    }

    /// Load a DSv4 1D `i64` table (hash routing `gate.tid2eid`) into a host
    /// `Vec<i64>`. The loader also uploads a device copy for the on-device
    /// router; the host copy remains the A/B oracle.
    pub(crate) fn load_dsv4_i64_host(&self, name: &str) -> Result<Vec<i64>> {
        use safetensors::tensor::Dtype;
        let tensor = self.load_raw_tensor(name)?;
        ensure!(
            tensor.dtype == Dtype::I64,
            "{name}: DSv4 tid2eid expected I64, got {:?}",
            tensor.dtype
        );
        ensure!(
            tensor.bytes.len() % 8 == 0,
            "{name}: I64 byte length {} is not a multiple of 8",
            tensor.bytes.len()
        );
        Ok(tensor
            .bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().expect("8-byte chunk")))
            .collect())
    }

    /// Load one DSv4 hyper-connection block (`base` bf16 vec, `mix_fn` matrix —
    /// bf16 or FP8/FP4 block-scaled, `scale` bf16 vec).
    pub(crate) fn load_dsv4_hyper_connection(
        &self,
        ctx: &DeviceContext,
        names: &deepseek_spec::DeepSeekV4HyperConnectionTensorNames,
    ) -> Result<crate::dsv4::Dsv4HyperConnection> {
        Ok(crate::dsv4::Dsv4HyperConnection {
            base: self.load_dsv4_vec(ctx, &names.base)?,
            mix_fn: self.load_dsv4_global_matrix(ctx, &names.mix_fn)?,
            scale: self.load_dsv4_vec(ctx, &names.scale)?,
        })
    }

    /// Load a DSv4 2D matrix dispatching on its on-disk dtype: BF16 →
    /// `from_safetensors`, F8_E4M3/I8 → block-scaled. Used for embed/head, which
    /// DSv4 checkpoints may ship in either precision.
    pub(crate) fn load_dsv4_global_matrix(
        &self,
        ctx: &DeviceContext,
        name: &str,
    ) -> Result<DeviceMatrix> {
        let dtype = self.load_raw_tensor(name)?.dtype;
        match dtype {
            Dtype::BF16 | Dtype::F32 => self.load_dsv4_bf16_matrix(ctx, name),
            Dtype::F8_E4M3 | Dtype::I8 => self.load_dsv4_block_scaled(ctx, name),
            other => bail!("{name}: unsupported DSv4 global matrix dtype {other:?}"),
        }
    }

    /// Build one DSv4 MLA attention block. The Q/KV/O LoRA matrices are FP8/FP4
    /// block-scaled; CSA/HCA layers also carry a `compressor` (and CSA an
    /// `indexer`) — their matrices may be FP8/FP4 or bf16, so they route through
    /// the dtype-dispatching [`Self::load_dsv4_global_matrix`].
    pub(crate) fn load_dsv4_attention(
        &self,
        ctx: &DeviceContext,
        config: &deepseek_spec::DeepSeekV4Config,
        names: &deepseek_spec::DeepSeekV4AttentionTensorNames,
        tp: &TpConfig,
    ) -> Result<crate::dsv4::Dsv4Attention> {
        Ok(crate::dsv4::Dsv4Attention {
            wq_a: self.load_dsv4_block_scaled(ctx, &names.wq_a)?,
            q_norm: self.load_dsv4_vec(ctx, &names.q_norm)?,
            wq_b: self.load_dsv4_block_scaled_sharded(
                ctx,
                &names.wq_b,
                names
                    .shard_for(config, &names.wq_b, tp.world_size)
                    .unwrap_or(Shard::Replicated),
                tp,
            )?,
            wkv: self.load_dsv4_block_scaled(ctx, &names.wkv)?,
            kv_norm: self.load_dsv4_vec(ctx, &names.kv_norm)?,
            wo_a: self.load_dsv4_block_scaled_sharded(
                ctx,
                &names.wo_a,
                names
                    .shard_for(config, &names.wo_a, tp.world_size)
                    .unwrap_or(Shard::Replicated),
                tp,
            )?,
            wo_b: self.load_dsv4_block_scaled_sharded(
                ctx,
                &names.wo_b,
                names
                    .shard_for(config, &names.wo_b, tp.world_size)
                    .unwrap_or(Shard::Replicated),
                tp,
            )?,
            attn_sink: self.load_dsv4_vec(ctx, &names.attn_sink)?,
            compressor: names
                .compressor
                .as_ref()
                .map(|c| self.load_dsv4_compressor(ctx, c))
                .transpose()?,
            indexer: names
                .indexer
                .as_ref()
                .map(|i| self.load_dsv4_indexer(ctx, i))
                .transpose()?,
        })
    }

    /// Load one compressor sub-block (`wkv`/`wgate`/`ape` matrices + `norm` vec).
    pub(crate) fn load_dsv4_compressor(
        &self,
        ctx: &DeviceContext,
        names: &deepseek_spec::DeepSeekV4CompressorTensorNames,
    ) -> Result<crate::dsv4::Dsv4Compressor> {
        Ok(crate::dsv4::Dsv4Compressor {
            wkv: self.load_dsv4_global_matrix(ctx, &names.wkv)?,
            wgate: self.load_dsv4_global_matrix(ctx, &names.wgate)?,
            ape: self.load_dsv4_global_matrix(ctx, &names.ape)?,
            norm: self.load_dsv4_vec(ctx, &names.norm)?,
        })
    }

    /// Load one CSA indexer sub-block (`wq_b`/`weights_proj` + a key compressor).
    pub(crate) fn load_dsv4_indexer(
        &self,
        ctx: &DeviceContext,
        names: &deepseek_spec::DeepSeekV4IndexerTensorNames,
    ) -> Result<crate::dsv4::Dsv4Indexer> {
        Ok(crate::dsv4::Dsv4Indexer {
            wq_b: self.load_dsv4_global_matrix(ctx, &names.wq_b)?,
            weights_proj: self.load_dsv4_global_matrix(ctx, &names.weights_proj)?,
            compressor: self.load_dsv4_compressor(ctx, &names.compressor)?,
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
    dtype: Dtype,
}

/// This EP rank's loaded MoE weights for one sparse layer: per-expert
/// gate/up/down stacks + their weight-pointer tables, the router gate, and the
/// shared expert. Built by [`SafetensorLoader::load_moe_layer_experts`],
/// consumed by [`crate::moe::moe_forward`].
pub(crate) struct MoeLayerWeights {
    pub(crate) gate: Vec<DeviceMatrix>,
    pub(crate) up: Vec<DeviceMatrix>,
    pub(crate) down: Vec<DeviceMatrix>,
    pub(crate) gate_ptrs: CudaSlice<u64>,
    pub(crate) up_ptrs: CudaSlice<u64>,
    pub(crate) down_ptrs: CudaSlice<u64>,
    pub(crate) router_gate: DeviceMatrix,
    pub(crate) shared_gate: DeviceMatrix,
    pub(crate) shared_up: DeviceMatrix,
    pub(crate) shared_down: DeviceMatrix,
    pub(crate) shared_gate_router: DeviceMatrix,
}
