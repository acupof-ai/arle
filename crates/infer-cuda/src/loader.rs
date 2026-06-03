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
        // Resolve tensor-parallel placement from the environment (TP-1). On a
        // single GPU this is `world_size == 1`, the no-op communicator, and the
        // full (unsharded) weight load — byte-identical to the pre-TP path.
        let tp = build_tp_runtime()?;
        Self::from_safetensors_with_tp(model_path, tp)
    }

    /// Load with an explicit [`crate::tp::TpRuntime`] (the env path threads its
    /// resolved runtime through here; tests can inject a single-GPU runtime).
    pub(crate) fn from_safetensors_with_tp(
        model_path: &Path,
        tp: crate::tp::TpRuntime,
    ) -> Result<Self> {
        let config = Qwen3Config::from_json_file(model_path.join("config.json"))
            .with_context(|| format!("load Qwen3 config from {}", model_path.display()))?;
        validate_clean_bf16_config(&config)?;

        let tp_cfg = *tp.config();
        // Per-rank head counts (GQA-aware). `head_shard` errors unless both head
        // counts divide the world size, which keeps every rank's attention shape
        // uniform — the precondition the kv8 TileLang kernels and the all-reduce
        // both rely on.
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

        // lm_head / embed_tokens stay REPLICATED across ranks (v1 design: avoids an
        // all-gather of logits; the final gemv runs the full vocab projection on
        // each rank). Only the per-layer Q/K/V/O and MLP projections are sharded.
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
                    // Single GPU: full tensors, identical to the pre-TP path.
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
                    // TP: Q/K/V are column-parallel but must split on whole-head
                    // boundaries (head-aligned), so the o_proj input shard and the
                    // attention head count agree. gate/up are plain column-parallel
                    // (intermediate dim); o_proj/down_proj are row-parallel.
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
                mlp: Mlp {
                    gate_proj,
                    up_proj,
                    down_proj,
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
            tp,
            local_q_heads,
            local_kv_heads,
        })
    }
}

/// Build the tensor-parallel runtime for model load.
///
/// On a multi-rank `nccl` build the launcher hands the NCCL `unique_id` in via
/// the `INFER_NCCL_UNIQUE_ID` env var (128 hex-encoded bytes — the launcher reads
/// it from `ncclGetUniqueId` on rank 0 and broadcasts it over its own transport;
/// see [`crate::tp::TpRuntime::from_env_with_nccl`]). Without that var, or on a
/// non-`nccl`/single-GPU build, this is the no-op single runtime.
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

/// Decode the launcher-supplied NCCL `unique_id` from `INFER_NCCL_UNIQUE_ID`
/// (128 lowercase-hex-encoded bytes, i.e. a 256-char string).
#[cfg(feature = "nccl")]
fn nccl_unique_id_from_env() -> Result<cuda_kernels::ffi::nccl::ncclUniqueId> {
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
    // NOTE: Qwen3 DECOUPLES head_dim from hidden_size/num_heads — e.g. Qwen3-0.6B
    // has hidden_size=1024, num_attention_heads=16, head_dim=128 (q_proj maps
    // 1024 -> 16*128=2048, o_proj maps back to 1024). So `hidden_size ==
    // num_attention_heads * head_dim` is NOT an invariant; the forward already
    // uses q_dim = num_attention_heads * head_dim for the projections. The real
    // constraints are head_dim==128 (TileLang HD128, checked in model.rs) + GQA
    // divisibility below.
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

    /// Load a 2D BF16 weight, slice it to this TP rank, and upload the shard.
    ///
    /// The host-side byte slicing is the feature-agnostic
    /// [`crate::shard_slice`] math:
    /// - [`ParallelLinearKind::Column`] (`q/k/v/gate/up_proj`) slices the output
    ///   dim (rows) via [`infer_topo::column_shard`].
    /// - [`ParallelLinearKind::Row`] (`o_proj/down_proj`) slices the input dim
    ///   (cols) via [`infer_topo::row_shard`].
    ///
    /// On a single-GPU [`TpConfig`] this is the identity slice — same bytes as
    /// [`Self::load_matrix`].
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
    /// Q/K/V are column-parallel (the output dim = `num_heads * head_dim` is
    /// split), but the split MUST land on whole-head boundaries so the per-rank
    /// attention head count, the o_proj input shard, and the kernel's RoPE/RMSNorm
    /// all agree. A plain [`infer_topo::column_shard`] on the raw output dim would
    /// dump the GQA remainder onto the last rank mid-head; instead we shard the
    /// HEAD dimension (`head_shard`, already done by the caller) and convert to a
    /// row [`ShardingSpec`] of `local_heads * head_dim` rows at the matching offset.
    ///
    /// `local_heads` is this rank's head count from [`infer_topo::head_shard`];
    /// since `head_shard` requires the global head count to divide the world size,
    /// every rank owns exactly `local_heads` and the offset is `rank * local_heads`.
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

    /// Load this EP rank's per-expert MoE weight stacks for one layer
    /// (`gate_proj` / `up_proj` / `down_proj`), the router gate, and the shared
    /// expert, then build the device-resident per-expert weight-pointer tables.
    ///
    /// Tensor naming follows the mlx-lm `qwen3_5_moe` HF convention:
    /// `<layer>.mlp.experts.{e}.{gate,up,down}_proj.weight`,
    /// `<layer>.mlp.gate.weight` (router), and `<layer>.mlp.shared_expert.*`.
    ///
    /// Only the experts owned by `split` (per [`crate::moe_config::ExpertSplit`])
    /// are loaded; single-GPU (`ep_size == 1`) loads every expert. The owned
    /// range is `split.local_expert_start .. split.local_expert_end()`, which is
    /// the contiguous slice the EP group from [`infer_topo::build_moe_ep_groups`]
    /// assigns to this rank.
    #[allow(dead_code)]
    fn load_moe_layer_experts(
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

/// This EP rank's loaded MoE weights for one sparse layer (MoE-3).
///
/// Holds the per-rank-owned routed-expert `gate`/`up`/`down` stacks (one
/// [`DeviceMatrix`] per owned expert), their device-resident weight-pointer
/// tables for the grouped GEMM, the router gate, and the single shared expert
/// (gate/up/down + its sigmoid gate). Built by
/// [`SafetensorLoader::load_moe_layer_experts`].
#[allow(dead_code)]
struct MoeLayerWeights {
    gate: Vec<DeviceMatrix>,
    up: Vec<DeviceMatrix>,
    down: Vec<DeviceMatrix>,
    gate_ptrs: CudaSlice<u64>,
    up_ptrs: CudaSlice<u64>,
    down_ptrs: CudaSlice<u64>,
    router_gate: DeviceMatrix,
    shared_gate: DeviceMatrix,
    shared_up: DeviceMatrix,
    shared_down: DeviceMatrix,
    shared_gate_router: DeviceMatrix,
}
