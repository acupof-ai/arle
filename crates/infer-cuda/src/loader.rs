//! Cold path: safetensors loading, paging metadata, and config validation.
//!
//! Holds the BF16 safetensors loader, `CudaModel::from_safetensors` weight
//! upload, the per-step paging metadata (`PageMeta`), and the BF16 config gate.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use cuda_kernels::KVFormat;
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, PagedKVPool};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
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
/// single runtime. Shared by the dense Qwen3 and Qwen3.5/3.6 hybrid loaders.
pub(crate) fn build_tp_runtime() -> Result<crate::tp::TpRuntime> {
    #[cfg(feature = "nccl")]
    {
        let cfg = crate::tp::resolve_tp_config_from_env().map_err(|e| anyhow!("{e}"))?;
        if !cfg.is_single() {
            // Bind this rank's CUDA device BEFORE ncclCommInitRank — NCCL pins
            // the communicator to the current device at init, so without this
            // every rank would init on device 0 (mirrors the proven
            // `dsv4::build_dsv4_tp_runtime` flow).
            let ordinal = cuda_kernels::tensor::parse_device_ordinal_from_env()?;
            cudarc::runtime::result::device::set(ordinal as i32)
                .map_err(|e| anyhow!("cudaSetDevice({ordinal}) before NCCL init failed: {e}"))?;
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

/// Mint a fresh NCCL `unique_id` (rank-0 launcher / multiproc coordinator) and
/// return it as 256 hex chars, ready to publish via `INFER_NCCL_UNIQUE_ID` so
/// every spawned worker rank inherits the SAME rendezvous handle (which the DSv4
/// executor decodes via [`nccl_unique_id_from_env`] during construction).
/// `ncclGetUniqueId` is a host call — no CUDA context / GPU is required to mint.
#[cfg(feature = "nccl")]
pub fn mint_nccl_unique_id_hex() -> Result<String> {
    use cuda_kernels::ffi::nccl;
    let mut id = nccl::ncclUniqueId {
        internal: [0i8; 128],
    };
    // SAFETY: `id` is a valid, fully-initialized 128-byte ncclUniqueId; NCCL
    // writes the rendezvous handle into it. Single-threaded, no aliasing.
    let res = unsafe { nccl::ncclGetUniqueId(&mut id) };
    nccl::check(res).context("ncclGetUniqueId failed")?;
    let mut hex = String::with_capacity(256);
    for &b in &id.internal {
        use std::fmt::Write;
        write!(hex, "{:02x}", b as u8).expect("write to String is infallible");
    }
    debug_assert_eq!(hex.len(), 256);
    Ok(hex)
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
    /// Host copy of the request's prefix length (tokens already in the pool
    /// before this forward). Quant formats use it to size the prefix refill.
    pub(crate) start_pos: usize,
    /// Global pool token rows for the NEW tokens [start_pos, start_pos+seq_len).
    /// Quant formats only (refill/quantize row lists); None for BF16.
    pub(crate) new_token_rows: Option<CudaSlice<i32>>,
    /// Global pool token rows for the prefix [0, start_pos). Quant + start_pos>0 only.
    pub(crate) prefix_token_rows: Option<CudaSlice<i32>>,
    /// Packed fused-decode metadata `[page_indptr(b+1) | last_page_len(b)]`
    /// from `build_quantized_decode_indptr`. Quant formats only.
    pub(crate) quant_decode_meta: Option<CudaSlice<i32>>,
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
        // Quant formats (INT8/FP8) need explicit token-row lists: the prefix
        // refill + new-row quantize kernels address the pool by global token
        // row, and the fused decode kernel consumes the packed
        // `[page_indptr | last_page_len]` metadata. BF16 carries None — zero
        // overhead on the default path.
        let quant = matches!(pool.format, KVFormat::INT8 | KVFormat::FP8E4M3);
        let (new_token_rows, prefix_token_rows, quant_decode_meta) = if quant {
            let new_rows = pool
                .token_rows_for_range(slot, start_pos, seq_len)
                .into_iter()
                .map(|row| row as i32)
                .collect::<Vec<_>>();
            let prefix_rows = if start_pos > 0 {
                let rows = pool
                    .token_rows_for_range(slot, 0, start_pos)
                    .into_iter()
                    .map(|row| row as i32)
                    .collect::<Vec<_>>();
                Some(upload_i32(ctx, &rows)?)
            } else {
                None
            };
            (
                Some(upload_i32(ctx, &new_rows)?),
                prefix_rows,
                Some(upload_i32(
                    ctx,
                    &pool.build_quantized_decode_indptr(&[slot]),
                )?),
            )
        } else {
            (None, None, None)
        };
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
            start_pos,
            new_token_rows,
            prefix_token_rows,
            quant_decode_meta,
        })
    }
}

pub(crate) struct SafetensorLoader {
    base: PathBuf,
    shards: Vec<PathBuf>,
    weight_map: HashMap<String, usize>,
    /// Read-once cache of shard bytes: without it, loading N tensors re-reads the
    /// whole shard N times (O(N × file_size) I/O). `Rc` so [`SharedTensor`] can
    /// expose a tensor's byte range zero-copy (no `to_vec` of the multi-GiB
    /// stacked expert tensors) without holding a `RefCell` guard across further
    /// loads (which would panic on the next shard-cache fill).
    /// `Rc<Vec<u8>>` over clippy's `Rc<[u8]>`: the conversion would copy the
    /// multi-GiB shard once more — the exact copy this cache exists to avoid.
    #[allow(clippy::rc_buffer)]
    shard_cache: std::cell::RefCell<HashMap<usize, std::rc::Rc<Vec<u8>>>>,
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
    pub(crate) fn load_matrix_sharded(
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
    ///
    /// `head_dim` is the PER-HEAD ROW COUNT, not necessarily the attention
    /// head_dim: the gated Qwen3.5/3.6 q_proj interleaves `[query; gate]` per
    /// head, so its per-head row block is `2 * head_dim`.
    pub(crate) fn load_qkv_head_sharded(
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

    /// Load this EP rank's MoE weights for one layer (routed gate/up/down +
    /// router gate + shared expert) and build the per-expert weight-pointer
    /// tables. Only the experts in
    /// `split.local_expert_start..local_expert_end()` are loaded (single-GPU
    /// loads all).
    ///
    /// Routed experts ship in one of two layouts (see
    /// [`qwen35_spec::Qwen35MoeTensorNames`]), auto-detected per layer:
    ///   • per-expert `experts.{i}.{gate,up,down}_proj.weight` — loaded as
    ///     separate matrices (byte-identical to the pre-stacked-support path);
    ///   • stacked+fused `experts.gate_up_proj` `[E, 2*moe_inter, hidden]`
    ///     (gate rows `[0, moe_inter)`, up rows `[moe_inter, 2*moe_inter)`)
    ///     + `experts.down_proj` `[E, hidden, moe_inter]` (production
    ///     Qwen3.6-35B-A3B) — each local expert's contiguous 2D block is
    ///     sliced out by byte range and uploaded into the same
    ///     `Vec<DeviceMatrix>` the per-expert path builds.
    ///
    /// Under TP the router gate (and the shared-expert sigmoid gate) stay
    /// replicated — routing must be computed identically on every rank — while
    /// the shared expert is column/row-sharded like a dense MLP so its partial
    /// lands in the same post-MoE all-reduce as the routed partial.
    pub(crate) fn load_moe_layer_experts(
        &self,
        ctx: &DeviceContext,
        names: &qwen35_spec::Qwen35MoeTensorNames,
        split: &crate::moe_config::ExpertSplit,
        tp: &TpConfig,
        moe_intermediate_size: usize,
        hidden_size: usize,
    ) -> Result<MoeLayerWeights> {
        const BF16_ELEM_SIZE: usize = 2;
        let mut gate = Vec::with_capacity(split.experts_per_rank);
        let mut up = Vec::with_capacity(split.experts_per_rank);
        let mut down = Vec::with_capacity(split.experts_per_rank);
        let per_expert_probe = names.expert_gate_proj(split.local_expert_start);
        // The stacked tensors are HF `nn.Parameter`s — no `.weight` suffix on
        // the real Qwen3.6-35B-A3B checkpoint — but accept a `.weight`-suffixed
        // export too.
        let resolve_stacked = |base: &str| -> Option<String> {
            [base.to_string(), format!("{base}.weight")]
                .into_iter()
                .find(|name| self.has_tensor(name))
        };
        if self.has_tensor(&per_expert_probe) {
            for e in split.local_expert_start..split.local_expert_end() {
                gate.push(self.load_matrix(ctx, &names.expert_gate_proj(e))?);
                up.push(self.load_matrix(ctx, &names.expert_up_proj(e))?);
                down.push(self.load_matrix(ctx, &names.expert_down_proj(e))?);
            }
        } else if let Some(gate_up_name) = resolve_stacked(&names.experts_stacked_gate_up_proj) {
            let down_name = resolve_stacked(&names.experts_stacked_down_proj).ok_or_else(|| {
                anyhow!(
                    "MoE layer `{}`: found stacked `{gate_up_name}` but no `{}` \
                     (expected [{}, {hidden_size}, {moe_intermediate_size}])",
                    names.mlp_prefix,
                    names.experts_stacked_down_proj,
                    split.num_experts
                )
            })?;
            ensure!(
                moe_intermediate_size > 0 && hidden_size > 0,
                "MoE layer `{}`: stacked expert load needs non-zero config dims \
                 (moe_intermediate_size={moe_intermediate_size}, hidden_size={hidden_size})",
                names.mlp_prefix
            );
            let stacked_rows = 2 * moe_intermediate_size;
            // Borrow each stacked tensor ONCE from the read-once shard cache and
            // slice every local expert directly out of the cached bytes — the
            // previous owned load (`view.data().to_vec()`) added ~1 GiB + 512 MiB
            // of host memcpy per MoE layer on top of the cache. Uploads unchanged.
            let gate_up_t = self.borrow_bf16_tensor(&gate_up_name)?;
            ensure!(
                gate_up_t.shape == [split.num_experts, stacked_rows, hidden_size],
                "{gate_up_name}: expected stacked fused gate‖up tensor \
                 [{}, {stacked_rows}, {hidden_size}] \
                 ([num_experts, 2*moe_intermediate_size, hidden_size]), got {:?}",
                split.num_experts,
                gate_up_t.shape
            );
            let down_t = self.borrow_bf16_tensor(&down_name)?;
            ensure!(
                down_t.shape == [split.num_experts, hidden_size, moe_intermediate_size],
                "{down_name}: expected stacked down tensor \
                 [{}, {hidden_size}, {moe_intermediate_size}] \
                 ([num_experts, hidden_size, moe_intermediate_size]), got {:?}",
                split.num_experts,
                down_t.shape
            );
            for e in split.local_expert_start..split.local_expert_end() {
                // gate_up_proj [E, 2*mi, hidden]: gate = rows [0, mi),
                // up = rows [mi, 2*mi) of expert e's contiguous block.
                let gate_bytes = crate::shard_slice::slice_stacked_expert(
                    gate_up_t.bytes(),
                    split.num_experts,
                    stacked_rows,
                    hidden_size,
                    BF16_ELEM_SIZE,
                    e,
                    0,
                    moe_intermediate_size,
                )?;
                gate.push(
                    DeviceMatrix::from_safetensors(
                        ctx,
                        gate_bytes,
                        moe_intermediate_size,
                        hidden_size,
                    )
                    .with_context(|| format!("upload expert {e} gate slice of {gate_up_name}"))?,
                );
                let up_bytes = crate::shard_slice::slice_stacked_expert(
                    gate_up_t.bytes(),
                    split.num_experts,
                    stacked_rows,
                    hidden_size,
                    BF16_ELEM_SIZE,
                    e,
                    moe_intermediate_size,
                    moe_intermediate_size,
                )?;
                up.push(
                    DeviceMatrix::from_safetensors(
                        ctx,
                        up_bytes,
                        moe_intermediate_size,
                        hidden_size,
                    )
                    .with_context(|| format!("upload expert {e} up slice of {gate_up_name}"))?,
                );
                // down_proj [E, hidden, mi]: the whole expert block.
                let down_bytes = crate::shard_slice::slice_stacked_expert(
                    down_t.bytes(),
                    split.num_experts,
                    hidden_size,
                    moe_intermediate_size,
                    BF16_ELEM_SIZE,
                    e,
                    0,
                    hidden_size,
                )?;
                down.push(
                    DeviceMatrix::from_safetensors(
                        ctx,
                        down_bytes,
                        hidden_size,
                        moe_intermediate_size,
                    )
                    .with_context(|| format!("upload expert {e} down slice of {down_name}"))?,
                );
            }
        } else {
            let legacy_switch_mlp =
                resolve_stacked(&format!("{}.switch_mlp.gate_proj", names.mlp_prefix)).is_some();
            bail!(
                "MoE layer `{}`: no recognized routed-expert layout — need per-expert \
                 `{per_expert_probe}` (+ up/down siblings) or stacked+fused \
                 `{}` [{}, {}, {hidden_size}] + `{}` [{}, {hidden_size}, {moe_intermediate_size}]{}",
                names.mlp_prefix,
                names.experts_stacked_gate_up_proj,
                split.num_experts,
                2 * moe_intermediate_size,
                names.experts_stacked_down_proj,
                split.num_experts,
                if legacy_switch_mlp {
                    " (found unsupported legacy `switch_mlp.*`)"
                } else {
                    ""
                }
            );
        }
        let router_gate = self.load_matrix(ctx, &names.router_gate)?;
        let (shared_gate, shared_up, shared_down) = if tp.is_single() {
            (
                self.load_matrix(ctx, &names.shared_expert_gate_proj)?,
                self.load_matrix(ctx, &names.shared_expert_up_proj)?,
                self.load_matrix(ctx, &names.shared_expert_down_proj)?,
            )
        } else {
            (
                self.load_matrix_sharded(
                    ctx,
                    &names.shared_expert_gate_proj,
                    infer_topo::ParallelLinearKind::Column,
                    tp,
                )?,
                self.load_matrix_sharded(
                    ctx,
                    &names.shared_expert_up_proj,
                    infer_topo::ParallelLinearKind::Column,
                    tp,
                )?,
                self.load_matrix_sharded(
                    ctx,
                    &names.shared_expert_down_proj,
                    infer_topo::ParallelLinearKind::Row,
                    tp,
                )?,
            )
        };
        let shared_gate_router = self.load_matrix(ctx, &names.shared_expert_gate)?;

        // DeepGEMM grouped-B caches (opt-in): concat the per-expert matrices
        // into one contiguous [G, n, k] buffer per projection, repoint the
        // pointer tables into it, and DROP the per-expert copies — keeping
        // both would double the routed-expert VRAM (~2x model weights on
        // Qwen3.6-35B). The hand kernels keep working through the rebuilt
        // tables (same [n, k] row-major slabs, new addresses).
        // Default-ON safety: a build-time stub bridge (no
        // ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1) must degrade to the hand-kernel
        // path instead of erroring at the first MoE forward — probe once here
        // and skip the grouped caches so `use_deepgemm` self-disables.
        let deepgemm_ready = crate::moe::qwen35_deepgemm_enabled()
            && match cuda_kernels::moe::dsv4_deepgemm_native_preflight() {
                Ok(_) => true,
                Err(err) => {
                    log::warn!(
                        "Qwen3.5 DeepGEMM MoE disabled: native bridge unavailable ({err}); \
                         falling back to the hand grouped kernels"
                    );
                    false
                }
            };
        let (gate_grouped, up_grouped, down_grouped) = if deepgemm_ready {
            let gate_g = MoeExpertGroup::concat(ctx, &gate)?;
            let up_g = MoeExpertGroup::concat(ctx, &up)?;
            let down_g = MoeExpertGroup::concat(ctx, &down)?;
            // Event tracking is disabled: dropping the per-expert sources
            // frees device memory at Rust last-use, so the async D2D concats
            // MUST have completed first.
            ctx.sync()?;
            gate.clear();
            up.clear();
            down.clear();
            (Some(gate_g), Some(up_g), Some(down_g))
        } else {
            (None, None, None)
        };

        // SGLang fused_moe stacked cache (opt-in, `ARLE_QWEN35_MOE_FUSED_SGLANG`),
        // mutually exclusive with DeepGEMM (both restack the same routed-expert
        // weights). Build-and-replace exactly like the DeepGEMM block above:
        // restack the per-expert Vecs into w1/w2, sync, then DROP the Vecs —
        // keeping both would double routed-expert VRAM (the lazy first-forward
        // build OOM'd the 35B BF16 shard at ~layer 20). The fused lane reads
        // w1/w2 directly, so the per-expert ptr tables below stay empty here.
        let fused_sglang = if !deepgemm_ready && crate::moe::qwen35_moe_fused_sglang_enabled() {
            let built = MoeFusedSglangWeights::build(ctx, &gate, &up, &down, hidden_size)?;
            // Event tracking disabled: the async D2D restacks MUST complete
            // before the drop frees the per-expert sources.
            ctx.sync()?;
            gate.clear();
            up.clear();
            down.clear();
            Some(built)
        } else {
            None
        };

        // Per-expert weight-pointer tables (one device pointer per owned
        // expert) — built from the per-expert matrices on the default path,
        // from the grouped buffer offsets on the DeepGEMM path, or empty when
        // the fused lane has freed the Vecs (it reads w1/w2 directly).
        let (gate_ptrs, up_ptrs, down_ptrs) = match (&gate_grouped, &up_grouped, &down_grouped) {
            (Some(g), Some(u), Some(d)) => {
                (g.ptr_table(ctx)?, u.ptr_table(ctx)?, d.ptr_table(ctx)?)
            }
            _ if fused_sglang.is_some() => (
                cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &[])?,
                cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &[])?,
                cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &[])?,
            ),
            _ => {
                let gate_refs: Vec<&DeviceMatrix> = gate.iter().collect();
                let up_refs: Vec<&DeviceMatrix> = up.iter().collect();
                let down_refs: Vec<&DeviceMatrix> = down.iter().collect();
                (
                    cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &gate_refs)?,
                    cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &up_refs)?,
                    cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &down_refs)?,
                )
            }
        };

        Ok(MoeLayerWeights {
            gate,
            up,
            down,
            gate_ptrs,
            up_ptrs,
            down_ptrs,
            gate_grouped,
            up_grouped,
            down_grouped,
            router_gate,
            shared_gate,
            shared_up,
            shared_down,
            shared_gate_router,
            fused_sglang,
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

    /// Whether `name` exists in the checkpoint: weight-map lookup when an
    /// index is present, otherwise each shard header is parsed (from the
    /// read-once byte cache). Used to probe which routed-expert layout a MoE
    /// checkpoint ships before committing to a load path.
    pub(crate) fn has_tensor(&self, name: &str) -> bool {
        if !self.weight_map.is_empty() {
            return self.weight_map.contains_key(name);
        }
        (0..self.shards.len()).any(|idx| self.shard_has_tensor(idx, name).unwrap_or(false))
    }

    /// Read-once shard bytes: fill the cache on first touch, then hand out a
    /// cheap `Rc` clone (no `RefCell` guard escapes, so nested loads that fill
    /// OTHER shards never hit a `BorrowMutError`).
    #[allow(clippy::rc_buffer)] // Rc<[u8]> conversion would re-copy the shard
    fn shard_bytes(&self, idx: usize) -> Result<std::rc::Rc<Vec<u8>>> {
        let path = self
            .shards
            .get(idx)
            .ok_or_else(|| anyhow!("shard index {idx} out of range"))?;
        if let Some(bytes) = self.shard_cache.borrow().get(&idx) {
            return Ok(std::rc::Rc::clone(bytes));
        }
        let bytes =
            std::rc::Rc::new(fs::read(path).with_context(|| format!("read {}", path.display()))?);
        self.shard_cache
            .borrow_mut()
            .insert(idx, std::rc::Rc::clone(&bytes));
        Ok(bytes)
    }

    /// Header-only existence check against one shard — no tensor bytes are
    /// copied (unlike `load_tensor_from_shard`).
    fn shard_has_tensor(&self, idx: usize, name: &str) -> Result<bool> {
        let bytes = self.shard_bytes(idx)?;
        let tensors = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("deserialize {}", self.shards[idx].display()))?;
        Ok(tensors.tensor(name).is_ok())
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
        let tensor = self.borrow_raw_from_shard(idx, name)?;
        Ok(OwnedTensor {
            shape: tensor.shape.clone(),
            bytes: tensor.bytes().to_vec(),
            dtype: tensor.dtype,
        })
    }

    /// Zero-copy shard read: the returned [`SharedTensor`] aliases the tensor's
    /// byte range inside the read-once shard cache (an `Rc` clone, no host
    /// memcpy). The stacked-expert loader slices ~1.5 GiB per MoE layer out of
    /// these bytes; the owned path (`load_raw_from_shard`) copied that whole
    /// range per tensor on top of the cache. (audit MOE-P2-1)
    fn borrow_raw_from_shard(&self, idx: usize, name: &str) -> Result<SharedTensor> {
        let shard = self.shard_bytes(idx)?;
        let path = &self.shards[idx];
        let tensors = SafeTensors::deserialize(&shard)
            .with_context(|| format!("deserialize {}", path.display()))?;
        let view = tensors
            .tensor(name)
            .with_context(|| format!("find tensor {name} in {}", path.display()))?;
        let shape = view.shape().to_vec();
        let dtype = view.dtype();
        let data = view.data();
        // `view.data()` borrows from `shard`'s allocation; record its range so
        // the `Rc`-owning SharedTensor can re-slice it without the borrow.
        let base = shard.as_ptr() as usize;
        let offset = data.as_ptr() as usize - base;
        let len = data.len();
        ensure!(
            offset + len <= shard.len(),
            "{name}: tensor byte range [{offset}, {}) exceeds shard size {}",
            offset + len,
            shard.len()
        );
        Ok(SharedTensor {
            shape,
            dtype,
            shard,
            offset,
            len,
        })
    }

    /// Zero-copy tensor lookup across shards (the borrow-path twin of
    /// [`Self::load_raw_tensor`]).
    fn borrow_raw_tensor(&self, name: &str) -> Result<SharedTensor> {
        if let Some(&idx) = self.weight_map.get(name) {
            return self.borrow_raw_from_shard(idx, name);
        }
        for idx in 0..self.shards.len() {
            if let Ok(tensor) = self.borrow_raw_from_shard(idx, name) {
                return Ok(tensor);
            }
        }
        Err(anyhow!(
            "tensor {name} not found in safetensors under {}",
            self.base.display()
        ))
    }

    /// Zero-copy BF16 tensor borrow (the borrow-path twin of `load_tensor`'s
    /// dtype gate).
    fn borrow_bf16_tensor(&self, name: &str) -> Result<SharedTensor> {
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.dtype == Dtype::BF16,
            "{name}: R6 clean CUDA path accepts BF16 only, got {:?}",
            tensor.dtype
        );
        Ok(tensor)
    }
}

/// A tensor whose bytes alias the loader's read-once shard cache (`Rc` share,
/// zero host copies). [`Self::bytes`] yields the tensor's exact byte range.
pub(crate) struct SharedTensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: Dtype,
    #[allow(clippy::rc_buffer)] // shares the shard cache's Rc — see shard_cache
    shard: std::rc::Rc<Vec<u8>>,
    offset: usize,
    len: usize,
}

impl SharedTensor {
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.shard[self.offset..self.offset + self.len]
    }
}

// DSv4 FP8/FP4 + E8M0 loaders. Loader-only milestone: reachable from
// `Dsv4Model::from_dsv4_fp8_safetensors`, which the executor enum branch wires
// with the Piece 2/3 forward (see `feedback_necessity_not_callers`).
#[allow(dead_code)]
impl SafetensorLoader {
    /// Dtype-agnostic full-tensor read (shape + raw bytes + dtype). The Qwen3.5
    /// hybrid TP loader slices fused-block tensors (`in_proj_qkv`, `conv1d`,
    /// per-v-head vectors) from these bytes before upload.
    pub(crate) fn load_raw_tensor(&self, name: &str) -> Result<OwnedTensor> {
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

    /// Normalize a small 1D/2D tensor to BF16 bytes — these ship as BF16
    /// (norms) or F32 (attn_sink, router gate, Qwen3.5 `dt_bias`) depending on
    /// the checkpoint. Shared by the DSv4 vec loaders and the Qwen3.5 hybrid
    /// TP slicers (which must match `load_vec_any`'s conversion exactly so the
    /// sharded load is byte-identical to slicing the single-GPU upload).
    pub(crate) fn dsv4_bytes_to_bf16<'a>(
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
            gemv_tables: std::sync::OnceLock::new(),
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
        let attn_sink = self.load_dsv4_vec(ctx, &names.attn_sink)?;
        let attn_sink_f32 = {
            let mut dst = ctx
                .stream
                .alloc_zeros::<f32>(attn_sink.len)
                .map_err(|e| anyhow!("DSv4 attn_sink f32 mirror alloc failed: {e}"))?;
            let (src_ptr, _sg) = attn_sink.data.device_ptr(&ctx.stream);
            let (dst_ptr, _dg) = dst.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::arle_bf16_to_f32_cuda(
                    src_ptr as *const ffi::Half,
                    dst_ptr as *mut f32,
                    attn_sink.len as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("DSv4 attn_sink bf16->f32 mirror failed: {e}"))?;
            }
            drop(_dg);
            dst
        };
        let wq_a = self.load_dsv4_block_scaled(ctx, &names.wq_a)?;
        let wkv = self.load_dsv4_block_scaled(ctx, &names.wkv)?;
        let wqkv_a_deepgemm = if crate::attention::dsv4_fused_wqkv_decode_alloc_enabled()? {
            Some(
                cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight_pair_rows(
                    ctx, &wq_a, &wkv,
                )?,
            )
        } else {
            None
        };
        let wq_b = self.load_dsv4_block_scaled_sharded(
            ctx,
            &names.wq_b,
            names
                .shard_for(config, &names.wq_b, tp.world_size)
                .unwrap_or(Shard::Replicated),
            tp,
        )?;
        // DeepGEMM-layout cache for the decode wq_b projection (lever #1: residual
        // scalar GEMV → tensor-core). Built under the same gate as the fused
        // wq_a|wkv cache so the runtime ARLE_DSV4_DECODE_PROJ_DEEPGEMM flag can A/B
        // it without a rebuild.
        let wq_b_deepgemm = if crate::attention::dsv4_fused_wqkv_decode_alloc_enabled()? {
            Some(cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight(ctx, &wq_b)?)
        } else {
            None
        };
        let wo_a = self.load_dsv4_block_scaled_sharded(
            ctx,
            &names.wo_a,
            names
                .shard_for(config, &names.wo_a, tp.world_size)
                .unwrap_or(Shard::Replicated),
            tp,
        )?;
        let wo_b = self.load_dsv4_block_scaled_sharded(
            ctx,
            &names.wo_b,
            names
                .shard_for(config, &names.wo_b, tp.world_size)
                .unwrap_or(Shard::Replicated),
            tp,
        )?;
        // DeepGEMM caches for the decode output projection (lever #1b), same gate.
        let (wo_a_deepgemm, wo_b_deepgemm) =
            if crate::attention::dsv4_fused_wqkv_decode_alloc_enabled()? {
                (
                    Some(
                        cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight(
                            ctx, &wo_a,
                        )?,
                    ),
                    Some(
                        cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight(
                            ctx, &wo_b,
                        )?,
                    ),
                )
            } else {
                (None, None)
            };
        // Replicated decode attention: FULL-width wq_b/wo_a copies alongside
        // the shards (decode reads full, prefill keeps the sharded math).
        let (wq_b_full, wo_a_full) =
            if crate::attention::dsv4_replicated_attn_enabled() && tp.world_size > 1 {
                (
                    Some(self.load_dsv4_block_scaled_sharded(
                        ctx,
                        &names.wq_b,
                        Shard::Replicated,
                        tp,
                    )?),
                    Some(self.load_dsv4_block_scaled_sharded(
                        ctx,
                        &names.wo_a,
                        Shard::Replicated,
                        tp,
                    )?),
                )
            } else {
                (None, None)
            };
        Ok(crate::dsv4::Dsv4Attention {
            wq_a,
            wqkv_a_deepgemm,
            q_norm: self.load_dsv4_vec(ctx, &names.q_norm)?,
            wq_b,
            wq_b_deepgemm,
            wkv,
            kv_norm: self.load_dsv4_vec(ctx, &names.kv_norm)?,
            wo_a,
            wo_b,
            wo_a_deepgemm,
            wo_b_deepgemm,
            wq_b_full,
            wo_a_full,
            attn_sink,
            attn_sink_f32,
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
        let wq_b = self.load_dsv4_global_matrix(ctx, &names.wq_b)?;
        let wq_b_deepgemm = if crate::attention::dsv4_fused_wqkv_decode_alloc_enabled()? {
            Some(cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight(ctx, &wq_b)?)
        } else {
            None
        };
        Ok(crate::dsv4::Dsv4Indexer {
            wq_b,
            weights_proj: self.load_dsv4_global_matrix(ctx, &names.weights_proj)?,
            compressor: self.load_dsv4_compressor(ctx, &names.compressor)?,
            wq_b_deepgemm,
        })
    }
}

#[derive(serde::Deserialize)]
struct SafetensorIndex {
    weight_map: HashMap<String, String>,
}

pub(crate) struct OwnedTensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) bytes: Vec<u8>,
    pub(crate) dtype: Dtype,
}

/// This EP rank's loaded MoE weights for one sparse layer: per-expert
/// gate/up/down stacks + their weight-pointer tables, the router gate, and the
/// shared expert. Built by [`SafetensorLoader::load_moe_layer_experts`],
/// consumed by [`crate::moe::moe_forward`].
pub(crate) struct MoeLayerWeights {
    /// Per-expert weight matrices (hand grouped-GEMM path). EMPTY when the
    /// DeepGEMM grouped caches below are built (`ARLE_QWEN35_DEEPGEMM=1` at
    /// load) — the grouped buffer then owns the only copy of the bytes and
    /// the `*_ptrs` tables point into it, so the hand kernels stay runnable.
    pub(crate) gate: Vec<DeviceMatrix>,
    pub(crate) up: Vec<DeviceMatrix>,
    pub(crate) down: Vec<DeviceMatrix>,
    pub(crate) gate_ptrs: CudaSlice<u64>,
    pub(crate) up_ptrs: CudaSlice<u64>,
    pub(crate) down_ptrs: CudaSlice<u64>,
    /// DeepGEMM grouped-B caches (`[groups, n, k]` contiguous row-major BF16,
    /// this rank's EP experts only). `Some` iff `ARLE_QWEN35_DEEPGEMM=1` at
    /// load; the default load path is byte-identical (fields stay `None`).
    pub(crate) gate_grouped: Option<MoeExpertGroup>,
    pub(crate) up_grouped: Option<MoeExpertGroup>,
    pub(crate) down_grouped: Option<MoeExpertGroup>,
    pub(crate) router_gate: DeviceMatrix,
    pub(crate) shared_gate: DeviceMatrix,
    pub(crate) shared_up: DeviceMatrix,
    pub(crate) shared_down: DeviceMatrix,
    pub(crate) shared_gate_router: DeviceMatrix,
    /// SGLang `fused_moe` stacked weight cache (`w1 [E, 2N, K]` gate-then-up +
    /// `w2 [E, K, N]` down, contiguous BF16, this rank's local experts). Built
    /// at LOAD when `ARLE_QWEN35_MOE_FUSED_SGLANG=1` AND DeepGEMM is off
    /// (mutually exclusive grouped caches): the per-expert Vecs are restacked
    /// into w1/w2 and then FREED — build-and-replace, mirroring the DeepGEMM
    /// grouped path, so resident VRAM does NOT double (the lazy first-forward
    /// build OOM'd on the 35B BF16 shard: w1/w2 ≈ 1.6 GB/layer on top of the
    /// still-resident 1.6 GB/layer Vecs × 40 layers). `None` by default —
    /// the hand path load is byte-identical. Restacked size ≈ source size
    /// (w1 ≈ 256·1024·2048·2 B ≈ 1.0 GB, w2 ≈ 256·2048·512·2 B ≈ 0.5 GB/layer).
    pub(crate) fused_sglang: Option<MoeFusedSglangWeights>,
}

/// SGLang `fused_moe` stacked expert-weight cache for one MoE layer (built
/// lazily by the U3 fused path; see [`MoeLayerWeights::fused_sglang`]).
///
/// `w1` is `[E_l, 2N, K]` row-major BF16: for local expert `e`, rows `[0, N)`
/// are the gate `[N, K]` and rows `[N, 2N)` are the up `[N, K]` (gate-then-up,
/// matching SGLang's stacked `w13_weight`). `w2` is `[E_l, K, N]` row-major
/// BF16: the per-expert down `[K, N]` verbatim. `num_experts` = this rank's
/// local expert count, `gate_up_rows` = `2N`, `hidden` = `K`, `moe_inter` = `N`.
pub(crate) struct MoeFusedSglangWeights {
    pub(crate) w1: CudaSlice<half::bf16>,
    pub(crate) w2: CudaSlice<half::bf16>,
    pub(crate) num_experts: usize,
    pub(crate) gate_up_rows: usize,
    pub(crate) hidden: usize,
    pub(crate) moe_inter: usize,
}

impl MoeFusedSglangWeights {
    /// Build the SGLang stacked `w1 [E, 2N, K]` (gate rows then up rows per
    /// expert) + `w2 [E, K, N]` (down verbatim) cache from the per-expert
    /// `[N, K]` / `[K, N]` matrices by D2D gather-concat. Mirrors
    /// [`MoeExpertGroup::concat`]: the caller MUST `ctx.sync()` before dropping
    /// the source Vecs (event tracking is disabled, so a Rust drop frees device
    /// memory immediately — before the async copies may have run). `num_experts`
    /// is the passed slices' length (this rank's local experts); `hidden_dim`
    /// pins `K`.
    fn build(
        ctx: &DeviceContext,
        gate: &[DeviceMatrix],
        up: &[DeviceMatrix],
        down: &[DeviceMatrix],
        hidden_dim: usize,
    ) -> Result<Self> {
        let num_experts = gate.len();
        ensure!(
            num_experts > 0 && up.len() == num_experts && down.len() == num_experts,
            "fused MoE build needs per-expert Vecs (gate={} up={} down={})",
            gate.len(),
            up.len(),
            down.len()
        );
        // Per-expert dims (gate/up `[N, K]`, down `[K, N]`); uniform across
        // experts (the concat path enforces uniformity too).
        let moe_inter = gate[0].rows; // N
        let k = gate[0].cols; // K (= hidden_dim)
        ensure!(
            k == hidden_dim,
            "fused MoE gate K {k} != hidden_dim {hidden_dim}"
        );
        let gate_up_rows = 2 * moe_inter; // 2N
        let w1_elems = num_experts * gate_up_rows * k;
        let w2_elems = num_experts * k * moe_inter;
        let mut w1 = ctx
            .stream
            .alloc_zeros::<half::bf16>(w1_elems)
            .map_err(|e| anyhow!("fused MoE w1 alloc failed: {e}"))?;
        let mut w2 = ctx
            .stream
            .alloc_zeros::<half::bf16>(w2_elems)
            .map_err(|e| anyhow!("fused MoE w2 alloc failed: {e}"))?;

        let expert_nk = moe_inter * k; // gate / up slab elements
        let down_kn = k * moe_inter; // down slab elements
        for e in 0..num_experts {
            let g = &gate[e];
            let u = &up[e];
            let d = &down[e];
            ensure!(
                g.rows == moe_inter && g.cols == k && u.rows == moe_inter && u.cols == k,
                "fused MoE expert {e} gate/up shape ({}x{} / {}x{}) != {moe_inter}x{k}",
                g.rows,
                g.cols,
                u.rows,
                u.cols
            );
            ensure!(
                d.rows == k && d.cols == moe_inter,
                "fused MoE expert {e} down shape {}x{} != {k}x{moe_inter}",
                d.rows,
                d.cols
            );
            ensure!(
                g.qweight.is_none()
                    && u.qweight.is_none()
                    && d.qweight.is_none()
                    && g.group_size == 0
                    && u.group_size == 0
                    && d.group_size == 0,
                "fused MoE expert {e} is quantized — the BF16 fused cache needs dense BF16"
            );
            // w1[e]: rows [0, N) = gate `[N, K]`, rows [N, 2N) = up `[N, K]`.
            let w1_base = e * gate_up_rows * k;
            ctx.stream
                .memcpy_dtod(&g.data, &mut w1.slice_mut(w1_base..w1_base + expert_nk))
                .map_err(|err| anyhow!("fused MoE w1 gate D2D failed: {err}"))?;
            ctx.stream
                .memcpy_dtod(
                    &u.data,
                    &mut w1.slice_mut(w1_base + expert_nk..w1_base + 2 * expert_nk),
                )
                .map_err(|err| anyhow!("fused MoE w1 up D2D failed: {err}"))?;
            // w2[e]: down `[K, N]` verbatim.
            let w2_base = e * down_kn;
            ctx.stream
                .memcpy_dtod(&d.data, &mut w2.slice_mut(w2_base..w2_base + down_kn))
                .map_err(|err| anyhow!("fused MoE w2 down D2D failed: {err}"))?;
        }

        Ok(Self {
            w1,
            w2,
            num_experts,
            gate_up_rows,
            hidden: k,
            moe_inter,
        })
    }
}

/// One contiguous `[groups, rows, cols]` row-major BF16 expert-weight buffer —
/// DeepGEMM's grouped-B layout (group `g` starts at `g * rows * cols`).
pub(crate) struct MoeExpertGroup {
    pub(crate) data: CudaSlice<half::bf16>,
    pub(crate) groups: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl MoeExpertGroup {
    /// Concatenate per-expert `[rows, cols]` matrices into one contiguous
    /// group-major buffer (D2D). Weights are static after load; the source
    /// matrices may be dropped afterwards **only after a stream sync** (event
    /// tracking is disabled, so a Rust drop frees device memory immediately —
    /// before the async copies may have run).
    fn concat(ctx: &DeviceContext, experts: &[DeviceMatrix]) -> Result<Self> {
        let first = experts
            .first()
            .ok_or_else(|| anyhow!("MoE expert group concat: no local experts"))?;
        let (rows, cols) = (first.rows, first.cols);
        let stride = rows * cols;
        let groups = experts.len();
        let mut data = ctx
            .stream
            .alloc_zeros::<half::bf16>(groups * stride)
            .map_err(|e| anyhow!("MoE expert group alloc failed: {e}"))?;
        for (g, expert) in experts.iter().enumerate() {
            ensure!(
                expert.rows == rows && expert.cols == cols && expert.data.len() == stride,
                "MoE expert group {g} non-uniform: {}x{} (data len {}) != {rows}x{cols}",
                expert.rows,
                expert.cols,
                expert.data.len()
            );
            ensure!(
                expert.qweight.is_none() && expert.group_size == 0,
                "MoE expert group {g} is quantized — DeepGEMM BF16 grouped cache needs dense BF16"
            );
            let mut dst = data.slice_mut(g * stride..(g + 1) * stride);
            ctx.stream
                .memcpy_dtod(&expert.data, &mut dst)
                .map_err(|e| anyhow!("MoE expert group {g} D2D failed: {e}"))?;
        }
        Ok(Self {
            data,
            groups,
            rows,
            cols,
        })
    }

    /// Device table of per-group base pointers (`data + g * rows * cols`) in
    /// the same `*const u64` format as [`cuda_kernels::moe::build_expert_weight_ptr_table`],
    /// so the hand grouped-GEMM kernels run unchanged on the grouped memory.
    fn ptr_table(&self, ctx: &DeviceContext) -> Result<CudaSlice<u64>> {
        let (base, _guard) = self.data.device_ptr(&ctx.stream);
        let stride_bytes = (self.rows * self.cols * std::mem::size_of::<half::bf16>()) as u64;
        let host: Vec<u64> = (0..self.groups as u64)
            .map(|g| base + g * stride_bytes)
            .collect();
        ctx.stream
            .clone_htod(&host)
            .map_err(|e| anyhow!("MoE expert group ptr table H2D failed: {e}"))
    }
}
