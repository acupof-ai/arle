//! Cold path: safetensors loading, paging metadata, and config validation.
//!
//! Holds the BF16 safetensors loader, `CudaModel::from_safetensors` weight
//! upload, the per-step paging metadata (`PageMeta`), and the BF16 config gate.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use anyhow::{anyhow, bail, ensure, Context, Result};
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, PagedKVPool};
use cuda_kernels::tensor::WeightFormat;
use cuda_kernels::KVFormat;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use deepseek_spec::Shard;
use infer_topo::{ShardingSpec, TpConfig};
use qwen3_spec::Qwen3Config;
use safetensors::{tensor::Dtype, SafeTensors};

use crate::model::{Attention, CudaModel, Mlp, TransformerBlock};
use crate::ops::{precompute_rope, upload_i32};
use crate::quant_format::{
    detect_quant_format, read_quant_manifest, reject_dsv4_e8m0_scale_abi, QuantFormat,
    QuantManifest, QuantTensorView, ScaleApply, TensorHeader,
};

const DEFAULT_ROPE_CACHE_LEN: usize = 32_768;
const DEFAULT_SHARD_CACHE_BYTES: usize = 8 * 1024 * 1024 * 1024;

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
    tensor_headers: std::cell::RefCell<Option<Rc<BTreeMap<String, TensorHeader>>>>,
    quant_manifest: Option<QuantManifest>,
    /// Bounded shard byte cache: without it, loading tensors that alternate
    /// across two shards re-reads multi-GiB files on every tensor touch. `Rc`
    /// lets [`SharedTensor`] expose a tensor's byte range zero-copy while the
    /// cache entry itself can be evicted after the layer no longer needs it.
    /// `Rc<Vec<u8>>` over clippy's `Rc<[u8]>`: the conversion would copy the
    /// multi-GiB shard once more — the exact copy this cache exists to avoid.
    #[allow(clippy::rc_buffer)]
    shard_cache: std::cell::RefCell<ShardByteCache>,
    shard_meta_cache: std::cell::RefCell<HashMap<usize, Rc<BTreeMap<String, ShardTensorMeta>>>>,
}

#[derive(Clone, Debug)]
enum QuantMatrixShard {
    Full,
    Rows(ShardingSpec),
    Cols(ShardingSpec),
}

struct Fp8BlockProjectionView {
    weight_name: String,
    scale_name: String,
    rows: usize,
    cols: usize,
    scale_rows: usize,
    scale_cols: usize,
    scale_apply: ScaleApply,
}

struct DirectFp8MoeRouted {
    w13: MoeFp8ExpertGroup,
    down: MoeFp8ExpertGroup,
    gate_up_quant_signature: ExpertQuantDispatchSignature,
    down_quant_signature: ExpertQuantDispatchSignature,
}

#[allow(clippy::rc_buffer)]
struct ShardByteCache {
    entries: HashMap<usize, Rc<Vec<u8>>>,
    order: VecDeque<usize>,
    bytes: usize,
    max_bytes: usize,
}

impl ShardByteCache {
    fn new(max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
            max_bytes,
        }
    }

    #[allow(clippy::rc_buffer)] // Rc<[u8]> conversion would re-copy the shard
    fn get(&mut self, idx: usize) -> Option<Rc<Vec<u8>>> {
        let bytes = Rc::clone(self.entries.get(&idx)?);
        self.touch(idx);
        Some(bytes)
    }

    #[allow(clippy::rc_buffer)] // Rc<[u8]> conversion would re-copy the shard
    fn insert(&mut self, idx: usize, bytes: Rc<Vec<u8>>) -> Vec<(usize, usize)> {
        if let Some(old) = self.entries.remove(&idx) {
            self.bytes = self.bytes.saturating_sub(old.len());
            self.remove_order(idx);
        }

        let incoming = bytes.len();
        let mut evicted = Vec::new();
        while !self.entries.is_empty() && self.bytes.saturating_add(incoming) > self.max_bytes {
            let Some(old_idx) = self.order.pop_front() else {
                break;
            };
            if let Some(old) = self.entries.remove(&old_idx) {
                self.bytes = self.bytes.saturating_sub(old.len());
                evicted.push((old_idx, old.len()));
            }
        }

        self.bytes = self.bytes.saturating_add(incoming);
        self.order.push_back(idx);
        self.entries.insert(idx, bytes);
        evicted
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn remove_order(&mut self, idx: usize) {
        if let Some(pos) = self.order.iter().position(|&entry| entry == idx) {
            self.order.remove(pos);
        }
    }

    fn touch(&mut self, idx: usize) {
        self.remove_order(idx);
        self.order.push_back(idx);
    }
}

#[derive(Clone, Debug)]
struct ShardTensorMeta {
    shape: Vec<usize>,
    dtype: Dtype,
    offset: usize,
    len: usize,
}

fn shard_cache_bytes_limit() -> usize {
    env::var("ARLE_CUDA_SHARD_CACHE_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_SHARD_CACHE_BYTES)
}

impl SafetensorLoader {
    pub(crate) fn new(base: &Path) -> Result<Self> {
        let t0 = Instant::now();
        let quant_manifest = if base.join("config.json").exists() {
            read_quant_manifest(base)?
        } else {
            None
        };
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
            let loader = Self {
                base: base.to_path_buf(),
                shards,
                weight_map,
                tensor_headers: std::cell::RefCell::new(None),
                quant_manifest,
                shard_cache: std::cell::RefCell::new(
                    ShardByteCache::new(shard_cache_bytes_limit()),
                ),
                shard_meta_cache: std::cell::RefCell::new(HashMap::new()),
            };
            loader.log_startup_phase(
                "new.index",
                t0,
                format_args!(
                    "shards={} weight_map={} quant_manifest={}",
                    loader.shards.len(),
                    loader.weight_map.len(),
                    loader.quant_manifest.is_some()
                ),
            );
            return Ok(loader);
        }

        let single = base.join("model.safetensors");
        if single.exists() {
            let loader = Self {
                base: base.to_path_buf(),
                shards: vec![single],
                weight_map: HashMap::new(),
                tensor_headers: std::cell::RefCell::new(None),
                quant_manifest,
                shard_cache: std::cell::RefCell::new(
                    ShardByteCache::new(shard_cache_bytes_limit()),
                ),
                shard_meta_cache: std::cell::RefCell::new(HashMap::new()),
            };
            loader.log_startup_phase(
                "new.single",
                t0,
                format_args!(
                    "shards=1 quant_manifest={}",
                    loader.quant_manifest.is_some()
                ),
            );
            return Ok(loader);
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
        let loader = Self {
            base: base.to_path_buf(),
            shards,
            weight_map: HashMap::new(),
            tensor_headers: std::cell::RefCell::new(None),
            quant_manifest,
            shard_cache: std::cell::RefCell::new(ShardByteCache::new(shard_cache_bytes_limit())),
            shard_meta_cache: std::cell::RefCell::new(HashMap::new()),
        };
        loader.log_startup_phase(
            "new.scan",
            t0,
            format_args!(
                "shards={} quant_manifest={}",
                loader.shards.len(),
                loader.quant_manifest.is_some()
            ),
        );
        Ok(loader)
    }

    fn startup_profile_enabled(&self) -> bool {
        std::env::var_os("ARLE_CUDA_STARTUP_PROFILE").is_some()
    }

    fn log_startup_phase(&self, phase: &str, start: Instant, extra: std::fmt::Arguments<'_>) {
        if self.startup_profile_enabled() {
            log::info!(
                target: "infer_cuda::startup",
                "cuda_startup phase=loader.{phase} elapsed_ms={:.1} {extra}",
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
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

    /// Load a Qwen3.5/3.6 matrix as resident quant when the checkpoint exposes
    /// a supported quant sidecar ABI; otherwise use the legacy BF16/F32 path.
    pub(crate) fn load_matrix_quant_aware(
        &self,
        ctx: &DeviceContext,
        name: &str,
    ) -> Result<DeviceMatrix> {
        match self.quant_view_for(name)? {
            Some(view) => self.load_quant_or_dense_view(ctx, &view, QuantMatrixShard::Full),
            None => self.load_matrix(ctx, name),
        }
    }

    /// Quant-aware twin of [`Self::load_matrix_sharded`].
    pub(crate) fn load_matrix_sharded_quant_aware(
        &self,
        ctx: &DeviceContext,
        name: &str,
        kind: infer_topo::ParallelLinearKind,
        tp: &infer_topo::TpConfig,
    ) -> Result<DeviceMatrix> {
        let Some(view) = self.quant_view_for(name)? else {
            return self.load_matrix_sharded(ctx, name, kind, tp);
        };
        ensure!(
            view.logical_shape.len() == 2,
            "{}: expected 2D quant-aware matrix, got {:?}",
            view.name,
            view.logical_shape
        );
        let (rows, cols) = (view.logical_shape[0], view.logical_shape[1]);
        let shard = match kind {
            infer_topo::ParallelLinearKind::Column => {
                QuantMatrixShard::Rows(infer_topo::column_shard(rows, tp))
            }
            infer_topo::ParallelLinearKind::Row => {
                QuantMatrixShard::Cols(infer_topo::row_shard(cols, tp))
            }
        };
        self.load_quant_or_dense_view(ctx, &view, shard)
    }

    /// Quant-aware twin of [`Self::load_qkv_head_sharded`].
    pub(crate) fn load_qkv_head_sharded_quant_aware(
        &self,
        ctx: &DeviceContext,
        name: &str,
        local_heads: usize,
        head_dim: usize,
        tp: &infer_topo::TpConfig,
    ) -> Result<DeviceMatrix> {
        let Some(view) = self.quant_view_for(name)? else {
            return self.load_qkv_head_sharded(ctx, name, local_heads, head_dim, tp);
        };
        ensure!(
            view.logical_shape.len() == 2,
            "{}: expected 2D quant-aware QKV matrix, got {:?}",
            view.name,
            view.logical_shape
        );
        let total_rows = view.logical_shape[0];
        let local_rows = local_heads * head_dim;
        let offset = tp.rank * local_rows;
        ensure!(
            offset + local_rows <= total_rows,
            "{}: head shard [{offset}, {}) exceeds rows {total_rows} \
             (local_heads={local_heads}, head_dim={head_dim}, rank={})",
            view.name,
            offset + local_rows,
            tp.rank
        );
        self.load_quant_or_dense_view(
            ctx,
            &view,
            QuantMatrixShard::Rows(ShardingSpec {
                offset,
                size: local_rows,
                total: total_rows,
            }),
        )
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
        let layer_t0 = Instant::now();
        const BF16_ELEM_SIZE: usize = 2;
        let mut gate = Vec::with_capacity(split.experts_per_rank);
        let mut up = Vec::with_capacity(split.experts_per_rank);
        let mut down = Vec::with_capacity(split.experts_per_rank);
        let per_expert_probe = names.expert_gate_proj(split.local_expert_start);
        let per_expert_quant_probe = self.quant_view_for(&per_expert_probe)?.is_some();
        let deepgemm_native_ready = crate::moe::qwen35_deepgemm_enabled()
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
        let mut direct_fp8_routed = None;
        // The stacked tensors are HF `nn.Parameter`s — no `.weight` suffix on
        // the real Qwen3.6-35B-A3B checkpoint — but accept a `.weight`-suffixed
        // export too.
        let resolve_stacked = |base: &str| -> Option<String> {
            [base.to_string(), format!("{base}.weight")]
                .into_iter()
                .find(|name| self.has_tensor(name))
        };
        if per_expert_quant_probe && deepgemm_native_ready {
            direct_fp8_routed = self.load_fp8_moe_groups_direct(
                ctx,
                names,
                split,
                moe_intermediate_size,
                hidden_size,
            )?;
        }
        if direct_fp8_routed.is_some() {
            // The direct FP8 path fills the resident DeepGEMM grouped caches
            // without constructing the transient per-expert DeviceMatrix list.
        } else if self.has_tensor(&per_expert_probe) || per_expert_quant_probe {
            for e in split.local_expert_start..split.local_expert_end() {
                gate.push(self.load_matrix_quant_aware(ctx, &names.expert_gate_proj(e))?);
                up.push(self.load_matrix_quant_aware(ctx, &names.expert_up_proj(e))?);
                down.push(self.load_matrix_quant_aware(ctx, &names.expert_down_proj(e))?);
            }
        } else if let Some(gate_up_name) = resolve_stacked(&names.experts_stacked_gate_up_proj) {
            let routed_t0 = Instant::now();
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
            self.log_startup_phase(
                "moe.stacked_routed_load",
                routed_t0,
                format_args!(
                    "layer={} local_experts={} gate={} up={} down={}",
                    names.mlp_prefix,
                    split.experts_per_rank,
                    gate.len(),
                    up.len(),
                    down.len()
                ),
            );
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
        self.log_startup_phase(
            "moe.routed_load",
            layer_t0,
            format_args!(
                "layer={} local_experts={} gate={} up={} down={} direct_fp8_grouped={}",
                names.mlp_prefix,
                split.experts_per_rank,
                gate.len(),
                up.len(),
                down.len(),
                direct_fp8_routed.is_some()
            ),
        );
        let shared_t0 = Instant::now();
        let router_gate = self.load_matrix(ctx, &names.router_gate)?;
        let (shared_gate, shared_up, shared_down) = if tp.is_single() {
            (
                self.load_matrix_quant_aware(ctx, &names.shared_expert_gate_proj)?,
                self.load_matrix_quant_aware(ctx, &names.shared_expert_up_proj)?,
                self.load_matrix_quant_aware(ctx, &names.shared_expert_down_proj)?,
            )
        } else {
            (
                self.load_matrix_sharded_quant_aware(
                    ctx,
                    &names.shared_expert_gate_proj,
                    infer_topo::ParallelLinearKind::Column,
                    tp,
                )?,
                self.load_matrix_sharded_quant_aware(
                    ctx,
                    &names.shared_expert_up_proj,
                    infer_topo::ParallelLinearKind::Column,
                    tp,
                )?,
                self.load_matrix_sharded_quant_aware(
                    ctx,
                    &names.shared_expert_down_proj,
                    infer_topo::ParallelLinearKind::Row,
                    tp,
                )?,
            )
        };
        let shared_gate_router = self.load_matrix(ctx, &names.shared_expert_gate)?;
        self.log_startup_phase(
            "moe.shared_load",
            shared_t0,
            format_args!("layer={}", names.mlp_prefix),
        );

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
        let (expert_weight_format, gate_sig, down_sig) =
            if let Some(direct) = direct_fp8_routed.as_ref() {
                (
                    WeightFormat::Fp8BlockScaled,
                    Some(direct.gate_up_quant_signature),
                    Some(direct.down_quant_signature),
                )
            } else {
                routed_expert_weight_format(&gate, &up, &down)?
            };
        let routed_quant = expert_weight_format.is_quantized();
        let grouped_t0 = Instant::now();
        let deepgemm_ready = !routed_quant && deepgemm_native_ready;
        let fp8_deepgemm_ready =
            expert_weight_format == WeightFormat::Fp8BlockScaled && deepgemm_native_ready;
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
        let (w13_fp8_grouped, down_fp8_grouped) = if let Some(direct) = direct_fp8_routed.take() {
            (Some(direct.w13), Some(direct.down))
        } else if fp8_deepgemm_ready {
            let w13_g = MoeFp8ExpertGroup::concat_pair_rows(
                ctx,
                &gate,
                &up,
                moe_intermediate_size,
                hidden_size,
            )?;
            let down_g = MoeFp8ExpertGroup::concat(ctx, &down, hidden_size, moe_intermediate_size)?;
            // The grouped caches now own the resident FP8 expert bytes. Sync
            // before dropping sources because cudarc event tracking is disabled
            // and the D2D concats above are async on the compute stream.
            ctx.sync()?;
            (Some(w13_g), Some(down_g))
        } else {
            (None, None)
        };

        // Per-expert weight-pointer tables (one device pointer per owned
        // expert) — built from the per-expert matrices on the default path,
        // or from the grouped buffer offsets on the DeepGEMM path.
        let (gate_ptrs, up_ptrs, down_ptrs) = match (
            &gate_grouped,
            &up_grouped,
            &down_grouped,
            &w13_fp8_grouped,
            &down_fp8_grouped,
        ) {
            (Some(g), Some(u), Some(d), _, _) => {
                (g.ptr_table(ctx)?, u.ptr_table(ctx)?, d.ptr_table(ctx)?)
            }
            (None, None, None, Some(w13), Some(down_g)) => (
                w13.qweight_ptr_table(ctx, 0)?,
                w13.qweight_ptr_table(ctx, moe_intermediate_size)?,
                down_g.qweight_ptr_table(ctx, 0)?,
            ),
            _ => {
                let gate_refs: Vec<&DeviceMatrix> = gate.iter().collect();
                let up_refs: Vec<&DeviceMatrix> = up.iter().collect();
                let down_refs: Vec<&DeviceMatrix> = down.iter().collect();
                if routed_quant {
                    (
                        cuda_kernels::moe::build_expert_qweight_u8_ptr_table(ctx, &gate_refs)?,
                        cuda_kernels::moe::build_expert_qweight_u8_ptr_table(ctx, &up_refs)?,
                        cuda_kernels::moe::build_expert_qweight_u8_ptr_table(ctx, &down_refs)?,
                    )
                } else {
                    (
                        cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &gate_refs)?,
                        cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &up_refs)?,
                        cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &down_refs)?,
                    )
                }
            }
        };
        let (gate_scale_ptrs, up_scale_ptrs, down_scale_ptrs) = if fp8_deepgemm_ready {
            let w13 = w13_fp8_grouped
                .as_ref()
                .ok_or_else(|| anyhow!("FP8 DeepGEMM w13 cache missing after build"))?;
            let down_g = down_fp8_grouped
                .as_ref()
                .ok_or_else(|| anyhow!("FP8 DeepGEMM down cache missing after build"))?;
            (
                Some(w13.scale_ptr_table(ctx, 0)?),
                Some(w13.scale_ptr_table(ctx, moe_intermediate_size)?),
                Some(down_g.scale_ptr_table(ctx, 0)?),
            )
        } else if routed_quant
            && matches!(
                expert_weight_format,
                WeightFormat::Fp8BlockScaled | WeightFormat::Fp8PerShard
            )
        {
            let gate_refs: Vec<&DeviceMatrix> = gate.iter().collect();
            let up_refs: Vec<&DeviceMatrix> = up.iter().collect();
            let down_refs: Vec<&DeviceMatrix> = down.iter().collect();
            (
                Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                    ctx, &gate_refs,
                )?),
                Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                    ctx, &up_refs,
                )?),
                Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                    ctx, &down_refs,
                )?),
            )
        } else if routed_quant && expert_weight_format == WeightFormat::Fp4E2M1Group {
            let gate_refs: Vec<&DeviceMatrix> = gate.iter().collect();
            let up_refs: Vec<&DeviceMatrix> = up.iter().collect();
            let down_refs: Vec<&DeviceMatrix> = down.iter().collect();
            (
                Some(cuda_kernels::moe::build_expert_qscale_fp8_ptr_table(
                    ctx, &gate_refs,
                )?),
                Some(cuda_kernels::moe::build_expert_qscale_fp8_ptr_table(
                    ctx, &up_refs,
                )?),
                Some(cuda_kernels::moe::build_expert_qscale_fp8_ptr_table(
                    ctx, &down_refs,
                )?),
            )
        } else {
            (None, None, None)
        };
        let (gate_global_ptrs, up_global_ptrs, down_global_ptrs) =
            if routed_quant && expert_weight_format == WeightFormat::Fp4E2M1Group {
                let gate_refs: Vec<&DeviceMatrix> = gate.iter().collect();
                let up_refs: Vec<&DeviceMatrix> = up.iter().collect();
                let down_refs: Vec<&DeviceMatrix> = down.iter().collect();
                (
                    Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                        ctx, &gate_refs,
                    )?),
                    Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                        ctx, &up_refs,
                    )?),
                    Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                        ctx, &down_refs,
                    )?),
                )
            } else {
                (None, None, None)
            };
        if fp8_deepgemm_ready {
            gate.clear();
            up.clear();
            down.clear();
        }
        self.log_startup_phase(
            "moe.grouped_cache",
            grouped_t0,
            format_args!(
                "layer={} format={expert_weight_format:?} fp8_deepgemm_ready={} routed_quant={} retained_gate={} retained_up={} retained_down={}",
                names.mlp_prefix,
                fp8_deepgemm_ready,
                routed_quant,
                gate.len(),
                up.len(),
                down.len()
            ),
        );

        Ok(MoeLayerWeights {
            gate,
            up,
            down,
            expert_weight_format,
            gate_up_quant_signature: gate_sig,
            down_quant_signature: down_sig,
            gate_ptrs,
            up_ptrs,
            down_ptrs,
            gate_scale_ptrs,
            up_scale_ptrs,
            down_scale_ptrs,
            gate_global_ptrs,
            up_global_ptrs,
            down_global_ptrs,
            gate_grouped,
            up_grouped,
            down_grouped,
            w13_fp8_grouped,
            down_fp8_grouped,
            router_gate,
            shared_gate,
            shared_up,
            shared_down,
            shared_gate_router,
        })
    }

    fn load_fp8_moe_groups_direct(
        &self,
        ctx: &DeviceContext,
        names: &qwen35_spec::Qwen35MoeTensorNames,
        split: &crate::moe_config::ExpertSplit,
        moe_intermediate_size: usize,
        hidden_size: usize,
    ) -> Result<Option<DirectFp8MoeRouted>> {
        let t0 = Instant::now();
        ensure!(
            moe_intermediate_size.is_multiple_of(128) && hidden_size.is_multiple_of(128),
            "Qwen3.6 FP8 direct grouped MoE needs 128-aligned dims, got mi={moe_intermediate_size} hidden={hidden_size}"
        );
        let groups = split.experts_per_rank;
        let w13_rows = 2 * moe_intermediate_size;
        let w13_scale_rows = w13_rows / 128;
        let w13_scale_cols = hidden_size / 128;
        let down_scale_rows = hidden_size / 128;
        let down_scale_cols = moe_intermediate_size / 128;
        let mut expert_views = Vec::with_capacity(groups);
        let mut shard_idx = None;
        let gate_up_sig = ExpertQuantDispatchSignature {
            rows: moe_intermediate_size,
            cols: hidden_size,
            quant_scale_rows: moe_intermediate_size / 128,
            quant_scale_cols: hidden_size / 128,
            quant_block_m: 128,
            quant_block_k: 128,
            group_size: 0,
        };
        let down_sig = ExpertQuantDispatchSignature {
            rows: hidden_size,
            cols: moe_intermediate_size,
            quant_scale_rows: hidden_size / 128,
            quant_scale_cols: moe_intermediate_size / 128,
            quant_block_m: 128,
            quant_block_k: 128,
            group_size: 0,
        };

        for e in split.local_expert_start..split.local_expert_end() {
            let gate = match self.fp8_block_projection_view(
                &names.expert_gate_proj(e),
                moe_intermediate_size,
                hidden_size,
            )? {
                Some(view) => view,
                None => return Ok(None),
            };
            let up = match self.fp8_block_projection_view(
                &names.expert_up_proj(e),
                moe_intermediate_size,
                hidden_size,
            )? {
                Some(view) => view,
                None => return Ok(None),
            };
            let down = match self.fp8_block_projection_view(
                &names.expert_down_proj(e),
                hidden_size,
                moe_intermediate_size,
            )? {
                Some(view) => view,
                None => return Ok(None),
            };
            for view in [&gate, &up, &down] {
                let Some(weight_idx) = self.weight_map.get(&view.weight_name).copied() else {
                    return Ok(None);
                };
                let Some(scale_idx) = self.weight_map.get(&view.scale_name).copied() else {
                    return Ok(None);
                };
                if weight_idx != scale_idx {
                    return Ok(None);
                }
                match shard_idx {
                    Some(idx) if idx != weight_idx => return Ok(None),
                    Some(_) => {}
                    None => shard_idx = Some(weight_idx),
                }
            }
            expert_views.push((gate, up, down));
        }
        let Some(shard_idx) = shard_idx else {
            return Ok(None);
        };

        let mut w13_weight = vec![0u8; groups * w13_rows * hidden_size];
        let mut w13_scales = vec![0f32; groups * w13_scale_rows * w13_scale_cols];
        let mut down_weight = vec![0u8; groups * hidden_size * moe_intermediate_size];
        let mut down_scales = vec![0f32; groups * down_scale_rows * down_scale_cols];
        let shard = self.shard_bytes(shard_idx)?;
        let tensors = SafeTensors::deserialize(&shard)
            .with_context(|| format!("deserialize {}", self.shards[shard_idx].display()))?;

        for (g, (gate, up, down)) in expert_views.iter().enumerate() {
            let w13_weight_base = g * w13_rows * hidden_size;
            let gate_weight = &mut w13_weight
                [w13_weight_base..w13_weight_base + moe_intermediate_size * hidden_size];
            self.copy_fp8_projection_from_shard(&tensors, gate, gate_weight)?;
            let up_weight_start = w13_weight_base + moe_intermediate_size * hidden_size;
            let up_weight = &mut w13_weight
                [up_weight_start..up_weight_start + moe_intermediate_size * hidden_size];
            self.copy_fp8_projection_from_shard(&tensors, up, up_weight)?;

            let w13_scale_base = g * w13_scale_rows * w13_scale_cols;
            let gate_scales =
                &mut w13_scales[w13_scale_base..w13_scale_base + gate.scale_rows * gate.scale_cols];
            self.copy_fp8_scales_from_shard(&tensors, gate, gate_scales)?;
            let up_scale_start = w13_scale_base + (moe_intermediate_size / 128) * w13_scale_cols;
            let up_scales =
                &mut w13_scales[up_scale_start..up_scale_start + up.scale_rows * up.scale_cols];
            self.copy_fp8_scales_from_shard(&tensors, up, up_scales)?;

            let down_weight_base = g * hidden_size * moe_intermediate_size;
            let down_weight_dst = &mut down_weight
                [down_weight_base..down_weight_base + hidden_size * moe_intermediate_size];
            self.copy_fp8_projection_from_shard(&tensors, down, down_weight_dst)?;
            let down_scale_base = g * down_scale_rows * down_scale_cols;
            let down_scales_dst = &mut down_scales
                [down_scale_base..down_scale_base + down.scale_rows * down.scale_cols];
            self.copy_fp8_scales_from_shard(&tensors, down, down_scales_dst)?;
        }

        let w13 = MoeFp8ExpertGroup::from_host(
            ctx,
            &w13_weight,
            &w13_scales,
            groups,
            w13_rows,
            hidden_size,
        )?;
        let down = MoeFp8ExpertGroup::from_host(
            ctx,
            &down_weight,
            &down_scales,
            groups,
            hidden_size,
            moe_intermediate_size,
        )?;
        self.log_startup_phase(
            "moe.direct_fp8_grouped_load",
            t0,
            format_args!(
                "layer={} shard_idx={} local_experts={} w13_bytes={} down_bytes={}",
                names.mlp_prefix,
                shard_idx,
                groups,
                w13_weight.len(),
                down_weight.len()
            ),
        );
        Ok(Some(DirectFp8MoeRouted {
            w13,
            down,
            gate_up_quant_signature: gate_up_sig,
            down_quant_signature: down_sig,
        }))
    }

    fn fp8_block_projection_view(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Option<Fp8BlockProjectionView>> {
        let Some(view) = self.quant_view_for(name)? else {
            return Ok(None);
        };
        let QuantFormat::Fp8BlockScaled {
            block_m,
            block_k,
            scale_apply,
        } = view.format
        else {
            return Ok(None);
        };
        ensure!(
            block_m == 128 && block_k == 128,
            "{}: direct FP8 grouped MoE supports 128x128 block scales, got {block_m}x{block_k}",
            view.name
        );
        ensure!(
            view.storage_dtype == Dtype::F8_E4M3 && view.logical_shape == [rows, cols],
            "{}: expected FP8 projection [{rows}, {cols}], got {:?} {:?}",
            view.name,
            view.storage_dtype,
            view.logical_shape
        );
        let scale_name = view
            .scale_names
            .first()
            .ok_or_else(|| anyhow!("{}: FP8 projection missing scale tensor", view.name))?
            .clone();
        Ok(Some(Fp8BlockProjectionView {
            weight_name: view.name,
            scale_name,
            rows,
            cols,
            scale_rows: rows / 128,
            scale_cols: cols / 128,
            scale_apply,
        }))
    }

    fn copy_fp8_projection_from_shard(
        &self,
        tensors: &SafeTensors<'_>,
        view: &Fp8BlockProjectionView,
        dst: &mut [u8],
    ) -> Result<()> {
        let tensor = tensors
            .tensor(&view.weight_name)
            .with_context(|| format!("find tensor {}", view.weight_name))?;
        ensure!(
            tensor.dtype() == Dtype::F8_E4M3 && tensor.shape() == [view.rows, view.cols],
            "{}: expected F8_E4M3 [{}, {}], got {:?} {:?}",
            view.weight_name,
            view.rows,
            view.cols,
            tensor.dtype(),
            tensor.shape()
        );
        let data = tensor.data();
        ensure!(
            data.len() == dst.len(),
            "{}: FP8 weight bytes {} != destination {}",
            view.weight_name,
            data.len(),
            dst.len()
        );
        dst.copy_from_slice(data);
        Ok(())
    }

    fn copy_fp8_scales_from_shard(
        &self,
        tensors: &SafeTensors<'_>,
        view: &Fp8BlockProjectionView,
        dst: &mut [f32],
    ) -> Result<()> {
        let tensor = tensors
            .tensor(&view.scale_name)
            .with_context(|| format!("find tensor {}", view.scale_name))?;
        ensure!(
            (tensor.dtype() == Dtype::BF16 || tensor.dtype() == Dtype::F32)
                && tensor.shape() == [view.scale_rows, view.scale_cols],
            "{}: expected BF16/F32 scale [{}, {}], got {:?} {:?}",
            view.scale_name,
            view.scale_rows,
            view.scale_cols,
            tensor.dtype(),
            tensor.shape()
        );
        let scales = tensor_bytes_to_f32(
            &view.scale_name,
            tensor.dtype(),
            tensor.data(),
            view.scale_apply,
        )?;
        ensure!(
            scales.len() == dst.len(),
            "{}: FP8 scale values {} != destination {}",
            view.scale_name,
            scales.len(),
            dst.len()
        );
        dst.copy_from_slice(&scales);
        Ok(())
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
        self.tensor_headers()
            .map(|headers| headers.contains_key(name))
            .unwrap_or(false)
    }

    fn quant_view_for(&self, name: &str) -> Result<Option<QuantTensorView>> {
        if self.quant_manifest.is_none() {
            return Ok(None);
        }
        let headers = self.tensor_headers()?;
        let mut candidates = vec![name.to_owned()];
        if let Some(base) = name.strip_suffix(".weight") {
            candidates.push(format!("{base}.weight_packed"));
        }
        for candidate in candidates {
            if !headers.contains_key(&candidate) {
                continue;
            }
            reject_dsv4_e8m0_scale_abi(&candidate, headers.as_ref())?;
            if let Some(view) =
                detect_quant_format(&candidate, headers.as_ref(), self.quant_manifest.as_ref())?
            {
                return Ok(Some(view));
            }
        }
        Ok(None)
    }

    fn tensor_headers(&self) -> Result<Rc<BTreeMap<String, TensorHeader>>> {
        if let Some(headers) = self.tensor_headers.borrow().as_ref() {
            return Ok(Rc::clone(headers));
        }
        let t0 = Instant::now();
        let mut headers = BTreeMap::new();
        for idx in 0..self.shards.len() {
            let shard_t0 = Instant::now();
            let shard_headers = self.read_shard_headers(idx)?;
            let tensor_count = shard_headers.len();
            let header_bytes = self.safetensors_header_len(idx)?;
            self.log_startup_phase(
                "tensor_headers.shard",
                shard_t0,
                format_args!(
                    "idx={idx} header_bytes={} tensors={} path={}",
                    header_bytes,
                    tensor_count,
                    self.shards[idx].display()
                ),
            );
            headers.extend(shard_headers);
        }
        let headers = Rc::new(headers);
        *self.tensor_headers.borrow_mut() = Some(Rc::clone(&headers));
        self.log_startup_phase(
            "tensor_headers.total",
            t0,
            format_args!(
                "shards={} tensors={} cached_shards={}",
                self.shards.len(),
                headers.len(),
                self.shard_cache.borrow().len()
            ),
        );
        Ok(headers)
    }

    fn read_shard_headers(&self, idx: usize) -> Result<BTreeMap<String, TensorHeader>> {
        let path = self
            .shards
            .get(idx)
            .ok_or_else(|| anyhow!("shard index {idx} out of range"))?;
        let header = self.read_safetensors_header_bytes(idx)?;
        let raw: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header)
            .with_context(|| format!("parse safetensors header {}", path.display()))?;
        let mut headers = BTreeMap::new();
        for (name, value) in raw {
            if name == "__metadata__" {
                continue;
            }
            let tensor: SafetensorHeaderTensor = serde_json::from_value(value)
                .with_context(|| format!("parse safetensors tensor header {name}"))?;
            headers.insert(
                name,
                TensorHeader {
                    dtype: tensor.dtype,
                    shape: tensor.shape,
                },
            );
        }
        Ok(headers)
    }

    fn safetensors_header_len(&self, idx: usize) -> Result<usize> {
        Ok(self.read_safetensors_header_len(idx)?.1)
    }

    fn read_safetensors_header_bytes(&self, idx: usize) -> Result<Vec<u8>> {
        let (mut file, header_len) = self.read_safetensors_header_len(idx)?;
        let mut header = vec![0u8; header_len];
        file.read_exact(&mut header)
            .with_context(|| format!("read safetensors header {}", self.shards[idx].display()))?;
        Ok(header)
    }

    fn read_safetensors_header_len(&self, idx: usize) -> Result<(fs::File, usize)> {
        let path = self
            .shards
            .get(idx)
            .ok_or_else(|| anyhow!("shard index {idx} out of range"))?;
        let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut len_bytes = [0u8; 8];
        file.read_exact(&mut len_bytes)
            .with_context(|| format!("read safetensors header length {}", path.display()))?;
        let header_len = usize::try_from(u64::from_le_bytes(len_bytes)).with_context(|| {
            format!("safetensors header length too large in {}", path.display())
        })?;
        ensure!(
            header_len > 0,
            "{}: safetensors header length is zero",
            path.display()
        );
        Ok((file, header_len))
    }

    fn load_quant_or_dense_view(
        &self,
        ctx: &DeviceContext,
        view: &QuantTensorView,
        shard: QuantMatrixShard,
    ) -> Result<DeviceMatrix> {
        match view.format {
            QuantFormat::DenseBf16 => match shard {
                QuantMatrixShard::Full => self.load_matrix(ctx, &view.name),
                QuantMatrixShard::Rows(spec) => self.load_matrix_sharded_by_spec(
                    ctx,
                    &view.name,
                    infer_topo::ParallelLinearKind::Column,
                    &spec,
                ),
                QuantMatrixShard::Cols(spec) => self.load_matrix_sharded_by_spec(
                    ctx,
                    &view.name,
                    infer_topo::ParallelLinearKind::Row,
                    &spec,
                ),
            },
            QuantFormat::DenseF32 => {
                let tensor = self.load_raw_tensor(&view.name)?;
                ensure!(
                    tensor.shape.len() == 2,
                    "{}: expected 2D F32 tensor, got {:?}",
                    view.name,
                    tensor.shape
                );
                let sharded =
                    self.shard_raw_2d(&tensor.bytes, tensor.shape[0], tensor.shape[1], 4, &shard)?;
                let owned = OwnedTensor {
                    shape: vec![sharded.rows, sharded.cols],
                    bytes: sharded.bytes,
                    dtype: Dtype::F32,
                };
                DeviceMatrix::from_safetensors(
                    ctx,
                    Self::dsv4_bytes_to_bf16(&view.name, &owned)?.as_ref(),
                    owned.shape[0],
                    owned.shape[1],
                )
                .with_context(|| format!("upload dense F32 tensor {}", view.name))
            }
            QuantFormat::Fp8BlockScaled {
                block_m,
                block_k,
                scale_apply,
            } => self.load_fp8_block_scaled_view(ctx, view, &shard, block_m, block_k, scale_apply),
            QuantFormat::Fp8PerShard { scale_apply } => {
                self.load_fp8_per_shard_view(ctx, view, &shard, scale_apply)
            }
            QuantFormat::Fp4E2M1Group {
                group_size,
                global_scale_apply,
            } => self.load_fp4_group_view(ctx, view, &shard, group_size, global_scale_apply),
        }
    }

    fn load_matrix_sharded_by_spec(
        &self,
        ctx: &DeviceContext,
        name: &str,
        kind: infer_topo::ParallelLinearKind,
        spec: &ShardingSpec,
    ) -> Result<DeviceMatrix> {
        const BF16_ELEM_SIZE: usize = 2;
        let tensor = self.load_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D BF16 tensor, got shape {:?}",
            tensor.shape
        );
        let sharded = match kind {
            infer_topo::ParallelLinearKind::Column => crate::shard_slice::shard_column_parallel(
                &tensor.bytes,
                tensor.shape[0],
                tensor.shape[1],
                BF16_ELEM_SIZE,
                spec,
            )?,
            infer_topo::ParallelLinearKind::Row => crate::shard_slice::shard_row_parallel(
                &tensor.bytes,
                tensor.shape[0],
                tensor.shape[1],
                BF16_ELEM_SIZE,
                spec,
            )?,
        };
        DeviceMatrix::from_safetensors(ctx, &sharded.bytes, sharded.rows, sharded.cols)
            .with_context(|| format!("upload sharded tensor {name}"))
    }

    fn shard_raw_2d(
        &self,
        bytes: &[u8],
        rows: usize,
        cols: usize,
        elem_size: usize,
        shard: &QuantMatrixShard,
    ) -> Result<crate::shard_slice::ShardedBytes> {
        match shard {
            QuantMatrixShard::Full => Ok(crate::shard_slice::ShardedBytes {
                bytes: bytes.to_vec(),
                rows,
                cols,
            }),
            QuantMatrixShard::Rows(spec) => {
                crate::shard_slice::shard_column_parallel(bytes, rows, cols, elem_size, spec)
            }
            QuantMatrixShard::Cols(spec) => {
                crate::shard_slice::shard_row_parallel(bytes, rows, cols, elem_size, spec)
            }
        }
    }

    fn load_fp8_block_scaled_view(
        &self,
        ctx: &DeviceContext,
        view: &QuantTensorView,
        shard: &QuantMatrixShard,
        block_m: usize,
        block_k: usize,
        scale_apply: ScaleApply,
    ) -> Result<DeviceMatrix> {
        let weight = self.load_raw_tensor(&view.name)?;
        ensure!(
            weight.dtype == Dtype::F8_E4M3 && weight.shape == view.logical_shape,
            "{}: expected F8_E4M3 {:?}, got {:?} {:?}",
            view.name,
            view.logical_shape,
            weight.dtype,
            weight.shape
        );
        let rows = view.logical_shape[0];
        let cols = view.logical_shape[1];
        let weight_shard = self.shard_raw_2d(&weight.bytes, rows, cols, 1, shard)?;
        let scale = self.load_raw_tensor(&view.scale_names[0])?;
        let scale_elem = float_elem_size(&view.scale_names[0], scale.dtype)?;
        let scale_rows = rows.div_ceil(block_m);
        let scale_cols = cols.div_ceil(block_k);
        ensure!(
            scale.shape == [scale_rows, scale_cols],
            "{}: scale shape {:?} != [{scale_rows}, {scale_cols}]",
            view.scale_names[0],
            scale.shape
        );
        let scale_shard = self.shard_fp8_block_scales(
            &scale.bytes,
            scale_elem,
            rows,
            cols,
            block_m,
            block_k,
            shard,
        )?;
        let scales = tensor_bytes_to_f32(
            &view.scale_names[0],
            scale.dtype,
            &scale_shard.bytes,
            scale_apply,
        )?;
        DeviceMatrix::from_fp8_block_scaled(
            ctx,
            &weight_shard.bytes,
            &scales,
            weight_shard.rows,
            weight_shard.cols,
            block_m,
            block_k,
        )
        .with_context(|| format!("upload FP8 block-scaled tensor {}", view.name))
    }

    fn shard_fp8_block_scales(
        &self,
        bytes: &[u8],
        elem_size: usize,
        rows: usize,
        cols: usize,
        block_m: usize,
        block_k: usize,
        shard: &QuantMatrixShard,
    ) -> Result<crate::shard_slice::ShardedBytes> {
        let scale_rows = rows.div_ceil(block_m);
        let scale_cols = cols.div_ceil(block_k);
        match shard {
            QuantMatrixShard::Full => Ok(crate::shard_slice::ShardedBytes {
                bytes: bytes.to_vec(),
                rows: scale_rows,
                cols: scale_cols,
            }),
            QuantMatrixShard::Rows(spec) => {
                ensure!(
                    spec.offset.is_multiple_of(block_m)
                        && (spec.end() == rows || spec.end().is_multiple_of(block_m)),
                    "FP8 block row shard {:?} must align to block_m={block_m} for rows={rows}",
                    spec.range()
                );
                let scale_spec = ShardingSpec {
                    offset: spec.offset / block_m,
                    size: spec.size.div_ceil(block_m),
                    total: scale_rows,
                };
                crate::shard_slice::shard_column_parallel(
                    bytes,
                    scale_rows,
                    scale_cols,
                    elem_size,
                    &scale_spec,
                )
            }
            QuantMatrixShard::Cols(spec) => {
                ensure!(
                    spec.offset.is_multiple_of(block_k)
                        && (spec.end() == cols || spec.end().is_multiple_of(block_k)),
                    "FP8 block col shard {:?} must align to block_k={block_k} for cols={cols}",
                    spec.range()
                );
                let scale_spec = ShardingSpec {
                    offset: spec.offset / block_k,
                    size: spec.size.div_ceil(block_k),
                    total: scale_cols,
                };
                crate::shard_slice::shard_row_parallel(
                    bytes,
                    scale_rows,
                    scale_cols,
                    elem_size,
                    &scale_spec,
                )
            }
        }
    }

    fn load_fp8_per_shard_view(
        &self,
        ctx: &DeviceContext,
        view: &QuantTensorView,
        shard: &QuantMatrixShard,
        scale_apply: ScaleApply,
    ) -> Result<DeviceMatrix> {
        let weight = self.load_raw_tensor(&view.name)?;
        ensure!(
            weight.dtype == Dtype::F8_E4M3 && weight.shape == view.logical_shape,
            "{}: expected F8_E4M3 {:?}, got {:?} {:?}",
            view.name,
            view.logical_shape,
            weight.dtype,
            weight.shape
        );
        let rows = view.logical_shape[0];
        let cols = view.logical_shape[1];
        let weight_shard = self.shard_raw_2d(&weight.bytes, rows, cols, 1, shard)?;
        let scale = self.load_raw_tensor(&view.scale_names[0])?;
        let input_scale = self.load_raw_tensor(&view.scale_names[1])?;
        let scales = tensor_to_f32_vec(&view.scale_names[0], &scale, scale_apply)?;
        let input_scales =
            tensor_to_f32_vec(&view.scale_names[1], &input_scale, ScaleApply::Multiply)?;
        ensure!(
            scales.len() == 1 && input_scales.len() == 1,
            "{}: FP8 per-shard dispatch currently requires scalar weight/input scales, got {}/{}",
            view.name,
            scales.len(),
            input_scales.len()
        );
        DeviceMatrix::from_fp8_per_shard(
            ctx,
            &weight_shard.bytes,
            &scales,
            &input_scales,
            weight_shard.rows,
            weight_shard.cols,
        )
        .with_context(|| format!("upload FP8 per-shard tensor {}", view.name))
    }

    fn load_fp4_group_view(
        &self,
        ctx: &DeviceContext,
        view: &QuantTensorView,
        shard: &QuantMatrixShard,
        group_size: usize,
        global_scale_apply: ScaleApply,
    ) -> Result<DeviceMatrix> {
        let weight = self.load_raw_tensor(&view.name)?;
        let rows = view.logical_shape[0];
        let cols = view.logical_shape[1];
        ensure!(
            weight.dtype == Dtype::U8 && weight.shape == [rows, cols / 2],
            "{}: expected packed U8 [{rows}, {}], got {:?} {:?}",
            view.name,
            cols / 2,
            weight.dtype,
            weight.shape
        );
        let weight_shard =
            self.shard_fp4_packed_weight(&weight.bytes, rows, cols, group_size, shard)?;
        let scale = self.load_raw_tensor(&view.scale_names[0])?;
        ensure!(
            scale.dtype == Dtype::F8_E4M3 && scale.shape == [rows, cols / group_size],
            "{}: expected FP8 group scale [{rows}, {}], got {:?} {:?}",
            view.scale_names[0],
            cols / group_size,
            scale.dtype,
            scale.shape
        );
        let scale_shard =
            self.shard_fp4_group_scales(&scale.bytes, rows, cols, group_size, shard)?;
        let global = self.load_raw_tensor(&view.scale_names[1])?;
        let global_scales = tensor_to_f32_vec(&view.scale_names[1], &global, global_scale_apply)?;
        ensure!(
            global_scales.len() == 1,
            "{}: FP4 global scale must be scalar, got {} values",
            view.scale_names[1],
            global_scales.len()
        );
        let input_scales = if view.scale_names.len() > 2 {
            let input = self.load_raw_tensor(&view.scale_names[2])?;
            Some(tensor_to_f32_vec(
                &view.scale_names[2],
                &input,
                ScaleApply::Multiply,
            )?)
        } else {
            None
        };
        DeviceMatrix::from_fp4_e2m1_group(
            ctx,
            &weight_shard.bytes,
            &scale_shard.bytes,
            &global_scales,
            input_scales.as_deref(),
            weight_shard.rows,
            weight_shard.cols * 2,
            group_size,
        )
        .with_context(|| format!("upload FP4 E2M1 tensor {}", view.name))
    }

    fn shard_fp4_packed_weight(
        &self,
        bytes: &[u8],
        rows: usize,
        logical_cols: usize,
        group_size: usize,
        shard: &QuantMatrixShard,
    ) -> Result<crate::shard_slice::ShardedBytes> {
        let packed_cols = logical_cols / 2;
        match shard {
            QuantMatrixShard::Full => Ok(crate::shard_slice::ShardedBytes {
                bytes: bytes.to_vec(),
                rows,
                cols: packed_cols,
            }),
            QuantMatrixShard::Rows(spec) => {
                crate::shard_slice::shard_column_parallel(bytes, rows, packed_cols, 1, spec)
            }
            QuantMatrixShard::Cols(spec) => {
                ensure!(
                    spec.offset.is_multiple_of(group_size)
                        && spec.size.is_multiple_of(group_size)
                        && spec.offset.is_multiple_of(2)
                        && spec.size.is_multiple_of(2),
                    "FP4 col shard {:?} must align to group_size={group_size} and packed pairs",
                    spec.range()
                );
                let packed_spec = ShardingSpec {
                    offset: spec.offset / 2,
                    size: spec.size / 2,
                    total: logical_cols / 2,
                };
                crate::shard_slice::shard_row_parallel(bytes, rows, packed_cols, 1, &packed_spec)
            }
        }
    }

    fn shard_fp4_group_scales(
        &self,
        bytes: &[u8],
        rows: usize,
        logical_cols: usize,
        group_size: usize,
        shard: &QuantMatrixShard,
    ) -> Result<crate::shard_slice::ShardedBytes> {
        let scale_cols = logical_cols / group_size;
        match shard {
            QuantMatrixShard::Full => Ok(crate::shard_slice::ShardedBytes {
                bytes: bytes.to_vec(),
                rows,
                cols: scale_cols,
            }),
            QuantMatrixShard::Rows(spec) => {
                crate::shard_slice::shard_column_parallel(bytes, rows, scale_cols, 1, spec)
            }
            QuantMatrixShard::Cols(spec) => {
                ensure!(
                    spec.offset.is_multiple_of(group_size) && spec.size.is_multiple_of(group_size),
                    "FP4 scale col shard {:?} must align to group_size={group_size}",
                    spec.range()
                );
                let scale_spec = ShardingSpec {
                    offset: spec.offset / group_size,
                    size: spec.size / group_size,
                    total: scale_cols,
                };
                crate::shard_slice::shard_row_parallel(bytes, rows, scale_cols, 1, &scale_spec)
            }
        }
    }

    /// Shard bytes: fill the bounded LRU on first touch, then hand out a cheap
    /// `Rc` clone (no `RefCell` guard escapes, so nested loads that fill other
    /// shards never hit a `BorrowMutError`). Loading beyond the byte budget
    /// evicts older entries; outstanding [`SharedTensor`] borrows keep their
    /// shard alive through their own `Rc`.
    #[allow(clippy::rc_buffer)] // Rc<[u8]> conversion would re-copy the shard
    fn shard_bytes(&self, idx: usize) -> Result<Rc<Vec<u8>>> {
        let path = self
            .shards
            .get(idx)
            .ok_or_else(|| anyhow!("shard index {idx} out of range"))?;
        if let Some(bytes) = self.shard_cache.borrow_mut().get(idx) {
            return Ok(bytes);
        }
        let t0 = Instant::now();
        let bytes = Rc::new(fs::read(path).with_context(|| format!("read {}", path.display()))?);
        let mut cache = self.shard_cache.borrow_mut();
        let evicted = cache.insert(idx, Rc::clone(&bytes));
        drop(cache);
        for (evicted_idx, evicted_bytes) in evicted {
            self.log_startup_phase(
                "shard_cache_evict",
                Instant::now(),
                format_args!("idx={evicted_idx} bytes={evicted_bytes}"),
            );
        }
        self.log_startup_phase(
            "shard_read",
            t0,
            format_args!("idx={idx} bytes={} path={}", bytes.len(), path.display()),
        );
        Ok(bytes)
    }

    #[allow(clippy::rc_buffer)] // shares the shard cache's Rc without copying
    fn shard_tensor_metas(
        &self,
        idx: usize,
        shard: &Rc<Vec<u8>>,
    ) -> Result<Rc<BTreeMap<String, ShardTensorMeta>>> {
        if let Some(metas) = self.shard_meta_cache.borrow().get(&idx) {
            return Ok(Rc::clone(metas));
        }
        let t0 = Instant::now();
        let path = &self.shards[idx];
        let tensors = SafeTensors::deserialize(shard)
            .with_context(|| format!("deserialize {}", path.display()))?;
        let base = shard.as_ptr() as usize;
        let mut metas = BTreeMap::new();
        for (name, view) in tensors.tensors() {
            let data = view.data();
            let offset = data.as_ptr() as usize - base;
            let len = data.len();
            ensure!(
                offset + len <= shard.len(),
                "{name}: tensor byte range [{offset}, {}) exceeds shard size {}",
                offset + len,
                shard.len()
            );
            metas.insert(
                name,
                ShardTensorMeta {
                    shape: view.shape().to_vec(),
                    dtype: view.dtype(),
                    offset,
                    len,
                },
            );
        }
        let metas = Rc::new(metas);
        self.shard_meta_cache
            .borrow_mut()
            .insert(idx, Rc::clone(&metas));
        self.log_startup_phase(
            "shard_deserialize",
            t0,
            format_args!(
                "idx={idx} tensors={} bytes={} path={}",
                metas.len(),
                shard.len(),
                path.display()
            ),
        );
        Ok(metas)
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
        let t0 = Instant::now();
        let tensor = self.borrow_raw_from_shard(idx, name)?;
        let owned = OwnedTensor {
            shape: tensor.shape.clone(),
            bytes: tensor.bytes().to_vec(),
            dtype: tensor.dtype,
        };
        self.log_startup_phase(
            "tensor.owned_copy",
            t0,
            format_args!(
                "name={name} idx={idx} bytes={} dtype={:?} shape={:?}",
                owned.bytes.len(),
                owned.dtype,
                owned.shape
            ),
        );
        Ok(owned)
    }

    /// Zero-copy shard read: the returned [`SharedTensor`] aliases the tensor's
    /// byte range inside the read-once shard cache (an `Rc` clone, no host
    /// memcpy). The stacked-expert loader slices ~1.5 GiB per MoE layer out of
    /// these bytes; the owned path (`load_raw_from_shard`) copied that whole
    /// range per tensor on top of the cache. (audit MOE-P2-1)
    fn borrow_raw_from_shard(&self, idx: usize, name: &str) -> Result<SharedTensor> {
        let shard = self.shard_bytes(idx)?;
        let path = &self.shards[idx];
        let metas = self.shard_tensor_metas(idx, &shard)?;
        let meta = metas
            .get(name)
            .with_context(|| format!("find tensor {name} in {}", path.display()))?;
        ensure!(
            meta.offset + meta.len <= shard.len(),
            "{name}: tensor byte range [{}, {}) exceeds shard size {}",
            meta.offset,
            meta.offset + meta.len,
            shard.len()
        );
        Ok(SharedTensor {
            shape: meta.shape.clone(),
            dtype: meta.dtype,
            shard,
            offset: meta.offset,
            len: meta.len,
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

fn float_elem_size(name: &str, dtype: Dtype) -> Result<usize> {
    match dtype {
        Dtype::F32 => Ok(4),
        Dtype::BF16 => Ok(2),
        other => bail!("{name}: expected BF16/F32 scale tensor, got {other:?}"),
    }
}

fn tensor_to_f32_vec(name: &str, tensor: &OwnedTensor, apply: ScaleApply) -> Result<Vec<f32>> {
    tensor_bytes_to_f32(name, tensor.dtype, &tensor.bytes, apply)
}

fn tensor_bytes_to_f32(
    name: &str,
    dtype: Dtype,
    bytes: &[u8],
    apply: ScaleApply,
) -> Result<Vec<f32>> {
    let mut values = match dtype {
        Dtype::F32 => {
            ensure!(
                bytes.len().is_multiple_of(4),
                "{name}: F32 scale byte length {} is not divisible by 4",
                bytes.len()
            );
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>()
        }
        Dtype::BF16 => {
            ensure!(
                bytes.len().is_multiple_of(2),
                "{name}: BF16 scale byte length {} is not divisible by 2",
                bytes.len()
            );
            bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect::<Vec<_>>()
        }
        other => bail!("{name}: expected BF16/F32 scale tensor, got {other:?}"),
    };
    if matches!(apply, ScaleApply::Divide) {
        for value in &mut values {
            ensure!(*value != 0.0, "{name}: divide-scale contains zero");
            *value = 1.0 / *value;
        }
    }
    Ok(values)
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

#[derive(serde::Deserialize)]
struct SafetensorHeaderTensor {
    dtype: Dtype,
    shape: Vec<usize>,
}

pub(crate) struct OwnedTensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) bytes: Vec<u8>,
    pub(crate) dtype: Dtype,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExpertQuantDispatchSignature {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) quant_scale_rows: usize,
    pub(crate) quant_scale_cols: usize,
    pub(crate) quant_block_m: usize,
    pub(crate) quant_block_k: usize,
    pub(crate) group_size: usize,
}

impl ExpertQuantDispatchSignature {
    fn from_matrix(matrix: &DeviceMatrix) -> Self {
        Self {
            rows: matrix.rows,
            cols: matrix.cols,
            quant_scale_rows: matrix.quant_scale_rows,
            quant_scale_cols: matrix.quant_scale_cols,
            quant_block_m: matrix.quant_block_m,
            quant_block_k: matrix.quant_block_k,
            group_size: matrix.group_size,
        }
    }
}

fn validate_expert_projection_dispatch_signature(
    name: &str,
    experts: &[DeviceMatrix],
    format: WeightFormat,
) -> Result<Option<ExpertQuantDispatchSignature>> {
    let first = experts
        .first()
        .ok_or_else(|| anyhow!("MoE layer has no local {name} experts"))?;
    let first_sig = ExpertQuantDispatchSignature::from_matrix(first);
    for (idx, expert) in experts.iter().enumerate() {
        ensure!(
            expert.weight_format() == format,
            "Qwen3.6 MoE {name} expert {idx} format {} != {format}",
            expert.weight_format()
        );
        if format.is_quantized() {
            let sig = ExpertQuantDispatchSignature::from_matrix(expert);
            ensure!(
                sig == first_sig,
                "Qwen3.6 MoE {name} expert {idx} quant dispatch signature {sig:?} != {first_sig:?}"
            );
        }
    }
    Ok(format.is_quantized().then_some(first_sig))
}

fn routed_expert_weight_format(
    gate: &[DeviceMatrix],
    up: &[DeviceMatrix],
    down: &[DeviceMatrix],
) -> Result<(
    WeightFormat,
    Option<ExpertQuantDispatchSignature>,
    Option<ExpertQuantDispatchSignature>,
)> {
    let first = gate
        .first()
        .ok_or_else(|| anyhow!("MoE layer has no local gate experts"))?
        .weight_format();
    ensure!(
        matches!(
            first,
            WeightFormat::DenseBf16
                | WeightFormat::Fp8BlockScaled
                | WeightFormat::Fp8PerShard
                | WeightFormat::Fp4E2M1Group
        ),
        "Qwen3.6 MoE routed expert format {first} is not supported"
    );
    let gate_sig = validate_expert_projection_dispatch_signature("gate", gate, first)?;
    let up_sig = validate_expert_projection_dispatch_signature("up", up, first)?;
    let down_sig = validate_expert_projection_dispatch_signature("down", down, first)?;
    if let (Some(gate_sig), Some(up_sig)) = (gate_sig, up_sig) {
        ensure!(
            gate_sig == up_sig,
            "Qwen3.6 MoE gate/up quant dispatch signature mismatch: gate={gate_sig:?} up={up_sig:?}"
        );
    }
    Ok((first, gate_sig, down_sig))
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
    pub(crate) expert_weight_format: WeightFormat,
    pub(crate) gate_up_quant_signature: Option<ExpertQuantDispatchSignature>,
    pub(crate) down_quant_signature: Option<ExpertQuantDispatchSignature>,
    pub(crate) gate_ptrs: CudaSlice<u64>,
    pub(crate) up_ptrs: CudaSlice<u64>,
    pub(crate) down_ptrs: CudaSlice<u64>,
    pub(crate) gate_scale_ptrs: Option<CudaSlice<u64>>,
    pub(crate) up_scale_ptrs: Option<CudaSlice<u64>>,
    pub(crate) down_scale_ptrs: Option<CudaSlice<u64>>,
    pub(crate) gate_global_ptrs: Option<CudaSlice<u64>>,
    pub(crate) up_global_ptrs: Option<CudaSlice<u64>>,
    pub(crate) down_global_ptrs: Option<CudaSlice<u64>>,
    /// DeepGEMM grouped-B caches (`[groups, n, k]` contiguous row-major BF16,
    /// this rank's EP experts only). `Some` iff `ARLE_QWEN35_DEEPGEMM=1` at
    /// load; the default load path is byte-identical (fields stay `None`).
    pub(crate) gate_grouped: Option<MoeExpertGroup>,
    pub(crate) up_grouped: Option<MoeExpertGroup>,
    pub(crate) down_grouped: Option<MoeExpertGroup>,
    /// DeepGEMM FP8 grouped-B cache for quantized routed experts. `w13`
    /// fuses gate rows followed by up rows per expert, so the DeepGEMM
    /// prefill lane can run one FP8 GEMM then SwiGLU+requantize.
    pub(crate) w13_fp8_grouped: Option<MoeFp8ExpertGroup>,
    pub(crate) down_fp8_grouped: Option<MoeFp8ExpertGroup>,
    pub(crate) router_gate: DeviceMatrix,
    pub(crate) shared_gate: DeviceMatrix,
    pub(crate) shared_up: DeviceMatrix,
    pub(crate) shared_down: DeviceMatrix,
    pub(crate) shared_gate_router: DeviceMatrix,
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

/// One contiguous `[groups, rows, cols]` row-major FP8 expert-weight buffer
/// with DeepGEMM-compatible FP32 `[groups, rows/128, cols/128]` block scales.
pub(crate) struct MoeFp8ExpertGroup {
    pub(crate) weight: CudaSlice<u8>,
    pub(crate) scales: CudaSlice<f32>,
    pub(crate) groups: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) scale_rows: usize,
    pub(crate) scale_cols: usize,
}

impl MoeFp8ExpertGroup {
    fn from_host(
        ctx: &DeviceContext,
        weight: &[u8],
        scales: &[f32],
        groups: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        ensure!(groups > 0, "FP8 MoE expert group: groups must be non-zero");
        ensure!(
            rows.is_multiple_of(128) && cols.is_multiple_of(128),
            "FP8 MoE DeepGEMM group needs rows/cols 128-aligned, got {rows}x{cols}"
        );
        let scale_rows = rows / 128;
        let scale_cols = cols / 128;
        ensure!(
            weight.len() == groups * rows * cols,
            "FP8 MoE grouped host weight bytes {} != expected {}",
            weight.len(),
            groups * rows * cols
        );
        ensure!(
            scales.len() == groups * scale_rows * scale_cols,
            "FP8 MoE grouped host scale values {} != expected {}",
            scales.len(),
            groups * scale_rows * scale_cols
        );
        Ok(Self {
            weight: ctx
                .stream
                .clone_htod(weight)
                .map_err(|e| anyhow!("FP8 MoE grouped weight H2D failed: {e}"))?,
            scales: ctx
                .stream
                .clone_htod(scales)
                .map_err(|e| anyhow!("FP8 MoE grouped scales H2D failed: {e}"))?,
            groups,
            rows,
            cols,
            scale_rows,
            scale_cols,
        })
    }

    fn concat(
        ctx: &DeviceContext,
        experts: &[DeviceMatrix],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        let groups = experts.len();
        ensure!(groups > 0, "FP8 MoE expert group concat: no local experts");
        let mut group = Self::empty(ctx, groups, rows, cols)?;
        for (g, expert) in experts.iter().enumerate() {
            group.copy_matrix_rows(ctx, g, 0, expert)?;
        }
        Ok(group)
    }

    fn concat_pair_rows(
        ctx: &DeviceContext,
        first: &[DeviceMatrix],
        second: &[DeviceMatrix],
        rows_each: usize,
        cols: usize,
    ) -> Result<Self> {
        ensure!(
            first.len() == second.len() && !first.is_empty(),
            "FP8 MoE fused group needs matching non-empty gate/up experts"
        );
        ensure!(
            rows_each.is_multiple_of(128),
            "FP8 MoE fused group first half rows must be 128-aligned, got {rows_each}"
        );
        let mut group = Self::empty(ctx, first.len(), rows_each * 2, cols)?;
        for (g, (a, b)) in first.iter().zip(second.iter()).enumerate() {
            group.copy_matrix_rows(ctx, g, 0, a)?;
            group.copy_matrix_rows(ctx, g, rows_each, b)?;
        }
        Ok(group)
    }

    fn empty(ctx: &DeviceContext, groups: usize, rows: usize, cols: usize) -> Result<Self> {
        ensure!(groups > 0, "FP8 MoE expert group: groups must be non-zero");
        ensure!(
            rows.is_multiple_of(128) && cols.is_multiple_of(128),
            "FP8 MoE DeepGEMM group needs rows/cols 128-aligned, got {rows}x{cols}"
        );
        let scale_rows = rows.div_ceil(128);
        let scale_cols = cols.div_ceil(128);
        Ok(Self {
            weight: ctx
                .stream
                .alloc_zeros::<u8>(groups * rows * cols)
                .map_err(|e| anyhow!("FP8 MoE grouped weight alloc failed: {e}"))?,
            scales: ctx
                .stream
                .alloc_zeros::<f32>(groups * scale_rows * scale_cols)
                .map_err(|e| anyhow!("FP8 MoE grouped scale alloc failed: {e}"))?,
            groups,
            rows,
            cols,
            scale_rows,
            scale_cols,
        })
    }

    fn copy_matrix_rows(
        &mut self,
        ctx: &DeviceContext,
        group: usize,
        row_offset: usize,
        matrix: &DeviceMatrix,
    ) -> Result<()> {
        ensure!(
            group < self.groups,
            "FP8 MoE group index {group} outside groups {}",
            self.groups
        );
        ensure!(
            matrix.weight_format() == WeightFormat::Fp8BlockScaled,
            "FP8 MoE grouped cache needs FP8 block-scaled experts, got {}",
            matrix.weight_format()
        );
        ensure!(
            matrix.rows + row_offset <= self.rows && matrix.cols == self.cols,
            "FP8 MoE grouped cache shape mismatch: matrix {}x{} at row_offset {} into group {}x{}",
            matrix.rows,
            matrix.cols,
            row_offset,
            self.rows,
            self.cols
        );
        ensure!(
            row_offset.is_multiple_of(128)
                && matrix.quant_block_m == 128
                && matrix.quant_block_k == 128,
            "FP8 MoE grouped cache needs 128x128 block metadata, row_offset={} block={}x{}",
            row_offset,
            matrix.quant_block_m,
            matrix.quant_block_k
        );
        let matrix_scale_rows = matrix.rows.div_ceil(128);
        let matrix_scale_cols = matrix.cols.div_ceil(128);
        ensure!(
            matrix.quant_scale_rows == matrix_scale_rows
                && matrix.quant_scale_cols == matrix_scale_cols
                && matrix_scale_cols == self.scale_cols,
            "FP8 MoE grouped cache scale shape {}x{} != expected {}x{}",
            matrix.quant_scale_rows,
            matrix.quant_scale_cols,
            matrix_scale_rows,
            self.scale_cols
        );
        let qweight = matrix
            .qweight_u8
            .as_ref()
            .ok_or_else(|| anyhow!("FP8 MoE grouped cache source missing weight bytes"))?;
        let scales = matrix
            .scale_f32
            .as_ref()
            .ok_or_else(|| anyhow!("FP8 MoE grouped cache source missing f32 scales"))?;
        ensure!(
            qweight.len() == matrix.rows * matrix.cols
                && scales.len() == matrix_scale_rows * self.scale_cols,
            "FP8 MoE grouped cache source lengths mismatch: weight={} scale={}",
            qweight.len(),
            scales.len()
        );

        {
            let src = qweight.slice(0..qweight.len());
            let group_weight_base = group * self.rows * self.cols;
            let start = group_weight_base + row_offset * self.cols;
            let mut dst = self.weight.slice_mut(start..start + qweight.len());
            ctx.stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| anyhow!("FP8 MoE grouped weight D2D failed: {e}"))?;
        }
        {
            let src = scales.slice(0..scales.len());
            let group_scale_base = group * self.scale_rows * self.scale_cols;
            let start = group_scale_base + (row_offset / 128) * self.scale_cols;
            let mut dst = self.scales.slice_mut(start..start + scales.len());
            ctx.stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| anyhow!("FP8 MoE grouped scale D2D failed: {e}"))?;
        }
        Ok(())
    }

    fn qweight_ptr_table(&self, ctx: &DeviceContext, row_offset: usize) -> Result<CudaSlice<u64>> {
        ensure!(
            row_offset < self.rows && row_offset.is_multiple_of(128),
            "FP8 MoE qweight ptr row offset {row_offset} invalid for rows {}",
            self.rows
        );
        let (base, _guard) = self.weight.device_ptr(&ctx.stream);
        let group_stride = self.rows * self.cols;
        let row_offset_elems = row_offset * self.cols;
        let host: Vec<u64> = (0..self.groups)
            .map(|g| base + (g * group_stride + row_offset_elems) as u64)
            .collect();
        ctx.stream
            .clone_htod(&host)
            .map_err(|e| anyhow!("FP8 MoE qweight ptr table H2D failed: {e}"))
    }

    fn scale_ptr_table(&self, ctx: &DeviceContext, row_offset: usize) -> Result<CudaSlice<u64>> {
        ensure!(
            row_offset < self.rows && row_offset.is_multiple_of(128),
            "FP8 MoE scale ptr row offset {row_offset} invalid for rows {}",
            self.rows
        );
        let (base, _guard) = self.scales.device_ptr(&ctx.stream);
        let group_stride = self.scale_rows * self.scale_cols;
        let row_offset_elems = (row_offset / 128) * self.scale_cols;
        let elem_size = std::mem::size_of::<f32>() as u64;
        let host: Vec<u64> = (0..self.groups)
            .map(|g| base + ((g * group_stride + row_offset_elems) as u64 * elem_size))
            .collect();
        ctx.stream
            .clone_htod(&host)
            .map_err(|e| anyhow!("FP8 MoE scale ptr table H2D failed: {e}"))
    }
}
