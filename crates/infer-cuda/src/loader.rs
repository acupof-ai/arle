//! Cold path: safetensors loading, paging metadata, and config validation.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs;
use std::io::Read;
use std::ops::Deref;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::rc::Rc;
use std::slice;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail, ensure};
use cuda_kernels::KVFormat;
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, PagedKVPool};
use cuda_kernels::tensor::{HostMatrixSnapshot, WeightFormat};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use deepseek_spec::Shard;
use infer_topo::{ShardingSpec, TpConfig};
use qwen3_spec::Qwen3Config;
use safetensors::{SafeTensors, tensor::Dtype};

use crate::model::{Attention, CudaModel, Mlp, TransformerBlock};
use crate::ops::{precompute_rope, upload_i32};
use crate::quant_format::{
    QuantFormat, QuantManifest, QuantTensorView, ScaleApply, TensorHeader, detect_quant_format,
    read_quant_manifest, reject_dsv4_e8m0_scale_abi,
};

const DEFAULT_ROPE_CACHE_LEN: usize = 32_768;
const DEFAULT_SHARD_CACHE_BYTES: usize = 8 * 1024 * 1024 * 1024;
const PREFETCH_CACHE_HEADROOM: usize = 64 * 1024 * 1024 * 1024;

fn available_ram_bytes() -> Option<usize> {
    fs::read_to_string("/proc/meminfo")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("MemAvailable:")?
                .split_whitespace()
                .next()?
                .parse::<usize>()
                .ok()
                .map(|kb| kb.saturating_mul(1024))
        })
}

impl CudaModel {
    pub(crate) fn from_safetensors(model_path: &Path) -> Result<Self> {
        let tp = build_tp_runtime(false)?;
        Self::from_safetensors_with_tp(model_path, tp)
    }

    pub(crate) fn from_safetensors_with_tp(
        model_path: &Path,
        tp: crate::tp::TpRuntime,
    ) -> Result<Self> {
        let config = Qwen3Config::from_json_file(model_path.join("config.json"))
            .with_context(|| format!("load Qwen3 config from {}", model_path.display()))?;
        validate_clean_bf16_config(&config)?;

        let tp_cfg = *tp.config();
        // kv8 kernels and the all-reduce require a uniform per-rank attention shape.
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

        // lm_head / embed_tokens stay replicated — sharding them would need an
        // all-gather
        // of logits.
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
                    // Whole-head boundaries keep o_proj's input shard and head count in
                    // agreement; K/V index by replica-aware KV block, not by rank.
                    let kv_block =
                        infer_topo::kv_load_block_index(config.num_key_value_heads, &tp_cfg)?;
                    (
                        loader.load_qkv_head_sharded(
                            &ctx,
                            &names.q_proj,
                            local_q_heads,
                            head_dim,
                            tp_cfg.rank,
                        )?,
                        loader.load_qkv_head_sharded(
                            &ctx,
                            &names.k_proj,
                            local_kv_heads,
                            head_dim,
                            kv_block,
                        )?,
                        loader.load_qkv_head_sharded(
                            &ctx,
                            &names.v_proj,
                            local_kv_heads,
                            head_dim,
                            kv_block,
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

/// Multi-rank `nccl` builds read the NCCL `unique_id` from `INFER_NCCL_UNIQUE_ID`;
/// otherwise the no-op single runtime.
pub(crate) fn build_tp_runtime(
    #[cfg_attr(not(feature = "nccl"), allow(unused_variables))] pin_numa: bool,
) -> Result<crate::tp::TpRuntime> {
    #[cfg(feature = "nccl")]
    {
        let cfg = crate::tp::resolve_tp_config_from_env().map_err(|e| anyhow!("{e}"))?;
        if !cfg.is_single() {
            let ordinal = cuda_kernels::tensor::parse_device_ordinal_from_env()?;
            if pin_numa {
                crate::numa_pin::pin_to_gpu_numa(ordinal as usize, cfg.world_size as usize);
            }
            cudarc::runtime::result::device::set(ordinal as i32)
                .map_err(|e| anyhow!("cudaSetDevice({ordinal}) before NCCL init failed: {e}"))?;
            let unique_id = nccl_unique_id_from_env()?;
            return crate::tp::TpRuntime::from_env_with_nccl(unique_id);
        }
    }
    crate::tp::TpRuntime::from_env().map_err(|e| anyhow!("{e}"))
}

/// Decode the NCCL `unique_id` from `INFER_NCCL_UNIQUE_ID` (256 hex chars = 128 bytes).
#[cfg(feature = "nccl")]
pub fn nccl_unique_id_from_env() -> Result<cuda_kernels::ffi::nccl::ncclUniqueId> {
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

/// Mint a fresh NCCL `unique_id` as 256 hex chars for `INFER_NCCL_UNIQUE_ID` — every
/// rank
/// must inherit the SAME handle. Host call: no CUDA context required.
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
    // Qwen3 decouples head_dim from hidden_size/num_heads, so `hidden_size ==
    // heads*head_dim`
    // is NOT an invariant; only GQA divisibility is.
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
    /// Host mirrors of `q_indptr` / `kv_indptr` — the prep kernel is single-row, so it
    /// is
    /// launched per row at these offsets.
    pub(crate) q_offsets: Vec<usize>,
    pub(crate) page_offsets: Vec<usize>,
    /// Host mirror of each row's total KV length (prefix + this forward's new
    /// tokens).
    pub(crate) kv_lens: Vec<usize>,
    /// Device `[batch]` copy of `kv_lens` — FA3's `seqused_k`.
    pub(crate) kv_lens_dev: CudaSlice<i32>,
    /// RECTANGULAR page table `[batch, page_table_stride]`, each row padded with its
    /// own last
    /// page: FA3 strides rows by a scalar, so the ragged `kv_indices` cannot be shared;
    /// rows
    /// never read past `kv_lens`.
    pub(crate) page_table_rect: CudaSlice<i32>,
    /// Row stride of `page_table_rect` — the longest row's page count.
    pub(crate) page_table_stride: usize,
    /// Longest row's new-token count — the kernel's `max_qlen`.
    pub(crate) seq_len: usize,
    /// Sum of every row's new-token count — the kernel's `total_q_tokens`.
    pub(crate) total_q: usize,
    pub(crate) num_pages: usize,
    pub(crate) batch: usize,
    /// Tokens already in the pool before this forward.
    pub(crate) start_pos: usize,
    /// Global pool token rows for the NEW tokens [start_pos, start_pos+seq_len).
    /// Quant formats only; None for BF16.
    pub(crate) new_token_rows: Option<CudaSlice<i32>>,
    /// Global pool token rows for the prefix [0, start_pos). Quant + start_pos>0 only.
    pub(crate) prefix_token_rows: Option<CudaSlice<i32>>,
    /// Packed fused-decode metadata `[page_indptr(b+1) | last_page_len(b)]`
    /// from `build_quantized_decode_indptr`. Quant formats only.
    pub(crate) quant_decode_meta: Option<CudaSlice<i32>>,
    /// When set, the FA3 launch uses this as `seqlen_k` instead of the step's true max
    /// KV
    /// length — required under CUDA graph capture, where the true length is only on
    /// device.
    pub(crate) seqlen_k_capture: Option<usize>,
}

impl PageMeta {
    /// Under CUDA graph capture the host `kv_lens` can be stale, so `seqlen_k_capture`
    /// wins.
    pub(crate) fn max_kv_len(&self) -> usize {
        self.seqlen_k_capture
            .unwrap_or_else(|| self.kv_lens.iter().copied().max().unwrap_or(0))
    }

    /// Ragged page table over `rows` of `(slot, start_pos, len)`.
    pub(crate) fn for_rows(
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        rows: &[(usize, usize, usize)],
    ) -> Result<Self> {
        ensure!(!rows.is_empty(), "page table needs at least one row");
        let batch = rows.len();
        let mut q_indptr = Vec::with_capacity(batch + 1);
        let mut kv_indptr = Vec::with_capacity(batch + 1);
        let mut kv_indices = Vec::new();
        let mut last_page_lens = Vec::with_capacity(batch);
        let mut start_positions = Vec::with_capacity(batch);
        let mut kv_lens = Vec::with_capacity(batch);
        let mut positions = Vec::with_capacity(batch);
        q_indptr.push(0);
        kv_indptr.push(0);
        let (mut total_q, mut total_pages, mut max_len) = (0usize, 0usize, 0usize);
        for &(slot, start_pos, len) in rows {
            ensure!(
                len > 0,
                "page-table row for slot {slot} has no query tokens"
            );
            let total_len = start_pos + len;
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
            kv_indices.extend(pages[..num_pages].iter().map(|&page| page as i32));
            let last_page_len = total_len % pool.page_size;
            last_page_lens.push(if last_page_len == 0 {
                pool.page_size as i32
            } else {
                last_page_len as i32
            });
            start_positions.push(start_pos as i32);
            positions.push((total_len - 1) as i32);
            kv_lens.push(total_len);
            total_q += len;
            total_pages += num_pages;
            max_len = max_len.max(len);
            q_indptr.push(total_q as i32);
            kv_indptr.push(total_pages as i32);
        }
        // Quant formats address the pool by global token row and consume packed decode
        // metadata; BF16 carries None.
        let quant = matches!(pool.format, KVFormat::INT8 | KVFormat::FP8E4M3);
        let (new_token_rows, prefix_token_rows, quant_decode_meta) = if quant {
            // Row lists concatenate in row order — the quantize kernel indexes them by
            // flat
            // Q-token position.
            let mut new_rows = Vec::new();
            for &(slot, start_pos, len) in rows {
                new_rows.extend(
                    pool.token_rows_for_range(slot, start_pos, len)
                        .into_iter()
                        .map(|row| row as i32),
                );
            }
            // The prefix refill feeds the single-row prefill path only; a batched
            // decode reads
            // the already-quantized prefix straight from the pool planes.
            let prefix_rows = if batch == 1 && rows[0].1 > 0 {
                let (slot, start_pos, _) = rows[0];
                let rows = pool
                    .token_rows_for_range(slot, 0, start_pos)
                    .into_iter()
                    .map(|row| row as i32)
                    .collect::<Vec<_>>();
                Some(upload_i32(ctx, &rows)?)
            } else {
                None
            };
            let slots: Vec<usize> = rows.iter().map(|&(slot, _, _)| slot).collect();
            (
                Some(upload_i32(ctx, &new_rows)?),
                prefix_rows,
                Some(upload_i32(
                    ctx,
                    &pool.build_quantized_decode_indptr(&slots),
                )?),
            )
        } else {
            (None, None, None)
        };
        // Pad each row with its own last page so a stray read stays inside the
        // request's KV.
        let stride = (1..=batch)
            .map(|r| kv_indptr[r] as usize - kv_indptr[r - 1] as usize)
            .max()
            .unwrap_or(0);
        let mut rect = Vec::with_capacity(batch * stride);
        for r in 0..batch {
            let (lo, hi) = (kv_indptr[r] as usize, kv_indptr[r + 1] as usize);
            rect.extend_from_slice(&kv_indices[lo..hi]);
            let pad = kv_indices[hi - 1];
            rect.resize(rect.len() + stride - (hi - lo), pad);
        }
        Ok(Self {
            q_indptr: upload_i32(ctx, &q_indptr)?,
            kv_indptr: upload_i32(ctx, &kv_indptr)?,
            kv_indices: upload_i32(ctx, &kv_indices)?,
            kv_lens_dev: upload_i32(ctx, &kv_lens.iter().map(|&l| l as i32).collect::<Vec<_>>())?,
            page_table_rect: upload_i32(ctx, &rect)?,
            page_table_stride: stride,
            kv_last_page_len: upload_i32(ctx, &last_page_lens)?,
            page_table_offsets: upload_i32(ctx, &[0])?,
            start_positions: upload_i32(ctx, &start_positions)?,
            positions: upload_i32(ctx, &positions)?,
            q_offsets: q_indptr.iter().map(|&q| q as usize).collect(),
            page_offsets: kv_indptr.iter().map(|&p| p as usize).collect(),
            kv_lens,
            seq_len: max_len,
            total_q,
            num_pages: total_pages,
            batch,
            start_pos: rows[0].1,
            new_token_rows,
            prefix_token_rows,
            quant_decode_meta,
            seqlen_k_capture: None,
        })
    }

    pub(crate) fn for_slot(
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<Self> {
        Self::for_rows(ctx, pool, &[(slot, start_pos, seq_len)])
    }

    /// Fixed-capacity single-slot decode metadata whose device buffers never move:
    /// [`Self::refresh_decode`] rewrites the CONTENTS each step, so a captured CUDA
    /// graph
    /// keeps reading the same addresses. `seqlen_k_capture` is pinned to the capacity
    /// so the
    /// FA3 scheduling ceiling is capture-stable. BF16 only.
    pub(crate) fn persistent_decode(
        ctx: &DeviceContext,
        page_size: usize,
        capacity_pages: usize,
    ) -> Result<Self> {
        let cap = capacity_pages.max(1);
        Ok(Self {
            q_indptr: upload_i32(ctx, &[0, 1])?,
            kv_indptr: upload_i32(ctx, &[0, 0])?,
            kv_indices: upload_i32(ctx, &vec![0i32; cap])?,
            kv_lens_dev: upload_i32(ctx, &[0])?,
            page_table_rect: upload_i32(ctx, &vec![0i32; cap])?,
            page_table_stride: cap,
            kv_last_page_len: upload_i32(ctx, &[0])?,
            page_table_offsets: upload_i32(ctx, &[0])?,
            start_positions: upload_i32(ctx, &[0])?,
            positions: upload_i32(ctx, &[0])?,
            q_offsets: vec![0, 1],
            page_offsets: vec![0, 0],
            kv_lens: vec![0],
            seq_len: 1,
            total_q: 1,
            num_pages: 0,
            batch: 1,
            start_pos: 0,
            new_token_rows: None,
            prefix_token_rows: None,
            quant_decode_meta: None,
            seqlen_k_capture: Some(cap * page_size),
        })
    }

    /// Rewrite a [`Self::persistent_decode`] meta for one decode step — contents only,
    /// no
    /// reallocation. Must run OUTSIDE any graph capture/replay.
    pub(crate) fn refresh_decode(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot: usize,
        start_pos: usize,
    ) -> Result<()> {
        ensure!(
            pool.format == KVFormat::BF16,
            "persistent decode page table is BF16-only, got {:?}",
            pool.format
        );
        let total_len = start_pos + 1;
        ensure!(
            pool.seq_len(slot) == total_len,
            "persistent decode refresh: pool seq_len {} != total_len {total_len} for slot {slot}",
            pool.seq_len(slot),
        );
        let num_pages = total_len.div_ceil(pool.page_size);
        let cap = self.page_table_stride;
        ensure!(
            num_pages <= cap,
            "slot {slot} needs {num_pages} pages, persistent capacity is {cap}"
        );
        let pages = pool.page_indices(slot);
        ensure!(
            pages.len() >= num_pages,
            "slot {slot} has {} pages, expected at least {num_pages}",
            pages.len()
        );
        let pages_i32: Vec<i32> = pages[..num_pages].iter().map(|&p| p as i32).collect();
        let stream = &ctx.stream;
        stream
            .memcpy_htod(&pages_i32, &mut self.kv_indices.slice_mut(0..num_pages))
            .map_err(|e| anyhow!("refresh kv_indices: {e}"))?;
        stream
            .memcpy_htod(
                &pages_i32,
                &mut self.page_table_rect.slice_mut(0..num_pages),
            )
            .map_err(|e| anyhow!("refresh page_table_rect: {e}"))?;
        let last_page_len = match total_len % pool.page_size {
            0 => pool.page_size as i32,
            l => l as i32,
        };
        stream
            .memcpy_htod(&[0, num_pages as i32], &mut self.kv_indptr.slice_mut(0..2))
            .map_err(|e| anyhow!("refresh kv_indptr: {e}"))?;
        stream
            .memcpy_htod(&[last_page_len], &mut self.kv_last_page_len.slice_mut(0..1))
            .map_err(|e| anyhow!("refresh kv_last_page_len: {e}"))?;
        stream
            .memcpy_htod(
                &[start_pos as i32],
                &mut self.start_positions.slice_mut(0..1),
            )
            .map_err(|e| anyhow!("refresh start_positions: {e}"))?;
        stream
            .memcpy_htod(
                &[(total_len - 1) as i32],
                &mut self.positions.slice_mut(0..1),
            )
            .map_err(|e| anyhow!("refresh positions: {e}"))?;
        stream
            .memcpy_htod(&[total_len as i32], &mut self.kv_lens_dev.slice_mut(0..1))
            .map_err(|e| anyhow!("refresh kv_lens_dev: {e}"))?;
        self.page_offsets = vec![0, num_pages];
        self.kv_lens = vec![total_len];
        self.num_pages = num_pages;
        self.start_pos = start_pos;
        Ok(())
    }

    /// Session KV-recall decode page table: build the metadata over a SELECTED page
    /// subset
    /// (`recall_pages`, ascending, ending with the current local page) instead of the
    /// slot's
    /// full page list. Correct because RoPE is baked into the cached K at write time
    /// and the
    /// query's RoPE position is independent of which KV pages are attended.
    /// Decode-only,
    /// BF16 only.
    pub(crate) fn for_recall_decode(
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        total_len: usize,
        recall_pages: &[u32],
    ) -> Result<Self> {
        ensure!(
            pool.format == KVFormat::BF16,
            "session KV-recall decode page table is BF16-only, got {:?}",
            pool.format
        );
        ensure!(
            !recall_pages.is_empty(),
            "recall decode page table needs at least one selected page"
        );
        let num_pages = recall_pages.len();
        let last_page_len = total_len % pool.page_size;
        let last_page_len = if last_page_len == 0 {
            pool.page_size
        } else {
            last_page_len
        };
        let page_ids = recall_pages.iter().map(|&p| p as i32).collect::<Vec<_>>();
        Ok(Self {
            q_indptr: upload_i32(ctx, &[0, 1])?,
            kv_indptr: upload_i32(ctx, &[0, num_pages as i32])?,
            kv_indices: upload_i32(ctx, &page_ids)?,
            kv_lens_dev: upload_i32(ctx, &[total_len as i32])?,
            page_table_rect: upload_i32(ctx, &page_ids)?,
            page_table_stride: num_pages,
            kv_last_page_len: upload_i32(ctx, &[last_page_len as i32])?,
            page_table_offsets: upload_i32(ctx, &[0])?,
            start_positions: upload_i32(ctx, &[(total_len - 1) as i32])?,
            positions: upload_i32(ctx, &[(total_len - 1) as i32])?,
            q_offsets: vec![0, 1],
            page_offsets: vec![0, num_pages],
            kv_lens: vec![total_len],
            seq_len: 1,
            total_q: 1,
            num_pages,
            batch: 1,
            start_pos: total_len - 1,
            new_token_rows: None,
            prefix_token_rows: None,
            quant_decode_meta: None,
            seqlen_k_capture: None,
        })
    }

    /// Batched decode page table. `total_len` INCLUDES this step's just-appended token.
    pub(crate) fn for_decode_batch(
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        rows: &[(usize, usize)],
    ) -> Result<Self> {
        let rows = rows
            .iter()
            .map(|&(slot, total_len)| {
                ensure!(total_len > 0, "decode row slot {slot} has empty cache");
                Ok((slot, total_len - 1, 1))
            })
            .collect::<Result<Vec<_>>>()?;
        Self::for_rows(ctx, pool, &rows)
    }
}

pub(crate) struct SafetensorLoader {
    base: PathBuf,
    shards: Vec<PathBuf>,
    weight_map: HashMap<String, usize>,
    tensor_headers: std::cell::RefCell<Option<Rc<BTreeMap<String, TensorHeader>>>>,
    quant_manifest: Option<QuantManifest>,
    /// Bounded mmap shard cache: tensors that alternate across two shards would
    /// otherwise
    /// re-open multi-GiB files on every touch. `Rc` lets [`SharedTensor`] alias a byte
    /// range
    /// zero-copy while the entry itself stays evictable.
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

/// Transpose each group's `[rows, cols]` row-major block-scale slab to `[cols, rows]`
/// in
/// place: maps the checkpoint's K-contiguous `weight_scale_inv` to the CUTLASS sm_120
/// N-contiguous SFB layout.
fn transpose_group_block_scales(scales: &mut [f32], groups: usize, rows: usize, cols: usize) {
    let per = rows * cols;
    debug_assert_eq!(scales.len(), groups * per);
    let mut tmp = vec![0f32; per];
    for g in 0..groups {
        let block = &mut scales[g * per..(g + 1) * per];
        for r in 0..rows {
            for c in 0..cols {
                tmp[c * rows + r] = block[r * cols + c];
            }
        }
        block.copy_from_slice(&tmp);
    }
}

struct ShardByteCache {
    entries: HashMap<usize, Rc<ShardBytes>>,
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

    fn get(&mut self, idx: usize) -> Option<Rc<ShardBytes>> {
        let bytes = Rc::clone(self.entries.get(&idx)?);
        self.touch(idx);
        Some(bytes)
    }

    fn insert(&mut self, idx: usize, bytes: Rc<ShardBytes>) -> Vec<(usize, usize)> {
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

struct ShardBytes {
    mmap: MmapShard,
}

impl ShardBytes {
    fn map(path: &Path) -> Result<Self> {
        Ok(Self {
            mmap: MmapShard::map(path)?,
        })
    }
}

impl Deref for ShardBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.mmap.as_slice()
    }
}

struct MmapShard {
    ptr: NonNull<u8>,
    len: usize,
}

impl MmapShard {
    fn map(path: &Path) -> Result<Self> {
        let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let len: usize = file
            .metadata()
            .with_context(|| format!("stat {}", path.display()))?
            .len()
            .try_into()
            .with_context(|| format!("{} is too large to mmap on this host", path.display()))?;
        ensure!(len > 0, "{} is empty", path.display());
        ensure!(
            len <= isize::MAX as usize,
            "{} length {len} exceeds mmap addressable range",
            path.display()
        );
        // SAFETY: fd is a live read-only file; 0 < len <= isize::MAX checked above;
        // MAP_FAILED handled below.
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("mmap {}", path.display()));
        }
        Ok(Self {
            ptr: NonNull::new(mapped.cast::<u8>()).expect("mmap returned null but not MAP_FAILED"),
            len,
        })
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr/len are the live PROT_READ mapping owned by self.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for MmapShard {
    fn drop(&mut self) {
        // SAFETY: ptr/len are the exact mmap result; unmapped once, in Drop.
        let rc = unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.len) };
        if rc != 0 {
            log::warn!(
                "munmap failed for safetensors shard mapping: {}",
                std::io::Error::last_os_error()
            );
        }
    }
}

#[derive(Clone, Debug)]
struct ShardTensorMeta {
    shape: Vec<usize>,
    dtype: Dtype,
    offset: usize,
    len: usize,
}

struct ShardedBytesCow<'a> {
    bytes: Cow<'a, [u8]>,
    rows: usize,
    cols: usize,
}

fn shard_cache_bytes_limit() -> usize {
    crate::runtime_flags::shard_cache_bytes().unwrap_or(DEFAULT_SHARD_CACHE_BYTES)
}

impl SafetensorLoader {
    /// Rank zero warms the page cache by advising the kernel to read ahead
    /// all checkpoint shards (`posix_fadvise(WILLNEED)`). Nonblocking —
    /// the kernel reads asynchronously and pages stay reclaimable under
    /// memory pressure (unlike the old read-to-buffer approach which pinned
    /// every page). Other ranks wait on the broadcast until rank 0 finishes.
    pub(crate) fn prefetch_shards_rank0(
        &self,
        ctx: &DeviceContext,
        tp: &crate::tp::TpRuntime,
    ) -> Result<()> {
        let rank = tp.config().rank;
        let enabled = std::env::var_os("ARLE_LOADER_PREFETCH").is_none_or(|value| value != "0");
        let should_prefetch = enabled && rank == 0 && self.has_memory_for_prefetch();

        if should_prefetch {
            self.prefetch_all_shards();
        }

        let rank0_prefetched = tp.broadcast_rank0_i32(ctx, &[i32::from(should_prefetch)])?[0] != 0;
        if rank == 0 && enabled && !rank0_prefetched {
            log::info!("loader prefetch skipped: insufficient memory or disabled");
        }
        Ok(())
    }

    /// Check if the host has enough free memory to benefit from prefetch.
    /// With fadvise, pages are reclaimable, so this is a soft check to
    /// avoid thrashing when memory is already tight.
    fn has_memory_for_prefetch(&self) -> bool {
        let checkpoint: usize = self
            .shards
            .iter()
            .filter_map(|path| fs::metadata(path).ok().map(|m| m.len() as usize))
            .sum();
        if checkpoint == 0 {
            return false;
        }
        let available = available_ram_bytes().unwrap_or(0);
        available >= checkpoint.saturating_add(PREFETCH_CACHE_HEADROOM)
    }

    /// Advise the kernel to read ahead all shards. Each shard is mmapped and
    /// `madvise(MADV_WILLNEED)`d — this operates on the same page-cache pages
    /// the subsequent weight-load mmap reads, unlike `posix_fadvise` on a
    /// separate fd which on some kernels does not populate the mmap page cache
    /// (observed: rank-0 deadlocked in page faults after fadvise). Pages stay
    /// reclaimable under memory pressure.
    fn prefetch_all_shards(&self) {
        use std::os::fd::AsRawFd;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let t0 = Instant::now();
        let n = self.shards.len();
        if n == 0 {
            return;
        }
        let threads = std::thread::available_parallelism()
            .map_or(8, |p| p.get())
            .clamp(1, 16)
            .min(n);
        let shards = &self.shards;
        let next = AtomicUsize::new(0);
        let bytes = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        if let Ok(meta) = fs::metadata(&shards[i]) {
                            let len = meta.len() as usize;
                            if len == 0 {
                                continue;
                            }
                            if let Ok(file) = fs::File::open(&shards[i]) {
                                // SAFETY: mmap the whole shard read-only, then
                                // madvise WILLNEED to fault the pages into the
                                // page cache. The munmap on drop does not
                                // reclaim the cached pages.
                                #[cfg(target_os = "linux")]
                                unsafe {
                                    let ptr = libc::mmap(
                                        std::ptr::null_mut(),
                                        len,
                                        libc::PROT_READ,
                                        libc::MAP_PRIVATE,
                                        file.as_raw_fd(),
                                        0,
                                    );
                                    if ptr != libc::MAP_FAILED {
                                        libc::madvise(ptr, len, libc::MADV_WILLNEED);
                                        libc::munmap(ptr, len);
                                    }
                                }
                                bytes.fetch_add(len, Ordering::Relaxed);
                            }
                        }
                    }
                });
            }
        });
        let gb = bytes.load(Ordering::Relaxed) as f64 / 1e9;
        let secs = t0.elapsed().as_secs_f64().max(1e-6);
        log::info!(
            "loader prefetch (mmap+madvise): {gb:.1} GB across {n} shards in {secs:.1}s ({:.2} GB/s, {threads} threads)",
            gb / secs
        );
    }

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
            crate::executor::cuda_startup_log(
                "loader.new.index",
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
            crate::executor::cuda_startup_log(
                "loader.new.single",
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
        crate::executor::cuda_startup_log(
            "loader.new.scan",
            t0,
            format_args!(
                "shards={} quant_manifest={}",
                loader.shards.len(),
                loader.quant_manifest.is_some()
            ),
        );
        Ok(loader)
    }

    pub(crate) fn load_vec(&self, ctx: &DeviceContext, name: &str) -> Result<DeviceVec> {
        let tensor = self.borrow_bf16_tensor(name)?;
        ensure!(
            tensor.shape.len() == 1,
            "{name}: expected 1D BF16 tensor, got shape {:?}",
            tensor.shape
        );
        DeviceVec::from_safetensors(ctx, tensor.bytes())
            .with_context(|| format!("upload tensor {name}"))
    }

    /// Load a 1D vector that is BF16 or F32 on disk, normalized to BF16 so the
    /// recurrent
    /// kernel's bf16 input ABI holds.
    pub(crate) fn load_vec_any(&self, ctx: &DeviceContext, name: &str) -> Result<DeviceVec> {
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 1,
            "{name}: expected 1D tensor, got shape {:?}",
            tensor.shape
        );
        DeviceVec::from_safetensors(
            ctx,
            Self::tensor_bytes_to_bf16(name, tensor.dtype, tensor.bytes())?.as_ref(),
        )
        .with_context(|| format!("upload vec {name}"))
    }

    /// Load a 1D F32 tensor into a device `f32` slice — the recurrent + gated-RMSNorm
    /// kernels
    /// read these as `*const f32`. Accepts F32 (passthrough) or BF16 (widened to F32).
    pub(crate) fn load_f32_vec(&self, ctx: &DeviceContext, name: &str) -> Result<CudaSlice<f32>> {
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 1,
            "{name}: expected 1D tensor, got shape {:?}",
            tensor.shape
        );
        let host: Vec<f32> = match tensor.dtype {
            Dtype::F32 => tensor
                .bytes()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::BF16 => tensor
                .bytes()
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            other => bail!("{name}: expected F32/BF16 1D tensor, got {other:?}"),
        };
        ctx.stream
            .clone_htod(&host)
            .map_err(|e| anyhow!("upload f32 vec {name}: {e}"))
    }

    /// Load a Qwen3.5 depthwise conv1d weight (`[qkv_dim, 1, kernel]` BF16) as a flat
    /// `[qkv_dim*kernel]` [`DeviceVec`] — the conv1d kernel's channel-major ABI.
    pub(crate) fn load_conv1d_vec(&self, ctx: &DeviceContext, name: &str) -> Result<DeviceVec> {
        let tensor = self.borrow_bf16_tensor(name)?;
        ensure!(
            !tensor.shape.is_empty(),
            "{name}: expected conv1d tensor, got rank-0"
        );
        DeviceVec::from_safetensors(ctx, tensor.bytes())
            .with_context(|| format!("upload conv1d {name}"))
    }

    pub(crate) fn load_matrix(&self, ctx: &DeviceContext, name: &str) -> Result<DeviceMatrix> {
        let tensor = self.borrow_bf16_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D BF16 tensor, got shape {:?}",
            tensor.shape
        );
        DeviceMatrix::from_safetensors(ctx, tensor.bytes(), tensor.shape[0], tensor.shape[1])
            .with_context(|| format!("upload tensor {name}"))
    }

    /// Load a 2D BF16 weight sliced to this TP rank. Column kind (`q/k/v/gate/up`)
    /// slices
    /// rows; row kind (`o/down`) slices cols. Single-GPU is the identity slice.
    pub(crate) fn load_matrix_sharded(
        &self,
        ctx: &DeviceContext,
        name: &str,
        kind: infer_topo::ParallelLinearKind,
        tp: &infer_topo::TpConfig,
    ) -> Result<DeviceMatrix> {
        const BF16_ELEM_SIZE: usize = 2;
        let tensor = self.borrow_bf16_tensor(name)?;
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
                    tensor.bytes(),
                    rows,
                    cols,
                    BF16_ELEM_SIZE,
                    &spec,
                )?
            }
            infer_topo::ParallelLinearKind::Row => {
                let spec = infer_topo::row_shard(cols, tp);
                crate::shard_slice::shard_row_parallel(
                    tensor.bytes(),
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

    /// Load a head-aligned column-parallel Q/K/V projection for this TP rank. The split
    /// MUST
    /// land on whole-head boundaries; a plain `column_shard` on the raw output dim
    /// would
    /// split a head on the last rank.
    ///
    /// `head_dim` is the PER-HEAD ROW COUNT: the gated Qwen3.5/3.6 q_proj interleaves
    /// `[query; gate]` per head, so its per-head row block is `2 * head_dim`.
    ///
    /// `block_index` selects the head block. Q passes `tp.rank`; KV passes
    /// [`infer_topo::kv_load_block_index`], which makes ranks in a replica group load
    /// IDENTICAL K/V weights — the cache-identity invariant behind independent
    /// per-replica GQA.
    pub(crate) fn load_qkv_head_sharded(
        &self,
        ctx: &DeviceContext,
        name: &str,
        local_heads: usize,
        head_dim: usize,
        block_index: usize,
    ) -> Result<DeviceMatrix> {
        const BF16_ELEM_SIZE: usize = 2;
        let tensor = self.borrow_bf16_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D BF16 tensor, got shape {:?}",
            tensor.shape
        );
        let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
        let total_rows = rows;
        let local_rows = local_heads * head_dim;
        let offset = block_index * local_rows;
        ensure!(
            offset + local_rows <= total_rows,
            "{name}: head shard [{offset}, {}) exceeds rows {total_rows} \
             (local_heads={local_heads}, head_dim={head_dim}, block_index={block_index})",
            offset + local_rows,
        );
        let spec = infer_topo::ShardingSpec {
            offset,
            size: local_rows,
            total: total_rows,
        };
        let sharded = crate::shard_slice::shard_column_parallel(
            tensor.bytes(),
            rows,
            cols,
            BF16_ELEM_SIZE,
            &spec,
        )?;
        DeviceMatrix::from_safetensors(ctx, &sharded.bytes, sharded.rows, sharded.cols)
            .with_context(|| format!("upload head-sharded tensor {name}"))
    }

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

    /// Load two same-K projections as ONE row-fused matrix (`[a; b]` along output rows)
    /// so a
    /// single GEMM serves both. W8A16 pairs fuse before the Marlin repack; every other
    /// format
    /// loads normally and fuses on device.
    pub(crate) fn load_matrix_pair_fused(
        &self,
        ctx: &DeviceContext,
        name_a: &str,
        name_b: &str,
    ) -> Result<DeviceMatrix> {
        self.load_matrix_pair_fused_inner(ctx, name_a, name_b, None)
    }

    /// Column-sharded (TP) twin of [`Self::load_matrix_pair_fused`]: each half
    /// is column-sharded by its own rows, then the local shards fuse.
    pub(crate) fn load_matrix_pair_fused_column_sharded(
        &self,
        ctx: &DeviceContext,
        name_a: &str,
        name_b: &str,
        tp: &infer_topo::TpConfig,
    ) -> Result<DeviceMatrix> {
        self.load_matrix_pair_fused_inner(ctx, name_a, name_b, Some(tp))
    }

    fn load_matrix_pair_fused_inner(
        &self,
        ctx: &DeviceContext,
        name_a: &str,
        name_b: &str,
        tp: Option<&infer_topo::TpConfig>,
    ) -> Result<DeviceMatrix> {
        let spec_for = |name: &str| -> Result<Option<infer_topo::ShardingSpec>> {
            let Some(tp) = tp else { return Ok(None) };
            let rows = self.logical_rows(name)?;
            Ok(Some(infer_topo::column_shard(rows, tp)))
        };
        self.load_matrices_row_fused(
            ctx,
            &[(name_a, spec_for(name_a)?), (name_b, spec_for(name_b)?)],
        )
    }

    fn load_bf16_row_sharded(
        &self,
        ctx: &DeviceContext,
        name: &str,
        spec: &infer_topo::ShardingSpec,
    ) -> Result<DeviceMatrix> {
        const BF16_ELEM_SIZE: usize = 2;
        let tensor = self.borrow_bf16_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D BF16 tensor, got shape {:?}",
            tensor.shape
        );
        let sharded = crate::shard_slice::shard_column_parallel(
            tensor.bytes(),
            tensor.shape[0],
            tensor.shape[1],
            BF16_ELEM_SIZE,
            spec,
        )?;
        DeviceMatrix::from_safetensors(ctx, &sharded.bytes, sharded.rows, sharded.cols)
            .with_context(|| format!("upload row-sharded tensor {name}"))
    }

    /// Logical (unsharded) output-row count of a matrix, quant-aware.
    pub(crate) fn logical_rows(&self, name: &str) -> Result<usize> {
        match self.quant_view_for(name)? {
            Some(view) => {
                ensure!(
                    view.logical_shape.len() == 2,
                    "{name}: expected 2D matrix, got {:?}",
                    view.logical_shape
                );
                Ok(view.logical_shape[0])
            }
            None => {
                let tensor = self.borrow_bf16_tensor(name)?;
                ensure!(
                    tensor.shape.len() == 2,
                    "{name}: expected 2D BF16 tensor, got shape {:?}",
                    tensor.shape
                );
                Ok(tensor.shape[0])
            }
        }
    }

    /// Load N same-K projections as ONE row-fused matrix (concatenated along output
    /// rows, in
    /// `parts` order). Each part carries its own optional row-shard spec (None = full
    /// matrix).
    /// W8A16 parts fuse before the Marlin repack; every other format fuses on device.
    pub(crate) fn load_matrices_row_fused(
        &self,
        ctx: &DeviceContext,
        parts: &[(&str, Option<infer_topo::ShardingSpec>)],
    ) -> Result<DeviceMatrix> {
        ensure!(parts.len() >= 2, "row fuse needs at least 2 parts");
        // Returns (matrix, needs_marlin_repack) — W8A16 parts stay INT8 until
        // the fused matrix repacks once.
        let load_one =
            |name: &str, spec: &Option<infer_topo::ShardingSpec>| -> Result<(DeviceMatrix, bool)> {
                match self.quant_view_for(name)? {
                    Some(view) => {
                        let shard = match spec {
                            Some(spec) => QuantMatrixShard::Rows(spec.clone()),
                            None => QuantMatrixShard::Full,
                        };
                        if let QuantFormat::W8A16 { group_size } = view.format {
                            Ok((
                                self.load_w8a16_view_unpacked(ctx, &view, &shard, group_size)?,
                                true,
                            ))
                        } else {
                            Ok((self.load_quant_or_dense_view(ctx, &view, shard)?, false))
                        }
                    }
                    None => match spec {
                        Some(spec) => Ok((self.load_bf16_row_sharded(ctx, name, spec)?, false)),
                        None => Ok((self.load_matrix(ctx, name)?, false)),
                    },
                }
            };
        let (mut fused, repack) = load_one(parts[0].0, &parts[0].1)?;
        for (name, spec) in &parts[1..] {
            let (m, r) = load_one(name, spec)?;
            ensure!(
                r == repack,
                "row fuse {}: mixed W8A16/non-W8A16 parts",
                name
            );
            fused = DeviceMatrix::fuse_rows(ctx, &fused, &m)
                .with_context(|| format!("row fuse + {name}"))?;
        }
        if repack {
            let names: Vec<&str> = parts.iter().map(|(n, _)| *n).collect();
            fused
                .repack_for_marlin_w8a16(ctx)
                .with_context(|| format!("Marlin W8A16 repack fused {}", names.join("+")))?;
        }
        Ok(fused)
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

    /// Quant-aware twin of [`Self::load_qkv_head_sharded`]. `block_index` has the
    /// same Q-vs-replicated-KV meaning (see that method's docs).
    pub(crate) fn load_qkv_head_sharded_quant_aware(
        &self,
        ctx: &DeviceContext,
        name: &str,
        local_heads: usize,
        head_dim: usize,
        block_index: usize,
    ) -> Result<DeviceMatrix> {
        let Some(view) = self.quant_view_for(name)? else {
            return self.load_qkv_head_sharded(ctx, name, local_heads, head_dim, block_index);
        };
        ensure!(
            view.logical_shape.len() == 2,
            "{}: expected 2D quant-aware QKV matrix, got {:?}",
            view.name,
            view.logical_shape
        );
        let total_rows = view.logical_shape[0];
        let local_rows = local_heads * head_dim;
        let offset = block_index * local_rows;
        ensure!(
            offset + local_rows <= total_rows,
            "{}: head shard [{offset}, {}) exceeds rows {total_rows} \
             (local_heads={local_heads}, head_dim={head_dim}, block_index={block_index})",
            view.name,
            offset + local_rows,
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

    /// Quant-aware twin of the BF16 fused-qkv head shard: shard the F8_E4M3 weight AND
    /// its
    /// block-scale sidecar with the SAME head-block helper, or return None so the
    /// caller
    /// keeps its BF16 path.
    ///
    /// Scale rows map 1:1 to head-block rows only because `head_rows` is a whole
    /// multiple of
    /// `block_m`, so the blocks can be re-expressed in scale units and fed through the
    /// identical helper.
    pub(crate) fn load_linear_qkv_fp8_head_sharded(
        &self,
        ctx: &DeviceContext,
        name: &str,
        blocks: &[crate::shard_slice::HeadBlock],
        tp: &TpConfig,
    ) -> Result<Option<DeviceMatrix>> {
        let Some(view) = self.quant_view_for(name)? else {
            return Ok(None);
        };
        let QuantFormat::Fp8BlockScaled {
            block_m,
            block_k,
            scale_apply,
        } = view.format
        else {
            // A non-FP8 quant sidecar on the fused qkv would silently mis-shard here.
            return Ok(None);
        };
        ensure!(
            view.logical_shape.len() == 2,
            "{name}: expected 2D fused qkv FP8 matrix, got {:?}",
            view.logical_shape
        );
        let rows = view.logical_shape[0];
        let cols = view.logical_shape[1];

        let weight = self.borrow_raw_tensor(&view.name)?;
        ensure!(
            weight.dtype == Dtype::F8_E4M3 && weight.shape == view.logical_shape,
            "{name}: expected F8_E4M3 {:?}, got {:?} {:?}",
            view.logical_shape,
            weight.dtype,
            weight.shape
        );
        let scale = self.borrow_raw_tensor(&view.scale_names[0])?;
        let scale_elem = float_elem_size(&view.scale_names[0], scale.dtype)?;
        let scale_rows = rows.div_ceil(block_m);
        let scale_cols = cols.div_ceil(block_k);
        ensure!(
            scale.shape == [scale_rows, scale_cols],
            "{}: scale shape {:?} != [{scale_rows}, {scale_cols}]",
            view.scale_names[0],
            scale.shape
        );

        // A head block's rows must tile the scale-block grid, or a scale row straddles
        // two.
        let scale_blocks = blocks
            .iter()
            .enumerate()
            .map(|(i, b)| {
                ensure!(
                    b.head_rows.is_multiple_of(block_m),
                    "{name}: fused block {i} head_rows {} not a multiple of block_m {block_m} \
                     (FP8 head shard requires head rows to tile the scale grid)",
                    b.head_rows
                );
                Ok(crate::shard_slice::HeadBlock {
                    heads: b.heads,
                    head_rows: b.head_rows / block_m,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            cols.is_multiple_of(block_k),
            "{name}: cols {cols} not a multiple of block_k {block_k}"
        );

        let weight_shard = crate::shard_slice::shard_head_blocks_column_parallel(
            weight.bytes(),
            cols,
            1,
            blocks,
            tp,
        )?;
        let scale_shard = crate::shard_slice::shard_head_blocks_column_parallel(
            scale.bytes(),
            scale_cols,
            scale_elem,
            &scale_blocks,
            tp,
        )?;
        let scales = tensor_bytes_to_f32(
            &view.scale_names[0],
            scale.dtype,
            &scale_shard.bytes,
            scale_apply,
        )?;
        let matrix = DeviceMatrix::from_fp8_block_scaled(
            ctx,
            &weight_shard.bytes,
            &scales,
            weight_shard.rows,
            weight_shard.cols,
            block_m,
            block_k,
        )
        .with_context(|| format!("upload sharded FP8 fused qkv {name}"))?;
        Ok(Some(matrix))
    }

    /// Load this EP rank's MoE weights for one layer (routed gate/up/down + router gate
    /// +
    /// shared expert) and build the per-expert weight-pointer tables. Only the experts
    /// in
    /// `split.local_expert_start..local_expert_end()` are loaded.
    ///
    /// Routed experts ship either per-expert (`experts.{i}.{gate,up,down}_proj.weight`)
    /// or
    /// stacked+fused (`experts.gate_up_proj` `[E, 2*moe_inter, hidden]`, gate rows
    /// first, plus
    /// `experts.down_proj` `[E, hidden, moe_inter]`), auto-detected per layer.
    ///
    /// Under TP the router gate and the shared-expert sigmoid gate stay replicated —
    /// routing
    /// must be computed identically on every rank — while the shared expert is sharded
    /// like a
    /// dense MLP so its partial lands in the same post-MoE all-reduce.
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
        let deepgemm_native_ready = crate::runtime_flags::qwen35_deepgemm()
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
        // sm_120 has no DeepGEMM native bridge, but the CUTLASS sm_120a grouped
        // collective
        // consumes the SAME contiguous grouped FP8 caches — build them regardless of
        // the
        // Hopper-only preflight, with weight scales transposed to N-contiguous SFB.
        let sm120 = ctx.is_sm120();
        let mut direct_fp8_routed = None;
        // The OPD rollout student re-merges LoRA into experts each step, which needs a
        // mutable
        // per-expert BF16 `DeviceMatrix` — suppress the fused grouped-FP8 path.
        let experts_bf16_resident = crate::runtime_flags::qwen35_moe_experts_bf16_resident();
        // The stacked tensors are HF `nn.Parameter`s (no `.weight` suffix), but accept
        // a
        // `.weight`-suffixed export too.
        let resolve_stacked = |base: &str| -> Option<String> {
            [base.to_string(), format!("{base}.weight")]
                .into_iter()
                .find(|name| self.has_tensor(name))
        };
        if !experts_bf16_resident && per_expert_quant_probe && (deepgemm_native_ready || sm120) {
            direct_fp8_routed = self.load_fp8_moe_groups_direct(
                ctx,
                names,
                split,
                moe_intermediate_size,
                hidden_size,
                sm120,
            )?;
        }
        if direct_fp8_routed.is_some() {
            // The direct FP8 path already filled the grouped caches; no per-expert
            // list.
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
            // Borrow each stacked tensor ONCE and slice every local expert out of the
            // cached
            // bytes — an owned load costs ~1 GiB + 512 MiB of host memcpy per MoE
            // layer.
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
                // gate = rows [0, mi), up = rows [mi, 2*mi) of expert e's contiguous
                // block.
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
            crate::executor::cuda_startup_log(
                "loader.moe.stacked_routed_load",
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
        crate::executor::cuda_startup_log(
            "loader.moe.routed_load",
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
        crate::executor::cuda_startup_log(
            "loader.moe.shared_load",
            shared_t0,
            format_args!("layer={}", names.mlp_prefix),
        );

        // Concat the per-expert matrices into one contiguous [G, n, k] buffer per
        // projection
        // and DROP the per-expert copies — keeping both doubles routed-expert VRAM (~2x
        // model
        // weights on Qwen3.6-35B). An unavailable native bridge must skip the grouped
        // caches
        // so `use_deepgemm` self-disables instead of erroring at the first MoE forward.
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
        // BF16-resident student: dequantize the per-expert FP8 experts in place so the
        // layer
        // is one BF16 kernel with a stable ptr table the LoRA re-merge can fold into.
        // Must run
        // before the grouped-cache decision and pointer-table build below.
        let mut expert_weight_format = expert_weight_format;
        let mut gate_sig = gate_sig;
        let mut down_sig = down_sig;
        if experts_bf16_resident && expert_weight_format == WeightFormat::Fp8BlockScaled {
            for (proj, experts) in [("gate", &mut gate), ("up", &mut up), ("down", &mut down)] {
                for (e, m) in experts.iter_mut().enumerate() {
                    dequantize_fp8_expert_to_bf16_in_place(ctx, m).with_context(|| {
                        format!("dequantize FP8 {proj} expert {e} of `{}`", names.mlp_prefix)
                    })?;
                }
            }
            expert_weight_format = WeightFormat::DenseBf16;
            // FP8 quant signatures are stale now the experts are dense BF16.
            gate_sig = None;
            down_sig = None;
        }
        let routed_quant = expert_weight_format.is_quantized();
        let grouped_t0 = Instant::now();
        // BF16 grouped DeepGEMM is Hopper-only: the contiguous kernel reads m_indices,
        // which
        // the sm_120 path leaves None, so building the caches there would panic on
        // first
        // prefill. Also skipped when BF16-resident: the concat clears the per-expert
        // Vecs the
        // LoRA re-merge needs mutable.
        let deepgemm_ready =
            !routed_quant && deepgemm_native_ready && !sm120 && !experts_bf16_resident;
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
            // Event tracking is disabled: sync before dropping the sources, whose bytes
            // the
            // async D2D concats above may still be reading.
            ctx.sync()?;
            (Some(w13_g), Some(down_g))
        } else {
            (None, None)
        };

        let ptr_tables = build_moe_layer_pointer_tables(
            ctx,
            expert_weight_format,
            &gate,
            &up,
            &down,
            gate_grouped.as_ref(),
            up_grouped.as_ref(),
            down_grouped.as_ref(),
            w13_fp8_grouped.as_ref(),
            down_fp8_grouped.as_ref(),
        )?;
        if fp8_deepgemm_ready {
            gate.clear();
            up.clear();
            down.clear();
        }
        crate::executor::cuda_startup_log(
            "loader.moe.grouped_cache",
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
            gate_ptrs: ptr_tables.gate_ptrs,
            up_ptrs: ptr_tables.up_ptrs,
            down_ptrs: ptr_tables.down_ptrs,
            gate_scale_ptrs: ptr_tables.gate_scale_ptrs,
            up_scale_ptrs: ptr_tables.up_scale_ptrs,
            down_scale_ptrs: ptr_tables.down_scale_ptrs,
            gate_global_ptrs: ptr_tables.gate_global_ptrs,
            up_global_ptrs: ptr_tables.up_global_ptrs,
            down_global_ptrs: ptr_tables.down_global_ptrs,
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
        // Transpose each expert's block scales from the checkpoint's K-contiguous
        // `[n_blocks, k_blocks]` to CUTLASS's N-contiguous SFB. Hopper keeps the raw
        // layout.
        transpose_sfb: bool,
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

        // CUTLASS SFB is N-contiguous per group; the checkpoint is K-contiguous.
        if transpose_sfb {
            transpose_group_block_scales(&mut w13_scales, groups, w13_scale_rows, w13_scale_cols);
            transpose_group_block_scales(
                &mut down_scales,
                groups,
                down_scale_rows,
                down_scale_cols,
            );
        }

        // `transpose_sfb` (== sm_120) also records the layout on the cache, so the
        // executor's
        // dispatch reads one source of truth.
        let w13 = MoeFp8ExpertGroup::from_host(
            ctx,
            &w13_weight,
            &w13_scales,
            groups,
            w13_rows,
            hidden_size,
            transpose_sfb,
        )?;
        let down = MoeFp8ExpertGroup::from_host(
            ctx,
            &down_weight,
            &down_scales,
            groups,
            hidden_size,
            moe_intermediate_size,
            transpose_sfb,
        )?;
        crate::executor::cuda_startup_log(
            "loader.moe.direct_fp8_grouped_load",
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

    /// Whether `name` exists in the checkpoint: weight-map lookup when an index is
    /// present,
    /// otherwise each shard header is parsed.
    pub(crate) fn has_tensor(&self, name: &str) -> bool {
        if !self.weight_map.is_empty() {
            return self.weight_map.contains_key(name);
        }
        self.tensor_headers()
            .map(|headers| headers.contains_key(name))
            .unwrap_or(false)
    }

    pub(crate) fn quant_view_for(&self, name: &str) -> Result<Option<QuantTensorView>> {
        if self.quant_manifest.is_none() {
            return Ok(None);
        }
        let headers = self.tensor_headers()?;
        let mut candidates = vec![name.to_owned()];
        if let Some(base) = name.strip_suffix(".weight") {
            candidates.push(format!("{base}.weight_packed"));
            candidates.push(format!("{base}.qweight"));
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
            crate::executor::cuda_startup_log(
                "loader.tensor_headers.shard",
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
        crate::executor::cuda_startup_log(
            "loader.tensor_headers.total",
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
                let tensor = self.borrow_raw_tensor(&view.name)?;
                ensure!(
                    tensor.shape.len() == 2,
                    "{}: expected 2D F32 tensor, got {:?}",
                    view.name,
                    tensor.shape
                );
                let sharded = self.shard_raw_2d_cow(
                    tensor.bytes(),
                    tensor.shape[0],
                    tensor.shape[1],
                    4,
                    &shard,
                )?;
                DeviceMatrix::from_safetensors(
                    ctx,
                    Self::tensor_bytes_to_bf16(&view.name, Dtype::F32, sharded.bytes.as_ref())?
                        .as_ref(),
                    sharded.rows,
                    sharded.cols,
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
            QuantFormat::W4A16 { group_size } => {
                self.load_w4a16_view(ctx, view, &shard, group_size)
            }
            QuantFormat::GptqW4A16 { group_size } => {
                self.load_gptq_w4a16_view(ctx, view, &shard, group_size)
            }
            QuantFormat::W8A16 { group_size } => {
                self.load_w8a16_view(ctx, view, &shard, group_size)
            }
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
        let tensor = self.borrow_bf16_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D BF16 tensor, got shape {:?}",
            tensor.shape
        );
        let sharded = match kind {
            infer_topo::ParallelLinearKind::Column => crate::shard_slice::shard_column_parallel(
                tensor.bytes(),
                tensor.shape[0],
                tensor.shape[1],
                BF16_ELEM_SIZE,
                spec,
            )?,
            infer_topo::ParallelLinearKind::Row => crate::shard_slice::shard_row_parallel(
                tensor.bytes(),
                tensor.shape[0],
                tensor.shape[1],
                BF16_ELEM_SIZE,
                spec,
            )?,
        };
        DeviceMatrix::from_safetensors(ctx, &sharded.bytes, sharded.rows, sharded.cols)
            .with_context(|| format!("upload sharded tensor {name}"))
    }

    fn shard_raw_2d_cow<'a>(
        &self,
        bytes: &'a [u8],
        rows: usize,
        cols: usize,
        elem_size: usize,
        shard: &QuantMatrixShard,
    ) -> Result<ShardedBytesCow<'a>> {
        match shard {
            QuantMatrixShard::Full => Ok(ShardedBytesCow {
                bytes: Cow::Borrowed(bytes),
                rows,
                cols,
            }),
            QuantMatrixShard::Rows(spec) => {
                let sharded =
                    crate::shard_slice::shard_column_parallel(bytes, rows, cols, elem_size, spec)?;
                Ok(ShardedBytesCow {
                    bytes: Cow::Owned(sharded.bytes),
                    rows: sharded.rows,
                    cols: sharded.cols,
                })
            }
            QuantMatrixShard::Cols(spec) => {
                let sharded =
                    crate::shard_slice::shard_row_parallel(bytes, rows, cols, elem_size, spec)?;
                Ok(ShardedBytesCow {
                    bytes: Cow::Owned(sharded.bytes),
                    rows: sharded.rows,
                    cols: sharded.cols,
                })
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
        let weight = self.borrow_raw_tensor(&view.name)?;
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
        let weight_shard = self.shard_raw_2d_cow(weight.bytes(), rows, cols, 1, shard)?;
        let scale = self.borrow_raw_tensor(&view.scale_names[0])?;
        let scale_elem = float_elem_size(&view.scale_names[0], scale.dtype)?;
        let scale_rows = rows.div_ceil(block_m);
        let scale_cols = cols.div_ceil(block_k);
        ensure!(
            scale.shape == [scale_rows, scale_cols],
            "{}: scale shape {:?} != [{scale_rows}, {scale_cols}]",
            view.scale_names[0],
            scale.shape
        );
        let scale_shard = self.shard_fp8_block_scales_cow(
            scale.bytes(),
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
            scale_shard.bytes.as_ref(),
            scale_apply,
        )?;
        DeviceMatrix::from_fp8_block_scaled(
            ctx,
            weight_shard.bytes.as_ref(),
            &scales,
            weight_shard.rows,
            weight_shard.cols,
            block_m,
            block_k,
        )
        .with_context(|| format!("upload FP8 block-scaled tensor {}", view.name))
    }

    fn shard_fp8_block_scales_cow<'a>(
        &self,
        bytes: &'a [u8],
        elem_size: usize,
        rows: usize,
        cols: usize,
        block_m: usize,
        block_k: usize,
        shard: &QuantMatrixShard,
    ) -> Result<ShardedBytesCow<'a>> {
        let scale_rows = rows.div_ceil(block_m);
        let scale_cols = cols.div_ceil(block_k);
        match shard {
            QuantMatrixShard::Full => Ok(ShardedBytesCow {
                bytes: Cow::Borrowed(bytes),
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
                let sharded = crate::shard_slice::shard_column_parallel(
                    bytes,
                    scale_rows,
                    scale_cols,
                    elem_size,
                    &scale_spec,
                )?;
                Ok(ShardedBytesCow {
                    bytes: Cow::Owned(sharded.bytes),
                    rows: sharded.rows,
                    cols: sharded.cols,
                })
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
                let sharded = crate::shard_slice::shard_row_parallel(
                    bytes,
                    scale_rows,
                    scale_cols,
                    elem_size,
                    &scale_spec,
                )?;
                Ok(ShardedBytesCow {
                    bytes: Cow::Owned(sharded.bytes),
                    rows: sharded.rows,
                    cols: sharded.cols,
                })
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
        let weight = self.borrow_raw_tensor(&view.name)?;
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
        let weight_shard = self.shard_raw_2d_cow(weight.bytes(), rows, cols, 1, shard)?;
        let scale = self.borrow_raw_tensor(&view.scale_names[0])?;
        let input_scale = self.borrow_raw_tensor(&view.scale_names[1])?;
        let scales = tensor_bytes_to_f32(
            &view.scale_names[0],
            scale.dtype,
            scale.bytes(),
            scale_apply,
        )?;
        let input_scales = tensor_bytes_to_f32(
            &view.scale_names[1],
            input_scale.dtype,
            input_scale.bytes(),
            ScaleApply::Multiply,
        )?;
        ensure!(
            scales.len() == 1 && input_scales.len() == 1,
            "{}: FP8 per-shard dispatch currently requires scalar weight/input scales, got {}/{}",
            view.name,
            scales.len(),
            input_scales.len()
        );
        DeviceMatrix::from_fp8_per_shard(
            ctx,
            weight_shard.bytes.as_ref(),
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
        let weight = self.borrow_raw_tensor(&view.name)?;
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
            self.shard_fp4_packed_weight_cow(weight.bytes(), rows, cols, group_size, shard)?;
        let scale = self.borrow_raw_tensor(&view.scale_names[0])?;
        ensure!(
            scale.dtype == Dtype::F8_E4M3 && scale.shape == [rows, cols / group_size],
            "{}: expected FP8 group scale [{rows}, {}], got {:?} {:?}",
            view.scale_names[0],
            cols / group_size,
            scale.dtype,
            scale.shape
        );
        let scale_shard =
            self.shard_fp4_group_scales_cow(scale.bytes(), rows, cols, group_size, shard)?;
        let global = self.borrow_raw_tensor(&view.scale_names[1])?;
        let global_scales = tensor_bytes_to_f32(
            &view.scale_names[1],
            global.dtype,
            global.bytes(),
            global_scale_apply,
        )?;
        ensure!(
            global_scales.len() == 1,
            "{}: FP4 global scale must be scalar, got {} values",
            view.scale_names[1],
            global_scales.len()
        );
        let input_scales = if view.scale_names.len() > 2 {
            let input = self.borrow_raw_tensor(&view.scale_names[2])?;
            Some(tensor_bytes_to_f32(
                &view.scale_names[2],
                input.dtype,
                input.bytes(),
                ScaleApply::Multiply,
            )?)
        } else {
            None
        };
        DeviceMatrix::from_fp4_e2m1_group(
            ctx,
            weight_shard.bytes.as_ref(),
            scale_shard.bytes.as_ref(),
            &global_scales,
            input_scales.as_deref(),
            weight_shard.rows,
            weight_shard.cols * 2,
            group_size,
        )
        .with_context(|| format!("upload FP4 E2M1 tensor {}", view.name))
    }

    fn load_w4a16_view(
        &self,
        ctx: &DeviceContext,
        view: &QuantTensorView,
        shard: &QuantMatrixShard,
        group_size: usize,
    ) -> Result<DeviceMatrix> {
        let weight = self.borrow_raw_tensor(&view.name)?;
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
            self.shard_fp4_packed_weight_cow(weight.bytes(), rows, cols, group_size, shard)?;
        let scale = self.borrow_raw_tensor(&view.scale_names[0])?;
        ensure!(
            scale.dtype == Dtype::BF16 && scale.shape == [rows, cols / group_size],
            "{}: expected BF16 group scale [{rows}, {}], got {:?} {:?}",
            view.scale_names[0],
            cols / group_size,
            scale.dtype,
            scale.shape
        );
        let scale_shard =
            self.shard_w4a16_scales_cow(scale.bytes(), rows, cols, group_size, shard)?;
        let scale_bf16 = Self::tensor_bytes_to_bf16(
            &view.scale_names[0],
            scale.dtype,
            scale_shard.bytes.as_ref(),
        )?;
        // SAFETY: BF16 bytes (2 bytes/elem) are a valid `&[half::bf16]` slice
        // (align 2, every bit pattern valid).
        let scales_data: &[half::bf16] = unsafe {
            std::slice::from_raw_parts(
                scale_bf16.as_ptr().cast::<half::bf16>(),
                scale_bf16.len() / 2,
            )
        };
        DeviceMatrix::from_quantized_int4(
            ctx,
            weight_shard.bytes.as_ref(),
            scales_data,
            weight_shard.rows,
            weight_shard.cols * 2,
            group_size,
        )
        .with_context(|| format!("upload W4A16 tensor {}", view.name))
    }

    /// Load a GPTQ/AutoRound W4A16 view and convert to ARLE's internal W4A16
    /// layout. GPTQ stores qweight as I32 [k//8, n] (8 int4 per word), scales as
    /// BF16 [groups, n], and qzeros as I32 [groups, n//8]. ARLE expects U8
    /// [n, k//2] (2 int4 per byte, low nibble first) and BF16 [n, groups].
    fn load_gptq_w4a16_view(
        &self,
        ctx: &DeviceContext,
        view: &QuantTensorView,
        shard: &QuantMatrixShard,
        group_size: usize,
    ) -> Result<DeviceMatrix> {
        let qweight = self.borrow_raw_tensor(&view.name)?;
        let n = view.logical_shape[0];
        let k = view.logical_shape[1];
        let packed_k = k / 8;
        ensure!(
            qweight.dtype == Dtype::I32 && qweight.shape == [packed_k, n],
            "{}: expected GPTQ qweight I32 [{}, {}], got {:?} {:?}",
            view.name,
            packed_k,
            n,
            qweight.dtype,
            qweight.shape
        );
        let num_groups = k / group_size;

        let qw_bytes = qweight.bytes();
        // SAFETY: qweight is I32 [packed_k, n], so its bytes are a valid i32 slice.
        let qw: &[i32] = unsafe {
            std::slice::from_raw_parts(qw_bytes.as_ptr().cast::<i32>(), qw_bytes.len() / 4)
        };

        let scales = self.borrow_raw_tensor(&view.scale_names[0])?;
        ensure!(
            (scales.dtype == Dtype::BF16 || scales.dtype == Dtype::F16)
                && scales.shape == [num_groups, n],
            "{}: expected GPTQ scales BF16/F16 [{}, {}], got {:?} {:?}",
            view.scale_names[0],
            num_groups,
            n,
            scales.dtype,
            scales.shape
        );
        let scales_bytes = scales.bytes();
        let scales_bf16: Cow<[u8]> = if scales.dtype == Dtype::F16 {
            Cow::Owned(
                scales_bytes
                    .chunks_exact(2)
                    .flat_map(|c| {
                        let f16 = half::f16::from_le_bytes([c[0], c[1]]);
                        half::bf16::from_f32(f16.to_f32()).to_le_bytes()
                    })
                    .collect(),
            )
        } else {
            Cow::Borrowed(scales_bytes)
        };
        let scales_bytes = scales_bf16.as_ref();

        let qzeros = self.borrow_raw_tensor(&view.scale_names[1])?;
        let packed_n = n / 8;
        ensure!(
            qzeros.dtype == Dtype::I32 && qzeros.shape == [num_groups, packed_n],
            "{}: expected GPTQ qzeros I32 [{}, {}], got {:?} {:?}",
            view.scale_names[1],
            num_groups,
            packed_n,
            qzeros.dtype,
            qzeros.shape
        );
        let qz_bytes = qzeros.bytes();
        // SAFETY: qzeros is I32 [num_groups, packed_n], so its bytes are a valid i32
        // slice.
        let qz: &[i32] = unsafe {
            std::slice::from_raw_parts(qz_bytes.as_ptr().cast::<i32>(), qz_bytes.len() / 4)
        };

        // GPTQ packs 8 uint4 per i32; ARLE packs 2 per u8 (lo=even, hi=odd), logical
        // [n, k].
        let arle_packed_cols = k / 2;
        let mut arle_weight = vec![0u8; n * arle_packed_cols];
        let mut arle_scales = vec![0u8; n * num_groups * 2]; // BF16

        for g in 0..num_groups {
            let k_start = g * group_size;
            let k_end = k_start + group_size;
            let mut zeros = vec![0u8; n];
            for (j, zero) in zeros.iter_mut().enumerate() {
                let word_idx = j / 8;
                let shift = (j % 8) * 4;
                let raw = ((qz[g * packed_n + word_idx] >> shift) & 0xF) as u8;
                *zero = raw + 1; // GPTQ stores actual_zero - 1
            }
            let scale_offset = g * n * 2;
            for j in 0..n {
                arle_scales[j * num_groups * 2 + g * 2] = scales_bytes[scale_offset + j * 2];
                arle_scales[j * num_groups * 2 + g * 2 + 1] =
                    scales_bytes[scale_offset + j * 2 + 1];
            }
            for k_idx in k_start..k_end {
                let packed_row = k_idx / 8;
                let nibble = k_idx % 8;
                let shift = nibble * 4;
                let arle_byte_idx_base = k_idx / 2;
                let is_high = k_idx % 2 == 1;
                for j in 0..n {
                    let raw = ((qw[packed_row * n + j] >> shift) & 0xF) as u8;
                    // GPTQ (raw - zero)*scale vs ARLE (uint4 - 8)*scale ⇒ uint4 = raw -
                    // zero + 8.
                    let arle_val = raw.wrapping_sub(zeros[j]).wrapping_add(8);
                    let arle_idx = j * arle_packed_cols + arle_byte_idx_base;
                    if is_high {
                        arle_weight[arle_idx] |= (arle_val & 0x0F) << 4;
                    } else {
                        arle_weight[arle_idx] = arle_val & 0x0F;
                    }
                }
            }
        }

        // Shard the converted ARLE-layout tensors: GPTQ packs both K (into i32 words)
        // and N
        // (into qzeros words), so sharding the raw tensors directly is error-prone.
        let weight_shard =
            self.shard_fp4_packed_weight_cow(&arle_weight, n, k, group_size, shard)?;
        let scale_shard = self.shard_w4a16_scales_cow(&arle_scales, n, k, group_size, shard)?;

        // SAFETY: BF16 bytes are valid &[half::bf16]
        let scales_data: &[half::bf16] = unsafe {
            std::slice::from_raw_parts(
                scale_shard.bytes.as_ptr().cast::<half::bf16>(),
                scale_shard.bytes.len() / 2,
            )
        };

        DeviceMatrix::from_quantized_int4(
            ctx,
            weight_shard.bytes.as_ref(),
            scales_data,
            weight_shard.rows,
            weight_shard.cols * 2,
            group_size,
        )
        .with_context(|| format!("upload GPTQ W4A16 tensor {}", view.name))
    }

    /// Load a W8A16 view: signed INT8 weights (non-packed, [rows, cols]) with per-row
    /// per-column-group BF16 scales — the CUDA `w8a16_gemv` kernel reads 1 byte per
    /// weight.
    fn load_w8a16_view(
        &self,
        ctx: &DeviceContext,
        view: &QuantTensorView,
        shard: &QuantMatrixShard,
        group_size: usize,
    ) -> Result<DeviceMatrix> {
        let mut matrix = self.load_w8a16_view_unpacked(ctx, view, shard, group_size)?;
        // Build the Marlin tensor-core layout (Ampere+); no-op below sm_80 or on
        // non-tile-aligned shapes → dispatch falls back to scalar/dequant.
        matrix
            .repack_for_marlin_w8a16(ctx)
            .with_context(|| format!("Marlin W8A16 repack {}", view.name))?;
        Ok(matrix)
    }

    /// W8A16 view load WITHOUT the Marlin repack — the row-fusion path concats two INT8
    /// sources first so the fused matrix repacks once.
    fn load_w8a16_view_unpacked(
        &self,
        ctx: &DeviceContext,
        view: &QuantTensorView,
        shard: &QuantMatrixShard,
        group_size: usize,
    ) -> Result<DeviceMatrix> {
        let weight = self.borrow_raw_tensor(&view.name)?;
        let rows = view.logical_shape[0];
        let cols = view.logical_shape[1];
        ensure!(
            weight.dtype == Dtype::I8 && weight.shape == [rows, cols],
            "{}: expected INT8 [{rows}, {cols}], got {:?} {:?}",
            view.name,
            weight.dtype,
            weight.shape
        );
        let weight_shard = self.shard_raw_2d_cow(weight.bytes(), rows, cols, 1, shard)?;
        let scale = self.borrow_raw_tensor(&view.scale_names[0])?;
        ensure!(
            scale.dtype == Dtype::BF16 && scale.shape == [rows, cols / group_size],
            "{}: expected BF16 group scale [{rows}, {}], got {:?} {:?}",
            view.scale_names[0],
            cols / group_size,
            scale.dtype,
            scale.shape
        );
        let scale_shard =
            self.shard_w4a16_scales_cow(scale.bytes(), rows, cols, group_size, shard)?;
        let scale_bf16 = Self::tensor_bytes_to_bf16(
            &view.scale_names[0],
            scale.dtype,
            scale_shard.bytes.as_ref(),
        )?;
        // SAFETY: BF16 bytes (2 bytes/elem) are a valid `&[half::bf16]` slice.
        let scales_data: &[half::bf16] = unsafe {
            std::slice::from_raw_parts(
                scale_bf16.as_ptr().cast::<half::bf16>(),
                scale_bf16.len() / 2,
            )
        };
        // SAFETY: I8 bytes are a valid `&[i8]` slice (align 1, all patterns valid).
        let qweight: &[i8] = unsafe {
            std::slice::from_raw_parts(
                weight_shard.bytes.as_ref().as_ptr().cast::<i8>(),
                weight_shard.bytes.as_ref().len(),
            )
        };
        DeviceMatrix::from_quantized_int8(
            ctx,
            qweight,
            scales_data,
            weight_shard.rows,
            weight_shard.cols,
            group_size,
        )
        .with_context(|| format!("upload W8A16 tensor {}", view.name))
    }

    fn shard_w4a16_scales_cow<'a>(
        &self,
        bytes: &'a [u8],
        rows: usize,
        logical_cols: usize,
        group_size: usize,
        shard: &QuantMatrixShard,
    ) -> Result<ShardedBytesCow<'a>> {
        let scale_cols = logical_cols / group_size;
        match shard {
            QuantMatrixShard::Full => Ok(ShardedBytesCow {
                bytes: Cow::Borrowed(bytes),
                rows,
                cols: scale_cols,
            }),
            QuantMatrixShard::Rows(spec) => {
                let sharded =
                    crate::shard_slice::shard_column_parallel(bytes, rows, scale_cols, 2, spec)?;
                Ok(ShardedBytesCow {
                    bytes: Cow::Owned(sharded.bytes),
                    rows: sharded.rows,
                    cols: sharded.cols,
                })
            }
            QuantMatrixShard::Cols(spec) => {
                ensure!(
                    spec.offset.is_multiple_of(group_size) && spec.size.is_multiple_of(group_size),
                    "W4A16 scale col shard {:?} must align to group_size={group_size}",
                    spec.range()
                );
                let scale_spec = ShardingSpec {
                    offset: spec.offset / group_size,
                    size: spec.size / group_size,
                    total: scale_cols,
                };
                let sharded = crate::shard_slice::shard_row_parallel(
                    bytes,
                    rows,
                    scale_cols,
                    2,
                    &scale_spec,
                )?;
                Ok(ShardedBytesCow {
                    bytes: Cow::Owned(sharded.bytes),
                    rows: sharded.rows,
                    cols: sharded.cols,
                })
            }
        }
    }

    fn shard_fp4_packed_weight_cow<'a>(
        &self,
        bytes: &'a [u8],
        rows: usize,
        logical_cols: usize,
        group_size: usize,
        shard: &QuantMatrixShard,
    ) -> Result<ShardedBytesCow<'a>> {
        let packed_cols = logical_cols / 2;
        match shard {
            QuantMatrixShard::Full => Ok(ShardedBytesCow {
                bytes: Cow::Borrowed(bytes),
                rows,
                cols: packed_cols,
            }),
            QuantMatrixShard::Rows(spec) => {
                let sharded =
                    crate::shard_slice::shard_column_parallel(bytes, rows, packed_cols, 1, spec)?;
                Ok(ShardedBytesCow {
                    bytes: Cow::Owned(sharded.bytes),
                    rows: sharded.rows,
                    cols: sharded.cols,
                })
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
                let sharded = crate::shard_slice::shard_row_parallel(
                    bytes,
                    rows,
                    packed_cols,
                    1,
                    &packed_spec,
                )?;
                Ok(ShardedBytesCow {
                    bytes: Cow::Owned(sharded.bytes),
                    rows: sharded.rows,
                    cols: sharded.cols,
                })
            }
        }
    }

    fn shard_fp4_group_scales_cow<'a>(
        &self,
        bytes: &'a [u8],
        rows: usize,
        logical_cols: usize,
        group_size: usize,
        shard: &QuantMatrixShard,
    ) -> Result<ShardedBytesCow<'a>> {
        let scale_cols = logical_cols / group_size;
        match shard {
            QuantMatrixShard::Full => Ok(ShardedBytesCow {
                bytes: Cow::Borrowed(bytes),
                rows,
                cols: scale_cols,
            }),
            QuantMatrixShard::Rows(spec) => {
                let sharded =
                    crate::shard_slice::shard_column_parallel(bytes, rows, scale_cols, 1, spec)?;
                Ok(ShardedBytesCow {
                    bytes: Cow::Owned(sharded.bytes),
                    rows: sharded.rows,
                    cols: sharded.cols,
                })
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
                let sharded = crate::shard_slice::shard_row_parallel(
                    bytes,
                    rows,
                    scale_cols,
                    1,
                    &scale_spec,
                )?;
                Ok(ShardedBytesCow {
                    bytes: Cow::Owned(sharded.bytes),
                    rows: sharded.rows,
                    cols: sharded.cols,
                })
            }
        }
    }

    /// Shard bytes: mmap into the bounded LRU on first touch, then hand out an `Rc`
    /// clone (no
    /// `RefCell` guard escapes, so nested loads that fill other shards never hit a
    /// `BorrowMutError`). Loading beyond the byte budget evicts older entries;
    /// outstanding
    /// [`SharedTensor`] borrows keep their shard alive through their own `Rc`.
    fn shard_bytes(&self, idx: usize) -> Result<Rc<ShardBytes>> {
        let path = self
            .shards
            .get(idx)
            .ok_or_else(|| anyhow!("shard index {idx} out of range"))?;
        if let Some(bytes) = self.shard_cache.borrow_mut().get(idx) {
            return Ok(bytes);
        }
        let t0 = Instant::now();
        let bytes = Rc::new(ShardBytes::map(path)?);
        let mut cache = self.shard_cache.borrow_mut();
        let evicted = cache.insert(idx, Rc::clone(&bytes));
        drop(cache);
        for (evicted_idx, evicted_bytes) in evicted {
            crate::executor::cuda_startup_log(
                "loader.shard_cache_evict",
                Instant::now(),
                format_args!("idx={evicted_idx} bytes={evicted_bytes}"),
            );
        }
        crate::executor::cuda_startup_log(
            "loader.shard_mmap",
            t0,
            format_args!("idx={idx} bytes={} path={}", bytes.len(), path.display()),
        );
        Ok(bytes)
    }

    fn shard_tensor_metas(
        &self,
        idx: usize,
        shard: &Rc<ShardBytes>,
    ) -> Result<Rc<BTreeMap<String, ShardTensorMeta>>> {
        if let Some(metas) = self.shard_meta_cache.borrow().get(&idx) {
            return Ok(Rc::clone(metas));
        }
        let t0 = Instant::now();
        let path = &self.shards[idx];
        let tensors = SafeTensors::deserialize(&shard[..])
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
        crate::executor::cuda_startup_log(
            "loader.shard_deserialize",
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

    /// Dtype-agnostic shard read. Same read-once cache as the BF16 path; the typed gate
    /// lives
    /// in the callers.
    fn load_raw_from_shard(&self, idx: usize, name: &str) -> Result<OwnedTensor> {
        let t0 = Instant::now();
        let tensor = self.borrow_raw_from_shard(idx, name)?;
        let owned = OwnedTensor {
            shape: tensor.shape.clone(),
            bytes: tensor.bytes().to_vec(),
            dtype: tensor.dtype,
        };
        crate::executor::cuda_startup_log(
            "loader.tensor.owned_copy",
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

    /// Zero-copy shard read: the returned [`SharedTensor`] aliases the tensor's byte
    /// range
    /// inside the read-once shard cache. The stacked-expert loader slices ~1.5 GiB per
    /// MoE
    /// layer out of these bytes.
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

    /// Zero-copy tensor lookup across shards.
    pub(crate) fn borrow_raw_tensor(&self, name: &str) -> Result<SharedTensor> {
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

/// A tensor whose bytes alias the loader's mmap shard cache (`Rc` share, zero host
/// copies).
pub(crate) struct SharedTensor {
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: Dtype,
    shard: std::rc::Rc<ShardBytes>,
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

/// Saturating encode of `val` into FP8 E4M3FN: 1 sign + 4 exp + 3 mant, bias 7, max
/// normal
/// 0x7E = 448.0. Values are clamped to ±448; NaN input maps to zero.
fn encode_f8_e4m3fn_sat(val: f32) -> u8 {
    const E4M3_MAX: f32 = 448.0;
    const BIAS: i32 = 7;
    let val = if val.is_nan() {
        0.0
    } else {
        val.clamp(-E4M3_MAX, E4M3_MAX)
    };
    let sign: u8 = if val.is_sign_negative() { 0x80 } else { 0x00 };
    let abs = val.abs();
    if abs == 0.0 {
        return sign;
    }
    let bits = abs.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let fp8_biased_exp = f32_exp + BIAS;
    if fp8_biased_exp <= 0 {
        // Subnormal: 0.mmmm × 2^(1−7), i.e. value = mant × 2^(−9)
        let mant = ((abs * 512.0).round() as i32).clamp(0, 7) as u8;
        return sign | mant;
    }
    if fp8_biased_exp >= 15 {
        return sign | 0x7E; // saturate to ±448.0
    }
    let fp8_biased_exp = fp8_biased_exp as u8;
    let f32_mant = bits & 0x007F_FFFF;
    // Round-to-nearest: add the guard bit (bit 19) before truncating to 3 bits.
    let fp8_mant_raw = (f32_mant + (1 << 19)) >> 20;
    if fp8_mant_raw > 7 {
        let new_exp = fp8_biased_exp + 1;
        return if new_exp >= 15 {
            sign | 0x7E
        } else {
            sign | (new_exp << 3)
        };
    }
    sign | (fp8_biased_exp << 3) | (fp8_mant_raw as u8)
}

// DSv4 FP8/FP4 + E8M0 loaders. The `allow(dead_code)` is retained on necessity grounds,
// not
// caller count: individual dtype loaders are config-selected, so some read as dead
// under a
// given build.
#[allow(dead_code)]
impl SafetensorLoader {
    /// Dtype-agnostic full-tensor read (shape + raw bytes + dtype).
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

    /// Normalize a small 1D/2D tensor to BF16 bytes — these ship BF16 or F32 depending
    /// on the
    /// checkpoint. Must match `load_vec_any`'s conversion exactly so a sharded load
    /// stays
    /// byte-identical to slicing the single-GPU upload.
    pub(crate) fn dsv4_bytes_to_bf16<'a>(
        name: &str,
        tensor: &'a OwnedTensor,
    ) -> Result<Cow<'a, [u8]>> {
        Self::tensor_bytes_to_bf16(name, tensor.dtype, tensor.bytes.as_slice())
    }

    pub(crate) fn tensor_bytes_to_bf16<'a>(
        name: &str,
        dtype: Dtype,
        bytes: &'a [u8],
    ) -> Result<Cow<'a, [u8]>> {
        match dtype {
            Dtype::BF16 => Ok(Cow::Borrowed(bytes)),
            Dtype::F32 => {
                ensure!(
                    bytes.len().is_multiple_of(4),
                    "{name}: F32 tensor byte length {} is not divisible by 4",
                    bytes.len()
                );
                Ok(Cow::Owned(
                    bytes
                        .chunks_exact(4)
                        .flat_map(|c| {
                            half::bf16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .to_le_bytes()
                        })
                        .collect(),
                ))
            }
            other => anyhow::bail!("{name}: DSv4 tensor expected BF16/F32, got {other:?}"),
        }
    }

    /// Load a DSv4 1D norm/bias vector — BF16 or F32 in the checkpoint, normalized to
    /// BF16.
    pub(crate) fn load_dsv4_vec(&self, ctx: &DeviceContext, name: &str) -> Result<DeviceVec> {
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 1,
            "{name}: expected 1D tensor, got shape {:?}",
            tensor.shape
        );
        DeviceVec::from_safetensors(
            ctx,
            Self::tensor_bytes_to_bf16(name, tensor.dtype, tensor.bytes())?.as_ref(),
        )
        .with_context(|| format!("upload DSv4 vec {name}"))
    }

    /// Load a DSv4 2D router gate (the only non-FP8 2D weight), normalized to BF16.
    pub(crate) fn load_dsv4_bf16_matrix(
        &self,
        ctx: &DeviceContext,
        name: &str,
    ) -> Result<DeviceMatrix> {
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D tensor, got shape {:?}",
            tensor.shape
        );
        DeviceMatrix::from_safetensors(
            ctx,
            Self::tensor_bytes_to_bf16(name, tensor.dtype, tensor.bytes())?.as_ref(),
            tensor.shape[0],
            tensor.shape[1],
        )
        .with_context(|| format!("upload DSv4 gate {name}"))
    }

    /// Sharded dense BF16/F32 load, for checkpoints that leave the tiny low-rank `wo_a`
    /// unquantized. Converts to bf16, then slices rows.
    /// ponytail: only Column{0}/Replicated — the only shards `wo_a` uses; bail else.
    pub(crate) fn load_dsv4_bf16_sharded(
        &self,
        ctx: &DeviceContext,
        name: &str,
        shard: Shard,
        tp: &TpConfig,
    ) -> Result<DeviceMatrix> {
        if tp.is_single() || shard == Shard::Replicated {
            return self.load_dsv4_bf16_matrix(ctx, name);
        }
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D dense tensor, got shape {:?}",
            tensor.shape
        );
        let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
        let bf16 = Self::tensor_bytes_to_bf16(name, tensor.dtype, tensor.bytes())?;
        match shard {
            Shard::Column { dim: 0 } => {
                let spec = infer_topo::column_shard(rows, tp);
                let sharded =
                    crate::shard_slice::shard_column_parallel(bf16.as_ref(), rows, cols, 2, &spec)?;
                DeviceMatrix::from_safetensors(ctx, &sharded.bytes, sharded.rows, sharded.cols)
                    .with_context(|| format!("upload DSv4 dense shard {name}"))
            }
            other => bail!(
                "{name}: DSv4 dense sharded load supports Column{{dim:0}}/Replicated, got {other:?}"
            ),
        }
    }

    /// Normalize a DSv4 block scale to E8M0 bytes. Accepts native `F8_E8M0`, or the
    /// common
    /// `F32` power-of-two serialization: a power-of-two f32's biased exponent byte IS
    /// its
    /// E8M0 code, so `(bits >> 23) & 0xff` is lossless. Errors on a non-power-of-two
    /// F32.
    fn dsv4_block_scale_e8m0(scale_name: &str, dtype: Dtype, bytes: &[u8]) -> Result<Vec<u8>> {
        match dtype {
            Dtype::F8_E8M0 => Ok(bytes.to_vec()),
            Dtype::F32 => {
                ensure!(
                    bytes.len().is_multiple_of(4),
                    "{scale_name}: F32 scale byte length {} not a multiple of 4",
                    bytes.len()
                );
                bytes
                    .chunks_exact(4)
                    .map(|c| {
                        let bits = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                        ensure!(
                            bits & 0x007f_ffff == 0,
                            "{scale_name}: F32 block scale {} is not a power of two; cannot map losslessly to E8M0",
                            f32::from_bits(bits)
                        );
                        Ok(((bits >> 23) & 0xff) as u8)
                    })
                    .collect()
            }
            other => {
                bail!(
                    "{scale_name}: expected F8_E8M0 or F32 power-of-two block scale, got {other:?}"
                )
            }
        }
    }

    pub(crate) fn load_dsv4_block_scaled(
        &self,
        ctx: &DeviceContext,
        name: &str,
    ) -> Result<DeviceMatrix> {
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D quantized tensor, got shape {:?}",
            tensor.shape
        );
        let scale_name = name
            .strip_suffix(".weight")
            .map(|prefix| format!("{prefix}.scale"))
            .ok_or_else(|| anyhow!("{name}: quantized DSv4 tensor must end with .weight"))?;
        let scale = self.borrow_raw_tensor(&scale_name)?;
        ensure!(
            scale.shape.len() == 2,
            "{scale_name}: expected 2D scale, got shape {:?}",
            scale.shape
        );
        let scale_e8m0 = Self::dsv4_block_scale_e8m0(&scale_name, scale.dtype, scale.bytes())?;
        let (scale_rows, scale_cols) = (scale.shape[0], scale.shape[1]);

        match tensor.dtype {
            Dtype::F8_E4M3 => {
                let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
                DeviceMatrix::from_dsv4_fp8_block_scaled(
                    ctx,
                    tensor.bytes(),
                    &scale_e8m0,
                    rows,
                    cols,
                    scale_rows,
                    scale_cols,
                )
                .with_context(|| format!("upload DSv4 FP8 matrix {name}"))
            }
            // FAIL-CLOSED: the FP4/MX lane NaNs from the first compressed-attention
            // layer
            // (#137); the FP4 dequant plumbing below stays for a future re-license.
            Dtype::I8 => bail!(
                "{name}: FP4/MX DSv4 checkpoints are unsupported — the compressed-attention \
                 path NaNs (#137). Use the FP8-native export."
            ),
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

    /// Load a DSv4 block-scaled FP8 matrix and apply a TP shard before upload. The FP8
    /// payload
    /// and E8M0 block scales must be sliced together, or the shard reads valid FP8
    /// bytes with
    /// the wrong scale blocks.
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

        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D quantized tensor, got shape {:?}",
            tensor.shape
        );
        let scale_name = name
            .strip_suffix(".weight")
            .map(|prefix| format!("{prefix}.scale"))
            .ok_or_else(|| anyhow!("{name}: quantized DSv4 tensor must end with .weight"))?;
        let scale = self.borrow_raw_tensor(&scale_name)?;
        ensure!(
            scale.shape.len() == 2,
            "{scale_name}: expected 2D scale, got shape {:?}",
            scale.shape
        );
        let scale_e8m0 = Self::dsv4_block_scale_e8m0(&scale_name, scale.dtype, scale.bytes())?;

        match tensor.dtype {
            Dtype::F8_E4M3 => {
                let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
                let (scale_rows, scale_cols) = (scale.shape[0], scale.shape[1]);
                let (weight, scales) = match shard {
                    Shard::Column { dim: 0 } => {
                        let spec = infer_topo::column_shard(rows, tp);
                        let weight = crate::shard_slice::shard_column_parallel(
                            tensor.bytes(),
                            rows,
                            cols,
                            1,
                            &spec,
                        )?;
                        let scale_spec =
                            Self::dsv4_scale_shard_for_value_shard(name, &spec, scale_rows, 128)?;
                        let scales = crate::shard_slice::shard_column_parallel(
                            &scale_e8m0,
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
                            tensor.bytes(),
                            rows,
                            cols,
                            1,
                            &spec,
                        )?;
                        let scale_spec =
                            Self::dsv4_scale_shard_for_value_shard(name, &spec, scale_cols, 128)?;
                        let scales = crate::shard_slice::shard_row_parallel(
                            &scale_e8m0,
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

    /// Dense-bf16 dequant copy of one DSv4 FP8 block-scaled tensor, TP-sharded like the
    /// FP8
    /// original. Costs 2× the F8 bytes in VRAM.
    fn load_dsv4_block_scaled_bf16_copy(
        &self,
        ctx: &DeviceContext,
        name: &str,
        shard: Shard,
        tp: &TpConfig,
    ) -> Result<DeviceMatrix> {
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D quantized tensor, got shape {:?}",
            tensor.shape
        );
        let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
        let f32 = self.dequantize_dsv4_block_scaled_to_f32_host(name, rows, cols)?;
        let bytes: Vec<u8> = f32
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        if tp.is_single() || shard == Shard::Replicated {
            return DeviceMatrix::from_safetensors(ctx, &bytes, rows, cols)
                .with_context(|| format!("upload bf16 dequant copy {name}"));
        }
        match shard {
            Shard::Column { dim: 0 } => {
                let spec = infer_topo::column_shard(rows, tp);
                let sharded =
                    crate::shard_slice::shard_column_parallel(&bytes, rows, cols, 2, &spec)?;
                DeviceMatrix::from_safetensors(ctx, &sharded.bytes, sharded.rows, sharded.cols)
                    .with_context(|| format!("upload sharded bf16 dequant copy {name}"))
            }
            other => bail!("{name}: unsupported bf16-copy TP shard policy {other:?}"),
        }
    }

    fn build_dsv4_wo_a_group_tables(
        ctx: &DeviceContext,
        weight: &DeviceMatrix,
        rows_per_group: usize,
    ) -> Result<crate::dsv4::Dsv4WoAGroupTables> {
        ensure!(
            rows_per_group > 0 && weight.rows.is_multiple_of(rows_per_group),
            "DSv4 wo_a grouped table needs rows {} divisible by rows_per_group {}",
            weight.rows,
            rows_per_group
        );
        let groups = weight.rows / rows_per_group;
        ensure!(groups > 0, "DSv4 wo_a grouped table has zero groups");
        // Dense BF16 wo_a: the grouped route-GEMV kernel is FP8/FP4-only, so these
        // tables are
        // consumed only when groups>1, which the dense path rejects. Carry the shape
        // and build
        // trivial base-pointer tables so the struct stays well-formed.
        if weight.weight_format == WeightFormat::DenseBf16 {
            let (base, _bg) = weight.data.device_ptr(&ctx.stream);
            let stride_bytes = rows_per_group
                .checked_mul(weight.cols)
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| anyhow!("DSv4 dense wo_a group stride overflow"))?;
            let ptrs: Vec<u64> = (0..groups)
                .map(|g| base + (g * stride_bytes) as u64)
                .collect();
            return Ok(crate::dsv4::Dsv4WoAGroupTables {
                weight_ptrs: ctx
                    .stream
                    .clone_htod(&ptrs)
                    .map_err(|e| anyhow!("DSv4 dense wo_a ptr table H2D failed: {e}"))?,
                scale_ptrs: ctx
                    .stream
                    .clone_htod(&ptrs)
                    .map_err(|e| anyhow!("DSv4 dense wo_a scale ptr table H2D failed: {e}"))?,
                groups,
                rows_per_group,
                cols_per_group: weight.cols,
                scale_rows_per_group: 0,
                scale_cols: 0,
            });
        }
        ensure!(
            weight.dsv4_scale_rows > 0
                && weight.dsv4_scale_cols > 0
                && weight.dsv4_scale_rows.is_multiple_of(groups),
            "DSv4 wo_a scale rows {} must be non-zero and divisible by groups {groups}",
            weight.dsv4_scale_rows
        );
        let scale_rows_per_group = weight.dsv4_scale_rows / groups;
        let qweight = weight
            .qweight
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 wo_a grouped table missing quantized weight bytes"))?;
        let scales = weight
            .dsv4_scales
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 wo_a grouped table missing E8M0 scales"))?;
        let weight_stride_bytes = match weight.weight_format {
            WeightFormat::Dsv4Fp8BlockScaled => rows_per_group.checked_mul(weight.cols),
            WeightFormat::Dsv4Fp4BlockScaled => {
                ensure!(
                    weight.cols.is_multiple_of(2),
                    "DSv4 wo_a FP4 grouped table needs even cols, got {}",
                    weight.cols
                );
                rows_per_group.checked_mul(weight.cols / 2)
            }
            other => bail!("DSv4 wo_a grouped table expected FP8/FP4 block-scaled, got {other:?}"),
        }
        .ok_or_else(|| {
            anyhow!(
                "DSv4 wo_a grouped table weight stride overflow rows_per_group={} cols={}",
                rows_per_group,
                weight.cols
            )
        })?;
        let scale_stride_bytes = scale_rows_per_group
            .checked_mul(weight.dsv4_scale_cols)
            .ok_or_else(|| {
                anyhow!(
                    "DSv4 wo_a grouped table scale stride overflow scale_rows={} scale_cols={}",
                    scale_rows_per_group,
                    weight.dsv4_scale_cols
                )
            })?;
        ensure!(
            qweight.len() == weight_stride_bytes * groups,
            "DSv4 wo_a grouped qweight len {} != groups({groups})*stride({weight_stride_bytes})",
            qweight.len()
        );
        ensure!(
            scales.len() == scale_stride_bytes * groups,
            "DSv4 wo_a grouped scale len {} != groups({groups})*stride({scale_stride_bytes})",
            scales.len()
        );

        let (weight_base, _wg) = qweight.device_ptr(&ctx.stream);
        let (scale_base, _sg) = scales.device_ptr(&ctx.stream);
        let weight_ptrs: Vec<u64> = (0..groups)
            .map(|g| weight_base + (g * weight_stride_bytes) as u64)
            .collect();
        let scale_ptrs: Vec<u64> = (0..groups)
            .map(|g| scale_base + (g * scale_stride_bytes) as u64)
            .collect();
        Ok(crate::dsv4::Dsv4WoAGroupTables {
            weight_ptrs: ctx
                .stream
                .clone_htod(&weight_ptrs)
                .map_err(|e| anyhow!("DSv4 wo_a weight ptr table H2D failed: {e}"))?,
            scale_ptrs: ctx
                .stream
                .clone_htod(&scale_ptrs)
                .map_err(|e| anyhow!("DSv4 wo_a scale ptr table H2D failed: {e}"))?,
            groups,
            rows_per_group,
            cols_per_group: weight.cols,
            scale_rows_per_group,
            scale_cols: weight.dsv4_scale_cols,
        })
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
        // GLM (`weight_scale_inv` FP8) ⇒ re-encode experts into the DSv4 FP8+E8M0
        // layout the grouped DeepGEMM cache consumes; DSv4 loads E8M0 directly.
        glm: bool,
    ) -> Result<crate::dsv4::Dsv4MoeLayer> {
        use cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache;
        use deepseek_spec::DeepSeekV4MoeRoutingKind;

        let mega_moe = matches!(
            crate::runtime_flags::dsv4_moe_transport()?,
            crate::runtime_flags::Dsv4MoeTransport::MegaMoe
        );

        // DSv4 ships FP8 E4M3 + E8M0 `<prefix>.scale`; GLM ships FP8 E4M3 + F32
        // `weight_scale_inv` (128×128 block scales), consumed losslessly by the 1D2D
        // `sfb`
        // path — no E8M0 re-encode, no dequant. Both ride the SAME
        // `build_grouped_cache`.
        let build_w13 =
            |first: &DeviceMatrix, second: &DeviceMatrix| -> Result<Dsv4Fp8DeepGemmWeightCache> {
                if glm {
                    Dsv4Fp8DeepGemmWeightCache::from_fp8_block_scaled_weight_pair_rows(
                        ctx, first, second,
                    )
                } else {
                    Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight_pair_rows(ctx, first, second)
                }
            };
        let build_w2 = |down: &DeviceMatrix| -> Result<Dsv4Fp8DeepGemmWeightCache> {
            if glm {
                Dsv4Fp8DeepGemmWeightCache::from_fp8_block_scaled_weight(ctx, down)
            } else {
                Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight(ctx, down)
            }
        };
        let load_fp8 = |name: &str| -> Result<DeviceMatrix> {
            if glm {
                self.load_dsv4_glm_fp8_as_block_scaled(ctx, name)
            } else {
                self.load_dsv4_block_scaled(ctx, name)
            }
        };

        let mut w13 = Vec::with_capacity(split.experts_per_rank);
        let mut w2 = Vec::with_capacity(split.experts_per_rank);
        for e in split.local_expert_start..split.local_expert_end() {
            let expert = names.expert(e);
            // w1 (gate) over w3 (up), row-stacked into one fused FP8 cache so the
            // masked grouped GEMM produces [gate | up] in a single launch.
            let w1 = load_fp8(&expert.w1)?;
            let w3 = load_fp8(&expert.w3)?;
            w13.push(build_w13(&w1, &w3)?);
            let down = load_fp8(&expert.w2)?;
            w2.push(build_w2(&down)?);
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
        let w13_layout = if mega_moe {
            crate::moe::GroupedWeightLayout::InterleavedL1
        } else {
            crate::moe::GroupedWeightLayout::Normal
        };
        let w13_grouped = crate::moe::build_grouped_cache(
            ctx,
            w13.as_slice(),
            2 * intermediate,
            hidden_dim,
            w13_layout,
        )?;
        let w2_grouped = crate::moe::build_grouped_cache(
            ctx,
            w2.as_slice(),
            hidden_dim,
            intermediate,
            crate::moe::GroupedWeightLayout::Normal,
        )?;
        let num_groups = w13_grouped.groups;
        ensure!(
            num_groups == split.experts_per_rank && w2_grouped.groups == num_groups,
            "DSv4 grouped expert count mismatch: w13={} w2={} expected {}",
            w13_grouped.groups,
            w2_grouped.groups,
            split.experts_per_rank
        );

        let gate = self.load_dsv4_bf16_matrix(ctx, &names.gate_weight)?;
        let (gate_bias, hash_tid2eid_device) = match routing_kind {
            DeepSeekV4MoeRoutingKind::LearnedBias => {
                let bias_name = names
                    .gate_bias
                    .as_ref()
                    .ok_or_else(|| anyhow!("DSv4 bias-routed MoE layer missing gate.bias"))?;
                (Some(self.load_dsv4_vec(ctx, bias_name)?), None)
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
                (None, Some(device))
            }
        };

        // SHARED expert (always-on, n_shared_experts == 1); same builders as the routed
        // ones.
        let shared = names
            .shared_experts
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 expects an always-on shared expert"))?;
        let shared_w1 = load_fp8(&shared.w1)?;
        let shared_w3 = load_fp8(&shared.w3)?;
        let shared_w13 = build_w13(&shared_w1, &shared_w3)?;
        let shared_down = load_fp8(&shared.w2)?;
        let shared_w2 = build_w2(&shared_down)?;

        Ok(crate::dsv4::Dsv4MoeLayer {
            w13_grouped,
            w2_grouped,
            num_groups,
            hidden_dim,
            intermediate,
            gate,
            gate_bias,
            hash_tid2eid_device,
            routing_kind,
            shared_w13,
            shared_w2,
            gemv_tables: std::sync::OnceLock::new(),
        })
    }

    /// Load a GLM dense-MLP layer (`first_k_dense_replace` layers). GLM ships FP8 + F32
    /// `weight_scale_inv`; each projection is dequantized to dense bf16 because the FP8
    /// grouped caches need E8M0 scales GLM does not carry.
    pub(crate) fn load_dsv4_dense_mlp(
        &self,
        ctx: &DeviceContext,
        names: &deepseek_spec::DeepSeekV4ExpertTensorNames,
    ) -> Result<crate::dsv4::Dsv4DenseMlp> {
        // w1 = gate_proj, w3 = up_proj, w2 = down_proj.
        let gate = self.load_dsv4_block_scaled_dialect(ctx, &names.w1)?;
        let up = self.load_dsv4_block_scaled_dialect(ctx, &names.w3)?;
        let down = self.load_dsv4_block_scaled_dialect(ctx, &names.w2)?;
        let hidden_dim = gate.cols;
        let intermediate = gate.rows;
        ensure!(
            up.rows == intermediate && up.cols == hidden_dim,
            "GLM dense up_proj shape {}x{} != gate {}x{}",
            up.rows,
            up.cols,
            intermediate,
            hidden_dim
        );
        ensure!(
            down.rows == hidden_dim && down.cols == intermediate,
            "GLM dense down_proj shape {}x{} != [{hidden_dim}, {intermediate}]",
            down.rows,
            down.cols
        );
        Ok(crate::dsv4::Dsv4DenseMlp {
            gate,
            up,
            down,
            hidden_dim,
            intermediate,
        })
    }

    /// Load a DSv4 1D `i64` hash-routing table (`gate.tid2eid`) into host memory.
    pub(crate) fn load_dsv4_i64_host(&self, name: &str) -> Result<Vec<i64>> {
        use safetensors::tensor::Dtype;
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.dtype == Dtype::I64,
            "{name}: DSv4 tid2eid expected I64, got {:?}",
            tensor.dtype
        );
        ensure!(
            tensor.bytes().len() % 8 == 0,
            "{name}: I64 byte length {} is not a multiple of 8",
            tensor.bytes().len()
        );
        Ok(tensor
            .bytes()
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

    /// Load a DSv4 2D matrix dispatching on its on-disk dtype: BF16/F32 are quantized
    /// to FP8
    /// E4M3FN block-scaled (128×128 blocks, E8M0 scales) on the host so downstream
    /// linears
    /// route through `mla_linear`; F8_E4M3/I8 keep the native block-scaled path.
    pub(crate) fn load_dsv4_global_matrix(
        &self,
        ctx: &DeviceContext,
        name: &str,
    ) -> Result<DeviceMatrix> {
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D tensor, got shape {:?}",
            tensor.shape
        );
        let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
        match tensor.dtype {
            Dtype::BF16 | Dtype::F32 => {
                let (fp8, scales, scale_rows, scale_cols) = Self::quantize_to_dsv4_fp8_host(
                    name,
                    tensor.dtype,
                    tensor.bytes(),
                    rows,
                    cols,
                )?;
                DeviceMatrix::from_dsv4_fp8_block_scaled(
                    ctx, &fp8, &scales, rows, cols, scale_rows, scale_cols,
                )
                .with_context(|| format!("upload quantized DSv4 FP8 matrix {name}"))
            }
            Dtype::F8_E4M3 | Dtype::I8 => self.load_dsv4_block_scaled(ctx, name),
            other => bail!("{name}: unsupported DSv4 global matrix dtype {other:?}"),
        }
    }

    /// Build one DSv4 MLA attention block. The Q/KV/O LoRA matrices are FP8/FP4
    /// block-scaled;
    /// `compressor` / `indexer` matrices may be FP8/FP4 or bf16, so they route through
    /// the
    /// dtype-dispatching [`Self::load_dsv4_global_matrix`].
    pub(crate) fn load_dsv4_attention(
        &self,
        ctx: &DeviceContext,
        config: &deepseek_spec::DeepSeekV4Config,
        names: &deepseek_spec::DeepSeekV4AttentionTensorNames,
        tp: &TpConfig,
    ) -> Result<crate::dsv4::Dsv4Attention> {
        // GLM (`plain_o_proj`) ships no `attn_sink` — its MLA has no per-head sink
        // logit.
        let (attn_sink, attn_sink_f32) = if config.plain_o_proj {
            (None, None)
        } else {
            let attn_sink = self.load_dsv4_vec(ctx, &names.attn_sink)?;
            let mut dst = ctx
                .stream
                .alloc_zeros::<f32>(attn_sink.len)
                .map_err(|e| anyhow!("DSv4 attn_sink f32 mirror alloc failed: {e}"))?;
            let (src_ptr, _sg) = attn_sink.data.device_ptr(&ctx.stream);
            let (dst_ptr, _dg) = dst.device_ptr_mut(&ctx.stream);
            // SAFETY: ptrs from live device allocations sized to the dims passed.
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
            drop(_sg);
            (Some(attn_sink), Some(dst))
        };
        // GLM (`plain_o_proj`) ships FP8 + F32 `weight_scale_inv`: dequant the Q/KV
        // projections
        // to dense bf16, since the FP8 DeepGEMM caches below need the E8M0
        // `dsv4_scales`
        // layout.
        let glm = config.plain_o_proj;
        let (wq_a, wkv) = if glm {
            (
                self.load_dsv4_block_scaled_dialect(ctx, &names.wq_a)?,
                self.load_dsv4_block_scaled_dialect(ctx, &names.wkv)?,
            )
        } else {
            (
                self.load_dsv4_block_scaled(ctx, &names.wq_a)?,
                self.load_dsv4_block_scaled(ctx, &names.wkv)?,
            )
        };
        let wqkv_a_deepgemm = if !glm && crate::attention::dsv4_fused_wqkv_decode_enabled()? {
            Some(
                cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight_pair_rows(
                    ctx, &wq_a, &wkv,
                )?,
            )
        } else {
            None
        };
        let wq_b = if glm {
            self.load_dsv4_block_scaled_dialect(ctx, &names.wq_b)?
        } else {
            self.load_dsv4_block_scaled_sharded(
                ctx,
                &names.wq_b,
                names
                    .shard_for(config, &names.wq_b, tp.world_size)
                    .unwrap_or(Shard::Replicated),
                tp,
            )?
        };
        let wq_b_deepgemm = self.decode_proj_cache(ctx, &wq_b)?;
        // Output projection. DSv4: low-rank wo_a→wo_b (+ per-group tables). GLM
        // (`plain_o_proj`): a single plain `o_proj`, plus the kv_b absorption split
        // (w_kc/w_vc).
        let (
            wo_a,
            wo_a_groups,
            wo_b,
            wo_a_deepgemm,
            wo_a_group_deepgemm,
            wo_b_deepgemm,
            o_proj,
            w_kc,
            w_vc,
        ) = if config.plain_o_proj {
            let o_proj = self.load_dsv4_block_scaled_dialect(
                ctx,
                names
                    .o_proj
                    .as_ref()
                    .ok_or_else(|| anyhow!("GLM plain_o_proj layer missing o_proj name"))?,
            )?;
            let (w_kc, w_vc) = self.load_dsv4_kv_b_absorb(ctx, config, names, tp)?;
            (
                None,
                None,
                None,
                None,
                None,
                None,
                Some(o_proj),
                Some(w_kc),
                Some(w_vc),
            )
        } else {
            // Some FP8 re-serializations leave the tiny low-rank `wo_a` as dense BF16.
            // Route by
            // dtype: block-scaled keeps the grouped route-GEMV + DeepGEMM levers; dense
            // BF16
            // rides `dsv4_linear` (gemm_batch). `wo_b` stays FP8.
            let wo_a_shard = names
                .shard_for(config, &names.wo_a, tp.world_size)
                .unwrap_or(Shard::Replicated);
            let wo_a = match self.borrow_raw_tensor(&names.wo_a)?.dtype {
                Dtype::F8_E4M3 | Dtype::I8 => {
                    self.load_dsv4_block_scaled_sharded(ctx, &names.wo_a, wo_a_shard, tp)?
                }
                _ => self.load_dsv4_bf16_sharded(ctx, &names.wo_a, wo_a_shard, tp)?,
            };
            let wo_a_is_block_scaled = matches!(
                wo_a.weight_format,
                WeightFormat::Dsv4Fp8BlockScaled | WeightFormat::Dsv4Fp4BlockScaled
            );
            let wo_a_groups = Self::build_dsv4_wo_a_group_tables(ctx, &wo_a, config.o_lora_rank)?;
            let wo_b = self.load_dsv4_block_scaled_sharded(
                ctx,
                &names.wo_b,
                names
                    .shard_for(config, &names.wo_b, tp.world_size)
                    .unwrap_or(Shard::Replicated),
                tp,
            )?;
            // DeepGEMM caches for the decode output projection. `wo_b` is always FP8;
            // `wo_a`
            // caches only when it is itself block-scaled (dense BF16 uses gemm_batch).
            let decode_alloc = crate::attention::dsv4_fused_wqkv_decode_enabled()?;
            let (wo_a_deepgemm, wo_a_group_deepgemm) = if decode_alloc && wo_a_is_block_scaled {
                let group_caches = if wo_a_groups.groups > 1 {
                    let mut caches = Vec::with_capacity(wo_a_groups.groups);
                    for group in 0..wo_a_groups.groups {
                        caches.push(
                            cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight_row_range(
                                ctx,
                                &wo_a,
                                group * wo_a_groups.rows_per_group,
                                wo_a_groups.rows_per_group,
                            )?,
                        );
                    }
                    Some(caches)
                } else {
                    None
                };
                let flat = (wo_a_groups.groups == 1)
                    .then(|| {
                        cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight(
                            ctx, &wo_a,
                        )
                    })
                    .transpose()?;
                (flat, group_caches)
            } else {
                (None, None)
            };
            let wo_b_deepgemm = if decode_alloc {
                Some(
                    cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight(ctx, &wo_b)?,
                )
            } else {
                None
            };
            (
                Some(wo_a),
                Some(wo_a_groups),
                Some(wo_b),
                wo_a_deepgemm,
                wo_a_group_deepgemm,
                wo_b_deepgemm,
                None,
                None,
                None,
            )
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
            wo_a_groups,
            wo_b,
            wo_a_deepgemm,
            wo_a_group_deepgemm,
            wo_b_deepgemm,
            attn_sink,
            attn_sink_f32,
            compressor: names
                .compressor
                .as_ref()
                .map(|c| self.load_dsv4_compressor(ctx, c, glm))
                .transpose()?,
            indexer: names
                .indexer
                .as_ref()
                .map(|i| self.load_dsv4_indexer(ctx, config, i))
                .transpose()?,
            o_proj,
            w_kc,
            w_vc,
        })
    }

    /// GLM `kv_b` absorption split (SGLang `deepseek_weight_loader.py:567-590`, the
    /// non-`use_deep_gemm_bmm` path).
    ///
    /// `kv_b_proj.weight` is `[num_heads*(qk_nope_head_dim + v_head_dim),
    /// kv_lora_rank]`, FP8
    /// block-scaled. Dequantize to bf16, split per head, and emit BOTH halves in
    /// `gemm_batch`
    /// orientation `[out, in]`: `w_kc` is the per-head transpose → `[num_heads*kv_lora,
    /// qk_nope]`, `w_vc` is as-is → `[num_heads*v_head, kv_lora]`. SGLang's `bmm`
    /// orientation
    /// is transposed from this because ARLE's `gemm_batch` computes `weight·x`, not
    /// `x·weight^T`; the contraction is identical.
    pub(crate) fn load_dsv4_kv_b_absorb(
        &self,
        ctx: &DeviceContext,
        config: &deepseek_spec::DeepSeekV4Config,
        names: &deepseek_spec::DeepSeekV4AttentionTensorNames,
        tp: &TpConfig,
    ) -> Result<(DeviceMatrix, DeviceMatrix)> {
        let kv_b_name = names
            .kv_b_proj
            .as_ref()
            .ok_or_else(|| anyhow!("GLM attention missing kv_b_proj name"))?;
        let qk_nope = config.qk_nope_head_dim;
        let v_head = config.v_head_dim;
        let kv_lora = config.kv_lora_rank;
        let num_heads = config.num_attention_heads;
        ensure!(
            qk_nope > 0 && v_head > 0 && kv_lora > 0,
            "GLM kv_b split needs qk_nope_head_dim/v_head_dim/kv_lora_rank > 0, \
             got {qk_nope}/{v_head}/{kv_lora}"
        );
        let rows = num_heads * (qk_nope + v_head);
        let kv_b = self.dequantize_dsv4_block_scaled_to_f32_host(kv_b_name, rows, kv_lora)?;
        let per_head = qk_nope + v_head;
        // w_kc[h] = w_kc_src[h] transposed: [kv_lora(out), qk_nope(in)].
        let mut w_kc = vec![0.0f32; num_heads * kv_lora * qk_nope];
        // w_vc[h] = w_vc_src[h] as-is: [v_head(out), kv_lora(in)].
        let mut w_vc = vec![0.0f32; num_heads * v_head * kv_lora];
        for h in 0..num_heads {
            let head_base = h * per_head * kv_lora;
            for i in 0..qk_nope {
                for j in 0..kv_lora {
                    w_kc[h * kv_lora * qk_nope + j * qk_nope + i] =
                        kv_b[head_base + i * kv_lora + j];
                }
            }
            let vc_src_base = head_base + qk_nope * kv_lora;
            for r in 0..v_head {
                for c in 0..kv_lora {
                    w_vc[h * v_head * kv_lora + r * kv_lora + c] =
                        kv_b[vc_src_base + r * kv_lora + c];
                }
            }
        }
        let w_kc_bytes: Vec<u8> = w_kc
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        let w_vc_bytes: Vec<u8> = w_vc
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        // Replicated at the head grain; TP head-sharding of w_kc/w_vc is not
        // implemented.
        let _ = tp;
        let w_kc = DeviceMatrix::from_safetensors(ctx, &w_kc_bytes, num_heads * kv_lora, qk_nope)
            .with_context(|| format!("upload GLM w_kc from {kv_b_name}"))?;
        let w_vc = DeviceMatrix::from_safetensors(ctx, &w_vc_bytes, num_heads * v_head, kv_lora)
            .with_context(|| format!("upload GLM w_vc from {kv_b_name}"))?;
        Ok((w_kc, w_vc))
    }

    /// Load a GLM (`weight_scale_inv`) FP8 E4M3 matrix as a `Fp8BlockScaled`
    /// [`DeviceMatrix`]
    /// (raw FP8 bytes + F32 per-128×128-block scales), WITHOUT dequantizing: GLM's
    /// `weight_scale_inv` is already the `[N/128, K/128]` row-major F32 grid the
    /// `sm90_fp8_gemm_1d2d` kernel reads as `sfb` — no transpose, no lossy E8M0
    /// re-encode.
    fn load_dsv4_glm_fp8_as_block_scaled(
        &self,
        ctx: &DeviceContext,
        name: &str,
    ) -> Result<DeviceMatrix> {
        const BLOCK: usize = 128;
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape.len() == 2,
            "{name}: expected 2D FP8 tensor, got {:?}",
            tensor.shape
        );
        ensure!(
            tensor.dtype == Dtype::F8_E4M3,
            "{name}: GLM MoE FP8 weight expects F8_E4M3, got {:?}",
            tensor.dtype
        );
        let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
        let weight = tensor.bytes();
        ensure!(
            weight.len() == rows * cols,
            "{name}: FP8 byte len {} != rows*cols {}",
            weight.len(),
            rows * cols
        );
        // GLM block scale: `<prefix>.weight_scale_inv`, F32, `[N/128, K/128]`.
        let base = name
            .strip_suffix(".weight")
            .ok_or_else(|| anyhow!("{name}: quantized tensor name must end with .weight"))?;
        let scale_name = format!("{base}.weight_scale_inv");
        let scale = self.borrow_raw_tensor(&scale_name)?;
        ensure!(
            scale.dtype == Dtype::F32,
            "{scale_name}: GLM block scale expected F32, got {:?}",
            scale.dtype
        );
        ensure!(
            scale.shape.len() == 2,
            "{scale_name}: expected 2D scale, got {:?}",
            scale.shape
        );
        let (scale_rows, scale_cols) = (scale.shape[0], scale.shape[1]);
        let want_rows = rows.div_ceil(BLOCK);
        let want_cols = cols.div_ceil(BLOCK);
        ensure!(
            scale_rows == want_rows && scale_cols == want_cols,
            "{scale_name}: GLM block scale {scale_rows}x{scale_cols} != [N/128, K/128] = \
             {want_rows}x{want_cols} for weight {rows}x{cols} (block 128x128)"
        );
        let scales: Vec<f32> = scale
            .bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        DeviceMatrix::from_fp8_block_scaled(ctx, weight, &scales, rows, cols, BLOCK, BLOCK)
            .with_context(|| format!("upload GLM FP8 block-scaled MoE weight {name}"))
    }

    /// Quantize a BF16 or F32 row-major matrix to DSv4 FP8 E4M3FN block-scaled on the
    /// host:
    /// 128×128 blocks with E8M0 per-block scales, the layout natively quantized DSv4
    /// weights
    /// use. Returns `(fp8_bytes, e8m0_scale_bytes, scale_rows, scale_cols)`.
    fn quantize_to_dsv4_fp8_host(
        name: &str,
        dtype: Dtype,
        bytes: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<(Vec<u8>, Vec<u8>, usize, usize)> {
        const BLOCK: usize = 128;
        const E4M3_MAX: f32 = 448.0;
        let vals: Vec<f32> = match dtype {
            Dtype::BF16 => {
                ensure!(
                    bytes.len() == rows * cols * 2,
                    "{name}: BF16 byte length {} != rows*cols*2={}",
                    bytes.len(),
                    rows * cols * 2
                );
                bytes
                    .chunks_exact(2)
                    .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect()
            }
            Dtype::F32 => {
                ensure!(
                    bytes.len() == rows * cols * 4,
                    "{name}: F32 byte length {} != rows*cols*4={}",
                    bytes.len(),
                    rows * cols * 4
                );
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            other => bail!("{name}: quantize_to_dsv4_fp8_host: unexpected dtype {other:?}"),
        };
        let scale_rows = rows.div_ceil(BLOCK);
        let scale_cols = cols.div_ceil(BLOCK);
        let mut fp8_out = vec![0u8; rows * cols];
        let mut scale_out = vec![0u8; scale_rows * scale_cols];
        for br in 0..scale_rows {
            let row_start = br * BLOCK;
            let row_end = (row_start + BLOCK).min(rows);
            for bc in 0..scale_cols {
                let col_start = bc * BLOCK;
                let col_end = (col_start + BLOCK).min(cols);
                let mut max_abs = 0.0f32;
                for r in row_start..row_end {
                    for c in col_start..col_end {
                        let v = vals[r * cols + c].abs();
                        if v > max_abs {
                            max_abs = v;
                        }
                    }
                }
                // Smallest power-of-2 scale s.t. max_abs / scale ≤ E4M3_MAX.
                // E8M0 byte b ↔ scale = 2^(b−127).
                let scale_byte: u8 = if max_abs == 0.0 {
                    127 // 2^0 = 1; arbitrary for all-zero block
                } else {
                    let b = ((max_abs / E4M3_MAX).log2() + 127.0).ceil() as i32;
                    b.clamp(1, 254) as u8
                };
                let scale = 2.0f32.powi(scale_byte as i32 - 127);
                scale_out[br * scale_cols + bc] = scale_byte;
                for r in row_start..row_end {
                    for c in col_start..col_end {
                        fp8_out[r * cols + c] = encode_f8_e4m3fn_sat(vals[r * cols + c] / scale);
                    }
                }
            }
        }
        Ok((fp8_out, scale_out, scale_rows, scale_cols))
    }

    /// Dequantize a DSv4/GLM block-scaled FP8 E4M3 matrix to host f32. DSv4 ships
    /// `<prefix>.scale` (F8_E8M0, 1 byte/block); GLM ships `<name>.weight_scale_inv`
    /// (F32,
    /// block [128,128]). The block scale multiplies.
    fn dequantize_dsv4_block_scaled_to_f32_host(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        use crate::quant_format::decode_f8_e4m3fn;
        let tensor = self.borrow_raw_tensor(name)?;
        ensure!(
            tensor.shape == [rows, cols],
            "{name}: expected FP8 shape [{rows}, {cols}], got {:?}",
            tensor.shape
        );
        ensure!(
            tensor.dtype == Dtype::F8_E4M3,
            "{name}: GLM kv_b dequant expects F8_E4M3, got {:?}",
            tensor.dtype
        );
        let weight = tensor.bytes();
        ensure!(
            weight.len() == rows * cols,
            "{name}: FP8 byte len {} != rows*cols {}",
            weight.len(),
            rows * cols
        );
        // Prefer the GLM `weight_scale_inv` (F32) dialect, else DSv4 `<prefix>.scale`
        // (E8M0).
        let base = name
            .strip_suffix(".weight")
            .ok_or_else(|| anyhow!("{name}: quantized tensor name must end with .weight"))?;
        let glm_scale = format!("{base}.weight_scale_inv");
        let (scale_f32, scale_rows, scale_cols, block_m, block_k) =
            if let Ok(s) = self.borrow_raw_tensor(&glm_scale) {
                ensure!(
                    s.dtype == Dtype::F32,
                    "{glm_scale}: GLM block scale expected F32, got {:?}",
                    s.dtype
                );
                ensure!(s.shape.len() == 2, "{glm_scale}: expected 2D scale");
                let (sr, sc) = (s.shape[0], s.shape[1]);
                let vals: Vec<f32> = s
                    .bytes()
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                // Block dims inferred from the weight/scale shapes (GLM uses
                // [128,128]).
                let block_m = rows.div_ceil(sr);
                let block_k = cols.div_ceil(sc);
                (vals, sr, sc, block_m, block_k)
            } else {
                let dsv4_scale = format!("{base}.scale");
                let s = self.borrow_raw_tensor(&dsv4_scale)?;
                ensure!(s.shape.len() == 2, "{dsv4_scale}: expected 2D scale");
                let (sr, sc) = (s.shape[0], s.shape[1]);
                // E8M0 bytes or F32 power-of-two, normalized to E8M0 → scale =
                // 2^(byte-127).
                let e8m0 = Self::dsv4_block_scale_e8m0(&dsv4_scale, s.dtype, s.bytes())?;
                let vals: Vec<f32> = e8m0.iter().map(|&b| 2.0f32.powi(b as i32 - 127)).collect();
                let block_m = rows.div_ceil(sr);
                let block_k = cols.div_ceil(sc);
                (vals, sr, sc, block_m, block_k)
            };
        ensure!(
            scale_f32.len() == scale_rows * scale_cols,
            "{name}: block-scale element count {} != {scale_rows}*{scale_cols}",
            scale_f32.len()
        );
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let sr = r / block_m;
            for c in 0..cols {
                let sc = c / block_k;
                let scale = scale_f32[sr.min(scale_rows - 1) * scale_cols + sc.min(scale_cols - 1)];
                out[r * cols + c] = decode_f8_e4m3fn(weight[r * cols + c]) * scale;
            }
        }
        Ok(out)
    }

    /// DeepGEMM repack of one decode projection weight, or `None`. Built only when the
    /// decode-DeepGEMM alloc gate is on AND the weight is raw FP8 block-scaled — the
    /// GLM
    /// dialect dequantizes to bf16, so the FP8 check alone excludes it.
    fn decode_proj_cache(
        &self,
        ctx: &DeviceContext,
        weight: &DeviceMatrix,
    ) -> Result<Option<cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache>> {
        if crate::attention::dsv4_fused_wqkv_decode_enabled()?
            && weight.weight_format == WeightFormat::Dsv4Fp8BlockScaled
        {
            Ok(Some(
                cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache::from_dsv4_weight(ctx, weight)?,
            ))
        } else {
            Ok(None)
        }
    }

    /// Load one compressor sub-block (`wkv`/`wgate`/`ape` matrices + `norm` vec). GLM's
    /// F32
    /// `weight_scale_inv` scales mean its compressor projections are dequantized to
    /// bf16.
    pub(crate) fn load_dsv4_compressor(
        &self,
        ctx: &DeviceContext,
        names: &deepseek_spec::DeepSeekV4CompressorTensorNames,
        glm: bool,
    ) -> Result<crate::dsv4::Dsv4Compressor> {
        let load_matrix = |name: &str| {
            if glm {
                self.load_dsv4_block_scaled_dialect(ctx, name)
            } else {
                self.load_dsv4_global_matrix(ctx, name)
            }
        };
        let wkv = load_matrix(&names.wkv)?;
        let wgate = load_matrix(&names.wgate)?;
        let fp32_probe = {
            let tensor = self.borrow_raw_tensor(&names.ape)?;
            ensure!(
                tensor.shape.len() == 2,
                "{}: expected 2D compressor APE, got {:?}",
                names.ape,
                tensor.shape
            );
            let values = match tensor.dtype {
                Dtype::F32 | Dtype::BF16 => tensor_bytes_to_f32(
                    &names.ape,
                    tensor.dtype,
                    tensor.bytes(),
                    ScaleApply::Multiply,
                )?,
                Dtype::F8_E4M3 => self.dequantize_dsv4_block_scaled_to_f32_host(
                    &names.ape,
                    tensor.shape[0],
                    tensor.shape[1],
                )?,
                other => bail!("{}: unsupported compressor APE dtype {other:?}", names.ape),
            };
            crate::dsv4::Dsv4CompressorFp32Probe {
                wkv: self.load_dsv4_bf16_matrix(ctx, &names.wkv)?,
                wgate: self.load_dsv4_bf16_matrix(ctx, &names.wgate)?,
                ape: ctx
                    .stream
                    .clone_htod(&values)
                    .map_err(|e| anyhow!("upload f32 compressor APE {}: {e}", names.ape))?,
            }
        };
        Ok(crate::dsv4::Dsv4Compressor {
            wkv_deepgemm: self.decode_proj_cache(ctx, &wkv)?,
            wgate_deepgemm: self.decode_proj_cache(ctx, &wgate)?,
            // `ape` is read RAW as bf16 by the compressor kernel, not through
            // `dsv4_linear`, so
            // it must be dense bf16 (#138).
            ape: self.load_dsv4_block_scaled_dialect(ctx, &names.ape)?,
            fp32_probe,
            norm: self.load_dsv4_vec(ctx, &names.norm)?,
            wkv,
            wgate,
        })
    }

    /// Load one indexer sub-block. DSv4 CSA: `wq_b`/`weights_proj` + a key
    /// compressor. GLM DSA (`SparseIndexed`): `wq_b`/`weights_proj` + `wk` key
    /// projection + `k_norm` (weight+bias), no compressor.
    pub(crate) fn load_dsv4_indexer(
        &self,
        ctx: &DeviceContext,
        config: &deepseek_spec::DeepSeekV4Config,
        names: &deepseek_spec::DeepSeekV4IndexerTensorNames,
    ) -> Result<crate::dsv4::Dsv4Indexer> {
        let glm = config.plain_o_proj;
        // GLM indexer wq_b is FP8 + `weight_scale_inv`; dequant to dense bf16.
        let wq_b = if glm {
            self.load_dsv4_block_scaled_dialect(ctx, &names.wq_b)?
        } else {
            self.load_dsv4_global_matrix(ctx, &names.wq_b)?
        };
        let wq_b_deepgemm = self.decode_proj_cache(ctx, &wq_b)?;
        let weights_proj = self.load_dsv4_global_matrix(ctx, &names.weights_proj)?;
        let weights_proj_deepgemm = self.decode_proj_cache(ctx, &weights_proj)?;
        let (compressor, wk, k_norm) = if glm {
            let wk = match names.wk.as_ref() {
                Some(n) => Some(self.load_dsv4_block_scaled_dialect(ctx, n)?),
                None => None,
            };
            let k_norm = match names.k_norm.as_ref() {
                Some(n) => Some(self.load_dsv4_vec(ctx, n)?),
                None => None,
            };
            (None, wk, k_norm)
        } else {
            let compressor = self.load_dsv4_compressor(
                ctx,
                names
                    .compressor
                    .as_ref()
                    .expect("DSv4 indexer always has a compressor"),
                false,
            )?;
            (Some(compressor), None, None)
        };
        Ok(crate::dsv4::Dsv4Indexer {
            wq_b,
            weights_proj,
            compressor,
            wk,
            k_norm,
            wq_b_deepgemm,
            weights_proj_deepgemm,
        })
    }

    /// Load a block-scaled FP8 matrix choosing the scale dialect by sibling: GLM
    /// `<base>.weight_scale_inv` (F32) ⇒ dequantize to a dense bf16 [`DeviceMatrix`]
    /// (the F32
    /// block scale cannot ride the E8M0 path losslessly); DSv4 `<base>.scale` (E8M0) ⇒
    /// the
    /// FP8 block-scaled path. BF16/F32 on disk falls through to the dense loader.
    pub(crate) fn load_dsv4_block_scaled_dialect(
        &self,
        ctx: &DeviceContext,
        name: &str,
    ) -> Result<DeviceMatrix> {
        let tensor = self.borrow_raw_tensor(name)?;
        if tensor.dtype == Dtype::BF16 || tensor.dtype == Dtype::F32 {
            return self.load_dsv4_bf16_matrix(ctx, name);
        }
        let base = name
            .strip_suffix(".weight")
            .ok_or_else(|| anyhow!("{name}: quantized tensor name must end with .weight"))?;
        // Dequant when a block scale is present in EITHER dialect (#138).
        let has_scale = self
            .borrow_raw_tensor(&format!("{base}.weight_scale_inv"))
            .is_ok()
            || self.borrow_raw_tensor(&format!("{base}.scale")).is_ok();
        if has_scale {
            ensure!(
                tensor.shape.len() == 2,
                "{name}: expected 2D quantized tensor, got {:?}",
                tensor.shape
            );
            let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
            let f32 = self.dequantize_dsv4_block_scaled_to_f32_host(name, rows, cols)?;
            let bytes: Vec<u8> = f32
                .iter()
                .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                .collect();
            DeviceMatrix::from_safetensors(ctx, &bytes, rows, cols)
                .with_context(|| format!("upload dequant bf16 {name}"))
        } else {
            self.load_dsv4_block_scaled(ctx, name)
        }
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

/// Dequantize one FP8-block-scaled routed expert to dense BF16 in place. Runs at load
/// for the
/// whole layer because grouped MoE dispatches one kernel per layer off a static ptr
/// table,
/// so the per-expert lazy promote cannot apply.
fn dequantize_fp8_expert_to_bf16_in_place(
    ctx: &DeviceContext,
    matrix: &mut DeviceMatrix,
) -> Result<()> {
    if matrix.weight_format == WeightFormat::DenseBf16 {
        return Ok(());
    }
    ensure!(
        matrix.weight_format == WeightFormat::Fp8BlockScaled
            && matrix.quant_block_m > 0
            && matrix.quant_block_k > 0
            && matrix.quant_scale_rows > 0
            && matrix.quant_scale_cols > 0,
        "BF16-resident expert dequant needs FP8 block-scaled metadata; got {:?}",
        matrix.weight_format
    );
    let mut dense = ctx
        .stream
        .alloc_zeros::<half::bf16>(matrix.rows * matrix.cols)
        .map_err(|e| anyhow!("expert BF16 dequant alloc failed: {e}"))?;
    {
        let qweight = matrix
            .qweight_u8
            .as_ref()
            .ok_or_else(|| anyhow!("FP8 expert missing qweight"))?;
        let scales = matrix
            .scale_f32
            .as_ref()
            .ok_or_else(|| anyhow!("FP8 expert missing f32 scales"))?;
        ensure!(
            qweight.len() == matrix.rows * matrix.cols,
            "FP8 expert qweight len {} != rows*cols {}",
            qweight.len(),
            matrix.rows * matrix.cols
        );
        let (qw_ptr, _gq) = qweight.device_ptr(&ctx.stream);
        let (scale_ptr, _gs) = scales.device_ptr(&ctx.stream);
        let (dense_ptr, _gd) = dense.device_ptr_mut(&ctx.stream);
        // SAFETY: ptrs from live device allocations sized to the dims passed.
        unsafe {
            cuda_kernels::ffi::dequantize_fp8_block_scaled_to_bf16_cuda(
                qw_ptr as *const u8,
                scale_ptr as *const f32,
                dense_ptr as *mut cuda_kernels::ffi::Half,
                matrix.rows as i32,
                matrix.cols as i32,
                matrix.quant_scale_rows as i32,
                matrix.quant_scale_cols as i32,
                matrix.quant_block_m as i32,
                matrix.quant_block_k as i32,
                ctx.stream.cu_stream(),
            )
        }
        .result()
        .map_err(|e| anyhow!("FP8->BF16 expert dequant failed: {e}"))?;
    }
    // Event tracking is disabled: sync before dropping the FP8 source so the
    // async dequant kernel has finished reading it (mirrors the grouped path).
    ctx.sync()?;
    matrix.data = dense;
    matrix.weight_format = WeightFormat::DenseBf16;
    matrix.qweight_u8 = None;
    matrix.scale_f32 = None;
    matrix.quant_scale_rows = 0;
    matrix.quant_scale_cols = 0;
    matrix.quant_block_m = 0;
    matrix.quant_block_k = 0;
    Ok(())
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
                | WeightFormat::W4A16
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

/// This EP rank's loaded MoE weights for one sparse layer. Built by
/// [`SafetensorLoader::load_moe_layer_experts`], consumed by
/// [`crate::moe::moe_forward`].
pub(crate) struct MoeLayerWeights {
    /// Per-expert weight matrices (hand grouped-GEMM path). EMPTY when the grouped
    /// caches
    /// below are built — the grouped buffer then owns the only copy of the bytes and
    /// the
    /// `*_ptrs` tables point into it, so the hand kernels stay runnable.
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
    /// DeepGEMM grouped-B caches (`[groups, n, k]` contiguous row-major BF16, this
    /// rank's EP
    /// experts only).
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

pub(crate) struct MoeLayerHostSnapshot {
    gate: Vec<HostMatrixSnapshot>,
    up: Vec<HostMatrixSnapshot>,
    down: Vec<HostMatrixSnapshot>,
    gate_grouped: Option<MoeExpertGroupHostSnapshot>,
    up_grouped: Option<MoeExpertGroupHostSnapshot>,
    down_grouped: Option<MoeExpertGroupHostSnapshot>,
    w13_fp8_grouped: Option<MoeFp8ExpertGroupHostSnapshot>,
    down_fp8_grouped: Option<MoeFp8ExpertGroupHostSnapshot>,
    router_gate: HostMatrixSnapshot,
    shared_gate: HostMatrixSnapshot,
    shared_up: HostMatrixSnapshot,
    shared_down: HostMatrixSnapshot,
    shared_gate_router: HostMatrixSnapshot,
    freed_bytes: usize,
}

impl MoeLayerHostSnapshot {
    #[must_use]
    pub(crate) fn freed_bytes(&self) -> usize {
        self.freed_bytes
    }
}

struct MoeExpertGroupHostSnapshot {
    data: Vec<half::bf16>,
    groups: usize,
    rows: usize,
    cols: usize,
}

struct MoeFp8ExpertGroupHostSnapshot {
    weight: Vec<u8>,
    scales: Vec<f32>,
    groups: usize,
    rows: usize,
    cols: usize,
    sfb_n_contiguous: bool,
}

impl MoeLayerWeights {
    pub(crate) fn offload_to_host(&mut self, ctx: &DeviceContext) -> Result<MoeLayerHostSnapshot> {
        let mut freed = 0usize;
        let gate = offload_matrix_vec(ctx, &mut self.gate, "moe.gate", &mut freed)?;
        let up = offload_matrix_vec(ctx, &mut self.up, "moe.up", &mut freed)?;
        let down = offload_matrix_vec(ctx, &mut self.down, "moe.down", &mut freed)?;
        let gate_grouped =
            offload_group_opt(ctx, &mut self.gate_grouped, "moe.gate_grouped", &mut freed)?;
        let up_grouped =
            offload_group_opt(ctx, &mut self.up_grouped, "moe.up_grouped", &mut freed)?;
        let down_grouped =
            offload_group_opt(ctx, &mut self.down_grouped, "moe.down_grouped", &mut freed)?;
        let w13_fp8_grouped = offload_fp8_group_opt(
            ctx,
            &mut self.w13_fp8_grouped,
            "moe.w13_fp8_grouped",
            &mut freed,
        )?;
        let down_fp8_grouped = offload_fp8_group_opt(
            ctx,
            &mut self.down_fp8_grouped,
            "moe.down_fp8_grouped",
            &mut freed,
        )?;
        let router_gate = self.router_gate.offload_to_host(ctx)?;
        freed += router_gate.freed_bytes();
        let shared_gate = self.shared_gate.offload_to_host(ctx)?;
        freed += shared_gate.freed_bytes();
        let shared_up = self.shared_up.offload_to_host(ctx)?;
        freed += shared_up.freed_bytes();
        let shared_down = self.shared_down.offload_to_host(ctx)?;
        freed += shared_down.freed_bytes();
        let shared_gate_router = self.shared_gate_router.offload_to_host(ctx)?;
        freed += shared_gate_router.freed_bytes();

        Ok(MoeLayerHostSnapshot {
            gate,
            up,
            down,
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
            freed_bytes: freed,
        })
    }

    pub(crate) fn reload_from_host(
        &mut self,
        ctx: &DeviceContext,
        snapshot: &MoeLayerHostSnapshot,
    ) -> Result<()> {
        reload_matrix_vec(ctx, &mut self.gate, &snapshot.gate, "moe.gate")?;
        reload_matrix_vec(ctx, &mut self.up, &snapshot.up, "moe.up")?;
        reload_matrix_vec(ctx, &mut self.down, &snapshot.down, "moe.down")?;
        self.gate_grouped = reload_group_opt(ctx, &snapshot.gate_grouped, "moe.gate_grouped")?;
        self.up_grouped = reload_group_opt(ctx, &snapshot.up_grouped, "moe.up_grouped")?;
        self.down_grouped = reload_group_opt(ctx, &snapshot.down_grouped, "moe.down_grouped")?;
        self.w13_fp8_grouped =
            reload_fp8_group_opt(ctx, &snapshot.w13_fp8_grouped, "moe.w13_fp8_grouped")?;
        self.down_fp8_grouped =
            reload_fp8_group_opt(ctx, &snapshot.down_fp8_grouped, "moe.down_fp8_grouped")?;
        self.router_gate
            .reload_from_host(ctx, &snapshot.router_gate)?;
        self.shared_gate
            .reload_from_host(ctx, &snapshot.shared_gate)?;
        self.shared_up.reload_from_host(ctx, &snapshot.shared_up)?;
        self.shared_down
            .reload_from_host(ctx, &snapshot.shared_down)?;
        self.shared_gate_router
            .reload_from_host(ctx, &snapshot.shared_gate_router)?;
        self.rebuild_pointer_tables(ctx)
    }

    fn rebuild_pointer_tables(&mut self, ctx: &DeviceContext) -> Result<()> {
        let ptr_tables = build_moe_layer_pointer_tables(
            ctx,
            self.expert_weight_format,
            &self.gate,
            &self.up,
            &self.down,
            self.gate_grouped.as_ref(),
            self.up_grouped.as_ref(),
            self.down_grouped.as_ref(),
            self.w13_fp8_grouped.as_ref(),
            self.down_fp8_grouped.as_ref(),
        )?;
        self.gate_ptrs = ptr_tables.gate_ptrs;
        self.up_ptrs = ptr_tables.up_ptrs;
        self.down_ptrs = ptr_tables.down_ptrs;
        self.gate_scale_ptrs = ptr_tables.gate_scale_ptrs;
        self.up_scale_ptrs = ptr_tables.up_scale_ptrs;
        self.down_scale_ptrs = ptr_tables.down_scale_ptrs;
        self.gate_global_ptrs = ptr_tables.gate_global_ptrs;
        self.up_global_ptrs = ptr_tables.up_global_ptrs;
        self.down_global_ptrs = ptr_tables.down_global_ptrs;
        Ok(())
    }
}

struct MoeLayerPointerTables {
    gate_ptrs: CudaSlice<u64>,
    up_ptrs: CudaSlice<u64>,
    down_ptrs: CudaSlice<u64>,
    gate_scale_ptrs: Option<CudaSlice<u64>>,
    up_scale_ptrs: Option<CudaSlice<u64>>,
    down_scale_ptrs: Option<CudaSlice<u64>>,
    gate_global_ptrs: Option<CudaSlice<u64>>,
    up_global_ptrs: Option<CudaSlice<u64>>,
    down_global_ptrs: Option<CudaSlice<u64>>,
}

struct MoeExpertRefs<'a> {
    gate: Vec<&'a DeviceMatrix>,
    up: Vec<&'a DeviceMatrix>,
    down: Vec<&'a DeviceMatrix>,
}

fn moe_expert_refs<'a>(
    gate: &'a [DeviceMatrix],
    up: &'a [DeviceMatrix],
    down: &'a [DeviceMatrix],
) -> Result<MoeExpertRefs<'a>> {
    ensure!(
        !gate.is_empty() && gate.len() == up.len() && gate.len() == down.len(),
        "MoE pointer-table rebuild requires matching non-empty per-expert matrices: gate={} up={} down={}",
        gate.len(),
        up.len(),
        down.len()
    );
    Ok(MoeExpertRefs {
        gate: gate.iter().collect(),
        up: up.iter().collect(),
        down: down.iter().collect(),
    })
}

#[allow(clippy::too_many_arguments)]
fn build_moe_layer_pointer_tables(
    ctx: &DeviceContext,
    expert_weight_format: WeightFormat,
    gate: &[DeviceMatrix],
    up: &[DeviceMatrix],
    down: &[DeviceMatrix],
    gate_grouped: Option<&MoeExpertGroup>,
    up_grouped: Option<&MoeExpertGroup>,
    down_grouped: Option<&MoeExpertGroup>,
    w13_fp8_grouped: Option<&MoeFp8ExpertGroup>,
    down_fp8_grouped: Option<&MoeFp8ExpertGroup>,
) -> Result<MoeLayerPointerTables> {
    let routed_quant = expert_weight_format.is_quantized();
    let bf16_grouped = match (gate_grouped, up_grouped, down_grouped) {
        (Some(g), Some(u), Some(d)) => Some((g, u, d)),
        (None, None, None) => None,
        _ => bail!("MoE pointer-table rebuild found partial BF16 grouped cache"),
    };
    let fp8_grouped = match (w13_fp8_grouped, down_fp8_grouped) {
        (Some(w13), Some(down_g)) => Some((w13, down_g)),
        (None, None) => None,
        _ => bail!("MoE pointer-table rebuild found partial FP8 grouped cache"),
    };
    ensure!(
        bf16_grouped.is_none() || fp8_grouped.is_none(),
        "MoE pointer-table rebuild cannot use both BF16 and FP8 grouped caches"
    );

    let (gate_ptrs, up_ptrs, down_ptrs) = if let Some((g, u, d)) = bf16_grouped {
        (g.ptr_table(ctx)?, u.ptr_table(ctx)?, d.ptr_table(ctx)?)
    } else if let Some((w13, down_g)) = fp8_grouped {
        let up_offset = w13.rows / 2;
        (
            w13.qweight_ptr_table(ctx, 0)?,
            w13.qweight_ptr_table(ctx, up_offset)?,
            down_g.qweight_ptr_table(ctx, 0)?,
        )
    } else {
        let refs = moe_expert_refs(gate, up, down)?;
        if routed_quant {
            // W4A16 packs INT4 nibbles into `i8` qweight; FP8/FP4 use `u8`.
            if expert_weight_format == WeightFormat::W4A16 {
                (
                    cuda_kernels::moe::build_expert_qweight_i8_ptr_table(ctx, &refs.gate)?,
                    cuda_kernels::moe::build_expert_qweight_i8_ptr_table(ctx, &refs.up)?,
                    cuda_kernels::moe::build_expert_qweight_i8_ptr_table(ctx, &refs.down)?,
                )
            } else {
                (
                    cuda_kernels::moe::build_expert_qweight_u8_ptr_table(ctx, &refs.gate)?,
                    cuda_kernels::moe::build_expert_qweight_u8_ptr_table(ctx, &refs.up)?,
                    cuda_kernels::moe::build_expert_qweight_u8_ptr_table(ctx, &refs.down)?,
                )
            }
        } else {
            (
                cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &refs.gate)?,
                cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &refs.up)?,
                cuda_kernels::moe::build_expert_weight_ptr_table(ctx, &refs.down)?,
            )
        }
    };

    let (gate_scale_ptrs, up_scale_ptrs, down_scale_ptrs) = if let Some((w13, down_g)) = fp8_grouped
    {
        let up_offset = w13.rows / 2;
        (
            Some(w13.scale_ptr_table(ctx, 0)?),
            Some(w13.scale_ptr_table(ctx, up_offset)?),
            Some(down_g.scale_ptr_table(ctx, 0)?),
        )
    } else if routed_quant
        && matches!(
            expert_weight_format,
            WeightFormat::Fp8BlockScaled | WeightFormat::Fp8PerShard
        )
    {
        let refs = moe_expert_refs(gate, up, down)?;
        (
            Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                ctx, &refs.gate,
            )?),
            Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                ctx, &refs.up,
            )?),
            Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                ctx, &refs.down,
            )?),
        )
    } else if routed_quant && expert_weight_format == WeightFormat::Fp4E2M1Group {
        let refs = moe_expert_refs(gate, up, down)?;
        (
            Some(cuda_kernels::moe::build_expert_qscale_fp8_ptr_table(
                ctx, &refs.gate,
            )?),
            Some(cuda_kernels::moe::build_expert_qscale_fp8_ptr_table(
                ctx, &refs.up,
            )?),
            Some(cuda_kernels::moe::build_expert_qscale_fp8_ptr_table(
                ctx, &refs.down,
            )?),
        )
    } else if routed_quant && expert_weight_format == WeightFormat::W4A16 {
        let refs = moe_expert_refs(gate, up, down)?;
        (
            Some(cuda_kernels::moe::build_expert_qscale_bf16_ptr_table(
                ctx, &refs.gate,
            )?),
            Some(cuda_kernels::moe::build_expert_qscale_bf16_ptr_table(
                ctx, &refs.up,
            )?),
            Some(cuda_kernels::moe::build_expert_qscale_bf16_ptr_table(
                ctx, &refs.down,
            )?),
        )
    } else {
        (None, None, None)
    };

    let (gate_global_ptrs, up_global_ptrs, down_global_ptrs) =
        if routed_quant && expert_weight_format == WeightFormat::Fp4E2M1Group {
            let refs = moe_expert_refs(gate, up, down)?;
            (
                Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                    ctx, &refs.gate,
                )?),
                Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                    ctx, &refs.up,
                )?),
                Some(cuda_kernels::moe::build_expert_scale_f32_ptr_table(
                    ctx, &refs.down,
                )?),
            )
        } else {
            (None, None, None)
        };

    Ok(MoeLayerPointerTables {
        gate_ptrs,
        up_ptrs,
        down_ptrs,
        gate_scale_ptrs,
        up_scale_ptrs,
        down_scale_ptrs,
        gate_global_ptrs,
        up_global_ptrs,
        down_global_ptrs,
    })
}

fn offload_matrix_vec(
    ctx: &DeviceContext,
    matrices: &mut [DeviceMatrix],
    label: &str,
    freed: &mut usize,
) -> Result<Vec<HostMatrixSnapshot>> {
    matrices
        .iter_mut()
        .enumerate()
        .map(|(idx, matrix)| {
            let snapshot = matrix
                .offload_to_host(ctx)
                .with_context(|| format!("offload {label}[{idx}]"))?;
            *freed += snapshot.freed_bytes();
            Ok(snapshot)
        })
        .collect()
}

fn reload_matrix_vec(
    ctx: &DeviceContext,
    matrices: &mut [DeviceMatrix],
    snapshots: &[HostMatrixSnapshot],
    label: &str,
) -> Result<()> {
    ensure!(
        matrices.len() == snapshots.len(),
        "reload {label}: matrix count {} != snapshot count {}",
        matrices.len(),
        snapshots.len()
    );
    for (idx, (matrix, snapshot)) in matrices.iter_mut().zip(snapshots).enumerate() {
        matrix
            .reload_from_host(ctx, snapshot)
            .with_context(|| format!("reload {label}[{idx}]"))?;
    }
    Ok(())
}

fn offload_group_opt(
    ctx: &DeviceContext,
    group: &mut Option<MoeExpertGroup>,
    label: &str,
    freed: &mut usize,
) -> Result<Option<MoeExpertGroupHostSnapshot>> {
    match group {
        Some(group) => {
            let (snapshot, bytes) = group
                .offload_to_host(ctx)
                .with_context(|| format!("offload {label}"))?;
            *freed += bytes;
            Ok(Some(snapshot))
        }
        None => Ok(None),
    }
}

fn reload_group_opt(
    ctx: &DeviceContext,
    snapshot: &Option<MoeExpertGroupHostSnapshot>,
    label: &str,
) -> Result<Option<MoeExpertGroup>> {
    snapshot
        .as_ref()
        .map(|snapshot| {
            MoeExpertGroup::from_host(ctx, snapshot).with_context(|| format!("reload {label}"))
        })
        .transpose()
}

fn offload_fp8_group_opt(
    ctx: &DeviceContext,
    group: &mut Option<MoeFp8ExpertGroup>,
    label: &str,
    freed: &mut usize,
) -> Result<Option<MoeFp8ExpertGroupHostSnapshot>> {
    match group {
        Some(group) => {
            let (snapshot, bytes) = group
                .offload_to_host(ctx)
                .with_context(|| format!("offload {label}"))?;
            *freed += bytes;
            Ok(Some(snapshot))
        }
        None => Ok(None),
    }
}

fn reload_fp8_group_opt(
    ctx: &DeviceContext,
    snapshot: &Option<MoeFp8ExpertGroupHostSnapshot>,
    label: &str,
) -> Result<Option<MoeFp8ExpertGroup>> {
    snapshot
        .as_ref()
        .map(|snapshot| {
            MoeFp8ExpertGroup::from_host(
                ctx,
                &snapshot.weight,
                &snapshot.scales,
                snapshot.groups,
                snapshot.rows,
                snapshot.cols,
                snapshot.sfb_n_contiguous,
            )
            .with_context(|| format!("reload {label}"))
        })
        .transpose()
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
    fn from_host(ctx: &DeviceContext, snapshot: &MoeExpertGroupHostSnapshot) -> Result<Self> {
        ensure!(
            snapshot.groups > 0,
            "MoE expert group: groups must be non-zero"
        );
        ensure!(
            snapshot.data.len() == snapshot.groups * snapshot.rows * snapshot.cols,
            "MoE grouped host data len {} != expected {}",
            snapshot.data.len(),
            snapshot.groups * snapshot.rows * snapshot.cols
        );
        Ok(Self {
            data: ctx
                .stream
                .clone_htod(snapshot.data.as_slice())
                .map_err(|e| anyhow!("MoE expert group H2D failed: {e}"))?,
            groups: snapshot.groups,
            rows: snapshot.rows,
            cols: snapshot.cols,
        })
    }

    fn offload_to_host(
        &mut self,
        ctx: &DeviceContext,
    ) -> Result<(MoeExpertGroupHostSnapshot, usize)> {
        let data = ctx
            .stream
            .clone_dtoh(&self.data)
            .map_err(|e| anyhow!("MoE expert group D2H failed: {e}"))?;
        let freed = data.len() * std::mem::size_of::<half::bf16>();
        ctx.sync()?;
        self.data = ctx
            .stream
            .alloc_zeros::<half::bf16>(1)
            .map_err(|e| anyhow!("MoE expert group placeholder alloc failed: {e}"))?;
        Ok((
            MoeExpertGroupHostSnapshot {
                data,
                groups: self.groups,
                rows: self.rows,
                cols: self.cols,
            },
            freed,
        ))
    }

    /// Concatenate per-expert `[rows, cols]` matrices into one contiguous group-major
    /// buffer
    /// (D2D). The source matrices may be dropped afterwards **only after a stream
    /// sync** —
    /// event tracking is disabled, so a Rust drop frees device memory immediately.
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

    /// Device table of per-group base pointers in the same `*const u64` format as
    /// [`cuda_kernels::moe::build_expert_weight_ptr_table`], so the hand kernels run
    /// unchanged.
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
    /// SFB scale layout, decided once at load: `true` = N-contiguous per group (CUTLASS
    /// sm_120a), `false` = the checkpoint's K-contiguous packing (Hopper DeepGEMM). The
    /// executor dispatches on this field and never re-derives the SM, so loader and
    /// executor
    /// cannot silently disagree on the operand layout.
    pub(crate) sfb_n_contiguous: bool,
}

impl MoeFp8ExpertGroup {
    fn from_host(
        ctx: &DeviceContext,
        weight: &[u8],
        scales: &[f32],
        groups: usize,
        rows: usize,
        cols: usize,
        sfb_n_contiguous: bool,
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
            sfb_n_contiguous,
        })
    }

    fn offload_to_host(
        &mut self,
        ctx: &DeviceContext,
    ) -> Result<(MoeFp8ExpertGroupHostSnapshot, usize)> {
        let weight = ctx
            .stream
            .clone_dtoh(&self.weight)
            .map_err(|e| anyhow!("FP8 MoE grouped weight D2H failed: {e}"))?;
        let scales = ctx
            .stream
            .clone_dtoh(&self.scales)
            .map_err(|e| anyhow!("FP8 MoE grouped scales D2H failed: {e}"))?;
        let freed =
            weight.len() * std::mem::size_of::<u8>() + scales.len() * std::mem::size_of::<f32>();
        ctx.sync()?;
        self.weight = ctx
            .stream
            .alloc_zeros::<u8>(1)
            .map_err(|e| anyhow!("FP8 MoE grouped weight placeholder alloc failed: {e}"))?;
        self.scales = ctx
            .stream
            .alloc_zeros::<f32>(1)
            .map_err(|e| anyhow!("FP8 MoE grouped scales placeholder alloc failed: {e}"))?;
        Ok((
            MoeFp8ExpertGroupHostSnapshot {
                weight,
                scales,
                groups: self.groups,
                rows: self.rows,
                cols: self.cols,
                sfb_n_contiguous: self.sfb_n_contiguous,
            },
            freed,
        ))
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
            // `empty` backs the Hopper DeepGEMM concat path — K-contiguous SFB.
            sfb_n_contiguous: false,
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

    /// Device weight+scale pointers for the `[num_rows, cols]` sub-matrix of expert
    /// `group`
    /// starting at `row_offset` rows, letting a borrower alias an expert's FP8 slice
    /// zero-copy. `row_offset` and `num_rows` must be 128-aligned (block grid). Returns
    /// (weight_ptr, scale_ptr, rows, cols, block_m=128, block_k=128).
    pub(crate) fn expert_slice_fp8_ptrs(
        &self,
        ctx: &DeviceContext,
        group: usize,
        row_offset: usize,
        num_rows: usize,
    ) -> Option<(u64, u64, usize, usize, usize, usize)> {
        if group >= self.groups || !row_offset.is_multiple_of(128) || !num_rows.is_multiple_of(128)
        {
            return None;
        }
        if row_offset + num_rows > self.rows {
            return None;
        }
        let (wbase, _wg) = self.weight.device_ptr(&ctx.stream);
        let (sbase, _sg) = self.scales.device_ptr(&ctx.stream);
        let group_stride = self.rows * self.cols;
        let weight_ptr = wbase + (group * group_stride + row_offset * self.cols) as u64;
        let scale_group_stride = self.scale_rows * self.scale_cols;
        let scale_elem_size = std::mem::size_of::<f32>() as u64;
        let scale_ptr = sbase
            + ((group * scale_group_stride + (row_offset / 128) * self.scale_cols) as u64
                * scale_elem_size);
        Some((weight_ptr, scale_ptr, num_rows, self.cols, 128, 128))
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
