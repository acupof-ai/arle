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
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, PagedKVPool};
use cudarc::driver::CudaSlice;
use infer_seam::ShardSpec;
use infer_topo::ShardingSpec;
use safetensors::{SafeTensors, tensor::Dtype};

// Re-exports keep `crate::loader::` paths stable for consumers of the moved
// Qwen MoE upload machinery (now in `crate::qwen35::load`).
use crate::ops::{
    fp4_deepgemm_available, fp8_deepgemm_per_channel_available, upload_i32,
    validate_quant_linear_storage,
};
use crate::quant_format::{
    QuantFormat, QuantManifest, QuantTensorView, ScaleApply, TensorHeader, detect_quant_format,
    read_quant_manifest, reject_dsv4_e8m0_scale_abi,
};
pub(crate) use crate::qwen35::load::{
    ExpertQuantDispatchSignature, MoeFp8ExpertGroup, MoeLayerHostSnapshot, MoeLayerWeights,
};

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

#[derive(Debug)]
pub(crate) struct PageMeta {
    pub(crate) q_indptr: CudaSlice<i32>,
    pub(crate) kv_indptr: CudaSlice<i32>,
    pub(crate) kv_indices: CudaSlice<i32>,
    pub(crate) kv_last_page_len: CudaSlice<i32>,
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
    /// When set, the FA3 launch uses this as `seqlen_k` instead of the step's true max
    /// KV
    /// length — required under CUDA graph capture, where the true length is only on
    /// device.
    pub(crate) seqlen_k_capture: Option<usize>,
    /// 2D decode: 1 if this rank owns the new token's page (writes KV), 0 if it
    /// skips. Computed once in `sharded_decode_meta` from the block-cyclic
    /// ownership predicate; 1 on all non-2D paths.
    pub(crate) write_kv: i32,
}

/// The KV extent a sharded decode page table actually addresses.
///
/// Returns `(kv_lens, last_page_len, write_kv)`. FA3 sizes its split-KV work
/// from `seqlen_k` and the combine kernel then indexes the table, so the length
/// must count the pages IN the table — passing the global `total_len` against a
/// shard or a recall subset aborts in `flash_fwd_combine`. A 1-wide shard owns
/// every page, so the same arithmetic serves with and without CP.
fn local_kv_extent(
    shard: ShardSpec,
    total_len: usize,
    page_size: usize,
    local_num_pages: usize,
) -> (usize, usize, i32) {
    let global_pages = total_len.div_ceil(page_size);
    let owns_last = global_pages > 0 && shard.owns_page(global_pages - 1);
    let overshoot = global_pages * page_size - total_len;
    let last_page_len = if owns_last {
        page_size - overshoot
    } else {
        page_size
    };
    let kv_lens = local_num_pages.saturating_sub(1) * page_size
        + if local_num_pages == 0 {
            0
        } else {
            last_page_len
        };
    (kv_lens, last_page_len, i32::from(owns_last))
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
        Self::for_rows_impl(ctx, pool, rows)
    }

    fn for_rows_impl(
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
        // Quant formats address the pool by global token row; BF16 carries None.
        let quant = matches!(pool.format, KVFormat::INT8 | KVFormat::FP8E4M3);
        let new_token_rows = if quant {
            // Row lists concatenate in row order — the quantize kernel indexes them by
            // flat Q-token position.
            let mut new_rows = Vec::new();
            for &(slot, start_pos, len) in rows {
                new_rows.extend(
                    pool.token_rows_for_range(slot, start_pos, len)
                        .into_iter()
                        .map(|row| row as i32),
                );
            }
            Some(upload_i32(ctx, &new_rows)?)
        } else {
            None
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
            seqlen_k_capture: None,
            write_kv: 1,
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

    /// 2D ring prefill: this rank's q-slice attends via the ring pass, which
    /// never reads the paged page table. The device page-table buffers are
    /// 1-element dummies; only `total_q`/`seq_len` (== this rank's q rows) and
    /// `start_pos` are live.
    pub(crate) fn for_ring_prefill(
        ctx: &DeviceContext,
        start_pos: usize,
        rows: usize,
    ) -> Result<Self> {
        let zero = upload_i32(ctx, &[0])?;
        Ok(Self {
            q_indptr: upload_i32(ctx, &[0, rows as i32])?,
            kv_indptr: upload_i32(ctx, &[0, 0])?,
            kv_indices: zero.clone(),
            kv_last_page_len: zero.clone(),
            start_positions: upload_i32(ctx, &[start_pos as i32])?,
            positions: upload_i32(ctx, &[(start_pos + rows.saturating_sub(1)) as i32])?,
            q_offsets: vec![0, rows],
            page_offsets: vec![0, 0],
            kv_lens: vec![0],
            kv_lens_dev: zero.clone(),
            page_table_rect: zero,
            page_table_stride: 1,
            seq_len: rows,
            total_q: rows,
            num_pages: 0,
            batch: 1,
            start_pos,
            new_token_rows: None,
            seqlen_k_capture: None,
            write_kv: 0,
        })
    }

    /// Fixed-capacity single-slot decode metadata whose device buffers never move:
    /// [`Self::refresh_decode`] rewrites the CONTENTS each step, so a captured CUDA
    /// graph
    /// keeps reading the same addresses. `seqlen_k_capture` is pinned to the capacity
    /// so the
    /// FA3 scheduling ceiling is capture-stable.
    ///
    /// A quantized pool needs two more buffers, and at B=1 both are fixed-size: one
    /// pool row for the new token, and the packed `[page_indptr(2) | last_page_len]`
    /// the split-KV kernel reads.
    pub(crate) fn persistent_decode(
        ctx: &DeviceContext,
        page_size: usize,
        capacity_pages: usize,
        format: KVFormat,
    ) -> Result<Self> {
        let cap = capacity_pages.max(1);
        let quant = format != KVFormat::BF16;
        Ok(Self {
            q_indptr: upload_i32(ctx, &[0, 1])?,
            kv_indptr: upload_i32(ctx, &[0, 0])?,
            kv_indices: upload_i32(ctx, &vec![0i32; cap])?,
            kv_lens_dev: upload_i32(ctx, &[0])?,
            page_table_rect: upload_i32(ctx, &vec![0i32; cap])?,
            page_table_stride: cap,
            kv_last_page_len: upload_i32(ctx, &[0])?,
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
            new_token_rows: quant.then(|| upload_i32(ctx, &[0])).transpose()?,
            seqlen_k_capture: Some(cap * page_size),
            write_kv: 1,
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
            (pool.format == KVFormat::BF16) == self.new_token_rows.is_none(),
            "persistent decode table was built for a {} pool but got {:?}",
            if self.new_token_rows.is_none() {
                "BF16"
            } else {
                "quantized"
            },
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
        if let Some(new_rows) = self.new_token_rows.as_mut() {
            let rows = pool.token_rows_for_range(slot, start_pos, 1);
            ensure!(
                rows.len() == 1,
                "decode token-row lookup returned {} rows for slot {slot}",
                rows.len()
            );
            stream
                .memcpy_htod(&[rows[0] as i32], &mut new_rows.slice_mut(0..1))
                .map_err(|e| anyhow!("refresh new_token_rows: {e}"))?;
        }
        self.page_offsets = vec![0, num_pages];
        self.kv_lens = vec![total_len];
        self.num_pages = num_pages;
        self.start_pos = start_pos;
        Ok(())
    }

    /// Fixed-capacity page table for the 2D-sharded eager decode lane, mirroring
    /// [`Self::persistent_decode`] but sized to this rank's LOCAL page shard
    /// (`max_local_pages = ceil(max_global_pages / cp_size)`). The pool under 2D
    /// holds only the owned pages, so `page_indices` already returns the shard.
    pub(crate) fn persistent_sharded_decode(
        ctx: &DeviceContext,
        max_local_pages: usize,
    ) -> Result<Self> {
        // Same shape as persistent_decode; the sharded lane pins no FA3
        // scheduling ceiling (the 2D merge handles variable local page counts).
        let mut meta = Self::persistent_decode(ctx, 1, max_local_pages, KVFormat::BF16)?;
        meta.seqlen_k_capture = None;
        Ok(meta)
    }

    /// Rewrite a [`Self::persistent_sharded_decode`] meta for one 2D decode step —
    /// contents only, no reallocation. `write_kv` flips per step: only the owner of
    /// the last global page appends the new token.
    pub(crate) fn refresh_sharded_decode(
        &mut self,
        ctx: &DeviceContext,
        pool: &PagedKVPool,
        slot: usize,
        start_pos: usize,
        shard: ShardSpec,
    ) -> Result<()> {
        ensure!(
            pool.format == KVFormat::BF16,
            "2D decode requires the BF16 KV pool (got {:?})",
            pool.format
        );
        let total_len = start_pos + 1;
        let page_size = pool.page_size;
        ensure!(
            pool.seq_len(slot) == total_len,
            "sharded decode refresh: pool seq_len {} != total_len {total_len} for slot {slot}",
            pool.seq_len(slot)
        );
        let local_pages: Vec<i32> = pool.page_indices(slot).iter().map(|&p| p as i32).collect();
        let local_num_pages = local_pages.len();
        let cap = self.page_table_stride;
        ensure!(
            local_num_pages <= cap,
            "slot {slot} sharded decode needs {local_num_pages} pages, persistent capacity is {cap}"
        );
        let (local_token_count, local_last_fill, write_kv) =
            local_kv_extent(shard, total_len, page_size, local_num_pages);
        let stream = &ctx.stream;
        // Empty shard: upload a dummy 1-entry table so the meta's table pointer
        // stays valid; FA3 is bypassed downstream (seqlen_k=0 rejected) and -inf
        // lse zeroes this shard's cross-cp merge weight.
        let table: &[i32] = if local_num_pages == 0 {
            &[0]
        } else {
            &local_pages
        };
        stream
            .memcpy_htod(table, &mut self.kv_indices.slice_mut(0..table.len()))
            .map_err(|e| anyhow!("refresh sharded kv_indices: {e}"))?;
        stream
            .memcpy_htod(table, &mut self.page_table_rect.slice_mut(0..table.len()))
            .map_err(|e| anyhow!("refresh sharded page_table_rect: {e}"))?;
        stream
            .memcpy_htod(
                &[0, local_num_pages as i32],
                &mut self.kv_indptr.slice_mut(0..2),
            )
            .map_err(|e| anyhow!("refresh sharded kv_indptr: {e}"))?;
        stream
            .memcpy_htod(
                &[local_last_fill as i32],
                &mut self.kv_last_page_len.slice_mut(0..1),
            )
            .map_err(|e| anyhow!("refresh sharded kv_last_page_len: {e}"))?;
        stream
            .memcpy_htod(
                &[start_pos as i32],
                &mut self.start_positions.slice_mut(0..1),
            )
            .map_err(|e| anyhow!("refresh sharded start_positions: {e}"))?;
        stream
            .memcpy_htod(
                &[(total_len - 1) as i32],
                &mut self.positions.slice_mut(0..1),
            )
            .map_err(|e| anyhow!("refresh sharded positions: {e}"))?;
        stream
            .memcpy_htod(
                &[local_token_count as i32],
                &mut self.kv_lens_dev.slice_mut(0..1),
            )
            .map_err(|e| anyhow!("refresh sharded kv_lens_dev: {e}"))?;
        self.page_offsets = vec![0, local_num_pages];
        self.kv_lens = vec![local_token_count];
        self.num_pages = local_num_pages;
        self.start_pos = start_pos;
        self.write_kv = write_kv;
        Ok(())
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
    pub(crate) shards: Vec<PathBuf>,
    pub(crate) weight_map: HashMap<String, usize>,
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

pub(crate) struct ShardBytes {
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
    /// all checkpoint shards (`mmap` + `madvise(MADV_WILLNEED)`). Nonblocking —
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
    /// Pages are reclaimable, so this is a soft check to avoid thrashing
    /// when memory is already tight.
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
    ///
    /// `madvise(MADV_WILLNEED)` is asynchronous: it schedules readahead but
    /// does not wait for the pages to be resident. By the time the loader
    /// touches a shard (after all shards have been advised), the I/O has
    /// typically completed.
    fn prefetch_all_shards(&self) {
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
                        // Reuse MmapShard for the open+mmap+munmap cycle; its
                        // Drop handles munmap. madvise(MADV_WILLNEED) on the
                        // mapping populates the same page-cache pages the
                        // weight-load mmap reads.
                        match MmapShard::map(&shards[i]) {
                            Ok(shard) => {
                                let slice = shard.as_slice();
                                // SAFETY: `slice` is the live mapping of this shard for
                                // the duration of the call; madvise only hints the kernel.
                                #[cfg(target_os = "linux")]
                                unsafe {
                                    libc::madvise(
                                        slice.as_ptr() as *mut libc::c_void,
                                        slice.len(),
                                        libc::MADV_WILLNEED,
                                    );
                                }
                                bytes.fetch_add(slice.len(), Ordering::Relaxed);
                            }
                            Err(e) => {
                                log::debug!(
                                    "prefetch mmap failed for {}: {e}",
                                    shards[i].display()
                                );
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

    /// Same load, for a matrix a dense projection will consume. Routed MoE
    /// experts must NOT use this — their grouped-GEMM path reads the packed
    /// nibbles that the Marlin repack replaces.
    pub(crate) fn load_dense_matrix_quant_aware(
        &self,
        ctx: &DeviceContext,
        name: &str,
    ) -> Result<DeviceMatrix> {
        let matrix = self.load_matrix_quant_aware(ctx, name)?;
        marlin_repack_dense(ctx, name, matrix, true)
    }

    /// Same load, for the output head. Every serving lane slices a single row
    /// before this GEMM, so the prefill-only DeepGEMM arm would never fire; the
    /// one caller that does present it a whole prompt is the OPD raw-logits
    /// forward, where crossing from one arm to the other partway up the prompt
    /// length would make the distillation target a function of prompt length.
    pub(crate) fn load_output_head_quant_aware(
        &self,
        ctx: &DeviceContext,
        name: &str,
    ) -> Result<DeviceMatrix> {
        let matrix = self.load_matrix_quant_aware(ctx, name)?;
        marlin_repack_dense(ctx, name, matrix, false)
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
        // NVFP4 fuses on device like every other format, then repacks once here.
        marlin_repack_dense(ctx, parts[0].0, fused, true)
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
        let matrix = self.load_quant_or_dense_view(ctx, &view, shard)?;
        marlin_repack_dense(ctx, name, matrix, true)
    }

    /// Quant-aware twin of [`Self::load_qkv_head_sharded`]. `block_index` has the
    /// same Q-vs-replicated-KV meaning (see that method's docs).
    ///
    /// Returns the shard un-repacked: both callers row-fuse it with a same-K
    /// sibling, and [`DeviceMatrix::fuse_rows`] reads the pre-repack buffers a
    /// repack releases. They call [`marlin_repack_dense`] on the fused matrix.
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
        self.quant_view_for_inner(name, true)
    }

    /// DSv4 loading path: E8M0 scales are native (block-scaled FP8), so skip
    /// the Qwen safety rejection.
    pub(crate) fn quant_view_for_dsv4(&self, name: &str) -> Result<Option<QuantTensorView>> {
        self.quant_view_for_inner(name, false)
    }

    fn quant_view_for_inner(
        &self,
        name: &str,
        reject_e8m0: bool,
    ) -> Result<Option<QuantTensorView>> {
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
            if reject_e8m0 {
                reject_dsv4_e8m0_scale_abi(&candidate, headers.as_ref())?;
            }
            if let Some(view) =
                detect_quant_format(&candidate, headers.as_ref(), self.quant_manifest.as_ref())?
            {
                return Ok(Some(view));
            }
        }
        Ok(None)
    }

    pub(crate) fn tensor_headers(&self) -> Result<Rc<BTreeMap<String, TensorHeader>>> {
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
            QuantFormat::W4Afp8 => bail!(
                "{}: W4AFP8 is loaded by the DSv4 MoE loader, not load_quant_or_dense_view",
                view.name
            ),
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
                // Per-channel scales (one column): a col shard keeps every row, so
                // each rank needs the whole scale column — replicate, don't slice.
                if scale_cols == 1 {
                    return Ok(ShardedBytesCow {
                        bytes: Cow::Borrowed(bytes),
                        rows: scale_rows,
                        cols: 1,
                    });
                }
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
    pub(crate) fn shard_bytes(&self, idx: usize) -> Result<Rc<ShardBytes>> {
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

    pub(crate) fn borrow_bf16_tensor(&self, name: &str) -> Result<SharedTensor> {
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

pub(crate) fn float_elem_size(name: &str, dtype: Dtype) -> Result<usize> {
    match dtype {
        Dtype::F32 => Ok(4),
        Dtype::BF16 => Ok(2),
        other => bail!("{name}: expected BF16/F32 scale tensor, got {other:?}"),
    }
}

pub(crate) fn tensor_bytes_to_f32(
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

/// Give a dense projection its Marlin tensor-core layout at load time. Both
/// repacks are format-gated no-ops, so one call covers NVFP4 (kFE2M1f) and
/// per-channel FP8 (kFE4M3fn) — the mixed Qwen3.8-27B-NVFP4 checkpoint carries
/// both. Every other format, and any shape or group size the vendored kernel is
/// not instantiated for, keeps the scalar GEMV.
///
/// Routed MoE experts must NOT come here: their grouped-GEMM path reads the
/// packed nibbles the repack replaces.
///
/// A row-fused matrix repacks once, after the fuse: [`DeviceMatrix::fuse_rows`]
/// reads the pre-repack buffers, which the repack releases.
///
/// `prefill_batched` is the caller's answer to "may a prefill chunk reach this
/// weight's DeepGEMM arm at all". It gates both formats' prefill arms, not just
/// NVFP4's `sfb` — the two arms compute at different precisions from the Marlin
/// arm beside them, so a weight that must not change precision with M says
/// false and stays on Marlin at every M.
pub(crate) fn marlin_repack_dense(
    ctx: &DeviceContext,
    name: &str,
    mut matrix: DeviceMatrix,
    prefill_batched: bool,
) -> Result<DeviceMatrix> {
    matrix
        .repack_for_marlin_fp4(ctx)
        .with_context(|| format!("Marlin NVFP4 repack {name}"))?;
    matrix
        .repack_for_marlin_fp8(ctx)
        .with_context(|| format!("Marlin FP8 per-channel repack {name}"))?;
    // Both repacks release the pre-repack bytes themselves: every arm that
    // reads a repacked weight reads the Marlin layout, DeepGEMM's prefill arms
    // included. The `sfb` is built from the S0E5M3 scale tail, so it comes
    // after the repack, not before.
    if prefill_batched && fp4_deepgemm_available(ctx, &matrix) {
        matrix
            .prepare_fp4_deepgemm_sfb(ctx)
            .with_context(|| format!("NVFP4 DeepGEMM sfb {name}"))?;
    }
    matrix.fp8_deepgemm_prefill =
        prefill_batched && fp8_deepgemm_per_channel_available(ctx, &matrix);
    // Final-state gate: every M (gemv and gemm_batch alike) must have a resident
    // consumer now that the repacks have released their sources. Fail the load
    // here, never a serve-time missing-buffer error.
    validate_quant_linear_storage(ctx, name, &matrix)?;
    Ok(matrix)
}
