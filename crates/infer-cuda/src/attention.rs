//! Paged attention kernel-call paths for the dense-BF16 Qwen3 forward (HOT axis).
//!
//! Prep kernels fuse Q/K RMSNorm + RoPE + KV-cache write; the TileLang kernels
//! run the HD128/kv8 paged attention.

use anyhow::{Result, anyhow, bail, ensure};
use cuda_kernels::attention as flash_kv;
use cuda_kernels::ffi;
use cuda_kernels::kv_quant;
use cuda_kernels::moe as cuda_moe;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates, PagedKVPool};
use cuda_kernels::tensor::{WeightFormat, cache_ptr};
use cuda_kernels::{KVFormat, TokenKVPool};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use deepseek_spec::{DeepSeekV4AttentionMode, DeepSeekV4Config};
use infer_seam::{KvBatchDescriptor, KvBatchRowKind};
use std::collections::HashSet;
use std::sync::atomic::{AtomicI8, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::dsv4::{
    Dsv4Attention, Dsv4Compressor, Dsv4ForwardKeepalive, Dsv4Indexer, Dsv4MlaKvArena,
};
use crate::loader::PageMeta;
use crate::moe_config::ExpertSplit;
use crate::paged_kv_table::{contiguous_page_table_byte_range, physical_page};
use crate::tp::TpRuntime;

const DSV4_FLASHMLA_MODEL1: i32 = 1;
const DSV4_FLASHMLA_S_Q: usize = 1;
const DSV4_FLASHMLA_OVERRIDE_ENV: i8 = -1;
const DSV4_FLASHMLA_OVERRIDE_OFF: i8 = 0;
const DSV4_FLASHMLA_OVERRIDE_ON: i8 = 1;

static DSV4_FLASHMLA_DECODE_OVERRIDE: AtomicI8 = AtomicI8::new(DSV4_FLASHMLA_OVERRIDE_ENV);
static DSV4_FUSED_WQKV_DECODE_OVERRIDE: AtomicI8 = AtomicI8::new(DSV4_FLASHMLA_OVERRIDE_ENV);

pub(crate) fn set_dsv4_flashmla_decode_override(enabled: Option<bool>) {
    let value = match enabled {
        Some(true) => DSV4_FLASHMLA_OVERRIDE_ON,
        Some(false) => DSV4_FLASHMLA_OVERRIDE_OFF,
        None => DSV4_FLASHMLA_OVERRIDE_ENV,
    };
    DSV4_FLASHMLA_DECODE_OVERRIDE.store(value, Ordering::Relaxed);
}

pub(crate) fn set_dsv4_fused_wqkv_decode_override(enabled: Option<bool>) {
    let value = match enabled {
        Some(true) => DSV4_FLASHMLA_OVERRIDE_ON,
        Some(false) => DSV4_FLASHMLA_OVERRIDE_OFF,
        None => DSV4_FLASHMLA_OVERRIDE_ENV,
    };
    DSV4_FUSED_WQKV_DECODE_OVERRIDE.store(value, Ordering::Relaxed);
}

/// Replicated decode attention (B=1 DP-attn degenerate form, opt-in
/// `ARLE_DSV4_REPLICATED_ATTN=1`): every rank loads FULL `wq_b`/`wo_a` and
/// computes the whole attention block redundantly at decode. Inputs are
/// rank-identical (the MoE all-reduce output), weights identical, kernels
/// deterministic ⇒ outputs identical on every rank — so the per-layer
/// attention AllGather (FlashMLA Q gather) AND AllReduce (O-LoRA partial
/// sum) both disappear: 43+43 latency-bound 8 KB collectives/token gone.
/// The attention KERNEL was already computed full-width on every rank
/// (h_q%64==0), so the added compute is only the two full-width projections.
/// Prefill keeps the sharded math + collectives.
pub(crate) fn dsv4_replicated_attn_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ARLE_DSV4_REPLICATED_ATTN").as_deref(),
            Ok("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
        )
    })
}

static DSV4_VERIFY_FROZEN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Frozen-KV MTP verify: while set, `mla_attention` SKIPS `dsv4_compressor_update`
/// so the speculative K-token verify forms no new compressed blocks / DSA packs and
/// mutates nothing compressed. The executor sets it around the speculative verify and
/// clears it for the accepted-prefix commit re-forward.
pub(crate) fn set_dsv4_verify_frozen(frozen: bool) {
    DSV4_VERIFY_FROZEN.store(frozen, Ordering::Relaxed);
}

pub(crate) fn dsv4_verify_frozen() -> bool {
    DSV4_VERIFY_FROZEN.load(Ordering::Relaxed)
}

fn maybe_probe_flashmla_decode_path(
    layer_idx: usize,
    mode: DeepSeekV4AttentionMode,
    flashmla_used: bool,
    token_count: usize,
    start_pos: usize,
) {
    if std::env::var("ARLE_DSV4_FLASHMLA_PROBE").as_deref() != Ok("1")
        || std::env::var("INFER_TP_RANK").as_deref() != Ok("0")
        || token_count != 1
        || start_pos == 0
    {
        return;
    }

    static SEEN: OnceLock<Mutex<HashSet<(usize, String)>>> = OnceLock::new();
    let mode_key = format!("{mode:?}");
    let set = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if set.lock().unwrap().insert((layer_idx, mode_key)) {
        eprintln!("[flashmla-probe] layer={layer_idx} mode={mode:?} flashmla_used={flashmla_used}");
    }
}

pub(crate) struct Dsv4CompressorState {
    pending_kv: CudaSlice<half::bf16>,
    pending_score: CudaSlice<half::bf16>,
    prev_overlap_kv: CudaSlice<half::bf16>,
    prev_overlap_score: CudaSlice<half::bf16>,
    compressed: HiddenStates,
}

impl Dsv4CompressorState {
    fn new(
        ctx: &DeviceContext,
        head_dim: usize,
        ratio: usize,
        overlap: bool,
        max_seq_len: usize,
    ) -> Result<Self> {
        let width = if overlap { 2 * head_dim } else { head_dim };
        let compressed_rows = max_seq_len.div_ceil(ratio).max(1);
        Ok(Self {
            pending_kv: ctx
                .stream
                .alloc_zeros::<half::bf16>(ratio * width)
                .map_err(|e| anyhow::anyhow!("DSv4 compressor pending kv alloc failed: {e}"))?,
            pending_score: ctx
                .stream
                .alloc_zeros::<half::bf16>(ratio * width)
                .map_err(|e| anyhow::anyhow!("DSv4 compressor pending score alloc failed: {e}"))?,
            prev_overlap_kv: ctx
                .stream
                .alloc_zeros::<half::bf16>(ratio * head_dim)
                .map_err(|e| anyhow::anyhow!("DSv4 compressor prev kv alloc failed: {e}"))?,
            prev_overlap_score: ctx
                .stream
                .alloc_zeros::<half::bf16>(ratio * head_dim)
                .map_err(|e| anyhow::anyhow!("DSv4 compressor prev score alloc failed: {e}"))?,
            compressed: HiddenStates::zeros(ctx, head_dim, compressed_rows)?,
        })
    }

    fn reset(&mut self, ctx: &DeviceContext) -> Result<()> {
        ctx.stream
            .memset_zeros(&mut self.pending_kv)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor pending kv reset failed: {e}"))?;
        ctx.stream
            .memset_zeros(&mut self.pending_score)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor pending score reset failed: {e}"))?;
        ctx.stream
            .memset_zeros(&mut self.prev_overlap_kv)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor prev kv reset failed: {e}"))?;
        ctx.stream
            .memset_zeros(&mut self.prev_overlap_score)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor prev score reset failed: {e}"))?;
        ctx.stream
            .memset_zeros(&mut self.compressed.data)
            .map_err(|e| anyhow::anyhow!("DSv4 compressor compressed reset failed: {e}"))?;
        self.compressed.seq_len = 0;
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Dsv4RollbackChecksum {
    len: usize,
    hash: u64,
    sum_abs: f64,
    first: [f32; 4],
}

impl Dsv4RollbackChecksum {
    fn from_host(host: &[half::bf16]) -> Self {
        let mut hash: u64 = 0xcbf29ce484222325;
        let mut sum_abs = 0.0_f64;
        let mut first = [0.0_f32; 4];
        for (idx, value) in host.iter().enumerate() {
            hash ^= u64::from(value.to_bits());
            hash = hash.wrapping_mul(0x100000001b3);
            let value_f32 = value.to_f32();
            sum_abs += f64::from(value_f32.abs());
            if idx < first.len() {
                first[idx] = value_f32;
            }
        }
        Self {
            len: host.len(),
            hash,
            sum_abs,
            first,
        }
    }
}

impl std::fmt::Display for Dsv4RollbackChecksum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "len={} hash={:016x} sum_abs={:.6} first={:?}",
            self.len, self.hash, self.sum_abs, self.first
        )
    }
}

fn dsv4_mtp_rollback_dump_enabled() -> bool {
    matches!(
        std::env::var("ARLE_DSV4_MTP_ROLLBACK_DUMP").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
    ) && std::env::var("INFER_TP_RANK").as_deref() == Ok("0")
}

fn dsv4_checksum_bf16_slice(
    ctx: &DeviceContext,
    slice: &CudaSlice<half::bf16>,
) -> Result<Dsv4RollbackChecksum> {
    ctx.sync()?;
    let host: Vec<half::bf16> = ctx
        .stream
        .clone_dtoh(slice)
        .map_err(|e| anyhow!("DSv4 rollback dump D2H failed: {e}"))?;
    Ok(Dsv4RollbackChecksum::from_host(&host))
}

fn dsv4_checksum_hidden(
    ctx: &DeviceContext,
    hidden: &HiddenStates,
) -> Result<Dsv4RollbackChecksum> {
    dsv4_checksum_bf16_slice(ctx, &hidden.data)
}

pub(crate) struct Dsv4KvAdapter {
    layers: Vec<Dsv4LayerKvLayout>,
    num_slots: usize,
    slot_epochs: Vec<Option<u64>>,
    /// One shared official-DSA selector scratch for ALL CSA layers and slots
    /// (issue #67). `None` when the model has no CSA layer or the official
    /// indexer is disabled (`ARLE_DSV4_DSA_INDEXER=0`).
    dsa_shared: Option<Dsv4DsaSharedScratch>,
    /// One shared MoE decode scratch for ALL layers and slots (issue #60).
    /// `None` when the GPU router decode scratch path is disabled.
    moe_decode_shared: Option<crate::moe::Dsv4MoeDecodeScratch>,
    /// One shared-expert decode output for ALL layers and slots (issue #60).
    /// `None` when the GPU router decode scratch path is disabled.
    shared_expert_out: Option<HiddenStates>,
}

pub(crate) struct Dsv4LayerKvLayout {
    /// Shared FP8 MLA latent pool for this layer (#85 P2 Stage A): a
    /// `TokenKVPool` of opaque packed records (`KVFormat::PackedBytes`,
    /// 584 B/token), page = FlashMLA block = 64 tokens, single plane (records
    /// in the K plane, no V/scale buffers). Every slot's band is addressed
    /// ONLY through its block table ([`Self::flashmla_page_table`] /
    /// [`Self::flashmla_pages_byte_range`]) — never by `slot_idx × slot_bytes`
    /// arithmetic. Stage A allocates each slot's pages up-front in slot
    /// order, so tables are contiguous identity runs and the physical layout
    /// is byte-identical to the pre-paging band arena.
    flashmla_kv_pool: Option<TokenKVPool>,
    dsa_key_cache: Option<CudaSlice<u8>>,
    /// Slot-logical FlashMLA blocks per slot (`sw_blocks + comp_blocks` for
    /// this layer's shape) — every slot's block-table length.
    flashmla_slot_pages: usize,
    /// Bytes of one pool page (`page_block_size × packed record bytes`).
    flashmla_page_bytes: usize,
    dsa_slot_bytes: usize,
    num_slots: usize,
}

pub(crate) trait ModelKvAdapter {
    type BatchView;

    fn prepare_kv_batch(&mut self, desc: &KvBatchDescriptor) -> Result<Self::BatchView>;
}

#[derive(Debug, Clone)]
pub(crate) struct Dsv4KvBatchView {
    pub(crate) rows: Vec<Dsv4KvBatchRowView>,
    pub(crate) flat_page_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
pub(crate) struct Dsv4KvBatchRowView {
    pub(crate) slot: usize,
    pub(crate) kind: KvBatchRowKind,
    pub(crate) seq_len: usize,
    pub(crate) append_pos: usize,
    pub(crate) append_len: usize,
    #[allow(dead_code)]
    pub(crate) slot_epoch: u64,
    pub(crate) page_range: std::ops::Range<usize>,
}

impl Dsv4KvAdapter {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        layer_specs: &[(DeepSeekV4AttentionMode, usize, usize)],
        max_seq_len: usize,
        kv_arena: &Dsv4MlaKvArena,
        tp_world: usize,
        num_slots: usize,
        moe_decode: Option<(
            &infer_moe::MoeConfig,
            &ExpertSplit,
            &crate::dsv4::Dsv4MoeLayer,
        )>,
        hidden_size: usize,
    ) -> Result<Self> {
        ensure!(num_slots > 0, "DSv4 attention pool needs at least one slot");
        let mut layers = Vec::with_capacity(layer_specs.len());
        for &(mode, compress_ratio, local_heads) in layer_specs {
            layers.push(Dsv4LayerKvLayout::new(
                ctx,
                config,
                mode,
                compress_ratio,
                max_seq_len,
                kv_arena,
                local_heads,
                tp_world,
                num_slots,
            )?);
        }
        // Build the ONE shared official-DSA scratch when any CSA layer exists.
        // All CSA layers must agree on compress_ratio: the scratch's
        // compressed-capacity sizing is shared (uniform cr=4 on DSv4-Flash).
        let mut csa_ratios = layer_specs
            .iter()
            .filter(|(mode, _, _)| *mode == DeepSeekV4AttentionMode::CompressedSparse)
            .map(|&(_, compress_ratio, _)| compress_ratio);
        let dsa_shared = match csa_ratios.next() {
            Some(first_cr) if dsv4_dsa_official_enabled()? => {
                ensure!(
                    csa_ratios.all(|cr| cr == first_cr),
                    "DSv4 shared DSA scratch requires a uniform CSA compress_ratio"
                );
                Some(Dsv4DsaSharedScratch::new(
                    ctx,
                    config,
                    first_cr,
                    max_seq_len,
                )?)
            }
            _ => None,
        };
        let moe_decode_shared = moe_decode
            .map(|(cfg, split, layer)| {
                crate::moe::Dsv4MoeDecodeScratch::new(ctx, cfg, split, layer)
            })
            .transpose()?;
        // ALWAYS allocate the model-wide B=1 shared-expert decode output: the
        // shared expert runs on every decode step (`use_comm_overlap || seq_len
        // == 1`) regardless of the GPU-router scratch path, so this must not be
        // gated on `moe_decode_shared`. One tiny instance (hidden_size×1 BF16 ≈
        // 14 KiB) replaces the old per-slot×per-layer `shared_decode_out` (#60),
        // and keeping it pre-allocated avoids a per-step `uninit` on the default
        // decode path.
        let shared_expert_out = Some(unsafe { HiddenStates::uninit(ctx, hidden_size, 1)? });
        Ok(Self {
            layers,
            num_slots,
            slot_epochs: vec![None; num_slots],
            dsa_shared,
            moe_decode_shared,
            shared_expert_out,
        })
    }

    /// Split-borrow accessor: one layer's KV layout plus the model-wide shared
    /// DSA scratch (disjoint fields, so both can be `&mut` at once).
    pub(crate) fn layer_and_dsa_shared_mut(
        &mut self,
        layer_idx: usize,
    ) -> Result<(&mut Dsv4LayerKvLayout, Option<&mut Dsv4DsaSharedScratch>)> {
        let len = self.layers.len();
        let layer = self
            .layers
            .get_mut(layer_idx)
            .ok_or_else(|| anyhow!("DSv4 attention pool layer {layer_idx} outside len {len}"))?;
        Ok((layer, self.dsa_shared.as_mut()))
    }

    /// Split-borrow accessor: model-wide shared MoE decode scratch plus the
    /// shared-expert output buffer (disjoint fields, so both can be `&mut`).
    pub(crate) fn moe_decode_shared_mut(
        &mut self,
    ) -> (
        Option<&mut crate::moe::Dsv4MoeDecodeScratch>,
        Option<&mut HiddenStates>,
    ) {
        (
            self.moe_decode_shared.as_mut(),
            self.shared_expert_out.as_mut(),
        )
    }

    pub(crate) fn layer_mut(&mut self, layer_idx: usize) -> Result<&mut Dsv4LayerKvLayout> {
        let len = self.layers.len();
        self.layers
            .get_mut(layer_idx)
            .ok_or_else(|| anyhow!("DSv4 attention pool layer {layer_idx} outside len {len}"))
    }

    pub(crate) fn layer(&self, layer_idx: usize) -> Result<&Dsv4LayerKvLayout> {
        let len = self.layers.len();
        self.layers
            .get(layer_idx)
            .ok_or_else(|| anyhow!("DSv4 attention pool layer {layer_idx} outside len {len}"))
    }
}

impl ModelKvAdapter for Dsv4KvAdapter {
    type BatchView = Dsv4KvBatchView;

    fn prepare_kv_batch(&mut self, desc: &KvBatchDescriptor) -> Result<Self::BatchView> {
        let mut rows = Vec::with_capacity(desc.rows.len());
        for (idx, row) in desc.rows.iter().enumerate() {
            ensure!(
                row.slot < self.num_slots,
                "DSv4 KV batch row {idx} slot {} outside adapter slots {}",
                row.slot,
                self.num_slots
            );
            ensure!(
                row.page_range.end <= desc.flat_page_ids.len(),
                "DSv4 KV batch row {idx} page range {:?} outside flat page len {}",
                row.page_range,
                desc.flat_page_ids.len()
            );
            ensure!(
                row.token_range.end <= desc.flat_token_ids.len(),
                "DSv4 KV batch row {idx} token range {:?} outside flat token len {}",
                row.token_range,
                desc.flat_token_ids.len()
            );
            ensure!(
                row.page_range.start < row.page_range.end,
                "DSv4 KV batch row {idx} has empty page range"
            );
            self.slot_epochs[row.slot] = Some(row.slot_epoch);
            rows.push(Dsv4KvBatchRowView {
                slot: row.slot,
                kind: row.kind,
                seq_len: row.seq_len,
                append_pos: row.append_pos,
                append_len: row.append_len,
                slot_epoch: row.slot_epoch,
                page_range: row.page_range.clone(),
            });
        }

        Ok(Dsv4KvBatchView {
            rows,
            flat_page_ids: desc.flat_page_ids.clone(),
        })
    }
}

impl Dsv4LayerKvLayout {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        max_seq_len: usize,
        kv_arena: &Dsv4MlaKvArena,
        local_heads: usize,
        tp_world: usize,
        num_slots: usize,
    ) -> Result<Self> {
        let flashmla_slot_pages = if dsv4_flashmla_decode_alloc_enabled()? {
            let shape = Dsv4FlashMlaDecodeShape::new(
                config,
                mode,
                compress_ratio,
                max_seq_len,
                kv_arena,
                local_heads,
                tp_world,
            )?;
            shape.total_blocks
        } else {
            0
        };
        let flashmla_page_bytes = kv_arena
            .page_block_size
            .checked_mul(kv_arena.bytes_per_token)
            .ok_or_else(|| anyhow!("DSv4 shared FlashMLA page byte size overflow"))?;
        let flashmla_kv_pool = if flashmla_slot_pages > 0 {
            // #85 P2 Stage A: the raw band arena is replaced by a shared
            // `TokenKVPool` of packed MLA latent records, sized to the SAME
            // total token budget (`total_blocks × 64 × num_slots` records).
            let format = KVFormat::PackedBytes {
                bytes_per_token: kv_arena.bytes_per_token,
            };
            ensure!(
                format.default_page_size() == kv_arena.page_block_size,
                "DSv4 FlashMLA pool page size {} != arena block size {}",
                format.default_page_size(),
                kv_arena.page_block_size
            );
            let tokens_per_slot = flashmla_slot_pages
                .checked_mul(kv_arena.page_block_size)
                .ok_or_else(|| anyhow!("DSv4 shared FlashMLA slot token overflow"))?;
            let total_tokens = tokens_per_slot
                .checked_mul(num_slots)
                .ok_or_else(|| anyhow!("DSv4 shared FlashMLA pool total overflow"))?;
            let budget_bytes = TokenKVPool::budget_bytes_for_tokens(
                1, // single "layer": one pool per Dsv4LayerKvLayout
                1, // MLA latent is head-less (kv_heads = 1)
                config.head_dim,
                total_tokens,
                format,
            );
            let mut pool = TokenKVPool::with_format(
                ctx,
                1,
                1,
                config.head_dim,
                num_slots,
                budget_bytes,
                format,
            )
            .map_err(|e| anyhow!("DSv4 shared FlashMLA pool alloc failed: {e}"))?;
            ensure!(
                pool.page_size == kv_arena.page_block_size
                    && pool.max_total_pages == flashmla_slot_pages * num_slots,
                "DSv4 FlashMLA pool sizing mismatch: page_size={} pages={} expected page_size={} pages={}",
                pool.page_size,
                pool.max_total_pages,
                kv_arena.page_block_size,
                flashmla_slot_pages * num_slots
            );
            // Stage A identity tables: every slot's pages are allocated
            // up-front in slot order, so slot `i` owns the contiguous run
            // `[i × total_blocks, (i+1) × total_blocks)` — byte-identical to
            // the pre-paging band arena. TP lockstep invariant: the tables
            // derive only from construction constants (num_slots + the
            // plan-pinned decode shape), so they are identical on every rank;
            // the NCCL min-reduced slot budget (`kv_budget_num_slots`) stays
            // the cross-rank capacity gate. Stage B replaces this loop with
            // host-pool mirroring (Qwen `mirror_slot` pattern).
            for slot in 0..num_slots {
                let pages = pool
                    .alloc_tokens(slot, tokens_per_slot)
                    .map_err(|e| anyhow!("DSv4 FlashMLA slot {slot} page alloc failed: {e}"))?;
                let first = (slot * flashmla_slot_pages) as u32;
                ensure!(
                    pages.len() == flashmla_slot_pages
                        && pages
                            .iter()
                            .enumerate()
                            .all(|(i, &p)| p == first + i as u32),
                    "DSv4 FlashMLA Stage A identity layout violated for slot {slot}"
                );
            }
            Some(pool)
        } else {
            None
        };

        let dsa_slot_bytes =
            if mode == DeepSeekV4AttentionMode::CompressedSparse && dsv4_dsa_official_enabled()? {
                dsv4_dsa_key_cache_bytes(config, compress_ratio, max_seq_len)?
            } else {
                0
            };
        let dsa_key_cache = if dsa_slot_bytes > 0 {
            Some(
                ctx.stream
                    .alloc_zeros::<u8>(
                        dsa_slot_bytes
                            .checked_mul(num_slots)
                            .ok_or_else(|| anyhow!("DSv4 shared DSA key-cache total overflow"))?,
                    )
                    .map_err(|e| anyhow!("DSv4 shared DSA key-cache alloc failed: {e}"))?,
            )
        } else {
            None
        };
        Ok(Self {
            flashmla_kv_pool,
            dsa_key_cache,
            flashmla_slot_pages,
            flashmla_page_bytes,
            dsa_slot_bytes,
            num_slots,
        })
    }

    fn slot_range(
        slot_idx: usize,
        slot_bytes: usize,
        num_slots: usize,
    ) -> Result<std::ops::Range<usize>> {
        ensure!(
            slot_idx < num_slots,
            "DSv4 attention pool slot {slot_idx} outside num_slots {num_slots}"
        );
        let start = slot_idx
            .checked_mul(slot_bytes)
            .ok_or_else(|| anyhow!("DSv4 attention pool slot offset overflow"))?;
        let end = start
            .checked_add(slot_bytes)
            .ok_or_else(|| anyhow!("DSv4 attention pool slot end overflow"))?;
        Ok(start..end)
    }

    /// The layer's shared FlashMLA latent pool (present iff the decode-alloc
    /// gate is on and this layer has a non-empty FlashMLA shape).
    fn flashmla_pool(&self) -> Result<&TokenKVPool> {
        self.flashmla_kv_pool
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 FlashMLA shared pool missing"))
    }

    fn flashmla_pool_mut(&mut self) -> Result<&mut TokenKVPool> {
        self.flashmla_kv_pool
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 FlashMLA shared pool missing"))
    }

    /// Whole-pool data plane (packed records live in the K plane only).
    fn flashmla_pool_data(&self) -> Result<&CudaSlice<u8>> {
        Ok(self.flashmla_pool()?.k_data_slice(0))
    }

    fn flashmla_pool_data_mut(&mut self) -> Result<&mut CudaSlice<u8>> {
        Ok(self.flashmla_pool_mut()?.k_data_slice_mut(0))
    }

    /// One slot's page table: slot-logical page → physical pool page
    /// (FlashMLA's FFI calls our 64-token page a "block"). The ONLY source of band addresses (#85 P2) — token counts and
    /// block counts come from the pool's table, never re-derived from
    /// `slot_idx` arithmetic.
    fn flashmla_page_table(&self, slot_idx: usize) -> Result<&[u32]> {
        ensure!(
            slot_idx < self.num_slots,
            "DSv4 attention pool slot {slot_idx} outside num_slots {}",
            self.num_slots
        );
        Ok(self.flashmla_pool()?.page_indices(slot_idx))
    }

    /// Table-routed byte range of one slot's FlashMLA band.
    ///
    /// Invariant (#85 P2 Stage A): the range derives from the slot's PAGE
    /// TABLE, and the helper errors unless the table is a contiguous identity
    /// run — that contiguity is what licenses the band-base addressing the
    /// device-side pack/index kernels still use. Stage B must hand those
    /// kernels a device-resident table before tables may fragment.
    fn flashmla_pages_byte_range(&self, slot_idx: usize) -> Result<std::ops::Range<usize>> {
        let table = self.flashmla_page_table(slot_idx)?;
        let range = contiguous_page_table_byte_range(
            table,
            self.flashmla_slot_pages,
            self.flashmla_page_bytes,
        )?;
        let pool_bytes = self.flashmla_pool_data()?.len();
        ensure!(
            range.end <= pool_bytes,
            "DSv4 FlashMLA table range {:?} outside pool bytes {}",
            range,
            pool_bytes
        );
        Ok(range)
    }

    fn dsa_slot_range(&self, slot_idx: usize) -> Result<std::ops::Range<usize>> {
        Self::slot_range(slot_idx, self.dsa_slot_bytes, self.num_slots)
    }

    fn reset_flashmla_slot(
        &mut self,
        ctx: &DeviceContext,
        flash: &Dsv4FlashMlaDecodeState,
    ) -> Result<()> {
        // Table-routed (#85 P2): the zeroed range derives from the slot's
        // page table. Stage A contiguity makes it one span (a single memset,
        // same launch count as the pre-paging band reset); Stage B zeroes —
        // or frees — per page once tables may fragment.
        let range = self.flashmla_pages_byte_range(flash.slot_idx)?;
        ensure!(
            range.len() == flash.fp8_kv_pool_len,
            "DSv4 FlashMLA shared pool range {:?} invalid for slot_len={}",
            range,
            flash.fp8_kv_pool_len
        );
        let pool_buf = self.flashmla_pool_data_mut()?;
        let mut view = pool_buf.slice_mut(range);
        ctx.stream
            .memset_zeros(&mut view)
            .map_err(|e| anyhow!("DSv4 shared FlashMLA slot reset failed: {e}"))?;
        Ok(())
    }

    fn reset_dsa_slot(&mut self, ctx: &DeviceContext, dsa: &Dsv4DsaOfficialState) -> Result<()> {
        let range = self.dsa_slot_range(dsa.slot_idx)?;
        let pool = self
            .dsa_key_cache
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 DSA shared key-cache missing"))?;
        ensure!(
            range.end <= pool.len() && range.len() == dsa.key_cache_len,
            "DSv4 DSA shared key-cache range {:?} invalid for pool_len={} slot_len={}",
            range,
            pool.len(),
            dsa.key_cache_len
        );
        let mut view = pool.slice_mut(range);
        ctx.stream
            .memset_zeros(&mut view)
            .map_err(|e| anyhow!("DSv4 shared DSA key-cache reset failed: {e}"))?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Dsv4FlashMlaDecodeShape {
    sw_blocks: usize,
    comp_blocks: usize,
    max_compressed_keys: usize,
    topk_unified: usize,
    total_blocks: usize,
    h_q: usize,
}

impl Dsv4FlashMlaDecodeShape {
    #[allow(clippy::too_many_arguments)]
    fn new(
        config: &DeepSeekV4Config,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        max_seq_len: usize,
        kv_arena: &Dsv4MlaKvArena,
        local_heads: usize,
        tp_world: usize,
    ) -> Result<Self> {
        ensure!(
            config.head_dim == 512 && kv_arena.bytes_per_token == 584,
            "DSv4 FlashMLA decode only wires MODEL1 head_dim=512 bytes/token=584"
        );
        ensure!(
            local_heads > 0 && tp_world > 0,
            "DSv4 FlashMLA decode requires non-zero local_heads and tp_world"
        );
        let h_q = local_heads
            .checked_mul(tp_world)
            .ok_or_else(|| anyhow!("DSv4 FlashMLA h_q overflow"))?;
        ensure!(
            matches!(h_q, 64 | 128),
            "DSv4 FlashMLA decode requires global h_q 64 or 128, got {h_q}"
        );
        ensure!(
            kv_arena.page_block_size == 64,
            "DSv4 FlashMLA decode requires page_block_size=64"
        );
        let sw_blocks = config.sliding_window.div_ceil(kv_arena.page_block_size);
        let compressed_rows = if mode == DeepSeekV4AttentionMode::SlidingWindow {
            0
        } else {
            ensure!(
                compress_ratio > 0,
                "DSv4 FlashMLA compressed decode requires non-zero ratio"
            );
            max_seq_len.div_ceil(compress_ratio).max(1)
        };
        let comp_blocks = compressed_rows.div_ceil(kv_arena.page_block_size);
        let max_compressed_keys = match mode {
            DeepSeekV4AttentionMode::SlidingWindow => 0,
            DeepSeekV4AttentionMode::CompressedSparse => config.index_topk,
            DeepSeekV4AttentionMode::HybridCompressed => compressed_rows.div_ceil(128) * 128,
        };
        let topk_unified = config
            .sliding_window
            .checked_add(max_compressed_keys)
            .ok_or_else(|| anyhow!("DSv4 FlashMLA topk_unified overflow"))?;
        ensure!(
            topk_unified.is_multiple_of(128),
            "DSv4 FlashMLA topk_unified {topk_unified} must be multiple of 128"
        );
        let total_blocks = sw_blocks
            .checked_add(comp_blocks)
            .ok_or_else(|| anyhow!("DSv4 FlashMLA total block overflow"))?;
        Ok(Self {
            sw_blocks,
            comp_blocks,
            max_compressed_keys,
            topk_unified,
            total_blocks,
            h_q,
        })
    }
}

struct Dsv4FlashMlaDecodeState {
    slot_idx: usize,
    fp8_kv_pool_len: usize,
    sw_blocks: usize,
    comp_blocks: usize,
    max_compressed_keys: usize,
    topk_unified: usize,
    fp8_kv_sw_bootstrapped: bool,
    fp8_kv_comp_packed_rows: usize,
    sw_bulk_block_ids: CudaSlice<i32>,
    sw_bulk_rows: CudaSlice<i32>,
    one_block_id: CudaSlice<i32>,
    one_row: CudaSlice<i32>,
    comp_block_ids: CudaSlice<i32>,
    comp_rows: CudaSlice<i32>,
    indices: CudaSlice<i32>,
    topk_length: CudaSlice<i32>,
    lse_out: CudaSlice<f32>,
    lse_accum: CudaSlice<f32>,
    o_accum: CudaSlice<f32>,
    sched_meta: CudaSlice<i32>,
    num_splits: CudaSlice<i32>,
    tp_gathered_q: CudaSlice<half::bf16>,
    tp_packed_q: CudaSlice<half::bf16>,
    tp_full_out: CudaSlice<half::bf16>,
    num_sm_parts: i32,
    fixed_overhead_num_blocks: i32,
    block_size_topk: i32,
}

impl Dsv4FlashMlaDecodeState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        max_seq_len: usize,
        kv_arena: &Dsv4MlaKvArena,
        local_heads: usize,
        tp_world: usize,
        slot_idx: usize,
        pool: &Dsv4LayerKvLayout,
    ) -> Result<Self> {
        let shape = Dsv4FlashMlaDecodeShape::new(
            config,
            mode,
            compress_ratio,
            max_seq_len,
            kv_arena,
            local_heads,
            tp_world,
        )?;
        // Table-routed (#85 P2): the slot band length comes from the slot's
        // block table; the decode shape must agree with the layout's table
        // length (both derive from the same plan-pinned construction params).
        let range = pool.flashmla_pages_byte_range(slot_idx)?;
        ensure!(
            shape.total_blocks == pool.flashmla_slot_pages && !range.is_empty(),
            "DSv4 FlashMLA shared slot band missing/invalid for slot {slot_idx} \
             (shape blocks {} vs table blocks {})",
            shape.total_blocks,
            pool.flashmla_slot_pages
        );

        let mut num_sm_parts = 0_i32;
        let mut fixed_overhead_num_blocks = 0_i32;
        let mut block_size_topk = 0_i32;
        unsafe {
            ffi::arle_flashmla_sm90_sparse_decode_get_meta(
                shape.h_q as i32,
                DSV4_FLASHMLA_S_Q as i32,
                DSV4_FLASHMLA_MODEL1,
                &mut num_sm_parts,
                &mut fixed_overhead_num_blocks,
                &mut block_size_topk,
            )
            .result()
            .map_err(|e| anyhow!("DSv4 FlashMLA decode meta failed: {e}"))?;
        }
        let num_sm_parts_max = (num_sm_parts as usize).max(256);
        let h_q_d = shape
            .h_q
            .checked_mul(config.head_dim)
            .ok_or_else(|| anyhow!("DSv4 FlashMLA h_q*d overflow"))?;
        let accum_rows = num_sm_parts_max + 1;
        let sw_slots = config.sliding_window;
        let comp_slots = if mode == DeepSeekV4AttentionMode::SlidingWindow {
            1
        } else {
            max_seq_len.div_ceil(compress_ratio).max(1)
        };

        let mut state = Self {
            slot_idx,
            fp8_kv_pool_len: range.len(),
            sw_blocks: shape.sw_blocks,
            comp_blocks: shape.comp_blocks,
            max_compressed_keys: shape.max_compressed_keys,
            topk_unified: shape.topk_unified,
            fp8_kv_sw_bootstrapped: false,
            fp8_kv_comp_packed_rows: 0,
            sw_bulk_block_ids: ctx.stream.alloc_zeros::<i32>(sw_slots)?,
            sw_bulk_rows: ctx.stream.alloc_zeros::<i32>(sw_slots)?,
            one_block_id: ctx.stream.alloc_zeros::<i32>(1)?,
            one_row: ctx.stream.alloc_zeros::<i32>(1)?,
            comp_block_ids: ctx.stream.alloc_zeros::<i32>(comp_slots)?,
            comp_rows: ctx.stream.alloc_zeros::<i32>(comp_slots)?,
            indices: ctx.stream.alloc_zeros::<i32>(shape.topk_unified)?,
            topk_length: ctx.stream.alloc_zeros::<i32>(1)?,
            lse_out: ctx.stream.alloc_zeros::<f32>(shape.h_q)?,
            lse_accum: ctx.stream.alloc_zeros::<f32>(accum_rows * shape.h_q)?,
            o_accum: ctx.stream.alloc_zeros::<f32>(accum_rows * h_q_d)?,
            sched_meta: ctx.stream.alloc_zeros::<i32>(num_sm_parts_max * 8)?,
            num_splits: ctx.stream.alloc_zeros::<i32>(2)?,
            tp_gathered_q: ctx.stream.alloc_zeros::<half::bf16>(h_q_d)?,
            tp_packed_q: ctx.stream.alloc_zeros::<half::bf16>(h_q_d)?,
            tp_full_out: ctx.stream.alloc_zeros::<half::bf16>(h_q_d)?,
            num_sm_parts,
            fixed_overhead_num_blocks,
            block_size_topk,
        };
        state.init_constant_sched_meta(ctx)?;
        Ok(state)
    }

    /// Fill `topk_length` and the FlashMLA scheduler metadata ONCE: both
    /// depend only on slot constants (`topk_unified`, sm-part shape), so
    /// computing them per decode step was (a) wasted work (43 calls/token)
    /// and (b) a CUDA-graph capture hazard — the per-step
    /// `memcpy_htod(&[topk], ..)` recorded a memcpy node whose HOST source
    /// was a dead stack temporary, so replay read a dangling pointer
    /// (garbage topk → insane splits → the 2026-06-10 IMA).
    fn init_constant_sched_meta(&mut self, ctx: &DeviceContext) -> Result<()> {
        let topk = i32::try_from(self.topk_unified)
            .map_err(|_| anyhow!("DSv4 FlashMLA topk {} overflows i32", self.topk_unified))?;
        ctx.stream
            .memcpy_htod(&[topk], &mut self.topk_length)
            .map_err(|e| anyhow!("DSv4 FlashMLA topk_length H2D failed: {e}"))?;
        let (topk_ptr, _tg) = self.topk_length.device_ptr(&ctx.stream);
        let (sched_ptr, _sg) = self.sched_meta.device_ptr_mut(&ctx.stream);
        let (splits_ptr, _pg) = self.num_splits.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::arle_flashmla_sm90_sparse_decode_sched_meta(
                1,
                1,
                self.block_size_topk,
                self.fixed_overhead_num_blocks,
                topk,
                0,
                topk_ptr as *const i32,
                std::ptr::null(),
                sched_ptr as *mut i32,
                splits_ptr as *mut i32,
                self.num_sm_parts,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| anyhow!("DSv4 FlashMLA sched_meta failed: {e}"))?;
        }
        Ok(())
    }

    fn reset(&mut self, ctx: &DeviceContext, pool: &mut Dsv4LayerKvLayout) -> Result<()> {
        self.fp8_kv_sw_bootstrapped = false;
        self.fp8_kv_comp_packed_rows = 0;
        pool.reset_flashmla_slot(ctx, self)?;
        Ok(())
    }
}

struct Dsv4FusedWqkvDecodeScratch {
    input_fp8: CudaSlice<u8>,
    input_scales: CudaSlice<f32>,
    qkv_raw: HiddenStates,
    active_experts: CudaSlice<i32>,
    active_offsets: CudaSlice<i32>,
    active_counts: CudaSlice<i32>,
    max_m: usize,
    scale_stride_m: usize,
    hidden_dim: usize,
    q_lora_rank: usize,
    head_dim: usize,
}

impl Dsv4FusedWqkvDecodeScratch {
    fn new(ctx: &DeviceContext, config: &DeepSeekV4Config) -> Result<Self> {
        let max_m = 128;
        let scale_stride_m = 128;
        let hidden_dim = config.hidden_size;
        let q_lora_rank = config.q_lora_rank;
        let head_dim = config.head_dim;
        let scale_cols = hidden_dim.div_ceil(128);
        Ok(Self {
            input_fp8: ctx
                .stream
                .alloc_zeros::<u8>(max_m * hidden_dim)
                .map_err(|e| anyhow!("DSv4 fused wqkv input fp8 scratch alloc failed: {e}"))?,
            input_scales: ctx
                .stream
                .alloc_zeros::<f32>(scale_stride_m * scale_cols)
                .map_err(|e| anyhow!("DSv4 fused wqkv input scale scratch alloc failed: {e}"))?,
            qkv_raw: unsafe { HiddenStates::uninit(ctx, q_lora_rank + head_dim, 1)? },
            active_experts: ctx
                .stream
                .clone_htod(&[0_i32])
                .map_err(|e| anyhow!("DSv4 fused wqkv active_experts H2D failed: {e}"))?,
            active_offsets: ctx
                .stream
                .clone_htod(&[0_i32])
                .map_err(|e| anyhow!("DSv4 fused wqkv active_offsets H2D failed: {e}"))?,
            active_counts: ctx
                .stream
                .clone_htod(&[1_i32])
                .map_err(|e| anyhow!("DSv4 fused wqkv active_counts H2D failed: {e}"))?,
            max_m,
            scale_stride_m,
            hidden_dim,
            q_lora_rank,
            head_dim,
        })
    }
}

struct Dsv4PrefillDeepGemmLinearScratch {
    input_fp8: CudaSlice<u8>,
    input_scales: CudaSlice<f32>,
    qkv_raw: HiddenStates,
    active_experts: CudaSlice<i32>,
    active_offsets: CudaSlice<i32>,
    active_counts: CudaSlice<i32>,
    max_m: usize,
    max_k: usize,
    max_scale_stride_m: usize,
    hidden_dim: usize,
    q_lora_rank: usize,
    head_dim: usize,
}

impl Dsv4PrefillDeepGemmLinearScratch {
    fn new(ctx: &DeviceContext, config: &DeepSeekV4Config, max_seq_len: usize) -> Result<Self> {
        // M (query/token) dimension is chunk-bounded: under chunked prefill the
        // layer forward processes at most `chunked_prefill_size` (<= chunk) query
        // tokens per call, so the M×K `input_fp8` / `input_scales` / `qkv_raw`
        // activation scratch is sized by `query_chunk`, not the slot's full
        // `max_seq_len` context (which would OOM at 900K). K = hidden_size stays
        // full. A debug assert at each forward call site (`prefill_proj_deepgemm`,
        // `run_fused_wqkv_prefill`) guards `seq_len <= max_m`.
        let query_chunk = DSV4_PREFILL_QUERY_CHUNK.min(max_seq_len.max(1));
        let max_m = query_chunk;
        let max_k = config.hidden_size;
        let q_lora_rank = config.q_lora_rank;
        let head_dim = config.head_dim;
        let max_scale_stride_m = max_m.div_ceil(4) * 4;
        let scale_cols = max_k.div_ceil(128);
        Ok(Self {
            input_fp8: ctx
                .stream
                .alloc_zeros::<u8>(max_m.checked_mul(max_k).ok_or_else(|| {
                    anyhow!(
                        "DSv4 prefill DeepGEMM linear input scratch overflow: M={} K={}",
                        max_m,
                        max_k
                    )
                })?)
                .map_err(|e| anyhow!("DSv4 prefill DeepGEMM linear input alloc failed: {e}"))?,
            input_scales: ctx
                .stream
                .alloc_zeros::<f32>(max_scale_stride_m.checked_mul(scale_cols).ok_or_else(
                    || {
                        anyhow!(
                            "DSv4 prefill DeepGEMM linear scale scratch overflow: M={} K={}",
                            max_scale_stride_m,
                            max_k
                        )
                    },
                )?)
                .map_err(|e| anyhow!("DSv4 prefill DeepGEMM linear scales alloc failed: {e}"))?,
            qkv_raw: unsafe { HiddenStates::uninit(ctx, q_lora_rank + head_dim, max_m)? },
            active_experts: ctx
                .stream
                .clone_htod(&[0_i32])
                .map_err(|e| anyhow!("DSv4 prefill DeepGEMM active_experts H2D failed: {e}"))?,
            active_offsets: ctx
                .stream
                .clone_htod(&[0_i32])
                .map_err(|e| anyhow!("DSv4 prefill DeepGEMM active_offsets H2D failed: {e}"))?,
            active_counts: ctx
                .stream
                .clone_htod(&[0_i32])
                .map_err(|e| anyhow!("DSv4 prefill DeepGEMM active_counts H2D failed: {e}"))?,
            max_m,
            max_k,
            max_scale_stride_m,
            hidden_dim: config.hidden_size,
            q_lora_rank,
            head_dim,
        })
    }
}

/// Query-dimension tile for the official DSA indexer prefill. The logits scratch
/// is `TILE × compressed_capacity` (f32); tiling the query axis keeps it bounded
/// (e.g. compress_ratio=4 @ 900K ctx: 4096 × 225024 × 4B ≈ 3.7 GB instead of ~810 GB).
/// This is the only path — long prompts loop in tiles, never materialize full-N logits.
///
/// Per-layer (each CSA layer owns its own `Dsv4DsaOfficialState` and thus its own
/// `logits` scratch — no cross-layer sharing, which is unsafe under this codebase's
/// disabled event-tracking + forward-level keepalive). The tile is 1024 (not 4096) so
/// all ~43 per-layer `logits` buffers fit at 900K: cr=4 → 1024 × ~225024 × 4B ≈ 0.92 GB
/// per cr=4 layer × ~20 such layers ≈ 18.4 GB, within the ~31 GB free budget. The 4096
/// tile OOMs (4096 × ~225024 × 4B ≈ 3.7 GB/layer × ~20 ≈ 74 GB).
/// `csa_select_official` loops sub-chunks when a forward passes more than `query_tile`
/// query tokens, so correctness is unchanged — just more sub-iterations.
const DSV4_DSA_PREFILL_QUERY_TILE: usize = 1024;

/// Query/token (M) dimension bound for DSv4 per-layer prefill scratch buffers
/// (e.g. [`Dsv4PrefillDeepGemmLinearScratch`]). Under chunked prefill the layer
/// forward only ever processes `chunked_prefill_size` query tokens per call
/// (scheduler default 2048), so M-dimension scratch must be sized by this chunk
/// bound — NOT by the slot's full `max_seq_len` context — or long-context prompts
/// (e.g. 900K) OOM allocating M×K activation buffers they never fill. Context/KV
/// dimensions (K=hidden_dim, compressed capacity, per-slot KV) keep `max_seq_len`
/// sizing. `chunked_prefill_size <= DSV4_PREFILL_QUERY_CHUNK` is asserted at each
/// scratch-writing call site; the one-shot `dsv4_parity` example at long context
/// is not the chunked-prefill path and is expected to trip that assert.
const DSV4_PREFILL_QUERY_CHUNK: usize = 4096;

/// Per-(slot, CSA-layer) STATEFUL half of the official DSA selector.
///
/// Only the pieces that carry cross-call state live here: the `rotated_keys`
/// mirror (incrementally written as compressed keys arrive), the
/// `packed_rows` progress counter, and the slot's key-cache band binding.
/// Every per-forward scratch buffer and every constant table is shared across
/// slots AND layers in [`Dsv4DsaSharedScratch`] (issue #67 — the per-slot ×
/// per-layer copies of the `logits` tile alone made 256K boot impossible).
struct Dsv4DsaOfficialState {
    slot_idx: usize,
    key_cache_len: usize,
    rotated_keys: CudaSlice<half::bf16>,
    packed_rows: usize,
}

impl Dsv4DsaOfficialState {
    fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        compress_ratio: usize,
        max_seq_len: usize,
        slot_idx: usize,
        pool: &Dsv4LayerKvLayout,
    ) -> Result<Self> {
        let compressed_capacity = max_seq_len.div_ceil(compress_ratio.max(1)).max(1);
        let key_cache_bytes = dsv4_dsa_key_cache_bytes(config, compress_ratio, max_seq_len)?;
        let range = pool.dsa_slot_range(slot_idx)?;
        ensure!(
            pool.dsa_key_cache.is_some()
                && range.len() == key_cache_bytes
                && range.len() == pool.dsa_slot_bytes
                && !range.is_empty(),
            "DSv4 official DSA shared slot band missing/invalid for slot {slot_idx}"
        );
        Ok(Self {
            slot_idx,
            key_cache_len: key_cache_bytes,
            rotated_keys: ctx
                .stream
                .alloc_zeros::<half::bf16>(compressed_capacity * config.index_head_dim)
                .map_err(|e| anyhow!("DSv4 official DSA rotated key alloc failed: {e}"))?,
            packed_rows: 0,
        })
    }

    fn reset(&mut self, ctx: &DeviceContext, pool: &mut Dsv4LayerKvLayout) -> Result<()> {
        self.packed_rows = 0;
        pool.reset_dsa_slot(ctx, self)?;
        Ok(())
    }
}

/// Per-MODEL shared half of the official DSA selector — ONE instance per
/// executor, shared across every CSA layer and every slot (issue #67).
///
/// Sharing safety: every kernel that touches these buffers is enqueued on the
/// single `ctx.stream`, so a later (slot, layer) call's writes are
/// stream-ordered after the earlier call's reads — the overwrite-before-read
/// discipline that already held per-tile within one call holds across calls
/// for free. The disabled-event-tracking hazard is about FREEING device memory
/// while async kernels may still touch it; this scratch lives as long as the
/// KV adapter and is never dropped mid-forward, so no premature-reuse window
/// exists. Contents carry NO cross-call state:
///
/// - per-forward scratch, overwritten before every read: `logits`, `q_fp8`,
///   `weights`, `context_lens`, `positions`, `sched_meta`, `raw_indices`;
/// - constants of (config, compress_ratio, max_seq): `cache_locs`,
///   `page_table_identity`, `freqs_cis`.
///
/// The stateful per-(slot, layer) pieces stay in [`Dsv4DsaOfficialState`].
pub(crate) struct Dsv4DsaSharedScratch {
    cache_locs: CudaSlice<i64>,
    q_fp8: CudaSlice<u8>,
    weights: CudaSlice<f32>,
    context_lens: CudaSlice<i32>,
    positions: CudaSlice<i32>,
    page_table_identity: CudaSlice<i32>,
    freqs_cis: CudaSlice<f32>,
    sched_meta: CudaSlice<i32>,
    logits: CudaSlice<f32>,
    raw_indices: CudaSlice<i32>,
    max_tokens: usize,
    query_tile: usize,
    query_chunk: usize,
    compressed_capacity: usize,
    num_pages: usize,
    num_heads: usize,
    head_dim: usize,
    logits_stride: usize,
    num_sms: usize,
}

impl Dsv4DsaSharedScratch {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        compress_ratio: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        ensure!(
            config.index_head_dim == 128,
            "Official DSv4 DSA indexer requires index_head_dim=128, got {}",
            config.index_head_dim
        );
        ensure!(
            config.index_n_heads == 32 || config.index_n_heads == 64,
            "Official DSv4 DSA indexer requires 32/64 heads, got {}",
            config.index_n_heads
        );
        let compressed_capacity = max_seq_len.div_ceil(compress_ratio.max(1)).max(1);
        let page_size = 64usize;
        let num_pages = compressed_capacity.div_ceil(page_size).max(1);
        let max_tokens = max_seq_len.max(1);
        // Query-dimension scratch is tiled: bounded by `query_tile` regardless of how
        // many query tokens a single call passes. When token_count <= query_tile the
        // compute loop runs a single iteration (offset 0), behavior-identical to the
        // pre-tiling code. Key-dimension / full-N buffers (cache_locs, freqs_cis)
        // stay sized by max_tokens/compressed_capacity; `raw_indices` (the
        // per-forward topk output) is chunk-sized — see below.
        let query_tile = DSV4_DSA_PREFILL_QUERY_TILE.min(max_tokens);
        // `raw_indices` is the topk OUTPUT: written per forward over the forward's
        // query tokens and read only by the gated `ARLE_DSV4_CSA_DUMP` block for those
        // same <= chunk queries — never the full max_tokens context. Size it by the
        // chunked-prefill query bound (`DSV4_PREFILL_QUERY_CHUNK`), not `max_tokens`,
        // which at 900K x topk would be ~1.9 GB/layer. `csa_select_official` asserts
        // `q_i.seq_len <= query_chunk` before the tile loop.
        let query_chunk = DSV4_PREFILL_QUERY_CHUNK.min(max_tokens);
        let q_elems = query_tile
            .checked_mul(config.index_n_heads)
            .and_then(|v| v.checked_mul(config.index_head_dim))
            .ok_or_else(|| anyhow!("DSv4 official DSA q scratch size overflow"))?;
        let logits_stride = compressed_capacity.div_ceil(256) * 256;
        let logits_elems = query_tile
            .checked_mul(logits_stride)
            .ok_or_else(|| anyhow!("DSv4 official DSA logits scratch size overflow"))?;
        let cache_locs_h: Vec<i64> = (0..compressed_capacity)
            .map(|v| i64::try_from(v).expect("compressed capacity fits i64"))
            .collect();
        let freqs_cis_h = dsv4_dsa_freqs_cis_real(config, compress_ratio, max_seq_len)?;
        let page_table_elems = query_tile
            .checked_mul(num_pages)
            .ok_or_else(|| anyhow!("DSv4 official DSA page table size overflow"))?;
        let mut page_table_h = Vec::with_capacity(page_table_elems);
        for _ in 0..query_tile {
            page_table_h
                .extend((0..num_pages).map(|v| i32::try_from(v).expect("page table fits i32")));
        }
        let num_sms = std::env::var("ARLE_DSV4_DSA_INDEXER_SMS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(78);
        Ok(Self {
            cache_locs: ctx
                .stream
                .clone_htod(&cache_locs_h)
                .map_err(|e| anyhow!("DSv4 official DSA cache loc upload failed: {e}"))?,
            q_fp8: ctx
                .stream
                .alloc_zeros::<u8>(q_elems)
                .map_err(|e| anyhow!("DSv4 official DSA q fp8 alloc failed: {e}"))?,
            weights: ctx
                .stream
                .alloc_zeros::<f32>(query_tile * config.index_n_heads)
                .map_err(|e| anyhow!("DSv4 official DSA weights alloc failed: {e}"))?,
            context_lens: ctx
                .stream
                .alloc_zeros::<i32>(query_tile)
                .map_err(|e| anyhow!("DSv4 official DSA context lens alloc failed: {e}"))?,
            positions: ctx
                .stream
                .alloc_zeros::<i32>(query_tile)
                .map_err(|e| anyhow!("DSv4 official DSA positions alloc failed: {e}"))?,
            page_table_identity: ctx
                .stream
                .clone_htod(&page_table_h)
                .map_err(|e| anyhow!("DSv4 official DSA page table upload failed: {e}"))?,
            freqs_cis: ctx
                .stream
                .clone_htod(&freqs_cis_h)
                .map_err(|e| anyhow!("DSv4 official DSA freqs_cis upload failed: {e}"))?,
            sched_meta: ctx
                .stream
                .alloc_zeros::<i32>((num_sms + 1) * 2)
                .map_err(|e| anyhow!("DSv4 official DSA sched meta alloc failed: {e}"))?,
            logits: ctx
                .stream
                .alloc_zeros::<f32>(logits_elems)
                .map_err(|e| anyhow!("DSv4 official DSA logits alloc failed: {e}"))?,
            raw_indices: ctx
                .stream
                .alloc_zeros::<i32>(query_chunk * config.index_topk)
                .map_err(|e| anyhow!("DSv4 official DSA raw indices alloc failed: {e}"))?,
            max_tokens,
            query_tile,
            query_chunk,
            compressed_capacity,
            num_pages,
            num_heads: config.index_n_heads,
            head_dim: config.index_head_dim,
            logits_stride,
            num_sms,
        })
    }
}

/// Device bytes of the ONE [`Dsv4DsaSharedScratch`] (per model, NOT per slot).
/// MUST mirror [`Dsv4DsaSharedScratch::new`]'s allocations (kept adjacent so
/// drift is visible). Feeds `Dsv4Model::kv_budget_num_slots` as a one-off
/// subtraction from the budget.
pub(crate) fn dsv4_dsa_shared_scratch_bytes(
    config: &DeepSeekV4Config,
    compress_ratio: usize,
    max_seq_len: usize,
) -> usize {
    let cc = max_seq_len.div_ceil(compress_ratio.max(1)).max(1);
    let num_pages = cc.div_ceil(64).max(1);
    let query_tile = DSV4_DSA_PREFILL_QUERY_TILE.min(max_seq_len.max(1));
    let query_chunk = DSV4_PREFILL_QUERY_CHUNK.min(max_seq_len.max(1));
    let logits_stride = cc.div_ceil(256) * 256;
    let logits = query_tile.saturating_mul(logits_stride).saturating_mul(4);
    let cache_locs = cc.saturating_mul(8);
    let q_fp8 = query_tile
        .saturating_mul(config.index_n_heads)
        .saturating_mul(config.index_head_dim);
    let weights = query_tile
        .saturating_mul(config.index_n_heads)
        .saturating_mul(4);
    let lens_positions = query_tile.saturating_mul(8);
    let page_table = query_tile.saturating_mul(num_pages).saturating_mul(4);
    // freqs_cis covers max_tokens positions x rope dim (f32).
    let freqs_cis = max_seq_len
        .saturating_mul(config.qk_rope_head_dim)
        .saturating_mul(4);
    let raw_indices = query_chunk
        .saturating_mul(config.index_topk)
        .saturating_mul(4);
    logits
        .saturating_add(cache_locs)
        .saturating_add(q_fp8)
        .saturating_add(weights)
        .saturating_add(lens_positions)
        .saturating_add(page_table)
        .saturating_add(freqs_cis)
        .saturating_add(raw_indices)
}

/// Device bytes of ONE per-(slot, CSA-layer) [`Dsv4DsaOfficialState`] — the
/// stateful `rotated_keys` mirror. Feeds the per-slot term of
/// `Dsv4Model::kv_budget_num_slots`.
pub(crate) fn dsv4_dsa_rotated_keys_bytes(
    config: &DeepSeekV4Config,
    compress_ratio: usize,
    max_seq_len: usize,
) -> usize {
    let cc = max_seq_len.div_ceil(compress_ratio.max(1)).max(1);
    cc.saturating_mul(config.index_head_dim).saturating_mul(2)
}

pub(crate) fn dsv4_dsa_key_cache_bytes(
    config: &DeepSeekV4Config,
    compress_ratio: usize,
    max_seq_len: usize,
) -> Result<usize> {
    let compressed_capacity = max_seq_len.div_ceil(compress_ratio).max(1);
    let page_size = 64usize;
    let num_pages = compressed_capacity.div_ceil(page_size).max(1);
    num_pages
        .checked_mul(page_size * (config.index_head_dim + std::mem::size_of::<f32>()))
        .ok_or_else(|| anyhow!("DSv4 official DSA key cache size overflow"))
}

fn dsv4_dsa_freqs_cis_real(
    config: &DeepSeekV4Config,
    compress_ratio: usize,
    max_seq_len: usize,
) -> Result<Vec<f32>> {
    ensure!(
        config.qk_rope_head_dim.is_multiple_of(2),
        "DSv4 official DSA RoPE dim {} must be even",
        config.qk_rope_head_dim
    );
    let dim = config.qk_rope_head_dim;
    let half = dim / 2;
    let base = if compress_ratio > 0 {
        config.compress_rope_theta
    } else {
        config.rope_theta
    } as f64;
    let original_seq_len = if compress_ratio > 0 {
        config.rope_parameters.original_max_position_embeddings
    } else {
        0
    };
    let factor = config.rope_parameters.factor as f64;
    let beta_fast = config.rope_parameters.beta_fast as f64;
    let beta_slow = config.rope_parameters.beta_slow as f64;
    let mut inv_freq = Vec::with_capacity(half);
    for pair in 0..half {
        let mut freq = 1.0f64 / base.powf((2 * pair) as f64 / dim as f64);
        if original_seq_len > 0 {
            let find_correction_dim = |num_rotations: f64| -> f64 {
                dim as f64
                    * ((original_seq_len as f64 / (num_rotations * 2.0 * std::f64::consts::PI))
                        .ln())
                    / (2.0 * base.ln())
            };
            let low = find_correction_dim(beta_fast).floor().max(0.0);
            let high = find_correction_dim(beta_slow).ceil().min((dim - 1) as f64);
            let mut high_adj = high;
            if (low - high_adj).abs() < f64::EPSILON {
                high_adj += 0.001;
            }
            let ramp = ((pair as f64 - low) / (high_adj - low)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            freq = freq / factor * (1.0 - smooth) + freq * smooth;
        }
        inv_freq.push(freq);
    }

    let mut out = vec![0.0f32; max_seq_len * dim];
    for pos in 0..max_seq_len {
        for pair in 0..half {
            let theta = pos as f64 * inv_freq[pair];
            out[pos * dim + 2 * pair] = theta.cos() as f32;
            out[pos * dim + 2 * pair + 1] = theta.sin() as f32;
        }
    }
    Ok(out)
}

pub(crate) struct Dsv4LayerAttentionState {
    sw_window_cache: CudaSlice<half::bf16>,
    compressor: Option<Dsv4CompressorState>,
    indexer: Option<Dsv4CompressorState>,
    flashmla: Option<Dsv4FlashMlaDecodeState>,
    fused_wqkv: Option<Dsv4FusedWqkvDecodeScratch>,
    prefill_linear: Option<Dsv4PrefillDeepGemmLinearScratch>,
    dsa_official: Option<Dsv4DsaOfficialState>,
}

/// Per-layer K+1-slot snapshot of the speculative-verify ring writes (frozen-KV
/// MTP P1-2). The depth-K verify forward over `[pending, d0..d_{depth-1}]` at
/// absolute positions `[start_pos .. start_pos+depth]` writes the BF16 SW ring
/// (`sw_window_cache`) and the FP8 ring (FlashMLA pool, block-table routed) for
/// each of the K+1 tokens — the draft chain needs its own KV. On a partial-accept
/// reject where `start_pos >= sliding_window`, the rejected drafts' ring writes
/// alias still-active window slots; truncate alone only resets lengths, so the
/// next decode would read corrupted slots. We snapshot ALL K+1 verify slots
/// pre-verify and, after the commit truncate, restore the rejected tail
/// `(accepted_n+1 ..= depth)`. The accepted slots `[0 ..= accepted_n]` are left
/// to the commit re-forward (which overwrites them).
///
/// §0.1 buffer enumeration — every verify-mutated, NON-frozen device buffer this
/// snapshot covers:
/// - `sw_window_cache` (BF16 SW ring): `sw_slots` holds K+1 head-dim slots.
/// - FP8 ring (FlashMLA pool data+scale bytes): `fp8_slots` holds K+1
///   `fp8_bytes_per_token` slots (`None` when this layer has no FlashMLA/FP8).
/// - `flash.fp8_kv_comp_packed_rows` (counter): `fp8_packed_rows_before` saves
///   the pre-verify value so the re-forward re-advances from the correct base.
///
/// Buffers pre-allocated ONCE at slot construction (no per-step `CudaSlice`
/// alloc — alloc churn + the disabled-event-tracking premature-free hazard). The
/// per-slot D2D copy math is recovered verbatim from the deleted single-slot
/// rollback snapshot (`Dsv4LayerAttentionSnapshot`, git 7f305a1e): SW slot =
/// `(draft_abs_pos % sliding_window) * head_dim`; FP8 via `fp8_sw_offsets` →
/// `physical_page(table, logical_page)` → block_base → data/scale byte D2D. The
/// only change here: loop over the K+1 positions and store K+1 slots.
pub(crate) struct Dsv4SpecRingSnapshot {
    /// `(max_depth+1) * head_dim` BF16 SW ring slots, slot `i` at `[i*head_dim..]`.
    sw_slots: CudaSlice<half::bf16>,
    /// `(max_depth+1) * fp8_bytes_per_token` FP8 ring slots (data+scale per slot);
    /// `None` when this layer has no FlashMLA/FP8 ring.
    fp8_slots: Option<CudaSlice<u8>>,
    /// `flash.fp8_kv_comp_packed_rows` captured once pre-verify; restored on reject.
    fp8_packed_rows_before: Option<usize>,
    /// `flash.fp8_kv_sw_bootstrapped` captured once pre-verify; restored on reject
    /// (P1-B). On the first spec decode after a long prefill the flag is still
    /// false, so the FP8 ring is unbootstrapped (stale) when captured. Restoring
    /// the flag means a wrap-reject re-bootstraps the whole window on the next
    /// decode (overwriting the stale FP8 bytes restore put back) instead of
    /// skipping the repack and reading corruption.
    fp8_bootstrapped_before: Option<bool>,
    /// Layout metadata `fp8_sw_offsets` needs — copied verbatim from the deleted
    /// single-slot snapshot struct's fields.
    head_dim: usize,
    sliding_window: usize,
    fp8_page_block_size: usize,
    fp8_token_data_bytes: usize,
    fp8_scale_bytes: usize,
    fp8_bytes_per_token: usize,
    /// Max draft depth this snapshot was sized for (K); valid slot count is K+1.
    max_depth: usize,
    /// Capture-time `start_pos`/`depth`, asserted in restore so a stale snapshot
    /// can never be replayed against a different verify window.
    captured_start_pos: usize,
    captured_depth: usize,
}

impl Dsv4SpecRingSnapshot {
    /// `(logical ring block, data offset in block, scale offset in block)` for
    /// one draft token's FP8 SW ring slot. Recovered verbatim from the deleted
    /// `Dsv4LayerAttentionSnapshot::fp8_sw_offsets` (git 7f305a1e). The block id
    /// is slot-LOGICAL — the caller translates it to a physical pool page through
    /// the slot's block table (#85 P2), so this math never bakes in band
    /// contiguity. Returns `None` when this layer has no FP8 ring.
    fn fp8_sw_offsets(&self, draft_abs_pos: usize) -> Option<(usize, usize, usize)> {
        self.fp8_slots.as_ref()?;
        let ring_idx = draft_abs_pos % self.sliding_window;
        let block_id = ring_idx / self.fp8_page_block_size;
        let row = ring_idx % self.fp8_page_block_size;
        let data_in_block = row * self.fp8_token_data_bytes;
        let scale_in_block =
            self.fp8_page_block_size * self.fp8_token_data_bytes + row * self.fp8_scale_bytes;
        Some((block_id, data_in_block, scale_in_block))
    }
}

/// Host-side image of one slot's per-layer device state for the whole-slot KV
/// tier (#84/#85 Route B, `docs/plans/2026-06-11-dsv4-whole-slot-kv-swap.md`).
/// Executor-internal, NOT byte-packed: plain host vectors per buffer.
/// Full-allocation copies by construction — extent-proof for v1; computing
/// written extents (e.g. `seq_len * 584B` of the FP8 band) is a perf TODO.
pub(crate) struct Dsv4LayerImage {
    sw_window_cache: Vec<half::bf16>,
    compressor: Option<Dsv4CompressorImage>,
    indexer: Option<Dsv4CompressorImage>,
    flashmla: Option<Dsv4FlashMlaImage>,
    dsa_official: Option<Dsv4DsaOfficialImage>,
}

/// Whole-slot host image of one [`Dsv4CompressorState`].
///
/// Unlike spec-decode rollback (truncate + re-forward, which only ever SHRINKS
/// a live slot so the rolled-back rows are overwritten by the next real
/// decode), this MUST carry `compressed.data`: a swapped-out slot is freed and
/// reused by another request, so every committed compressed row has to survive
/// in the image.
struct Dsv4CompressorImage {
    pending_kv: Vec<half::bf16>,
    pending_score: Vec<half::bf16>,
    prev_overlap_kv: Vec<half::bf16>,
    prev_overlap_score: Vec<half::bf16>,
    compressed_data: Vec<half::bf16>,
    compressed_seq_len: usize,
}

impl Dsv4CompressorImage {
    fn capture(ctx: &DeviceContext, state: &Dsv4CompressorState) -> Result<Self> {
        Ok(Self {
            pending_kv: ctx
                .stream
                .clone_dtoh(&state.pending_kv)
                .map_err(|e| anyhow!("DSv4 swap compressor pending kv D2H failed: {e}"))?,
            pending_score: ctx
                .stream
                .clone_dtoh(&state.pending_score)
                .map_err(|e| anyhow!("DSv4 swap compressor pending score D2H failed: {e}"))?,
            prev_overlap_kv: ctx
                .stream
                .clone_dtoh(&state.prev_overlap_kv)
                .map_err(|e| anyhow!("DSv4 swap compressor prev kv D2H failed: {e}"))?,
            prev_overlap_score: ctx
                .stream
                .clone_dtoh(&state.prev_overlap_score)
                .map_err(|e| anyhow!("DSv4 swap compressor prev score D2H failed: {e}"))?,
            compressed_data: ctx
                .stream
                .clone_dtoh(&state.compressed.data)
                .map_err(|e| anyhow!("DSv4 swap compressor compressed D2H failed: {e}"))?,
            compressed_seq_len: state.compressed.seq_len,
        })
    }

    fn restore_to(&self, ctx: &DeviceContext, state: &mut Dsv4CompressorState) -> Result<()> {
        ensure!(
            self.pending_kv.len() == state.pending_kv.len()
                && self.pending_score.len() == state.pending_score.len()
                && self.prev_overlap_kv.len() == state.prev_overlap_kv.len()
                && self.prev_overlap_score.len() == state.prev_overlap_score.len()
                && self.compressed_data.len() == state.compressed.data.len(),
            "DSv4 swap compressor image shape mismatch"
        );
        ctx.stream
            .memcpy_htod(&self.pending_kv, &mut state.pending_kv)
            .map_err(|e| anyhow!("DSv4 swap compressor pending kv H2D failed: {e}"))?;
        ctx.stream
            .memcpy_htod(&self.pending_score, &mut state.pending_score)
            .map_err(|e| anyhow!("DSv4 swap compressor pending score H2D failed: {e}"))?;
        ctx.stream
            .memcpy_htod(&self.prev_overlap_kv, &mut state.prev_overlap_kv)
            .map_err(|e| anyhow!("DSv4 swap compressor prev kv H2D failed: {e}"))?;
        ctx.stream
            .memcpy_htod(&self.prev_overlap_score, &mut state.prev_overlap_score)
            .map_err(|e| anyhow!("DSv4 swap compressor prev score H2D failed: {e}"))?;
        ctx.stream
            .memcpy_htod(&self.compressed_data, &mut state.compressed.data)
            .map_err(|e| anyhow!("DSv4 swap compressor compressed H2D failed: {e}"))?;
        state.compressed.seq_len = self.compressed_seq_len;
        Ok(())
    }
}

/// Whole-slot host image of one [`Dsv4FlashMlaDecodeState`]: the two mutable
/// host scalars plus the slot's pages of the shared FP8 KV pool.
struct Dsv4FlashMlaImage {
    fp8_kv_sw_bootstrapped: bool,
    fp8_kv_comp_packed_rows: usize,
    /// Whole-band payload in slot-logical block order — the per-page host
    /// images `TokenKVPool::copy_pages_to_host` produces. Slot-agnostic:
    /// restore re-resolves the TARGET slot's block table. Perf TODO
    /// unchanged: only `seq_len`-derived rows are written; v1 copies every
    /// page (extent-proof by construction).
    fp8_kv_pool_pages: Vec<u8>,
}

impl Dsv4FlashMlaImage {
    fn capture(
        ctx: &DeviceContext,
        pool: &Dsv4LayerKvLayout,
        flash: &Dsv4FlashMlaDecodeState,
    ) -> Result<Self> {
        // Table-routed (#85 P2): the image is the slot's PAGES (page-table lookup),
        // moved by the pool's own per-page transport — the same
        // `copy_pages_to_host` the page tier (#82/#83) consumes — so this
        // path stays valid when Stage B fragments the table.
        let table = pool.flashmla_page_table(flash.slot_idx)?.to_vec();
        let payload = pool
            .flashmla_pool()?
            .copy_pages_to_host(ctx, &table)
            .map_err(|e| anyhow!("DSv4 swap FlashMLA pool page D2H failed: {e}"))?;
        ensure!(
            payload.len() == flash.fp8_kv_pool_len,
            "DSv4 swap FlashMLA page payload {} != slot band bytes {}",
            payload.len(),
            flash.fp8_kv_pool_len
        );
        Ok(Self {
            fp8_kv_sw_bootstrapped: flash.fp8_kv_sw_bootstrapped,
            fp8_kv_comp_packed_rows: flash.fp8_kv_comp_packed_rows,
            fp8_kv_pool_pages: payload,
        })
    }

    fn restore_to(
        &self,
        ctx: &DeviceContext,
        pool: &mut Dsv4LayerKvLayout,
        flash: &mut Dsv4FlashMlaDecodeState,
    ) -> Result<()> {
        // Table-routed (#85 P2): restore lands on the TARGET slot's pages
        // (page-table lookup), mirroring capture; `copy_pages_from_host`
        // re-validates payload length against the page count.
        let table = pool.flashmla_page_table(flash.slot_idx)?.to_vec();
        ensure!(
            self.fp8_kv_pool_pages.len() == flash.fp8_kv_pool_len,
            "DSv4 swap FlashMLA restore payload {} != slot band bytes {}",
            self.fp8_kv_pool_pages.len(),
            flash.fp8_kv_pool_len
        );
        pool.flashmla_pool_mut()?
            .copy_pages_from_host(ctx, &table, &self.fp8_kv_pool_pages)
            .map_err(|e| anyhow!("DSv4 swap FlashMLA pool page H2D failed: {e}"))?;
        flash.fp8_kv_sw_bootstrapped = self.fp8_kv_sw_bootstrapped;
        flash.fp8_kv_comp_packed_rows = self.fp8_kv_comp_packed_rows;
        Ok(())
    }
}

/// Whole-slot host image of one [`Dsv4DsaOfficialState`]: the `packed_rows`
/// progress counter, the `rotated_keys` mirror, and the slot's exclusive band
/// of the shared official-DSA key cache.
struct Dsv4DsaOfficialImage {
    packed_rows: usize,
    /// Incrementally-written rotated-key mirror. Already-packed rows are not
    /// provably re-read (only the newly-packed slice feeds the cache store in
    /// `csa_select_official`), so this MIGHT be reconstructible — uncertain
    /// from source, so snapshot the full buffer (always safe).
    rotated_keys: Vec<half::bf16>,
    /// Full `dsa_slot_range` band: the FP8 indexer key cache the paged-MQA
    /// logits kernel reads in full every step. Perf TODO: written extent is
    /// `packed_rows`-derived; v1 copies the whole band.
    key_cache_slot: Vec<u8>,
}

impl Dsv4DsaOfficialImage {
    fn capture(
        ctx: &DeviceContext,
        pool: &Dsv4LayerKvLayout,
        official: &Dsv4DsaOfficialState,
    ) -> Result<Self> {
        let range = pool.dsa_slot_range(official.slot_idx)?;
        let pool_buf = pool
            .dsa_key_cache
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 swap DSA shared key-cache missing"))?;
        ensure!(
            range.end <= pool_buf.len() && range.len() == official.key_cache_len,
            "DSv4 swap DSA key-cache range {:?} invalid for pool_len={} slot_len={}",
            range,
            pool_buf.len(),
            official.key_cache_len
        );
        Ok(Self {
            packed_rows: official.packed_rows,
            rotated_keys: ctx
                .stream
                .clone_dtoh(&official.rotated_keys)
                .map_err(|e| anyhow!("DSv4 swap DSA rotated keys D2H failed: {e}"))?,
            key_cache_slot: ctx
                .stream
                .clone_dtoh(&pool_buf.slice(range))
                .map_err(|e| anyhow!("DSv4 swap DSA key-cache band D2H failed: {e}"))?,
        })
    }

    fn restore_to(
        &self,
        ctx: &DeviceContext,
        pool: &mut Dsv4LayerKvLayout,
        official: &mut Dsv4DsaOfficialState,
    ) -> Result<()> {
        ensure!(
            self.rotated_keys.len() == official.rotated_keys.len(),
            "DSv4 swap DSA rotated keys image len {} != state len {}",
            self.rotated_keys.len(),
            official.rotated_keys.len()
        );
        let range = pool.dsa_slot_range(official.slot_idx)?;
        let pool_buf = pool
            .dsa_key_cache
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 swap DSA shared key-cache missing"))?;
        ensure!(
            range.end <= pool_buf.len()
                && range.len() == official.key_cache_len
                && range.len() == self.key_cache_slot.len(),
            "DSv4 swap DSA restore range {:?} invalid for pool_len={} image_len={}",
            range,
            pool_buf.len(),
            self.key_cache_slot.len()
        );
        ctx.stream
            .memcpy_htod(&self.rotated_keys, &mut official.rotated_keys)
            .map_err(|e| anyhow!("DSv4 swap DSA rotated keys H2D failed: {e}"))?;
        let mut view = pool_buf.slice_mut(range);
        ctx.stream
            .memcpy_htod(&self.key_cache_slot, &mut view)
            .map_err(|e| anyhow!("DSv4 swap DSA key-cache band H2D failed: {e}"))?;
        official.packed_rows = self.packed_rows;
        Ok(())
    }
}

impl Dsv4LayerAttentionState {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        mode: DeepSeekV4AttentionMode,
        compress_ratio: usize,
        max_seq_len: usize,
        kv_arena: &Dsv4MlaKvArena,
        local_heads: usize,
        tp_world: usize,
        slot_idx: usize,
        pool: &Dsv4LayerKvLayout,
    ) -> Result<Self> {
        let sw_len = config.sliding_window * config.head_dim;
        ensure!(
            sw_len > 0,
            "DSv4 SW window cache len is zero (sliding_window={} head_dim={})",
            config.sliding_window,
            config.head_dim
        );
        let sw_window_cache = ctx
            .stream
            .alloc_zeros::<half::bf16>(sw_len)
            .map_err(|e| anyhow::anyhow!("DSv4 SW window cache alloc failed: {e}"))?;
        let overlap = compress_ratio < 16;
        let compressor = if mode == DeepSeekV4AttentionMode::SlidingWindow {
            None
        } else {
            Some(Dsv4CompressorState::new(
                ctx,
                config.head_dim,
                compress_ratio,
                overlap,
                max_seq_len,
            )?)
        };
        let indexer = if mode == DeepSeekV4AttentionMode::CompressedSparse {
            Some(Dsv4CompressorState::new(
                ctx,
                config.index_head_dim,
                compress_ratio,
                true,
                max_seq_len,
            )?)
        } else {
            None
        };
        let flashmla = if dsv4_flashmla_decode_alloc_enabled()? {
            Some(Dsv4FlashMlaDecodeState::new(
                ctx,
                config,
                mode,
                compress_ratio,
                max_seq_len,
                kv_arena,
                local_heads,
                tp_world,
                slot_idx,
                pool,
            )?)
        } else {
            None
        };
        let fused_wqkv = if dsv4_fused_wqkv_decode_alloc_enabled()? {
            Some(Dsv4FusedWqkvDecodeScratch::new(ctx, config)?)
        } else {
            None
        };
        let prefill_linear = if dsv4_fp8_linear_deepgemm_enabled()? {
            Some(Dsv4PrefillDeepGemmLinearScratch::new(
                ctx,
                config,
                max_seq_len,
            )?)
        } else {
            None
        };
        let dsa_official =
            if mode == DeepSeekV4AttentionMode::CompressedSparse && dsv4_dsa_official_enabled()? {
                Some(Dsv4DsaOfficialState::new(
                    ctx,
                    config,
                    compress_ratio,
                    max_seq_len,
                    slot_idx,
                    pool,
                )?)
            } else {
                None
            };
        Ok(Self {
            sw_window_cache,
            compressor,
            indexer,
            flashmla,
            fused_wqkv,
            prefill_linear,
            dsa_official,
        })
    }

    pub(crate) fn reset(
        &mut self,
        ctx: &DeviceContext,
        pool: &mut Dsv4LayerKvLayout,
    ) -> Result<()> {
        ctx.stream
            .memset_zeros(&mut self.sw_window_cache)
            .map_err(|e| anyhow::anyhow!("DSv4 SW window cache reset failed: {e}"))?;
        if let Some(compressor) = &mut self.compressor {
            compressor.reset(ctx)?;
        }
        if let Some(indexer) = &mut self.indexer {
            indexer.reset(ctx)?;
        }
        if let Some(flashmla) = &mut self.flashmla {
            flashmla.reset(ctx, pool)?;
        }
        if let Some(dsa) = &mut self.dsa_official {
            dsa.reset(ctx, pool)?;
        }
        Ok(())
    }

    /// Serialize this layer's COMPLETE per-slot device state into a host image
    /// (whole-slot KV swap, #84/#85 Route B). The caller (slot-level entry in
    /// `dsv4.rs`) syncs the stream once after all layers, so the D2H copies are
    /// complete before the engine frees the slot.
    ///
    /// §0.1 per-buffer verdict — every `Dsv4LayerAttentionState` field:
    /// - `sw_window_cache`: SNAPSHOT (full allocation). The live BF16 SW ring;
    ///   whole-buffer copy is extent-proof by construction (the EAGLE lesson:
    ///   ring self-heal only holds below `sliding_window`, so never infer a
    ///   written extent).
    /// - `compressor` / `indexer` ([`Dsv4CompressorState`]): SNAPSHOT (full) —
    ///   `pending_kv`/`pending_score`/`prev_overlap_kv`/`prev_overlap_score`
    ///   partial-row accumulators, plus `compressed.data` + `compressed.seq_len`
    ///   (see [`Dsv4CompressorImage`] for why the rollback snapshot's data-skip
    ///   does NOT apply to swap).
    /// - `flashmla` ([`Dsv4FlashMlaDecodeState`]), field by field:
    ///   - `fp8_kv_sw_bootstrapped` / `fp8_kv_comp_packed_rows`: SNAPSHOT — the
    ///     only host scalars written after init (`flashmla_pack_sw_ring`,
    ///     `flashmla_pack_compressed_delta`; everything else is set once in
    ///     `new`/`init_constant_sched_meta`).
    ///   - shared FP8 pool pages (the slot's `flashmla_page_table` into the
    ///     packed `TokenKVPool`): SNAPSHOT via `copy_pages_to_host` (every
    ///     page; perf TODO: written extent is `seq_len`-derived).
    ///   - `sw_bulk_block_ids`/`sw_bulk_rows`: SCRATCH — written only in
    ///     `flashmla_pack_sw_ring` (ring-identity constants) immediately before
    ///     their single read in the same call; never read elsewhere.
    ///   - `one_block_id`/`one_row`: SCRATCH — device-written from
    ///     `start_pos_device` at the top of every `flashmla_pack_one_sw_token`
    ///     before their single read in the same call.
    ///   - `comp_block_ids`/`comp_rows`: SCRATCH — H2D-written immediately
    ///     before their single read inside `flashmla_pack_compressed_delta`.
    ///   - `indices`: SCRATCH — the unified top-k select writes it each decode
    ///     step before the sparse-decode kernel reads it in the same step.
    ///   - `topk_length`/`sched_meta`/`num_splits`: CONSTANT-after-init
    ///     (`init_constant_sched_meta` — slot-shape constants, written once).
    ///   - `lse_out`/`lse_accum`/`o_accum`: SCRATCH — decode-kernel split
    ///     accumulators/outputs, fully overwritten per launch before read.
    ///   - `tp_gathered_q`/`tp_packed_q`/`tp_full_out`: SCRATCH — per-step Q
    ///     gather/output staging, written each decode step before read.
    ///   - `slot_idx`/`fp8_kv_pool_len`/`sw_blocks`/`comp_blocks`/
    ///     `max_compressed_keys`/`topk_unified`/`num_sm_parts`/
    ///     `fixed_overhead_num_blocks`/`block_size_topk`: CONSTANT-after-init.
    /// - `fused_wqkv` ([`Dsv4FusedWqkvDecodeScratch`]): SCRATCH — `input_fp8`/
    ///   `input_scales`/`qkv_raw` are quantize→GEMM staging overwritten from
    ///   the step's activations before every read; `active_experts`/
    ///   `active_offsets`/`active_counts` are `[0]`/`[0]`/`[1]` constants.
    /// - `prefill_linear` ([`Dsv4PrefillDeepGemmLinearScratch`]): SCRATCH —
    ///   same quantize→GEMM staging pattern, M-chunk-bounded, fully written
    ///   from the chunk's activations before read each call.
    /// - `dsa_official` ([`Dsv4DsaOfficialState`]): SNAPSHOT — `packed_rows`
    ///   (host progress counter), `rotated_keys` (incremental mirror; old rows
    ///   not provably re-read → uncertain → snapshot full, always safe), and
    ///   the shared `dsa_key_cache` band (`dsa_slot_range(slot_idx)`), which
    ///   the paged-MQA logits kernel reads in full every step.
    /// - [`Dsv4DsaSharedScratch`] (adapter-level, NOT per-slot): NO SNAPSHOT —
    ///   its doc proves "contents carry NO cross-call state" (per-forward
    ///   scratch overwritten before read + config constants), shared across
    ///   every slot and layer.
    pub(crate) fn swap_out_image(
        &self,
        ctx: &DeviceContext,
        pool: &Dsv4LayerKvLayout,
    ) -> Result<Dsv4LayerImage> {
        Ok(Dsv4LayerImage {
            sw_window_cache: ctx
                .stream
                .clone_dtoh(&self.sw_window_cache)
                .map_err(|e| anyhow!("DSv4 swap SW window D2H failed: {e}"))?,
            compressor: self
                .compressor
                .as_ref()
                .map(|state| Dsv4CompressorImage::capture(ctx, state))
                .transpose()?,
            indexer: self
                .indexer
                .as_ref()
                .map(|state| Dsv4CompressorImage::capture(ctx, state))
                .transpose()?,
            flashmla: self
                .flashmla
                .as_ref()
                .map(|flash| Dsv4FlashMlaImage::capture(ctx, pool, flash))
                .transpose()?,
            dsa_official: self
                .dsa_official
                .as_ref()
                .map(|official| Dsv4DsaOfficialImage::capture(ctx, pool, official))
                .transpose()?,
        })
    }

    /// Exact inverse of [`Self::swap_out_image`] into (possibly another) slot's
    /// state at the same per-buffer granularity. Every stateful buffer is fully
    /// rewritten from the image, so leftover state from a previous occupant of
    /// the target slot cannot leak; scratch buffers stay untouched (overwritten
    /// before read per the verdicts above). The slot-level caller syncs once
    /// after all layers, before the engine resumes decode / drops the image.
    pub(crate) fn swap_in_image(
        &mut self,
        ctx: &DeviceContext,
        pool: &mut Dsv4LayerKvLayout,
        image: &Dsv4LayerImage,
    ) -> Result<()> {
        ensure!(
            image.sw_window_cache.len() == self.sw_window_cache.len(),
            "DSv4 swap SW window image len {} != state len {}",
            image.sw_window_cache.len(),
            self.sw_window_cache.len()
        );
        ctx.stream
            .memcpy_htod(&image.sw_window_cache, &mut self.sw_window_cache)
            .map_err(|e| anyhow!("DSv4 swap SW window H2D failed: {e}"))?;
        match (&mut self.compressor, &image.compressor) {
            (Some(state), Some(image)) => image.restore_to(ctx, state)?,
            (None, None) => {}
            _ => bail!("DSv4 swap compressor image presence mismatch"),
        }
        match (&mut self.indexer, &image.indexer) {
            (Some(state), Some(image)) => image.restore_to(ctx, state)?,
            (None, None) => {}
            _ => bail!("DSv4 swap indexer image presence mismatch"),
        }
        match (&mut self.flashmla, &image.flashmla) {
            (Some(flash), Some(image)) => image.restore_to(ctx, pool, flash)?,
            (None, None) => {}
            _ => bail!("DSv4 swap FlashMLA image presence mismatch"),
        }
        match (&mut self.dsa_official, &image.dsa_official) {
            (Some(official), Some(image)) => image.restore_to(ctx, pool, official)?,
            (None, None) => {}
            _ => bail!("DSv4 swap DSA image presence mismatch"),
        }
        Ok(())
    }

    pub(crate) fn advance_decode_len(
        &mut self,
        mode: DeepSeekV4AttentionMode,
        ratio: usize,
        total_len: usize,
    ) {
        if mode == DeepSeekV4AttentionMode::SlidingWindow {
            return;
        }
        let compressed_rows = total_len / ratio;
        if let Some(compressor) = &mut self.compressor {
            compressor.compressed.seq_len = compressed_rows;
        }
        if let Some(indexer) = &mut self.indexer {
            indexer.compressed.seq_len = compressed_rows;
        }
    }

    pub(crate) fn truncate_decode_len(
        &mut self,
        mode: DeepSeekV4AttentionMode,
        ratio: usize,
        total_len: usize,
    ) {
        self.advance_decode_len(mode, ratio, total_len);
        // SGLang frozen-KV discipline: the official DSA key cache is a
        // deterministic function of committed `seq_len`. Truncating
        // sw/fp8/compressor/indexer back to `total_len` is enough for them, but
        // NOT for `dsa_official` — its `packed_rows` progress counter advances when a
        // speculative draft crosses a compression boundary (csa_select_official,
        // `packed_rows = indexer_rows_after`) and was never rolled back on draft
        // rejection. The stale draft key then stays in `dsa_key_cache` and the
        // next pack is skipped (`newly_packed == 0`), corrupting top-k selection
        // and accumulating into degenerate output. Clamp `packed_rows` DOWN to
        // the rolled-back compressed-row count so the next real decode re-packs
        // (overwrites) the row from the restored indexer KV — self-heal, no
        // snapshot (mirrors SGLang's seq_len-as-single-source-of-truth design).
        if let Some(dsa) = &mut self.dsa_official {
            let compressed_rows = total_len / ratio.max(1);
            dsa.packed_rows = dsa.packed_rows.min(compressed_rows);
        }
    }

    pub(crate) fn dump_mtp_rollback_state(
        &self,
        ctx: &DeviceContext,
        layer_idx: usize,
        label: &str,
        abs_len: usize,
    ) -> Result<()> {
        if !dsv4_mtp_rollback_dump_enabled() {
            return Ok(());
        }
        let sw = dsv4_checksum_bf16_slice(ctx, &self.sw_window_cache)?;
        let compressor_pending = self
            .compressor
            .as_ref()
            .map(|state| dsv4_checksum_bf16_slice(ctx, &state.pending_kv))
            .transpose()?;
        let compressor_prev = self
            .compressor
            .as_ref()
            .map(|state| dsv4_checksum_bf16_slice(ctx, &state.prev_overlap_kv))
            .transpose()?;
        let compressor_compressed = self
            .compressor
            .as_ref()
            .map(|state| dsv4_checksum_hidden(ctx, &state.compressed))
            .transpose()?;
        let compressor_seq = self
            .compressor
            .as_ref()
            .map(|state| state.compressed.seq_len)
            .unwrap_or(0);
        let indexer_pending = self
            .indexer
            .as_ref()
            .map(|state| dsv4_checksum_bf16_slice(ctx, &state.pending_kv))
            .transpose()?;
        let indexer_prev = self
            .indexer
            .as_ref()
            .map(|state| dsv4_checksum_bf16_slice(ctx, &state.prev_overlap_kv))
            .transpose()?;
        let indexer_compressed = self
            .indexer
            .as_ref()
            .map(|state| dsv4_checksum_hidden(ctx, &state.compressed))
            .transpose()?;
        let indexer_seq = self
            .indexer
            .as_ref()
            .map(|state| state.compressed.seq_len)
            .unwrap_or(0);
        eprintln!(
            "[dsv4-mtp-rollback-dump] label={label} layer={layer_idx} abs_len={abs_len} sw=({sw}) comp_seq={compressor_seq} comp_pending=({}) comp_prev=({}) comp_compressed=({}) index_seq={indexer_seq} index_pending=({}) index_prev=({}) index_compressed=({})",
            compressor_pending.map_or_else(|| "none".to_string(), |v| v.to_string()),
            compressor_prev.map_or_else(|| "none".to_string(), |v| v.to_string()),
            compressor_compressed.map_or_else(|| "none".to_string(), |v| v.to_string()),
            indexer_pending.map_or_else(|| "none".to_string(), |v| v.to_string()),
            indexer_prev.map_or_else(|| "none".to_string(), |v| v.to_string()),
            indexer_compressed.map_or_else(|| "none".to_string(), |v| v.to_string()),
        );
        Ok(())
    }

    /// Allocate this layer's K+1-slot spec-ring snapshot ONCE at slot
    /// construction (no per-step alloc). Sizes mirror the deleted single-slot
    /// `rollback_snapshot` (git 7f305a1e) ×(max_depth+1): the SW slot
    /// (`config.head_dim` BF16) and — when this layer has a FlashMLA decode
    /// state — the FP8 ring slot (`kv_arena.bytes_per_token` bytes, data+scale).
    pub(crate) fn alloc_spec_ring_snapshot(
        &self,
        ctx: &DeviceContext,
        config: &DeepSeekV4Config,
        kv_arena: &Dsv4MlaKvArena,
        max_depth: usize,
    ) -> Result<Dsv4SpecRingSnapshot> {
        let slots = max_depth
            .checked_add(1)
            .ok_or_else(|| anyhow!("DSv4 spec-ring max_depth {max_depth} overflow"))?;
        // FP8 token data bytes = NoPE FP8 (1 B/dim) + RoPE bf16 (2 B/dim);
        // scale bytes = bytes_per_token - data bytes. Verbatim from the deleted
        // `rollback_snapshot` (git 7f305a1e).
        let fp8_token_data_bytes = kv_arena
            .nope_dim
            .checked_add(kv_arena.rope_dim * std::mem::size_of::<half::bf16>())
            .ok_or_else(|| anyhow!("DSv4 spec-ring FP8 token data byte overflow"))?;
        ensure!(
            kv_arena.bytes_per_token >= fp8_token_data_bytes,
            "DSv4 spec-ring FP8 bytes/token {} smaller than token data bytes {}",
            kv_arena.bytes_per_token,
            fp8_token_data_bytes
        );
        let fp8_scale_bytes = kv_arena.bytes_per_token - fp8_token_data_bytes;
        Ok(Dsv4SpecRingSnapshot {
            sw_slots: ctx
                .stream
                .alloc_zeros::<half::bf16>(slots * config.head_dim)
                .map_err(|e| anyhow!("DSv4 spec-ring SW slots alloc failed: {e}"))?,
            fp8_slots: self
                .flashmla
                .as_ref()
                .map(|_| {
                    ctx.stream
                        .alloc_zeros::<u8>(slots * kv_arena.bytes_per_token)
                        .map_err(|e| anyhow!("DSv4 spec-ring FP8 slots alloc failed: {e}"))
                })
                .transpose()?,
            fp8_packed_rows_before: None,
            fp8_bootstrapped_before: None,
            head_dim: config.head_dim,
            sliding_window: config.sliding_window,
            fp8_page_block_size: kv_arena.page_block_size,
            fp8_token_data_bytes,
            fp8_scale_bytes,
            fp8_bytes_per_token: kv_arena.bytes_per_token,
            max_depth,
            captured_start_pos: 0,
            captured_depth: 0,
        })
    }

    /// Snapshot the K+1 verify ring slots BEFORE the frozen depth-K verify
    /// forward. For `i in 0..=depth`: D2D the SW slot for `draft_abs_pos =
    /// start_pos+i` into `snap.sw_slots[i*head_dim..]`, and — when this layer has
    /// an FP8 ring — D2D the FP8 data+scale for that position into
    /// `snap.fp8_slots[i*bytes_per_token..]`. Saves `flash.fp8_kv_comp_packed_rows`
    /// once. Borrows match the deleted code: `pool: &mut`, `flash: &`.
    pub(crate) fn capture_spec_rings(
        &self,
        ctx: &DeviceContext,
        pool: &mut Dsv4LayerKvLayout,
        snap: &mut Dsv4SpecRingSnapshot,
        start_pos: usize,
        depth: usize,
    ) -> Result<()> {
        ensure!(
            depth <= snap.max_depth,
            "DSv4 spec-ring capture depth {depth} exceeds snapshot max_depth {}",
            snap.max_depth
        );
        ensure!(
            snap.sliding_window > 0 && snap.head_dim > 0,
            "DSv4 spec-ring capture invalid shape sliding_window={} head_dim={}",
            snap.sliding_window,
            snap.head_dim
        );
        snap.fp8_packed_rows_before = self.flashmla.as_ref().map(|f| f.fp8_kv_comp_packed_rows);
        snap.fp8_bootstrapped_before = self.flashmla.as_ref().map(|f| f.fp8_kv_sw_bootstrapped);
        for i in 0..=depth {
            let draft_abs_pos = start_pos + i;
            snap.capture_sw_slot(ctx, &self.sw_window_cache, i, draft_abs_pos)?;
            if let Some(flash) = &self.flashmla {
                snap.capture_fp8_slot(ctx, pool, flash, i, draft_abs_pos)?;
            }
        }
        snap.captured_start_pos = start_pos;
        snap.captured_depth = depth;
        Ok(())
    }

    /// Restore the REJECTED tail ring slots `(accepted_n+1 ..= depth)` AFTER the
    /// commit truncate and BEFORE the accepted-prefix re-forward. The accepted
    /// slots `[0 ..= accepted_n]` are left to the re-forward (which overwrites
    /// them). Restores `flash.fp8_kv_comp_packed_rows` to the pre-verify base.
    /// `pool: &mut`; the FP8 `slot_idx` is read from `self.flashmla` up front so
    /// the per-slot restore re-resolves the page table without a live `&flash`
    /// borrow across the `&mut sw_window_cache` restore.
    pub(crate) fn restore_spec_ring_tail(
        &mut self,
        ctx: &DeviceContext,
        pool: &mut Dsv4LayerKvLayout,
        snap: &Dsv4SpecRingSnapshot,
        start_pos: usize,
        accepted_n: usize,
        depth: usize,
    ) -> Result<()> {
        ensure!(
            snap.captured_start_pos == start_pos && snap.captured_depth == depth,
            "DSv4 spec-ring restore window mismatch captured=({},{}) restore=({start_pos},{depth})",
            snap.captured_start_pos,
            snap.captured_depth
        );
        ensure!(
            accepted_n <= depth,
            "DSv4 spec-ring restore accepted_n {accepted_n} exceeds depth {depth}"
        );
        // The FP8 ring is slot-pinned; its page table is keyed by the layer's
        // `flash.slot_idx`. Read it once up front so the per-slot restore can
        // re-resolve the physical page WITHOUT holding a `&self.flashmla` borrow
        // across the `&mut self.sw_window_cache` restore (mirrors the deleted
        // `restore_fp8_sw`, which re-read the table from `flash.slot_idx`).
        let fp8_slot_idx = self.flashmla.as_ref().map(|f| f.slot_idx);
        for i in (accepted_n + 1)..=depth {
            let draft_abs_pos = start_pos + i;
            snap.restore_sw_slot(ctx, &mut self.sw_window_cache, i, draft_abs_pos)?;
            if let Some(slot_idx) = fp8_slot_idx {
                snap.restore_fp8_slot(ctx, pool, slot_idx, i, draft_abs_pos)?;
            }
        }
        if let Some(flash) = &mut self.flashmla {
            if let Some(rows) = snap.fp8_packed_rows_before {
                flash.fp8_kv_comp_packed_rows = rows;
            }
            // P1-B: restore the bootstrap flag too. If it was false pre-verify the
            // captured FP8 slots are stale, so leaving the flag true would skip the
            // next decode's repack; restoring false forces a full re-bootstrap that
            // overwrites the stale bytes restore_sw/fp8 put back.
            if let Some(bootstrapped) = snap.fp8_bootstrapped_before {
                flash.fp8_kv_sw_bootstrapped = bootstrapped;
            }
        }
        Ok(())
    }
}

impl Dsv4SpecRingSnapshot {
    /// D2D the SW ring slot for `draft_abs_pos` into snapshot slot `i`. Ring
    /// offset = `(draft_abs_pos % sliding_window) * head_dim` — verbatim from the
    /// deleted `Dsv4LayerAttentionSnapshot::capture_sw` (git 7f305a1e).
    fn capture_sw_slot(
        &mut self,
        ctx: &DeviceContext,
        sw_window_cache: &CudaSlice<half::bf16>,
        i: usize,
        draft_abs_pos: usize,
    ) -> Result<()> {
        ensure!(
            self.sliding_window > 0 && self.head_dim > 0,
            "DSv4 spec-ring SW snapshot has invalid shape sliding_window={} head_dim={}",
            self.sliding_window,
            self.head_dim
        );
        let ring_idx = draft_abs_pos % self.sliding_window;
        let src_offset = ring_idx * self.head_dim;
        let dst_offset = i * self.head_dim;
        ensure!(
            src_offset + self.head_dim <= sw_window_cache.len()
                && dst_offset + self.head_dim <= self.sw_slots.len(),
            "DSv4 spec-ring SW slot out of range src={} dst={} head_dim={} cache_len={} slots_len={}",
            src_offset,
            dst_offset,
            self.head_dim,
            sw_window_cache.len(),
            self.sw_slots.len()
        );
        let src = sw_window_cache.slice(src_offset..src_offset + self.head_dim);
        let mut dst = self
            .sw_slots
            .slice_mut(dst_offset..dst_offset + self.head_dim);
        ctx.stream
            .memcpy_dtod(&src, &mut dst)
            .map_err(|e| anyhow!("DSv4 spec-ring SW slot D2D snapshot failed: {e}"))?;
        Ok(())
    }

    /// Restore the SW ring slot for `draft_abs_pos` from snapshot slot `i`.
    /// Verbatim ring math from the deleted `restore_sw`.
    fn restore_sw_slot(
        &self,
        ctx: &DeviceContext,
        sw_window_cache: &mut CudaSlice<half::bf16>,
        i: usize,
        draft_abs_pos: usize,
    ) -> Result<()> {
        let ring_idx = draft_abs_pos % self.sliding_window;
        let dst_offset = ring_idx * self.head_dim;
        let src_offset = i * self.head_dim;
        ensure!(
            dst_offset + self.head_dim <= sw_window_cache.len()
                && src_offset + self.head_dim <= self.sw_slots.len(),
            "DSv4 spec-ring SW restore out of range src={} dst={} head_dim={} cache_len={} slots_len={}",
            src_offset,
            dst_offset,
            self.head_dim,
            sw_window_cache.len(),
            self.sw_slots.len()
        );
        let src = self.sw_slots.slice(src_offset..src_offset + self.head_dim);
        let mut dst = sw_window_cache.slice_mut(dst_offset..dst_offset + self.head_dim);
        ctx.stream
            .memcpy_dtod(&src, &mut dst)
            .map_err(|e| anyhow!("DSv4 spec-ring SW slot D2D restore failed: {e}"))?;
        Ok(())
    }

    /// D2D the FP8 ring data+scale bytes for `draft_abs_pos` into snapshot slot
    /// `i`. Table-routed physical-page math verbatim from the deleted
    /// `capture_fp8_sw` (git 7f305a1e); early-returns when this layer has no FP8
    /// ring (`fp8_sw_offsets`/`fp8_slots` is `None`).
    fn capture_fp8_slot(
        &mut self,
        ctx: &DeviceContext,
        pool: &mut Dsv4LayerKvLayout,
        flash: &Dsv4FlashMlaDecodeState,
        i: usize,
        draft_abs_pos: usize,
    ) -> Result<()> {
        // `fp8_sw_offsets` returns `None` exactly when `fp8_slots` is `None`
        // (SW-only / non-FlashMLA layer), so this early-return also guards the
        // `as_mut().ok_or_else` below — verbatim shape from the deleted
        // `capture_fp8_sw`.
        let Some((logical_page, data_in_block, scale_in_block)) =
            self.fp8_sw_offsets(draft_abs_pos)
        else {
            return Ok(());
        };
        // Table-routed (#85 P2): the ring block's byte base is its PHYSICAL pool
        // page (block-table lookup), so this path stays valid when Stage B
        // fragments the table.
        let page = physical_page(pool.flashmla_page_table(flash.slot_idx)?, logical_page)?;
        let block_base = page as usize * (self.fp8_page_block_size * self.fp8_bytes_per_token);
        let data_offset = block_base + data_in_block;
        let scale_offset = block_base + scale_in_block;
        let pool_buf = pool.flashmla_pool_data()?;
        let slot_base = i * self.fp8_bytes_per_token;
        let slots = self
            .fp8_slots
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 spec-ring FP8 slots missing during capture"))?;
        ensure!(
            data_offset + self.fp8_token_data_bytes <= pool_buf.len()
                && scale_offset + self.fp8_scale_bytes <= pool_buf.len()
                && slot_base + self.fp8_bytes_per_token <= slots.len(),
            "DSv4 spec-ring FP8 slot out of range data={} scale={} pool_len={} slot_base={} slots_len={}",
            data_offset,
            scale_offset,
            pool_buf.len(),
            slot_base,
            slots.len()
        );
        let src_data = pool_buf.slice(data_offset..data_offset + self.fp8_token_data_bytes);
        let mut dst_data = slots.slice_mut(slot_base..slot_base + self.fp8_token_data_bytes);
        ctx.stream
            .memcpy_dtod(&src_data, &mut dst_data)
            .map_err(|e| anyhow!("DSv4 spec-ring FP8 data snapshot failed: {e}"))?;
        let src_scale = pool_buf.slice(scale_offset..scale_offset + self.fp8_scale_bytes);
        let mut dst_scale = slots
            .slice_mut(slot_base + self.fp8_token_data_bytes..slot_base + self.fp8_bytes_per_token);
        ctx.stream
            .memcpy_dtod(&src_scale, &mut dst_scale)
            .map_err(|e| anyhow!("DSv4 spec-ring FP8 scale snapshot failed: {e}"))?;
        Ok(())
    }

    /// Restore the FP8 ring data+scale bytes for `draft_abs_pos` from snapshot
    /// slot `i`. Table-routed physical-page math verbatim from the deleted
    /// `restore_fp8_sw`; early-returns when this layer has no FP8 ring. (The
    /// `flash` slot index is re-resolved through the table by the page lookup;
    /// the caller restores `fp8_kv_comp_packed_rows` separately.)
    fn restore_fp8_slot(
        &self,
        ctx: &DeviceContext,
        pool: &mut Dsv4LayerKvLayout,
        slot_idx: usize,
        i: usize,
        draft_abs_pos: usize,
    ) -> Result<()> {
        let Some((logical_page, data_in_block, scale_in_block)) =
            self.fp8_sw_offsets(draft_abs_pos)
        else {
            return Ok(());
        };
        let Some(slots) = &self.fp8_slots else {
            return Ok(());
        };
        // Table-routed (#85 P2): same physical-page translation as capture. The
        // table is re-resolved from the layer's `slot_idx` (read by the caller
        // before the `&mut sw_window_cache` borrow) — verbatim translation from
        // the deleted `restore_fp8_sw`.
        let page = physical_page(pool.flashmla_page_table(slot_idx)?, logical_page)?;
        let block_base = page as usize * (self.fp8_page_block_size * self.fp8_bytes_per_token);
        let data_offset = block_base + data_in_block;
        let scale_offset = block_base + scale_in_block;
        let pool_buf = pool.flashmla_pool_data_mut()?;
        let slot_base = i * self.fp8_bytes_per_token;
        ensure!(
            data_offset + self.fp8_token_data_bytes <= pool_buf.len()
                && scale_offset + self.fp8_scale_bytes <= pool_buf.len()
                && slot_base + self.fp8_bytes_per_token <= slots.len(),
            "DSv4 spec-ring FP8 restore out of range data={} scale={} pool_len={} slot_base={} slots_len={}",
            data_offset,
            scale_offset,
            pool_buf.len(),
            slot_base,
            slots.len()
        );
        let src_data = slots.slice(slot_base..slot_base + self.fp8_token_data_bytes);
        let mut dst_data = pool_buf.slice_mut(data_offset..data_offset + self.fp8_token_data_bytes);
        ctx.stream
            .memcpy_dtod(&src_data, &mut dst_data)
            .map_err(|e| anyhow!("DSv4 spec-ring FP8 data restore failed: {e}"))?;
        let src_scale = slots
            .slice(slot_base + self.fp8_token_data_bytes..slot_base + self.fp8_bytes_per_token);
        let mut dst_scale = pool_buf.slice_mut(scale_offset..scale_offset + self.fp8_scale_bytes);
        ctx.stream
            .memcpy_dtod(&src_scale, &mut dst_scale)
            .map_err(|e| anyhow!("DSv4 spec-ring FP8 scale restore failed: {e}"))?;
        Ok(())
    }
}

/// P2 commit fold (fast-path plan): commit the ACCEPTED verify prefix into ONE
/// layer's persistent state WITHOUT re-running the full forward. §0.1 mutated
/// buffers and their dispositions:
/// - compressor + indexer state (`pending/overlap/compressed{,seq_len}`):
///   re-ingested here by the same NON-frozen batched `compressor_forward` the
///   re-forward would have run, over the PERSISTED attn-normed rows.
/// - bf16 SW ring: re-derive `k_prepared` (wkv → kv_norm → rope at the chain
///   positions) from the persisted rows, then the same window roll the
///   prefill path uses.
/// - FP8 SW ring: strided pack of those K rows at `pos % sliding_window`
///   (table-routed, mirrors `flashmla_pack_sw_ring`).
/// - `fp8_kv_comp_packed_rows` / `dsa_official.packed_rows`: untouched —
///   the next decode's `flashmla_pack_compressed_delta` / `csa_select` bulk
///   paths self-heal off the advanced `compressed.seq_len`.
///
/// The Q-side compute is discarded (the verify already produced argmax and
/// hiddens); `q_dummy` feeds the prepare kernel's Q arm with zeros.
#[allow(clippy::too_many_arguments)]
pub(crate) fn commit_layer_fold(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    state: &mut Dsv4LayerAttentionState,
    pool: &mut Dsv4LayerKvLayout,
    gathered: &HiddenStates,
    start_pos: usize,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    let m = gathered.seq_len;
    ensure!(m > 0, "DSv4 commit fold needs at least the pending row");
    let head_dim = config.head_dim;
    let rope = &config.rope_parameters;
    let (rope_base, original_seq_len) = if compress_ratio > 0 {
        let osl = i32::try_from(rope.original_max_position_embeddings).map_err(|_| {
            anyhow!(
                "DSv4 original_max_position_embeddings {} overflows i32",
                rope.original_max_position_embeddings
            )
        })?;
        (config.compress_rope_theta, osl)
    } else {
        (config.rope_theta, 0i32)
    };

    // ── Compressor + indexer ingestion (compressed layers only), exactly the
    // calls the re-forward's mla_attention would have made, non-frozen.
    if mode != DeepSeekV4AttentionMode::SlidingWindow {
        let compressor = attention.compressor.as_ref().ok_or_else(|| {
            anyhow!("DSv4 commit fold: {mode:?} layer without compressor weights")
        })?;
        let overlap = compress_ratio < 16;
        let compressor_state = state
            .compressor
            .as_mut()
            .ok_or_else(|| anyhow!("DSv4 commit fold: {mode:?} layer without compressor state"))?;
        compressor_forward(
            ctx,
            config,
            compressor,
            gathered,
            compressor_state,
            head_dim,
            compress_ratio,
            overlap,
            start_pos,
            None,
            true,
            original_seq_len,
            keepalive,
        )?;
        if mode == DeepSeekV4AttentionMode::CompressedSparse {
            let indexer = attention
                .indexer
                .as_ref()
                .ok_or_else(|| anyhow!("DSv4 commit fold: CSA layer without indexer weights"))?;
            let use_official_dsa = dsv4_dsa_official_enabled()?;
            let indexer_rope_original_seq_len = if use_official_dsa {
                i32::try_from(config.rope_parameters.original_max_position_embeddings)
                    .map_err(|_| anyhow!("DSv4 commit fold indexer rope len overflows i32"))?
            } else {
                0
            };
            let indexer_state = state
                .indexer
                .as_mut()
                .ok_or_else(|| anyhow!("DSv4 commit fold: CSA layer without indexer state"))?;
            compressor_forward(
                ctx,
                config,
                &indexer.compressor,
                gathered,
                indexer_state,
                config.index_head_dim,
                compress_ratio,
                true,
                start_pos,
                None,
                use_official_dsa,
                indexer_rope_original_seq_len,
                keepalive,
            )?;
        }
    }

    // ── K re-derivation: wkv → kv_norm → rope at chain positions.
    let mut kv_raw = unsafe { HiddenStates::uninit(ctx, head_dim, m)? };
    dsv4_linear(ctx, &attention.wkv, gathered, &mut kv_raw)?;
    keepalive.keep_hidden(&kv_raw);
    let kv_normed = mla_rms_norm(ctx, &kv_raw, &attention.kv_norm, config.rms_norm_eps)?;
    keepalive.keep_hidden(&kv_normed);
    let local_width = attention.wq_b.rows;
    let local_heads = local_width / head_dim;
    let q_dummy = HiddenStates {
        data: ctx
            .stream
            .alloc_zeros::<half::bf16>(local_width * m)
            .map_err(|e| anyhow!("DSv4 commit fold q scratch alloc failed: {e}"))?,
        hidden_dim: local_width,
        seq_len: m,
    };
    let mut q_discard = unsafe { HiddenStates::uninit(ctx, local_width, m)? };
    let mut k_prepared = unsafe { HiddenStates::uninit(ctx, head_dim, m)? };
    {
        let (q_raw_ptr, _qr) = q_dummy.data.device_ptr(&ctx.stream);
        let (k_raw_ptr, _kr) = kv_normed.data.device_ptr(&ctx.stream);
        let (q_out_ptr, _qo) = q_discard.data.device_ptr_mut(&ctx.stream);
        let (k_out_ptr, _ko) = k_prepared.data.device_ptr_mut(&ctx.stream);
        // SAFETY: buffers sized above; q arm runs on zeros and is discarded.
        unsafe {
            ffi::dsv4_prepare_qk_cuda(
                q_raw_ptr as *const ffi::Half,
                k_raw_ptr as *const ffi::Half,
                q_out_ptr as *mut ffi::Half,
                k_out_ptr as *mut ffi::Half,
                m as i32,
                local_heads as i32,
                head_dim as i32,
                config.qk_rope_head_dim as i32,
                start_pos as i32,
                config.rms_norm_eps,
                rope_base,
                original_seq_len,
                rope.factor,
                rope.beta_fast,
                rope.beta_slow,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    keepalive.keep_hidden(&q_dummy);
    keepalive.keep_hidden(&q_discard);
    keepalive.keep_hidden(&k_prepared);

    // ── bf16 SW ring roll (chain shape — identical to the prefill path).
    update_bf16_sw_window(
        ctx,
        &mut state.sw_window_cache,
        &k_prepared,
        start_pos,
        None,
        config,
    )?;

    // ── FP8 SW ring pack for the accepted positions (table-routed strided
    // pack, mirrors flashmla_pack_sw_ring's math for m explicit slots).
    if let Some(flash) = &mut state.flashmla {
        let page_block_size = 64;
        let mut block_ids = Vec::with_capacity(m);
        let mut rows = Vec::with_capacity(m);
        {
            let table = pool.flashmla_page_table(flash.slot_idx)?;
            for i in 0..m {
                let slot_idx = (start_pos + i) % config.sliding_window;
                let page = physical_page(table, slot_idx / page_block_size)?;
                block_ids.push(
                    i32::try_from(page)
                        .map_err(|_| anyhow!("DSv4 commit fold FP8 page overflows i32"))?,
                );
                rows.push((slot_idx % page_block_size) as i32);
            }
        }
        ctx.stream
            .memcpy_htod(&block_ids, &mut flash.sw_bulk_block_ids)
            .map_err(|e| anyhow!("DSv4 commit fold FP8 block_ids H2D failed: {e}"))?;
        ctx.stream
            .memcpy_htod(&rows, &mut flash.sw_bulk_rows)
            .map_err(|e| anyhow!("DSv4 commit fold FP8 rows H2D failed: {e}"))?;
        let (k_ptr, _kg) = k_prepared.data.device_ptr(&ctx.stream);
        let pool_buf = pool.flashmla_pool_data_mut()?;
        let (pool_ptr, _pg) = pool_buf.device_ptr_mut(&ctx.stream);
        let nope_ptr = k_ptr;
        let rope_ptr = nope_ptr + (config.head_dim - config.qk_rope_head_dim) as u64 * 2;
        flash_kv::dsv4_fp8_kv_pack_strided_raw(
            ctx,
            nope_ptr,
            rope_ptr,
            pool_ptr,
            &flash.sw_bulk_block_ids,
            &flash.sw_bulk_rows,
            m,
            page_block_size,
            config.head_dim,
            config.head_dim,
        )?;
    }
    Ok(())
}

/// Device-side draft-tree topology for ONE spec-verify forward: per-row
/// absolute positions (`start_pos + depth`, repeats across siblings) and
/// per-row branch ancestors as flattened chunk-row indices (`[n, max_anc]`,
/// root included, self implicit, -1 padded). Feeds the positions-array
/// prepare-QK/inverse-RoPE kernels and `arle_flashmla_tree_build_indices`,
/// so ONE batched forward verifies the whole tree with every row attending
/// exactly its own branch — no ring writes, no per-row replay.
pub(crate) struct Dsv4TreeAttnMeta {
    pub(crate) positions: CudaSlice<i32>,
    pub(crate) ancestors: CudaSlice<i32>,
    pub(crate) max_anc: usize,
    pub(crate) n_rows: usize,
}

impl Dsv4TreeAttnMeta {
    /// Upload host topology. `ancestors[r]` lists row r's branch chunk-rows
    /// shallow→deep (root included, self excluded).
    pub(crate) fn new(
        ctx: &DeviceContext,
        positions: &[usize],
        ancestors: &[Vec<usize>],
    ) -> Result<Self> {
        let n = positions.len();
        ensure!(
            n > 0 && ancestors.len() == n,
            "DSv4 tree meta shape mismatch: positions={} ancestors={}",
            n,
            ancestors.len()
        );
        let max_anc = ancestors.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let pos_host: Vec<i32> = positions
            .iter()
            .map(|&p| i32::try_from(p).map_err(|_| anyhow!("DSv4 tree position {p} overflows i32")))
            .collect::<Result<_>>()?;
        let mut anc_host = vec![-1i32; n * max_anc];
        for (r, chain) in ancestors.iter().enumerate() {
            for (j, &a) in chain.iter().enumerate() {
                ensure!(a < n, "DSv4 tree ancestor row {a} out of {n} rows");
                anc_host[r * max_anc + j] = a as i32;
            }
        }
        Ok(Self {
            positions: ctx
                .stream
                .clone_htod(&pos_host)
                .map_err(|e| anyhow!("DSv4 tree positions H2D failed: {e}"))?,
            ancestors: ctx
                .stream
                .clone_htod(&anc_host)
                .map_err(|e| anyhow!("DSv4 tree ancestors H2D failed: {e}"))?,
            max_anc,
            n_rows: n,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn paged_attention(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &mut HiddenStates,
    k_batch: &mut HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    rms_eps: f32,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    if meta.seq_len == 1 {
        decode_attention(
            ctx,
            layer_idx,
            pool,
            q_batch,
            k_batch,
            v_batch,
            q_norm,
            k_norm,
            cos_cache,
            sin_cache,
            rms_eps,
            meta,
            num_q_heads,
            num_kv_heads,
            head_dim,
            out,
        )
    } else {
        prefill_attention(
            ctx,
            layer_idx,
            pool,
            q_batch,
            k_batch,
            v_batch,
            q_norm,
            k_norm,
            cos_cache,
            sin_cache,
            rms_eps,
            meta,
            num_q_heads,
            num_kv_heads,
            head_dim,
            out,
        )
    }
}

/// Re-materialize the quantized prefix rows of `layer_idx` into the shared
/// bf16 work buffers (pool→work dequant) before the prefill prep kernel
/// appends the new chunk's rows. The work buffer is shared across layers and
/// overwritten every layer/forward, and prefix pages may arrive via radix
/// attach / COW detach / tier promote — the quantized plane is the only
/// durable source, so the prefix is unconditionally re-materialized. K uses
/// the per-channel KIVI dequant (per-channel K quantize never writes
/// per-token K scales); V uses the per-token sibling.
fn refill_prefix_rows_to_work(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    meta: &PageMeta,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    if meta.start_pos == 0 {
        return Ok(());
    }
    let stream = &ctx.stream;
    let prefix_rows = meta.prefix_token_rows.as_ref().ok_or_else(|| {
        anyhow!(
            "quant KV prefill missing prefix_token_rows for start_pos={}",
            meta.start_pos
        )
    })?;
    let k_static_scales_ptr = pool
        .k_static_scales_ptr(layer_idx, stream)
        .ok_or_else(|| anyhow!("quant KV pool missing KIVI k_static_scales (layer {layer_idx})"))?;
    match pool.format {
        KVFormat::FP8E4M3 => {
            kv_quant::dequantize_paged_kv_fp8_per_channel_k_to_hnd(
                ctx,
                pool.k_data_ptr(layer_idx, stream),
                k_static_scales_ptr,
                pool.k_work_ptr(stream),
                prefix_rows,
                num_kv_heads,
                head_dim,
                pool.kv_dim,
                meta.start_pos,
            )?;
            kv_quant::dequantize_paged_kv_fp8_to_hnd(
                ctx,
                pool.v_data_ptr(layer_idx, stream),
                pool.v_scales_ptr(layer_idx, stream),
                pool.v_work_ptr(stream),
                prefix_rows,
                num_kv_heads,
                head_dim,
                pool.kv_dim,
                meta.start_pos,
            )
        }
        KVFormat::INT8 => {
            kv_quant::dequantize_paged_kv_int8_per_channel_k_to_hnd(
                ctx,
                pool.k_data_ptr(layer_idx, stream),
                k_static_scales_ptr,
                pool.k_work_ptr(stream),
                prefix_rows,
                num_kv_heads,
                head_dim,
                pool.kv_dim,
                meta.start_pos,
            )?;
            kv_quant::dequantize_paged_kv_int8_to_hnd(
                ctx,
                pool.v_data_ptr(layer_idx, stream),
                pool.v_scales_ptr(layer_idx, stream),
                pool.v_work_ptr(stream),
                prefix_rows,
                num_kv_heads,
                head_dim,
                pool.kv_dim,
                meta.start_pos,
            )
        }
        other => bail!("quant KV prefix refill does not support format {other:?}"),
    }
}

/// Quantize this forward's new bf16 rows work→pool for `layer_idx`,
/// calibrating the KIVI per-channel K scale table first if the layer's latch
/// is still unset. K goes through the per-channel quantize against the static
/// table; V keeps per-(token, head) scales.
fn calibrate_and_quantize_new_rows(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    meta: &PageMeta,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let stream = &ctx.stream;
    let new_rows = meta
        .new_token_rows
        .as_ref()
        .ok_or_else(|| anyhow!("quant KV forward missing new_token_rows"))?;
    let k_static_scales_ptr = pool
        .k_static_scales_ptr(layer_idx, stream)
        .ok_or_else(|| anyhow!("quant KV pool missing KIVI k_static_scales (layer {layer_idx})"))?;
    let batch = meta.seq_len;
    // Latch-once calibration is REQUIRED under chunked prefill: recalibrating
    // on a later chunk would rescale the table while earlier chunks' K bytes
    // remain quantized under the old scale, corrupting every prior row at
    // decode. First batch through the layer calibrates (absmax → finalize),
    // then the latch flips and the table is read-only.
    if !pool.k_kivi_calibrated[layer_idx].load(Ordering::Relaxed) {
        kv_quant::compute_k_per_channel_absmax(
            ctx,
            pool.k_work_ptr(stream),
            k_static_scales_ptr,
            new_rows,
            num_kv_heads,
            head_dim,
            pool.kv_dim,
            batch,
        )?;
        match pool.format {
            KVFormat::FP8E4M3 => kv_quant::finalize_k_per_channel_scales(
                ctx,
                k_static_scales_ptr,
                num_kv_heads * head_dim,
            )?,
            KVFormat::INT8 => kv_quant::finalize_k_per_channel_scales_int8(
                ctx,
                k_static_scales_ptr,
                num_kv_heads * head_dim,
            )?,
            other => bail!("quant KV calibration does not support format {other:?}"),
        }
        pool.k_kivi_calibrated[layer_idx].store(true, Ordering::Relaxed);
    }
    match pool.format {
        KVFormat::FP8E4M3 => {
            kv_quant::quantize_paged_kv_fp8_per_channel(
                ctx,
                pool.k_work_ptr(stream),
                pool.k_data_ptr(layer_idx, stream),
                k_static_scales_ptr,
                new_rows,
                num_kv_heads,
                head_dim,
                pool.kv_dim,
                batch,
            )?;
            kv_quant::quantize_paged_kv_fp8(
                ctx,
                pool.v_work_ptr(stream),
                pool.v_data_ptr(layer_idx, stream),
                pool.v_scales_ptr(layer_idx, stream),
                new_rows,
                num_kv_heads,
                head_dim,
                pool.kv_dim,
                batch,
            )?;
        }
        KVFormat::INT8 => {
            kv_quant::quantize_paged_kv_int8_per_channel(
                ctx,
                pool.k_work_ptr(stream),
                pool.k_data_ptr(layer_idx, stream),
                k_static_scales_ptr,
                new_rows,
                num_kv_heads,
                head_dim,
                pool.kv_dim,
                batch,
            )?;
            kv_quant::quantize_paged_kv_single(
                ctx,
                pool.v_work_ptr(stream),
                pool.v_data_ptr(layer_idx, stream),
                pool.v_scales_ptr(layer_idx, stream),
                new_rows,
                num_kv_heads,
                head_dim,
                pool.kv_dim,
                batch,
            )?;
        }
        other => bail!("quant KV new-row quantize does not support format {other:?}"),
    }
    Ok(())
}

/// Fused-dequant decode attention over the quantized pool planes (replaces
/// the TileLang bf16 kernel for INT8/FP8 pools). NOT
/// `decode_attention_varlen_fp8` — that kernel consumes per-token K scales
/// and is incompatible with per-channel K.
#[allow(clippy::too_many_arguments)]
fn run_quant_decode(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &HiddenStates,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    let stream = &ctx.stream;
    let quant_meta = meta
        .quant_decode_meta
        .as_ref()
        .ok_or_else(|| anyhow!("quant KV decode missing quant_decode_meta"))?;
    let k_static_scales_ptr = pool
        .k_static_scales_ptr(layer_idx, stream)
        .ok_or_else(|| anyhow!("quant KV pool missing KIVI k_static_scales (layer {layer_idx})"))?;
    let ws = pool
        .int8_attn_workspace
        .as_ref()
        .ok_or_else(|| anyhow!("quant KV pool missing split-KV attention workspace"))?;
    // The kernel adapts its split count to the workspace it is given
    // (`choose_decode_num_splits` clamps by workspace_bytes, ≥1 split), so the
    // only unfittable case is a single split not fitting. The pool sizes the
    // workspace from an approximate-max-heads heuristic (paged_kv.rs) that can
    // undershoot the full 32-split footprint for q40/q64 dense shapes at small
    // num_slots — gating on 32 splits here would falsely reject those.
    let needed = kv_quant::decode_attention_int8_workspace_bytes(1, num_q_heads, head_dim, 1);
    ensure!(
        needed <= pool.int8_attn_workspace_bytes,
        "quant decode workspace cannot fit a single split: needs {needed} bytes, pool allocated {}",
        pool.int8_attn_workspace_bytes
    );
    let sm_scale = 1.0 / (head_dim as f32).sqrt();
    match pool.format {
        KVFormat::FP8E4M3 => kv_quant::decode_attention_fp8_per_channel_k(
            ctx,
            q_batch,
            pool.k_data_ptr(layer_idx, stream),
            pool.v_data_ptr(layer_idx, stream),
            k_static_scales_ptr,
            pool.v_scales_ptr(layer_idx, stream),
            &meta.kv_indices,
            quant_meta,
            out,
            1,
            num_q_heads,
            num_kv_heads,
            head_dim,
            pool.kv_dim,
            sm_scale,
            ws,
            pool.int8_attn_workspace_bytes,
        ),
        KVFormat::INT8 => kv_quant::decode_attention_int8_per_channel_k(
            ctx,
            q_batch,
            pool.k_data_ptr(layer_idx, stream),
            pool.v_data_ptr(layer_idx, stream),
            k_static_scales_ptr,
            pool.v_scales_ptr(layer_idx, stream),
            &meta.kv_indices,
            quant_meta,
            out,
            1,
            num_q_heads,
            num_kv_heads,
            head_dim,
            pool.kv_dim,
            sm_scale,
            ws,
            pool.int8_attn_workspace_bytes,
        ),
        other => bail!("quant KV decode does not support format {other:?}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn prefill_attention(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &mut HiddenStates,
    k_batch: &mut HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    rms_eps: f32,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    let quant = matches!(pool.format, KVFormat::INT8 | KVFormat::FP8E4M3);
    if !quant && pool.format != KVFormat::BF16 {
        // Defensive: CudaKvCacheDtype::resolve admits only BF16/INT8/FP8.
        bail!(
            "dense-Qwen3 paged prefill supports BF16/INT8/FP8E4M3 KV pools, got {:?}",
            pool.format
        );
    }
    if quant {
        // Prefix rows must be back in the bf16 work buffer before the prep
        // kernel appends this chunk's rows (TileLang reads the whole [0,
        // start_pos + seq_len) span from the work buffer via pool.k_ptr).
        refill_prefix_rows_to_work(ctx, layer_idx, pool, meta, num_kv_heads, head_dim)?;
    }
    {
        let (q_ptr, _gq) = q_batch.data.device_ptr_mut(&ctx.stream);
        let (k_ptr, _gk) = k_batch.data.device_ptr_mut(&ctx.stream);
        let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
        let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
        let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
        let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
        let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
        let (indices_ptr, _gi) = meta.kv_indices.device_ptr(&ctx.stream);
        let (offsets_ptr, _goff) = meta.page_table_offsets.device_ptr(&ctx.stream);
        let (start_ptr, _gstart) = meta.start_positions.device_ptr(&ctx.stream);
        let k_pool_ptr = pool.k_ptr(layer_idx, &ctx.stream);
        let v_pool_ptr = pool.v_ptr(layer_idx, &ctx.stream);

        unsafe {
            ffi::prefill_attention_paged_prep_cuda(
                q_ptr as *mut ffi::Half,
                k_ptr as *mut ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                indices_ptr as *const i32,
                offsets_ptr as *const i32,
                pool.page_size as i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                meta.seq_len as i32,
                start_ptr as *const i32,
                rms_eps,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    run_tilelang_paged(
        ctx,
        false,
        layer_idx,
        pool,
        q_batch,
        meta,
        num_q_heads,
        num_kv_heads,
        head_dim,
        out,
    )?;
    if quant {
        // FINALIZE after TileLang has consumed the bf16 work buffer: calibrate
        // (latch-once) and persist this chunk's new rows into the quant planes.
        calibrate_and_quantize_new_rows(ctx, layer_idx, pool, meta, num_kv_heads, head_dim)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_attention(
    ctx: &DeviceContext,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &mut HiddenStates,
    k_batch: &HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    rms_eps: f32,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    let quant = matches!(pool.format, KVFormat::INT8 | KVFormat::FP8E4M3);
    if !quant && pool.format != KVFormat::BF16 {
        // Defensive: CudaKvCacheDtype::resolve admits only BF16/INT8/FP8.
        bail!(
            "dense-Qwen3 paged decode supports BF16/INT8/FP8E4M3 KV pools, got {:?}",
            pool.format
        );
    }
    {
        let (q_ptr, _gq) = q_batch.data.device_ptr_mut(&ctx.stream);
        let (k_ptr, _gk) = k_batch.data.device_ptr(&ctx.stream);
        let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
        let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
        let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
        let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
        let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
        let (pos_ptr, _gp) = meta.positions.device_ptr(&ctx.stream);
        let (indices_ptr, _gi) = meta.kv_indices.device_ptr(&ctx.stream);
        let (indptr_ptr, _gind) = meta.kv_indptr.device_ptr(&ctx.stream);
        let (last_ptr, _glp) = meta.kv_last_page_len.device_ptr(&ctx.stream);
        let k_pool_ptr = pool.k_ptr(layer_idx, &ctx.stream);
        let v_pool_ptr = pool.v_ptr(layer_idx, &ctx.stream);
        let stride_page = pool.kv_dim * pool.page_size;

        unsafe {
            ffi::decode_prep_paged_cuda(
                q_ptr as *mut ffi::Half,
                k_ptr as *const ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                pos_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                indices_ptr as *const i32,
                indptr_ptr as *const i32,
                last_ptr as *const i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                stride_page as i32,
                1,
                rms_eps,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    if quant {
        // Calibrate-if-unlatched covers the 1-token-prompt edge — a seq_len==1
        // first forward routes here with start_pos==0 and zero-init static
        // scales; quantizing K against a zero table would write garbage for
        // the whole request. Then quantize this step's row and run the fused
        // dequant decode kernel (graph is hard-disabled for quant pools).
        calibrate_and_quantize_new_rows(ctx, layer_idx, pool, meta, num_kv_heads, head_dim)?;
        return run_quant_decode(
            ctx,
            layer_idx,
            pool,
            q_batch,
            meta,
            num_q_heads,
            num_kv_heads,
            head_dim,
            out,
        );
    }
    run_tilelang_paged(
        ctx,
        true,
        layer_idx,
        pool,
        q_batch,
        meta,
        num_q_heads,
        num_kv_heads,
        head_dim,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_tilelang_paged(
    ctx: &DeviceContext,
    decode: bool,
    layer_idx: usize,
    pool: &PagedKVPool,
    q_batch: &HiddenStates,
    meta: &PageMeta,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(head_dim == 128, "only HD128 TileLang kernels are wired");
    ensure!(num_kv_heads == 8, "only kv8 TileLang kernels are wired");

    let (q_ptr, _gq) = q_batch.data.device_ptr(&ctx.stream);
    let (qo_ptr, _gqo) = meta.q_indptr.device_ptr(&ctx.stream);
    let (kv_indptr_ptr, _gki) = meta.kv_indptr.device_ptr(&ctx.stream);
    let (kv_indices_ptr, _gkx) = meta.kv_indices.device_ptr(&ctx.stream);
    let (last_ptr, _glp) = meta.kv_last_page_len.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let k_pool_ptr = pool.k_ptr(layer_idx, &ctx.stream);
    let v_pool_ptr = pool.v_ptr(layer_idx, &ctx.stream);
    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();

    // Set R6_ATTN_DEBUG=1 to dump the scalar args + device arrays fed to the
    // TileLang paged kernel.
    if std::env::var("R6_ATTN_DEBUG").is_ok() {
        eprintln!(
            "[r6-attn] decode={decode} layer={layer_idx} q_heads={num_q_heads} kv_heads={num_kv_heads} head_dim={head_dim} seq_len={} num_pages(meta)={} max_total_pages={} page_size={} kv_dim={} sm_scale={sm_scale}",
            meta.seq_len, meta.num_pages, pool.max_total_pages, pool.page_size, pool.kv_dim
        );
        for (name, slice) in [
            ("q_indptr", &meta.q_indptr),
            ("kv_indptr", &meta.kv_indptr),
            ("kv_indices", &meta.kv_indices),
            ("kv_last_page_len", &meta.kv_last_page_len),
        ] {
            match ctx.stream.clone_dtoh(slice) {
                Ok(v) => eprintln!("[r6-attn]   {name} = {v:?}"),
                Err(e) => eprintln!("[r6-attn]   {name} dtoh err: {e}"),
            }
        }
    }

    // TileLang arg order (load-bearing): `num_pages` (arg 12) = pool capacity
    // (`pool.max_total_pages`, the k_pool/v_pool first-dim extent); `total_pages`
    // (arg 13) = page-table length (`meta.num_pages`). Swapping them gives wrong
    // pool strides + an OOB kv_indices walk that hangs the kernel (Xid 43).
    unsafe {
        match (decode, num_q_heads) {
            (false, 16) => ffi::tilelang_batch_prefill_paged_hd128_q16_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                meta.seq_len as i32,
                meta.seq_len as i32,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (false, 32) => ffi::tilelang_batch_prefill_paged_hd128_q32_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                meta.seq_len as i32,
                meta.seq_len as i32,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (false, 40) => ffi::tilelang_batch_prefill_paged_hd128_q40_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                meta.seq_len as i32,
                meta.seq_len as i32,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (false, 64) => ffi::tilelang_batch_prefill_paged_hd128_q64_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                meta.seq_len as i32,
                meta.seq_len as i32,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (true, 16) => ffi::tilelang_batch_decode_paged_hd128_q16_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                1,
                1,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (true, 32) => ffi::tilelang_batch_decode_paged_hd128_q32_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                1,
                1,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (true, 40) => ffi::tilelang_batch_decode_paged_hd128_q40_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                1,
                1,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            (true, 64) => ffi::tilelang_batch_decode_paged_hd128_q64_kv8_run_cuda(
                q_ptr as *mut ffi::Half,
                qo_ptr as *const i32,
                k_pool_ptr as *mut ffi::Half,
                v_pool_ptr as *mut ffi::Half,
                kv_indptr_ptr as *const i32,
                kv_indices_ptr as *const i32,
                last_ptr as *const i32,
                out_ptr as *mut ffi::Half,
                1,
                1,
                1,
                pool.max_total_pages as i32,
                meta.num_pages as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pool.page_size as i32,
                sm_scale,
                ctx.stream.cu_stream(),
            )
            .result()?,
            _ => bail!("unsupported HD128 q/kv head config q{num_q_heads}_kv{num_kv_heads}"),
        }
    }
    Ok(())
}

// ============================================================================
// DSv4-Flash MLA attention core
// ============================================================================
//
// The MLA attention is a genuinely new subsystem next to the dense-BF16 paged
// path above (it is NOT a GEMM swap): a low-rank Q/KV projection (`wq_a → q_norm
// → wq_b` for Q; `wkv → kv_norm` for the single compressed KV latent), partial
// RoPE on the trailing `rope_dim` columns, a windowed attention with a per-head
// sink logit + (on CSA/HCA layers) a compressed-key stream, and a low-rank O
// projection (`wo_a → wo_b`).
//
// All three modes run through the bf16 correctness core (the perf-optimized
// FlashMLA sparse path stays gated and uses the shared per-layer FP8 KV pool:
//   - SlidingWindow (`compress_ratio == 0`): Q/K prep RoPE + `dsv4_swa_attention`
//     over the bf16 SW ring cache, with the output inverse-RoPE fused.
//   - CompressedSparse (`0 < ratio < 16`): a compressor produces compressed keys,
//     an indexer + `dsv4_csa_select_cuda` picks the top-k blocks, then
//     `dsv4_hybrid_attention_cuda` (mode 1) attends over SW window + selected
//     compressed blocks.
//   - HybridCompressed (`ratio >= 16`): compressor + `dsv4_hybrid_attention_cuda`
//     (mode 2) attending over SW window + ALL compressed blocks (no selector).
//
// Shared kernels: `dsv4_{fp8,fp4}_gemv_batch_cuda` / `gemm_cuda` (LoRA matmuls),
// `dsv4_prepare_qk_cuda`, `dsv4_swa_attention_cuda`, `dsv4_compressor_update_cuda`,
// `dsv4_csa_select_cuda`, `dsv4_hybrid_attention_cuda`.

/// Run one DSv4 FP8/FP4 block-scaled LoRA matmul: `out[N, T] = W[N, K] · x[K, T]`.
///
/// The MLA LoRA weights (`wq_a/wq_b/wkv/wo_a/wo_b`) load as
/// [`WeightFormat::Dsv4Fp8BlockScaled`] / [`WeightFormat::Dsv4Fp4BlockScaled`]
/// (raw quant bytes in `qweight`, E8M0 block scales in `dsv4_scales`), so the
/// dense bf16 [`gemm_batch`] cannot run them — this dispatches the shared
/// `dsv4_*_gemv_batch_cuda` kernels instead. `batch_size` is the token count.
pub(crate) fn mla_linear(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        weight.cols == x.hidden_dim,
        "mla_linear input dim mismatch: weight cols {}, x hidden_dim {}",
        weight.cols,
        x.hidden_dim
    );
    ensure!(
        weight.rows == out.hidden_dim && x.seq_len == out.seq_len,
        "mla_linear output shape mismatch: weight rows {}, out hidden_dim {}, x seq {}, out seq {}",
        weight.rows,
        out.hidden_dim,
        x.seq_len,
        out.seq_len
    );
    let qw = weight
        .qweight
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DSv4 MLA matrix missing raw quant bytes (qweight)"))?;
    let scales = weight
        .dsv4_scales
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DSv4 MLA matrix missing block scales (dsv4_scales)"))?;
    let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
    let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();
    // SAFETY: all buffers are valid on ctx.stream; shapes are checked above and
    // the scale-row/col extents come from the matrix the loader built.
    unsafe {
        let res = match weight.weight_format {
            WeightFormat::Dsv4Fp8BlockScaled => ffi::dsv4_fp8_gemv_batch_cuda(
                qw_ptr as *const u8,
                scales_ptr as *const u8,
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                x.seq_len as i32,
                weight.rows as i32,
                weight.cols as i32,
                weight.dsv4_scale_rows as i32,
                weight.dsv4_scale_cols as i32,
                stream,
            ),
            WeightFormat::Dsv4Fp4BlockScaled => ffi::dsv4_fp4_gemv_batch_cuda(
                qw_ptr as *const u8,
                scales_ptr as *const u8,
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                x.seq_len as i32,
                weight.rows as i32,
                weight.cols as i32,
                weight.dsv4_scale_rows as i32,
                weight.dsv4_scale_cols as i32,
                stream,
            ),
            other => bail!("mla_linear: expected DSv4 FP8/FP4 block-scaled weight, got {other:?}"),
        };
        res.result()?;
    }
    Ok(())
}

/// Decode (M=1) FP8 projection through tensor-core DeepGEMM: quantize `input`
/// (K columns) into the fused-wqkv FP8 scratch, then `dsv4_deepgemm_fp8_gemm_nt`
/// with the pre-repacked weight `cache`. Used for the residual decode projections
/// (wo_a/wo_b; lever #1b) when K ≤ the scratch hidden_dim. The scratch may have
/// been consumed by an earlier projection this step — safe, all on `ctx.stream`.
fn decode_proj_deepgemm(
    ctx: &DeviceContext,
    scratch: &Dsv4FusedWqkvDecodeScratch,
    cache: &cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache,
    input: &HiddenStates,
    out: &mut HiddenStates,
    k: usize,
) -> Result<()> {
    ensure!(
        cache.cols == k
            && cache.rows == out.hidden_dim
            && input.hidden_dim == k
            && input.seq_len == 1
            && out.seq_len == 1,
        "DSv4 decode_proj_deepgemm shape mismatch: cache {}x{} k={k} in {}x{} out {}x{}",
        cache.rows,
        cache.cols,
        input.hidden_dim,
        input.seq_len,
        out.hidden_dim,
        out.seq_len
    );
    let stream = ctx.stream.cu_stream();
    // SAFETY: all buffers live on ctx.stream; K ≤ scratch hidden_dim so the fused
    // FP8 + scale scratch covers the extents.
    unsafe {
        cuda_moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
            cache_ptr(&input.data, ctx),
            cache_ptr(&scratch.input_fp8, ctx),
            cache_ptr(&scratch.input_scales, ctx),
            cache_ptr(&scratch.active_experts, ctx),
            cache_ptr(&scratch.active_offsets, ctx),
            cache_ptr(&scratch.active_counts, ctx),
            1,
            scratch.max_m,
            k,
            scratch.scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 decode proj activation quantize failed: {e}"))?;
        cuda_moe::dsv4_deepgemm_fp8_gemm_nt(
            cache_ptr(&scratch.input_fp8, ctx),
            cache_ptr(&scratch.input_scales, ctx),
            cache_ptr(&cache.weight, ctx),
            cache_ptr(&cache.scales, ctx),
            cache_ptr(&out.data, ctx),
            1,
            cache.rows,
            cache.cols,
            scratch.scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 decode proj DeepGEMM dense failed: {e}"))?;
    }
    Ok(())
}

/// Prefill (M=token_count) residual projection via DeepGEMM: quantize `input`
/// [m, k] into the prefill FP8 scratch, then `dsv4_deepgemm_fp8_gemm_nt` with the
/// pre-repacked weight `cache`. The M>1 analogue of [`decode_proj_deepgemm`] —
/// moves the prefill wq_b / wo / indexer projections off the scalar
/// `dsv4_fp8_gemv_batch` (62% of mla_attn prefill) onto tensor-core DeepGEMM.
/// K ≤ scratch.max_k (the fused-wqkv scratch is sized for the largest K=hidden_dim).
fn prefill_proj_deepgemm(
    ctx: &DeviceContext,
    scratch: &mut Dsv4PrefillDeepGemmLinearScratch,
    cache: &cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache,
    input: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    let m = input.seq_len;
    let k = cache.cols;
    let n = cache.rows;
    ensure!(
        input.hidden_dim == k && out.hidden_dim == n && out.seq_len == m,
        "DSv4 prefill_proj_deepgemm shape mismatch: cache {n}x{k} in {}x{} out {}x{}",
        input.hidden_dim,
        input.seq_len,
        out.hidden_dim,
        out.seq_len
    );
    // M (query/token) dim is chunk-bounded: scratch.max_m = DSV4_PREFILL_QUERY_CHUNK
    // (>= chunked_prefill_size). Chunked prefill guarantees seq_len <=
    // chunked_prefill_size <= DSV4_PREFILL_QUERY_CHUNK, so this assert only trips on
    // a misconfigured chunk size or the one-shot dsv4_parity long-context example —
    // fail loud rather than write past the chunk-sized M×K scratch.
    ensure!(
        m <= scratch.max_m && k <= scratch.max_k,
        "DSv4 prefill_proj_deepgemm M={m} > query chunk {} (or K={k} > {}): chunked \
         prefill must keep seq_len <= chunked_prefill_size <= DSV4_PREFILL_QUERY_CHUNK",
        scratch.max_m,
        scratch.max_k
    );
    let scale_stride_m = m.div_ceil(4) * 4;
    let scale_cols = k.div_ceil(128);
    ensure!(
        scale_stride_m <= scratch.max_scale_stride_m
            && scale_stride_m * scale_cols <= scratch.input_scales.len()
            && m * k <= scratch.input_fp8.len(),
        "DSv4 prefill_proj_deepgemm scratch extent mismatch: M={m} K={k} stride={scale_stride_m}"
    );
    let active_count = i32::try_from(m)
        .map_err(|_| anyhow!("DSv4 prefill_proj_deepgemm token count {m} overflows i32"))?;
    ctx.stream
        .memcpy_htod(&[active_count], &mut scratch.active_counts)
        .map_err(|e| anyhow!("DSv4 prefill_proj_deepgemm active_counts H2D failed: {e}"))?;
    let stream = ctx.stream.cu_stream();
    // SAFETY: all buffers on ctx.stream; M/K within scratch extents (checked above).
    unsafe {
        cuda_moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
            cache_ptr(&input.data, ctx),
            cache_ptr(&scratch.input_fp8, ctx),
            cache_ptr(&scratch.input_scales, ctx),
            cache_ptr(&scratch.active_experts, ctx),
            cache_ptr(&scratch.active_offsets, ctx),
            cache_ptr(&scratch.active_counts, ctx),
            1,
            m,
            k,
            scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 prefill_proj_deepgemm activation quantize failed: {e}"))?;
        cuda_moe::dsv4_deepgemm_fp8_gemm_nt(
            cache_ptr(&scratch.input_fp8, ctx),
            cache_ptr(&scratch.input_scales, ctx),
            cache_ptr(&cache.weight, ctx),
            cache_ptr(&cache.scales, ctx),
            cache_ptr(&out.data, ctx),
            m,
            n,
            k,
            scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 prefill_proj_deepgemm DeepGEMM dense failed: {e}"))?;
    }
    Ok(())
}

fn run_fused_wqkv_prefill(
    ctx: &DeviceContext,
    attention: &Dsv4Attention,
    hidden: &HiddenStates,
    scratch: &mut Dsv4PrefillDeepGemmLinearScratch,
    c_q: &mut HiddenStates,
    kv_raw: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        hidden.seq_len > 1,
        "DSv4 fused wqkv prefill path requires seq_len>1, got {}",
        hidden.seq_len
    );
    ensure!(
        hidden.hidden_dim == scratch.hidden_dim,
        "DSv4 fused wqkv prefill hidden dim mismatch: hidden={} scratch={}",
        hidden.hidden_dim,
        scratch.hidden_dim
    );
    ensure!(
        c_q.hidden_dim == scratch.q_lora_rank
            && kv_raw.hidden_dim == scratch.head_dim
            && c_q.seq_len == hidden.seq_len
            && kv_raw.seq_len == hidden.seq_len,
        "DSv4 fused wqkv prefill output shape mismatch: c_q={}x{} kv={}x{} scratch q={} kv={} tokens={}",
        c_q.hidden_dim,
        c_q.seq_len,
        kv_raw.hidden_dim,
        kv_raw.seq_len,
        scratch.q_lora_rank,
        scratch.head_dim,
        hidden.seq_len
    );
    let cache = attention.wqkv_a_deepgemm.as_ref().ok_or_else(|| {
        anyhow!("ARLE_DSV4_FP8_LINEAR_DEEPGEMM=1 but DSv4 fused wqkv cache was not loaded")
    })?;
    ensure!(
        cache.rows == scratch.q_lora_rank + scratch.head_dim && cache.cols == scratch.hidden_dim,
        "DSv4 fused wqkv prefill cache shape {}x{} != expected {}x{}",
        cache.rows,
        cache.cols,
        scratch.q_lora_rank + scratch.head_dim,
        scratch.hidden_dim
    );
    let m = hidden.seq_len;
    let n = cache.rows;
    let k = cache.cols;
    // M (query/token) dim is chunk-bounded: scratch.max_m = DSV4_PREFILL_QUERY_CHUNK
    // (>= chunked_prefill_size). Chunked prefill guarantees seq_len <=
    // chunked_prefill_size <= DSV4_PREFILL_QUERY_CHUNK, so this assert only trips on
    // a misconfigured chunk size or the one-shot dsv4_parity long-context example —
    // fail loud rather than write past the chunk-sized M×K activation scratch.
    ensure!(
        m <= scratch.max_m && k <= scratch.max_k,
        "DSv4 fused wqkv prefill M={} > query chunk {} (or K={} > {}): chunked prefill \
         must keep seq_len <= chunked_prefill_size <= DSV4_PREFILL_QUERY_CHUNK",
        m,
        scratch.max_m,
        k,
        scratch.max_k
    );
    let scale_stride_m = m.div_ceil(4) * 4;
    let scale_cols = k.div_ceil(128);
    ensure!(
        scale_stride_m <= scratch.max_scale_stride_m
            && scale_stride_m * scale_cols <= scratch.input_scales.len()
            && m * k <= scratch.input_fp8.len(),
        "DSv4 fused wqkv prefill scratch extent mismatch: M={} K={} scale_stride={} scales={} fp8={}",
        m,
        k,
        scale_stride_m,
        scratch.input_scales.len(),
        scratch.input_fp8.len()
    );
    let active_count = i32::try_from(m)
        .map_err(|_| anyhow!("DSv4 fused wqkv prefill token count {m} overflows i32"))?;
    ctx.stream
        .memcpy_htod(&[active_count], &mut scratch.active_counts)
        .map_err(|e| anyhow!("DSv4 fused wqkv prefill active_counts H2D failed: {e}"))?;
    let stream = ctx.stream.cu_stream();
    unsafe {
        cuda_moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
            cache_ptr(&hidden.data, ctx),
            cache_ptr(&scratch.input_fp8, ctx),
            cache_ptr(&scratch.input_scales, ctx),
            cache_ptr(&scratch.active_experts, ctx),
            cache_ptr(&scratch.active_offsets, ctx),
            cache_ptr(&scratch.active_counts, ctx),
            1,
            m,
            k,
            scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 fused wqkv prefill activation quantize failed: {e}"))?;
        cuda_moe::dsv4_deepgemm_fp8_gemm_nt(
            cache_ptr(&scratch.input_fp8, ctx),
            cache_ptr(&scratch.input_scales, ctx),
            cache_ptr(&cache.weight, ctx),
            cache_ptr(&cache.scales, ctx),
            cache_ptr(&scratch.qkv_raw.data, ctx),
            m,
            n,
            k,
            scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 fused wqkv prefill DeepGEMM dense failed: {e}"))?;
        let (qkv_ptr, _qkv_guard) = scratch.qkv_raw.data.device_ptr(&ctx.stream);
        let (cq_ptr, _cq_guard) = c_q.data.device_ptr_mut(&ctx.stream);
        ffi::dsv4_tp_out_slice_cuda(
            qkv_ptr as *const ffi::Half,
            cq_ptr as *mut ffi::Half,
            m as i32,
            n as i32,
            scratch.q_lora_rank as i32,
            0,
            stream,
        )
        .result()
        .map_err(|e| anyhow!("DSv4 fused wqkv prefill c_q slice failed: {e}"))?;
        let (kv_ptr, _kv_guard) = kv_raw.data.device_ptr_mut(&ctx.stream);
        ffi::dsv4_tp_out_slice_cuda(
            qkv_ptr as *const ffi::Half,
            kv_ptr as *mut ffi::Half,
            m as i32,
            n as i32,
            scratch.head_dim as i32,
            scratch.q_lora_rank as i32,
            stream,
        )
        .result()
        .map_err(|e| anyhow!("DSv4 fused wqkv prefill kv slice failed: {e}"))?;
    }
    Ok(())
}

pub(crate) fn mla_linear_vec(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &DeviceVec,
    out: &mut HiddenStates,
) -> Result<()> {
    ensure!(
        weight.cols == x.len,
        "mla_linear_vec input dim mismatch: weight cols {}, x len {}",
        weight.cols,
        x.len
    );
    ensure!(
        weight.rows == out.hidden_dim && out.seq_len == 1,
        "mla_linear_vec output shape mismatch: weight rows {}, out {}x{}",
        weight.rows,
        out.hidden_dim,
        out.seq_len
    );
    let qw = weight
        .qweight
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DSv4 MLA matrix missing raw quant bytes (qweight)"))?;
    let scales = weight
        .dsv4_scales
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("DSv4 MLA matrix missing block scales (dsv4_scales)"))?;
    let (qw_ptr, _gqw) = qw.device_ptr(&ctx.stream);
    let (scales_ptr, _gs) = scales.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let stream = ctx.stream.cu_stream();
    // SAFETY: all buffers are valid on ctx.stream; shapes are checked above.
    unsafe {
        let res = match weight.weight_format {
            WeightFormat::Dsv4Fp8BlockScaled => ffi::dsv4_fp8_gemv_batch_cuda(
                qw_ptr as *const u8,
                scales_ptr as *const u8,
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                1,
                weight.rows as i32,
                weight.cols as i32,
                weight.dsv4_scale_rows as i32,
                weight.dsv4_scale_cols as i32,
                stream,
            ),
            WeightFormat::Dsv4Fp4BlockScaled => ffi::dsv4_fp4_gemv_batch_cuda(
                qw_ptr as *const u8,
                scales_ptr as *const u8,
                x_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                1,
                weight.rows as i32,
                weight.cols as i32,
                weight.dsv4_scale_rows as i32,
                weight.dsv4_scale_cols as i32,
                stream,
            ),
            other => {
                bail!("mla_linear_vec: expected DSv4 FP8/FP4 block-scaled weight, got {other:?}")
            }
        };
        res.result()?;
    }
    Ok(())
}

/// Run one DSv4 linear `out = W · x` dispatching on the weight's on-disk format:
/// bf16 dense → [`crate::ops::gemm_batch`]; FP8/FP4 block-scaled → [`mla_linear`].
/// DSv4 checkpoints ship the compressor / indexer / HC-mix matrices in either
/// precision, so callers route every non-router linear through here.
pub(crate) fn dsv4_linear(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    match weight.weight_format {
        WeightFormat::DenseBf16 => crate::ops::gemm_batch(ctx, weight, x, out),
        WeightFormat::Dsv4Fp8BlockScaled | WeightFormat::Dsv4Fp4BlockScaled => {
            mla_linear(ctx, weight, x, out)
        }
        other => bail!("dsv4_linear: unsupported weight format {other:?}"),
    }
}

pub(crate) fn dsv4_flashmla_decode_enabled() -> Result<bool> {
    match DSV4_FLASHMLA_DECODE_OVERRIDE.load(Ordering::Relaxed) {
        DSV4_FLASHMLA_OVERRIDE_OFF => return Ok(false),
        DSV4_FLASHMLA_OVERRIDE_ON => return Ok(true),
        _ => {}
    }
    // Default ON: FlashMLA SM90 sparse decode is the adopted decode attention — the
    // same vendored kernel SGLang uses. Licensed 2026-06-06 on the TP=8/EP=8 pod:
    // 64-tok resident same-load A/B token-exact vs scalar, 29.47 -> 36.59 tok/s
    // (+24%). `dsv4_flashmla_decode_alloc_enabled` falls through to this, so the
    // arena allocates under the default. Opt out with ARLE_DSV4_FLASHMLA_DECODE=0.
    Ok(!matches!(
        std::env::var("ARLE_DSV4_FLASHMLA_DECODE").as_deref(),
        Ok("0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO")
    ))
}

fn dsv4_flashmla_prefill_enabled() -> Result<bool> {
    // Default ON: vendored FlashMLA sparse prefill replaces the scalar
    // SW/CSA/HCA attention math. Licensed 2026-06-07 on the TP=8/EP=8 H20 pod:
    // 4096-token warm prefill 7189 -> 4299 ms, and the 2048-token edge case is
    // within the legacy same-config floor on both synthetic and real-prose prompts.
    // Opt out with ARLE_DSV4_FLASHMLA_PREFILL=0.
    Ok(!matches!(
        std::env::var("ARLE_DSV4_FLASHMLA_PREFILL").as_deref(),
        Ok("0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO")
    ))
}

fn dsv4_fp8_linear_deepgemm_enabled() -> Result<bool> {
    // Default ON: prefill wq_a|wkv projection fusion routes the shared hidden
    // activation through FP8 DeepGEMM instead of the scalar FP8 GEMV path. Licensed
    // 2026-06-07 by the six-shape within-floor gate; keep the scalar fallback via
    // ARLE_DSV4_FP8_LINEAR_DEEPGEMM=0.
    Ok(!matches!(
        std::env::var("ARLE_DSV4_FP8_LINEAR_DEEPGEMM").as_deref(),
        Ok("0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO")
    ))
}

fn dsv4_decode_proj_deepgemm_enabled() -> bool {
    // Lever #1 (nsys decode breakdown): route the residual decode projection GEMVs
    // (wq_b now; wo_a/wo_b next) through tensor-core DeepGEMM instead of the scalar
    // `dsv4_fp8_gemv_batch` (3.62ms, #1 decode GPU kernel). Default ON: licensed
    // 2026-06-07 on the TP=8/EP=8 pod, same-load A/B 38.2 -> 39.2 tok/s (+2.5%,
    // reproduced ×2) with the 37-tok needle retrieved bit-identically (divergence
    // only in the free-continuation tail = legitimate FP8 numerics). Opt out with
    // ARLE_DSV4_DECODE_PROJ_DEEPGEMM=0.
    !matches!(
        std::env::var("ARLE_DSV4_DECODE_PROJ_DEEPGEMM").as_deref(),
        Ok("0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO")
    )
}

/// Prefill residual projections (wq_b now; wo/indexer next) → tensor-core DeepGEMM
/// instead of the scalar `dsv4_fp8_gemv_batch` (62% of mla_attn prefill per the P/D
/// nsys breakdown). Default ON: licensed 2026-06-08 on the TP=8 pod — at M=1024 the
/// prefill wq_b A/B cut total prefill_ms 14382 → 7628 (−47%) with the needle answer
/// retrieved byte-identically (scalar fp8_gemv scales O(M); it's a decode GEMV).
/// Opt out with ARLE_DSV4_PREFILL_PROJ_DEEPGEMM=0.
fn dsv4_prefill_proj_deepgemm_enabled() -> bool {
    !matches!(
        std::env::var("ARLE_DSV4_PREFILL_PROJ_DEEPGEMM").as_deref(),
        Ok("0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO")
    )
}

/// Prefill DSA indexer query projection → DeepGEMM (134.9 → 6.05ms, −95.5% at M=1024).
/// **Default ON (licensed 2026-06-09).** This was the #1 prefill GPU kernel — the
/// nsys 64K breakdown pinned `dsv4_fp8_gemv_batch_tiled` (this indexer query proj)
/// at **38.4% of all GPU time** (25ms/call, scalar token-looped GEMV). It feeds the
/// top-k block SELECTOR, so it was gated OFF pending a planted-answer long-context
/// needle (an FP8 flip could shift selection). That gate is now MET: with it ON, the
/// planted needle (738291) **retrieves** — 64K hit `738291` exact, 128K hit `738291`
/// exact, and every run finds the needle region (selection intact). Same-binary A/B:
/// 64K prefill 17.6s → 11.0s (−37%), 128K 42.7s → 23.0s (−46%). The exact-digit
/// borderline at ≥2K is the pre-existing compression-fidelity residual (tracked
/// separately), NOT a selection break. Opt out with ARLE_DSV4_PREFILL_INDEXER_DEEPGEMM=0.
fn dsv4_prefill_indexer_deepgemm_enabled() -> bool {
    !matches!(
        std::env::var("ARLE_DSV4_PREFILL_INDEXER_DEEPGEMM").as_deref(),
        Ok("0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO")
    )
}

pub(crate) fn dsv4_dsa_official_enabled() -> Result<bool> {
    // Default ON: official/vendored DSA indexer replaces the legacy scalar
    // csa_select selector. Licensed by the variable-shape legacy-floor gate:
    // official diverges no earlier than legacy diverges from itself across
    // 64/256/512/1024/2048/4096 prompts, with the legacy selector retained as
    // an explicit fallback via ARLE_DSV4_DSA_INDEXER=0.
    Ok(!matches!(
        std::env::var("ARLE_DSV4_DSA_INDEXER").as_deref(),
        Ok("0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO")
    ))
}

/// Per-layer attention-output localizer (Track A FlashMLA-prefill diagnosis).
///
/// When `ARLE_DSV4_ATTN_DUMP=1`, every layer prints a stable FNV-1a hash of its
/// full bf16 `local_attn` output plus the first 8 values of row 0, on rank 0
/// only. Run the same prompt twice — once scalar (default), once with
/// `ARLE_DSV4_FLASHMLA_PREFILL=1` — and diff the two logs: the *first* CSA/HCA
/// layer whose hash differs is exactly where FlashMLA-prefill diverges from the
/// scalar reference. SW layers run scalar in both passes and match by
/// construction, so a mismatch localizes the bug to one layer in one build —
/// replacing the end-to-end-token guess loop. Adds one `ctx.sync()` per layer,
/// so it is strictly opt-in.
fn dsv4_attn_dump_enabled() -> bool {
    matches!(
        std::env::var("ARLE_DSV4_ATTN_DUMP").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
    ) && std::env::var("INFER_TP_RANK").as_deref() == Ok("0")
}

/// Debug (ARLE_DSV4_KNEW_DUMP, rank 0, layer 0): dump each row's L2 + first4 of the
/// prepared key. Compares token_a's key in a batched [token_a,wrong_b] forward (sp=5
/// seq=2, row 0) vs the per-token [token_a]@5 forward (sp=5 seq=1, row 0) to test
/// whether the multi-token prepare mis-computes token_a (the col1-bug hypothesis).
fn dsv4_dump_kprep(
    ctx: &DeviceContext,
    layer_idx: usize,
    label: &str,
    h: &HiddenStates,
    start_pos: usize,
) {
    if layer_idx != 0
        || std::env::var_os("ARLE_DSV4_KNEW_DUMP").is_none()
        || std::env::var("INFER_TP_RANK").as_deref() != Ok("0")
    {
        return;
    }
    if ctx.sync().is_err() {
        return;
    }
    let host: Vec<half::bf16> = match ctx.stream.clone_dtoh(&h.data) {
        Ok(v) => v,
        Err(_) => return,
    };
    let n = h.hidden_dim;
    for row in 0..h.seq_len {
        let base = row * n;
        let mut l2 = 0.0f32;
        for i in 0..n {
            let x = host[base + i].to_f32();
            l2 += x * x;
        }
        let first4: Vec<f32> = (0..4.min(n)).map(|i| host[base + i].to_f32()).collect();
        eprintln!(
            "[knew-dump] {label} sp={start_pos} seq={} row={row} dim={n} l2={:.5} first4={first4:?}",
            h.seq_len,
            l2.sqrt()
        );
    }
}

fn dsv4_dump_attn_output(
    ctx: &DeviceContext,
    layer_idx: usize,
    mode: DeepSeekV4AttentionMode,
    out: &HiddenStates,
) -> Result<()> {
    ctx.sync()?;
    let host: Vec<half::bf16> = ctx
        .stream
        .clone_dtoh(&out.data)
        .map_err(|e| anyhow!("DSv4 attn-dump D2H failed: {e}"))?;
    let mut hash: u64 = 0xcbf29ce484222325;
    for v in &host {
        hash ^= u64::from(v.to_bits());
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let row0: Vec<f32> = host.iter().take(8).map(|v| v.to_f32()).collect();
    eprintln!(
        "[dsv4-attn-dump] layer={layer_idx} mode={mode:?} seq_len={} hidden={} hash={hash:016x} row0={row0:?}",
        out.seq_len, out.hidden_dim
    );
    Ok(())
}

fn dsv4_flashmla_decode_alloc_enabled() -> Result<bool> {
    if env_flag("ARLE_DSV4_FLASHMLA_DECODE_ALLOC")? {
        return Ok(true);
    }
    dsv4_flashmla_decode_enabled()
}

pub(crate) fn dsv4_fused_wqkv_decode_alloc_enabled() -> Result<bool> {
    if env_flag("ARLE_DSV4_FUSED_WQKV_DECODE_ALLOC")? {
        return Ok(true);
    }
    dsv4_fused_wqkv_decode_enabled()
}

fn dsv4_fused_wqkv_decode_enabled() -> Result<bool> {
    match DSV4_FUSED_WQKV_DECODE_OVERRIDE.load(Ordering::Relaxed) {
        DSV4_FLASHMLA_OVERRIDE_OFF => return Ok(false),
        DSV4_FLASHMLA_OVERRIDE_ON => return Ok(true),
        _ => {}
    }
    // Default ON: fuse wq_a|wkv_a into one FP8 DeepGEMM instead of the scalar
    // `dsv4_fp8_gemv_batch_kernel` (which the clean decode profile pinned at 16.9% of
    // decode GPU — the #1 real decode kernel). Licensed 2026-06-06 on the TP=8/EP=8
    // pod, 64-tok same-binary env A/B: 31.774 -> 37.633 tok/s (+18.4%), token-exact.
    // `dsv4_fused_wqkv_decode_alloc_enabled` falls through to this, so the fused
    // scratch allocates under the default. Opt out with ARLE_DSV4_FUSED_WQKV_DECODE=0.
    Ok(!matches!(
        std::env::var("ARLE_DSV4_FUSED_WQKV_DECODE").as_deref(),
        Ok("0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO")
    ))
}

fn env_flag(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) => match value.as_str() {
            "1" | "true" | "TRUE" | "yes" | "on" | "ON" => Ok(true),
            "0" | "false" | "FALSE" | "no" | "off" | "OFF" | "" => Ok(false),
            other => bail!("unsupported {name} `{other}` (expected 0/1, true/false, on/off)"),
        },
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(e) => bail!("{name} invalid env: {e}"),
    }
}

fn flashmla_mode_int(mode: DeepSeekV4AttentionMode) -> i32 {
    match mode {
        DeepSeekV4AttentionMode::CompressedSparse => 1,
        DeepSeekV4AttentionMode::SlidingWindow | DeepSeekV4AttentionMode::HybridCompressed => 2,
    }
}

fn flashmla_pack_sw_ring(
    ctx: &DeviceContext,
    flash: &mut Dsv4FlashMlaDecodeState,
    pool: &mut Dsv4LayerKvLayout,
    window_cache: &CudaSlice<half::bf16>,
    config: &DeepSeekV4Config,
) -> Result<()> {
    if flash.fp8_kv_sw_bootstrapped {
        return Ok(());
    }
    let sliding_window = config.sliding_window;
    let page_block_size = 64;
    let mut block_ids = Vec::with_capacity(sliding_window);
    let mut rows = Vec::with_capacity(sliding_window);
    {
        // Table-routed (#85 P2): this host-built bulk pack hands the kernel
        // PHYSICAL pool pages (page-table lookup) and the pool BASE pointer,
        // so it stays valid when Stage B fragments the table. Token/block
        // counts come from the slot's table, never re-derived.
        let table = pool.flashmla_page_table(flash.slot_idx)?;
        for slot in 0..sliding_window {
            let page = physical_page(table, slot / page_block_size)?;
            block_ids.push(
                i32::try_from(page)
                    .map_err(|_| anyhow!("DSv4 FlashMLA SW page {page} overflows i32"))?,
            );
            rows.push((slot % page_block_size) as i32);
        }
    }
    ctx.stream
        .memcpy_htod(&block_ids, &mut flash.sw_bulk_block_ids)
        .map_err(|e| anyhow!("DSv4 FlashMLA SW block_ids H2D failed: {e}"))?;
    ctx.stream
        .memcpy_htod(&rows, &mut flash.sw_bulk_rows)
        .map_err(|e| anyhow!("DSv4 FlashMLA SW rows H2D failed: {e}"))?;
    let (window_ptr, _wg) = window_cache.device_ptr(&ctx.stream);
    let pool_buf = pool.flashmla_pool_data_mut()?;
    let (pool_ptr, _pg) = pool_buf.device_ptr_mut(&ctx.stream);
    let nope_ptr = window_ptr;
    let rope_ptr = nope_ptr + (config.head_dim - config.qk_rope_head_dim) as u64 * 2;
    flash_kv::dsv4_fp8_kv_pack_strided_raw(
        ctx,
        nope_ptr,
        rope_ptr,
        pool_ptr,
        &flash.sw_bulk_block_ids,
        &flash.sw_bulk_rows,
        sliding_window,
        page_block_size,
        config.head_dim,
        config.head_dim,
    )?;
    flash.fp8_kv_sw_bootstrapped = true;
    Ok(())
}

fn flashmla_pack_one_sw_token(
    ctx: &DeviceContext,
    flash: &mut Dsv4FlashMlaDecodeState,
    pool: &mut Dsv4LayerKvLayout,
    k_prepared: &HiddenStates,
    start_pos_device: &CudaSlice<i32>,
    config: &DeepSeekV4Config,
) -> Result<()> {
    let (bid_ptr, bid_guard) = flash.one_block_id.device_ptr_mut(&ctx.stream);
    let (row_ptr, row_guard) = flash.one_row.device_ptr_mut(&ctx.stream);
    let (start_ptr, _sg) = start_pos_device.device_ptr(&ctx.stream);
    flash_kv::dsv4_fp8_kv_fill_one_sw_slot_from_start_pos_raw(
        ctx,
        bid_ptr,
        row_ptr,
        start_ptr,
        config.sliding_window,
        64,
    )?;
    drop(bid_guard);
    drop(row_guard);

    let (k_ptr, _kg) = k_prepared.data.device_ptr(&ctx.stream);
    let range = pool.flashmla_pages_byte_range(flash.slot_idx)?;
    let pool_buf = pool.flashmla_pool_data_mut()?;
    ensure!(
        range.end <= pool_buf.len() && range.len() == flash.fp8_kv_pool_len,
        "DSv4 FlashMLA shared one-token table range {:?} invalid pool_len={} slot_len={}",
        range,
        pool_buf.len(),
        flash.fp8_kv_pool_len
    );
    let mut pool_view = pool_buf.slice_mut(range);
    let (pool_ptr, _pg) = pool_view.device_ptr_mut(&ctx.stream);
    let nope_ptr = k_ptr;
    let rope_ptr = nope_ptr + (config.head_dim - config.qk_rope_head_dim) as u64 * 2;
    flash_kv::dsv4_fp8_kv_pack_strided_raw(
        ctx,
        nope_ptr,
        rope_ptr,
        pool_ptr,
        &flash.one_block_id,
        &flash.one_row,
        1,
        64,
        config.head_dim,
        config.head_dim,
    )
}

fn flashmla_pack_compressed_delta(
    ctx: &DeviceContext,
    flash: &mut Dsv4FlashMlaDecodeState,
    pool: &mut Dsv4LayerKvLayout,
    compressed: Option<&HiddenStates>,
    start_pos_device: &CudaSlice<i32>,
    compress_ratio: usize,
    config: &DeepSeekV4Config,
) -> Result<()> {
    let Some(compressed) = compressed else {
        return Ok(());
    };
    // Steady-state decode adds AT MOST one compressed row per step
    // ((pos+1) % ratio == 0). That row is packed by the DEVICE kernel below —
    // fully derived from `start_pos_device`, so it records into CUDA-graph
    // captures and stays correct on replay (the old host Vec + H2D path is
    // skipped on replay entirely, which stalled the pool → garbage/IMA).
    // The host bulk path remains ONLY for multi-row gaps (first decode after
    // prefill / request boundaries), which always execute eagerly (the graph
    // warm pass — see `CudaGraphState::rearm_warm`).
    {
        let range = pool.flashmla_pages_byte_range(flash.slot_idx)?;
        let pool_buf = pool.flashmla_pool_data_mut()?;
        ensure!(
            range.end <= pool_buf.len() && range.len() == flash.fp8_kv_pool_len,
            "DSv4 FlashMLA shared compressed-delta table range {:?} invalid pool_len={} slot_len={}",
            range,
            pool_buf.len(),
            flash.fp8_kv_pool_len
        );
        let mut pool_view = pool_buf.slice_mut(range);
        let (pool_ptr, _pg) = pool_view.device_ptr_mut(&ctx.stream);
        let (compressed_ptr, _cg) = compressed.data.device_ptr(&ctx.stream);
        let (start_ptr, _sg) = start_pos_device.device_ptr(&ctx.stream);
        flash_kv::dsv4_fp8_kv_pack_completed_compressor_row_start_pos_raw(
            ctx,
            compressed_ptr,
            pool_ptr,
            start_ptr,
            compress_ratio,
            flash.sw_blocks,
            64,
            config.head_dim,
        )?;
    }
    let start_row = flash.fp8_kv_comp_packed_rows;
    let end_row = compressed.seq_len;
    if end_row <= start_row {
        return Ok(());
    }
    // Host bookkeeping below runs in eager contexts only (warm pass / no
    // graph); the device kernel above already covered the single-row case, so
    // bulk-pack only multi-row gaps. The boundary row may be packed by both
    // paths — idempotent overwrite of identical data.
    flash.fp8_kv_comp_packed_rows = end_row;
    if end_row == start_row + 1 {
        return Ok(());
    }
    let n = end_row - start_row;
    let mut block_ids = Vec::with_capacity(n);
    let mut rows = Vec::with_capacity(n);
    for row in start_row..end_row {
        block_ids.push((flash.sw_blocks + row / 64) as i32);
        rows.push((row % 64) as i32);
    }
    ctx.stream
        .memcpy_htod(&block_ids, &mut flash.comp_block_ids)
        .map_err(|e| anyhow!("DSv4 FlashMLA compressed block_ids H2D failed: {e}"))?;
    ctx.stream
        .memcpy_htod(&rows, &mut flash.comp_rows)
        .map_err(|e| anyhow!("DSv4 FlashMLA compressed rows H2D failed: {e}"))?;

    let (compressed_ptr, _cg) = compressed.data.device_ptr(&ctx.stream);
    let range = pool.flashmla_pages_byte_range(flash.slot_idx)?;
    let pool_buf = pool.flashmla_pool_data_mut()?;
    ensure!(
        range.end <= pool_buf.len() && range.len() == flash.fp8_kv_pool_len,
        "DSv4 FlashMLA shared compressed table range {:?} invalid pool_len={} slot_len={}",
        range,
        pool_buf.len(),
        flash.fp8_kv_pool_len
    );
    let mut pool_view = pool_buf.slice_mut(range);
    let (pool_ptr, _pg) = pool_view.device_ptr_mut(&ctx.stream);
    let row_offset_bytes = start_row as u64 * config.head_dim as u64 * 2;
    let nope_ptr = compressed_ptr + row_offset_bytes;
    let rope_ptr = nope_ptr + (config.head_dim - config.qk_rope_head_dim) as u64 * 2;
    flash_kv::dsv4_fp8_kv_pack_strided_raw(
        ctx,
        nope_ptr,
        rope_ptr,
        pool_ptr,
        &flash.comp_block_ids,
        &flash.comp_rows,
        n,
        64,
        config.head_dim,
        config.head_dim,
    )?;
    flash.fp8_kv_comp_packed_rows = end_row;
    Ok(())
}

fn update_bf16_sw_window(
    ctx: &DeviceContext,
    sw_window_cache: &mut CudaSlice<half::bf16>,
    k_prepared: &HiddenStates,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    config: &DeepSeekV4Config,
) -> Result<()> {
    let (k_ptr, _kg) = k_prepared.data.device_ptr(&ctx.stream);
    let (window_ptr, _wg) = sw_window_cache.device_ptr_mut(&ctx.stream);
    unsafe {
        if let Some(start_pos_device) = start_pos_device {
            let (start_ptr, _sg) = start_pos_device.device_ptr(&ctx.stream);
            ffi::dsv4_update_window_cache_start_pos_ptr_cuda(
                k_ptr as *const ffi::Half,
                window_ptr as *mut ffi::Half,
                k_prepared.seq_len as i32,
                start_ptr as *const i32,
                config.sliding_window as i32,
                config.head_dim as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        } else {
            ffi::dsv4_update_window_cache_cuda(
                k_ptr as *const ffi::Half,
                window_ptr as *mut ffi::Half,
                k_prepared.seq_len as i32,
                start_pos as i32,
                config.sliding_window as i32,
                config.head_dim as i32,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn try_flashmla_prefill_attention(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    q_prepared: &HiddenStates,
    k_prepared: &HiddenStates,
    selected: Option<&CudaSlice<i32>>,
    compressed: Option<&HiddenStates>,
    sw_window_cache: &mut CudaSlice<half::bf16>,
    start_pos: usize,
    tree: Option<&Dsv4TreeAttnMeta>,
    tp: &TpRuntime,
    local_heads: usize,
    local_attn: &mut HiddenStates,
    sm_scale: f32,
    rope_base: f32,
    original_seq_len: i32,
    rope_factor: f32,
    rope_beta_fast: f32,
    rope_beta_slow: f32,
) -> Result<bool> {
    if !dsv4_flashmla_prefill_enabled()? {
        return Ok(false);
    }
    if q_prepared.seq_len <= 1 {
        return Ok(false);
    }
    // Pure-SW layers go through this path ONLY for tree-verify chunks (the
    // unified pool degenerates to [SW cache | chunk K] and the tree indices
    // carry the branch mask); plain prefill keeps the cheaper contiguous swa.
    if mode == DeepSeekV4AttentionMode::SlidingWindow && tree.is_none() {
        return Ok(false);
    }
    ensure!(
        config.head_dim == 512 && config.qk_rope_head_dim == 64,
        "DSv4 FlashMLA prefill only supports MODEL1 head_dim=512 rope_dim=64"
    );
    ensure!(
        q_prepared.seq_len == k_prepared.seq_len && local_attn.seq_len == q_prepared.seq_len,
        "DSv4 FlashMLA prefill shape mismatch: q={} k={} out={}",
        q_prepared.seq_len,
        k_prepared.seq_len,
        local_attn.seq_len
    );

    let token_count = q_prepared.seq_len;
    if let Some(meta) = tree {
        ensure!(
            meta.n_rows == token_count,
            "DSv4 tree meta rows {} != verify rows {token_count}",
            meta.n_rows
        );
    }
    let tp_world = tp.config().world_size;
    let tp_rank = tp.config().rank;
    let global_heads = local_heads
        .checked_mul(tp_world)
        .ok_or_else(|| anyhow!("DSv4 FlashMLA prefill global head overflow"))?;
    ensure!(
        matches!(global_heads, 64 | 128),
        "DSv4 FlashMLA prefill requires global heads 64/128, got {global_heads}"
    );

    let compressed_count = compressed.map_or(0, |c| c.seq_len);
    let max_compressed_keys = match mode {
        DeepSeekV4AttentionMode::CompressedSparse => config.index_topk,
        DeepSeekV4AttentionMode::HybridCompressed => compressed_count.div_ceil(128) * 128,
        // Tree-only: no compressed region in the unified pool.
        DeepSeekV4AttentionMode::SlidingWindow => 0,
    };
    // Tree rows attend committed-window + branch + compressed: the branch part
    // (<= max_anc + 1 <= 128) needs its own padded slot block on top of the
    // chain layout's sw_window + compressed budget.
    let tree_pad = if tree.is_some() { 128 } else { 0 };
    let topk_unified = config
        .sliding_window
        .checked_add(tree_pad)
        .and_then(|v| v.checked_add(max_compressed_keys))
        .ok_or_else(|| anyhow!("DSv4 FlashMLA prefill topk overflow"))?;
    ensure!(
        topk_unified.is_multiple_of(128),
        "DSv4 FlashMLA prefill topk {topk_unified} must be multiple of 128"
    );
    let kv_rows = config
        .sliding_window
        .checked_add(token_count)
        .and_then(|v| v.checked_add(compressed_count))
        .ok_or_else(|| anyhow!("DSv4 FlashMLA prefill unified KV rows overflow"))?;
    ensure!(kv_rows > 0, "DSv4 FlashMLA prefill needs non-empty KV pool");

    // FlashMLA prefill consumes one unified bf16 pool:
    // [rolling SW cache rebased | current chunk K | compressed pool].
    let mut kv_unified = unsafe { HiddenStates::uninit(ctx, config.head_dim, kv_rows)? };
    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_prefill_pack_kv");
        let (kv_ptr, _kvg) = kv_unified.data.device_ptr_mut(&ctx.stream);
        let (window_ptr, _wg) = sw_window_cache.device_ptr(&ctx.stream);
        let (k_ptr, _kg) = k_prepared.data.device_ptr(&ctx.stream);
        let (comp_ptr, _cg) = match compressed.filter(|_| compressed_count > 0) {
            Some(c) => {
                let (p, g) = c.data.device_ptr(&ctx.stream);
                (p as *const ffi::Half, Some(g))
            }
            None => (std::ptr::null(), None),
        };
        unsafe {
            ffi::arle_flashmla_csa_pack_kv(
                kv_ptr as *mut ffi::Half,
                window_ptr as *const ffi::Half,
                k_ptr as *const ffi::Half,
                comp_ptr,
                start_pos as i32,
                config.sliding_window as i32,
                token_count as i32,
                compressed_count as i32,
                config.head_dim as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| anyhow!("DSv4 FlashMLA prefill KV pack failed: {e}"))?;
        }
    }

    let mut indices = ctx
        .stream
        .alloc_zeros::<i32>(token_count * topk_unified)
        .map_err(|e| anyhow!("DSv4 FlashMLA prefill indices alloc failed: {e}"))?;
    let mut topk_length = ctx
        .stream
        .alloc_zeros::<i32>(token_count)
        .map_err(|e| anyhow!("DSv4 FlashMLA prefill topk_length alloc failed: {e}"))?;
    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_prefill_build_indices");
        let (indices_ptr, _ig) = indices.device_ptr_mut(&ctx.stream);
        let (topk_ptr, _tg) = topk_length.device_ptr_mut(&ctx.stream);
        if let Some(meta) = tree {
            // Tree-verify chunk: per-row positions + branch ancestors replace
            // the causal-contiguous window arithmetic. CSA passes the selector
            // output; HCA passes its identity cap; pure-SW passes neither.
            let (pos_ptr, _pg) = meta.positions.device_ptr(&ctx.stream);
            let (anc_ptr, _ag) = meta.ancestors.device_ptr(&ctx.stream);
            let (sel_ptr, _sg) = match (mode, selected) {
                (DeepSeekV4AttentionMode::CompressedSparse, Some(sel)) => {
                    let (p, g) = sel.device_ptr(&ctx.stream);
                    (p as *const i32, Some(g))
                }
                (DeepSeekV4AttentionMode::CompressedSparse, None) => {
                    bail!("DSv4 FlashMLA CSA tree verify missing selected topk")
                }
                _ => (std::ptr::null(), None),
            };
            let max_compressed_arg = if mode == DeepSeekV4AttentionMode::HybridCompressed {
                max_compressed_keys
            } else {
                0
            };
            unsafe {
                ffi::arle_flashmla_tree_build_indices(
                    indices_ptr as *mut i32,
                    topk_ptr as *mut i32,
                    pos_ptr as *const i32,
                    anc_ptr as *const i32,
                    meta.max_anc as i32,
                    sel_ptr,
                    token_count as i32,
                    start_pos as i32,
                    config.sliding_window as i32,
                    if sel_ptr.is_null() {
                        0
                    } else {
                        config.index_topk as i32
                    },
                    max_compressed_arg as i32,
                    topk_unified as i32,
                    compressed_count as i32,
                    compress_ratio as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("DSv4 FlashMLA tree verify indices failed: {e}"))?;
            }
        } else {
            unsafe {
                match mode {
                    DeepSeekV4AttentionMode::CompressedSparse => {
                        let selected = selected.ok_or_else(|| {
                            anyhow!("DSv4 FlashMLA CSA prefill missing selected topk")
                        })?;
                        let (selected_ptr, _sg) = selected.device_ptr(&ctx.stream);
                        ffi::arle_flashmla_csa_build_indices(
                            indices_ptr as *mut i32,
                            topk_ptr as *mut i32,
                            selected_ptr as *const i32,
                            token_count as i32,
                            start_pos as i32,
                            config.sliding_window as i32,
                            config.index_topk as i32,
                            compressed_count as i32,
                            compress_ratio as i32,
                            ctx.stream.cu_stream(),
                        )
                        .result()
                        .map_err(|e| anyhow!("DSv4 FlashMLA CSA prefill indices failed: {e}"))?;
                    }
                    DeepSeekV4AttentionMode::HybridCompressed => {
                        ffi::arle_flashmla_hca_build_indices(
                            indices_ptr as *mut i32,
                            topk_ptr as *mut i32,
                            token_count as i32,
                            start_pos as i32,
                            config.sliding_window as i32,
                            max_compressed_keys as i32,
                            compressed_count as i32,
                            compress_ratio as i32,
                            ctx.stream.cu_stream(),
                        )
                        .result()
                        .map_err(|e| anyhow!("DSv4 FlashMLA HCA prefill indices failed: {e}"))?;
                    }
                    DeepSeekV4AttentionMode::SlidingWindow => unreachable!(),
                }
            }
        }
    }

    let mut max_logits = ctx
        .stream
        .alloc_zeros::<f32>(token_count * global_heads)
        .map_err(|e| anyhow!("DSv4 FlashMLA prefill max_logits alloc failed: {e}"))?;
    let mut lse = ctx
        .stream
        .alloc_zeros::<f32>(token_count * global_heads)
        .map_err(|e| anyhow!("DSv4 FlashMLA prefill lse alloc failed: {e}"))?;

    let (q_ptr, q_guard) = q_prepared.data.device_ptr(&ctx.stream);
    let (kv_ptr, kv_guard) = kv_unified.data.device_ptr(&ctx.stream);
    let (indices_ptr, indices_guard) = indices.device_ptr(&ctx.stream);
    let (topk_ptr, topk_guard) = topk_length.device_ptr(&ctx.stream);
    let (out_ptr, out_guard) = local_attn.data.device_ptr_mut(&ctx.stream);
    let (max_ptr, max_guard) = max_logits.device_ptr_mut(&ctx.stream);
    let (lse_ptr, lse_guard) = lse.device_ptr_mut(&ctx.stream);

    let local_width = local_heads * config.head_dim;
    let global_width = global_heads * config.head_dim;
    let (q_for_flashmla, flash_out_ptr, mut tp_gathered_q, mut tp_packed_q, mut tp_full_out) =
        if tp_world > 1 {
            let mut gathered = ctx
                .stream
                .alloc_zeros::<half::bf16>(tp_world * token_count * local_width)
                .map_err(|e| anyhow!("DSv4 FlashMLA prefill TP Q gather alloc failed: {e}"))?;
            let mut packed = ctx
                .stream
                .alloc_zeros::<half::bf16>(token_count * global_width)
                .map_err(|e| anyhow!("DSv4 FlashMLA prefill TP Q pack alloc failed: {e}"))?;
            let full_out = ctx
                .stream
                .alloc_zeros::<half::bf16>(token_count * global_width)
                .map_err(|e| anyhow!("DSv4 FlashMLA prefill TP output alloc failed: {e}"))?;
            let (gather_ptr, gather_guard) = gathered.device_ptr_mut(&ctx.stream);
            {
                let _nvtx = crate::nvtx::range("dsv4/flashmla_prefill_q_allgather");
                unsafe {
                    tp.all_gather_bf16_raw(
                        ctx,
                        q_ptr as *const std::ffi::c_void,
                        token_count * local_width,
                        gather_ptr as *mut std::ffi::c_void,
                    )?;
                }
            }
            drop(gather_guard);
            let (packed_ptr, packed_guard) = packed.device_ptr_mut(&ctx.stream);
            {
                let _nvtx = crate::nvtx::range("dsv4/flashmla_prefill_q_repack");
                unsafe {
                    ffi::dsv4_tp_q_repack_cuda(
                        gather_ptr as *const ffi::Half,
                        packed_ptr as *mut ffi::Half,
                        tp_world as i32,
                        token_count as i32,
                        local_heads as i32,
                        config.head_dim as i32,
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .map_err(|e| anyhow!("DSv4 FlashMLA prefill TP Q repack failed: {e}"))?;
                }
            }
            drop(packed_guard);
            let (full_out_ptr, full_out_guard) = full_out.device_ptr(&ctx.stream);
            drop(full_out_guard);
            (
                packed_ptr as *const ffi::Half,
                full_out_ptr as *mut ffi::Half,
                Some(gathered),
                Some(packed),
                Some(full_out),
            )
        } else {
            (
                q_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                None,
                None,
                None,
            )
        };

    let (sink_base, sink_guard) = attention.attn_sink_f32.device_ptr(&ctx.stream);
    ensure!(
        if tp_world > 1 {
            attention.attn_sink_f32.len() >= global_heads
        } else {
            attention.attn_sink_f32.len() >= tp_rank * local_heads + local_heads
        },
        "DSv4 FlashMLA prefill attn_sink_f32 len {} cannot cover heads",
        attention.attn_sink_f32.len()
    );
    let sink_ptr = if tp_world > 1 {
        sink_base as *const f32
    } else {
        unsafe { (sink_base as *const f32).add(tp_rank * local_heads) }
    };

    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_prefill_fwd");
        unsafe {
            ffi::arle_flashmla_sm90_sparse_prefill_fwd(
                q_for_flashmla,
                kv_ptr as *const ffi::Half,
                indices_ptr as *const i32,
                sink_ptr,
                topk_ptr as *const i32,
                flash_out_ptr,
                max_ptr as *mut f32,
                lse_ptr as *mut f32,
                token_count as i32,
                kv_rows as i32,
                global_heads as i32,
                1,
                config.head_dim as i32,
                config.head_dim as i32,
                topk_unified as i32,
                sm_scale,
                global_width as i32,
                config.head_dim as i32,
                config.head_dim as i32,
                0,
                topk_unified as i32,
                0,
                0,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| anyhow!("DSv4 FlashMLA sparse prefill failed: {e}"))?;
        }
    }

    if tp_world > 1 {
        let full_out = tp_full_out
            .as_ref()
            .ok_or_else(|| anyhow!("DSv4 FlashMLA prefill missing TP full output"))?;
        let (full_out_ptr, full_out_guard) = full_out.device_ptr(&ctx.stream);
        {
            let _nvtx = crate::nvtx::range("dsv4/flashmla_prefill_out_slice");
            unsafe {
                ffi::dsv4_tp_out_slice_cuda(
                    full_out_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    token_count as i32,
                    global_width as i32,
                    local_width as i32,
                    (tp_rank * local_width) as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("DSv4 FlashMLA prefill TP out slice failed: {e}"))?;
            }
        }
        drop(full_out_guard);
    }

    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_prefill_inverse_rope");
        unsafe {
            if let Some(meta) = tree {
                let (pos_ptr, _pg) = meta.positions.device_ptr(&ctx.stream);
                ffi::arle_dsv4_output_inverse_rope_batch_start_pos_cuda(
                    out_ptr as *mut ffi::Half,
                    token_count as i32,
                    local_heads as i32,
                    config.head_dim as i32,
                    config.qk_rope_head_dim as i32,
                    pos_ptr as *const i32,
                    rope_base,
                    original_seq_len,
                    rope_factor,
                    rope_beta_fast,
                    rope_beta_slow,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("DSv4 FlashMLA tree output inverse-rope failed: {e}"))?;
            } else {
                ffi::arle_dsv4_output_inverse_rope_cuda(
                    out_ptr as *mut ffi::Half,
                    token_count as i32,
                    local_heads as i32,
                    config.head_dim as i32,
                    config.qk_rope_head_dim as i32,
                    start_pos as i32,
                    rope_base,
                    original_seq_len,
                    rope_factor,
                    rope_beta_fast,
                    rope_beta_slow,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("DSv4 FlashMLA prefill output inverse-rope failed: {e}"))?;
            }
        }
    }

    // Tree verify is PURE: the frozen forward owns no ring state — the commit
    // path re-establishes the rings for exactly the accepted prefix. Plain
    // prefill keeps rolling the window with the chunk K.
    if tree.is_none() {
        update_bf16_sw_window(ctx, sw_window_cache, k_prepared, start_pos, None, config)?;
    }

    if env_flag("ARLE_DSV4_FLASHMLA_PREFILL_SYNC")? {
        ctx.sync()?;
    }

    // Keep temporary buffers in scope until all launches that use their raw
    // pointers have been enqueued. Optional sync above is available for
    // diagnostics and for conservative lifetime validation on pod.
    drop(tp_gathered_q.take());
    drop(tp_packed_q.take());
    drop(tp_full_out.take());
    drop(q_guard);
    drop(kv_guard);
    drop(indices_guard);
    drop(topk_guard);
    drop(out_guard);
    drop(max_guard);
    drop(lse_guard);
    drop(sink_guard);

    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn try_flashmla_decode_attention(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    q_prepared: &HiddenStates,
    k_prepared: &HiddenStates,
    selected: Option<&CudaSlice<i32>>,
    compressed: Option<&HiddenStates>,
    sw_window_cache: &mut CudaSlice<half::bf16>,
    flash: &mut Dsv4FlashMlaDecodeState,
    pool: &mut Dsv4LayerKvLayout,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    tp: &TpRuntime,
    local_heads: usize,
    // Replicated decode: q already carries ALL heads — no TP gather/repack,
    // and the full output lands in local_attn directly (no slice).
    replicated: bool,
    local_attn: &mut HiddenStates,
    sm_scale: f32,
    rope_base: f32,
    original_seq_len: i32,
    rope_factor: f32,
    rope_beta_fast: f32,
    rope_beta_slow: f32,
) -> Result<bool> {
    if !dsv4_flashmla_decode_enabled()? {
        return Ok(false);
    }
    if q_prepared.seq_len != 1 {
        return Ok(false);
    }
    let start_pos_device = start_pos_device.ok_or_else(|| {
        anyhow!("DSv4 FlashMLA decode requires device start_pos for token_count=1")
    })?;
    ensure!(
        config.head_dim == 512 && config.qk_rope_head_dim == 64,
        "DSv4 FlashMLA decode only supports MODEL1 head_dim=512 rope_dim=64"
    );
    ensure!(
        local_attn.seq_len == 1,
        "DSv4 FlashMLA decode writes exactly one token"
    );

    let tp_world = tp.config().world_size;
    let tp_rank = tp.config().rank;
    let global_heads = if replicated {
        local_heads
    } else {
        local_heads
            .checked_mul(tp_world)
            .ok_or_else(|| anyhow!("DSv4 FlashMLA global head overflow"))?
    };
    ensure!(
        matches!(global_heads, 64 | 128),
        "DSv4 FlashMLA decode requires global heads 64/128, got {global_heads}"
    );

    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_pack_sw_ring");
        flashmla_pack_sw_ring(ctx, flash, pool, sw_window_cache, config)?;
    }

    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_pack_one");
        flashmla_pack_one_sw_token(ctx, flash, pool, k_prepared, start_pos_device, config)?;
    }
    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_pack_compressed");
        flashmla_pack_compressed_delta(
            ctx,
            flash,
            pool,
            compressed,
            start_pos_device,
            compress_ratio,
            config,
        )?;
    }

    let mode_int = flashmla_mode_int(mode);
    let selected_ptr_u64 = if mode == DeepSeekV4AttentionMode::CompressedSparse {
        let selected =
            selected.ok_or_else(|| anyhow!("DSv4 FlashMLA CSA missing selected topk"))?;
        let (ptr, guard) = selected.device_ptr(&ctx.stream);
        let ptr_u64 = ptr;
        drop(guard);
        ptr_u64
    } else {
        0
    };
    let (indices_ptr, indices_guard) = flash.indices.device_ptr_mut(&ctx.stream);
    let (start_ptr, start_guard) = start_pos_device.device_ptr(&ctx.stream);
    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_build_indices");
        flash_kv::dsv4_flashmla_decode_build_indices_start_pos_ptr_raw(
            ctx,
            indices_ptr,
            selected_ptr_u64,
            flash.sw_blocks,
            config.sliding_window,
            start_ptr,
            flash.max_compressed_keys,
            if mode == DeepSeekV4AttentionMode::SlidingWindow {
                1
            } else {
                compress_ratio
            },
            mode_int,
            64,
        )?;
    }
    drop(indices_guard);
    drop(start_guard);

    // topk_length + scheduler metadata are slot constants, computed once at
    // state init (`init_constant_sched_meta`) — see the capture-hazard note
    // there. Saves 43 sched-meta calls/token as a side effect.

    let (q_ptr, q_guard) = q_prepared.data.device_ptr(&ctx.stream);
    let pool_range = pool.flashmla_pages_byte_range(flash.slot_idx)?;
    let pool_buf = pool.flashmla_pool_data()?;
    ensure!(
        pool_range.end <= pool_buf.len() && pool_range.len() == flash.fp8_kv_pool_len,
        "DSv4 FlashMLA shared fwd table range {:?} invalid pool_len={} slot_len={}",
        pool_range,
        pool_buf.len(),
        flash.fp8_kv_pool_len
    );
    let pool_view = pool_buf.slice(pool_range);
    let (pool_ptr, pool_guard) = pool_view.device_ptr(&ctx.stream);
    let (out_ptr, out_guard) = local_attn.data.device_ptr_mut(&ctx.stream);
    let (lse_out_ptr, lse_guard) = flash.lse_out.device_ptr_mut(&ctx.stream);
    let (lse_accum_ptr, lse_accum_guard) = flash.lse_accum.device_ptr_mut(&ctx.stream);
    let (o_accum_ptr, o_accum_guard) = flash.o_accum.device_ptr_mut(&ctx.stream);
    let (indices_ptr, indices_guard) = flash.indices.device_ptr(&ctx.stream);
    let (topk_ptr, topk_guard) = flash.topk_length.device_ptr(&ctx.stream);
    let (sched_ptr, sched_guard) = flash.sched_meta.device_ptr(&ctx.stream);
    let (splits_ptr, splits_guard) = flash.num_splits.device_ptr(&ctx.stream);

    let q_for_flashmla = if !replicated && tp_world > 1 {
        let (gather_ptr, gather_guard) = flash.tp_gathered_q.device_ptr_mut(&ctx.stream);
        {
            let _nvtx = crate::nvtx::range("dsv4/flashmla_q_allgather");
            unsafe {
                tp.all_gather_bf16_raw(
                    ctx,
                    q_ptr as *const std::ffi::c_void,
                    local_heads * config.head_dim,
                    gather_ptr as *mut std::ffi::c_void,
                )?;
            }
        }
        drop(gather_guard);
        let (packed_ptr, packed_guard) = flash.tp_packed_q.device_ptr_mut(&ctx.stream);
        {
            let _nvtx = crate::nvtx::range("dsv4/flashmla_q_repack");
            unsafe {
                ffi::dsv4_tp_q_repack_cuda(
                    gather_ptr as *const ffi::Half,
                    packed_ptr as *mut ffi::Half,
                    tp_world as i32,
                    1,
                    local_heads as i32,
                    config.head_dim as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("DSv4 FlashMLA TP Q repack failed: {e}"))?;
            }
        }
        drop(packed_guard);
        packed_ptr as *const ffi::Half
    } else {
        q_ptr as *const ffi::Half
    };

    let (sink_base, sink_guard) = attention.attn_sink_f32.device_ptr(&ctx.stream);
    ensure!(
        if tp_world > 1 {
            attention.attn_sink_f32.len() >= global_heads
        } else {
            attention.attn_sink_f32.len() >= tp_rank * local_heads + local_heads
        },
        "DSv4 FlashMLA attn_sink_f32 len {} cannot cover heads",
        attention.attn_sink_f32.len()
    );
    let sink_ptr = if tp_world > 1 {
        sink_base as *const f32
    } else {
        unsafe { (sink_base as *const f32).add(tp_rank * local_heads) }
    };

    let flash_out_ptr = if !replicated && tp_world > 1 {
        let (full_out_ptr, full_out_guard) = flash.tp_full_out.device_ptr_mut(&ctx.stream);
        drop(full_out_guard);
        full_out_ptr as *mut ffi::Half
    } else {
        out_ptr as *mut ffi::Half
    };

    let bytes_per_token = 584_i32;
    let stride_kv_block_bytes = 64_i32 * bytes_per_token;
    let stride_q = (global_heads * config.head_dim) as i32;
    let stride_o = stride_q;
    let stride_indices = flash.topk_unified as i32;
    let stride_lse = global_heads as i32;
    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_fwd");
        unsafe {
            ffi::arle_flashmla_sm90_sparse_decode_fwd(
                q_for_flashmla,
                pool_ptr as *const ffi::Half,
                indices_ptr as *const i32,
                topk_ptr as *const i32,
                sink_ptr,
                flash_out_ptr,
                lse_out_ptr as *mut f32,
                lse_accum_ptr as *mut f32,
                o_accum_ptr as *mut f32,
                sched_ptr as *const i32,
                splits_ptr as *const i32,
                1,
                1,
                global_heads as i32,
                1,
                config.head_dim as i32,
                config.head_dim as i32,
                (flash.sw_blocks + flash.comp_blocks) as i32,
                64,
                stride_indices,
                flash.num_sm_parts,
                DSV4_FLASHMLA_MODEL1,
                sm_scale,
                stride_q,
                stride_q,
                config.head_dim as i32,
                stride_kv_block_bytes,
                bytes_per_token,
                stride_indices,
                stride_indices,
                stride_lse,
                1,
                stride_o,
                stride_o,
                config.head_dim as i32,
                global_heads as i32,
                global_heads as i32,
                stride_o,
                stride_o,
                config.head_dim as i32,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| anyhow!("DSv4 FlashMLA sparse decode failed: {e}"))?;
        }
    }

    if !replicated && tp_world > 1 {
        let (full_out_ptr, full_out_guard) = flash.tp_full_out.device_ptr(&ctx.stream);
        {
            let _nvtx = crate::nvtx::range("dsv4/flashmla_out_slice");
            unsafe {
                ffi::dsv4_tp_out_slice_cuda(
                    full_out_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    1,
                    (global_heads * config.head_dim) as i32,
                    (local_heads * config.head_dim) as i32,
                    (tp_rank * local_heads * config.head_dim) as i32,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("DSv4 FlashMLA TP out slice failed: {e}"))?;
            }
        }
        drop(full_out_guard);
    }

    {
        let _nvtx = crate::nvtx::range("dsv4/flashmla_inverse_rope");
        unsafe {
            ffi::arle_dsv4_output_inverse_rope_start_pos_ptr_cuda(
                out_ptr as *mut ffi::Half,
                1,
                local_heads as i32,
                config.head_dim as i32,
                config.qk_rope_head_dim as i32,
                start_ptr as *const i32,
                rope_base,
                original_seq_len,
                rope_factor,
                rope_beta_fast,
                rope_beta_slow,
                ctx.stream.cu_stream(),
            )
            .result()
            .map_err(|e| anyhow!("DSv4 FlashMLA output inverse-rope failed: {e}"))?;
        }
    }

    drop(q_guard);
    drop(pool_guard);
    drop(out_guard);
    drop(lse_guard);
    drop(lse_accum_guard);
    drop(o_accum_guard);
    drop(indices_guard);
    drop(topk_guard);
    drop(sched_guard);
    drop(splits_guard);
    drop(sink_guard);

    update_bf16_sw_window(
        ctx,
        sw_window_cache,
        k_prepared,
        start_pos,
        Some(start_pos_device),
        config,
    )?;
    Ok(true)
}

/// RMSNorm a `HiddenStates` in place into a fresh buffer (the MLA Q/KV LoRA
/// norms `q_norm` / `kv_norm`). Thin wrapper over the shared batched RMSNorm.
fn mla_rms_norm(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
) -> Result<HiddenStates> {
    // SAFETY: rms_norm_batched_cuda writes the full output buffer.
    let mut out = unsafe { HiddenStates::uninit(ctx, x.hidden_dim, x.seq_len)? };
    {
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
        // SAFETY: buffers valid on ctx.stream; out matches x shape.
        unsafe {
            ffi::rms_norm_batched_cuda(
                x_ptr as *const ffi::Half,
                w_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                x.hidden_dim as i32,
                x.seq_len as i32,
                eps,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    Ok(out)
}

fn mla_rms_norm_decode_slice(
    ctx: &DeviceContext,
    x: &HiddenStates,
    offset: usize,
    width: usize,
    weight: &DeviceVec,
    eps: f32,
) -> Result<HiddenStates> {
    ensure!(
        x.seq_len == 1,
        "DSv4 fused wqkv slice RMSNorm is decode-only, got seq_len={}",
        x.seq_len
    );
    ensure!(
        offset + width <= x.hidden_dim,
        "DSv4 fused wqkv slice out of range: offset={offset} width={width} hidden_dim={}",
        x.hidden_dim
    );
    ensure!(
        weight.len == width,
        "DSv4 fused wqkv slice norm weight len {} != slice width {width}",
        weight.len
    );
    let mut out = unsafe { HiddenStates::uninit(ctx, width, 1)? };
    {
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
        let x_ptr = unsafe { (x_ptr as *const ffi::Half).add(offset) };
        unsafe {
            ffi::rms_norm_batched_cuda(
                x_ptr,
                w_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                width as i32,
                1,
                eps,
                ctx.stream.cu_stream(),
            )
            .result()?;
        }
    }
    Ok(out)
}

fn run_fused_wqkv_decode(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    hidden: &HiddenStates,
    scratch: &mut Dsv4FusedWqkvDecodeScratch,
) -> Result<(HiddenStates, HiddenStates, HiddenStates)> {
    ensure!(
        hidden.seq_len == 1,
        "DSv4 fused wqkv decode path requires seq_len=1, got {}",
        hidden.seq_len
    );
    ensure!(
        hidden.hidden_dim == scratch.hidden_dim && hidden.hidden_dim == config.hidden_size,
        "DSv4 fused wqkv hidden dim mismatch: hidden={} scratch={} config={}",
        hidden.hidden_dim,
        scratch.hidden_dim,
        config.hidden_size
    );
    ensure!(
        scratch.q_lora_rank == attention.wq_a.rows && scratch.head_dim == attention.wkv.rows,
        "DSv4 fused wqkv scratch shape mismatch: scratch q={} kv={} weights q={} kv={}",
        scratch.q_lora_rank,
        scratch.head_dim,
        attention.wq_a.rows,
        attention.wkv.rows
    );
    let cache = attention.wqkv_a_deepgemm.as_ref().ok_or_else(|| {
        anyhow!("DSv4 fused wqkv decode requested but fused cache was not loaded")
    })?;
    ensure!(
        cache.rows == scratch.q_lora_rank + scratch.head_dim && cache.cols == scratch.hidden_dim,
        "DSv4 fused wqkv cache shape {}x{} != expected {}x{}",
        cache.rows,
        cache.cols,
        scratch.q_lora_rank + scratch.head_dim,
        scratch.hidden_dim
    );
    let scale_cols = scratch.hidden_dim.div_ceil(128);
    ensure!(
        scratch.input_scales.len() >= scratch.scale_stride_m * scale_cols,
        "DSv4 fused wqkv scale scratch too small"
    );
    let stream = ctx.stream.cu_stream();
    unsafe {
        cuda_moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
            cache_ptr(&hidden.data, ctx),
            cache_ptr(&scratch.input_fp8, ctx),
            cache_ptr(&scratch.input_scales, ctx),
            cache_ptr(&scratch.active_experts, ctx),
            cache_ptr(&scratch.active_offsets, ctx),
            cache_ptr(&scratch.active_counts, ctx),
            1,
            scratch.max_m,
            scratch.hidden_dim,
            scratch.scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 fused wqkv activation quantize failed: {e}"))?;
        cuda_moe::dsv4_deepgemm_fp8_gemm_nt(
            cache_ptr(&scratch.input_fp8, ctx),
            cache_ptr(&scratch.input_scales, ctx),
            cache_ptr(&cache.weight, ctx),
            cache_ptr(&cache.scales, ctx),
            cache_ptr(&scratch.qkv_raw.data, ctx),
            1,
            cache.rows,
            cache.cols,
            scratch.scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 fused wqkv DeepGEMM dense failed: {e}"))?;
    }
    let c_q_normed = mla_rms_norm_decode_slice(
        ctx,
        &scratch.qkv_raw,
        0,
        scratch.q_lora_rank,
        &attention.q_norm,
        config.rms_norm_eps,
    )?;
    let kv_normed = mla_rms_norm_decode_slice(
        ctx,
        &scratch.qkv_raw,
        scratch.q_lora_rank,
        scratch.head_dim,
        &attention.kv_norm,
        config.rms_norm_eps,
    )?;

    let mut q_raw = unsafe { HiddenStates::uninit(ctx, attention.wq_b.rows, 1)? };
    let nvtx_wq_b = crate::nvtx::range("dsv4/linear/wq_b");
    match (
        dsv4_decode_proj_deepgemm_enabled(),
        attention.wq_b_deepgemm.as_ref(),
    ) {
        (true, Some(cache)) => {
            // Lever #1: wq_b (M=1) through tensor-core DeepGEMM instead of the
            // scalar GEMV. Quantize c_q_normed (K=q_lora_rank) into the fused
            // scratch FP8 buffer (already consumed by the wq_a|wkv GEMM above, so
            // safe to reuse on this stream), then DeepGEMM dense GEMM.
            let k = scratch.q_lora_rank;
            ensure!(
                cache.cols == k && cache.rows == attention.wq_b.rows,
                "DSv4 wq_b DeepGEMM cache shape {}x{} != expected {}x{}",
                cache.rows,
                cache.cols,
                attention.wq_b.rows,
                k
            );
            crate::linear_profile::profile(ctx, "dsv4/linear/wq_b", || -> Result<()> {
                let stream = ctx.stream.cu_stream();
                // SAFETY: all buffers live on ctx.stream; K=q_lora_rank ≤ hidden_dim
                // so the fused scratch (sized for hidden_dim) covers the FP8 +
                // scale extents.
                unsafe {
                    cuda_moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
                        cache_ptr(&c_q_normed.data, ctx),
                        cache_ptr(&scratch.input_fp8, ctx),
                        cache_ptr(&scratch.input_scales, ctx),
                        cache_ptr(&scratch.active_experts, ctx),
                        cache_ptr(&scratch.active_offsets, ctx),
                        cache_ptr(&scratch.active_counts, ctx),
                        1,
                        scratch.max_m,
                        k,
                        scratch.scale_stride_m,
                        stream,
                    )
                    .map_err(|e| anyhow!("DSv4 wq_b activation quantize failed: {e}"))?;
                    cuda_moe::dsv4_deepgemm_fp8_gemm_nt(
                        cache_ptr(&scratch.input_fp8, ctx),
                        cache_ptr(&scratch.input_scales, ctx),
                        cache_ptr(&cache.weight, ctx),
                        cache_ptr(&cache.scales, ctx),
                        cache_ptr(&q_raw.data, ctx),
                        1,
                        cache.rows,
                        cache.cols,
                        scratch.scale_stride_m,
                        stream,
                    )
                    .map_err(|e| anyhow!("DSv4 wq_b DeepGEMM dense failed: {e}"))?;
                }
                Ok(())
            })?;
        }
        _ => {
            crate::linear_profile::profile(ctx, "dsv4/linear/wq_b", || {
                dsv4_linear(ctx, &attention.wq_b, &c_q_normed, &mut q_raw)
            })?;
        }
    }
    drop(nvtx_wq_b);
    Ok((c_q_normed, q_raw, kv_normed))
}

/// One DSv4 MLA attention block (SlidingWindow / CompressedSparse /
/// HybridCompressed, dispatched on `mode` / `compress_ratio`).
///
/// `hidden` is the post-attn-LN input `[hidden_size, token_count]`;
/// `state` holds this layer's per-slot bf16 sliding-window ring plus compressor
/// pending/compressed pools. `start_pos` is the absolute position of `hidden`'s
/// first token (0 for a fresh prefill). Writes `[hidden_size, token_count]` into
/// `out` (the O-LoRA output, pre-TP-all-reduce — the model layer-loop owns the
/// row-parallel sum). FlashMLA-FP8 decode stays gated (perf path).
///
/// `tp_rank` is this rank's tensor-parallel index. The per-head `attn_sink`
/// vector is loaded WHOLE on every rank (no TP slice), so the SW/hybrid kernels
/// must skip to this rank's head block via `sink_offset = tp_rank * local_heads`
/// — otherwise every non-zero rank reads rank-0's sink logits and the attention
/// output diverges by a small head-dependent margin (multi-GPU only).
#[allow(clippy::too_many_arguments)]
pub(crate) fn mla_attention(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    attention: &Dsv4Attention,
    mode: DeepSeekV4AttentionMode,
    compress_ratio: usize,
    layer_idx: usize,
    hidden: &HiddenStates,
    state: &mut Dsv4LayerAttentionState,
    pool: &mut Dsv4LayerKvLayout,
    dsa_shared: Option<&mut Dsv4DsaSharedScratch>,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    // Spec tree-verify chunk: per-row positions + branch topology. Routes the
    // whole multi-row chunk through ONE FlashMLA sparse forward per layer
    // (every mode incl. pure-SW) with zero ring writes; `None` everywhere else.
    tree: Option<&Dsv4TreeAttnMeta>,
    tp: &TpRuntime,
    out: &mut HiddenStates,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    ensure!(
        hidden.hidden_dim == config.hidden_size,
        "DSv4 MLA hidden dim {} != hidden_size {}",
        hidden.hidden_dim,
        config.hidden_size
    );

    let head_dim = config.head_dim;
    let token_count = hidden.seq_len;
    // Replicated decode attention: single-row chunks compute the FULL block
    // per rank from the full-width weights — zero attention collectives
    // (caller skips the AR via the same predicate). Prefill / multi-row
    // chunks keep the sharded math.
    let replicated = token_count == 1
        && dsv4_replicated_attn_enabled()
        && attention.wq_b_full.is_some()
        && attention.wo_a_full.is_some();
    let wq_b_active = if replicated {
        attention
            .wq_b_full
            .as_ref()
            .expect("replicated gate checked")
    } else {
        &attention.wq_b
    };
    let local_width = wq_b_active.rows;
    ensure!(
        head_dim > 0 && local_width.is_multiple_of(head_dim),
        "DSv4 MLA local q width {local_width} is not a multiple of head_dim {head_dim}"
    );
    let local_heads = local_width / head_dim;
    ensure!(local_heads > 0, "DSv4 MLA requires at least one local head");
    let tp_rank = tp.config().rank;
    // This rank owns global heads [tp_rank*local_heads, +local_heads); the
    // whole-loaded attn_sink must be indexed from that offset (see fn docs).
    // Replicated: this rank owns ALL heads — offset 0.
    let sink_offset = if replicated { 0 } else { tp_rank * local_heads };
    let wo_a_active = if replicated {
        attention
            .wo_a_full
            .as_ref()
            .expect("replicated gate checked")
    } else {
        &attention.wo_a
    };
    ensure!(
        attention.wkv.rows == head_dim,
        "DSv4 MLA wkv rows {} != head_dim {head_dim}",
        attention.wkv.rows
    );
    ensure!(
        wo_a_active.cols == local_width,
        "DSv4 MLA wo_a cols {} != local attention width {local_width}",
        wo_a_active.cols
    );
    ensure!(
        attention.wo_b.rows == out.hidden_dim && out.seq_len == token_count,
        "DSv4 MLA output shape mismatch: wo_b rows {} out {}x{} expected {}x{}",
        attention.wo_b.rows,
        out.hidden_dim,
        out.seq_len,
        attention.wo_b.rows,
        token_count
    );
    ensure!(
        config.sliding_window > 0,
        "DSv4 MLA requires a non-zero sliding_window"
    );
    ensure!(
        config.qk_rope_head_dim <= head_dim,
        "DSv4 MLA rope dim {} exceeds head_dim {head_dim}",
        config.qk_rope_head_dim
    );
    ensure!(
        state.sw_window_cache.len() == config.sliding_window * head_dim,
        "DSv4 MLA SW window cache len {} != sliding_window*head_dim {}",
        state.sw_window_cache.len(),
        config.sliding_window * head_dim
    );
    ensure!(
        attention.attn_sink.len >= sink_offset + local_heads,
        "DSv4 MLA attn_sink len {} cannot cover rank {tp_rank} heads [{sink_offset}, {})",
        attention.attn_sink.len,
        sink_offset + local_heads
    );

    let rope = &config.rope_parameters;
    // RoPE base/YaRN is PER-LAYER, matching the canonical SGLang impl
    // (deepseek_v4.py:271 `rope_base = compress_rope_theta if compress_ratio else
    // rope_theta`, and fused_qk_norm_rope_swa_store which ropes Q + SW-K with ONE
    // per-layer cos/sin cache): compressed layers (CSA cr=4 / HCA cr=128) rope Q,
    // SW-K, the output inverse-rope AND the compressor with compress_rope_theta +
    // YaRN(original_max_position_embeddings); pure-SW layers (cr=0) use rope_theta
    // with no YaRN. Q MUST share the compressed-key theta or Q·compressed-K phase
    // mismatches and long context (>~80 tok) collapses to garbage. (The prior
    // "always rope_theta, no YaRN" matched the old-tree reference.rs /
    // errors/2026-05-29-dsv4-longctx-rope-conflation, not the canonical model —
    // SGLang ropes everything on a compressed layer at compress_rope_theta.)
    let (rope_base, original_seq_len) = if compress_ratio > 0 {
        let osl = i32::try_from(rope.original_max_position_embeddings).map_err(|_| {
            anyhow!(
                "DSv4 original_max_position_embeddings {} overflows i32",
                rope.original_max_position_embeddings
            )
        })?;
        (config.compress_rope_theta, osl)
    } else {
        (config.rope_theta, 0i32)
    };
    let start_pos_i32 = i32::try_from(start_pos)
        .map_err(|_| anyhow::anyhow!("DSv4 MLA start_pos {start_pos} overflows i32"))?;

    // ── 1+2. Q/KV LoRA. Decode uses the existing B=1 fused (`wq_a | wkv`)
    // path. Prefill can opt into the same fused weight cache via
    // ARLE_DSV4_FP8_LINEAR_DEEPGEMM; the default branch below preserves the
    // scalar reference order exactly. Replicated decode forces the plain arm:
    // the fused/DeepGEMM scratches and weight caches are shard-shaped.
    let fused_wqkv = token_count == 1 && !replicated && dsv4_fused_wqkv_decode_enabled()?;
    let (c_q_normed, q_raw, kv_normed) = if fused_wqkv {
        let scratch = state.fused_wqkv.as_mut().ok_or_else(|| {
            anyhow!("DSv4 fused wqkv decode requested but decode scratch was not allocated")
        })?;
        let nvtx_wqkv = crate::nvtx::range("dsv4/linear/wqkv_a_fused");
        let out = crate::linear_profile::profile(ctx, "dsv4/linear/wqkv_a_fused", || {
            run_fused_wqkv_decode(ctx, config, attention, hidden, scratch)
        })?;
        drop(nvtx_wqkv);
        out
    } else if token_count > 1 && dsv4_fp8_linear_deepgemm_enabled()? {
        let mut c_q = unsafe { HiddenStates::uninit(ctx, attention.wq_a.rows, token_count)? };
        let mut kv_raw = unsafe { HiddenStates::uninit(ctx, head_dim, token_count)? };
        let scratch = state.prefill_linear.as_mut().ok_or_else(|| {
            anyhow!(
                "ARLE_DSV4_FP8_LINEAR_DEEPGEMM=1 but prefill fused wqkv scratch was not allocated"
            )
        })?;
        let nvtx_wqkv = crate::nvtx::range("dsv4/linear/wqkv_a_fused_prefill");
        crate::linear_profile::profile(ctx, "dsv4/linear/wqkv_a_fused_prefill", || {
            run_fused_wqkv_prefill(ctx, attention, hidden, scratch, &mut c_q, &mut kv_raw)
        })?;
        drop(nvtx_wqkv);
        keepalive.keep_hidden(&c_q);
        let c_q_normed = mla_rms_norm(ctx, &c_q, &attention.q_norm, config.rms_norm_eps)?;
        keepalive.keep_hidden(&c_q_normed);
        let mut q_raw = unsafe { HiddenStates::uninit(ctx, local_width, token_count)? };
        let nvtx_wq_b = crate::nvtx::range("dsv4/linear/wq_b");
        // Prefill wq_b → DeepGEMM (off the scalar dsv4_fp8_gemv_batch, the 62% of
        // mla_attn prefill). Reuses the prefill fused-wqkv FP8 scratch since
        // K=q_lora_rank ≤ hidden_dim. Opt-in until the prefill A/B licenses it.
        if let Some(cache) = attention
            .wq_b_deepgemm
            .as_ref()
            .filter(|_| dsv4_prefill_proj_deepgemm_enabled())
        {
            let scratch = state.prefill_linear.as_mut().ok_or_else(|| {
                anyhow!("DSv4 prefill wq_b DeepGEMM requested but prefill scratch not allocated")
            })?;
            crate::linear_profile::profile(ctx, "dsv4/linear/wq_b", || {
                prefill_proj_deepgemm(ctx, scratch, cache, &c_q_normed, &mut q_raw)
            })?;
        } else {
            crate::linear_profile::profile(ctx, "dsv4/linear/wq_b", || {
                dsv4_linear(ctx, &attention.wq_b, &c_q_normed, &mut q_raw)
            })?;
        }
        drop(nvtx_wq_b);
        keepalive.keep_hidden(&q_raw);

        // KV latent: wkv (down to the single compressed latent) → kv_norm.
        keepalive.keep_hidden(&kv_raw);
        let kv_normed = mla_rms_norm(ctx, &kv_raw, &attention.kv_norm, config.rms_norm_eps)?;
        keepalive.keep_hidden(&kv_normed);
        (c_q_normed, q_raw, kv_normed)
    } else {
        // Q-LoRA: wq_a (down) → q_norm RMSNorm → wq_b (up to per-head Q).
        // SAFETY: dsv4_linear writes the full c_q buffer.
        let mut c_q = unsafe { HiddenStates::uninit(ctx, attention.wq_a.rows, token_count)? };
        let nvtx_wq_a = crate::nvtx::range("dsv4/linear/wq_a");
        crate::linear_profile::profile(ctx, "dsv4/linear/wq_a", || {
            dsv4_linear(ctx, &attention.wq_a, hidden, &mut c_q)
        })?;
        drop(nvtx_wq_a);
        keepalive.keep_hidden(&c_q);
        let c_q_normed = mla_rms_norm(ctx, &c_q, &attention.q_norm, config.rms_norm_eps)?;
        keepalive.keep_hidden(&c_q_normed);
        // SAFETY: dsv4_linear writes the full q_raw buffer.
        let mut q_raw = unsafe { HiddenStates::uninit(ctx, local_width, token_count)? };
        let nvtx_wq_b = crate::nvtx::range("dsv4/linear/wq_b");
        crate::linear_profile::profile(ctx, "dsv4/linear/wq_b", || {
            dsv4_linear(ctx, wq_b_active, &c_q_normed, &mut q_raw)
        })?;
        drop(nvtx_wq_b);
        keepalive.keep_hidden(&q_raw);

        // KV latent: wkv (down to the single compressed latent) → kv_norm.
        // SAFETY: dsv4_linear writes the full kv_raw buffer.
        let mut kv_raw = unsafe { HiddenStates::uninit(ctx, head_dim, token_count)? };
        let nvtx_wkv = crate::nvtx::range("dsv4/linear/wkv");
        crate::linear_profile::profile(ctx, "dsv4/linear/wkv", || {
            dsv4_linear(ctx, &attention.wkv, hidden, &mut kv_raw)
        })?;
        drop(nvtx_wkv);
        keepalive.keep_hidden(&kv_raw);
        let kv_normed = mla_rms_norm(ctx, &kv_raw, &attention.kv_norm, config.rms_norm_eps)?;
        keepalive.keep_hidden(&kv_normed);
        (c_q_normed, q_raw, kv_normed)
    };
    keepalive.keep_hidden(&c_q_normed);
    keepalive.keep_hidden(&q_raw);
    keepalive.keep_hidden(&kv_normed);

    // ── 3. Partial RoPE on the trailing rope_dim cols of Q (per head) and K.
    // SAFETY: dsv4_prepare_qk_cuda writes both full output buffers.
    let mut q_prepared = unsafe { HiddenStates::uninit(ctx, local_width, token_count)? };
    let mut k_prepared = unsafe { HiddenStates::uninit(ctx, head_dim, token_count)? };
    {
        let (q_raw_ptr, _qr) = q_raw.data.device_ptr(&ctx.stream);
        let (k_raw_ptr, _kr) = kv_normed.data.device_ptr(&ctx.stream);
        let (q_out_ptr, _qo) = q_prepared.data.device_ptr_mut(&ctx.stream);
        let (k_out_ptr, _ko) = k_prepared.data.device_ptr_mut(&ctx.stream);
        // SAFETY: all buffers valid on ctx.stream; head/dim args checked above.
        unsafe {
            if let Some(meta) = tree {
                // Tree-verify chunk: per-row absolute positions (siblings
                // repeat) instead of start_pos + row.
                let (pos_ptr, _pg) = meta.positions.device_ptr(&ctx.stream);
                ffi::dsv4_prepare_qk_fused_batch_start_pos_cuda(
                    q_raw_ptr as *const ffi::Half,
                    k_raw_ptr as *const ffi::Half,
                    q_out_ptr as *mut ffi::Half,
                    k_out_ptr as *mut ffi::Half,
                    token_count as i32,
                    local_heads as i32,
                    head_dim as i32,
                    config.qk_rope_head_dim as i32,
                    pos_ptr as *const i32,
                    config.rms_norm_eps,
                    rope_base,
                    original_seq_len,
                    rope.factor,
                    rope.beta_fast,
                    rope.beta_slow,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            } else if let Some(start_pos_device) = start_pos_device {
                let (start_ptr, _sg) = start_pos_device.device_ptr(&ctx.stream);
                ffi::dsv4_prepare_qk_start_pos_ptr_cuda(
                    q_raw_ptr as *const ffi::Half,
                    k_raw_ptr as *const ffi::Half,
                    q_out_ptr as *mut ffi::Half,
                    k_out_ptr as *mut ffi::Half,
                    token_count as i32,
                    local_heads as i32,
                    head_dim as i32,
                    config.qk_rope_head_dim as i32,
                    start_ptr as *const i32,
                    config.rms_norm_eps,
                    rope_base,
                    original_seq_len,
                    rope.factor,
                    rope.beta_fast,
                    rope.beta_slow,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            } else {
                ffi::dsv4_prepare_qk_cuda(
                    q_raw_ptr as *const ffi::Half,
                    k_raw_ptr as *const ffi::Half,
                    q_out_ptr as *mut ffi::Half,
                    k_out_ptr as *mut ffi::Half,
                    token_count as i32,
                    local_heads as i32,
                    head_dim as i32,
                    config.qk_rope_head_dim as i32,
                    start_pos_i32,
                    config.rms_norm_eps,
                    rope_base,
                    original_seq_len,
                    rope.factor,
                    rope.beta_fast,
                    rope.beta_slow,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
    }
    keepalive.keep_hidden(&q_prepared);
    keepalive.keep_hidden(&k_prepared);
    dsv4_dump_kprep(ctx, layer_idx, "k_prepared", &k_prepared, start_pos);
    dsv4_dump_kprep(ctx, layer_idx, "q_prepared", &q_prepared, start_pos);

    let sm_scale = 1.0f32 / (head_dim as f32).sqrt();
    // SAFETY: the SW/hybrid attention kernel writes the full local_attn buffer.
    let mut local_attn = unsafe { HiddenStates::uninit(ctx, local_width, token_count)? };

    if mode == DeepSeekV4AttentionMode::SlidingWindow {
        // ── 4a. SW: windowed attention + per-head sink + output inverse-RoPE.
        // The kernel reads the pre-roped q/k, attends over the bf16 SW ring cache
        // (which it also updates), adds the sink, and un-rotates the rope tail of
        // the OUTPUT (sign = -1) before returning.
        if let Some(meta) = tree {
            // Tree-verify chunk: the contiguous swa kernel cannot express the
            // branch mask (siblings share positions), so route through the
            // sparse forward over [SW cache | chunk K] with tree indices.
            let used = try_flashmla_prefill_attention(
                ctx,
                config,
                attention,
                mode,
                compress_ratio,
                &q_prepared,
                &k_prepared,
                None,
                None,
                &mut state.sw_window_cache,
                start_pos,
                Some(meta),
                tp,
                local_heads,
                &mut local_attn,
                sm_scale,
                rope_base,
                original_seq_len,
                rope.factor,
                rope.beta_fast,
                rope.beta_slow,
            )?;
            ensure!(
                used,
                "DSv4 spec tree verify requires the FlashMLA prefill path \
                 (ARLE_DSV4_FLASHMLA_PREFILL disabled?)"
            );
            // Falls through to the shared O-projection tail below.
        } else {
            let flashmla_used = if dsv4_flashmla_decode_enabled()? {
                let flash = state.flashmla.as_mut().ok_or_else(|| {
                    anyhow!("ARLE_DSV4_FLASHMLA_DECODE=1 but layer state has no FlashMLA arena")
                })?;
                try_flashmla_decode_attention(
                    ctx,
                    config,
                    attention,
                    mode,
                    compress_ratio,
                    &q_prepared,
                    &k_prepared,
                    None,
                    None,
                    &mut state.sw_window_cache,
                    flash,
                    pool,
                    start_pos,
                    start_pos_device,
                    tp,
                    local_heads,
                    replicated,
                    &mut local_attn,
                    sm_scale,
                    rope_base,
                    original_seq_len,
                    rope.factor,
                    rope.beta_fast,
                    rope.beta_slow,
                )?
            } else {
                false
            };
            maybe_probe_flashmla_decode_path(
                layer_idx,
                mode,
                flashmla_used,
                token_count,
                start_pos,
            );
            if !flashmla_used {
                let (q_ptr, _qg) = q_prepared.data.device_ptr(&ctx.stream);
                let (k_ptr, _kg) = k_prepared.data.device_ptr(&ctx.stream);
                let (window_ptr, _wg) = state.sw_window_cache.device_ptr_mut(&ctx.stream);
                let (sink_ptr, _sg) = attention.attn_sink.data.device_ptr(&ctx.stream);
                let (out_ptr, _og) = local_attn.data.device_ptr_mut(&ctx.stream);
                // SAFETY: all buffers valid on ctx.stream; window sized above; sink_offset
                // skips to this rank's head block in the whole-loaded attn_sink vector.
                unsafe {
                    if let Some(start_pos_device) = start_pos_device {
                        let (start_ptr, _spg) = start_pos_device.device_ptr(&ctx.stream);
                        ffi::dsv4_swa_attention_start_pos_ptr_cuda(
                            q_ptr as *const ffi::Half,
                            k_ptr as *const ffi::Half,
                            window_ptr as *mut ffi::Half,
                            sink_ptr as *const ffi::Half,
                            out_ptr as *mut ffi::Half,
                            token_count as i32,
                            local_heads as i32,
                            head_dim as i32,
                            config.sliding_window as i32,
                            start_ptr as *const i32,
                            sink_offset as i32,
                            sm_scale,
                            config.qk_rope_head_dim as i32,
                            rope_base,
                            original_seq_len,
                            rope.factor,
                            rope.beta_fast,
                            rope.beta_slow,
                            1,
                            ctx.stream.cu_stream(),
                        )
                        .result()?;
                    } else {
                        ffi::dsv4_swa_attention_cuda(
                            q_ptr as *const ffi::Half,
                            k_ptr as *const ffi::Half,
                            window_ptr as *mut ffi::Half,
                            sink_ptr as *const ffi::Half,
                            out_ptr as *mut ffi::Half,
                            token_count as i32,
                            local_heads as i32,
                            head_dim as i32,
                            config.sliding_window as i32,
                            start_pos_i32,
                            sink_offset as i32,
                            sm_scale,
                            config.qk_rope_head_dim as i32,
                            rope_base,
                            original_seq_len,
                            rope.factor,
                            rope.beta_fast,
                            rope.beta_slow,
                            1,
                            ctx.stream.cu_stream(),
                        )
                        .result()?;
                    }
                }
            }
        }
    } else {
        // ── 4b. CSA / HCA: compressor → (CSA) indexer top-k select → hybrid
        // windowed+compressed attention.
        let compressor = attention.compressor.as_ref().ok_or_else(|| {
            anyhow::anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no compressor weights")
        })?;
        let overlap = compress_ratio < 16;
        {
            let compressor_state = state.compressor.as_mut().ok_or_else(|| {
                anyhow::anyhow!("DSv4 layer {layer_idx} is {mode:?} but has no compressor state")
            })?;
            compressor_forward(
                ctx,
                config,
                compressor,
                hidden,
                compressor_state,
                head_dim,
                compress_ratio,
                overlap,
                start_pos,
                start_pos_device,
                true,
                // YaRN on for compressed layers (matches Q/SW-K + SGLang
                // compressor freqs_cis); original_seq_len = orig_max_pos here.
                original_seq_len,
                keepalive,
            )?;
        }

        let selected = if mode == DeepSeekV4AttentionMode::CompressedSparse {
            let indexer = attention.indexer.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "DSv4 layer {layer_idx} is CompressedSparse but has no indexer weights"
                )
            })?;
            let use_official_dsa = dsv4_dsa_official_enabled()?;
            let indexer_rope_original_seq_len = if use_official_dsa {
                i32::try_from(config.rope_parameters.original_max_position_embeddings).map_err(
                    |_| {
                        anyhow!(
                            "DSv4 official DSA original_max_position_embeddings {} overflows i32",
                            config.rope_parameters.original_max_position_embeddings
                        )
                    },
                )?
            } else {
                0
            };
            let indexer_rows_before = state
                .indexer
                .as_ref()
                .map(|s| s.compressed.seq_len)
                .unwrap_or(0);
            // Indexer keys: a second compressor over index_head_dim keys (no APE
            // gate on the keys — `apply_rope = true`, head_dim = index_head_dim).
            {
                let indexer_state = state.indexer.as_mut().ok_or_else(|| {
                    anyhow::anyhow!(
                        "DSv4 layer {layer_idx} is CompressedSparse but has no indexer state"
                    )
                })?;
                compressor_forward(
                    ctx,
                    config,
                    &indexer.compressor,
                    hidden,
                    indexer_state,
                    config.index_head_dim,
                    compress_ratio,
                    true,
                    start_pos,
                    start_pos_device,
                    use_official_dsa,
                    indexer_rope_original_seq_len,
                    keepalive,
                )?;
            }
            let indexer_rows_after = state
                .indexer
                .as_ref()
                .map(|s| s.compressed.seq_len)
                .unwrap_or(0);
            let index_keys = &state
                .indexer
                .as_ref()
                .expect("indexer state checked above")
                .compressed;
            let official = state.dsa_official.as_mut();
            Some(csa_select(
                ctx,
                config,
                layer_idx,
                indexer,
                hidden,
                &c_q_normed,
                index_keys,
                official,
                dsa_shared,
                pool,
                indexer_rows_before,
                indexer_rows_after,
                start_pos,
                start_pos_device,
                compress_ratio,
                state.prefill_linear.as_mut(),
                keepalive,
            )?)
        } else {
            None
        };

        let compressed = &state
            .compressor
            .as_ref()
            .expect("compressor state checked above")
            .compressed;
        let compressed_count = compressed.seq_len;
        let compressed_capacity = compressed.data.len() / head_dim;
        let compressed_count_arg = if start_pos_device.is_some() {
            // CUDA graph replay bakes scalar launch args. In decode, the causal
            // bound is `abs_pos / compress_ratio`, so the kernel may safely see
            // the fixed capacity instead of the current compressed seq_len.
            compressed_capacity
        } else {
            compressed_count
        };
        let mode_int = match mode {
            DeepSeekV4AttentionMode::CompressedSparse => 1,
            DeepSeekV4AttentionMode::HybridCompressed => 2,
            DeepSeekV4AttentionMode::SlidingWindow => unreachable!(),
        };
        let flashmla_used = if try_flashmla_prefill_attention(
            ctx,
            config,
            attention,
            mode,
            compress_ratio,
            &q_prepared,
            &k_prepared,
            selected.as_ref(),
            Some(compressed),
            &mut state.sw_window_cache,
            start_pos,
            tree,
            tp,
            local_heads,
            &mut local_attn,
            sm_scale,
            rope_base,
            original_seq_len,
            rope.factor,
            rope.beta_fast,
            rope.beta_slow,
        )? {
            true
        } else if tree.is_some() {
            bail!(
                "DSv4 spec tree verify requires the FlashMLA prefill path \
                 (ARLE_DSV4_FLASHMLA_PREFILL disabled?)"
            );
        } else if dsv4_flashmla_decode_enabled()? {
            let flash = state.flashmla.as_mut().ok_or_else(|| {
                anyhow!("ARLE_DSV4_FLASHMLA_DECODE=1 but layer state has no FlashMLA arena")
            })?;
            try_flashmla_decode_attention(
                ctx,
                config,
                attention,
                mode,
                compress_ratio,
                &q_prepared,
                &k_prepared,
                selected.as_ref(),
                Some(compressed),
                &mut state.sw_window_cache,
                flash,
                pool,
                start_pos,
                start_pos_device,
                tp,
                local_heads,
                replicated,
                &mut local_attn,
                sm_scale,
                rope_base,
                original_seq_len,
                rope.factor,
                rope.beta_fast,
                rope.beta_slow,
            )?
        } else {
            false
        };
        maybe_probe_flashmla_decode_path(layer_idx, mode, flashmla_used, token_count, start_pos);
        if !flashmla_used {
            let (q_ptr, _qg) = q_prepared.data.device_ptr(&ctx.stream);
            let (k_ptr, _kg) = k_prepared.data.device_ptr(&ctx.stream);
            let (window_ptr, _wg) = state.sw_window_cache.device_ptr_mut(&ctx.stream);
            let (sink_ptr, _sg) = attention.attn_sink.data.device_ptr(&ctx.stream);
            let (out_ptr, _og) = local_attn.data.device_ptr_mut(&ctx.stream);
            let (comp_ptr, _cguard) = if compressed_count_arg > 0 {
                let (p, g) = compressed.data.device_ptr(&ctx.stream);
                (p as *const ffi::Half, Some(g))
            } else {
                (std::ptr::null(), None)
            };
            let (sel_ptr, _sguard) = match selected.as_ref() {
                Some(sel) => {
                    let (p, g) = sel.device_ptr(&ctx.stream);
                    (p as *const i32, Some(g))
                }
                None => (std::ptr::null(), None),
            };
            // SAFETY: all buffers valid on ctx.stream; compressed/selected may be null
            // (the kernel branches on compressed_count / mode). write_window_cache=1
            // updates the bf16 SW ring inline.
            unsafe {
                if let Some(start_pos_device) = start_pos_device {
                    let (start_ptr, _spg) = start_pos_device.device_ptr(&ctx.stream);
                    ffi::dsv4_hybrid_attention_start_pos_ptr_cuda(
                        q_ptr as *const ffi::Half,
                        k_ptr as *const ffi::Half,
                        window_ptr as *mut ffi::Half,
                        comp_ptr,
                        sel_ptr,
                        sink_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        token_count as i32,
                        local_heads as i32,
                        head_dim as i32,
                        config.sliding_window as i32,
                        start_ptr as *const i32,
                        sink_offset as i32,
                        sm_scale,
                        config.qk_rope_head_dim as i32,
                        rope_base,
                        original_seq_len,
                        rope.factor,
                        rope.beta_fast,
                        rope.beta_slow,
                        mode_int,
                        compress_ratio as i32,
                        compressed_count_arg as i32,
                        config.index_topk as i32,
                        1,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                } else {
                    ffi::dsv4_hybrid_attention_cuda(
                        q_ptr as *const ffi::Half,
                        k_ptr as *const ffi::Half,
                        window_ptr as *mut ffi::Half,
                        comp_ptr,
                        sel_ptr,
                        sink_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        token_count as i32,
                        local_heads as i32,
                        head_dim as i32,
                        config.sliding_window as i32,
                        start_pos_i32,
                        sink_offset as i32,
                        sm_scale,
                        config.qk_rope_head_dim as i32,
                        rope_base,
                        original_seq_len,
                        rope.factor,
                        rope.beta_fast,
                        rope.beta_slow,
                        mode_int,
                        compress_ratio as i32,
                        compressed_count_arg as i32,
                        config.index_topk as i32,
                        1,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
        }
        if let Some(sel) = selected.as_ref() {
            keepalive.keep_i32(sel);
        }
    }
    keepalive.keep_hidden(&local_attn);

    if dsv4_attn_dump_enabled() {
        dsv4_dump_attn_output(ctx, layer_idx, mode, &local_attn)?;
    }

    // ── 5. O-LoRA: wo_a (per o-group, down to the output latent) → wo_b (up
    // back to hidden). Row-parallel: the all-reduce-sum is the model's concern.
    // SAFETY: dsv4_linear writes the full latent buffer.
    mla_oproj(
        ctx,
        attention,
        state,
        &local_attn,
        token_count,
        wo_a_active,
        replicated,
        keepalive,
        out,
    )
}

/// O-LoRA output projection, extracted from `mla_attention` so the batched-decode
/// path can call it ONCE over [N] rows (Phase 4): `wo_a` (down to the output latent)
/// → `wo_b` (up to hidden) into `out`. Row-parallel — the all-reduce-sum is the
/// caller's concern. Decode (token_count==1) and prefill (token_count>1) DeepGEMM
/// paths + the scalar fallback are preserved byte-for-byte; batched decode passes
/// token_count=N to hit the prefill DeepGEMM branch (M=N amortizes the wo weight read).
#[allow(clippy::too_many_arguments)]
fn mla_oproj(
    ctx: &DeviceContext,
    attention: &Dsv4Attention,
    state: &mut Dsv4LayerAttentionState,
    local_attn: &HiddenStates,
    token_count: usize,
    // Replicated decode: full-width wo_a (complete output, caller skips the
    // AR); the shard-shaped DeepGEMM caches are bypassed.
    wo_a_active: &DeviceMatrix,
    replicated: bool,
    keepalive: &mut Dsv4ForwardKeepalive,
    out: &mut HiddenStates,
) -> Result<()> {
    // SAFETY: dsv4_linear writes the full latent buffer.
    let mut latent = unsafe { HiddenStates::uninit(ctx, wo_a_active.rows, token_count)? };
    let wo_dg = token_count == 1
        && !replicated
        && dsv4_decode_proj_deepgemm_enabled()
        && state.fused_wqkv.is_some()
        && attention.wo_a_deepgemm.is_some()
        && attention.wo_b_deepgemm.is_some();
    if wo_dg {
        // Lever #1b: wo_a/wo_b (M=1) through tensor-core DeepGEMM, reusing the
        // fused-wqkv FP8 scratch (local_width == hidden_size on DSv4-Flash).
        let scratch = state.fused_wqkv.as_ref().expect("wo_dg gate checked");
        let wo_a_cache = attention
            .wo_a_deepgemm
            .as_ref()
            .expect("wo_dg gate checked");
        let wo_b_cache = attention
            .wo_b_deepgemm
            .as_ref()
            .expect("wo_dg gate checked");
        let nvtx_wo_a = crate::nvtx::range("dsv4/linear/wo_a");
        crate::linear_profile::profile(ctx, "dsv4/linear/wo_a", || {
            decode_proj_deepgemm(
                ctx,
                scratch,
                wo_a_cache,
                local_attn,
                &mut latent,
                attention.wo_a.cols,
            )
        })?;
        drop(nvtx_wo_a);
        keepalive.keep_hidden(&latent);
        let nvtx_wo_b = crate::nvtx::range("dsv4/linear/wo_b");
        crate::linear_profile::profile(ctx, "dsv4/linear/wo_b", || {
            decode_proj_deepgemm(ctx, scratch, wo_b_cache, &latent, out, attention.wo_b.cols)
        })?;
        drop(nvtx_wo_b);
    } else if token_count > 1
        && dsv4_prefill_proj_deepgemm_enabled()
        && attention.wo_a_deepgemm.is_some()
        && attention.wo_b_deepgemm.is_some()
        && state.prefill_linear.is_some()
    {
        // Prefill wo_a/wo_b (M=token_count) → DeepGEMM, off the scalar fp8_gemv
        // (same lever as prefill wq_b; reuses the prefill FP8 scratch).
        let wo_a_cache = attention
            .wo_a_deepgemm
            .as_ref()
            .expect("wo prefill gate checked");
        let wo_b_cache = attention
            .wo_b_deepgemm
            .as_ref()
            .expect("wo prefill gate checked");
        let nvtx_wo_a = crate::nvtx::range("dsv4/linear/wo_a");
        {
            let scratch = state
                .prefill_linear
                .as_mut()
                .expect("wo prefill gate checked");
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_a", || {
                prefill_proj_deepgemm(ctx, scratch, wo_a_cache, local_attn, &mut latent)
            })?;
        }
        drop(nvtx_wo_a);
        keepalive.keep_hidden(&latent);
        let nvtx_wo_b = crate::nvtx::range("dsv4/linear/wo_b");
        {
            let scratch = state
                .prefill_linear
                .as_mut()
                .expect("wo prefill gate checked");
            crate::linear_profile::profile(ctx, "dsv4/linear/wo_b", || {
                prefill_proj_deepgemm(ctx, scratch, wo_b_cache, &latent, out)
            })?;
        }
        drop(nvtx_wo_b);
    } else {
        let nvtx_wo_a = crate::nvtx::range("dsv4/linear/wo_a");
        crate::linear_profile::profile(ctx, "dsv4/linear/wo_a", || {
            dsv4_linear(ctx, wo_a_active, local_attn, &mut latent)
        })?;
        drop(nvtx_wo_a);
        keepalive.keep_hidden(&latent);
        let nvtx_wo_b = crate::nvtx::range("dsv4/linear/wo_b");
        crate::linear_profile::profile(ctx, "dsv4/linear/wo_b", || {
            dsv4_linear(ctx, &attention.wo_b, &latent, out)
        })?;
        drop(nvtx_wo_b);
    }
    Ok(())
}

/// Run one compressor sub-block over `hidden`, updating the per-slot bf16
/// compressed-key pool for the absolute `[0, start_pos + token_count)` range.
///
/// `wkv`/`wgate` project the hidden into the per-block KV / gating-score streams
/// (`width = 2*head_dim` when `overlap`, else `head_dim`); `dsv4_compressor_update_cuda`
/// folds them through `ape` + RMSNorm(`norm`) + compress-rope into one row per
/// `compress_ratio` tokens. `apply_rope = false` skips the rope tail (indexer
/// keys).
#[allow(clippy::too_many_arguments)]
fn compressor_forward(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    compressor: &Dsv4Compressor,
    hidden: &HiddenStates,
    state: &mut Dsv4CompressorState,
    head_dim: usize,
    ratio: usize,
    overlap: bool,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    apply_rope: bool,
    rope_original_seq_len: i32,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<()> {
    ensure!(ratio > 0, "DSv4 compressor ratio must be non-zero");
    let width = if overlap { 2 * head_dim } else { head_dim };
    ensure!(
        compressor.wkv.rows == width && compressor.wgate.rows == width,
        "DSv4 compressor rows mismatch: wkv={} wgate={} expected width={width}",
        compressor.wkv.rows,
        compressor.wgate.rows
    );
    let token_count = hidden.seq_len;
    let total = start_pos + token_count;
    let compressed_rows = total / ratio;
    let start_pos_i32 = i32::try_from(start_pos)
        .map_err(|_| anyhow::anyhow!("DSv4 compressor start_pos {start_pos} exceeds i32"))?;
    let pending_len = start_pos % ratio;
    let pending_len_i32 = i32::try_from(pending_len)
        .map_err(|_| anyhow::anyhow!("DSv4 compressor pending_len {pending_len} exceeds i32"))?;
    let compressed_base = start_pos / ratio;
    let compressed_base_i32 = i32::try_from(compressed_base).map_err(|_| {
        anyhow::anyhow!("DSv4 compressor compressed_base {compressed_base} exceeds i32")
    })?;
    ensure!(
        state.compressed.hidden_dim == head_dim,
        "DSv4 compressor state hidden_dim {} != head_dim {head_dim}",
        state.compressed.hidden_dim
    );
    let compressed_capacity = state.compressed.data.len() / head_dim;
    ensure!(
        compressed_rows <= compressed_capacity,
        "DSv4 compressor compressed rows {compressed_rows} exceed state capacity {compressed_capacity}"
    );

    // SAFETY: dsv4_linear writes the full compressor kv buffer.
    let mut kv_raw = unsafe { HiddenStates::uninit(ctx, width, token_count)? };
    let nvtx_compressor_wkv = crate::nvtx::range("dsv4/linear/compressor_wkv");
    crate::linear_profile::profile(ctx, "dsv4/linear/compressor_wkv", || {
        dsv4_linear(ctx, &compressor.wkv, hidden, &mut kv_raw)
    })?;
    drop(nvtx_compressor_wkv);
    keepalive.keep_hidden(&kv_raw);
    // SAFETY: dsv4_linear writes the full compressor score buffer.
    let mut score_raw = unsafe { HiddenStates::uninit(ctx, width, token_count)? };
    let nvtx_compressor_wgate = crate::nvtx::range("dsv4/linear/compressor_wgate");
    crate::linear_profile::profile(ctx, "dsv4/linear/compressor_wgate", || {
        dsv4_linear(ctx, &compressor.wgate, hidden, &mut score_raw)
    })?;
    drop(nvtx_compressor_wgate);
    keepalive.keep_hidden(&score_raw);

    let rope = &config.rope_parameters;
    // Compressed keys use compress_rope_theta with NO YaRN (original_seq_len = 0).
    let (rope_dim, rope_base) = if apply_rope {
        (config.qk_rope_head_dim, config.compress_rope_theta)
    } else {
        (0, config.compress_rope_theta)
    };
    {
        let (kv_ptr, _kg) = kv_raw.data.device_ptr(&ctx.stream);
        let (score_ptr, _scg) = score_raw.data.device_ptr(&ctx.stream);
        let (ape_ptr, _ag) = compressor.ape.data.device_ptr(&ctx.stream);
        let (norm_ptr, _ng) = compressor.norm.data.device_ptr(&ctx.stream);
        let (pkv_ptr, _pkg) = state.pending_kv.device_ptr_mut(&ctx.stream);
        let (psc_ptr, _psg) = state.pending_score.device_ptr_mut(&ctx.stream);
        let (prkv_ptr, _prkg) = state.prev_overlap_kv.device_ptr_mut(&ctx.stream);
        let (prsc_ptr, _prsg) = state.prev_overlap_score.device_ptr_mut(&ctx.stream);
        let (comp_ptr, _cg) = state.compressed.data.device_ptr_mut(&ctx.stream);
        let has_prev_overlap = i32::from(compressed_base > 0);
        // SAFETY: all buffers valid on ctx.stream; state carries the pending and
        // overlap rows from previous contiguous appends.
        if !dsv4_verify_frozen() {
            unsafe {
                if let Some(start_pos_device) = start_pos_device {
                    let (start_ptr, _spg) = start_pos_device.device_ptr(&ctx.stream);
                    ffi::dsv4_compressor_update_start_pos_ptr_cuda(
                        kv_ptr as *const ffi::Half,
                        score_ptr as *const ffi::Half,
                        ape_ptr as *const ffi::Half,
                        norm_ptr as *const ffi::Half,
                        pkv_ptr as *mut ffi::Half,
                        psc_ptr as *mut ffi::Half,
                        prkv_ptr as *mut ffi::Half,
                        prsc_ptr as *mut ffi::Half,
                        comp_ptr as *mut ffi::Half,
                        token_count as i32,
                        start_ptr as *const i32,
                        head_dim as i32,
                        ratio as i32,
                        width as i32,
                        i32::from(overlap),
                        config.rms_norm_eps,
                        rope_dim as i32,
                        rope_base,
                        rope_original_seq_len,
                        rope.factor,
                        rope.beta_fast,
                        rope.beta_slow,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                } else {
                    ffi::dsv4_compressor_update_cuda(
                        kv_ptr as *const ffi::Half,
                        score_ptr as *const ffi::Half,
                        ape_ptr as *const ffi::Half,
                        norm_ptr as *const ffi::Half,
                        pkv_ptr as *mut ffi::Half,
                        psc_ptr as *mut ffi::Half,
                        prkv_ptr as *mut ffi::Half,
                        prsc_ptr as *mut ffi::Half,
                        comp_ptr as *mut ffi::Half,
                        token_count as i32,
                        start_pos_i32,
                        pending_len_i32,
                        compressed_base_i32,
                        head_dim as i32,
                        ratio as i32,
                        width as i32,
                        i32::from(overlap),
                        has_prev_overlap,
                        config.rms_norm_eps,
                        rope_dim as i32,
                        rope_base,
                        rope_original_seq_len,
                        rope.factor,
                        rope.beta_fast,
                        rope.beta_slow,
                        ctx.stream.cu_stream(),
                    )
                    .result()?;
                }
            }
        }
    }
    // Frozen-KV (P1-1): a frozen verify SKIPS the compressor/indexer CUDA update
    // above, so it must NOT advance `compressed.seq_len` either — otherwise CSA /
    // FlashMLA in the same verify would attend to a compressed/indexer row whose
    // data was never produced, and `csa_select` would advance DSA `packed_rows`
    // off `indexer_rows_after`. Freezing the length keeps `indexer_rows_after ==
    // indexer_rows_before` so the whole compressed+sparse path stays frozen; the
    // accepted-prefix commit re-forward (non-frozen) advances it for real.
    if !dsv4_verify_frozen() {
        state.compressed.seq_len = compressed_rows;
    }
    Ok(())
}

/// CSA top-k block selection: project the index query (`wq_b`) + per-head gating
/// (`weights_proj`), then `dsv4_csa_select_cuda` scores each compressed-key block
/// and writes the top-`index_topk` block ids per token into `[seq * index_topk]`.
#[allow(clippy::too_many_arguments)]
fn csa_select(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    layer_idx: usize,
    indexer: &Dsv4Indexer,
    hidden: &HiddenStates,
    c_q_normed: &HiddenStates,
    keys: &HiddenStates,
    official: Option<&mut Dsv4DsaOfficialState>,
    dsa_shared: Option<&mut Dsv4DsaSharedScratch>,
    pool: &mut Dsv4LayerKvLayout,
    indexer_rows_before: usize,
    indexer_rows_after: usize,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    ratio: usize,
    prefill_scratch: Option<&mut Dsv4PrefillDeepGemmLinearScratch>,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<CudaSlice<i32>> {
    // SAFETY: dsv4_linear writes the full index-query buffer.
    let mut q_i = unsafe { HiddenStates::uninit(ctx, indexer.wq_b.rows, c_q_normed.seq_len)? };
    let nvtx_indexer_wq_b = crate::nvtx::range("dsv4/linear/indexer_wq_b");
    // Prefill index-query (M=token_count) → DeepGEMM, off the scalar fp8_gemv (the #1
    // remaining projection at M=1024). Decode (seq_len==1) / no-cache stays scalar.
    let indexer_wq_b_dg = c_q_normed.seq_len > 1
        && dsv4_prefill_indexer_deepgemm_enabled()
        && indexer.wq_b_deepgemm.is_some()
        && prefill_scratch.is_some();
    if indexer_wq_b_dg {
        let cache = indexer
            .wq_b_deepgemm
            .as_ref()
            .expect("indexer wq_b dg gate checked");
        let scratch = prefill_scratch.expect("indexer wq_b dg gate checked");
        crate::linear_profile::profile(ctx, "dsv4/linear/indexer_wq_b", || {
            prefill_proj_deepgemm(ctx, scratch, cache, c_q_normed, &mut q_i)
        })?;
    } else {
        crate::linear_profile::profile(ctx, "dsv4/linear/indexer_wq_b", || {
            dsv4_linear(ctx, &indexer.wq_b, c_q_normed, &mut q_i)
        })?;
    }
    drop(nvtx_indexer_wq_b);
    keepalive.keep_hidden(&q_i);
    // SAFETY: dsv4_linear writes the full index-weight buffer.
    let mut weights =
        unsafe { HiddenStates::uninit(ctx, indexer.weights_proj.rows, hidden.seq_len)? };
    let nvtx_indexer_weights = crate::nvtx::range("dsv4/linear/indexer_weights");
    crate::linear_profile::profile(ctx, "dsv4/linear/indexer_weights", || {
        dsv4_linear(ctx, &indexer.weights_proj, hidden, &mut weights)
    })?;
    drop(nvtx_indexer_weights);
    keepalive.keep_hidden(&weights);

    ensure!(
        q_i.hidden_dim.is_multiple_of(config.index_head_dim),
        "DSv4 indexer q width {} is not divisible by index_head_dim {}",
        q_i.hidden_dim,
        config.index_head_dim
    );
    let local_index_heads = q_i.hidden_dim / config.index_head_dim;
    ensure!(
        weights.hidden_dim == local_index_heads,
        "DSv4 indexer weights width {} != local index heads {local_index_heads}",
        weights.hidden_dim
    );

    let key_count = if start_pos_device.is_some() {
        if dsv4_verify_frozen() {
            // Frozen-KV (P1-A): the selector computes `available = min(key_count,
            // abs_pos / ratio)`. A frozen verify's `abs_pos` can cross a compression
            // boundary, so capacity-or-`abs_pos/ratio` would expose a compressed row
            // the frozen compressor never produced. Pin to the committed indexer row
            // count — `keys.seq_len` is frozen to that by P1-1, and abs_pos/ratio >=
            // it, so `available` stays at the committed set. (Frozen verify is the
            // spec path, never graph-replayed, so the replay-safety capacity rule
            // below does not apply here.)
            keys.seq_len
        } else {
            // Graph replay must not bake the current compressed-key seq_len. The
            // selector computes `available = min(key_count, abs_pos / ratio)`, so
            // capacity preserves the same causal set while staying replay-safe.
            keys.data.len() / keys.hidden_dim
        }
    } else {
        keys.seq_len
    };
    let score_scale =
        (config.index_head_dim as f32).powf(-0.5) * (config.index_n_heads as f32).powf(-0.5);
    if let Some(official) = official {
        let graph_replay = start_pos_device.is_some()
            && matches!(
                std::env::var("ARLE_DSV4_DECODE_GRAPH").as_deref(),
                Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
            );
        let shared = match (dsa_shared, graph_replay) {
            (Some(shared), _) => Some(shared),
            // Decode-graph replay: csa_select_official would early-return
            // before touching any scratch, so a missing shared handle (the
            // graph capture closures don't carry the adapter) is legal.
            (None, true) => None,
            (None, false) => anyhow::bail!(
                "DSv4 official DSA per-slot state present but shared scratch missing                  outside decode-graph replay"
            ),
        };
        if let Some(shared) = shared {
            if let Some(selected) = csa_select_official(
                ctx,
                config,
                &q_i,
                &weights,
                keys,
                official,
                shared,
                pool,
                indexer_rows_before,
                indexer_rows_after,
                key_count,
                start_pos,
                start_pos_device,
                layer_idx,
                ratio,
                local_index_heads,
                score_scale,
                keepalive,
            )? {
                return Ok(selected);
            }
        }
    }
    let mut selected = ctx
        .stream
        .alloc_zeros::<i32>(hidden.seq_len * config.index_topk)
        .map_err(|e| anyhow::anyhow!("DSv4 CSA selected alloc failed: {e}"))?;
    {
        let (q_ptr, _qg) = q_i.data.device_ptr(&ctx.stream);
        let (w_ptr, _wg) = weights.data.device_ptr(&ctx.stream);
        let (keys_ptr, _kg) = keys.data.device_ptr(&ctx.stream);
        let (sel_ptr, _sg) = selected.device_ptr_mut(&ctx.stream);
        // SAFETY: all buffers valid on ctx.stream; selected sized seq*index_topk.
        unsafe {
            if let Some(start_pos_device) = start_pos_device {
                let (start_ptr, _spg) = start_pos_device.device_ptr(&ctx.stream);
                ffi::dsv4_csa_select_start_pos_ptr_cuda(
                    q_ptr as *const ffi::Half,
                    w_ptr as *const ffi::Half,
                    keys_ptr as *const ffi::Half,
                    sel_ptr as *mut i32,
                    hidden.seq_len as i32,
                    q_i.hidden_dim as i32,
                    local_index_heads as i32,
                    config.index_head_dim as i32,
                    key_count as i32,
                    ratio as i32,
                    config.index_topk as i32,
                    score_scale,
                    start_ptr as *const i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            } else {
                ffi::dsv4_csa_select_cuda(
                    q_ptr as *const ffi::Half,
                    w_ptr as *const ffi::Half,
                    keys_ptr as *const ffi::Half,
                    sel_ptr as *mut i32,
                    hidden.seq_len as i32,
                    q_i.hidden_dim as i32,
                    local_index_heads as i32,
                    config.index_head_dim as i32,
                    key_count as i32,
                    ratio as i32,
                    config.index_topk as i32,
                    score_scale,
                    start_pos as i32,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
    }
    maybe_probe_deepgemm_dsa_logits(
        ctx,
        config,
        layer_idx,
        &q_i,
        &weights,
        keys,
        hidden.seq_len,
        local_index_heads,
        key_count,
        ratio,
        start_pos,
        start_pos_device,
        score_scale,
    )?;
    if std::env::var("ARLE_DSV4_CSA_DUMP").as_deref() == Ok("1")
        && std::env::var("INFER_TP_RANK").as_deref() == Ok("0")
    {
        let row_idx = hidden.seq_len.saturating_sub(1);
        let available = std::cmp::min(key_count, (start_pos + row_idx) / ratio);
        match ctx.stream.clone_dtoh(&selected) {
            Ok(host) => {
                let row_start = row_idx * config.index_topk;
                let selected_head: Vec<i32> =
                    host.iter().skip(row_start).copied().take(32).collect();
                let invalid_selected = host
                    .iter()
                    .skip(row_start)
                    .take(config.index_topk)
                    .filter(|&&v| v < 0 || v >= available as i32)
                    .count();
                eprintln!(
                    "[dsv4-csa-dump] layer={layer_idx} start_pos={start_pos} seq_len={} row={row_idx} ratio={ratio} available={available} topk={} invalid_selected={invalid_selected} selected_head={selected_head:?}",
                    hidden.seq_len, config.index_topk,
                );
            }
            Err(e) => {
                eprintln!("[dsv4-csa-dump] layer={layer_idx} start_pos={start_pos} dtoh_err={e}")
            }
        }
    }
    Ok(selected)
}

#[allow(clippy::too_many_arguments)]
fn csa_select_official(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    q_i: &HiddenStates,
    weights: &HiddenStates,
    keys: &HiddenStates,
    official: &mut Dsv4DsaOfficialState,
    shared: &mut Dsv4DsaSharedScratch,
    pool: &mut Dsv4LayerKvLayout,
    indexer_rows_before: usize,
    indexer_rows_after: usize,
    key_count: usize,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    layer_idx: usize,
    ratio: usize,
    local_index_heads: usize,
    score_scale: f32,
    keepalive: &mut Dsv4ForwardKeepalive,
) -> Result<Option<CudaSlice<i32>>> {
    if start_pos_device.is_some()
        && matches!(
            std::env::var("ARLE_DSV4_DECODE_GRAPH").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
        )
    {
        return Ok(None);
    }
    ensure!(
        local_index_heads == shared.num_heads && config.index_head_dim == shared.head_dim,
        "DSv4 official DSA shape mismatch local_heads={} official_heads={} dim={} official_dim={}",
        local_index_heads,
        shared.num_heads,
        config.index_head_dim,
        shared.head_dim
    );
    ensure!(
        q_i.seq_len <= shared.max_tokens,
        "DSv4 official DSA token_count {} exceeds scratch max {}",
        q_i.seq_len,
        shared.max_tokens
    );
    ensure!(
        key_count <= shared.compressed_capacity
            && indexer_rows_after <= shared.compressed_capacity
            && indexer_rows_before <= indexer_rows_after,
        "DSv4 official DSA key rows before={} after={} key_count={} capacity={}",
        indexer_rows_before,
        indexer_rows_after,
        key_count,
        shared.compressed_capacity
    );
    ensure!(
        start_pos + q_i.seq_len <= shared.max_tokens,
        "DSv4 official DSA positions {}..{} exceed freqs_cis max {}",
        start_pos,
        start_pos + q_i.seq_len,
        shared.max_tokens
    );

    ensure!(
        official.rotated_keys.len() == shared.compressed_capacity * shared.head_dim,
        "DSv4 official DSA rotated_keys len {} mismatches shared scratch capacity {}x{}",
        official.rotated_keys.len(),
        shared.compressed_capacity,
        shared.head_dim
    );

    let newly_packed = indexer_rows_after.saturating_sub(official.packed_rows);
    if newly_packed > 0 {
        ensure!(
            official.packed_rows <= indexer_rows_before,
            "DSv4 official DSA packed rows {} ahead of indexer rows before {}",
            official.packed_rows,
            indexer_rows_before
        );
        let src_offset = official.packed_rows * config.index_head_dim;
        let src = keys
            .data
            .slice(src_offset..src_offset + newly_packed * config.index_head_dim);
        {
            let mut rotated = official
                .rotated_keys
                .slice_mut(src_offset..src_offset + newly_packed * config.index_head_dim);
            let (src_ptr, _sg) = src.device_ptr(&ctx.stream);
            let (rot_ptr, _rg) = rotated.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::dsv4_dsa_hadamard128_bf16_cuda(
                    src_ptr as *const ffi::Half,
                    rot_ptr as *mut ffi::Half,
                    i32::try_from(newly_packed)?,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
        let locs = shared
            .cache_locs
            .slice(official.packed_rows..official.packed_rows + newly_packed);
        {
            let rotated = official
                .rotated_keys
                .slice(src_offset..src_offset + newly_packed * config.index_head_dim);
            let (rot_store_ptr, _rsg) = rotated.device_ptr(&ctx.stream);
            let cache_range = pool.dsa_slot_range(official.slot_idx)?;
            let cache_pool = pool
                .dsa_key_cache
                .as_mut()
                .ok_or_else(|| anyhow!("DSv4 official DSA shared key-cache missing"))?;
            ensure!(
                cache_range.end <= cache_pool.len() && cache_range.len() == official.key_cache_len,
                "DSv4 official DSA shared key-cache range {:?} invalid pool_len={} slot_len={}",
                cache_range,
                cache_pool.len(),
                official.key_cache_len
            );
            let mut cache_view = cache_pool.slice_mut(cache_range);
            let (cache_ptr_u8, _cg) = cache_view.device_ptr_mut(&ctx.stream);
            let (locs_ptr, _lg) = locs.device_ptr(&ctx.stream);
            unsafe {
                ffi::dsv4_dsa_fused_store_index_k_cache_cuda(
                    rot_store_ptr as *const ffi::Half,
                    cache_ptr_u8 as *mut u8,
                    locs_ptr as *const i64,
                    i32::try_from(newly_packed)?,
                    64,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }
        official.packed_rows = indexer_rows_after;
    }

    let token_count = q_i.seq_len;
    // `raw_indices` (topk output) is sized by `query_chunk`, not `max_tokens`. The
    // scheduler guarantees a single forward never passes more than
    // `chunked_prefill_size <= DSV4_PREFILL_QUERY_CHUNK == query_chunk` query tokens,
    // so the per-tile `raw_indices[t0..t0+tlen]` writes and the DUMP read
    // `raw_indices[0..seq_len*topk]` both stay within the chunk-sized buffer. Fail loud
    // rather than write past it (e.g. the one-shot long-context `dsv4_parity` example,
    // which is not the chunked-prefill path).
    ensure!(
        token_count <= shared.query_chunk,
        "DSv4 official DSA token_count {} exceeds prefill query chunk {} (raw_indices \
         scratch is chunk-sized; chunked prefill must keep seq_len <= \
         chunked_prefill_size <= DSV4_PREFILL_QUERY_CHUNK)",
        token_count,
        shared.query_chunk
    );
    let dump_csa = std::env::var("ARLE_DSV4_CSA_DUMP").as_deref() == Ok("1")
        && std::env::var("INFER_TP_RANK").as_deref() == Ok("0");
    let context_lens_h = if dump_csa {
        Some(
            (0..token_count)
                .map(|token| {
                    let abs_pos = start_pos + token;
                    i32::try_from(std::cmp::min(key_count, abs_pos / ratio))
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        None
    };

    // Full-N output, allocated once. Each tile writes its disjoint [t0..t0+tlen) slice.
    let mut selected = ctx
        .stream
        .alloc_zeros::<i32>(token_count * config.index_topk)
        .map_err(|e| anyhow!("DSv4 official DSA selected alloc failed: {e}"))?;

    // Query-axis tiling — the ONLY compute path. The logits scratch is bounded by
    // `tile × logits_stride`; long prompts loop in tiles and never materialize full-N
    // logits. When token_count <= tile this loop runs a single iteration with t0=0
    // (tlen=token_count), behavior-IDENTICAL to the pre-tiling code.
    //
    // Mutated-buffer enumeration (per-tile correctness):
    //   shared.logits [tile × stride]: overwritten each sub-chunk before topk reads it — safe.
    //   shared.q_fp8/weights/context_lens/positions [tile-sized]: overwritten each
    //     sub-chunk before use — safe.
    //   shared.page_table_identity [tile × num_pages]: identity, read-only, same for
    //     every sub-chunk — safe.
    //   selected / shared.raw_indices [full N × topk]: each sub-chunk writes its
    //     disjoint [t0..t0+tlen) slice — full output assembled, no overlap.
    //   key-packing buffers (rotated_keys, key cache, cache_locs, packed_rows):
    //     untouched by this change (handled in the query-independent block above).
    let tile = shared.query_tile;
    // q_i.data / weights.data are flat [seq_len * per_token_width]; derive per-token
    // strides so each tile slices the right sub-range of the (untiled) inputs.
    let q_stride = q_i.data.len() / token_count;
    ensure!(
        q_stride * token_count == q_i.data.len(),
        "DSv4 official DSA q input len {} not divisible by token_count {}",
        q_i.data.len(),
        token_count
    );
    let w_stride = weights.data.len() / token_count;
    ensure!(
        w_stride * token_count == weights.data.len(),
        "DSv4 official DSA weights input len {} not divisible by token_count {}",
        weights.data.len(),
        token_count
    );

    let mut t0 = 0usize;
    while t0 < token_count {
        let tlen = (token_count - t0).min(tile);

        // (a) per-tile context_lens / positions. Decode graph/eager decode carry
        // `start_pos` on device; fill tile metadata on GPU to avoid two tiny
        // H2D copies per CSA layer.
        {
            let mut context_lens = shared.context_lens.slice_mut(0..tlen);
            let mut positions = shared.positions.slice_mut(0..tlen);
            if let Some(start_pos_device) = start_pos_device {
                let (lens_ptr, _lg) = context_lens.device_ptr_mut(&ctx.stream);
                let (positions_ptr, _pg) = positions.device_ptr_mut(&ctx.stream);
                let (start_ptr, _sg) = start_pos_device.device_ptr(&ctx.stream);
                unsafe {
                    ffi::dsv4_dsa_fill_context_lens_positions_start_pos_cuda(
                        lens_ptr as *mut i32,
                        positions_ptr as *mut i32,
                        start_ptr as *const i32,
                        i32::try_from(t0)?,
                        i32::try_from(tlen)?,
                        i32::try_from(key_count)?,
                        i32::try_from(ratio)?,
                        ctx.stream.cu_stream(),
                    )
                    .result()
                    .map_err(|e| anyhow!("DSv4 official DSA GPU metadata fill failed: {e}"))?;
                }
            } else {
                let context_lens_tile: Vec<i32> = (0..tlen)
                    .map(|i| {
                        let abs_pos = start_pos + t0 + i;
                        i32::try_from(std::cmp::min(key_count, abs_pos / ratio))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let positions_tile: Vec<i32> = (0..tlen)
                    .map(|i| i32::try_from(start_pos + t0 + i))
                    .collect::<Result<Vec<_>, _>>()?;
                ctx.stream
                    .memcpy_htod(&context_lens_tile, &mut context_lens)
                    .map_err(|e| anyhow!("DSv4 official DSA context_lens H2D failed: {e}"))?;
                ctx.stream
                    .memcpy_htod(&positions_tile, &mut positions)
                    .map_err(|e| anyhow!("DSv4 official DSA positions H2D failed: {e}"))?;
            }
        }

        // (c) fused Q indexer rope+hadamard+quant over the tile's input sub-range.
        {
            let q_in = q_i.data.slice(t0 * q_stride..(t0 + tlen) * q_stride);
            let (q_ptr, _qg) = q_in.device_ptr(&ctx.stream);
            let (q_fp8_ptr, _qfg) = shared.q_fp8.device_ptr_mut(&ctx.stream);
            let w_in = weights.data.slice(t0 * w_stride..(t0 + tlen) * w_stride);
            let (w_ptr, _wg) = w_in.device_ptr(&ctx.stream);
            let (weights_out_ptr, _wog) = shared.weights.device_ptr_mut(&ctx.stream);
            let (freqs_ptr, _fg) = shared.freqs_cis.device_ptr(&ctx.stream);
            let positions = shared.positions.slice(0..tlen);
            let (positions_ptr, _pg) = positions.device_ptr(&ctx.stream);
            unsafe {
                ffi::dsv4_dsa_fused_q_indexer_rope_hadamard_quant_cuda(
                    q_ptr as *const ffi::Half,
                    q_fp8_ptr as *mut u8,
                    w_ptr as *const ffi::Half,
                    weights_out_ptr as *mut f32,
                    score_scale,
                    freqs_ptr as *const f32,
                    positions_ptr as *const i32,
                    i32::try_from(tlen)?,
                    i32::try_from(local_index_heads)?,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }

        // (d) paged MQA logits scheduling metadata for the tile.
        unsafe {
            cuda_moe::dsv4_deepgemm_paged_mqa_logits_metadata(
                cache_ptr(&shared.context_lens, ctx),
                cache_ptr(&shared.sched_meta, ctx),
                tlen,
                1,
                64,
                shared.num_sms,
                ctx.stream.cu_stream(),
            )
            .map_err(|e| anyhow!("DSv4 official DSA metadata failed: {e}"))?;
        }

        // (e) fused paged FP8 MQA logits → shared.logits (tlen rows).
        {
            let cache_range = pool.dsa_slot_range(official.slot_idx)?;
            let cache_pool = pool
                .dsa_key_cache
                .as_ref()
                .ok_or_else(|| anyhow!("DSv4 official DSA shared key-cache missing"))?;
            ensure!(
                cache_range.end <= cache_pool.len() && cache_range.len() == official.key_cache_len,
                "DSv4 official DSA shared key-cache range {:?} invalid pool_len={} slot_len={}",
                cache_range,
                cache_pool.len(),
                official.key_cache_len
            );
            let cache_view = cache_pool.slice(cache_range);
            let (q_ptr, _qg) = shared.q_fp8.device_ptr(&ctx.stream);
            let (cache_ptr_u8, _kg) = cache_view.device_ptr(&ctx.stream);
            let (weights_ptr, _wg) = shared.weights.device_ptr(&ctx.stream);
            let (lens_ptr, _lg) = shared.context_lens.device_ptr(&ctx.stream);
            let (page_ptr, _pg) = shared.page_table_identity.device_ptr(&ctx.stream);
            let (sched_ptr, _sg) = shared.sched_meta.device_ptr(&ctx.stream);
            let (logits_ptr, _og) = shared.logits.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::dsv4_deepgemm_fp8_paged_mqa_logits_fused_cache_cuda(
                    q_ptr as *const u8,
                    cache_ptr_u8 as *const u8,
                    weights_ptr as *const f32,
                    lens_ptr as *const i32,
                    page_ptr as *const i32,
                    sched_ptr as *const i32,
                    logits_ptr as *mut f32,
                    i32::try_from(tlen)?,
                    1,
                    i32::try_from(local_index_heads)?,
                    i32::try_from(config.index_head_dim)?,
                    i32::try_from(shared.num_pages)?,
                    64,
                    i32::try_from(shared.num_pages * 64)?,
                    i32::try_from(shared.logits_stride)?,
                    i32::try_from(shared.num_pages)?,
                    i32::try_from(64 * (config.index_head_dim + std::mem::size_of::<f32>()))?,
                    i32::try_from(shared.num_sms)?,
                    ctx.stream.cu_stream(),
                )
                .result()
                .map_err(|e| anyhow!("DSv4 official DSA paged logits failed: {e}"))?;
            }
        }

        // (f) topk transform: read shared.logits (base), write the tile's disjoint
        //     output slices of `selected` and `shared.raw_indices`.
        {
            let context_lens = shared.context_lens.slice(0..tlen);
            let (logits_ptr, _lg) = shared.logits.device_ptr(&ctx.stream);
            let (lens_ptr, _csg) = context_lens.device_ptr(&ctx.stream);
            let (page_ptr, _ptg) = shared.page_table_identity.device_ptr(&ctx.stream);
            let mut sel =
                selected.slice_mut(t0 * config.index_topk..(t0 + tlen) * config.index_topk);
            let (sel_ptr, _seg) = sel.device_ptr_mut(&ctx.stream);
            let mut raw = shared
                .raw_indices
                .slice_mut(t0 * config.index_topk..(t0 + tlen) * config.index_topk);
            let (raw_ptr, _rig) = raw.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::dsv4_deepseek_v4_topk_transform_512_cuda(
                    logits_ptr as *const f32,
                    lens_ptr as *const i32,
                    page_ptr as *const i32,
                    sel_ptr as *mut i32,
                    raw_ptr as *mut i32,
                    i64::try_from(shared.logits_stride)?,
                    i64::try_from(shared.num_pages)?,
                    i64::try_from(config.index_topk)?,
                    i32::try_from(tlen)?,
                    i32::try_from(config.index_topk)?,
                    64,
                    ctx.stream.cu_stream(),
                )
                .result()?;
            }
        }

        t0 += tlen;
    }
    keepalive.keep_u8(&shared.q_fp8);
    keepalive.keep_f32(&shared.weights);
    if dump_csa {
        let row_idx = q_i.seq_len.saturating_sub(1);
        let available = context_lens_h
            .as_ref()
            .and_then(|lens| lens.get(row_idx).copied())
            .unwrap_or_default();
        let selected_host = ctx.stream.clone_dtoh(&selected);
        let raw_host = ctx
            .stream
            .clone_dtoh(&shared.raw_indices.slice(0..q_i.seq_len * config.index_topk));
        match (selected_host, raw_host) {
            (Ok(selected_host), Ok(raw_host)) => {
                let invalid_selected = selected_host
                    .iter()
                    .skip(row_idx * config.index_topk)
                    .take(config.index_topk)
                    .filter(|&&v| v < 0 || v >= available)
                    .count();
                let invalid_raw = raw_host
                    .iter()
                    .skip(row_idx * config.index_topk)
                    .take(config.index_topk)
                    .filter(|&&v| v < 0 || v >= available)
                    .count();
                let selected_head: Vec<i32> = selected_host
                    .iter()
                    .skip(row_idx * config.index_topk)
                    .copied()
                    .take(32)
                    .collect();
                let raw_head: Vec<i32> = raw_host
                    .iter()
                    .skip(row_idx * config.index_topk)
                    .copied()
                    .take(32)
                    .collect();
                eprintln!(
                    "[dsv4-csa-dump-official] layer={layer_idx} start_pos={start_pos} seq_len={} row={row_idx} ratio={ratio} available={available} topk={} invalid_selected={invalid_selected} invalid_raw={invalid_raw} selected_head={selected_head:?} raw_head={raw_head:?}",
                    q_i.seq_len, config.index_topk,
                );
            }
            (Err(e), _) | (_, Err(e)) => {
                eprintln!(
                    "[dsv4-csa-dump-official] layer={layer_idx} start_pos={start_pos} dtoh_err={e}"
                );
            }
        }
    }
    Ok(Some(selected))
}

#[allow(clippy::too_many_arguments)]
fn maybe_probe_deepgemm_dsa_logits(
    ctx: &DeviceContext,
    config: &DeepSeekV4Config,
    layer_idx: usize,
    q_i: &HiddenStates,
    weights: &HiddenStates,
    keys: &HiddenStates,
    seq_len: usize,
    local_index_heads: usize,
    key_count: usize,
    ratio: usize,
    start_pos: usize,
    start_pos_device: Option<&CudaSlice<i32>>,
    score_scale: f32,
) -> Result<()> {
    if std::env::var("ARLE_DSV4_DSA_LOGITS_PROBE").as_deref() != Ok("1")
        || std::env::var("INFER_TP_RANK").as_deref() != Ok("0")
        || seq_len != 1
    {
        return Ok(());
    }
    static SEEN: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if !seen.lock().unwrap().insert(layer_idx) {
        return Ok(());
    }
    let effective_start_pos = if let Some(start_pos_device) = start_pos_device {
        let host = ctx
            .stream
            .clone_dtoh(start_pos_device)
            .map_err(|e| anyhow!("DSv4 DSA logits probe start_pos D2H failed: {e}"))?;
        usize::try_from(host.first().copied().unwrap_or_default())
            .map_err(|_| anyhow!("DSv4 DSA logits probe negative start_pos"))?
    } else {
        start_pos
    };
    ensure!(
        config.index_head_dim == 128,
        "DeepGEMM paged-MQA logits probe expects index_head_dim=128, got {}",
        config.index_head_dim
    );
    ensure!(
        local_index_heads == 32 || local_index_heads == 64,
        "DeepGEMM paged-MQA logits probe expects 32/64 heads, got {local_index_heads}"
    );
    let available = std::cmp::min(key_count, effective_start_pos / ratio);
    if available == 0 {
        eprintln!("[dsv4-dsa-logits-probe] layer={layer_idx} skipped reason=no_available_keys");
        return Ok(());
    }
    let sample_limit = std::env::var("ARLE_DSV4_DSA_LOGITS_PROBE_LIMIT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(16)
        .min(available);
    let num_sms = std::env::var("ARLE_DSV4_DSA_LOGITS_PROBE_SMS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(78);
    let block_kv = 64usize;
    let padded_keys = available.div_ceil(block_kv) * block_kv;
    let logits_stride = available.div_ceil(256) * 256;
    let q_scale_stride_m = local_index_heads.div_ceil(4) * 4;
    let key_scale_stride_m = padded_keys;
    let scale_cols = config.index_head_dim.div_ceil(128);

    let q_fp8 = ctx
        .stream
        .alloc_zeros::<u8>(local_index_heads * config.index_head_dim)
        .map_err(|e| anyhow!("DSv4 DSA logits probe q fp8 alloc failed: {e}"))?;
    let q_scales = ctx
        .stream
        .alloc_zeros::<f32>(q_scale_stride_m * scale_cols)
        .map_err(|e| anyhow!("DSv4 DSA logits probe q scales alloc failed: {e}"))?;
    let key_fp8 = ctx
        .stream
        .alloc_zeros::<u8>(padded_keys * config.index_head_dim)
        .map_err(|e| anyhow!("DSv4 DSA logits probe key fp8 alloc failed: {e}"))?;
    let key_scales = ctx
        .stream
        .alloc_zeros::<f32>(padded_keys * scale_cols)
        .map_err(|e| anyhow!("DSv4 DSA logits probe key scales alloc failed: {e}"))?;
    let active_experts = ctx
        .stream
        .clone_htod(&[0_i32])
        .map_err(|e| anyhow!("DSv4 DSA logits probe active_experts H2D failed: {e}"))?;
    let active_offsets = ctx
        .stream
        .clone_htod(&[0_i32])
        .map_err(|e| anyhow!("DSv4 DSA logits probe active_offsets H2D failed: {e}"))?;
    let mut active_counts = ctx
        .stream
        .clone_htod(&[local_index_heads as i32])
        .map_err(|e| anyhow!("DSv4 DSA logits probe active_counts H2D failed: {e}"))?;
    let stream = ctx.stream.cu_stream();
    unsafe {
        cuda_moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
            cache_ptr(&q_i.data, ctx),
            cache_ptr(&q_fp8, ctx),
            cache_ptr(&q_scales, ctx),
            cache_ptr(&active_experts, ctx),
            cache_ptr(&active_offsets, ctx),
            cache_ptr(&active_counts, ctx),
            1,
            local_index_heads,
            config.index_head_dim,
            q_scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 DSA logits probe q quantize failed: {e}"))?;
    }
    ctx.stream
        .memcpy_htod(&[available as i32], &mut active_counts)
        .map_err(|e| anyhow!("DSv4 DSA logits probe key active_counts H2D failed: {e}"))?;
    unsafe {
        cuda_moe::dsv4_deepgemm_pack_quantize_bf16_to_fp8(
            cache_ptr(&keys.data, ctx),
            cache_ptr(&key_fp8, ctx),
            cache_ptr(&key_scales, ctx),
            cache_ptr(&active_experts, ctx),
            cache_ptr(&active_offsets, ctx),
            cache_ptr(&active_counts, ctx),
            1,
            padded_keys,
            config.index_head_dim,
            key_scale_stride_m,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 DSA logits probe key quantize failed: {e}"))?;
    }

    let q_scales_host = ctx
        .stream
        .clone_dtoh(&q_scales)
        .map_err(|e| anyhow!("DSv4 DSA logits probe q scales D2H failed: {e}"))?;
    let weights_host = ctx
        .stream
        .clone_dtoh(&weights.data)
        .map_err(|e| anyhow!("DSv4 DSA logits probe weights D2H failed: {e}"))?;
    let fused_weights_host: Vec<f32> = (0..local_index_heads)
        .map(|head| weights_host[head].to_f32() * q_scales_host[head] * score_scale)
        .collect();
    let fused_weights = ctx
        .stream
        .clone_htod(&fused_weights_host)
        .map_err(|e| anyhow!("DSv4 DSA logits probe weights H2D failed: {e}"))?;
    let context_lens = ctx
        .stream
        .clone_htod(&[available as i32])
        .map_err(|e| anyhow!("DSv4 DSA logits probe context_lens H2D failed: {e}"))?;
    let block_table_host: Vec<i32> = (0..padded_keys / block_kv).map(|v| v as i32).collect();
    let block_table = ctx
        .stream
        .clone_htod(&block_table_host)
        .map_err(|e| anyhow!("DSv4 DSA logits probe block_table H2D failed: {e}"))?;
    let sched_meta = ctx
        .stream
        .alloc_zeros::<i32>((num_sms + 1) * 2)
        .map_err(|e| anyhow!("DSv4 DSA logits probe sched_meta alloc failed: {e}"))?;
    let logits = ctx
        .stream
        .alloc_zeros::<f32>(logits_stride)
        .map_err(|e| anyhow!("DSv4 DSA logits probe logits alloc failed: {e}"))?;
    unsafe {
        cuda_moe::dsv4_deepgemm_paged_mqa_logits_metadata(
            cache_ptr(&context_lens, ctx),
            cache_ptr(&sched_meta, ctx),
            1,
            1,
            block_kv,
            num_sms,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 DSA logits probe metadata failed: {e}"))?;
        cuda_moe::dsv4_deepgemm_fp8_paged_mqa_logits(
            cache_ptr(&q_fp8, ctx),
            cache_ptr(&key_fp8, ctx),
            cache_ptr(&key_scales, ctx),
            cache_ptr(&fused_weights, ctx),
            cache_ptr(&context_lens, ctx),
            cache_ptr(&block_table, ctx),
            cache_ptr(&sched_meta, ctx),
            cache_ptr(&logits, ctx),
            1,
            1,
            local_index_heads,
            config.index_head_dim,
            padded_keys / block_kv,
            block_kv,
            available,
            logits_stride,
            padded_keys / block_kv,
            block_kv * (config.index_head_dim + std::mem::size_of::<f32>()),
            num_sms,
            stream,
        )
        .map_err(|e| anyhow!("DSv4 DSA logits probe paged logits failed: {e}"))?;
    }
    let logits_host = ctx
        .stream
        .clone_dtoh(&logits)
        .map_err(|e| anyhow!("DSv4 DSA logits probe logits D2H failed: {e}"))?;
    let q_host = ctx
        .stream
        .clone_dtoh(&q_i.data)
        .map_err(|e| anyhow!("DSv4 DSA logits probe q D2H failed: {e}"))?;
    let keys_host = ctx
        .stream
        .clone_dtoh(&keys.data)
        .map_err(|e| anyhow!("DSv4 DSA logits probe keys D2H failed: {e}"))?;

    let mut max_abs = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut first = Vec::with_capacity(sample_limit);
    for (block, &official) in logits_host.iter().enumerate().take(sample_limit) {
        let mut reference = 0.0f32;
        for (head, weight) in weights_host.iter().enumerate().take(local_index_heads) {
            let q_base = head * config.index_head_dim;
            let key_base = block * config.index_head_dim;
            let mut dot = 0.0f32;
            for col in 0..config.index_head_dim {
                dot += q_host[q_base + col].to_f32() * keys_host[key_base + col].to_f32();
            }
            reference += weight.to_f32() * score_scale * dot.max(0.0);
        }
        let diff = (official - reference).abs();
        max_abs = max_abs.max(diff);
        sum_sq += diff * diff;
        if block < 4 {
            first.push(format!(
                "{block}:dg={official:.5}:ref={reference:.5}:diff={diff:.5}"
            ));
        }
    }
    let rms = (sum_sq / sample_limit as f32).sqrt();
    eprintln!(
        "[dsv4-dsa-logits-probe] layer={layer_idx} available={available} sample={sample_limit} heads={local_index_heads} max_abs={max_abs:.6} rms={rms:.6} first={}",
        first.join(",")
    );
    Ok(())
}
