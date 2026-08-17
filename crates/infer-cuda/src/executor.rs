//! Real CUDA executor: the engine-facing step driver and sampling tail.
//!
//! Wraps the loaded [`CudaModel`] + device [`PagedKVPool`], validates the
//! single-row plan, mirrors host→device page allocation, runs the forward, and
//! samples the next token (`sample_cuda_token`: greedy argmax / host sampling).

use std::collections::VecDeque;
use std::path::Path;
use std::time::Instant;

use anyhow::{Result, ensure};
use cuda_kernels::KVFormat;
use cuda_kernels::prelude::{DeviceContext, DeviceVec, PagedKVPool};
use cuda_kernels::tensor::{CudaPipelineFence, CudaPipelineFenceStatus, CudaPipelineStreamKind};
use infer_plan::{DecodeRow, ForwardPlan, SamplingParams, SlotToken, StepOutput};
use infer_seam::{
    KvBatchDescriptor, KvBatchRowKind, KvPool, PrefixBlock, pages_only_reusable_prefix_blocks,
};
use log::{info, warn};

use crate::attention::ModelKvAdapter;
use crate::decode_graph::DecodeGraphContext;
use crate::decode_graph_key::{DECODE_GRAPH_BATCH, DecodeGraphKey};
use crate::graph::GraphBucket;
use crate::loader::PageMeta;
use crate::model::CudaModel;
use crate::ops::argmax;

#[cfg(feature = "cuda")]
#[path = "executor/dsv4.rs"]
mod dsv4;
#[path = "executor/qwen.rs"]
mod qwen;
#[path = "executor/qwen35.rs"]
mod qwen35;
#[path = "executor/spec_decode.rs"]
mod spec_decode;

use dsv4::Dsv4CudaExecutor;
use qwen::QwenCudaExecutor;
use qwen35::Qwen35CudaExecutor;

pub(crate) const SUPPORTED_PAGE_SIZE: usize = 16;

/// Minimum KV length (tokens) before a decode step head-shards across the
/// attn_cp group (B2, T3.1). Below this the cp all-reduce costs more than the
/// halved qkv/KV/GDN traffic saves — keeps short decode a wash.
pub(crate) const CP_DECODE_MIN_KV_TOKENS: usize = 8192;

/// Recurrent-sidecar snapshot stride in KV pages (8192 tokens), bounding how
/// much a cross-conversation prefix hit must re-prefill. Every snapshot retains
/// a full state copy (~150 MB at 27B) until publish, so halving the stride
/// costs TTFT linearly: 2048 spent 3.13 s of a 33K prefill's 25.3 s.
pub(crate) const SIDECAR_SNAPSHOT_STRIDE_PAGES: usize = 512;

/// Flatten a `(session, block)` [`infer_seam::TierBlockKey`] into the
/// `KvTierStore`'s opaque `u64` key namespace. The prefix tier already keys
/// the same store by sequentially-assigned `u64`s (`next_tier_key`), so the
/// write-through namespace must never collide with it: we reserve the high bit
/// for write-through keys and pack `session` (low 31 bits) and `block` (low 32
/// bits) below it. Two distinct sessions therefore never alias (tenant
/// isolation, the "session A never prefetches into session B" gate), and no
/// write-through key can equal a prefix-tier key (which the engine assigns from 0
/// upward, never setting the high bit before wrapping past 2^63).
#[cfg(feature = "cuda")]
#[must_use]
pub(crate) fn tier_block_u64(session: u64, block: u64) -> u64 {
    const WRITETHROUGH_BIT: u64 = 1 << 63;
    WRITETHROUGH_BIT | ((session & 0x7FFF_FFFF) << 32) | (block & 0xFFFF_FFFF)
}
/// Seq-len budget the captured decode graph's fixed `kv_indices` is sized to;
/// pages beyond it fall back to eager rather than replay a stale graph.
pub(crate) const DECODE_GRAPH_MAX_SEQ_LEN: usize = 32_768;

/// Decode-graph gate. Set once at load from the `enable_cuda_graph` load flag
/// (CLI `--cuda-graph`/`--no-cuda-graph`, default on) via
/// [`set_decode_graph_default`]. Single-GPU Qwen dense only — `warmup` still
/// hard-disables the graph under TP (NCCL not graph-capturable) and MoE (host
/// routing per step).
static DECODE_GRAPH_DEFAULT_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Set the load-time decode-graph default. Wired from
/// `LoadedInferenceEngine::load(.., enable_cuda_graph)` so the CLI
/// `--cuda-graph`/`--no-cuda-graph` flag controls the graph — single set at
/// load, read once at `warmup`.
pub fn set_decode_graph_default(enabled: bool) {
    DECODE_GRAPH_DEFAULT_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Captured B=1 decode graph enabled? CLI-driven (`--cuda-graph`/
/// `--no-cuda-graph`, default on). The eager path stays the correctness floor;
/// `warmup` still gates TP and MoE off regardless of this.
pub(crate) fn decode_graph_enabled() -> bool {
    DECODE_GRAPH_DEFAULT_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

fn cuda_startup_profile_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("ARLE_CUDA_STARTUP_PROFILE").is_some())
}

pub(crate) fn cuda_startup_log(phase: &str, start: Instant, extra: std::fmt::Arguments<'_>) {
    if cuda_startup_profile_enabled() {
        info!(
            target: "infer_cuda::startup",
            "cuda_startup phase={phase} elapsed_ms={:.1} {extra}",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// CUDA KV-cache storage dtype, resolved from the backend-neutral
/// [`infer_seam::KvCacheDtype`] request against the CUDA support matrix at engine
/// construction. Mirrors `infer_metal::MetalKvCacheDtype`: the seam carries the
/// *request*, each backend resolves it against what it can actually honor and
/// fails loud rather than silently downgrading.
///
/// The CUDA path resolves BF16 (default) plus the paged quant-KV modes INT8 and
/// FP8 E4M3 (#68 T3) on the dense-Qwen3 paged pool: KIVI per-channel K scales +
/// per-(token, head) V scales, fused-dequant decode attention. `Tq4` stays a
/// loud explicit deferral until a paged-prefill kernel path exists for
/// TurboQuant — the enum trails the real, validated paths, it does not
/// pre-declare unimplemented ones.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CudaKvCacheDtype {
    /// Native BF16 per-layer caches (the default CUDA KV dtype).
    #[default]
    Bf16,
    /// Paged INT8 quant KV (KIVI per-channel K + per-token V scales, #68 T3).
    Int8,
    /// Paged FP8 E4M3 quant KV (KIVI per-channel K + per-token V scales, #68 T3).
    Fp8,
}

impl CudaKvCacheDtype {
    /// Stable lowercase label for logs and gate report lines.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Int8 => "int8",
            Self::Fp8 => "fp8",
        }
    }

    /// Resolve a backend-neutral requested dtype against the CUDA support matrix.
    /// `Auto`/`Bf16` → BF16; `Int8`/`Fp8` → the paged quant modes (#68 T3);
    /// `Tq4` fails loud with an explicit-deferral message rather than silently
    /// falling back to BF16.
    pub fn resolve(requested: infer_seam::KvCacheDtype) -> Result<Self> {
        use infer_seam::KvCacheDtype;
        match requested {
            KvCacheDtype::Auto | KvCacheDtype::Bf16 => Ok(Self::Bf16),
            KvCacheDtype::Int8 => Ok(Self::Int8),
            KvCacheDtype::Fp8 => Ok(Self::Fp8),
            KvCacheDtype::Tq4 => anyhow::bail!(
                "CUDA KV cache dtype tq4 is deferred: TurboQuant pools use \
                 page_size=1 while the TileLang paged prefill kernels are compiled \
                 for PAGE_SIZE=16 — no paged-prefill kernel path exists for TQ (the \
                 monolith never had one either); tracked in #68"
            ),
        }
    }

    #[must_use]
    pub fn kv_format(self) -> KVFormat {
        match self {
            Self::Bf16 => KVFormat::BF16,
            Self::Int8 => KVFormat::INT8,
            Self::Fp8 => KVFormat::FP8E4M3,
        }
    }
}

/// The real cuda-kernels executor. Dense Qwen3 uses the paged KV pool;
/// Qwen3.5/3.6 and DSv4 keep model-specific per-slot state inside their
/// executor arms.
pub(crate) enum RealCudaExecutor {
    Qwen(Box<QwenCudaExecutor>),
    Qwen35(Box<Qwen35CudaExecutor>),
    Dsv4(Box<Dsv4CudaExecutor>),
}

impl std::fmt::Debug for RealCudaExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Qwen(q) => q.fmt(f),
            Self::Qwen35(q) => q.fmt(f),
            Self::Dsv4(d) => d.fmt(f),
        }
    }
}

impl RealCudaExecutor {
    pub(crate) fn from_qwen3_bf16_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
        kv_dtype: CudaKvCacheDtype,
        mem_fraction_static: f64,
    ) -> Result<Self> {
        Ok(Self::Qwen(Box::new(
            QwenCudaExecutor::from_qwen3_bf16_safetensors(
                model_path,
                num_slots,
                total_pages,
                kv_dtype,
                mem_fraction_static,
            )?,
        )))
    }

    /// `dspark_draft_model`: `Some(dir)` loads the DSpark/DFlash block drafter
    /// from that checkpoint dir and enables `--spec-type dspark` decode.
    /// `mtp_draft_tokens`: `Some(n)` loads the checkpoint-native NextN-MTP head
    /// for `--spec-type mtp` speculative decode (draft depth `n`).
    pub(crate) fn from_qwen35_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
        max_total_tokens: usize,
        kv_dtype: CudaKvCacheDtype,
        mem_fraction_static: f64,
        dspark_draft_model: Option<&Path>,
        dspark_sps_bias_ms: f32,
        dspark_sps_row_ms: f32,
        markov_head_rank: Option<usize>,
        dspark_block_size: Option<usize>,
        mtp_draft_tokens: Option<usize>,
        memory_budget_bytes: Option<usize>,
    ) -> Result<Self> {
        Ok(Self::Qwen35(Box::new(
            Qwen35CudaExecutor::from_qwen35_safetensors(
                model_path,
                num_slots,
                total_pages,
                max_total_tokens,
                kv_dtype,
                mem_fraction_static,
                dspark_draft_model,
                dspark_sps_bias_ms,
                dspark_sps_row_ms,
                markov_head_rank,
                dspark_block_size,
                mtp_draft_tokens,
                memory_budget_bytes,
            )?,
        )))
    }

    /// Build the DSv4-Flash executor (MLA + HC + FP8 MoE, multi-GPU TP/EP).
    /// `mtp_draft_tokens`: `Some(n)` = config-driven MTP spec decode on (draft
    /// depth `n`); `mtp_draft_topk`: `Some(k)` = per-level MTP draft candidate
    /// width (`1` = chain-only candidates).
    /// `None` = no MTP head (DSpark or plain decode).
    pub(crate) fn from_dsv4_fp8_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        max_seq_len: usize,
        mtp_draft_tokens: Option<usize>,
        mtp_draft_topk: Option<usize>,
        dspark_draft_model: Option<&Path>,
        dspark_sps_bias_ms: f32,
        dspark_sps_row_ms: f32,
    ) -> Result<Self> {
        Ok(Self::Dsv4(Box::new(
            Dsv4CudaExecutor::from_dsv4_fp8_safetensors(
                model_path,
                num_slots,
                max_seq_len,
                mtp_draft_tokens,
                mtp_draft_topk,
                dspark_draft_model,
                dspark_sps_bias_ms,
                dspark_sps_row_ms,
            )?,
        )))
    }

    pub(crate) fn submit(
        &mut self,
        plan: &ForwardPlan,
        host_kv: &mut dyn KvPool,
    ) -> Result<StepOutput> {
        // Both Qwen arms lower the host pool's page table into their device pool
        // (host-authoritative mirror); DSv4 validates + adapts the descriptor.
        // Qwen3.5 reads the host pool directly rather than the flattened
        // descriptor — its spec path also GROWS the slot mid-step.
        match self {
            Self::Qwen(q) => {
                let kv_batch = KvBatchDescriptor::from_plan(plan, host_kv)?;
                q.submit(plan, host_kv, &kv_batch)
            }
            Self::Qwen35(q) => q.submit(plan, host_kv),
            Self::Dsv4(d) => {
                let kv_batch = KvBatchDescriptor::from_plan(plan, host_kv)?;
                d.submit(plan, &kv_batch)
            }
        }
    }

    pub(crate) fn warmup(&mut self) -> Result<()> {
        match self {
            Self::Qwen(q) => q.warmup(),
            // Qwen3.5/3.6 hybrid: whole-step decode graph (opt-in,
            // --qwen35-decode-graph) — warmup logs the gate verdict;
            // capture itself is lazy per slot.
            Self::Qwen35(q) => q.warmup(),
            // DSv4 drives its own per-portion/whole-step graph gates inside
            // the model (ARLE_DSV4_* envs).
            Self::Dsv4(_) => Ok(()),
        }
    }

    pub(crate) fn poll_background(&mut self) {
        if let Self::Dsv4(d) = self {
            d.poll_prefix_captures();
        }
    }

    /// Model-default stop tokens, engine-core's fallback stop set for
    /// requests that supply none. Without this every CUDA request ignores the
    /// model's EOS and pads to `max_tokens` with post-EOS degenerate text
    /// (found via the MTP P0 probe — the Metal executor always had the
    /// equivalent override).
    pub(crate) fn model_stop_token_ids(&self) -> Vec<u32> {
        match self {
            Self::Qwen(q) => q.model.config.eos_token_ids.clone(),
            Self::Qwen35(q) => q.model.config.stop_token_ids.clone(),
            Self::Dsv4(d) => d.model.config.eos_token_id.into_iter().collect(),
        }
    }

    /// Per-forward total-token cap. Only DSv4 on the deepep_ll transport has a
    /// hard limit (the LL NVSHMEM dispatch buffer's
    /// `num_max_dispatch_tokens_per_rank`); all other arms are unbounded.
    pub(crate) fn max_tokens_per_step(&self) -> usize {
        match self {
            Self::Qwen(_) | Self::Qwen35(_) => usize::MAX,
            Self::Dsv4(d) => d.model.max_tokens_per_step().unwrap_or(usize::MAX),
        }
    }

    /// Page-granular host-tier hooks. Dense Qwen3 is the only CUDA arm with a
    /// page-addressable device pool here; DSv4 whole-slot swap is a separate
    /// hook below.
    pub(crate) fn kv_tier_capacity_pages(&self) -> usize {
        match self {
            Self::Qwen(q) => q.kv_tier_capacity_pages(),
            Self::Dsv4(_) => 0,
            Self::Qwen35(_) => 0,
        }
    }

    pub(crate) fn kv_tier_page_bytes(&self) -> usize {
        match self {
            Self::Qwen(q) => q.kv_tier_page_bytes(),
            Self::Dsv4(_) => 0,
            Self::Qwen35(_) => 0,
        }
    }

    pub(crate) fn kv_tier_host_demoted_pages(&self) -> usize {
        match self {
            Self::Qwen(q) => q.kv_tier_host_demoted_pages(),
            Self::Dsv4(d) => d.kv_tier_host_demoted_pages(),
            Self::Qwen35(q) => q.kv_tier_host_demoted_pages(),
        }
    }

    /// Cumulative spec-decode counters: DSpark on Qwen3.5/3.6, MTP on DSv4.
    /// Host-side integers maintained in the accept-commit paths — no device sync.
    pub(crate) fn spec_decode_stats(&self) -> infer_seam::SpecDecodeStats {
        match self {
            Self::Qwen(_) => infer_seam::SpecDecodeStats::default(),
            Self::Qwen35(q) => {
                if let Some(ds) = q.dspark.as_ref() {
                    infer_seam::SpecDecodeStats {
                        chains: ds.chains as u64,
                        drafted: (ds.accepts + ds.rejects) as u64,
                        accepted: ds.accepts as u64,
                        rejected: ds.rejects as u64,
                        partial_ctx_chains: ds.partial_ctx_chains as u64,
                    }
                } else if let Some(m) = q.mtp.as_ref() {
                    infer_seam::SpecDecodeStats {
                        chains: m.chains as u64,
                        drafted: (m.accepts + m.rejects) as u64,
                        accepted: m.accepts as u64,
                        rejected: m.rejects as u64,
                        partial_ctx_chains: 0,
                    }
                } else {
                    infer_seam::SpecDecodeStats::default()
                }
            }
            Self::Dsv4(d) => infer_seam::SpecDecodeStats {
                chains: d.mtp_chains as u64,
                drafted: (d.mtp_accepts + d.mtp_rejects) as u64,
                accepted: d.mtp_accepts as u64,
                rejected: d.mtp_rejects as u64,
                partial_ctx_chains: 0,
            },
        }
    }

    /// KV tokens a decode row reaches in one submit. Only the Qwen3.5/3.6 arm
    /// grows the host pool for its spec chain (MTP/DSpark); DSv4 MTP grows its
    /// own device bands instead, and dense Qwen has no spec.
    pub(crate) fn spec_row_tokens(&self) -> usize {
        match self {
            Self::Qwen35(q) => q.spec_row_tokens(),
            Self::Qwen(_) | Self::Dsv4(_) => 1,
        }
    }

    /// Reads host atomics and performs no device synchronization.
    pub(crate) fn operator_dispatch_stats(&self) -> infer_seam::OperatorDispatchStats {
        crate::ops::qwen_fp8_dense_operator_stats()
    }

    pub(crate) fn op_timing_stats(&self) -> infer_seam::OpTimingStats {
        infer_seam::OpTimingStats {
            ops: crate::profile::get_op_stats()
                .into_iter()
                .map(|(name, total_micros, count)| infer_seam::OpTimingEntry {
                    name,
                    total_micros,
                    count,
                })
                .collect(),
        }
    }

    pub(crate) fn kv_tier_disk_pages(&self) -> usize {
        match self {
            Self::Qwen(q) => q.kv_tier_disk_pages(),
            Self::Dsv4(d) => d.kv_tier_disk_pages(),
            Self::Qwen35(q) => q.kv_tier_disk_pages(),
        }
    }

    pub(crate) fn kv_tier_read_hits(&self) -> infer_seam::KvTierReadHits {
        match self {
            Self::Dsv4(d) => d.kv_tier_read_hits(),
            Self::Qwen(q) => q.kv_tier_read_hits(),
            Self::Qwen35(q) => q.kv_tier_read_hits(),
        }
    }

    pub(crate) fn kv_tier_io_stats(&self) -> infer_seam::KvTierIoStats {
        match self {
            Self::Qwen(q) => q.kv_tier_io_stats(),
            Self::Dsv4(d) => d.kv_tier_io_stats(),
            Self::Qwen35(q) => q.kv_tier_io_stats(),
        }
    }

    pub(crate) fn kv_tier_location(&self, key: u64) -> Option<infer_seam::KvTierLocation> {
        match self {
            Self::Qwen(q) => q.kv_tier_location(key),
            Self::Dsv4(d) => d.kv_tier_location(key),
            Self::Qwen35(q) => q.kv_tier_location(key),
        }
    }

    pub(crate) fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
        match self {
            Self::Qwen(q) => q.reusable_prefix_blocks(blocks),
            Self::Qwen35(q) => q.reusable_prefix_blocks(blocks),
            Self::Dsv4(d) => d.reusable_prefix_blocks(blocks),
        }
    }

    /// Reuse length for admitting a request whose prompt is `tokens`. Only DSv4
    /// finish-write-through differs from the plain count (it content-verifies its
    /// sub-page tail against the prompt); the others ignore `tokens`.
    pub(crate) fn reusable_prefix_blocks_for_prompt(
        &self,
        blocks: &[PrefixBlock],
        tokens: &[u32],
    ) -> usize {
        match self {
            Self::Dsv4(d) => d.reusable_prefix_blocks_for_prompt(blocks, tokens),
            Self::Qwen(q) => q.reusable_prefix_blocks(blocks),
            Self::Qwen35(q) => q.reusable_prefix_blocks(blocks),
        }
    }

    pub(crate) fn demote_prefix_pages(&mut self, entries: &[(u32, u64)]) -> Result<usize> {
        match self {
            Self::Qwen(q) => q.demote_prefix_pages(entries),
            Self::Dsv4(_) => Ok(0),
            Self::Qwen35(_) => Ok(0),
        }
    }

    pub(crate) fn promote_prefix_pages(&mut self, entries: &[(u64, u32)]) -> Result<()> {
        match self {
            Self::Qwen(q) => q.promote_prefix_pages(entries),
            Self::Dsv4(_) => Ok(()),
            Self::Qwen35(_) => {
                anyhow::bail!(
                    "page-granular KV tier store is implemented only for dense Qwen3 CUDA"
                )
            }
        }
    }

    pub(crate) fn drop_kv_tier_entries(&mut self, keys: &[u64]) {
        if let Self::Qwen(q) = self {
            q.drop_kv_tier_entries(keys);
        }
    }

    /// Whole-slot KV tier hooks. Qwen3.6 and DSv4 park a demoted slot's full
    /// device image in the tier; dense Qwen3 reports no slot tier (its
    /// page-granular radix tier handles capacity).
    pub(crate) fn kv_slot_tier_enabled(&self) -> bool {
        matches!(self, Self::Qwen35(_) | Self::Dsv4(_))
    }

    pub(crate) fn demote_slot(&mut self, slot: usize, key: u64) -> Result<bool> {
        match self {
            Self::Qwen35(q) => q.demote_slot(slot, key),
            Self::Dsv4(d) => d.demote_slot(slot, key),
            Self::Qwen(_) => Ok(false),
        }
    }

    pub(crate) fn promote_slot(&mut self, key: u64, slot: usize, slot_pages: &[u32]) -> Result<()> {
        match self {
            Self::Qwen35(q) => q.promote_slot(key, slot, slot_pages),
            Self::Dsv4(d) => d.promote_slot(key, slot, slot_pages),
            Self::Qwen(_) => {
                anyhow::bail!("whole-slot KV tier store is only implemented for Qwen3.6/DSv4 CUDA")
            }
        }
    }

    pub(crate) fn drop_kv_slot_entries(&mut self, keys: &[u64]) {
        match self {
            Self::Qwen35(q) => q.drop_kv_slot_entries(keys),
            Self::Dsv4(d) => d.drop_kv_slot_entries(keys),
            Self::Qwen(_) => {}
        }
    }

    /// No arm implements a position-0 prefix store; always no match.
    pub(crate) fn cached_prefix_match_len(&self, _tokens: &[u32]) -> Result<usize> {
        Ok(0)
    }

    /// `BackendExecutor::tp_sync_min` — see there for why the scheduler needs
    /// this (2026-07-05 TP=4 admission livelock).
    pub(crate) fn tp_sync_min(&self, local: usize) -> Result<usize> {
        match self {
            Self::Qwen(q) => q.tp_sync_min(local),
            Self::Qwen35(q) => q.tp_sync_min(local),
            Self::Dsv4(d) => d.tp_sync_min(local),
        }
    }

    /// This rank's (cp_rank, cp_size) for the host pool's shard filter, or
    /// `None` when 2D is not engaged.
    pub(crate) fn kv_shard_spec(&self) -> Option<(usize, usize)> {
        match self {
            Self::Qwen35(q) => q.kv_shard_spec(),
            Self::Qwen(_) | Self::Dsv4(_) => None,
        }
    }

    pub(crate) fn restore_cached_prefix(
        &mut self,
        _slot: usize,
        _tokens: &[u32],
        _matched_len: usize,
        _slot_pages: &[u32],
    ) -> Result<()> {
        // Never called: cached_prefix_match_len always returns 0 today.
        anyhow::bail!("backend has no position-0 prefix store")
    }

    /// Restore backend side state for `slot` when reusing a prefix of length
    /// `matched_len`. Qwen3.5/3.6 hybrid restores the recurrent sidecar; DSv4
    /// restores the content-keyed prefix-state pool entries (#154 Phase 2);
    /// no-op for Qwen3 (pages-only). An `Err` makes the engine free the slot
    /// and fall back to full recompute — never a partial restore.
    pub(crate) fn restore_prefix_sidecar(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
        prefix_pages: &[u32],
    ) -> Result<usize> {
        match self {
            // Qwen3.5/3.6 returns the ABSOLUTE restored length (a periodic
            // boundary `B ≤ matched_len` on a cross-conversation hit, else
            // `matched_len`).
            Self::Qwen35(q) => q.restore_recurrent_sidecar(slot, tokens, matched_len, prefix_pages),
            // DSv4 returns EXTRA tokens beyond `matched_len`; lift to absolute.
            Self::Dsv4(d) => d
                .restore_prefix_state(slot, tokens, matched_len, prefix_pages)
                .map(|extra| matched_len + extra),
            Self::Qwen(_) => Ok(matched_len),
        }
    }

    /// Write the finishing slot's full frontier state through to the content-keyed
    /// prefix store (DSv4 finish-write-through, behind `--dsv4-decode-reuse`);
    /// no-op for the other backends. `slot_pages` is the slot's OWN host page
    /// chain (the pool keys entries by these ids, reconciled to the radix's
    /// canonical ids by the subsequent `save_prefix_sidecar` confirm/repair).
    pub(crate) fn capture_finish_frontier(
        &mut self,
        slot: usize,
        tokens: &[u32],
        slot_pages: &[u32],
    ) -> Result<()> {
        match self {
            Self::Dsv4(d) => d.capture_finish_frontier(slot, tokens, slot_pages),
            Self::Qwen(_) | Self::Qwen35(_) => {
                let _ = (slot, tokens, slot_pages);
                Ok(())
            }
        }
    }

    /// Extra chunk-end alignment beyond KV pages (seam contract): DSv4's
    /// boundary sections (overlap registers + ring) only materialize at a
    /// forward's own end, so every intermediate boundary must be forced.
    pub(crate) fn prefill_restore_boundary_alignment(&self) -> usize {
        match self {
            Self::Dsv4(d) => d.prefill_restore_boundary_alignment(),
            Self::Qwen(_) | Self::Qwen35(_) => 1,
        }
    }

    /// Largest single prefill forward (seam contract): DSv4 bounds it by its
    /// snapshot grain; the Qwen arms accept any chunk the config allows.
    pub(crate) fn max_prefill_chunk(&self) -> usize {
        match self {
            Self::Dsv4(d) => d.max_prefill_chunk(),
            Self::Qwen(_) | Self::Qwen35(_) => usize::MAX,
        }
    }

    /// Capture the page-radix sidecar for `slot` at the just-published prefix
    /// `tokens[..matched_len]`. Qwen3.5/3.6 hybrid stores recurrent + full-attn
    /// KV atomically into the tier; DSv4 confirms ONLY the newly-cached pages'
    /// pool entries (radix retention makes those ids recycle-proof — a deduped
    /// page frees and recycles, so confirming it would leave a stale confirmed
    /// entry under a reusable id), then repairs deduped positions whose
    /// canonical pool entry evicted (#157: re-key the slot's own provisional
    /// entry — content-identical by construction — else the canonical chain
    /// floor-0-locks until radix eviction); no-op for Qwen3.
    pub(crate) fn save_prefix_sidecar(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
        prefix_pages: &[u32],
        slot_pages: &[u32],
        newly_cached: &[u32],
    ) -> Result<()> {
        match self {
            Self::Qwen35(q) => q.save_recurrent_sidecar(slot, tokens, matched_len, prefix_pages),
            Self::Dsv4(d) => {
                d.confirm_prefix_pages(newly_cached);
                d.repair_prefix_pool_chain(prefix_pages, slot_pages);
                Ok(())
            }
            Self::Qwen(_) => Ok(()),
        }
    }

    /// Radix evicted these host-pool prefix pages: drop any state keyed to
    /// them (eviction rides the radix, no independent sidecar LRU). Qwen3.5/3.6
    /// drops recurrent sidecars; DSv4 drops prefix-state pool entries.
    pub(crate) fn release_prefix_pages(&mut self, pages: &[u32]) {
        match self {
            Self::Qwen35(q) => q.release_sidecar_pages(pages),
            Self::Dsv4(d) => d.release_prefix_pages(pages),
            Self::Qwen(_) => {}
        }
    }

    /// A slot's pages returned to the free pool: DSv4 drops their provisional
    /// (never-confirmed) pool entries so a recycled id can't resurrect stale
    /// content; no-op elsewhere.
    pub(crate) fn release_provisional_prefix_pages(&mut self, pages: &[u32]) {
        if let Self::Dsv4(d) = self {
            d.release_provisional_prefix_pages(pages);
        }
    }

    /// Engine freed `slot`'s host pages: the backend must return the slot's
    /// device pages in the same tick, or the host admission pool over-reports
    /// free capacity and the planner licenses chunks the device pool can't
    /// hold (engine-fatal at submit). DSv4 returns demand-paged FlashMLA band
    /// pages (#154 Phase 3b); Qwen3.5/3.6 drops its page mirror (and any recall
    /// keepalive); dense Qwen mirrors the host pool (nothing to free).
    pub(crate) fn release_kv_slot(&mut self, slot: usize) {
        match self {
            Self::Qwen35(q) => {
                if let Err(err) = q.release_kv_slot(slot) {
                    warn!("Qwen3.5 release_kv_slot({slot}) failed: {err:#}");
                }
            }
            Self::Dsv4(d) => {
                if let Err(err) = d.kv_adapter.flashmla_free_slot(slot) {
                    warn!("DSv4 release_kv_slot({slot}) failed: {err:#}");
                }
            }
            Self::Qwen(_) => {}
        }
    }

    /// Whether the engine must run the device-pool fit gate. Dense Qwen
    /// mirrors the host pool exactly — inert.
    pub(crate) fn kv_device_gate_active(&self) -> bool {
        match self {
            Self::Dsv4(d) => d.kv_device_gate_active(),
            Self::Qwen(_) | Self::Qwen35(_) => false,
        }
    }

    /// Per-row device-pool fit for the engine's gate. Only DSv4 still needs it
    /// (per-layer band growth paired with each layer pool, #160); the two Qwen
    /// arms mirror the host pool, which is the fit authority.
    pub(crate) fn kv_device_fit(
        &self,
        rows: &[infer_seam::DeviceRowDemand],
        unfit: &mut Vec<usize>,
    ) {
        if let Self::Dsv4(d) = self {
            d.kv_device_fit(rows, unfit);
        }
    }

    /// Re-budget the host-demoted tier store (`0` disables; pre-serve only). No-op on
    /// arms without a tier store.
    pub(crate) fn set_kv_tier_budget_bytes(&mut self, bytes: usize) {
        match self {
            Self::Qwen(q) => q.set_kv_tier_budget_bytes(bytes),
            // L2: the Qwen3.6 arm's G3 slot_tier also honors the explicit cap.
            Self::Qwen35(q) => q.set_kv_tier_budget_bytes(bytes),
            Self::Dsv4(d) => d.set_kv_tier_budget_bytes(bytes),
        }
    }

    /// Opt into session KV-recall (`--kv-recall`, default off). Wired for the
    /// dense-Qwen3 paged decode arm (the only CUDA arm with a paged page table +
    /// page-granular tier); the Qwen3.5/3.6 hybrid and DSv4 arms own per-slot KV
    /// state internally and ignore the request (logged at the call site). Off →
    /// the decode hot path is byte-identical.
    pub(crate) fn set_kv_recall(&mut self, enabled: bool) -> Result<()> {
        match self {
            Self::Qwen(q) => {
                q.set_kv_recall(enabled);
                Ok(())
            }
            Self::Qwen35(q) => q.set_kv_recall(enabled),
            Self::Dsv4(_) => {
                if enabled {
                    warn!(
                        "--kv-recall requested but DSv4 owns per-slot MLA KV; \
                         session KV-recall is wired for dense Qwen3 + Qwen3.6 only (ignored)"
                    );
                }
                Ok(())
            }
        }
    }

    /// Attach the opt-in disk spill level (pre-serve only). Returns whether
    /// any arm consumed it, so callers can fail closed instead of silently
    /// dropping an explicit `--kv-disk` request.
    pub(crate) fn set_kv_tier_disk(
        &mut self,
        root: std::path::PathBuf,
        budget_bytes: usize,
    ) -> bool {
        match self {
            Self::Qwen(q) => q.set_kv_tier_disk(root, budget_bytes),
            Self::Qwen35(q) => q.set_kv_tier_disk(root, budget_bytes),
            Self::Dsv4(d) => d.set_kv_tier_disk(root, budget_bytes),
        }
    }

    pub(crate) fn dsv4_verify_forward_selftest(&mut self, prompt: &[u32]) -> Result<()> {
        match self {
            Self::Dsv4(d) => d.verify_forward_selftest(prompt),
            Self::Qwen(_) | Self::Qwen35(_) => {
                anyhow::bail!("DSv4 verify-forward selftest requires a DSv4 executor")
            }
        }
    }

    /// Effective slot count after any model-side KV-budget clamp. The DSv4
    /// constructor may clamp below the requested `num_slots` (dynamic KV mem
    /// budget); the scheduler MUST admit against this count, not the requested
    /// one, or it admits requests to slots the executor has no arena for.
    pub(crate) fn effective_num_slots(&self) -> usize {
        match self {
            Self::Qwen(q) => q.num_slots,
            Self::Qwen35(q) => q.num_slots,
            Self::Dsv4(d) => d.num_slots,
        }
    }

    /// Actual shared device-pool page count, for the paged-pool models (dense
    /// Qwen3 + Qwen3.6 + DSv4 MLA latent pool). All three profile their pool
    /// from measured free VRAM at construction, so this is the page count the
    /// host admission pool MUST mirror 1:1 — not the requested `total_pages`.
    pub(crate) fn effective_total_pages(&self) -> Option<usize> {
        match self {
            Self::Qwen(q) => Some(q.kv.max_total_pages),
            Self::Qwen35(q) => q.full_attn_pool_pages(),
            Self::Dsv4(d) => d.kv_adapter.flashmla_total_pages(),
        }
    }

    /// Device pool page size (tokens/page), for arms whose host admission pool
    /// must mirror the device pool's page granularity. DSv4's MLA latent pool
    /// pages at `page_block_size` (64), NOT `config.page_size` (16) — the host
    /// `CudaKvPool` must use this or it gates at 1/4 device token capacity (H3).
    /// `None` where the host page size already matches the device default.
    pub(crate) fn effective_page_size(&self) -> Option<usize> {
        match self {
            Self::Qwen(_) | Self::Qwen35(_) => None,
            Self::Dsv4(d) => d.kv_adapter.flashmla_page_size(),
        }
    }

    pub(crate) fn effective_fixed_pages_per_slot(&self) -> Option<usize> {
        match self {
            Self::Qwen(_) | Self::Qwen35(_) => None,
            Self::Dsv4(d) => d.kv_adapter.flashmla_max_slot_pages(),
        }
    }

    /// OPD teacher raw-logits forward (Qwen3.5/3.6 hybrid only). Returns the full
    /// `[seq_len, vocab]` logits without sampling. Dense Qwen3 / DSv4 are not OPD
    /// teacher targets on this surface and bail.
    pub(crate) fn forward_token_logits(
        &mut self,
        input_ids: &[u32],
        positions: &[u32],
    ) -> Result<(DeviceVec, [usize; 2])> {
        match self {
            Self::Qwen35(q) => q.forward_token_logits(input_ids, positions),
            Self::Qwen(_) => {
                anyhow::bail!(
                    "forward_token_logits is wired for the Qwen3.5/3.6 hybrid OPD teacher, not dense Qwen3"
                )
            }
            Self::Dsv4(_) => {
                anyhow::bail!(
                    "forward_token_logits is wired for the Qwen3.5/3.6 hybrid OPD teacher, not DSv4"
                )
            }
        }
    }

    /// Trunk taps + final hidden states for offline DSpark draft training
    /// (Qwen3.5/3.6 hybrid only), host-side.
    pub(crate) fn forward_training_taps(
        &mut self,
        input_ids: &[u32],
        target_layer_ids: &[i64],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        match self {
            Self::Qwen35(q) => q.forward_training_taps(input_ids, target_layer_ids),
            Self::Qwen(_) | Self::Dsv4(_) => anyhow::bail!(
                "forward_training_taps is wired for the Qwen3.5/3.6 hybrid DSpark draft, \
                 not dense Qwen3 or DSv4"
            ),
        }
    }

    /// Device context of the underlying model (for the OPD raw-logits surface to
    /// build a `RawLogits` carrying a sync/consume handle).
    pub(crate) fn device(&self) -> &DeviceContext {
        match self {
            Self::Qwen35(q) => q.device(),
            Self::Qwen(q) => &q.model.ctx,
            Self::Dsv4(d) => &d.model.ctx,
        }
    }

    /// Offload the model's device weights to host RAM for the OPD teacher
    /// time-share, returning the device bytes freed. Only the Qwen3.5/3.6 hybrid
    /// arm (the OPD teacher target) supports this; the dense-Qwen3 and DSv4 arms
    /// bail (DSv4 is multi-GPU FP8 + the dense path is not an OPD teacher).
    pub(crate) fn offload_engine_weights(&mut self) -> Result<usize> {
        match self {
            Self::Qwen35(q) => q.offload_engine_weights(),
            Self::Qwen(_) => {
                anyhow::bail!(
                    "offload_engine_weights is only supported on the Qwen3.5/3.6 hybrid OPD teacher path"
                )
            }
            Self::Dsv4(_) => {
                anyhow::bail!(
                    "offload_engine_weights is not supported on the DSv4 multi-GPU FP8 path"
                )
            }
        }
    }

    /// Release inference forward scratch (Qwen3.5/3.6 only); a no-op on the
    /// dense-Qwen3 and DSv4 arms (no `Qwen35Workspace`), so this is safe to call
    /// unconditionally on the OPD writeback path.
    pub(crate) fn release_inference_scratch(&mut self) -> Result<()> {
        match self {
            Self::Qwen35(q) => q.release_inference_scratch(),
            Self::Qwen(_) => Ok(()),
            Self::Dsv4(_) => Ok(()),
        }
    }

    /// Drop the rollout engine's full-attn KV pool (Qwen3.5/3.6 only); a no-op on
    /// the other arms. agent-OPD writeback headroom — see `Qwen35CudaExecutor`.
    pub(crate) fn release_kv_pool(&mut self) -> Result<()> {
        match self {
            Self::Qwen35(q) => q.release_kv_pool(),
            Self::Qwen(_) => Ok(()),
            Self::Dsv4(_) => Ok(()),
        }
    }

    /// Re-acquire the rollout engine's full-attn KV pool (Qwen3.5/3.6 only) before
    /// the next agent-OPD round's rollout; a no-op on the other arms.
    pub(crate) fn ensure_kv_pool(&mut self) -> Result<()> {
        match self {
            Self::Qwen35(q) => q.ensure_kv_pool(),
            Self::Qwen(_) => Ok(()),
            Self::Dsv4(_) => Ok(()),
        }
    }

    pub(crate) fn reload_engine_weights(&mut self) -> Result<()> {
        match self {
            Self::Qwen35(q) => q.reload_engine_weights(),
            Self::Qwen(_) => {
                anyhow::bail!(
                    "reload_engine_weights is only supported on the Qwen3.5/3.6 hybrid OPD teacher path"
                )
            }
            Self::Dsv4(_) => {
                anyhow::bail!(
                    "reload_engine_weights is not supported on the DSv4 multi-GPU FP8 path"
                )
            }
        }
    }

    /// Per-step student LoRA re-merge (OPD P2). Only the Qwen3.5/3.6 hybrid
    /// executor carries the OPD student; dense Qwen3 / DSv4 are not student
    /// targets and reject the update.
    pub(crate) fn remerge_student_lora(
        &mut self,
        update: crate::qwen35::StudentLoraUpdate,
    ) -> Result<()> {
        match self {
            Self::Qwen35(q) => q.remerge_student_lora(update),
            Self::Qwen(_) => anyhow::bail!(
                "student LoRA re-merge is only wired for the Qwen3.5/3.6 hybrid OPD student; \
                 the dense Qwen3 executor is not a student target"
            ),
            Self::Dsv4(_) => anyhow::bail!(
                "student LoRA re-merge is only wired for the Qwen3.5/3.6 hybrid OPD student; \
                 the DSv4-Flash executor is not a student target"
            ),
        }
    }

    /// Hot-swap the DSpark Markov head weights.
    pub(crate) fn update_dspark_markov_weights(&mut self, w1: &[f32], w2: &[f32]) -> Result<()> {
        match self {
            Self::Qwen35(q) => q.update_dspark_markov_weights(w1, w2),
            Self::Dsv4(d) => d.update_dspark_markov_weights(w1, w2),
            Self::Qwen(_) => anyhow::bail!(
                "DSpark Markov weight update is only wired for the Qwen3.5/3.6 and DSv4-Flash executors; \
                 the dense Qwen3 executor has no DSpark head"
            ),
        }
    }

    /// Read-only borrow of resident FP8 block-scaled base projection pointers
    /// for train-infer weight sharing (`--share-frozen-base`). Only the
    /// Qwen3.5/3.6 hybrid student carries shareable FP8 base weights.
    pub(crate) fn frozen_base_fp8_pointers(
        &self,
    ) -> Result<Vec<crate::qwen35::SharedFp8BaseProjection>> {
        match self {
            Self::Qwen35(q) => q.frozen_base_fp8_pointers(),
            Self::Qwen(_) => anyhow::bail!(
                "frozen-base FP8 sharing is only wired for the Qwen3.5/3.6 hybrid OPD student; \
                 the dense Qwen3 executor is not a student target"
            ),
            Self::Dsv4(_) => anyhow::bail!(
                "frozen-base FP8 sharing is only wired for the Qwen3.5/3.6 hybrid OPD student; \
                 the DSv4-Flash executor is not a student target"
            ),
        }
    }

    /// Non-owning views of every resident dense-BF16 base projection's device
    /// pointer, for refreshing the train student's frozen base AFTER a LoRA
    /// re-merge.
    pub(crate) fn frozen_base_bf16_pointers(
        &self,
    ) -> Result<Vec<crate::qwen35::SharedBf16BaseProjection>> {
        match self {
            Self::Qwen35(q) => q.frozen_base_bf16_pointers(),
            Self::Qwen(_) => anyhow::bail!(
                "frozen-base BF16 sharing is only wired for the Qwen3.5/3.6 hybrid OPD student"
            ),
            Self::Dsv4(_) => anyhow::bail!(
                "frozen-base BF16 sharing is only wired for the Qwen3.5/3.6 hybrid OPD student"
            ),
        }
    }
}

use kv_native_sys::{BLOB_CHUNK_BYTES, KvTierStore, default_t1_budget_bytes};

/// Placeholder, not measurement-derived — dormant until `demote` is wired.
///
/// Construction-time default fraction of available host DRAM the L2 KV tier
/// may claim — the shared-box-safe 0.5 (the store is pageable host memory; see
/// `infer_seam::DramTierPolicy`). The engine builder re-budgets pre-serve with
/// the per-rank share resolved from `--kv-dram`.
pub(crate) const DEFAULT_DRAM_FRACTION: f64 = 0.5;

/// Per-rank L2 budget at construction: the deployment-total default
/// (`default_t1_budget_bytes(DEFAULT_DRAM_FRACTION)`) divided by the TP world
/// size. The engine builder re-budgets pre-serve with the per-rank share
/// resolved from `--kv-dram`; dividing here keeps the pre-setter window and
/// direct-construction paths (agent-bench) correct under TP>1. World resolves
/// from env — the builder's `TpEnvGuard` sets `INFER_TP_SIZE` before
/// construction — and defaults to 1 when unset or unreadable.
pub(crate) fn default_t1_budget_per_rank() -> usize {
    let world = crate::tp::resolve_tp_config_from_env()
        .map(|c| c.world_size)
        .unwrap_or(1);
    default_t1_budget_bytes(DEFAULT_DRAM_FRACTION) / world.max(1)
}

/// `slot_tier` key namespaces (top byte, see `kv_native_sys::tier_key`), so features
/// sharing THE store never collide and a future kind (e.g. a suffix cache) is
/// one new constant.
/// Parked whole-slot images (key = engine-minted swap key).
pub(crate) const NS_SLOT: u64 = 1;
pub(crate) const NS_SLOT_CHUNK: u64 = 2;
/// Qwen3.5/3.6 recurrent prefix sidecars (key = token-prefix hash). Disjoint
/// namespace from `NS_SLOT`, so the Qwen3.6 arm's ONE `slot_tier` holds both the
/// whole-slot spill images and the page-radix sidecars without key aliasing.
pub(crate) const NS_SIDECAR: u64 = 3;
pub(crate) const NS_SIDECAR_CHUNK: u64 = 4;

/// Borrow a plan row's penalty snapshot; empty when the request set no penalty.
pub(crate) fn penalty_of(
    history: &Option<std::sync::Arc<[u32]>>,
    prompt_len: usize,
) -> infer_plan::PenaltyHistory<'_> {
    infer_plan::PenaltyHistory {
        tokens: history.as_deref().unwrap_or(&[]),
        prompt_len,
    }
}

pub(crate) fn sample_cuda_token(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    params: &SamplingParams,
    position: u64,
    penalty: infer_plan::PenaltyHistory<'_>,
) -> Result<u32> {
    // A grammar mask must reach the argmax, so greedy takes the host path too.
    if params.is_raw_argmax() {
        let token = argmax(ctx, logits)?;
        return Ok(token);
    }

    let logits_host = logits.to_host(ctx)?;
    let token = infer_plan::sample_token_penalized(&logits_host, params, position, penalty);
    Ok(token)
}

/// [`sample_cuda_token`] with a caller-provided persistent argmax scratch (one
/// device i32): the Qwen3.5/3.6 steady-state decode sampler — greedy decode
/// performs ZERO device allocations per token (the last per-token
/// `alloc_zeros(1)` moved into the workspace `argmax_out` slot). Always runs
/// OUTSIDE any CUDA-graph capture (argmax syncs + reads D2H). Also returns the
/// behavior logprob of the drawn token under the filtered sampling dist
/// (`None` for greedy — the P6 sidecar skips greedy requests).
pub(crate) fn sample_cuda_token_scratched(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    params: &SamplingParams,
    position: u64,
    argmax_out: &mut cudarc::driver::CudaSlice<i32>,
    penalty: infer_plan::PenaltyHistory<'_>,
) -> Result<(u32, Option<f32>)> {
    if params.is_raw_argmax() {
        let token = crate::ops::argmax_into(ctx, logits, argmax_out)?;
        return Ok((token, None));
    }
    let logits_host = logits.to_host(ctx)?;
    let (token, logprob) =
        infer_plan::sample_token_logprob_penalized(&logits_host, params, position, penalty);
    Ok((token, logprob))
}

/// [`sample_cuda_token_scratched`] plus the OpenAI logprobs capture
/// (`SamplingParams::top_logprobs`) written into `capture`; one host copy
/// feeds both the draw and the capture. Requests without the capture take the
/// scratched path unchanged.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_cuda_token_captured(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    params: &SamplingParams,
    position: u64,
    argmax_out: &mut cudarc::driver::CudaSlice<i32>,
    penalty: infer_plan::PenaltyHistory<'_>,
    capture: &mut Vec<(u32, f32)>,
) -> Result<(u32, Option<f32>)> {
    if params.top_logprobs.is_none() {
        return sample_cuda_token_scratched(ctx, logits, params, position, argmax_out, penalty);
    }
    let logits_host = logits.to_host(ctx)?;
    let (token, logprob) =
        infer_plan::sample_token_logprob_penalized(&logits_host, params, position, penalty);
    *capture = infer_plan::sampled_top_logprobs(&logits_host, params, penalty, token);
    Ok((token, logprob))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_t1_budget_per_rank_divides_by_world() {
        // Env is process-global; infer-cuda has no other env-mutating tests, so
        // no serialization is needed today. Restore on exit.
        let saved = std::env::var("INFER_TP_SIZE").ok();
        unsafe {
            std::env::set_var("INFER_TP_SIZE", "1");
        }
        let w1 = default_t1_budget_per_rank();
        unsafe {
            std::env::set_var("INFER_TP_SIZE", "2");
        }
        let w2 = default_t1_budget_per_rank();
        match saved {
            Some(v) => unsafe { std::env::set_var("INFER_TP_SIZE", v) },
            None => unsafe { std::env::remove_var("INFER_TP_SIZE") },
        }
        assert!(w1 > 0, "world=1 budget must be non-zero");
        assert_eq!(w2, w1 / 2, "world=2 budget must be half of world=1");
    }
}
