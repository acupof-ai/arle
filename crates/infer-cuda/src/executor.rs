//! Real CUDA executor: the engine-facing step driver and sampling tail.
//!
//! Wraps the loaded [`CudaModel`] + device [`PagedKVPool`], validates the
//! single-row plan, mirrors host→device page allocation, runs the forward, and
//! samples the next token (`sample_cuda_token`: greedy argmax / host sampling).

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, ensure};
use cuda_kernels::KVFormat;
use cuda_kernels::prelude::{DeviceContext, DeviceVec, PagedKVPool};
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

#[path = "executor/spec_decode.rs"]
mod spec_decode;

const SUPPORTED_PAGE_SIZE: usize = 16;

/// Flatten a `(session, block)` [`infer_seam::TierBlockKey`] into the
/// `CudaKvTierStore`'s opaque `u64` key namespace. The prefix tier already keys
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
const DECODE_GRAPH_MAX_SEQ_LEN: usize = 32_768;

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
fn decode_graph_enabled() -> bool {
    DECODE_GRAPH_DEFAULT_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

fn cuda_startup_profile_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("ARLE_CUDA_STARTUP_PROFILE").is_some())
}

fn cuda_startup_log(phase: &str, start: Instant, extra: std::fmt::Arguments<'_>) {
    if cuda_startup_profile_enabled() {
        info!(
            target: "infer_cuda::startup",
            "cuda_startup phase=executor.{phase} elapsed_ms={:.1} {extra}",
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

    /// The cuda-kernels pool format this dtype selects.
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
    pub(crate) fn from_qwen35_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
        kv_dtype: CudaKvCacheDtype,
        mem_fraction_static: f64,
        dspark_draft_model: Option<&Path>,
        dspark_conf_threshold: f32,
    ) -> Result<Self> {
        Ok(Self::Qwen35(Box::new(
            Qwen35CudaExecutor::from_qwen35_safetensors(
                model_path,
                num_slots,
                total_pages,
                kv_dtype,
                mem_fraction_static,
                dspark_draft_model,
                dspark_conf_threshold,
            )?,
        )))
    }

    /// Build the DSv4-Flash executor (MLA + HC + FP8 MoE, multi-GPU TP/EP).
    /// `mtp_draft_tokens`: `Some(n)` = config-driven MTP spec decode on (draft
    /// depth `n`); `mtp_draft_topk`: `Some(k)` = per-level MTP draft candidate
    /// width (`1` = chain-only candidates).
    /// `None` falls back to the `ARLE_DSV4_SPEC_DECODE` env gate.
    pub(crate) fn from_dsv4_fp8_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        max_seq_len: usize,
        mtp_draft_tokens: Option<usize>,
        mtp_draft_topk: Option<usize>,
    ) -> Result<Self> {
        Ok(Self::Dsv4(Box::new(
            Dsv4CudaExecutor::from_dsv4_fp8_safetensors(
                model_path,
                num_slots,
                max_seq_len,
                mtp_draft_tokens,
                mtp_draft_topk,
            )?,
        )))
    }

    pub(crate) fn submit(
        &mut self,
        plan: &ForwardPlan,
        host_kv: &mut dyn KvPool,
    ) -> Result<StepOutput> {
        // The descriptor is built per arm: Qwen-dense lowers it into its device
        // page pool (host-authoritative mirror), DSv4 validates + adapts it.
        // Qwen3.5 hybrid owns per-slot KV state and consumes neither, so it
        // skips the per-step page-id flattening entirely.
        match self {
            Self::Qwen(q) => {
                let kv_batch = KvBatchDescriptor::from_plan(plan, host_kv)?;
                q.submit(plan, host_kv, &kv_batch)
            }
            Self::Qwen35(q) => q.submit(plan),
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

    pub(crate) fn kv_tier_disk_pages(&self) -> usize {
        match self {
            Self::Qwen(q) => q.kv_tier_disk_pages(),
            Self::Dsv4(d) => d.kv_tier_disk_pages(),
            Self::Qwen35(q) => q.kv_tier_disk_pages(),
        }
    }

    pub(crate) fn kv_tier_location(&self, key: u64) -> Option<infer_seam::KvTierLocation> {
        match self {
            Self::Qwen(q) => q.kv_tier_location(key),
            Self::Dsv4(_) => None,
            Self::Qwen35(_) => None,
        }
    }

    pub(crate) fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
        match self {
            Self::Qwen(q) => q.reusable_prefix_blocks(blocks),
            Self::Qwen35(q) => q.reusable_prefix_blocks(blocks),
            Self::Dsv4(d) => d.reusable_prefix_blocks(blocks),
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

    /// Whole-slot KV tier hooks. CUDA implements these for the DSv4 and Qwen3.6
    /// (G3 capacity spill) executor arms; dense Qwen3 reports no slot tier (its
    /// page-granular radix tier handles capacity).
    pub(crate) fn kv_slot_tier_enabled(&self) -> bool {
        match self {
            Self::Dsv4(d) => d.kv_slot_tier_enabled(),
            Self::Qwen35(q) => q.kv_slot_tier_enabled(),
            Self::Qwen(_) => false,
        }
    }

    pub(crate) fn demote_slot(&mut self, slot: usize, key: u64) -> Result<bool> {
        match self {
            Self::Dsv4(d) => d.demote_slot(slot, key),
            Self::Qwen35(q) => q.demote_slot(slot, key),
            Self::Qwen(_) => Ok(false),
        }
    }

    pub(crate) fn promote_slot(&mut self, key: u64, slot: usize, slot_pages: &[u32]) -> Result<()> {
        match self {
            Self::Dsv4(d) => d.promote_slot(key, slot, slot_pages),
            Self::Qwen35(q) => {
                let _ = slot_pages;
                q.promote_slot(key, slot)
            }
            Self::Qwen(_) => {
                anyhow::bail!("whole-slot KV tier store is not implemented for dense Qwen3 CUDA")
            }
        }
    }

    pub(crate) fn drop_kv_slot_entries(&mut self, keys: &[u64]) {
        match self {
            Self::Dsv4(d) => d.drop_kv_slot_entries(keys),
            Self::Qwen35(q) => q.drop_kv_slot_entries(keys),
            Self::Qwen(_) => {}
        }
    }

    /// Generic engine-level prefix match hook (`BackendExecutor` trait).
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
    ) -> Result<()> {
        match self {
            Self::Qwen35(q) => q.restore_recurrent_sidecar(slot, tokens, matched_len, prefix_pages),
            Self::Dsv4(d) => d.restore_prefix_state(slot, matched_len, prefix_pages),
            Self::Qwen(_) => Ok(()),
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

    /// Capture the page-radix sidecar for `slot` at the just-published prefix
    /// `tokens[..matched_len]`. Qwen3.5/3.6 hybrid stores recurrent + full-attn
    /// KV atomically into the tier; DSv4 confirms the pages' pool entries
    /// (radix retention makes their ids recycle-proof); no-op for Qwen3.
    pub(crate) fn save_prefix_sidecar(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
        prefix_pages: &[u32],
    ) -> Result<()> {
        match self {
            Self::Qwen35(q) => q.save_recurrent_sidecar(slot, tokens, matched_len, prefix_pages),
            Self::Dsv4(d) => {
                d.confirm_prefix_pages(prefix_pages);
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

    /// Reload the model's device weights from the host snapshot.
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
}

use crate::kv_tier::{BLOB_CHUNK_BYTES, CudaKvTierStore, default_t1_budget_bytes};

/// Placeholder, not measurement-derived — dormant until `demote` is wired.

/// Construction-time default fraction of available host DRAM the L2 KV tier
/// may claim — the shared-box-safe 0.5 (the store is pageable host memory; see
/// `infer_seam::DramTierPolicy`). The engine builder re-budgets pre-serve with
/// the per-rank share resolved from `--kv-dram`.
pub(crate) const DEFAULT_DRAM_FRACTION: f64 = 0.5;

pub(crate) struct QwenCudaExecutor {
    model: CudaModel,
    kv: PagedKVPool,
    tier: CudaKvTierStore,
    num_slots: usize,
    /// Per-slot (occupant epoch, materialized token count) continuity guard.
    ///
    /// The host `CudaKvPool` is the single page allocator; this executor only
    /// mirrors the host page table into the device pool per scheduled row
    /// (`TokenKVPool::mirror_slot`), so the device pool no longer proves that
    /// the KV rows behind a resumed position were actually written. This
    /// watermark restores that loud-error contract: within one occupant epoch,
    /// rows must append contiguously; a NEW epoch may start at `append_pos > 0`
    /// only because the engine attached a radix prefix whose retained pages
    /// still hold the publishing request's KV rows.
    slot_progress: Vec<SlotProgress>,
    /// Fixed device buffers for the B=1 captured decode path. Built lazily at
    /// warmup; `None` until then / on capture failure (capture is never
    /// load-bearing for correctness — eager is the floor).
    decode_ctx: Option<DecodeGraphContext>,
    /// Per-shape captured decode graphs, keyed by page-table length: batch is
    /// fixed at [`DECODE_GRAPH_BATCH`], so `num_pages` is the only varying capture
    /// scalar. A new page count recaptures.
    graphs: Option<GraphBucket>,
    /// Session KV-recall opt-in (`--kv-recall`, default off). When off the decode
    /// hot path does no scoring and the page table is the full contiguous cache →
    /// baseline byte-identical (CUDA is the Stable backend). When on, recall is
    /// BF16-only and **eager-only** (the captured decode graph bakes `num_pages`,
    /// and recall needs a host query read-back + restricted page table between
    /// steps), so a recall-active slot skips the graph. See
    /// `docs/plans/2026-06-23-session-infinite-kv-memory.md`.
    kv_recall: bool,
    /// Recall budget regions (validated defaults): sink 32 + local 256 + top-k 8
    /// blocks of 32. The working set is bounded regardless of history length.
    recall_cfg: infer_core::RecallConfig,
    /// Per-slot resident mean-key reps + next-step recall page plan. Indexed by
    /// slot; only mutated when `kv_recall` is on.
    recall: Vec<crate::recall::CudaRecallState>,
    /// One-time non-BF16-KV-with-recall fallback log latch.
    recall_quant_warned: bool,
}

/// One slot's executor-side materialization watermark (see
/// [`QwenCudaExecutor::slot_progress`]).
#[derive(Clone, Copy)]
struct SlotProgress {
    /// Host-pool occupant epoch the watermark belongs to.
    epoch: u64,
    /// Tokens materialized (KV rows written or prefix-attached) for that epoch.
    len: usize,
}

impl Default for SlotProgress {
    fn default() -> Self {
        // u64::MAX never collides with a real host epoch (epochs start at 0 and
        // bump by 1-2 per occupancy), so the first real row always takes the
        // fresh-occupant branch.
        Self {
            epoch: u64::MAX,
            len: 0,
        }
    }
}

impl std::fmt::Debug for QwenCudaExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QwenCudaExecutor")
            .field("model", &self.model)
            .field("num_slots", &self.num_slots)
            .field("page_size", &self.kv.page_size)
            .field("max_total_pages", &self.kv.max_total_pages)
            .field("decode_graph", &self.decode_ctx.is_some())
            .field(
                "captured_decode_shapes",
                &self.graphs.as_ref().map_or(0, GraphBucket::len),
            )
            .finish()
    }
}

impl QwenCudaExecutor {
    /// `BackendExecutor::tp_sync_min` — see there for why the scheduler needs
    /// this (2026-07-05 TP=4 admission livelock). Dense Qwen3 has no existing
    /// `tp_min_usize` helper (unlike DSv4/Qwen3.6, which already use one for
    /// KV-budget clamping), so this inlines the same all-reduce.
    pub(crate) fn tp_sync_min(&self, local: usize) -> Result<usize> {
        let capped = i32::try_from(local.min(i32::MAX as usize)).unwrap_or(i32::MAX);
        self.model
            .tp
            .all_reduce_min_scalar_i32(&self.model.ctx, capped)
            .map(|v| v.max(0) as usize)
            .map_err(|e| anyhow::anyhow!("Qwen3 TP min-reduce admission free pages failed: {e}"))
    }

    /// `mem_fraction_static` (default 0.9): the dense shared paged pool is sized
    /// from MEASURED free VRAM after weights load (`infer_seam::profile_kv_pool_tokens`,
    /// SGLang-style), NOT the requested `total_pages`. `total_pages` becomes a
    /// minimum-capacity floor: the profiled pool is the larger of the two so the
    /// internal `total_pages` default never shrinks the pool below it, but a large
    /// card gets the extra capacity for more concurrency.
    pub(crate) fn from_qwen3_bf16_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
        kv_dtype: CudaKvCacheDtype,
        mem_fraction_static: f64,
    ) -> Result<Self> {
        ensure!(num_slots > 0, "CudaExecutor requires at least one slot");
        ensure!(
            total_pages > 0,
            "CudaExecutor requires at least one KV page"
        );

        let model = CudaModel::from_safetensors(model_path.as_ref())?;
        let kv_format = kv_dtype.kv_format();

        // Profile the shared paged pool from MEASURED free VRAM now that the
        // weights are resident (SGLang's mem_fraction_static). Per-token cell cost
        // = `budget_bytes_for_tokens(.., 1 token)` (storage + work bytes the pool
        // actually charges). On a successful read, size the pool to the larger of
        // the profiled token budget and the requested `total_pages` floor; if the
        // VRAM probe fails (no active context / driver error), fall back to the
        // requested `total_pages` exactly (byte-identical to before profiling).
        let cell_bytes_per_token = PagedKVPool::budget_bytes_for_tokens(
            model.config.num_hidden_layers,
            model.config.num_key_value_heads,
            model.config.head_dim,
            1,
            kv_format,
        ) as u64;
        let requested_pages = total_pages;
        let total_pages = match model.ctx.mem_info_bytes() {
            Ok((free, total)) => {
                let profiled_tokens = infer_seam::profile_kv_pool_tokens(
                    free as u64,
                    total as u64,
                    cell_bytes_per_token,
                    mem_fraction_static,
                );
                let profiled_pages = (profiled_tokens / SUPPORTED_PAGE_SIZE as u64) as usize;
                let sized = profiled_pages.max(requested_pages).max(1);
                log::info!(
                    "CUDA dense Qwen3 KV pool profiled from measured VRAM: free {}MB / total \
                     {}MB, mem_fraction_static {mem_fraction_static}, cell {cell_bytes_per_token}B/tok \
                     -> max_total_tokens {profiled_tokens} ({profiled_pages} pages); requested \
                     floor {requested_pages} pages -> sizing {sized} pages",
                    free >> 20,
                    total >> 20,
                );
                sized
            }
            Err(e) => {
                log::warn!(
                    "CUDA dense Qwen3 KV pool: free-VRAM probe failed ({e}); falling back to \
                     requested total_pages={requested_pages}"
                );
                requested_pages
            }
        };

        let token_budget = total_pages * SUPPORTED_PAGE_SIZE;
        let budget_bytes = PagedKVPool::budget_bytes_for_tokens(
            model.config.num_hidden_layers,
            model.config.num_key_value_heads,
            model.config.head_dim,
            token_budget,
            kv_format,
        );
        let kv = PagedKVPool::with_format(
            &model.ctx,
            model.config.num_hidden_layers,
            model.config.num_key_value_heads,
            model.config.head_dim,
            num_slots,
            budget_bytes,
            kv_format,
        )?;
        ensure!(
            kv.page_size == SUPPORTED_PAGE_SIZE,
            "R6 paged Qwen3 expects cuda-kernels page_size={SUPPORTED_PAGE_SIZE}, got {}",
            kv.page_size
        );

        let slot_progress = vec![SlotProgress::default(); num_slots];
        let tier = CudaKvTierStore::with_budget(
            default_t1_budget_bytes(DEFAULT_DRAM_FRACTION),
            kv.storage_bytes_per_page(),
        );
        let recall = (0..num_slots)
            .map(|_| crate::recall::CudaRecallState::default())
            .collect();
        Ok(Self {
            model,
            kv,
            tier,
            num_slots,
            slot_progress,
            decode_ctx: None,
            graphs: None,
            kv_recall: false,
            recall_cfg: crate::recall::default_recall_config(),
            recall,
            recall_quant_warned: false,
        })
    }

    pub(crate) fn kv_tier_capacity_pages(&self) -> usize {
        self.tier.capacity_pages()
    }

    pub(crate) fn kv_tier_page_bytes(&self) -> usize {
        self.tier.page_bytes()
    }

    pub(crate) fn kv_tier_host_demoted_pages(&self) -> usize {
        self.tier.host_demoted_pages()
    }

    pub(crate) fn kv_tier_disk_pages(&self) -> usize {
        self.tier.disk_pages()
    }

    pub(crate) fn kv_tier_location(&self, key: u64) -> Option<infer_seam::KvTierLocation> {
        self.tier.location(key)
    }

    pub(crate) fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
        pages_only_reusable_prefix_blocks(blocks, |key| self.tier.contains(key))
    }

    /// Re-budget the host-demoted tier store (`0` disables). Pre-serve only: any
    /// existing entries are dropped, so callers configure this right after
    /// construction, before the engine demotes anything.
    pub(crate) fn set_kv_tier_budget_bytes(&mut self, bytes: usize) {
        self.tier = CudaKvTierStore::with_budget(bytes, self.kv.storage_bytes_per_page());
    }

    /// Opt into session KV-recall (`--kv-recall`, default off). Mirrors the Metal
    /// `set_kv_recall`: a post-construction setter so the constructor signature
    /// stays stable. With recall off the decode hot path is unchanged
    /// (byte-identical baseline — CUDA is the Stable backend).
    pub(crate) fn set_kv_recall(&mut self, enabled: bool) {
        self.kv_recall = enabled;
    }

    /// Attach the opt-in disk spill level (`--kv-disk`). Pre-serve only.
    pub(crate) fn set_kv_tier_disk(
        &mut self,
        root: std::path::PathBuf,
        budget_bytes: usize,
    ) -> bool {
        self.tier
            .set_disk(root, budget_bytes, self.kv.storage_bytes_per_page())
    }

    /// Copy device pages into the host tier store (synchronous: the copy is
    /// complete when this returns, so the engine may free the pages). Stops at
    /// capacity and reports the accepted prefix length.
    pub(crate) fn demote_prefix_pages(&mut self, entries: &[(u32, u64)]) -> Result<usize> {
        let mut accepted = 0usize;
        for &(page, key) in entries {
            if self.tier.is_full() {
                break;
            }
            let payload = self.kv.copy_pages_to_host(&self.model.ctx, &[page])?;
            if !self.tier.insert(key, payload) {
                break;
            }
            accepted += 1;
        }
        Ok(accepted)
    }

    /// Copy host tier entries back into freshly allocated device pages. The
    /// engine attaches the pages right after, so sync before returning.
    pub(crate) fn promote_prefix_pages(&mut self, entries: &[(u64, u32)]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let page_bytes = self.kv.storage_bytes_per_page();
        let mut buf = Vec::with_capacity(entries.len() * page_bytes);
        let mut pages: Vec<u32> = Vec::with_capacity(entries.len());
        for &(key, page) in entries {
            let payload = self
                .tier
                .read(key)
                .map_err(|err| anyhow::anyhow!("KV tier promote: {err}"))?;
            pages.push(page);
            buf.extend_from_slice(&payload);
        }
        self.kv
            .copy_pages_from_host_on_copy_stream(&self.model.ctx, &pages, &buf)?;
        self.model.ctx.sync_copy()?;
        Ok(())
    }

    pub(crate) fn drop_kv_tier_entries(&mut self, keys: &[u64]) {
        self.tier.remove(keys);
    }

    /// **Write-through**: mirror a filled device `page` into the host tier under
    /// `key`, so a later evict-drop of that page is free (the tier keeps the
    /// source of truth). Reuses the same `CudaKvTierStore` as the prefix tier —
    /// there is ONE session-keyed store (R5), not a parallel one. `key` is a
    /// `(session, block)` pair flattened to the store's `u64` namespace by
    /// [`tier_block_u64`].
    ///
    /// Synchronous today (the copy is complete on return, matching
    /// `demote_prefix_pages`); R4's side-stream async mirror is the remaining perf
    /// step — see the pending-remote wins entry. Returns `false` (page not
    /// mirrored, MUST NOT be evict-dropped) when the tier is full.
    pub(crate) fn write_through(&mut self, key: u64, page: u32) -> Result<bool> {
        if self.tier.is_full() {
            return Ok(false);
        }
        let payload = self.kv.copy_pages_to_host(&self.model.ctx, &[page])?;
        Ok(self.tier.insert(key, payload))
    }

    /// **Prefetch**: load blocks from the host tier back into freshly allocated
    /// device pages (`(key, page)`), complete on return. Identical transport to
    /// `promote_prefix_pages`; the difference is the entry point
    /// (relevance-prefetch at prefill vs prefix-hit promote), per R5.
    #[allow(dead_code)] // WIP: R5 relevance-prefetch entry point, not yet wired
    pub(crate) fn prefetch_pages(&mut self, entries: &[(u64, u32)]) -> Result<()> {
        self.promote_prefix_pages(entries)
    }

    pub(crate) fn submit(
        &mut self,
        plan: &ForwardPlan,
        host_kv: &mut dyn KvPool,
        kv_batch: &KvBatchDescriptor,
    ) -> Result<StepOutput> {
        ensure!(
            host_kv.page_size() == SUPPORTED_PAGE_SIZE,
            "host CudaKvPool page_size={} does not match CUDA BF16 page_size={SUPPORTED_PAGE_SIZE}",
            host_kv.page_size()
        );

        let rows = plan.decode_rows.len() + plan.prefill_rows.len();
        if rows == 0 {
            return Ok(StepOutput { tokens: Vec::new() });
        }
        // Multi-row plans (continuous batching): N prefill rows + M decode rows,
        // total > 1. Includes the MIXED case (a new request prefilling while
        // another decodes). Prefill rows run sequentially as single-row sub-steps,
        // then the M decode rows run as ONE batch. total == 1 falls through to the
        // single-row fast path below (byte-identical, incl. captured-graph/recall).
        if rows > 1 {
            return self.submit_multi_row(plan, host_kv, kv_batch);
        }
        ensure!(
            kv_batch.rows.len() == 1,
            "KV batch descriptor carries {} rows for a single-row plan",
            kv_batch.rows.len()
        );
        let kv_row = &kv_batch.rows[0];
        // Host page table covering [0, append_end) for this row — the engine
        // already allocated the append span in the host pool.
        let pages = &kv_batch.flat_page_ids[kv_row.page_range.clone()];

        let token = if let Some(row) = plan.prefill_rows.first() {
            ensure!(
                row.slot < self.num_slots,
                "prefill slot {} outside CUDA executor slots {}",
                row.slot,
                self.num_slots
            );
            ensure!(!row.tokens.is_empty(), "prefill row must carry tokens");
            let expected_len = row.start_pos + row.tokens.len();
            ensure!(
                kv_row.slot == row.slot
                    && kv_row.append_pos == row.start_pos
                    && kv_row.append_len == row.tokens.len(),
                "KV batch row (slot {}, append {}+{}) does not match prefill row (slot {}, {}+{})",
                kv_row.slot,
                kv_row.append_pos,
                kv_row.append_len,
                row.slot,
                row.start_pos,
                row.tokens.len()
            );
            self.advance_slot_progress(row.slot, kv_row.slot_epoch, row.start_pos, expected_len)?;
            self.kv.mirror_slot(row.slot, pages, expected_len)?;
            let position = expected_len as u64;
            self.model.forward_tokens(
                row.slot,
                &row.tokens,
                row.start_pos,
                &mut self.kv,
                &row.params,
                position,
            )?
        } else {
            let row = &plan.decode_rows[0];
            ensure!(
                row.slot < self.num_slots,
                "decode slot {} outside CUDA executor slots {}",
                row.slot,
                self.num_slots
            );
            ensure!(
                kv_row.slot == row.slot
                    && kv_row.append_pos == row.kv_seq_len
                    && kv_row.append_len == 1,
                "KV batch row (slot {}, append {}+{}) does not match decode row (slot {}, kv_seq_len {})",
                kv_row.slot,
                kv_row.append_pos,
                kv_row.append_len,
                row.slot,
                row.kv_seq_len
            );
            self.advance_slot_progress(
                row.slot,
                kv_row.slot_epoch,
                row.kv_seq_len,
                row.kv_seq_len + 1,
            )?;
            self.kv.mirror_slot(row.slot, pages, row.kv_seq_len + 1)?;
            let position = row.kv_seq_len.saturating_add(1) as u64;
            // Session KV-recall (eager-only, BF16-only): when on and this slot has
            // an active recall plan, attend the restricted page table + rescore
            // for the next step, then evict-drop the non-working-set middle pages
            // out of HBM (write-through tiered KV — the flat-VRAM win). Off / no
            // plan → the byte-identical default below.
            if let Some(token) = self.try_recall_decode(row, position, host_kv)? {
                token
            } else {
                // Try the captured graph; on any miss fall back to the eager path.
                match self.try_captured_decode(row.slot, row.last_token, row.kv_seq_len)? {
                    Some(()) => self.sample_decode_logits(&row.params, position)?,
                    None => self.model.forward_tokens(
                        row.slot,
                        &[row.last_token],
                        row.kv_seq_len,
                        &mut self.kv,
                        &row.params,
                        position,
                    )?,
                }
            }
        };

        Ok(StepOutput {
            tokens: vec![SlotToken {
                slot: plan
                    .prefill_rows
                    .first()
                    .map(|r| r.slot)
                    .unwrap_or_else(|| plan.decode_rows[0].slot),
                token,
                logprob: None,
                finish: None,
            }],
        })
    }

    /// Multi-row plan (continuous batching), total rows > 1. Handles pure-decode
    /// (N=0, M>1) and MIXED (N>=1 prefill + M>=0 decode) alike: the M decode
    /// rows run as ONE batch FIRST, then prefill rows run SEQUENTIALLY as
    /// single-row sub-steps. Decode-first minimises TTFT and ITL for in-flight
    /// requests — prefill sub-steps are expensive and must not stall decode.
    ///
    /// Plan rows always address disjoint slots (a request is either Prefilling or
    /// Decoding), so execution order is irrelevant for KV correctness. The
    /// KvBatchDescriptor layout is unchanged (prefill rows at 0..n_prefill, decode
    /// rows at n_prefill..total); only the GPU submission order is swapped.
    fn submit_multi_row(
        &mut self,
        plan: &ForwardPlan,
        host_kv: &mut dyn KvPool,
        kv_batch: &KvBatchDescriptor,
    ) -> Result<StepOutput> {
        let rows = plan.decode_rows.len() + plan.prefill_rows.len();
        ensure!(
            kv_batch.rows.len() == rows,
            "KV batch descriptor has {} rows for a {rows}-row plan",
            kv_batch.rows.len()
        );
        let mut seen_slots = std::collections::BTreeSet::new();
        for slot in plan
            .prefill_rows
            .iter()
            .map(|row| row.slot)
            .chain(plan.decode_rows.iter().map(|row| row.slot))
        {
            ensure!(
                seen_slots.insert(slot),
                "CUDA plan schedules slot {slot} more than once per tick"
            );
        }

        let n_prefill = plan.prefill_rows.len();
        let mut tokens = Vec::with_capacity(rows);
        // Decode first: active decode requests get their next token before any
        // prefill sub-step runs. Slots are disjoint so KV order is irrelevant.
        if !plan.decode_rows.is_empty() {
            let sub_batch = kv_batch.subset(n_prefill..kv_batch.rows.len())?;
            tokens.extend(self.submit_decode_batch(&plan.decode_rows, host_kv, &sub_batch)?);
        }
        for (idx, row) in plan.prefill_rows.iter().enumerate() {
            let sub_batch = kv_batch.subset(idx..idx + 1)?;
            tokens.push(self.submit_prefill_row(row, &sub_batch)?);
        }
        Ok(StepOutput { tokens })
    }

    /// One prefill row as its own single-row sub-step (mirror + forward_tokens).
    /// `kv_batch` is the row's single-row (sub-)descriptor — indistinguishable from
    /// what a prefill-only tick delivers. Byte-identical to the single-row prefill
    /// arm of [`Self::submit`].
    fn submit_prefill_row(
        &mut self,
        row: &infer_plan::PrefillRow,
        kv_batch: &KvBatchDescriptor,
    ) -> Result<SlotToken> {
        ensure!(
            kv_batch.rows.len() == 1,
            "prefill sub-batch carries {} rows (expected 1)",
            kv_batch.rows.len()
        );
        let kv_row = &kv_batch.rows[0];
        let pages = &kv_batch.flat_page_ids[kv_row.page_range.clone()];
        ensure!(
            row.slot < self.num_slots,
            "prefill slot {} outside CUDA executor slots {}",
            row.slot,
            self.num_slots
        );
        ensure!(!row.tokens.is_empty(), "prefill row must carry tokens");
        let expected_len = row.start_pos + row.tokens.len();
        ensure!(
            kv_row.slot == row.slot
                && kv_row.append_pos == row.start_pos
                && kv_row.append_len == row.tokens.len(),
            "KV batch row (slot {}, append {}+{}) does not match prefill row (slot {}, {}+{})",
            kv_row.slot,
            kv_row.append_pos,
            kv_row.append_len,
            row.slot,
            row.start_pos,
            row.tokens.len()
        );
        self.advance_slot_progress(row.slot, kv_row.slot_epoch, row.start_pos, expected_len)?;
        self.kv.mirror_slot(row.slot, pages, expected_len)?;
        let position = expected_len as u64;
        let token = self.model.forward_tokens(
            row.slot,
            &row.tokens,
            row.start_pos,
            &mut self.kv,
            &row.params,
            position,
        )?;
        Ok(SlotToken {
            slot: row.slot,
            token,
            logprob: None,
            finish: None,
        })
    }

    /// Run `decode_rows` (M >= 1) as ONE batched decode. `kv_batch` is the
    /// decode-only (sub-)descriptor whose rows align by index with `decode_rows`.
    ///
    /// BF16 + single-GPU + recall-inactive → the true batched path (one prep +
    /// one batched paged attention over `(M, M, 1)`, MLP/norm batched over M rows,
    /// per-row sample). Any of TP-collective / quant KV / active recall falls back
    /// to per-row sequential decode (the single-row machinery, correctness floor)
    /// rather than crash. Output order matches `decode_rows`.
    fn submit_decode_batch(
        &mut self,
        decode_rows: &[DecodeRow],
        host_kv: &mut dyn KvPool,
        kv_batch: &KvBatchDescriptor,
    ) -> Result<Vec<SlotToken>> {
        let batch = decode_rows.len();
        ensure!(
            kv_batch.rows.len() == batch,
            "KV batch descriptor carries {} rows for a {}-row decode plan",
            kv_batch.rows.len(),
            batch
        );

        // Per-row validate + mirror is shared by both lanes (the device KV pool
        // must hold every row's just-appended token before any forward reads it).
        for (row, kv_row) in decode_rows.iter().zip(&kv_batch.rows) {
            ensure!(
                row.slot < self.num_slots,
                "decode slot {} outside CUDA executor slots {}",
                row.slot,
                self.num_slots
            );
            ensure!(
                kv_row.slot == row.slot
                    && kv_row.append_pos == row.kv_seq_len
                    && kv_row.append_len == 1,
                "KV batch row (slot {}, append {}+{}) does not match decode row (slot {}, kv_seq_len {})",
                kv_row.slot,
                kv_row.append_pos,
                kv_row.append_len,
                row.slot,
                row.kv_seq_len
            );
            let pages = &kv_batch.flat_page_ids[kv_row.page_range.clone()];
            self.advance_slot_progress(
                row.slot,
                kv_row.slot_epoch,
                row.kv_seq_len,
                row.kv_seq_len + 1,
            )?;
            self.kv.mirror_slot(row.slot, pages, row.kv_seq_len + 1)?;
        }

        // Correctness floor: the batched paged kernels are BF16 single-GPU only,
        // and recall needs the per-row restricted page table the batched meta does
        // not carry. Any of these → per-row sequential decode.
        let recall_on = self.kv_recall && self.kv.format == KVFormat::BF16;
        let can_batch =
            self.kv.format == KVFormat::BF16 && !self.model.tp.is_collective() && !recall_on;
        if !can_batch {
            let mut tokens = Vec::with_capacity(batch);
            for row in decode_rows {
                let position = row.kv_seq_len.saturating_add(1) as u64;
                // KV already mirrored above; run the per-row forward. Recall reuses
                // its restricted-table path when active.
                let token = if let Some(token) = self.try_recall_decode(row, position, host_kv)? {
                    token
                } else {
                    self.model.forward_tokens(
                        row.slot,
                        &[row.last_token],
                        row.kv_seq_len,
                        &mut self.kv,
                        &row.params,
                        position,
                    )?
                };
                tokens.push(SlotToken {
                    slot: row.slot,
                    token,
                    logprob: None,
                    finish: None,
                });
            }
            return Ok(tokens);
        }

        // True batched decode. Build the M-row page table from the freshly mirrored
        // slots, run one forward, sample M tokens (one per row's params/position).
        let last_tokens: Vec<u32> = decode_rows.iter().map(|r| r.last_token).collect();
        let params: Vec<SamplingParams> = decode_rows.iter().map(|r| r.params.clone()).collect();
        let positions: Vec<u64> = decode_rows
            .iter()
            .map(|r| r.kv_seq_len.saturating_add(1) as u64)
            .collect();
        let batch_rows: Vec<(usize, usize)> = decode_rows
            .iter()
            .map(|r| (r.slot, r.kv_seq_len + 1))
            .collect();
        let meta = PageMeta::for_decode_batch(&self.model.ctx, &self.kv, &batch_rows)?;
        let out_tokens = self.model.forward_decode_batch(
            &last_tokens,
            &mut self.kv,
            &meta,
            &params,
            &positions,
        )?;
        ensure!(
            out_tokens.len() == batch,
            "batched decode returned {} tokens for {batch} rows",
            out_tokens.len()
        );
        Ok(decode_rows
            .iter()
            .zip(out_tokens)
            .map(|(row, token)| SlotToken {
                slot: row.slot,
                token,
                logprob: None,
                finish: None,
            })
            .collect())
    }

    /// Warmup the B=1 decode graph before serving.
    ///
    /// Captures the smallest shape (`num_pages = 1`) so the machinery is proven
    /// before the first request; later page counts capture lazily on first decode
    /// (capture-once, replay-after, per key). Opt-in via
    /// `--cuda-graph`; any capture failure downgrades to eager-only
    /// (never fatal — eager is the correctness floor).
    pub(crate) fn warmup(&mut self) -> Result<()> {
        // NCCL all-reduce is not graph-capturable, so multi-rank TP stays eager
        // (a captured graph would silently skip the collective → wrong logits).
        // CUDA-graph decode is a single-GPU optimization.
        if self.model.tp.is_collective() {
            info!(
                "CUDA decode graph disabled under tensor parallelism \
                 (world_size>1, NCCL collectives are not graph-capturable); \
                 using eager forward"
            );
            return Ok(());
        }
        if self.kv.format != KVFormat::BF16 {
            info!(
                "CUDA decode graph disabled for quantized KV (format {:?}); decode runs eager \
                 through the fused-dequant kernels (#68 T3 V1)",
                self.kv.format
            );
            return Ok(());
        }
        if !decode_graph_enabled() {
            info!("CUDA decode graph disabled (set --cuda-graph to enable)");
            return Ok(());
        }
        if self.decode_ctx.is_some() {
            return Ok(()); // idempotent
        }
        match self.build_and_capture_warmup() {
            Ok(()) => {
                info!("CUDA B=1 decode graph captured (warmup, num_pages=1)");
                Ok(())
            }
            Err(e) => {
                warn!("CUDA decode graph warmup capture failed, falling back to eager: {e}");
                // Drop any half-built state so submit stays on the eager path.
                self.decode_ctx = None;
                self.graphs = None;
                Ok(())
            }
        }
    }

    /// Allocate the fixed buffers + capture `num_pages = 1` on a dummy slot.
    /// Isolated so a capture failure leaves the executor eager-only.
    fn build_and_capture_warmup(&mut self) -> Result<()> {
        let cfg = &self.model.config;
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let ctx = DecodeGraphContext::new(
            &self.model.ctx,
            cfg.hidden_size,
            q_dim,
            kv_dim,
            cfg.intermediate_size,
            self.model.logits_dim(),
            self.kv.page_size,
            DECODE_GRAPH_MAX_SEQ_LEN,
        )?;
        self.decode_ctx = Some(ctx);
        self.graphs = Some(GraphBucket::new(self.model.ctx.stream.clone()));

        // Mirror dummy slot 0 onto page 0 for a single token so the page table
        // is valid for num_pages = 1, capture, then clear the view so serving
        // starts clean. The capture writes throwaway KV into page 0's rows; the
        // first real occupant of that page overwrites them. The device
        // allocator is never used — the host pool is the single allocator and
        // serving rows arrive via `mirror_slot`.
        ensure!(self.num_slots > 0, "warmup needs at least one slot");
        let dummy_slot = 0usize;
        self.kv.mirror_slot(dummy_slot, &[0], 1)?;
        // Two passes: `CudaGraphState` warms one eager run before its first
        // capture (universal JIT-in-capture guard), so a single call here
        // would only warm and silently push the capture onto the first real
        // request. First call = eager warm (also flushes lazy module loads
        // outside capture), second = the boot-time capture this warmup
        // promises.
        let capture_result = self
            .capture_decode_for_current_state(dummy_slot, 0, 0)
            .and_then(|()| self.capture_decode_for_current_state(dummy_slot, 0, 0));
        self.kv.mirror_slot(dummy_slot, &[], 0)?;
        capture_result?;
        self.model.ctx.sync()?;
        Ok(())
    }

    /// Session KV-recall decode step (#3/#4/#5), eager-only and BF16-only.
    ///
    /// Returns `Ok(Some(token))` when recall handled the step (restricted page
    /// table + rescore for the next step), `Ok(None)` to fall through to the
    /// default graph/eager decode. No-op (`None`) unless `--kv-recall` is on AND
    /// the session has exceeded the working-set budget — below the budget recall
    /// is a strict no-op, so the default path stays byte-identical. Non-BF16 KV
    /// falls back (logged once); recall is BF16-gated.
    fn try_recall_decode(
        &mut self,
        row: &DecodeRow,
        position: u64,
        host_kv: &mut dyn KvPool,
    ) -> Result<Option<u32>> {
        if !self.kv_recall {
            return Ok(None);
        }
        if self.kv.format != KVFormat::BF16 {
            if !self.recall_quant_warned {
                warn!(
                    "--kv-recall requested with a {:?} KV pool; recall is BF16-only — \
                     falling back to full attention (use --kv-cache-dtype bf16 to enable recall)",
                    self.kv.format
                );
                self.recall_quant_warned = true;
            }
            return Ok(None);
        }
        let cfg = self.recall_cfg;
        let cache_len = row.kv_seq_len + 1; // includes this step's appended token
        // Below the working-set budget recall is a strict no-op (mirrors
        // `plan_recall` returning the full contiguous range) → default path.
        if cache_len <= cfg.working_set_tokens() {
            return Ok(None);
        }

        // Correctness invariant for the restricted page table: the decode kernel
        // treats every page EXCEPT the last in `kv_indices` as FULL (`page_size`
        // tokens) and only the last as partial (`kv_last_page_len`). With all the
        // recall region boundaries (`n_init`, `l_bs`, `n_local`) multiples of
        // `page_size`, every range start/end is page-aligned except the final
        // local-window end (= `cache_len`), whose last page IS the current partial
        // page — so the only partial page is the last selected one, matching the
        // kernel. A future config that breaks this alignment would silently
        // mis-attend, so fail loud here.
        let ps = self.kv.page_size;
        ensure!(
            cfg.n_init.is_multiple_of(ps)
                && cfg.n_local.is_multiple_of(ps)
                && cfg.l_bs.is_multiple_of(ps),
            "KV-recall config (n_init {}, n_local {}, l_bs {}) must be multiples of \
             the KV page_size {} so the restricted page table has only its LAST page partial",
            cfg.n_init,
            cfg.n_local,
            cfg.l_bs,
            ps
        );

        // Page list for this step's attention: the slot's recall plan from the
        // previous step (stale-Q), or the full page list on the first recall step
        // (no plan yet) so the forward is still correct while we seed scoring.
        let recall_pages: Vec<u32> = match self.recall.get(row.slot).and_then(|s| s.recall_pages())
        {
            Some(p) => p.to_vec(),
            None => {
                let num_pages = cache_len.div_ceil(self.kv.page_size);
                self.kv.page_indices(row.slot)[..num_pages].to_vec()
            }
        };
        let recall_meta =
            PageMeta::for_recall_decode(&self.model.ctx, &self.kv, cache_len, &recall_pages)?;
        let (token, layer0_query) = self.model.forward_decode_recall(
            row.last_token,
            &mut self.kv,
            &recall_meta,
            &row.params,
            position,
        )?;

        // Score this step's query against the resident reps and plan the NEXT
        // step's recall (stale-Q, licensed).
        let num_q_heads = self.model.local_q_heads;
        let num_kv_heads = self.model.local_kv_heads;
        let head_dim = self.model.config.head_dim;
        let evict_pages = if let Some(state) = self.recall.get_mut(row.slot) {
            state.recompute_recall_plan(
                &self.model.ctx,
                &self.kv,
                row.slot,
                cache_len,
                &cfg,
                num_q_heads,
                num_kv_heads,
                head_dim,
                &layer0_query,
                // Dense arm has no L3 recall tier wired into the decode path:
                // evicted blocks stay -inf (no mid-decode re-recall). Byte-identical
                // to the prior behavior.
                /* allow_prefetch = */
                false,
            )?;
            state.take_evict_pages()
        } else {
            Vec::new()
        };

        // Write-through evict-drop (THE flat-VRAM win): free the cold middle pages
        // outside the working set out of HBM. For each, mirror it to the tier
        // (write_through, async D2H — the page already has a durable copy after
        // this) and only then return the physical page to BOTH the device pool and
        // the host single-allocator. The logical page table keeps its length (an
        // evict sentinel marks the slot), so `mirror_slot`/`SlotProgress` stay
        // valid. If the tier is full we skip that page (never lose KV).
        self.evict_drop_recall_pages(row.slot, &evict_pages, host_kv)?;

        Ok(Some(token))
    }

    /// Free the given logical pages of `slot` out of HBM (write-through tiered KV).
    ///
    /// Reused by [`Self::try_recall_decode`]. For each logical page: read its
    /// physical id, mirror it to the tier (so the drop is free), then evict-drop it
    /// from the device pool (`PagedKVPool::evict_slot_page` → recycles the HBM page)
    /// and the host single-allocator (`KvAllocator::evict_slot_page` → returns it to
    /// the free stack). Both pools' logical page tables stay the same length (an
    /// `EVICTED_PAGE` sentinel marks the freed slot), keeping the slot sparse
    /// without breaking the `mirror_slot` page-count or `SlotProgress` contiguity
    /// contracts. A tier-full `write_through` skips that page (KV must not be lost).
    fn evict_drop_recall_pages(
        &mut self,
        slot: usize,
        evict_pages: &[usize],
        host_kv: &mut dyn KvPool,
    ) -> Result<()> {
        for &logical in evict_pages {
            let table = self.kv.page_indices(slot);
            let Some(&physical) = table.get(logical) else {
                continue;
            };
            if physical == cuda_kernels::prelude::EVICTED_PAGE {
                continue; // already freed
            }
            // Mirror to the tier first; only drop if it took a durable copy.
            let key = tier_block_u64(slot as u64, logical as u64);
            if !self.write_through(key, physical)? {
                continue; // tier full → keep page resident (no KV loss)
            }
            // Free the physical page from BOTH pools (the real free). Host first so
            // the device pool's `page_indices` (read above) is still valid for the
            // device call; both replace the logical slot with the evict sentinel.
            let host_freed = host_kv.evict_slot_page(slot, logical);
            let dev_freed = self.kv.evict_slot_page(slot, logical);
            debug_assert_eq!(
                host_freed, dev_freed,
                "host/device pools disagreed on evicted page for slot {slot} logical {logical}"
            );
        }
        Ok(())
    }

    /// Write Stage-1 metadata and replay (or lazily capture) the decode graph for
    /// this step's page count. Returns `Ok(Some(()))` when the graph wrote
    /// `decode_ctx.logits` (sample from there), `Ok(None)` for eager fallback
    /// (disabled, or page count over budget). The step's KV token must already be
    /// allocated.
    fn try_captured_decode(
        &mut self,
        slot: usize,
        token: u32,
        kv_seq_len: usize,
    ) -> Result<Option<()>> {
        if self.decode_ctx.is_none() || self.graphs.is_none() {
            return Ok(None);
        }
        // Page count over budget → eager (never a stale replay).
        let total_len = kv_seq_len + 1;
        let num_pages = total_len.div_ceil(self.kv.page_size);
        if num_pages > DECODE_GRAPH_MAX_SEQ_LEN.div_ceil(self.kv.page_size) {
            return Ok(None);
        }
        self.capture_decode_for_current_state(slot, token, kv_seq_len)?;
        Ok(Some(()))
    }

    /// Stage-1 write + `run_or_capture` for the current decode state. Captures on
    /// first sight of a `num_pages` shape, replays after; a new page count gets
    /// its own graph entry.
    fn capture_decode_for_current_state(
        &mut self,
        slot: usize,
        token: u32,
        kv_seq_len: usize,
    ) -> Result<()> {
        // Split borrows: stage1 needs &kv + &mut decode_ctx; replay needs &model +
        // &mut kv + &mut decode_ctx + &mut graphs.
        let key: DecodeGraphKey = {
            let decode_ctx = self
                .decode_ctx
                .as_mut()
                .expect("decode_ctx present (checked by caller / warmup)");
            decode_ctx.stage1_write(&self.model.ctx, &self.kv, slot, token, kv_seq_len)?
        };
        debug_assert_eq!(key.batch_size, DECODE_GRAPH_BATCH);

        let model = &self.model;
        let kv = &mut self.kv;
        let decode_ctx = self
            .decode_ctx
            .as_mut()
            .expect("decode_ctx present (checked by caller / warmup)");
        let graphs = self
            .graphs
            .as_mut()
            .expect("graphs present (checked by caller / warmup)");
        let graph = graphs.entry(key.num_pages);
        graph.state.run_or_capture(|| {
            model.forward_decode_captured(kv, decode_ctx)?;
            Ok(())
        })
    }

    /// Sample the next token from the captured graph's fixed logits buffer.
    /// Sampling stays outside the graph (replay ends at `decode_ctx.logits`).
    fn sample_decode_logits(&self, params: &SamplingParams, position: u64) -> Result<u32> {
        let decode_ctx = self
            .decode_ctx
            .as_ref()
            .expect("decode_ctx present when sampling captured logits");
        sample_cuda_token(&self.model.ctx, &decode_ctx.logits, params, position)
    }

    /// Loud-error continuity guard replacing the old device-allocator
    /// materialized checks (see [`QwenCudaExecutor::slot_progress`]).
    ///
    /// Same occupant epoch ⇒ this executor must already have materialized
    /// exactly `append_pos` tokens (chunked prefill and decode append
    /// contiguously). A new epoch is a fresh occupant: `append_pos == 0` is a
    /// normal fresh prefill; `append_pos > 0` means the engine attached a radix
    /// prefix whose retained pages still hold the publishing request's KV rows
    /// (the host pool never recycled them), so the occupant starts materialized
    /// at that length.
    fn advance_slot_progress(
        &mut self,
        slot: usize,
        epoch: u64,
        append_pos: usize,
        end: usize,
    ) -> Result<()> {
        let progress = &mut self.slot_progress[slot];
        if progress.epoch == epoch {
            ensure!(
                progress.len == append_pos,
                "CUDA slot {slot} epoch {epoch} materialized {} tokens but the plan resumes at {append_pos} (non-contiguous append)",
                progress.len
            );
        } else if self.kv_recall {
            // New occupant: drop the prior session's recall reps + plan so the
            // fresh request starts from the byte-identical default page table.
            if let Some(state) = self.recall.get_mut(slot) {
                state.reset();
            }
        }
        self.slot_progress[slot] = SlotProgress { epoch, len: end };
        Ok(())
    }
}

/// DSv4-Flash executor: drives [`crate::dsv4::Dsv4Model::forward_tokens`].
/// Prefill/mixed still run one scheduled row. Pure decode uses B=1 as the
/// single-row reference and B>1 as the canonical layer-major batched lane for
/// MODEL1 FlashMLA decode. DSv4 owns its MLA KV state inside the forward (bf16
/// SW rings + compressor pending/compressed pools), so it does NOT use a
/// [`PagedKVPool`]. The decode graph is disabled (MLA host-routing per step).
pub(crate) struct Dsv4CudaExecutor {
    model: crate::dsv4::Dsv4Model,
    slots: Vec<crate::dsv4::Dsv4SlotState>,
    kv_adapter: crate::attention::Dsv4KvAdapter,
    spec_slots: Vec<Dsv4SpecSlotState>,
    /// `Some(n)` = config-driven MTP spec decode on (draft depth `n`, from the
    /// serve path's `--spec-type mtp`); `None` falls back to the
    /// `ARLE_DSV4_SPEC_DECODE` env gate at each spec branch.
    spec_draft_tokens: Option<usize>,
    /// MTP draft candidate width. `None`/`Some(1)` keeps chain-only candidates;
    /// `Some(k>1)` widens the draft matrix while verifier rows stay chain-shaped.
    spec_draft_topk: Option<usize>,
    num_slots: usize,
    mtp_accepts: usize,
    mtp_rejects: usize,
    /// Adaptive MTP gate (B=1): EMA of per-step accepted/depth, plus the count of
    /// consecutive gated skips since the last real spec step (drives the periodic
    /// probe). MTP only beats no-spec when it emits > t_mtp/t_nospec tok/step, so
    /// below that acceptance the gate runs a warm no-spec step instead. Init
    /// optimistic (1.0) so MTP runs until the running acceptance proves it loses.
    /// See `mtp_should_speculate`. Opt-in via `--mtp-adaptive` (bring-up;
    /// promote to a `--mtp-adaptive` CLI flag once pod-calibrated).
    mtp_accept_ema: f32,
    mtp_skip_streak: usize,
    /// THE tier store (L2 DRAM + optional L3 NVMe): every demoted blob — parked
    /// slots — lives here as 16 MiB chunks under a per-key manifest,
    /// partitioned by `NS_*` key namespaces.
    slot_tier: CudaKvTierStore,
    /// Content-keyed cross-request prefix-state pool (#154 Phase 2): one
    /// host-resident entry per completed host page, written from the
    /// post-forward choke point, consumed by the prefix restore path.
    prefix_state: crate::attention::Dsv4PrefixStatePool,
}

/// `slot_tier` key namespaces (top byte, see `kv_tier::tier_key`), so features
/// sharing THE store never collide and a future kind (e.g. a suffix cache) is
/// one new constant.
/// Parked whole-slot images (key = engine-minted swap key).
const NS_SLOT: u64 = 1;
const NS_SLOT_CHUNK: u64 = 2;
/// Qwen3.5/3.6 recurrent prefix sidecars (key = token-prefix hash). Disjoint
/// namespace from `NS_SLOT`, so the Qwen3.6 arm's ONE `slot_tier` holds both the
/// whole-slot spill images and the page-radix sidecars without key aliasing.
const NS_SIDECAR: u64 = 3;
const NS_SIDECAR_CHUNK: u64 = 4;

/// One demoted slot: the device-state image plus the executor-level MTP spec
/// chain. `spec_pending`/`spec_hidden` MUST ride along (not reset): under
/// `--spec-type mtp` the resumed decode hard-requires the pending token and
/// the previous MTP stream (`forward_decode_tokens` errors on a missing
/// pending), and the slot's spec state is overwritten by whichever request
/// occupies the slot while this one is demoted.
#[derive(Default)]
struct Dsv4SpecSlotState {
    pending: Option<u32>,
    hidden: Option<DeviceVec>,
}

#[derive(Clone)]
struct Dsv4DecodeBatchRow {
    slot: usize,
    last_token: u32,
    start_pos: usize,
    position: u64,
    params: SamplingParams,
}

struct Dsv4DecodeBatch {
    slot_ids: Vec<usize>,
    tokens: Vec<u32>,
    start_positions: Vec<usize>,
    positions: Vec<u64>,
    rows: Vec<Dsv4DecodeBatchRow>,
}

impl Dsv4DecodeBatch {
    fn from_rows(
        rows: &[DecodeRow],
        slots: &[crate::dsv4::Dsv4SlotState],
        num_slots: usize,
    ) -> Result<Self> {
        let mut seen = vec![false; num_slots];
        let mut slot_ids = Vec::with_capacity(rows.len());
        let mut tokens = Vec::with_capacity(rows.len());
        let mut start_positions = Vec::with_capacity(rows.len());
        let mut positions = Vec::with_capacity(rows.len());
        let mut batch_rows = Vec::with_capacity(rows.len());
        for row in rows {
            ensure!(
                row.slot < num_slots,
                "decode slot {} outside DSv4 executor slots {}",
                row.slot,
                num_slots
            );
            ensure!(
                !seen[row.slot],
                "DSv4 decode batch contains duplicate slot {}",
                row.slot
            );
            seen[row.slot] = true;
            ensure!(
                slots[row.slot].seq_len() == row.kv_seq_len,
                "DSv4 materialized state len {} != DecodeRow.kv_seq_len {} for slot {}",
                slots[row.slot].seq_len(),
                row.kv_seq_len,
                row.slot
            );
            let position = row.kv_seq_len.saturating_add(1) as u64;
            slot_ids.push(row.slot);
            tokens.push(row.last_token);
            start_positions.push(row.kv_seq_len);
            positions.push(position);
            batch_rows.push(Dsv4DecodeBatchRow {
                slot: row.slot,
                last_token: row.last_token,
                start_pos: row.kv_seq_len,
                position,
                params: row.params.clone(),
            });
        }
        Ok(Self {
            slot_ids,
            tokens,
            start_positions,
            positions,
            rows: batch_rows,
        })
    }
}

fn validate_dsv4_decode_kv_batch(rows: &[DecodeRow], kv_batch: &KvBatchDescriptor) -> Result<()> {
    ensure!(
        kv_batch.rows.len() == rows.len(),
        "DSv4 decode KV batch row count {} != plan rows {}",
        kv_batch.rows.len(),
        rows.len()
    );
    for (idx, (plan_row, kv_row)) in rows.iter().zip(&kv_batch.rows).enumerate() {
        ensure!(
            kv_row.kind == KvBatchRowKind::Decode,
            "DSv4 decode KV batch row {idx} has kind {:?}",
            kv_row.kind
        );
        ensure!(
            kv_row.slot == plan_row.slot,
            "DSv4 decode KV batch row {idx} slot {} != plan slot {}",
            kv_row.slot,
            plan_row.slot
        );
        ensure!(
            kv_row.seq_len == plan_row.kv_seq_len && kv_row.append_pos == plan_row.kv_seq_len,
            "DSv4 decode KV batch row {idx} seq/append ({},{}) != plan kv_seq_len {}",
            kv_row.seq_len,
            kv_row.append_pos,
            plan_row.kv_seq_len
        );
        ensure!(
            kv_row.append_len == 1,
            "DSv4 decode KV batch row {idx} append_len {} != 1",
            kv_row.append_len
        );
        ensure!(
            kv_row.page_range.start < kv_row.page_range.end,
            "DSv4 decode KV batch row {idx} has empty page range"
        );
        let tokens = &kv_batch.flat_token_ids[kv_row.token_range.clone()];
        ensure!(
            tokens == [plan_row.last_token],
            "DSv4 decode KV batch row {idx} tokens {:?} != plan token {}",
            tokens,
            plan_row.last_token
        );
    }
    Ok(())
}

fn validate_dsv4_prefill_kv_batch(
    row: &infer_plan::PrefillRow,
    kv_batch: &KvBatchDescriptor,
) -> Result<()> {
    ensure!(
        kv_batch.rows.len() == 1,
        "DSv4 prefill KV batch row count {} != 1",
        kv_batch.rows.len()
    );
    let kv_row = &kv_batch.rows[0];
    ensure!(
        kv_row.kind == KvBatchRowKind::Prefill,
        "DSv4 prefill KV batch row has kind {:?}",
        kv_row.kind
    );
    ensure!(
        kv_row.slot == row.slot,
        "DSv4 prefill KV batch slot {} != plan slot {}",
        kv_row.slot,
        row.slot
    );
    ensure!(
        kv_row.seq_len == row.start_pos && kv_row.append_pos == row.start_pos,
        "DSv4 prefill KV batch seq/append ({},{}) != plan start_pos {}",
        kv_row.seq_len,
        kv_row.append_pos,
        row.start_pos
    );
    ensure!(
        kv_row.append_len == row.tokens.len(),
        "DSv4 prefill KV batch append_len {} != plan token count {}",
        kv_row.append_len,
        row.tokens.len()
    );
    ensure!(
        kv_row.page_range.start < kv_row.page_range.end,
        "DSv4 prefill KV batch has empty page range"
    );
    let tokens = &kv_batch.flat_token_ids[kv_row.token_range.clone()];
    ensure!(
        tokens == row.tokens.as_slice(),
        "DSv4 prefill KV batch tokens do not match plan tokens"
    );
    Ok(())
}

fn validate_dsv4_decode_kv_view(
    rows: &[DecodeRow],
    view: &crate::attention::Dsv4KvBatchView,
) -> Result<()> {
    ensure!(
        view.rows.len() == rows.len(),
        "DSv4 decode KV adapter view row count {} != plan rows {}",
        view.rows.len(),
        rows.len()
    );
    for (idx, (plan_row, view_row)) in rows.iter().zip(&view.rows).enumerate() {
        ensure!(
            view_row.kind == KvBatchRowKind::Decode,
            "DSv4 decode KV adapter row {idx} has kind {:?}",
            view_row.kind
        );
        ensure!(
            view_row.slot == plan_row.slot,
            "DSv4 decode KV adapter row {idx} slot {} != plan slot {}",
            view_row.slot,
            plan_row.slot
        );
        ensure!(
            view_row.seq_len == plan_row.kv_seq_len
                && view_row.append_pos == plan_row.kv_seq_len
                && view_row.append_len == 1,
            "DSv4 decode KV adapter row {idx} seq/append ({},{},{}) != plan kv_seq_len {}",
            view_row.seq_len,
            view_row.append_pos,
            view_row.append_len,
            plan_row.kv_seq_len
        );
        ensure!(
            view_row.page_range.end <= view.flat_page_ids.len(),
            "DSv4 decode KV adapter row {idx} page range {:?} outside flat page len {}",
            view_row.page_range,
            view.flat_page_ids.len()
        );
        let pages = &view.flat_page_ids[view_row.page_range.clone()];
        ensure!(
            !pages.is_empty(),
            "DSv4 decode KV adapter row {idx} has no page ids"
        );
        ensure!(
            view_row.slot_page_range.end <= view.flat_slot_page_ids.len(),
            "DSv4 decode KV adapter row {idx} slot page range {:?} outside flat slot page len {}",
            view_row.slot_page_range,
            view.flat_slot_page_ids.len()
        );
        let slot_pages = &view.flat_slot_page_ids[view_row.slot_page_range.clone()];
        ensure!(
            !slot_pages.is_empty(),
            "DSv4 decode KV adapter row {idx} has no slot page ids"
        );
    }
    Ok(())
}

fn validate_dsv4_prefill_kv_view(
    row: &infer_plan::PrefillRow,
    view: &crate::attention::Dsv4KvBatchView,
) -> Result<()> {
    ensure!(
        view.rows.len() == 1,
        "DSv4 prefill KV adapter view row count {} != 1",
        view.rows.len()
    );
    let view_row = &view.rows[0];
    ensure!(
        view_row.kind == KvBatchRowKind::Prefill,
        "DSv4 prefill KV adapter row has kind {:?}",
        view_row.kind
    );
    ensure!(
        view_row.slot == row.slot
            && view_row.seq_len == row.start_pos
            && view_row.append_pos == row.start_pos
            && view_row.append_len == row.tokens.len(),
        "DSv4 prefill KV adapter row slot/seq/append ({},{},{},{}) != plan ({},{},{})",
        view_row.slot,
        view_row.seq_len,
        view_row.append_pos,
        view_row.append_len,
        row.slot,
        row.start_pos,
        row.tokens.len()
    );
    ensure!(
        view_row.page_range.end <= view.flat_page_ids.len(),
        "DSv4 prefill KV adapter row page range {:?} outside flat page len {}",
        view_row.page_range,
        view.flat_page_ids.len()
    );
    let pages = &view.flat_page_ids[view_row.page_range.clone()];
    ensure!(
        !pages.is_empty(),
        "DSv4 prefill KV adapter row has no page ids"
    );
    ensure!(
        view_row.slot_page_range.end <= view.flat_slot_page_ids.len(),
        "DSv4 prefill KV adapter row slot page range {:?} outside flat slot page len {}",
        view_row.slot_page_range,
        view.flat_slot_page_ids.len()
    );
    let slot_pages = &view.flat_slot_page_ids[view_row.slot_page_range.clone()];
    ensure!(
        !slot_pages.is_empty(),
        "DSv4 prefill KV adapter row has no slot page ids"
    );
    Ok(())
}

impl std::fmt::Debug for Dsv4CudaExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dsv4CudaExecutor")
            .field("model", &self.model)
            .field("num_slots", &self.num_slots)
            .finish()
    }
}

impl Dsv4CudaExecutor {
    fn tp_min_usize(&self, value: usize, what: &str) -> Result<usize> {
        let capped = i32::try_from(value.min(i32::MAX as usize)).unwrap_or(i32::MAX);
        self.model
            .tp
            .all_reduce_min_scalar_i32(&self.model.ctx, capped)
            .map(|v| v.max(0) as usize)
            .map_err(|e| anyhow::anyhow!("DSv4 TP min-reduce {what} failed: {e}"))
    }

    fn mirror_restore_pages(
        &mut self,
        slot: usize,
        slot_pages: &[u32],
        seq_len: usize,
    ) -> Result<()> {
        ensure!(
            !slot_pages.is_empty(),
            "DSv4 restore slot {slot} has empty host slot page table"
        );
        self.kv_adapter.mirror_slot_pages(slot, slot_pages, seq_len)
    }

    pub(crate) fn from_dsv4_fp8_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        max_seq_len: usize,
        mtp_draft_tokens: Option<usize>,
        mtp_draft_topk: Option<usize>,
    ) -> Result<Self> {
        ensure!(num_slots > 0, "Dsv4CudaExecutor requires at least one slot");
        ensure!(max_seq_len > 0, "Dsv4CudaExecutor requires max_seq_len > 0");
        let mtp_draft_tokens_for_load = mtp_draft_tokens
            .or_else(|| mtp_draft_topk.map(|_| crate::dsv4::DEFAULT_SPEC_DRAFT_DEPTH));
        let model = crate::dsv4::Dsv4Model::from_dsv4_fp8_safetensors(
            model_path.as_ref(),
            mtp_draft_tokens_for_load,
        )?;
        // [vram-probe] TEMP bit-exact VRAM attribution (remove after budget unification).
        // Returns measured-used bytes (or None when mem_get_info fails) so the
        // ledger can reconcile predicted device_bytes() against the measured used.
        let mem_dbg = |tag: &str| -> Option<usize> {
            match cudarc::driver::result::mem_get_info() {
                Ok((free, total)) => {
                    let used = total - free;
                    log::info!(
                        "[vram-probe] {tag}: used {}MB free {}MB",
                        used >> 20,
                        free >> 20
                    );
                    Some(used)
                }
                Err(_) => None,
            }
        };
        let weights_used_at_model_load = mem_dbg("after model load (weights+experts)");
        // Reclaim the cuMemAllocAsync pool BEFORE measuring free VRAM: weight
        // loading allocs+frees large device scratch (FP8 dequant, DeepGEMM cache
        // build, staging), and the retain-threshold=MAX pool holds it — so
        // `mem_get_info` in the budget would count freed loading scratch as USED
        // and starve the KV slot count. Trim returns it to the OS so the budget
        // sees the true free (recovers the KV slots those GB should fund).
        if let Err(e) = model.ctx.trim_memory_pool() {
            log::warn!("pre-KV-budget trim_memory_pool failed (non-fatal): {e}");
        }
        // Dynamic KV mem budget: clamp num_slots to what GPU free mem affords (was: fixed
        // num_slots → c=32 OOM crash at long max_seq_len). Deterministic ⇒ TP-consistent.
        let budget = model.kv_budget_plan(num_slots, max_seq_len)?;
        let num_slots = budget.num_slots;
        let kv_adapter = model.new_kv_adapter(max_seq_len, budget)?;
        mem_dbg("after new_kv_adapter (KV pools)");
        // [vram-ledger] PREDICTED adapter device bytes + per-component breakdown.
        log::info!(
            "[vram-ledger] adapter predicted {}MB; breakdown {:?}",
            kv_adapter.device_bytes() >> 20,
            kv_adapter
                .device_bytes_breakdown()
                .iter()
                .map(|(name, bytes)| (*name, bytes >> 20))
                .collect::<Vec<_>>()
        );
        let mut slots = Vec::with_capacity(num_slots);
        for slot_idx in 0..num_slots {
            slots.push(model.new_slot_state(max_seq_len, slot_idx, &kv_adapter)?);
            if slot_idx == 0 {
                mem_dbg("after slot 0 (per-slot state)");
                // [vram-ledger] PREDICTED slot-0 device bytes + per-component breakdown.
                log::info!(
                    "[vram-ledger] slot0 predicted {}MB; breakdown {:?}",
                    slots[0].device_bytes() >> 20,
                    slots[0]
                        .device_bytes_breakdown()
                        .iter()
                        .map(|(name, bytes)| (*name, bytes >> 20))
                        .collect::<Vec<_>>()
                );
                // [vram-ledger] Attribute the per-slot attention bulk to the exact
                // buffer family, summed across all layers (every bit's source named).
                log::info!(
                    "[vram-ledger] slot0 attention sub-totals (Σ layers) {:?}",
                    slots[0]
                        .attention_breakdown_total()
                        .iter()
                        .map(|(name, bytes)| (*name, bytes >> 20))
                        .collect::<Vec<_>>()
                );
                // Drift guard: the KV budget divides free VRAM by the STATIC
                // `per_slot_device_bytes`; if it drifts from the real slot alloc,
                // `affordable` mis-clamps num_slots and engine build OOMs (the
                // 43→382 MB under-count). Warn on >5% so it can't silently return.
                let predicted = model.per_slot_device_bytes(max_seq_len)?;
                let actual = slots[0].device_bytes();
                let drift = (predicted as i64 - actual as i64).unsigned_abs() as usize;
                if drift.saturating_mul(20) > actual {
                    log::warn!(
                        "[vram-ledger] DSv4 per-slot budget drift {}%: static per_slot_device_bytes {}MB vs \
                         slot0 device_bytes {}MB — reconcile per_slot_device_bytes with Dsv4SlotState::new",
                        drift.saturating_mul(100) / actual.max(1),
                        predicted >> 20,
                        actual >> 20,
                    );
                }
            }
        }
        let measured_used_after_all = mem_dbg("after all slots (build complete)");
        // [vram-ledger] Reconcile PREDICTED cumulative device_bytes against the
        // measured used. residual = measured_used
        //   - (weights_used_at_model_load + adapter.device_bytes() + Σ slot.device_bytes()).
        // The residual is everything NOT in the named-buffer ledger: CUDA context
        // + library reservations + per-cudaMalloc allocation rounding across the
        // ~258 tiny per-layer allocs/slot. A large residual points the gap there,
        // a small one means the named buffers fully account for the slot cost.
        let adapter_bytes = kv_adapter.device_bytes();
        let slots_bytes: usize = slots.iter().map(|s| s.device_bytes()).sum();
        log::info!(
            "[vram-ledger] cumulative predicted: weights {}MB + adapter {}MB + Σ {} slots {}MB = {}MB",
            weights_used_at_model_load.map_or(0, |b| b >> 20),
            adapter_bytes >> 20,
            num_slots,
            slots_bytes >> 20,
            (weights_used_at_model_load.unwrap_or(0) + adapter_bytes + slots_bytes) >> 20
        );
        if let (Some(measured), Some(weights)) =
            (measured_used_after_all, weights_used_at_model_load)
        {
            let predicted_total = weights + adapter_bytes + slots_bytes;
            // Saturating signed residual: usually positive (ctx/libs/rounding),
            // but guard the rare measured < predicted (measurement skew).
            let residual_mb = (measured as i64 - predicted_total as i64) >> 20;
            log::info!(
                "[vram-ledger] residual (ctx+libs+cudaMalloc rounding) = {residual_mb}MB \
                 (measured used {}MB - predicted {}MB)",
                measured >> 20,
                predicted_total >> 20
            );
        }
        let spec_slots = (0..num_slots)
            .map(|_| Dsv4SpecSlotState::default())
            .collect();
        let layer_specs: Vec<_> = model
            .layers
            .iter()
            .map(|layer| (layer.mode, layer.compress_ratio))
            .collect();
        let prefix_entry_bytes = crate::attention::dsv4_prefix_entry_max_bytes(
            &model.config,
            &model.kv_arena,
            &layer_specs,
            model.kv_arena.page_block_size,
        );
        let (park_budget, prefix_budget) =
            Self::split_dram_share(default_t1_budget_bytes(DEFAULT_DRAM_FRACTION));
        Ok(Self {
            model,
            slots,
            kv_adapter,
            spec_slots,
            spec_draft_tokens: mtp_draft_tokens_for_load,
            spec_draft_topk: mtp_draft_topk,
            num_slots,
            mtp_accepts: 0,
            mtp_rejects: 0,
            mtp_accept_ema: 1.0,
            mtp_skip_streak: 0,
            slot_tier: CudaKvTierStore::with_budget(park_budget, BLOB_CHUNK_BYTES),
            prefix_state: crate::attention::Dsv4PrefixStatePool::new(
                prefix_budget,
                prefix_entry_bytes,
            ),
        })
    }

    /// ONE `--kv-dram` share funds BOTH host stores — whole-slot park and
    /// prefix-state pool split it 50/50, an explicit deterministic partition,
    /// never two independent draws of the default fraction. Three tiers,
    /// three independent questions: HBM sizes slots (`kv_budget_plan`), this
    /// DRAM share sizes pool heat, `--kv-disk` alone sizes the spill.
    fn split_dram_share(bytes: usize) -> (usize, usize) {
        let park = bytes / 2;
        (park, bytes - park)
    }

    /// Publish every host page this forward completed for `slot` into the
    /// content-keyed prefix-state pool (#154 Phase 2 write hook). Best-effort:
    /// a publish failure only forfeits a future reuse, never the forward.
    fn publish_completed_prefix_pages(
        &mut self,
        slot: usize,
        slot_pages: &[u32],
        start_pos: usize,
        end_pos: usize,
    ) {
        if self.prefix_state.is_inactive() {
            return;
        }
        let page_tokens = self.model.kv_arena.page_block_size;
        if page_tokens == 0 || end_pos <= start_pos {
            return;
        }
        let align = self.model.config.sliding_window.max(1);
        for page_index in (start_pos / page_tokens)..(end_pos / page_tokens) {
            let page_end = (page_index + 1) * page_tokens;
            let Some(&page_id) = slot_pages.get(page_index) else {
                warn!(
                    "DSv4 prefix publish: slot {slot} page {page_index} outside host table ({} pages)",
                    slot_pages.len()
                );
                return;
            };
            // Boundary sections exist only when the forward ended exactly here;
            // restore commits only at `align` multiples, so skip odd page ends.
            let boundary = page_end == end_pos && page_end.is_multiple_of(align);
            match self.slots[slot].capture_prefix_page(
                &self.model.ctx,
                &self.model.layers,
                &self.kv_adapter,
                &self.model.kv_arena,
                self.model.config.index_head_dim,
                page_tokens,
                page_index,
                boundary,
            ) {
                Ok(entry) => {
                    if !self.prefix_state.publish(page_id, &entry) {
                        warn!(
                            "DSv4 prefix publish: pool refused host page {page_id} \
                             (slot {slot} idx {page_index})"
                        );
                    }
                }
                Err(err) => {
                    warn!("DSv4 prefix publish failed for slot {slot} page {page_index}: {err:#}")
                }
            }
        }
    }

    /// Radix evicted these host pages: drop their pool entries (the pool's
    /// lifetime rides the radix, no independent lifetime above the DRAM budget).
    pub(crate) fn release_prefix_pages(&mut self, pages: &[u32]) {
        self.prefix_state.remove_pages(pages);
    }

    /// Radix publish confirmed these pages (engine `save_prefix_sidecar` arm):
    /// flip their pool entries from provisional to readable.
    pub(crate) fn confirm_prefix_pages(&mut self, pages: &[u32]) {
        self.prefix_state.confirm_pages(pages);
    }

    /// Attach the opt-in NVMe disk spill level (pre-serve only) to BOTH host
    /// stores, splitting the ONE `--kv-disk` cap the same way as the DRAM
    /// share. The cap is soft for the prefix pool: over it, publish drops the
    /// pool's oldest entries — capacity never blocks a forward and never
    /// enters the reuse license.
    pub(crate) fn set_kv_tier_disk(
        &mut self,
        root: std::path::PathBuf,
        budget_bytes: usize,
    ) -> bool {
        let (park_disk, prefix_disk) = Self::split_dram_share(budget_bytes);
        let park_ok = self
            .slot_tier
            .set_disk(root.clone(), park_disk, BLOB_CHUNK_BYTES);
        let prefix_ok = self.prefix_state.set_disk(root, prefix_disk);
        if !park_ok || !prefix_ok {
            warn!("DSv4 --kv-disk attach incomplete: park={park_ok} prefix={prefix_ok}");
        }
        park_ok && prefix_ok
    }

    /// Pre-serve re-budget rebuilds THE stores, orphaning every stored blob.
    /// The share is split across both stores — see [`Self::split_dram_share`].
    pub(crate) fn set_kv_tier_budget_bytes(&mut self, bytes: usize) {
        let (park_budget, prefix_budget) = Self::split_dram_share(bytes);
        self.slot_tier = CudaKvTierStore::with_budget(park_budget, BLOB_CHUNK_BYTES);
        self.prefix_state.set_budget_bytes(prefix_budget);
    }

    pub(crate) fn kv_tier_host_demoted_pages(&self) -> usize {
        self.slot_tier.host_demoted_pages() + self.prefix_state.host_pages()
    }

    pub(crate) fn kv_tier_disk_pages(&self) -> usize {
        self.slot_tier.disk_pages() + self.prefix_state.disk_pages()
    }

    /// Content-keyed reuse license (#154 Phase 2): a leading page is
    /// attachable only while every page before and including it has a pool
    /// entry; the COMMIT point additionally requires the page to carry
    /// boundary sections at a restore-alignment end. Pages between commit
    /// points keep the scan alive without advancing it. Pool presence covers
    /// host DRAM and disk alike — licensing never does capacity math.
    pub(crate) fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
        let page_tokens = self.model.kv_arena.page_block_size;
        if page_tokens == 0 {
            return 0;
        }
        let align = self.model.config.sliding_window.max(1);
        let mut committed = 0usize;
        for (idx, block) in blocks.iter().enumerate() {
            // Fail closed on demoted keys: DSv4 pages never demote through
            // the radix tier (demote_prefix_pages returns 0).
            let PrefixBlock::ResidentPage(page_id) = *block else {
                break;
            };
            let Some(meta) = self.prefix_state.page_meta(page_id) else {
                break;
            };
            let page_end = (idx + 1) * page_tokens;
            if meta.boundary && page_end.is_multiple_of(align) {
                committed = idx + 1;
            }
        }
        committed
    }

    /// The boundary sections only exist at a forward's own end position —
    /// without this, a single-call prefill never visits an earlier boundary.
    pub(crate) fn prefill_restore_boundary_alignment(&self) -> usize {
        self.model.config.sliding_window.max(1)
    }

    /// Restore a radix-matched prefix into `slot` from the content-keyed pool
    /// (#154 Phase 2 read hook; engine calls via the restore_prefix_sidecar
    /// seam AFTER attaching the host pages). Every entry decodes to owned
    /// host state first, so a missing/undecodable page aborts before any
    /// device byte moves; an `Err` propagates to the engine's fallback
    /// (free slot, full recompute) — never a partial restore.
    pub(crate) fn restore_prefix_state(
        &mut self,
        slot: usize,
        matched_len: usize,
        prefix_pages: &[u32],
    ) -> Result<()> {
        ensure!(
            slot < self.num_slots,
            "DSv4 prefix restore slot {slot} outside executor slots {}",
            self.num_slots
        );
        let page_tokens = self.model.kv_arena.page_block_size;
        let align = self.model.config.sliding_window.max(1);
        ensure!(
            page_tokens > 0 && matched_len > 0,
            "DSv4 prefix restore needs a non-empty match (matched_len {matched_len})"
        );
        // D4 hard error: an unaligned match means the engine trim/clamp
        // ordering regressed — fail loud, never silently skip the ring restore.
        ensure!(
            matched_len.is_multiple_of(align) && matched_len.is_multiple_of(page_tokens),
            "DSv4 prefix restore matched_len {matched_len} not aligned to ring {align} / page {page_tokens}"
        );
        ensure!(
            prefix_pages.len() == matched_len / page_tokens,
            "DSv4 prefix restore {} pages != matched_len {matched_len} / page {page_tokens}",
            prefix_pages.len()
        );
        let entries = prefix_pages
            .iter()
            .map(|&page_id| self.prefix_state.read_entry(page_id))
            .collect::<Result<Vec<_>>>()?;
        self.kv_adapter.mirror_full_band(slot, matched_len)?;
        // A17: a restored occupant enters Decoding without a tail warm step
        // for its MTP chain; stale spec state belongs to the prior occupant.
        self.spec_slots[slot] = Dsv4SpecSlotState::default();
        self.slots[slot].restore_prefix_state(
            &self.model.ctx,
            &self.model.layers,
            &mut self.kv_adapter,
            &self.model.kv_arena,
            self.model.config.index_head_dim,
            &entries,
            matched_len,
            page_tokens,
        )
    }

    /// Whole-slot swap is rank-local bytes plus TP-wide scalar consensus.
    pub(crate) fn kv_slot_tier_enabled(&self) -> bool {
        true
    }

    /// Demote `slot`'s entire device state into the tier store under the
    /// engine-minted `key`. Contract (see
    /// `infer_seam::BackendExecutor::demote_slot`): the copy is complete
    /// before returning — `swap_out_image` ends in `ctx.sync()` — so the
    /// engine may free the slot immediately. `Ok(false)` = no room on some
    /// rank (engine falls back to plain recompute). Exactly TWO collectives
    /// on every path (capture consensus, insert consensus), so the lockstep
    /// collective count is rank-invariant.
    pub(crate) fn demote_slot(&mut self, slot: usize, key: u64) -> Result<bool> {
        ensure!(
            slot < self.num_slots,
            "DSv4 demote slot {slot} outside executor slots {}",
            self.num_slots
        );
        let image = self.slots[slot].swap_out_image(&self.model.ctx, &self.kv_adapter);
        let capture_ok = usize::from(image.is_ok());
        if self.tp_min_usize(capture_ok, "slot demote capture")? == 0 {
            return Err(image
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("peer rank failed DSv4 slot demote capture")));
        }
        let bytes = image?.to_bytes();
        let inserted = self
            .slot_tier
            .insert_chunked(NS_SLOT, NS_SLOT_CHUNK, key, &bytes);
        if self.tp_min_usize(usize::from(inserted), "slot demote insert")? == 0 {
            if inserted {
                self.slot_tier.remove_chunked(NS_SLOT, NS_SLOT_CHUNK, key);
            }
            return Ok(false);
        }
        Ok(true)
    }

    /// Restore the whole-slot snapshot stored under `key` into `slot`. The engine
    /// resumes decode at the demoted position right after this returns, and
    /// drops the entry via [`Self::drop_kv_slot_entries`] — the entry
    /// intentionally stays in the store here. `swap_in_image` ends in
    /// `ctx.sync()`, so both the device restore and the spec-hidden H2D (same
    /// stream, ordered before it) are complete before the host image can be
    /// dropped. Exactly TWO collectives on every path (read+parse consensus,
    /// restore consensus) — nothing rank-local errs between them.
    pub(crate) fn promote_slot(&mut self, key: u64, slot: usize, slot_pages: &[u32]) -> Result<()> {
        ensure!(
            slot < self.num_slots,
            "DSv4 promote slot {slot} outside executor slots {}",
            self.num_slots
        );
        let image = self
            .slot_tier
            .read_chunked(NS_SLOT, NS_SLOT_CHUNK, key)
            .and_then(|bytes| {
                crate::dsv4::Dsv4SlotSnapshot::from_bytes(&bytes)
                    .map_err(|e| anyhow::anyhow!("DSv4 promote slot: {e:#}"))
            });
        let image_ok = usize::from(image.is_ok());
        if self.tp_min_usize(image_ok, "slot promote read")? == 0 {
            return Err(image
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("peer rank failed DSv4 slot promote read")));
        }
        let image = image?;
        let restored = self
            .mirror_restore_pages(slot, slot_pages, image.seq_len())
            .and_then(|()| {
                self.spec_slots[slot] = Dsv4SpecSlotState::default();
                self.slots[slot].swap_in_image(&self.model.ctx, &mut self.kv_adapter, &image)
            });
        let restore_ok = usize::from(restored.is_ok());
        if self.tp_min_usize(restore_ok, "slot promote restore")? == 0 {
            return Err(restored
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("peer rank failed DSv4 slot promote restore")));
        }
        restored
    }

    pub(crate) fn drop_kv_slot_entries(&mut self, keys: &[u64]) {
        for &key in keys {
            self.slot_tier.remove_chunked(NS_SLOT, NS_SLOT_CHUNK, key);
        }
    }

    /// `BackendExecutor::tp_sync_min` — see there for why the scheduler needs
    /// this (2026-07-05 TP=4 admission livelock).
    pub(crate) fn tp_sync_min(&self, local: usize) -> Result<usize> {
        self.tp_min_usize(local, "admission free pages")
    }

    /// One no-spec greedy forward that ALSO stages the MTP draft state (pending
    /// token + stream hidden) so a subsequent `spec_step` can resume. Shared by
    /// final-prefill (seeds the first chain) and the adaptive gate's fallback (a
    /// low-acceptance step runs at no-spec cost but keeps the draft head warm).
    fn forward_mtp_warm_step(
        &mut self,
        slot_idx: usize,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<u32> {
        let (token, hidden) = self.model.forward_tokens_with_hidden(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            tokens,
            start_pos,
            params,
            position,
        )?;
        self.spec_slots[slot_idx].pending = Some(token);
        self.spec_slots[slot_idx].hidden = Some(hidden);
        Ok(token)
    }

    fn forward_prefill_tokens(
        &mut self,
        slot_idx: usize,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
        final_prefill: bool,
    ) -> Result<Vec<u32>> {
        let spec_on = self.spec_requested();
        if spec_on && params.is_greedy() {
            let token = if final_prefill {
                self.forward_mtp_warm_step(slot_idx, tokens, start_pos, params, position)?
            } else {
                // Non-final chunk: emit the token, drop any stale draft state.
                let (token, _hidden) = self.model.forward_tokens_with_hidden(
                    &mut self.slots[slot_idx],
                    &mut self.kv_adapter,
                    tokens,
                    start_pos,
                    params,
                    position,
                )?;
                self.spec_slots[slot_idx] = Dsv4SpecSlotState::default();
                token
            };
            Ok(vec![token])
        } else {
            if spec_on {
                self.spec_slots[slot_idx] = Dsv4SpecSlotState::default();
            }
            Ok(vec![self.model.forward_tokens(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                tokens,
                start_pos,
                params,
                position,
            )?])
        }
    }

    fn forward_decode_tokens(
        &mut self,
        slot_idx: usize,
        last_token: u32,
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<Vec<u32>> {
        if !self.spec_requested() {
            let token = self.model.forward_tokens(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                &[last_token],
                start_pos,
                params,
                position,
            )?;
            return Ok(vec![token]);
        }

        if !params.is_greedy() {
            self.spec_slots[slot_idx] = Dsv4SpecSlotState::default();
            let token = self.model.forward_tokens(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                &[last_token],
                start_pos,
                params,
                position,
            )?;
            return Ok(vec![token]);
        }
        // Self-heal an un-seeded / desynced MTP stream (#140). A request that
        // enters Decoding WITHOUT a tail prefill — a full position-0 prefix-cache
        // hit or a whole-slot promote, both of which reset `spec_slots` to
        // default (pending=None) and rely on a tail warm-step that never runs on
        // a full hit — reaches its first decode spec_step with no pending token.
        // The old code bailed the whole TP group (crash observed ~613 verify
        // ticks into sustained serving once the prefix cache warmed). Instead
        // re-seed via one warm no-spec step for `last_token`: it stages
        // pending + hidden and emits the token, so the NEXT step runs real MTP.
        // A pending that disagrees with `last_token` is a stream desync (the
        // staged hidden belongs to a different token) — same recovery, warned.
        let seeded = match self.spec_slots[slot_idx].pending {
            Some(pending) if pending == last_token => true,
            Some(pending) => {
                log::warn!(
                    "DSv4 MTP stream desync (slot {slot_idx}): pending {pending} != last_token \
                     {last_token}; re-seeding via a warm step"
                );
                false
            }
            None => false,
        };
        if !seeded {
            let token =
                self.forward_mtp_warm_step(slot_idx, &[last_token], start_pos, params, position)?;
            return Ok(vec![token]);
        }
        // Adaptive gate (B=1): when the running acceptance EMA predicts MTP would
        // lose to no-spec, run a warm no-spec step instead — keeps the draft head
        // staged so MTP resumes the moment acceptance recovers (a periodic probe
        // forces one real step to refresh the EMA). The warm step needs the MTP
        // hidden so it takes the eager forward (not the decode-graph fast path):
        // it costs eager-no-spec, a touch above t_nospec, which only narrows the
        // win — calibrate MIN_ACCEPT against that. NOTE: the EMA + skip_streak are
        // executor-global (fine for this B=1 bring-up flag); make them per-slot
        // before promoting the gate to a default.
        if self.mtp_adaptive_skip() {
            self.mtp_skip_streak += 1;
            let token =
                self.forward_mtp_warm_step(slot_idx, &[last_token], start_pos, params, position)?;
            return Ok(vec![token]);
        }
        self.spec_step(slot_idx, start_pos, position)
    }

    fn forward_decode_row(&mut self, row: &Dsv4DecodeBatchRow) -> Result<Vec<SlotToken>> {
        let tokens = self.forward_decode_tokens(
            row.slot,
            row.last_token,
            row.start_pos,
            &row.params,
            row.position,
        )?;
        Ok(tokens
            .into_iter()
            .map(|token| SlotToken {
                slot: row.slot,
                token,
                logprob: None,
                finish: None,
            })
            .collect())
    }

    /// One decode sub-batch, then the prefix-state publish hook: any row that
    /// crossed a host-page boundary this tick publishes its completed pages.
    /// The hook runs AFTER the inner forward's MTP verify+commit+truncate, so
    /// `seq_len` reflects only committed tokens — rejected drafts never enter
    /// the pool.
    fn forward_decode_batch(
        &mut self,
        rows: &[DecodeRow],
        kv_batch: &KvBatchDescriptor,
    ) -> Result<Vec<SlotToken>> {
        let out = self.forward_decode_batch_inner(rows, kv_batch)?;
        for (row, kv_row) in rows.iter().zip(&kv_batch.rows) {
            let slot_pages = &kv_batch.flat_slot_page_ids[kv_row.slot_page_range.clone()];
            let end_pos = self.slots[row.slot].seq_len();
            self.publish_completed_prefix_pages(row.slot, slot_pages, row.kv_seq_len, end_pos);
        }
        Ok(out)
    }

    fn forward_decode_batch_inner(
        &mut self,
        rows: &[DecodeRow],
        kv_batch: &KvBatchDescriptor,
    ) -> Result<Vec<SlotToken>> {
        validate_dsv4_decode_kv_batch(rows, kv_batch)?;
        let kv_view = self.kv_adapter.prepare_kv_batch(kv_batch)?;
        validate_dsv4_decode_kv_view(rows, &kv_view)?;
        // Host band changed since last sync (page growth at a page boundary) —
        // refresh the graph-referenced device tables BEFORE the forward. Decode
        // growth previously never resynced the device table (#154 Phase 0).
        for row in rows {
            if self.kv_adapter.take_device_table_dirty(row.slot) {
                self.slots[row.slot]
                    .refresh_flashmla_device_page_tables(&self.model.ctx, &self.kv_adapter)?;
            }
        }
        let batch = Dsv4DecodeBatch::from_rows(rows, &self.slots, self.num_slots)?;
        ensure!(
            batch.slot_ids.len() == batch.rows.len()
                && batch.tokens.len() == batch.rows.len()
                && batch.start_positions.len() == batch.rows.len()
                && batch.positions.len() == batch.rows.len(),
            "DSv4 decode batch surface length mismatch"
        );
        if batch.rows.len() == 1 {
            let out = self.forward_decode_row(&batch.rows[0])?;
            // B=1 bypasses `Dsv4Model::forward_decode_batch` (the CUDA-graph
            // decode-graph lane, `forward_tokens_decode_graph`) — mirror its
            // trace call here so the diagnostic covers both lanes uniformly.
            // See ARLE_DSV4_DECODE_TRACE, crate::dsv4::dsv4_decode_trace.
            crate::dsv4::dsv4_decode_trace(
                self.model.tp.config().rank,
                &batch.slot_ids,
                &batch.start_positions,
                &out.iter().map(|t| t.token).collect::<Vec<_>>(),
            );
            return Ok(out);
        }

        // Cross-slot batched MTP decode (batched-MTP Stage 1). B=1 already took
        // the single-row path above; B>1 drives all N chains through one batched
        // verify (MoE grouped over the verify rows, attention per-slot) instead
        // of the per-row `spec_step` loop.
        let spec_on = self.spec_requested();
        let all_greedy = batch.rows.iter().all(|row| row.params.is_greedy());
        if spec_on && all_greedy {
            // Self-heal an un-seeded / desynced MTP stream (#140), the batched
            // twin of the B=1 path: a slot entering Decoding without a tail
            // prefill (full prefix-cache hit / whole-slot promote) has pending=
            // None. Rather than bail the TP group, warm-step EVERY row this one
            // tick to (re)seed pending+hidden, then batched MTP resumes next
            // tick. Cheaper than splitting the batch; a one-tick per-slot
            // degradation only when a slot joins un-seeded.
            let needs_seed = batch
                .rows
                .iter()
                .any(|row| self.spec_slots[row.slot].pending != Some(row.last_token));
            if needs_seed {
                let mut tokens = Vec::with_capacity(batch.rows.len());
                for row in &batch.rows {
                    if let Some(pending) = self.spec_slots[row.slot].pending {
                        if pending != row.last_token {
                            log::warn!(
                                "DSv4 MTP batched stream desync (slot {}): pending {pending} != \
                                 last_token {}; re-seeding",
                                row.slot,
                                row.last_token
                            );
                        }
                    }
                    let token = self.forward_mtp_warm_step(
                        row.slot,
                        &[row.last_token],
                        row.start_pos,
                        &row.params,
                        row.position,
                    )?;
                    tokens.push(SlotToken {
                        slot: row.slot,
                        token,
                        logprob: None,
                        finish: None,
                    });
                }
                return Ok(tokens);
            }
            let committed = self.spec_step_batched(&batch.slot_ids, &batch.start_positions)?;
            ensure!(
                committed.len() == batch.rows.len(),
                "DSv4 batched MTP returned {} chains for {} rows",
                committed.len(),
                batch.rows.len()
            );
            let tokens: Vec<SlotToken> = batch
                .slot_ids
                .iter()
                .zip(committed)
                .flat_map(|(&slot, chain)| {
                    chain.into_iter().map(move |token| SlotToken {
                        slot,
                        token,
                        logprob: None,
                        finish: None,
                    })
                })
                .collect();
            return Ok(tokens);
        }

        // True batched decode (layer-major driver, batched attention + grouped
        // MoE). B=1 is the single-row reference above; B>1 always batches. If
        // spec was requested but sampling is not greedy, disable spec state for
        // these rows and use the same normal batched decode lane.
        if spec_on {
            for row in &batch.rows {
                self.spec_slots[row.slot] = Dsv4SpecSlotState::default();
            }
        }
        let params: Vec<SamplingParams> = batch.rows.iter().map(|r| r.params.clone()).collect();
        let out = self.model.forward_decode_batch(
            &mut self.slots,
            &mut self.kv_adapter,
            &batch.slot_ids,
            &batch.tokens,
            &batch.start_positions,
            &batch.positions,
            &params,
        )?;
        ensure!(
            out.len() == batch.rows.len(),
            "DSv4 batched decode returned {} tokens for {} rows",
            out.len(),
            batch.rows.len()
        );
        Ok(batch
            .slot_ids
            .iter()
            .zip(out)
            .map(|(&slot, token)| SlotToken {
                slot,
                token,
                logprob: None,
                finish: None,
            })
            .collect())
    }

    fn submit(&mut self, plan: &ForwardPlan, kv_batch: &KvBatchDescriptor) -> Result<StepOutput> {
        let rows = plan.decode_rows.len() + plan.prefill_rows.len();
        // Cross-rank lockstep debug surface: every rank logs a per-forward plan
        // fingerprint. Divergence at tick K (different rows on different ranks)
        // is the NCCL-deadlock root cause for multi-rank serve; diff the per-rank
        // streams with RUST_LOG=infer_cuda=debug.
        if rows > 0 && log::log_enabled!(log::Level::Debug) {
            static PLAN_TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let tick = PLAN_TICK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            log::debug!(
                "[dsv4-plan] rank={} tick={tick} prefill={:?} decode={:?}",
                self.model.tp.config().rank,
                plan.prefill_rows
                    .iter()
                    .map(|r| (r.slot, r.start_pos, r.tokens.len()))
                    .collect::<Vec<_>>(),
                plan.decode_rows
                    .iter()
                    .map(|r| (r.slot, r.kv_seq_len))
                    .collect::<Vec<_>>(),
            );
        }
        if rows == 0 {
            ensure!(
                kv_batch.rows.is_empty(),
                "DSv4 empty plan got non-empty KV batch descriptor"
            );
            return Ok(StepOutput { tokens: Vec::new() });
        }

        if plan.prefill_rows.is_empty() {
            return Ok(StepOutput {
                tokens: self.forward_decode_batch(&plan.decode_rows, kv_batch)?,
            });
        }

        // Mixed / multi-prefill plans split into per-prefill single-row
        // sub-steps plus one decode sub-batch. Plan rows always address
        // disjoint slots (a request is either Prefilling or Decoding), so the
        // sequential sub-steps are math-identical to consecutive single-mode
        // ticks. Descriptor commit order is prefill rows first, then decode
        // rows (`KvBatchDescriptor::from_plan`), mapping plan rows onto
        // descriptor rows by index.
        ensure!(
            kv_batch.rows.len() == rows,
            "DSv4 KV batch descriptor has {} rows for a {rows}-row plan",
            kv_batch.rows.len()
        );
        let mut seen_slots = std::collections::BTreeSet::new();
        for slot in plan
            .prefill_rows
            .iter()
            .map(|row| row.slot)
            .chain(plan.decode_rows.iter().map(|row| row.slot))
        {
            ensure!(
                seen_slots.insert(slot),
                "DSv4 plan schedules slot {slot} more than once per tick"
            );
        }

        let n_prefill = plan.prefill_rows.len();
        let mut tokens = Vec::with_capacity(rows);
        for (idx, row) in plan.prefill_rows.iter().enumerate() {
            let sub_batch = kv_batch.subset(idx..idx + 1)?;
            tokens.extend(self.submit_prefill_row(row, &sub_batch)?);
        }
        if !plan.decode_rows.is_empty() {
            let sub_batch = kv_batch.subset(n_prefill..kv_batch.rows.len())?;
            tokens.extend(self.forward_decode_batch(&plan.decode_rows, &sub_batch)?);
        }
        Ok(StepOutput { tokens })
    }

    /// One prefill row as its own single-row sub-step. `kv_batch` must be the
    /// row's single-row (sub-)descriptor — indistinguishable from what a
    /// prefill-only tick delivers.
    fn submit_prefill_row(
        &mut self,
        row: &infer_plan::PrefillRow,
        kv_batch: &KvBatchDescriptor,
    ) -> Result<Vec<SlotToken>> {
        validate_dsv4_prefill_kv_batch(row, kv_batch)?;
        ensure!(
            row.slot < self.num_slots,
            "prefill slot {} outside DSv4 executor slots {}",
            row.slot,
            self.num_slots
        );
        ensure!(!row.tokens.is_empty(), "prefill row must carry tokens");
        // C2: free+reset BEFORE prepare_kv_batch (free-then-alloc, matching the
        // Qwen3.5 executor). prepare_kv_batch draws pages + advances seq_len; if
        // the reset ran after, fresh prefill wrote into an EMPTY page table and a
        // reused slot aborted the prepare seq_len==append_pos invariant.
        if row.start_pos == 0 {
            self.kv_adapter.flashmla_free_slot(row.slot)?;
            self.slots[row.slot].reset(&self.model.ctx, &mut self.kv_adapter)?;
            self.spec_slots[row.slot] = Dsv4SpecSlotState::default();
        }
        let kv_view = self.kv_adapter.prepare_kv_batch(kv_batch)?;
        validate_dsv4_prefill_kv_view(row, &kv_view)?;
        if row.start_pos == 0 {
            self.kv_adapter.zero_slot_band(&self.model.ctx, row.slot)?;
        }
        // Host band changed since last sync — refresh the graph-referenced device tables.
        if self.kv_adapter.take_device_table_dirty(row.slot) {
            self.slots[row.slot]
                .refresh_flashmla_device_page_tables(&self.model.ctx, &self.kv_adapter)?;
        }
        let position = (row.start_pos + row.tokens.len()) as u64;
        let final_prefill = row.start_pos + row.tokens.len() >= row.total_tokens;
        let tokens = self.forward_prefill_tokens(
            row.slot,
            &row.tokens,
            row.start_pos,
            &row.params,
            position,
            final_prefill,
        )?;
        let slot = row.slot;
        // Prefix-state publish hook: this chunk completed pages
        // [start_pos/page, (start_pos+len)/page).
        let slot_pages = &kv_batch.flat_slot_page_ids[kv_batch.rows[0].slot_page_range.clone()];
        self.publish_completed_prefix_pages(
            slot,
            slot_pages,
            row.start_pos,
            row.start_pos + row.tokens.len(),
        );
        Ok(tokens
            .into_iter()
            .map(|token| SlotToken {
                slot,
                token,
                logprob: None,
                finish: None,
            })
            .collect())
    }

    pub(crate) fn verify_forward_selftest(&mut self, prompt: &[u32]) -> Result<()> {
        ensure!(
            !prompt.is_empty(),
            "DSv4 verify-forward selftest requires a non-empty prompt"
        );
        let slot_idx = 0;
        let params = SamplingParams::default();
        let start_pos = prompt.len();

        self.slots[slot_idx].reset(&self.model.ctx, &mut self.kv_adapter)?;
        let token_a = self.model.forward_tokens(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            prompt,
            0,
            &params,
            start_pos as u64,
        )?;
        let verify_one = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &[token_a],
            start_pos,
            (start_pos + 1) as u64,
        )?;

        self.slots[slot_idx].reset(&self.model.ctx, &mut self.kv_adapter)?;
        let token_a_again = self.model.forward_tokens(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            prompt,
            0,
            &params,
            start_pos as u64,
        )?;
        ensure!(
            token_a == token_a_again,
            "DSv4 verify selftest prefill token drifted: {token_a} != {token_a_again}"
        );
        let normal_one = self.model.forward_tokens(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &[token_a],
            start_pos,
            &params,
            (start_pos + 1) as u64,
        )?;
        ensure!(
            verify_one.argmax.first().copied() == Some(normal_one),
            "DSv4 verify selftest one-token mismatch: verify={:?} normal={normal_one}",
            verify_one.argmax
        );

        self.slots[slot_idx].reset(&self.model.ctx, &mut self.kv_adapter)?;
        let token_a = self.model.forward_tokens(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            prompt,
            0,
            &params,
            start_pos as u64,
        )?;
        let verify_one = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &[token_a],
            start_pos,
            (start_pos + 1) as u64,
        )?;
        let token_b = verify_one.argmax[0];
        let mut wrong_b = token_b.wrapping_add(2);
        if wrong_b == token_b {
            wrong_b = token_b.wrapping_add(3);
        }

        self.slots[slot_idx].reset(&self.model.ctx, &mut self.kv_adapter)?;
        let token_a = self.model.forward_tokens(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            prompt,
            0,
            &params,
            start_pos as u64,
        )?;
        let verify_two = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &[token_a, wrong_b],
            start_pos,
            (start_pos + 1) as u64,
        )?;
        ensure!(
            verify_two.argmax.first() == verify_one.argmax.first(),
            "DSv4 verify selftest two-token row0 mismatch: one={:?} two={:?}",
            verify_one.argmax,
            verify_two.argmax
        );

        // NOTE: no col1/bonus byte-identity gate here. The 2-token verify's col1
        // (bonus) on a FORCED-WRONG draft is DISCARDED in real decode (rejects emit
        // only base_next), and any byte-identity check is confounded by the M=2-vs-M=1
        // FP8 kernel path (SWA-prefill + prefill-DeepGEMM vs FlashMLA-decode +
        // decode-DeepGEMM). The real correctness gate is full-decode byte-identity vs
        // non-spec (validated 2026-06-08: batched MTP byte-identical on needle+capital,
        // +61/+70%). See errors/2026-06-08-dsv4-batched-verify-col1-wrong.md.

        // Depth-2 sequential MTP probe REMOVED 2026-06-08 (killed): measured depth-1 3/3,
        // depth-2-top1 1/3 (~33% accept) → sequential chain is only ~+15%, not the 6ms
        // path. The single MTP head caps multi-step drafts; tree-EAGLE (top-K) is the
        // amortization lever beyond depth-1. See the consolidated decode-6ms report.

        self.slots[slot_idx].reset(&self.model.ctx, &mut self.kv_adapter)?;
        self.spec_slots[slot_idx] = Dsv4SpecSlotState::default();
        if self.model.tp.config().rank == 0 {
            eprintln!(
                "[dsv4-mtp-selftest] PASS token_a={token_a} token_b={token_b} wrong_b={wrong_b} verify_two={:?}",
                verify_two.argmax
            );
        }
        Ok(())
    }
}

/// Whole-step Qwen3.5/3.6 decode graph enabled? `--qwen35-decode-graph true`
/// opt-in, default OFF until the pod license (≥ +10% tok/s + needle gate +
/// replay-reuse evidence per the bench spec). The eager path stays the
/// correctness floor; `Qwen35CudaExecutor::warmup` additionally gates TP and
/// host-routed MoE off regardless of this.
fn qwen35_decode_graph_enabled() -> bool {
    crate::runtime_flags::qwen35_decode_graph()
}

/// Batched rows>1 decode for Qwen3.5/3.6 (stage 1, contiguous KV) — re-port
/// of the deleted monolith's `decode_batch` design
/// ([`crate::qwen35::Qwen35BatchDecodeState`]). DEFAULT ON: before this path
/// existed, a rows>1 plan was a hard executor error (`rows == 1` ensure →
/// engine death), so even a conservative batched path strictly dominates the
/// status quo. `--qwen35-batched-decode false` is the escape hatch AND the
/// honest same-binary A/B arm: it processes rows>1 decode plans as sequential
/// per-row single-row forwards instead (a NEW loop — the old behavior was
/// death, not a fallback). rows==1 plans never consult this gate (the
/// single-row path is byte-identical either way).
fn qwen35_batched_decode_enabled() -> bool {
    crate::runtime_flags::qwen35_batched_decode()
}

/// Decode-graph replay/capture probes — static so the pod bench can PROVE
/// replay reuse from server logs (license requires reuse evidence, not
/// capture-exists). Expected steady state: captures == live slots (≤
/// num_slots keys; B=1/R=8/seq=1 are shape constants and kv enters via the
/// staged device scalar, so there is exactly ONE graph per slot), replays ≈
/// decoded tokens.
static QWEN35_GRAPH_CAPTURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static QWEN35_GRAPH_REPLAYS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Device addresses a slot's captured decode graph was baked against. The
/// graph replays kernel launches against FIXED pointers, so if any anchor
/// moved (workspace `release()` → re-alloc, a length-flip on the staging
/// slots) the capture is stale and must be dropped — replaying it would read
/// freed memory (the 2026-06-10 IMA class). `ws_epoch` covers wholesale
/// releases; the three pointer anchors cover the staged-input and output
/// buffers directly.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Qwen35GraphBake {
    token_ids_ptr: u64,
    start_pos_ptr: u64,
    logits_ptr: u64,
    ws_epoch: u64,
}

/// Per-executor whole-step decode-graph state: a DEDICATED decode workspace
/// (only ever sees `seq_len == 1` shapes, so its buffer addresses are stable
/// across prefill interleaves — the main workspace re-shapes on every
/// prefill chunk and would invalidate captures every request) plus one
/// [`CudaGraphState`] per slot (captured buffers bake the slot's k/v cache +
/// GDR/conv state addresses, which live for the executor's lifetime and are
/// only memset by `reset`).
struct Qwen35DecodeGraph {
    ws: crate::qwen35::Qwen35Workspace,
    graphs: Vec<crate::graph::CudaGraphState>,
    baked: Vec<Option<Qwen35GraphBake>>,
}

impl Qwen35DecodeGraph {
    fn new(num_slots: usize, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> Self {
        Self {
            ws: crate::qwen35::Qwen35Workspace::new(),
            graphs: (0..num_slots)
                .map(|_| crate::graph::CudaGraphState::new(stream.clone()))
                .collect(),
            baked: vec![None; num_slots],
        }
    }
}

/// Qwen3.5 / Qwen3.6 HYBRID executor: drives
/// [`crate::qwen35::Qwen35Model::forward_tokens`] per single-row sub-step and
/// [`crate::qwen35::Qwen35Model::forward_decode_batch`] for rows>1 pure-decode
/// sub-batches (stage-1 batched decode, contiguous KV). Owns per-slot KV state
/// inside the model (full-attn contiguous caches + gated-delta recurrent
/// state), so it does NOT use a [`PagedKVPool`]; it relies on the host
/// [`KvPool`] only for the slot's logical `seq_len` to derive `start_pos`.
/// The whole-step decode graph is opt-in (`--qwen35-decode-graph`,
/// single-GPU + device-routed MoE only, rows==1 plans only).
///
/// Scope: prefill stays single-row (mixed plans run per-prefill sub-steps,
/// then one decode sub-batch — the DSv4 executor pattern), uncached
/// full-prefix (each full-attn layer recomputes over its contiguous cache;
/// each linear-attn layer advances the recurrent state in place). A
/// continuous-batching paged + packed-batch path is the stage-2 follow-up
/// (legacy `infer/src/model/qwen35`).
pub(crate) struct Qwen35CudaExecutor {
    model: crate::qwen35::Qwen35Model,
    /// Per-slot KV + recurrent state (one [`crate::qwen35::Qwen35SlotState`] per slot).
    /// A slot's recurrent state is EMPTY until its first request activates it.
    slots: Vec<crate::qwen35::Qwen35SlotState>,
    /// G3 whole-slot capacity spill: an inactive/retracted request parks its
    /// whole slot snapshot here (instead of being dropped + recomputed) and is
    /// restored byte-exact on resume. Routes through the SAME `CudaKvTierStore`
    /// transport as the page-granular `recall_tier` (the unified-tier plan — all
    /// grains move bytes through ONE store kind by opaque `u64` key), so G3 gets
    /// the managed 850 GB DRAM budget + NVMe spill + durability for free instead
    /// of a private DRAM map. Sized for whole-slot snapshots (one entry ≈ one
    /// recurrent-block image), so its count cap = budget / image-bytes (the
    /// budget-aware ~5500 cap, not the old `num_slots*2`). Always built — G3 is
    /// default-on and independent of `--kv-recall` (the page grain). Keyed by the
    /// engine's session key, a namespace disjoint from `recall_tier`'s
    /// `tier_block_u64(slot, page)` keys (separate store ⇒ no aliasing).
    slot_tier: CudaKvTierStore,
    /// Per-unit size (one whole-slot recurrent-block image) for the `slot_tier`
    /// budget, stored so `set_kv_tier_budget_bytes` can
    /// re-budget the tier post-construction without recomputing it (L2).
    /// Free-list of detached recurrent blocks (~147 MiB each). Released here by a
    /// finished request, popped by the next — so only ACTIVE slots hold a block,
    /// not all `num_slots`. Grows to `num_slots` at full concurrency (then HBM ==
    /// the old upfront reservation); idle/partial load costs proportionally less.
    recurrent_pool: Vec<crate::qwen35::RecurrentBlock>,
    /// Persistent forward workspace (exact-shape buffer reuse): forwards are
    /// strictly serial on this executor, so ONE workspace serves every slot.
    /// Mirrors the DSv4 persistent decode scratch (passed `&mut` per forward).
    workspace: crate::qwen35::Qwen35Workspace,
    num_slots: usize,
    /// Whole-step decode graph armed (env gate + TP-single + device-routed
    /// MoE, resolved at construction; `warmup` logs the verdict). ANY capture
    /// failure clears this — eager is the permanent fallback, never fatal.
    decode_graph_armed: bool,
    /// Lazily-built per-slot graph state (`None` until the first gated decode,
    /// and re-`None`d whenever baked addresses go stale: OPD weight
    /// offload/reload, student-LoRA re-merge).
    decode_graph: Option<Qwen35DecodeGraph>,
    /// Lazily-built batched rows>1 decode state: a dedicated `[*, B]`
    /// workspace plus per-layer recurrent-state pointer tables (see
    /// [`crate::qwen35::Qwen35BatchDecodeState`]). `None` until the first
    /// batched decode.
    batch_decode: Option<crate::qwen35::Qwen35BatchDecodeState>,
    /// Session KV-recall opt-in (`--kv-recall`, default off). When true the
    /// recall *cycle* (score → evict → prefetch) layers a working-set
    /// restriction on the SAME paged `full_attn_kv` pool the default path uses;
    /// off → the default attends the full resident page set (no eviction). The
    /// full-attn KV is paged in BOTH cases since the shared-paged migration.
    kv_recall: bool,
    /// Recall budget (sink/local/block/top_k). Mirrors the dense arm.
    recall_cfg: infer_core::RecallConfig,
    /// Per-slot recall state (reps + next-step page plan + evict list).
    recall: Vec<crate::recall::CudaRecallState>,
    /// One-shot warn latch when --kv-recall is requested with a non-BF16 pool.
    #[allow(dead_code)]
    recall_quant_warned: bool,
    /// Shared paged full-attn KV pool (HD256, `num_full` layers), the DEFAULT
    /// full-attn KV substrate since the shared-paged migration — built eagerly
    /// in the constructor, profile-sized from measured free VRAM (not
    /// `num_slots × max_seq_len`). Both the default forward and the `--kv-recall`
    /// cycle read/write THIS one pool; `Option` only so the OPD-offload path can
    /// drop it. Device-only, self-allocating.
    full_attn_kv: Option<PagedKVPool>,
    /// KV pool format (BF16 / FP8E4M3 / INT8). Captured at construction from the
    /// `--kv-cache-dtype` flag; stored so `ensure_kv_pool` can rebuild the pool
    /// with the same format after `release_kv_pool` drops it on the OPD path.
    kv_format: KVFormat,
    /// L3 write-through tier for the recall cycle (host DRAM, optional NVMe spill):
    /// the source of truth for evict-dropped middle blocks. Allocated lazily on
    /// the first `set_kv_recall(true)` and sized to ONE pool page image. Keyed by
    /// `tier_block_u64(slot, logical_page)` — a session-scoped namespace, so slot A
    /// never prefetches slot B's KV. `None` until `--kv-recall` is enabled.
    recall_tier: Option<CudaKvTierStore>,
    /// Per-slot one-step eviction keepalive (the race fix). A page evict-dropped at
    /// decode step N is parked here, NOT returned to `free_pages`, until the START
    /// of step N+1 — by which point step N's attention (and its argmax `ctx.sync()`
    /// at the step boundary) has completed, so `alloc_tokens` can never hand the
    /// in-flight attention's page to the new token. Drained at the top of each
    /// `decode_row_recall` for that slot. Holds (logical, physical) so the parked
    /// physical page is the one actually returned to the pool one step later.
    recall_keepalive: Vec<Vec<(usize, u32)>>,
    /// Resolved per-rank L2 byte budget (`--kv-dram` ÷ world size). Recorded by
    /// `set_kv_tier_budget_bytes` alongside the `slot_tier` rebuild; consumed
    /// when `set_kv_recall(true)` lazily builds `recall_tier`.
    recall_budget_bytes: usize,

    /// Model checkpoint path for deriving the weights epoch tag at durable
    /// NVMe spill time (`set_kv_recall` / `set_kv_tier_disk`).
    #[allow(dead_code)]
    model_path: std::path::PathBuf,
    /// Weights-version tag from the checkpoint (`weights_epoch_tag`). Stamped
    /// into the durable recall manifest so a restart drops stale KV after an
    /// OPD weight update.
    weights_epoch: String,
    /// Operator-provided NVMe root for durable recall spill (`--kv-disk`).
    /// `None` until `set_kv_tier_disk` wires it.
    disk_root: Option<std::path::PathBuf>,
    /// Budget bytes for durable NVMe recall spill (`--kv-disk-limit`).
    /// `None` until `set_kv_tier_disk` wires it.
    disk_budget: Option<usize>,
    /// `mem_fraction_static` + requested page floor captured at construction so
    /// `ensure_kv_pool` can RE-PROFILE and rebuild `full_attn_kv` identically
    /// after `release_kv_pool` dropped it (agent-OPD rollout→writeback→rollout:
    /// the dead pool is freed for the co-resident autograd writeback, then
    /// re-acquired before the next round's rollout). Not on the serve path.
    kv_pool_mem_fraction_static: f64,
    kv_pool_requested_pages: usize,
    /// Recurrent prefix sidecars live in `slot_tier` under `NS_SIDECAR` (blob =
    /// recurrent state + full-attn KV, keyed by token-prefix hash), so they share
    /// the ONE budget-managed store instead of a private capped RAM map — the
    /// 32-cap that always-missed beyond the hottest prefixes is gone (issue #85).
    /// This map is eviction coordination ONLY: the tail host-pool page of each
    /// published prefix → its sidecar key. `release_prefix_pages` drops the blob
    /// when the radix evicts that page, so the sidecar's lifetime rides the radix
    /// blocks — no independent LRU. (The store's own budget is the memory
    /// backstop for any blob whose tail page LRU-evicted before drop reached it.)
    sidecar_page_key: std::collections::HashMap<u32, u64>,
    /// Per-slot recurrent snapshot captured DURING prefill exactly at the
    /// last-page boundary `L* = align_down16(prompt_len - 1)` — the restore
    /// target for an exact resend (un-popped for non-aligned, pop-trimmed for
    /// aligned; see `prefix.rs:72`). The device recurrent state has no position
    /// index and cannot rewind, so a snapshot taken at prompt-completion (`P`)
    /// keyed at `L*` would bake the residue `[L*..P]` and double-advance it on
    /// restore's re-prefill — corrupting page-aligned lengths. Capturing at `L*`
    /// makes snapshot-position == key-position. Consumed by
    /// `save_recurrent_sidecar`; cleared on the next request (`start_pos == 0`)
    /// so no cross-request state leaks.
    prefill_boundary_snapshot: Vec<Option<(usize, crate::qwen35::Qwen35RecurrentSnapshot)>>,
    /// DSpark block-draft runtime (`--spec-type dspark`): draft head + per-slot
    /// ctx caches + tap/scratch buffers. `None` (the default) keeps the
    /// baseline executor byte-identical — nothing is allocated or captured.
    dspark: Option<crate::qwen35::dspark::Qwen35DsparkExec>,
}

impl std::fmt::Debug for Qwen35CudaExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen35CudaExecutor")
            .field("model", &self.model)
            .field("num_slots", &self.num_slots)
            .field("decode_graph_armed", &self.decode_graph_armed)
            .field(
                "captured_decode_slots",
                &self
                    .decode_graph
                    .as_ref()
                    .map_or(0, |dg| dg.graphs.iter().filter(|g| g.is_captured()).count()),
            )
            .finish()
    }
}

impl Qwen35CudaExecutor {
    /// `|_| false`: demote/promote is a no-op for Qwen35, so demoted pages are never restorable.
    pub(crate) fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
        pages_only_reusable_prefix_blocks(blocks, |_| false)
    }

    /// Serialize the slot's recurrent state + full-attn KV as ONE atomic blob
    /// and store it into `slot_tier` under `NS_SIDECAR`, keyed by the token hash
    /// of the published radix prefix. Records the prefix's tail page → key so
    /// `release_sidecar_pages` drops the blob when the radix evicts that page.
    ///
    /// Boundary: the sidecar is keyed at `L* = align_down16(tokens.len() - 1)` —
    /// the largest page boundary strictly below the sequence length, i.e. the
    /// restore target a future exact resend lands on (pop-trimmed for aligned,
    /// un-popped for non-aligned; `prefix.rs:72`). When a prefill boundary
    /// snapshot was captured at exactly `L*` (`submit_prefill_row`), it is used
    /// verbatim — snapshot-position == key-position, so restore's re-prefill of
    /// `[L*..len]` advances the recurrent state exactly once. Otherwise (finish/
    /// decode publish, recall path, or tiny prompt) fall back to the live
    /// snapshot keyed at the sealed page grain — the pre-existing best-effort
    /// path for supersequence (agentic follow-up) reuse.
    pub(crate) fn save_recurrent_sidecar(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
        prefix_pages: &[u32],
    ) -> anyhow::Result<()> {
        if !self.slots[slot].has_recurrent() {
            return Ok(()); // pure full-attn path — nothing to capture
        }
        let boundary = tokens.len().saturating_sub(1) / SUPPORTED_PAGE_SIZE * SUPPORTED_PAGE_SIZE;
        let pending = self.prefill_boundary_snapshot[slot].take();
        let (mat_len, mut snap) = match pending {
            Some((pos, snap)) if pos == boundary && boundary > 0 => (pos, snap),
            _ => {
                // Legacy best-effort: raw seq_len aligns to a page ~1/16 of the
                // time, so page-align to keep capture-key == restore's matched_len.
                let mat_len = matched_len
                    .min(self.slots[slot].seq_len())
                    .min(tokens.len())
                    / SUPPORTED_PAGE_SIZE
                    * SUPPORTED_PAGE_SIZE;
                if mat_len == 0 {
                    return Ok(());
                }
                (
                    mat_len,
                    self.slots[slot].snapshot_recurrent(&self.model.ctx)?,
                )
            }
        };
        let key = crate::qwen35::hash_prefix_tokens(&tokens[..mat_len]);
        // snapshot_recurrent synchronizes the stream so pages are flushed before D2H.
        if let Some(pool) = self.full_attn_kv.as_ref() {
            // full-attn KV is position-indexed and append-only, so pages [0..mat_len)
            // are byte-identical whether copied at `mat_len` or at the current end.
            let n_pages = mat_len / SUPPORTED_PAGE_SIZE;
            let all_pages = pool.page_indices(slot);
            let pages = all_pages[..n_pages.min(all_pages.len())].to_vec();
            if !pages.is_empty() {
                match pool.copy_pages_to_host(&self.model.ctx, &pages) {
                    Ok(data) => snap.full_attn_kv = Some(data),
                    Err(e) => {
                        log::warn!(
                            "slot {slot}: full-attn KV D2H failed: {e}; skipping sidecar entry"
                        );
                        return Ok(()); // don't store an incomplete (KV-less) blob
                    }
                }
            }
        }
        // Atomic: recurrent + full-attn KV as ONE blob under ONE key.
        let bytes = snap.to_bytes();
        if self
            .slot_tier
            .insert_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, key, &bytes)
        {
            // Coordinate eviction off the last page the sidecar COVERS (leaves
            // evict deepest-first, so this drops the blob the moment its own
            // prefix erodes) — `mat_len` may be one page short of the sealed run.
            let cover_idx = (mat_len / SUPPORTED_PAGE_SIZE).saturating_sub(1);
            if let Some(&tail) = prefix_pages.get(cover_idx).or_else(|| prefix_pages.last()) {
                if let Some(old) = self.sidecar_page_key.insert(tail, key) {
                    if old != key {
                        self.slot_tier
                            .remove_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, old);
                    }
                }
            }
        }
        Ok(())
    }

    /// Radix evicted these host-pool prefix pages: drop the sidecar blob keyed to
    /// any of them. Eviction rides the radix — no independent sidecar LRU.
    pub(crate) fn release_sidecar_pages(&mut self, pages: &[u32]) {
        for &page in pages {
            if let Some(key) = self.sidecar_page_key.remove(&page) {
                self.slot_tier
                    .remove_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, key);
            }
        }
    }

    /// Acquires recurrent buffers before restore so the tail-prefill sees populated state.
    /// Also resets device KV pool seq_len to matched_len for the prefill_row_paged_default invariant.
    /// Cache miss → zeroed recurrent (graceful degradation, not a hard error).
    pub(crate) fn restore_recurrent_sidecar(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
        _prefix_pages: &[u32],
    ) -> anyhow::Result<()> {
        let matched_len = matched_len.min(tokens.len());
        if std::env::var_os("ARLE_KVDRIFT_DEBUG").is_some() {
            let pool_len = self
                .full_attn_kv
                .as_ref()
                .map_or(usize::MAX, |p| p.seq_len(slot));
            eprintln!(
                "[kvdrift] RESTORE-SIDECAR slot={} matched_len={} prompt_tokens={} prefix_pages={} slot.seq_len(before)={} pool.seq_len(before)={}",
                slot,
                matched_len,
                tokens.len(),
                _prefix_pages.len(),
                self.slots[slot].seq_len(),
                pool_len,
            );
        }
        let key = crate::qwen35::hash_prefix_tokens(&tokens[..matched_len]);
        let (num_linear, gdr_len, conv_len) = self.model.recurrent_dims();
        // Reused slot: start_pos != 0 skips the normal release+acquire in submit_prefill_row.
        self.slots[slot].release_recurrent(&mut self.recurrent_pool);
        self.slots[slot].acquire_recurrent(
            &self.model.ctx,
            num_linear,
            gdr_len,
            conv_len,
            &mut self.recurrent_pool,
        )?;

        // A sidecar-restored prefix has no draft ctx features for the matched
        // span — clear stale DSpark state; the tail prefill rebases the draft
        // ctx at start_pos (suffix-only drafting).
        if let Some(df) = self.dspark.as_mut().and_then(|ds| ds.slots[slot].as_mut()) {
            df.reset();
        }
        // Read the atomic blob from the tier (transparently spans L1/L2 host →
        // L3 disk); a MISS or corrupt/foreign payload deserializes to None and
        // falls through to the clean-recompute reset below — never a partial restore.
        let snap = self
            .slot_tier
            .read_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, key)
            .ok()
            .and_then(|bytes| crate::qwen35::Qwen35RecurrentSnapshot::from_bytes(&bytes).ok());

        // restore_recurrent_from_snapshot doesn't advance seq_len; set it for submit_decode_row invariant.
        match snap.as_ref() {
            Some(s) => self.slots[slot].restore_recurrent_from_snapshot(&self.model.ctx, s)?,
            None => {
                // ponytail: MISS → reset slot state, then return Err so the caller
                // falls back to full recompute. Must clean up full_attn_kv and seq_len
                // here before returning — the caller's Err handler won't do it.
                if let Some(pool) = self.full_attn_kv.as_mut() {
                    pool.free_slot(slot);
                }
                self.slots[slot].set_seq_len(0);
                return Err(anyhow::anyhow!(
                    "no recurrent sidecar for prefix matched_len={matched_len} \
                     (key={key:#018x}); falling back to full recompute"
                ));
            }
        }
        self.slots[slot].set_seq_len(matched_len);

        // Must run regardless of sidecar hit/miss: free prior occupant, alloc fresh pages.
        if let Some(pool) = self.full_attn_kv.as_mut() {
            pool.free_slot(slot);
            let new_pages = pool.alloc_tokens(slot, matched_len).map_err(|e| {
                anyhow::anyhow!("device pool prefix alloc failed for slot {slot}: {e}")
            })?;
            if let Some(kv_data) = snap.as_ref().and_then(|s| s.full_attn_kv.as_deref()) {
                pool.copy_pages_from_host(&mut self.model.ctx, &new_pages, kv_data)
                    .map_err(|e| {
                        anyhow::anyhow!("device pool KV H2D restore failed for slot {slot}: {e}")
                    })?;
            }
            // Stream-order: H2D above is ordered before subsequent kernels; no explicit sync needed.
        }
        Ok(())
    }

    /// Whole-slot swap is rank-local bytes plus TP-wide scalar consensus
    /// ([`Self::tp_min_usize`] in demote/promote), mirroring DSv4's arm.
    pub(crate) fn kv_slot_tier_enabled(&self) -> bool {
        true
    }

    pub(crate) fn kv_tier_host_demoted_pages(&self) -> usize {
        self.slot_tier.host_demoted_pages()
    }

    pub(crate) fn kv_tier_disk_pages(&self) -> usize {
        self.slot_tier.disk_pages()
    }

    fn tp_min_usize(&self, value: usize, what: &str) -> Result<usize> {
        let capped = i32::try_from(value.min(i32::MAX as usize)).unwrap_or(i32::MAX);
        self.model
            .tp
            .all_reduce_min_scalar_i32(&self.model.ctx, capped)
            .map(|v| v.max(0) as usize)
            .map_err(|e| anyhow::anyhow!("Qwen3.6 TP min-reduce {what} failed: {e}"))
    }

    /// `BackendExecutor::tp_sync_min` — see there for why the scheduler needs
    /// this (2026-07-05 TP=4 admission livelock).
    pub(crate) fn tp_sync_min(&self, local: usize) -> Result<usize> {
        self.tp_min_usize(local, "admission free pages")
    }

    /// Demote `slot`'s entire device state into the host `slot_tier` under `key`.
    /// The copy is complete before returning (`swap_out_image` ends in
    /// `ctx.sync()`), so the engine may free the slot immediately. The image is
    /// serialized byte-exact ([`Qwen35SlotImage::to_bytes`]) and handed to the
    /// shared `CudaKvTierStore` transport, which manages the 850 GB DRAM budget
    /// (+ optional NVMe spill). Returns `Ok(false)` when the tier is at budget
    /// on ANY rank (engine falls back to plain recompute, same contract as the
    /// old cap). Exactly TWO collectives on every path (capture consensus,
    /// insert consensus), so the lockstep collective count is rank-invariant.
    pub(crate) fn demote_slot(&mut self, slot: usize, key: u64) -> Result<bool> {
        ensure!(
            slot < self.num_slots,
            "Qwen3.6 demote slot {slot} outside executor slots {}",
            self.num_slots
        );
        let image = {
            let Self {
                model,
                slots,
                full_attn_kv,
                recurrent_pool,
                ..
            } = &mut *self;
            let pool = full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (whole-slot demote)");
            slots[slot].swap_out_image(&model.ctx, slot, pool, recurrent_pool)
        };
        let capture_ok = usize::from(image.is_ok());
        if self.tp_min_usize(capture_ok, "slot demote capture")? == 0 {
            return Err(image.err().unwrap_or_else(|| {
                anyhow::anyhow!("peer rank failed Qwen3.6 slot demote capture")
            }));
        }
        // Chunked like DSv4 (16 MiB store pages): the whole image never fit a
        // recurrent-floor-sized page — the review-6 blocker where every insert
        // was refused post-oversize-guard and park degraded to pure recompute.
        // Per-rank DRAM headroom can diverge, so the verdict is min-reduced; on
        // a mixed verdict, locally-successful ranks roll their insert back so
        // no rank keeps a blob the others lack.
        let bytes = image?.to_bytes();
        let inserted = self
            .slot_tier
            .insert_chunked(NS_SLOT, NS_SLOT_CHUNK, key, &bytes);
        if self.tp_min_usize(usize::from(inserted), "slot demote insert")? == 0 {
            if inserted {
                self.slot_tier.remove_chunked(NS_SLOT, NS_SLOT_CHUNK, key);
            }
            return Ok(false);
        }
        Ok(true)
    }

    /// Restore the whole-slot snapshot stored under `key` into `slot`. The engine
    /// resumes decode at the demoted position right after this returns, and drops
    /// the entry via [`Self::drop_kv_slot_entries`] — the entry intentionally
    /// stays in the tier here. `swap_in_image` ends in `ctx.sync()`, so the
    /// device restore is complete before the host image can be dropped. Exactly
    /// TWO collectives on every path (read+parse consensus, restore consensus) —
    /// nothing rank-local errs between them.
    pub(crate) fn promote_slot(&mut self, key: u64, slot: usize) -> Result<()> {
        ensure!(
            slot < self.num_slots,
            "Qwen3.6 promote slot {slot} outside executor slots {}",
            self.num_slots
        );
        // The slot image carries no draft ctx cache — clear stale state; the
        // first warm decode rebases the draft ctx at the promoted position.
        if let Some(df) = self.dspark.as_mut().and_then(|ds| ds.slots[slot].as_mut()) {
            df.reset();
        }
        // Read the serialized image out of the tier (host hit or NVMe read) and
        // deserialize it byte-exact before touching the device.
        let image = self
            .slot_tier
            .read_chunked(NS_SLOT, NS_SLOT_CHUNK, key)
            .map_err(|err| anyhow::anyhow!("Qwen3.6 whole-slot tier read key {key}: {err}"))
            .and_then(|bytes| crate::qwen35::Qwen35SlotImage::from_bytes(&bytes));
        let image_ok = usize::from(image.is_ok());
        if self.tp_min_usize(image_ok, "slot promote read")? == 0 {
            return Err(image
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("peer rank failed Qwen3.6 slot promote read")));
        }
        let image = image?;
        // Infallible config data — safe between the collectives.
        let (num_linear, gdr_len, conv_len) = self.model.recurrent_dims();
        let restored = {
            let Self {
                model,
                slots,
                full_attn_kv,
                recurrent_pool,
                ..
            } = &mut *self;
            let pool = full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (whole-slot promote)");
            slots[slot].swap_in_image(
                &model.ctx,
                slot,
                pool,
                recurrent_pool,
                num_linear,
                gdr_len,
                conv_len,
                &image,
            )
        };
        let restore_ok = usize::from(restored.is_ok());
        if self.tp_min_usize(restore_ok, "slot promote restore")? == 0 {
            return Err(restored.err().unwrap_or_else(|| {
                anyhow::anyhow!("peer rank failed Qwen3.6 slot promote restore")
            }));
        }
        restored
    }

    pub(crate) fn drop_kv_slot_entries(&mut self, keys: &[u64]) {
        for &key in keys {
            self.slot_tier.remove_chunked(NS_SLOT, NS_SLOT_CHUNK, key);
        }
    }

    pub(crate) fn from_qwen35_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
        kv_dtype: CudaKvCacheDtype,
        mem_fraction_static: f64,
        dspark_draft_model: Option<&Path>,
        dspark_conf_threshold: f32,
    ) -> Result<Self> {
        let total_t0 = Instant::now();
        ensure!(
            num_slots > 0,
            "Qwen35CudaExecutor requires at least one slot"
        );
        ensure!(
            total_pages > 0,
            "Qwen35CudaExecutor requires at least one KV page"
        );
        // `max_seq_len` is the per-request token ceiling (the model's positional
        // budget + the host CudaKvPool admission span). The full-attn KV is now a
        // SHARED profile-sized paged pool (built below), not a per-slot
        // contiguous cache, so this no longer scales VRAM by num_slots.
        let max_seq_len = total_pages * SUPPORTED_PAGE_SIZE;
        let model_t0 = Instant::now();
        // `None`: the checkpoint-native NextN-MTP head stays unwired here; the
        // baseline load is byte-identical (no draft head loaded). DSpark is the
        // external block drafter, loaded below from its own checkpoint dir.
        let mut model =
            crate::qwen35::Qwen35Model::from_safetensors(model_path.as_ref(), max_seq_len, None)?;
        cuda_startup_log(
            "qwen35_model_load",
            model_t0,
            format_args!(
                "requested_slots={num_slots} total_pages={total_pages} max_seq_len={max_seq_len}"
            ),
        );
        // DSpark draft head: loaded BEFORE the KV budget/pool profiling so its
        // weights are subtracted from the measured free VRAM.
        let dspark_head = dspark_draft_model
            .map(|dir| {
                ensure!(
                    model.tp.is_single(),
                    "--spec-type dspark is single-GPU only (draft lm_head/argmax are rank-local)"
                );
                let head = crate::qwen35::dspark::load_dspark_head(
                    &model.ctx,
                    dir,
                    max_seq_len,
                    model.config.hidden_size,
                    model.config.num_hidden_layers,
                    model.config.vocab_size,
                    dspark_conf_threshold,
                )?;
                model.set_spec_draft_tokens(head.block_size());
                log::info!(
                    "CUDA Qwen3.6 DSpark drafter loaded from {}: mode={} block={} taps={:?}",
                    dir.display(),
                    head.mode_label(),
                    head.block_size(),
                    head.target_layer_ids(),
                );
                Ok(head)
            })
            .transpose()?;
        // Dynamic KV mem budget (unified with DSv4 via the infer-seam kernel):
        // clamp num_slots to what post-weights free VRAM affords. Qwen3.5/3.6
        // previously admitted the requested count as-is → OOM at large
        // max_seq_len. Deterministic + NCCL min-reduced ⇒ TP-consistent.
        let budget_t0 = Instant::now();
        // Reclaim retained loading scratch before the budget measures free VRAM
        // (see the DSv4 path above — same starvation otherwise).
        if let Err(e) = model.ctx.trim_memory_pool() {
            log::warn!("pre-KV-budget trim_memory_pool failed (non-fatal): {e}");
        }
        let dspark_slot_bytes = dspark_head.as_ref().map_or(0, |h| h.slot_state_bytes());
        let num_slots = model.kv_budget_num_slots(num_slots, dspark_slot_bytes)?;
        cuda_startup_log(
            "qwen35_kv_budget",
            budget_t0,
            format_args!("effective_slots={num_slots}"),
        );
        let slots_t0 = Instant::now();
        // Empty slots — zero recurrent HBM upfront. Each slot's ~147 MiB
        // recurrent block is drawn from `recurrent_pool` on its first request
        // (`acquire_recurrent` at the `start_pos == 0` prefill) and returned on
        // request finish. The pool starts empty and grows on demand.
        let slots: Vec<_> = (0..num_slots).map(|_| model.new_slot_state()).collect();
        cuda_startup_log(
            "qwen35_slot_alloc",
            slots_t0,
            format_args!("slots={num_slots} max_seq_len={max_seq_len}"),
        );

        // Shared paged full-attn KV pool — the DEFAULT substrate (Phase 2 of the
        // shared-paged-KV migration). Built EAGERLY here, profile-sized from
        // MEASURED free VRAM after weights load (SGLang's mem_fraction_static),
        // NOT `num_slots × max_seq_len` (the per-slot contiguous waste). The same
        // build is re-run by `ensure_kv_pool` after `release_kv_pool` drops it on
        // the agent-OPD writeback path — extracted into `build_full_attn_kv_pool`.
        let pool_t0 = Instant::now();
        let requested_pages = total_pages.max(1);
        let kv_format = kv_dtype.kv_format();
        let full_attn_kv = Self::build_full_attn_kv_pool(
            &model,
            num_slots,
            requested_pages,
            mem_fraction_static,
            kv_format,
            dspark_slot_bytes,
        )?;
        cuda_startup_log("qwen35_paged_pool_alloc", pool_t0, format_args!("built"));

        // G3 whole-slot spill tier: snapshots stored as 16 MiB chunked blobs
        // (manifest + chunks, the DSv4 pattern) — a whole image never fits one
        // fixed page, and the store's size contract is per-page. Same
        // `CudaKvTierStore` transport as the page-granular recall tier — the
        // unified plan: every grain moves bytes through ONE store kind. NVMe
        // spill is opt-in via `set_kv_tier_disk` (shared with the dense arm).
        let tier_budget_bytes = default_t1_budget_bytes(DEFAULT_DRAM_FRACTION);
        let slot_tier = CudaKvTierStore::with_budget(tier_budget_bytes, BLOB_CHUNK_BYTES);

        // Whole-step decode graph: env opt-in ∧ single-GPU (NCCL all-reduce is
        // not graph-capturable on this stack — TP≥2 stays eager, same as
        // dense) ∧ every layer's decode step is a pure device-kernel sequence.
        let decode_graph_armed = qwen35_decode_graph_enabled()
            && model.tp.is_single()
            && model.decode_graph_unsupported_reason().is_none();
        let model_path_buf = model_path.as_ref().to_path_buf();
        let weights_epoch = crate::kv_tier::weights_epoch_tag(&model_path_buf);
        let executor = Self {
            model,
            slots,
            slot_tier,
            recurrent_pool: Vec::new(),
            workspace: crate::qwen35::Qwen35Workspace::new(),
            num_slots,
            decode_graph_armed,
            decode_graph: None,
            batch_decode: None,
            kv_recall: false,
            recall_cfg: crate::recall::default_recall_config(),
            recall: (0..num_slots)
                .map(|_| crate::recall::CudaRecallState::default())
                .collect(),
            recall_quant_warned: false,
            full_attn_kv: Some(full_attn_kv),
            kv_format,
            recall_tier: None,
            recall_keepalive: (0..num_slots).map(|_| Vec::new()).collect(),
            recall_budget_bytes: tier_budget_bytes,
            model_path: model_path_buf,
            weights_epoch,
            disk_root: None,
            disk_budget: None,
            kv_pool_mem_fraction_static: mem_fraction_static,
            kv_pool_requested_pages: total_pages.max(1),
            sidecar_page_key: std::collections::HashMap::new(),
            prefill_boundary_snapshot: (0..num_slots).map(|_| None).collect(),
            dspark: dspark_head.map(|h| crate::qwen35::dspark::Qwen35DsparkExec::new(h, num_slots)),
        };
        cuda_startup_log(
            "qwen35_executor_total",
            total_t0,
            format_args!("slots={num_slots} max_seq_len={max_seq_len}"),
        );
        Ok(executor)
    }

    /// Build the shared paged full-attn KV pool, profile-sized from MEASURED free
    /// VRAM (SGLang's `mem_fraction_static`), floored at `requested_pages`. The
    /// constructor's eager build and `ensure_kv_pool`'s post-release rebuild both
    /// go through here so the re-acquired pool matches the original sizing recipe.
    fn build_full_attn_kv_pool(
        model: &crate::qwen35::Qwen35Model,
        num_slots: usize,
        requested_pages: usize,
        mem_fraction_static: f64,
        kv_format: KVFormat,
        dspark_slot_bytes: usize,
    ) -> Result<PagedKVPool> {
        let num_full = model.config.num_full_attention_layers();
        let local_kv_heads = model.local_kv_heads();
        let head_dim = model.config.head_dim;
        let cell_bytes_per_token =
            PagedKVPool::budget_bytes_for_tokens(num_full, local_kv_heads, head_dim, 1, kv_format)
                as u64;
        // H1: the per-slot recurrent block (gdr + conv) is reserved out of the
        // SAME free VRAM `kv_budget_num_slots` clamped against. Subtract that
        // reservation from free BEFORE profiling the full-attn pool, so
        // weights + recurrent×slots + pool ≤ mem_fraction×free (mirrors DSv4
        // dsv4.rs:1687-1691). Per-rank free, num_slots already min-reduced.
        let (_per_slot, _kv_bytes, gdr_bytes, conv_bytes) = model.per_slot_kv_bytes();
        // DSpark draft ctx caches are per-slot lazily-allocated device buffers —
        // reserve them here too, or the pool eats the VRAM the first dspark
        // prefill needs.
        let per_slot_recurrent = gdr_bytes
            .saturating_add(conv_bytes)
            .saturating_add(dspark_slot_bytes);
        let recurrent_reservation = per_slot_recurrent.saturating_mul(num_slots) as u64;
        let total_pool_pages = match model.ctx.mem_info_bytes() {
            Ok((free, total)) => {
                let free_after_recurrent = (free as u64).saturating_sub(recurrent_reservation);
                let profiled_tokens = infer_seam::profile_kv_pool_tokens(
                    free_after_recurrent,
                    total as u64,
                    cell_bytes_per_token,
                    mem_fraction_static,
                );
                let profiled_pages = (profiled_tokens / SUPPORTED_PAGE_SIZE as u64) as usize;
                let sized = profiled_pages.max(requested_pages).max(1);
                log::info!(
                    "CUDA Qwen3.6 full-attn KV pool profiled from measured VRAM: free {}MB / \
                     total {}MB, recurrent reservation {}MB ({num_slots} slots × {}MB), \
                     free_after_recurrent {}MB, mem_fraction_static {mem_fraction_static}, cell \
                     {cell_bytes_per_token}B/tok ({num_full} full-attn layers × {local_kv_heads} \
                     kv-heads × {head_dim} hd) -> max_total_tokens {profiled_tokens} \
                     ({profiled_pages} pages); requested floor {requested_pages} pages \
                     -> sizing {sized} pages",
                    free >> 20,
                    total >> 20,
                    recurrent_reservation >> 20,
                    per_slot_recurrent >> 20,
                    free_after_recurrent >> 20,
                );
                sized
            }
            Err(e) => {
                log::warn!(
                    "CUDA Qwen3.6 full-attn KV pool: free-VRAM probe failed ({e}); falling back \
                     to requested floor {requested_pages} pages"
                );
                requested_pages
            }
        };
        let pool_token_budget = total_pool_pages * SUPPORTED_PAGE_SIZE;
        let pool_budget_bytes = PagedKVPool::budget_bytes_for_tokens(
            num_full,
            local_kv_heads,
            head_dim,
            pool_token_budget,
            kv_format,
        );
        let full_attn_kv = PagedKVPool::with_format(
            &model.ctx,
            num_full,
            local_kv_heads,
            head_dim,
            num_slots,
            pool_budget_bytes,
            kv_format,
        )?;
        ensure!(
            full_attn_kv.page_size == SUPPORTED_PAGE_SIZE,
            "Qwen3.6 full-attn paged pool page_size={} != {SUPPORTED_PAGE_SIZE}",
            full_attn_kv.page_size
        );
        Ok(full_attn_kv)
    }

    /// Drop the shared paged full-attn KV pool, returning its HBM to the device
    /// async pool (trimmed to the OS so a co-resident allocator sees it free).
    /// agent-OPD ONLY: the masked-CE writeback's `forward_hidden_states` is a
    /// FRESH autograd forward that does NOT read this engine's KV cache, so the
    /// rollout pool is DEAD during the writeback — freeing it widens the writeback
    /// headroom. `ensure_kv_pool` re-acquires it before the next round's rollout.
    /// Precondition: all in-flight rollout work has synced (the round's rollout
    /// completed + LoRA synced before the writeback closure runs). No-op if the
    /// pool was already released. Errors if any slot still holds resident pages
    /// (a live request would lose its KV) — agent-OPD finishes all rollouts before
    /// the writeback, so the pool is empty.
    pub(crate) fn release_kv_pool(&mut self) -> Result<()> {
        let Some(pool) = self.full_attn_kv.as_ref() else {
            return Ok(());
        };
        let freed = pool.device_bytes();
        // Drain in-flight device work referencing the pool BEFORE dropping it (no
        // kernel may still read the slices we are about to free).
        self.model.ctx.sync()?;
        // Drop enqueues `cuMemFreeAsync` for every pool slice on the model stream
        // — the frees are ASYNC, so they have NOT executed yet here.
        self.full_attn_kv = None;
        // Sync AGAIN so the enqueued async frees actually complete and the blocks
        // return to the device async pool; only THEN can the trim return them to
        // the OS. Without this second sync the trim runs before any block is back
        // in the pool → nothing trimmed → `mem_get_info` still shows them used (the
        // bug the first version hit: pool dropped but VRAM not reclaimed).
        self.model.ctx.sync()?;
        // Return the freed async-pool blocks to the OS so the co-resident autograd
        // writeback's allocator + `mem_get_info` see them free (the pool's release
        // threshold is u64::MAX, so a drop alone only caches the blocks). The
        // device default async pool is per-DEVICE (shared across the infer + train
        // contexts on this GPU), so the trimmed bytes are available to autograd.
        if let Err(e) = self.model.ctx.trim_memory_pool() {
            log::warn!("release_kv_pool: trim_memory_pool failed (non-fatal): {e}");
        }
        log::info!(
            "Qwen3.6 released full-attn KV pool: freed {}MB (agent-OPD writeback headroom)",
            freed >> 20
        );
        Ok(())
    }

    /// Re-acquire the shared paged full-attn KV pool after `release_kv_pool`
    /// dropped it (agent-OPD next-round rollout). Re-profiles from current free
    /// VRAM with the construction-time `mem_fraction_static` + requested floor.
    /// No-op (idempotent) if the pool is already resident.
    pub(crate) fn ensure_kv_pool(&mut self) -> Result<()> {
        if self.full_attn_kv.is_some() {
            return Ok(());
        }
        let pool = Self::build_full_attn_kv_pool(
            &self.model,
            self.num_slots,
            self.kv_pool_requested_pages,
            self.kv_pool_mem_fraction_static,
            self.kv_format,
            self.dspark
                .as_ref()
                .map_or(0, |ds| ds.head.slot_state_bytes()),
        )?;
        log::info!(
            "Qwen3.6 re-acquired full-attn KV pool: {}MB (agent-OPD next-round rollout)",
            pool.device_bytes() >> 20
        );
        self.full_attn_kv = Some(pool);
        Ok(())
    }

    /// Boot-time decode-graph verdict log (mirrors the dense `warmup` info
    /// messages). Capture itself is lazy — one whole-step capture per slot on
    /// its first gated decode (after `CudaGraphState`'s universal eager warm
    /// run), so unused slots never pay capture/instantiation memory.
    pub(crate) fn warmup(&mut self) -> Result<()> {
        let warmup_t0 = Instant::now();
        let dense_t0 = Instant::now();
        let (warmed_shapes, warm_m) = self.model.warm_fp8_deepgemm_dense_prefill()?;
        cuda_startup_log(
            "qwen35_warm_dense_deepgemm",
            dense_t0,
            format_args!("shapes={warmed_shapes} warm_m={warm_m}"),
        );
        if warmed_shapes > 0 {
            info!(
                "Qwen3.5 FP8 dense DeepGEMM warmed {warmed_shapes} projection shape(s) at M={warm_m}"
            );
        }
        let grouped_t0 = Instant::now();
        let (grouped_shapes, grouped_tokens, grouped_min_rows, grouped_max_rows) =
            self.model.warm_fp8_deepgemm_grouped_prefill()?;
        cuda_startup_log(
            "qwen35_warm_grouped_deepgemm",
            grouped_t0,
            format_args!(
                "shapes={grouped_shapes} tokens={grouped_tokens} rows={grouped_min_rows}..{grouped_max_rows}"
            ),
        );
        if grouped_shapes > 0 {
            info!(
                "Qwen3.5 FP8 grouped DeepGEMM warmed {grouped_shapes} GEMM shape(s) at tokens<={grouped_tokens} rows={grouped_min_rows}..{grouped_max_rows}"
            );
        }
        if !qwen35_decode_graph_enabled() {
            info!(
                "Qwen3.5 whole-step decode graph disabled \
                 (set --qwen35-decode-graph to enable)"
            );
            cuda_startup_log(
                "qwen35_warmup_total",
                warmup_t0,
                format_args!("graph=disabled"),
            );
            return Ok(());
        }
        if !self.model.tp.is_single() {
            info!(
                "Qwen3.5 whole-step decode graph disabled under tensor parallelism \
                 (world_size>1, NCCL collectives are not graph-capturable); \
                 using eager forward"
            );
            cuda_startup_log(
                "qwen35_warmup_total",
                warmup_t0,
                format_args!("graph=tp_disabled"),
            );
            return Ok(());
        }
        if let Some(reason) = self.model.decode_graph_unsupported_reason() {
            info!("Qwen3.5 whole-step decode graph disabled: {reason}; using eager forward");
            cuda_startup_log(
                "qwen35_warmup_total",
                warmup_t0,
                format_args!("graph=unsupported"),
            );
            return Ok(());
        }
        debug_assert!(self.decode_graph_armed);
        info!(
            "Qwen3.5 whole-step decode graph ARMED ({} slots; lazy capture per slot, \
             one eager warm run before each first capture; eager fallback on any failure)",
            self.num_slots
        );
        cuda_startup_log(
            "qwen35_warmup_total",
            warmup_t0,
            format_args!("graph=armed"),
        );
        Ok(())
    }

    /// Resolved per-rank L2 byte cap (`--kv-dram` ÷ world size) for the G3
    /// `slot_tier`, also recorded for the lazily-built recall tier. Pre-serve
    /// only (drops any existing entries).
    pub(crate) fn set_kv_tier_budget_bytes(&mut self, bytes: usize) {
        self.recall_budget_bytes = bytes;
        self.slot_tier = CudaKvTierStore::with_budget(bytes, BLOB_CHUNK_BYTES);
    }

    /// Attach NVMe spill (`--kv-disk`): an ephemeral disk level under the
    /// G3 `slot_tier` (whole-slot capacity spill) plus a durable level for the
    /// recall tier (attached now if built, else stashed for `set_kv_recall`).
    /// Pre-serve only. The budget is a per-store cap, not a reservation — both
    /// stores are sparse mmaps, so disk is consumed only by actual spill.
    pub(crate) fn set_kv_tier_disk(
        &mut self,
        root: std::path::PathBuf,
        budget_bytes: usize,
    ) -> bool {
        self.disk_root = Some(root.clone());
        self.disk_budget = Some(budget_bytes);
        let recall_attached = match self.recall_tier.as_mut() {
            Some(tier) => {
                let page_bytes = tier.page_bytes();
                tier.set_disk_durable(
                    root.clone(),
                    budget_bytes,
                    page_bytes,
                    self.weights_epoch.clone(),
                )
            }
            None => false,
        };
        self.slot_tier
            .set_disk(root, budget_bytes, BLOB_CHUNK_BYTES)
            || recall_attached
    }

    /// Opt into session KV-recall (`--kv-recall`, default off). The paged
    /// full-attn pool (`full_attn_kv`) is ALWAYS resident since the shared-paged
    /// migration — the DEFAULT forward already attends it (full resident set, no
    /// eviction). Enabling recall layers the eviction/scoring *cycle* on the
    /// SAME pool (working-set restriction) and lazily builds its L3 tier on the
    /// first enable. Default-off keeps the full resident attention.
    pub(crate) fn set_kv_recall(&mut self, enabled: bool) -> Result<()> {
        ensure!(
            !(enabled && self.dspark.is_some()),
            "--kv-recall is not supported with --spec-type dspark (the verify \
             forward would race the recall eviction cycle)"
        );
        self.kv_recall = enabled;
        if enabled && self.recall_tier.is_none() {
            // L3 write-through tier: source of truth for evict-dropped middle
            // blocks. One entry == one pool page image (all `num_full` layers,
            // K+V). Host-DRAM budget is the resolved per-rank `--kv-dram` share
            // (same cap as the slot tier); NVMe spill is opt-in via the
            // prefix-tier `--kv-disk` wiring. Reuses the SAME `CudaKvTierStore`
            // transport the dense arm's prefix/write-through tier uses (R5 —
            // one store kind).
            let page_bytes = self
                .full_attn_kv
                .as_ref()
                .map(|p| p.storage_bytes_per_page())
                .ok_or_else(|| {
                    anyhow::anyhow!("--kv-recall: full-attn paged pool not allocated")
                })?;
            let mut tier = CudaKvTierStore::with_budget(self.recall_budget_bytes, page_bytes);
            // Try loading prior session durable NVMe spill if disk is configured.
            // Falls through to set_disk_durable on first run or epoch mismatch.
            if let (Some(root), Some(budget)) = (self.disk_root.as_ref(), self.disk_budget) {
                let loaded =
                    tier.load(root.clone(), budget, page_bytes, self.weights_epoch.clone());
                if !loaded {
                    tier.set_disk_durable(
                        root.clone(),
                        budget,
                        page_bytes,
                        self.weights_epoch.clone(),
                    );
                }
            }
            self.recall_tier = Some(tier);
        }
        Ok(())
    }

    /// Whether the full-attn KV path is paged (the shared pool is resident).
    /// TRUE by default since the shared-paged migration — the default forward
    /// attends the full resident page set. (`Option` is `None` only after an
    /// OPD weight offload dropped the pool.)
    fn full_attn_paged(&self) -> bool {
        self.full_attn_kv.is_some()
    }

    /// Actual shared full-attn paged-pool page count, so the host admission pool
    /// mirrors the device pool 1:1 (like dense). `None` only if the pool was
    /// dropped (OPD offload), in which case the caller falls back to the
    /// requested config value.
    pub(crate) fn full_attn_pool_pages(&self) -> Option<usize> {
        self.full_attn_kv.as_ref().map(|p| p.max_total_pages)
    }

    /// Whether the recall eviction/scoring *cycle* runs this session
    /// (`--kv-recall`). When false the default paged path still runs — it just
    /// attends the full resident set (no working-set restriction, no tier I/O).
    fn recall_active(&self) -> bool {
        self.kv_recall && self.recall_tier.is_some() && self.full_attn_kv.is_some()
    }

    /// One DEFAULT paged prefill row (no recall cycle): self-allocate the prompt
    /// tokens into the shared `full_attn_kv` pool, build the FULL-resident page
    /// table (`PageMeta::for_slot`), and run the paged forward over it. This is
    /// the dense-Qwen3 model applied to Qwen3.6 — full attention over every
    /// resident page, no eviction/scoring/prefetch. Mirrors the prefill steps of
    /// [`Self::prefill_row_recall`] but stops after the forward.
    fn prefill_row_paged_default(
        &mut self,
        row: &infer_plan::PrefillRow,
        position: u64,
    ) -> Result<u32> {
        let slot = row.slot;
        if std::env::var_os("ARLE_KVDRIFT_DEBUG").is_some() {
            let pool_len = self
                .full_attn_kv
                .as_ref()
                .map_or(usize::MAX, |p| p.seq_len(slot));
            eprintln!(
                "[kvdrift] PREFILL slot={} start_pos={} tokens={} total={} slot.seq_len={} pool.seq_len={}",
                slot,
                row.start_pos,
                row.tokens.len(),
                row.total_tokens,
                self.slots[slot].seq_len(),
                pool_len,
            );
        }
        {
            let pool = self
                .full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (full_attn_paged)");
            // The pool must hold exactly `start_pos` tokens before appending the
            // tail. `free_slot` runs at `start_pos == 0` in `submit_prefill_row`;
            // prefix reuse (start_pos > 0) is gated off for hybrid models by
            // `reusable_prefix_blocks` returning 0, so this path always sees
            // start_pos == 0. A mismatch here would mean a soundness gate failed.
            ensure!(
                pool.seq_len(slot) == row.start_pos,
                "Qwen3.6 default-paged prefill: device pool seq_len {} != start_pos {} for slot {} \
                 (reusable_prefix_blocks soundness gate bypassed)",
                pool.seq_len(slot),
                row.start_pos,
                slot
            );
            pool.alloc_tokens(slot, row.tokens.len())?;
        }
        let meta = {
            let pool = self.full_attn_kv.as_ref().expect("full_attn_kv present");
            crate::loader::PageMeta::for_slot(
                &self.model.ctx,
                pool,
                slot,
                row.start_pos,
                row.tokens.len(),
            )?
        };
        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            dspark,
            ..
        } = self;
        let pool = full_attn_kv.as_mut().expect("full_attn_kv present");
        let mut rc = crate::qwen35::Qwen35RecallForward {
            pool,
            meta: &meta,
            layer0_query: Vec::new(),
        };
        let Some(ds) = dspark.as_mut() else {
            return model.forward_tokens_recall(
                &mut slots[slot],
                workspace,
                &row.tokens,
                row.start_pos,
                &row.params,
                position,
                &mut rc,
            );
        };
        // DSpark: capture the trunk taps, extend the draft ctx cache with this
        // chunk's rows, and stage the pending anchor on the final chunk.
        ds.taps.prepare(
            ds.head.target_layer_ids(),
            model.config.hidden_size,
            row.tokens.len(),
        );
        let token = model.forward_tokens_recall_tapped(
            &mut slots[slot],
            workspace,
            &row.tokens,
            row.start_pos,
            &row.params,
            position,
            &mut rc,
            Some(&mut ds.taps),
        )?;
        if ds.slots[slot].is_none() {
            ds.slots[slot] = Some(crate::qwen35::dspark::Qwen35DsparkSlotState::new(
                &model.ctx, &ds.head,
            )?);
        }
        let df = ds.slots[slot].as_mut().expect("built above");
        if df.ctx_end != row.start_pos {
            // Gap (prefix-cache / sidecar-restored prefix): rebase the draft
            // ctx at the suffix instead of degrading to plain decode — exact
            // for sliding draft layers once the tail ≥ window, approximate
            // only for the full-attention layer.
            df.rebase(row.start_pos);
        }
        model.dspark_append_ctx(
            &ds.head,
            df,
            &mut ds.taps,
            &mut ds.scratch,
            row.tokens.len(),
            row.start_pos,
        )?;
        let is_final = row.start_pos + row.tokens.len() == row.total_tokens;
        df.pending = (is_final && df.ctx_end == row.total_tokens).then_some(token);
        Ok(token)
    }

    /// One DEFAULT paged decode row (no recall cycle): append this step's token
    /// to the shared `full_attn_kv` pool and attend the FULL resident page set
    /// (`PageMeta::for_slot`). Mirrors [`Self::decode_row_recall`] but without
    /// the working-set restriction and without any tier I/O.
    fn decode_row_paged_default(&mut self, row: &DecodeRow, position: u64) -> Result<u32> {
        let slot = row.slot;
        {
            let pool = self
                .full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (full_attn_paged)");
            // The pool must already hold exactly this step's cache length before
            // appending the new token; a mismatch means an upstream prefill left
            // the slot's `seq_len` inconsistent with the engine's `kv_seq_len`
            // (e.g. a leaked radix-reuse prefill — see `prefill_row_paged_default`).
            // Catch it at the append, not via `PageMeta::for_slot` math downstream.
            ensure!(
                pool.seq_len(slot) == row.kv_seq_len,
                "Qwen3.6 default-paged decode: pool seq_len {} != kv_seq_len {} for slot {}",
                pool.seq_len(slot),
                row.kv_seq_len,
                slot
            );
            pool.alloc_tokens(slot, 1)?;
        }
        let meta = {
            let pool = self.full_attn_kv.as_ref().expect("full_attn_kv present");
            crate::loader::PageMeta::for_slot(&self.model.ctx, pool, slot, row.kv_seq_len, 1)?
        };
        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            ..
        } = self;
        let pool = full_attn_kv.as_mut().expect("full_attn_kv present");
        let mut rc = crate::qwen35::Qwen35RecallForward {
            pool,
            meta: &meta,
            layer0_query: Vec::new(),
        };
        model.forward_tokens_recall(
            &mut slots[slot],
            workspace,
            &[row.last_token],
            row.kv_seq_len,
            &row.params,
            position,
            &mut rc,
        )
    }

    /// One DSpark decode row: the block spec step when the slot is seeded
    /// (pending anchor + contiguous draft ctx), else a tapped warm step that
    /// re-seeds it. Emits 1..=block_size+1 tokens for this slot.
    fn dspark_decode_row(&mut self, row: &DecodeRow) -> Result<Vec<SlotToken>> {
        ensure!(
            row.slot < self.num_slots,
            "decode slot {} outside Qwen3.5 executor slots {}",
            row.slot,
            self.num_slots
        );
        ensure!(
            self.slots[row.slot].seq_len() == row.kv_seq_len,
            "Qwen3.5 materialized state len {} != DecodeRow.kv_seq_len {} for slot {}",
            self.slots[row.slot].seq_len(),
            row.kv_seq_len,
            row.slot
        );
        let position = row.kv_seq_len.saturating_add(1) as u64;
        let start = row.kv_seq_len;
        let ds = self
            .dspark
            .as_ref()
            .expect("dspark_decode_row without dspark");
        // The verify appends block_size+1 rows, so the whole chain must fit the
        // trunk cap; near the cap, degrade to single-token steps. Greedy rows
        // verify by argmax match; sampling rows by rejection sampling.
        let seeded = self.full_attn_paged()
            && start + ds.head.block_size() < self.model.max_seq_len()
            && matches!(
                ds.slots[row.slot].as_ref(),
                Some(s) if s.pending == Some(row.last_token) && s.ctx_end == start
            );
        if !seeded {
            let token = self.dspark_warm_decode_row(row, position)?;
            return Ok(vec![SlotToken {
                slot: row.slot,
                token,
                logprob: None,
                finish: None,
            }]);
        }
        self.dspark_spec_row(row)
    }

    /// One no-spec decode step that keeps the DSpark draft warm: paged decode
    /// with tap capture, extending the ctx cache + pending anchor when the
    /// cache is contiguous. Non-paged rows run plain and clear the draft state
    /// (speculation re-seeds at the next paged step).
    fn dspark_warm_decode_row(&mut self, row: &DecodeRow, position: u64) -> Result<u32> {
        let slot = row.slot;
        if !self.full_attn_paged() {
            if let Some(df) = self.dspark.as_mut().and_then(|ds| ds.slots[slot].as_mut()) {
                df.pending = None;
            }
            return self.submit_decode_row(row, false);
        }
        {
            let pool = self.full_attn_kv.as_mut().expect("paged (checked)");
            ensure!(
                pool.seq_len(slot) == row.kv_seq_len,
                "Qwen3.6 dspark warm decode: pool seq_len {} != kv_seq_len {} for slot {}",
                pool.seq_len(slot),
                row.kv_seq_len,
                slot
            );
            pool.alloc_tokens(slot, 1)?;
        }
        let meta = {
            let pool = self.full_attn_kv.as_ref().expect("paged (checked)");
            crate::loader::PageMeta::for_slot(&self.model.ctx, pool, slot, row.kv_seq_len, 1)?
        };
        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            dspark,
            ..
        } = self;
        let pool = full_attn_kv.as_mut().expect("paged (checked)");
        let mut rc = crate::qwen35::Qwen35RecallForward {
            pool,
            meta: &meta,
            layer0_query: Vec::new(),
        };
        let ds = dspark.as_mut().expect("dspark warm without dspark");
        // Gap (whole-slot promote / restored prefix): rebase at the current
        // position and start rebuilding the suffix-only ctx from this step.
        if let Some(df) = ds.slots[slot].as_mut()
            && df.ctx_end != row.kv_seq_len
        {
            df.rebase(row.kv_seq_len);
        }
        let taps = if ds.slots[slot].is_some() {
            ds.taps
                .prepare(ds.head.target_layer_ids(), model.config.hidden_size, 1);
            Some(&mut ds.taps)
        } else {
            None
        };
        let token = model.forward_tokens_recall_tapped(
            &mut slots[slot],
            workspace,
            &[row.last_token],
            row.kv_seq_len,
            &row.params,
            position,
            &mut rc,
            taps,
        )?;
        if let Some(df) = ds.slots[slot].as_mut() {
            model.dspark_append_ctx(
                &ds.head,
                df,
                &mut ds.taps,
                &mut ds.scratch,
                1,
                row.kv_seq_len,
            )?;
            df.pending = Some(token);
        }
        Ok(token)
    }

    /// One DSpark block spec step: draft a block from the draft head's ctx
    /// cache, verify the chain in ONE trunk forward over the paged pool
    /// (linear-capture + taps), accept the longest matching prefix, roll the
    /// trunk back on partial accept, crop the pool, and extend the draft ctx
    /// with the accepted rows' features. Greedy rows are token-exact to no-spec
    /// decode (every committed token is a trunk argmax); sampling rows commit
    /// rejection-sampled tokens distributed exactly as filtered target sampling.
    fn dspark_spec_row(&mut self, row: &DecodeRow) -> Result<Vec<SlotToken>> {
        let slot = row.slot;
        let start = row.kv_seq_len;
        if self.dspark.as_ref().expect("dspark").spec[slot].is_none() {
            let st = self.model.new_spec_slot_state()?;
            self.dspark.as_mut().expect("dspark").spec[slot] = Some(st);
        }
        let mut pt = crate::qwen35::dspark_phase_start(&self.model.ctx);
        // 1. Draft the block (draft-head-only; no trunk/pool state touched —
        //    the speculative noise K/V rows in the draft cache self-heal).
        let (chain, q) = {
            let Self { model, dspark, .. } = self;
            let ds = dspark.as_mut().expect("dspark");
            let df = ds.slots[slot].as_mut().expect("seeded slot");
            model.dspark_draft_block(
                &ds.head,
                df,
                &mut ds.scratch,
                row.last_token,
                start,
                &row.params,
            )?
        };
        let draft_ms = crate::qwen35::mtp_phase_lap(&self.model.ctx, &mut pt);
        if chain.len() < 2 {
            // Confidence head rejected the whole block — warm step instead.
            let position = start.saturating_add(1) as u64;
            let token = self.dspark_warm_decode_row(row, position)?;
            return Ok(vec![SlotToken {
                slot,
                token,
                logprob: None,
                finish: None,
            }]);
        }

        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            dspark,
            ..
        } = self;
        let ds = dspark.as_mut().expect("dspark");
        let spec = ds.spec[slot].as_mut().expect("built above");
        // 2. Snapshot the trunk linear state (partial-accept rollback base).
        spec.snapshot_trunk(&model.ctx, &slots[slot])?;
        let snap_ms = crate::qwen35::mtp_phase_lap(&model.ctx, &mut pt);
        // 3. Verify the chain over the paged pool, capturing linear inputs
        //    (replay substrate) + trunk taps (ctx features of accepted rows).
        let pool = full_attn_kv.as_mut().expect("paged (gated by seeded)");
        ensure!(
            pool.seq_len(slot) == start,
            "Qwen3.6 dspark verify: pool seq_len {} != start {start} for slot {slot}",
            pool.seq_len(slot)
        );
        pool.alloc_tokens(slot, chain.len())?;
        let meta = crate::loader::PageMeta::for_slot(&model.ctx, pool, slot, start, chain.len())?;
        let mut rc = crate::qwen35::Qwen35RecallForward {
            pool,
            meta: &meta,
            layer0_query: Vec::new(),
        };
        ds.taps.prepare(
            ds.head.target_layer_ids(),
            model.config.hidden_size,
            chain.len(),
        );
        let logits = model.dspark_verify_logits(
            &mut slots[slot],
            workspace,
            &chain,
            start,
            spec,
            &mut rc,
            &mut ds.taps,
        )?;
        let verify_ms = crate::qwen35::mtp_phase_lap(&model.ctx, &mut pt);
        // 4. Accept scan + trunk rollback (linear replay; full-attn KV
        //    self-heals under the pool crop below).
        let (emitted, bonus, k) = if row.params.is_greedy() {
            model.dspark_accept_commit(&mut slots[slot], spec, workspace, &chain, &logits, start)?
        } else {
            model.dspark_accept_commit_sampled(
                &mut slots[slot],
                spec,
                workspace,
                &chain,
                &q,
                &logits,
                start,
                &row.params,
            )?
        };
        drop(rc);
        if k + 1 < chain.len() {
            full_attn_kv
                .as_mut()
                .expect("paged (gated by seeded)")
                .truncate_slot(slot, start + k + 1)?;
        }
        let accept_ms = crate::qwen35::mtp_phase_lap(&model.ctx, &mut pt);
        // 5. Extend the draft ctx cache with the accepted rows' features and
        //    stage the bonus as the next block's anchor.
        let df = ds.slots[slot].as_mut().expect("seeded slot");
        model.dspark_append_ctx(&ds.head, df, &mut ds.taps, &mut ds.scratch, k + 1, start)?;
        let append_ms = crate::qwen35::mtp_phase_lap(&model.ctx, &mut pt);
        if pt.is_some() {
            eprintln!(
                "[dspark-phase] chain={} accept={k} base={} draft={draft_ms:.2} snap={snap_ms:.2} verify={verify_ms:.2} accept_commit={accept_ms:.2} append={append_ms:.2} ms",
                chain.len(),
                df.ctx_base
            );
        }
        df.pending = Some(bonus);
        ds.accepts += k;
        ds.rejects += chain.len() - 1 - k;
        Ok(emitted
            .into_iter()
            .map(|token| SlotToken {
                slot,
                token,
                logprob: None,
                finish: None,
            })
            .collect())
    }

    /// One prefill row over the recall pool (`--kv-recall`). The ONLY place the whole
    /// recall cycle runs (write-through model: "decode 不召回; prefetch 只在 prefill").
    ///
    /// 1. Self-allocate tokens, run paged forward — captures layer-0 query for scoring.
    /// 2. Score history (`recompute_recall_plan`): select evict + prefetch pages.
    /// 3. Batched-H2D prefetch tier-resident blocks, resolve fixed `recall_pages` working set.
    /// 4. Write-back-evict cold middle to tier, free physical pages immediately (no in-flight attn race).
    ///
    /// After return, `recall[slot].recall_pages()` is the immutable working set for decode.
    fn prefill_row_recall(&mut self, row: &infer_plan::PrefillRow, position: u64) -> Result<u32> {
        let slot = row.slot;
        let cfg = self.recall_cfg;
        {
            let pool = self
                .full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (recall_active)");
            pool.alloc_tokens(slot, row.tokens.len())?;
        }
        let meta = {
            let pool = self.full_attn_kv.as_ref().expect("full_attn_kv present");
            crate::loader::PageMeta::for_slot(
                &self.model.ctx,
                pool,
                slot,
                row.start_pos,
                row.tokens.len(),
            )?
        };
        // (1) Forward. Borrow split: &model + &mut slot + &mut workspace + &mut pool (via rc). `rc` carries back layer-0 query.
        let (token, layer0_query) = {
            let Self {
                model,
                slots,
                workspace,
                full_attn_kv,
                ..
            } = self;
            let pool = full_attn_kv.as_mut().expect("full_attn_kv present");
            let mut rc = crate::qwen35::Qwen35RecallForward {
                pool,
                meta: &meta,
                layer0_query: Vec::new(),
            };
            let token = model.forward_tokens_recall(
                &mut slots[slot],
                workspace,
                &row.tokens,
                row.start_pos,
                &row.params,
                position,
                &mut rc,
            )?;
            (token, rc.layer0_query)
        };

        let cache_len = row.start_pos + row.tokens.len();
        let ps = self.full_attn_kv.as_ref().expect("full_attn_kv").page_size;
        ensure!(
            cfg.n_init.is_multiple_of(ps)
                && cfg.n_local.is_multiple_of(ps)
                && cfg.l_bs.is_multiple_of(ps),
            "KV-recall config (n_init {}, n_local {}, l_bs {}) must be multiples of page_size {}",
            cfg.n_init,
            cfg.n_local,
            cfg.l_bs,
            ps
        );

        // Score history against prefill query, plan fixed working set. `allow_prefetch=true` lets tier-resident blocks re-enter.
        let num_q_heads = self.model.local_q_heads();
        let num_kv_heads = self.model.local_kv_heads();
        let head_dim = self.model.config.head_dim;
        let (evict_pages, prefetch_pages) = {
            let Self {
                recall,
                full_attn_kv,
                model,
                ..
            } = self;
            let pool = full_attn_kv.as_ref().expect("full_attn_kv");
            if let Some(state) = recall.get_mut(slot) {
                state.recompute_recall_plan(
                    &model.ctx,
                    pool,
                    slot,
                    cache_len,
                    &cfg,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    &layer0_query,
                    /* allow_prefetch = */ true,
                )?;
                (state.take_evict_pages(), state.take_prefetch_pages())
            } else {
                (Vec::new(), Vec::new())
            }
        };

        // Prefetch tier-resident blocks back into HBM. Decode never does this — one batched prefill sync point.
        for logical in prefetch_pages {
            let key = tier_block_u64(slot as u64, logical as u64);
            let Some(tier) = self.recall_tier.as_mut() else {
                break;
            };
            let payload = match tier.read(key) {
                Ok(p) => p.into_owned(),
                Err(_) => continue,
            };
            if let Some(pool) = self.full_attn_kv.as_mut() {
                if let Some(new_page) = pool.reinstate_slot_page(slot, logical) {
                    pool.copy_pages_from_host(&self.model.ctx, &[new_page], &payload)?;
                }
            }
        }

        if let Some(state) = self.recall.get_mut(slot) {
            if let Some(pool) = self.full_attn_kv.as_ref() {
                state.resolve_recall_pages(pool, slot);
            }
        }

        // Write-back-evict cold middle: mirror to L3, free physical pages. Prefill drained compute stream — no in-flight attn race.
        for logical in evict_pages {
            let physical = {
                let pool = self.full_attn_kv.as_ref().expect("full_attn_kv");
                pool.page_indices(slot)
                    .get(logical)
                    .copied()
                    .filter(|&p| p != cuda_kernels::prelude::EVICTED_PAGE)
            };
            let Some(physical) = physical else {
                continue;
            };
            let key = tier_block_u64(slot as u64, logical as u64);
            let mirrored = {
                let payload = {
                    let pool = self.full_attn_kv.as_ref().expect("full_attn_kv");
                    pool.copy_pages_to_host(&self.model.ctx, &[physical])?
                };
                match self.recall_tier.as_mut() {
                    Some(tier) if !tier.is_full() => tier.insert(key, payload),
                    _ => false,
                }
            };
            if !mirrored {
                continue; // tier full → keep page resident (no KV loss)
            }
            if let Some(pool) = self.full_attn_kv.as_mut() {
                pool.evict_slot_page(slot, logical);
            }
        }
        Ok(token)
    }

    /// One decode row over the recall pool (`--kv-recall`): append + attend, ZERO tier I/O.
    /// Working set fixed at prefill. Alloc token, read fixed `recall_pages`, forward + sample.
    fn decode_row_recall(&mut self, row: &DecodeRow, position: u64) -> Result<u32> {
        let slot = row.slot;
        {
            let pool = self
                .full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (recall_active)");
            pool.alloc_tokens(slot, 1)?;
        }
        let cache_len = row.kv_seq_len + 1;
        let recall_pages: Vec<u32> = match self.recall.get(slot).and_then(|s| s.recall_pages()) {
            Some(p) => p.to_vec(),
            None => {
                let pool = self.full_attn_kv.as_ref().expect("full_attn_kv");
                let num_pages = cache_len.div_ceil(pool.page_size);
                pool.page_indices(slot)[..num_pages].to_vec()
            }
        };
        let meta = {
            let pool = self.full_attn_kv.as_ref().expect("full_attn_kv");
            crate::loader::PageMeta::for_recall_decode(
                &self.model.ctx,
                pool,
                cache_len,
                &recall_pages,
            )?
        };
        // Forward (paged decode) + sample. `rc.layer0_query` collected but unused on decode.
        let Self {
            model,
            slots,
            workspace,
            full_attn_kv,
            ..
        } = self;
        let pool = full_attn_kv.as_mut().expect("full_attn_kv");
        let mut rc = crate::qwen35::Qwen35RecallForward {
            pool,
            meta: &meta,
            layer0_query: Vec::new(),
        };
        model.forward_tokens_recall(
            &mut slots[slot],
            workspace,
            &[row.last_token],
            row.kv_seq_len,
            &row.params,
            position,
            &mut rc,
        )
    }

    /// Run one decode step through the captured whole-step graph. Returns
    /// `Ok(None)` for eager fallback (gate off, out-of-budget, or capture/replay
    /// failure which permanently downgrades to eager, never fatal).
    ///
    /// See [`crate::qwen35::Qwen35Model::forward_decode_step_captured`] for
    /// capture-safety table and perf formula.
    fn try_graph_decode(&mut self, row: &DecodeRow, position: u64) -> Result<Option<u32>> {
        if !self.decode_graph_armed {
            return Ok(None);
        }
        // Budget check must run every step — host ensures inside captured closure
        // only run on warm/capture steps. Out-of-budget falls back to eager.
        if row.kv_seq_len + 1 > self.model.max_seq_len() {
            return Ok(None);
        }
        if self.decode_graph.is_none() {
            self.decode_graph = Some(Qwen35DecodeGraph::new(
                self.num_slots,
                &self.model.ctx.stream,
            ));
        }
        // Split borrows: capture closure needs &model + &mut slot + &mut workspace.
        let Self {
            model,
            slots,
            decode_graph,
            ..
        } = self;
        let dg = decode_graph
            .as_mut()
            .expect("decode_graph built above when armed");
        let Qwen35DecodeGraph { ws, graphs, baked } = dg;
        let slot_idx = row.slot;

        // Stage per-step device scalars outside graph (dense stage1_write pattern).
        let (token_ids_ptr, start_pos_ptr) =
            model.stage_step_inputs(ws, &[row.last_token], row.kv_seq_len)?;
        let logits_ptr = model.workspace_logits_ptr(ws)?;
        let bake = Qwen35GraphBake {
            token_ids_ptr,
            start_pos_ptr,
            logits_ptr,
            ws_epoch: ws.epoch(),
        };
        match baked[slot_idx] {
            Some(prev) if prev != bake => {
                // Workspace addresses drifted since capture (release → re-alloc).
                // Drop and re-capture against new addresses.
                info!(
                    "[qwen35-decode-graph] slot {slot_idx}: workspace addresses changed; \
                     dropping stale capture and recapturing"
                );
                graphs[slot_idx] = crate::graph::CudaGraphState::new(model.ctx.stream.clone());
                baked[slot_idx] = Some(bake);
            }
            None => baked[slot_idx] = Some(bake),
            _ => {}
        }

        let state = &mut graphs[slot_idx];
        let was_captured = state.is_captured();
        let will_replay = was_captured && !state.is_armed_warm();
        let slot_state = &mut slots[slot_idx];
        let run = state
            .run_or_capture(|| model.forward_decode_step_captured(slot_state, ws, row.kv_seq_len));
        if let Err(e) = run {
            // Any capture/replay failure downgrades to eager permanently — never fatal.
            // Mid-capture error only recorded kernels, so device state untouched.
            warn!(
                "Qwen3.5 whole-step decode graph failed (slot {slot_idx}), \
                 downgrading to eager forward: {e}"
            );
            self.decode_graph_armed = false;
            self.decode_graph = None;
            return Ok(None);
        }
        // Host-side state advance happens here — captured closure is host-state-free.
        slots[slot_idx].advance_seq_len(1);

        // Reuse-evidence probes (license needs replay counts, not capture-exists).
        let state = &self.decode_graph.as_ref().expect("still present").graphs[slot_idx];
        if !was_captured && state.is_captured() {
            let captures =
                QWEN35_GRAPH_CAPTURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let keys = self
                .decode_graph
                .as_ref()
                .expect("still present")
                .graphs
                .iter()
                .filter(|g| g.is_captured())
                .count();
            info!(
                "[qwen35-decode-graph] captured slot {slot_idx} \
                 (captures_total={captures}, live_keys={keys}, max_keys={})",
                self.num_slots
            );
        }
        if will_replay {
            let replays =
                QWEN35_GRAPH_REPLAYS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if replays.is_multiple_of(100) {
                info!(
                    "[qwen35-decode-graph] replay_total={replays} captures_total={}",
                    QWEN35_GRAPH_CAPTURES.load(std::sync::atomic::Ordering::Relaxed)
                );
            }
        }

        // Sampling stays OUTSIDE the graph (argmax + D2H + sync), reading the
        // logits the replay (or warm run) just wrote.
        let dg = self.decode_graph.as_mut().expect("still present");
        let token = self
            .model
            .sample_workspace_logits(&mut dg.ws, &row.params, position)?;
        Ok(Some(token))
    }

    /// Offload the model's device weights to host RAM (OPD teacher time-share),
    /// returning the device bytes freed. Per-slot KV / recurrent state is left
    /// resident — only the shared model weights move. The forward workspace is
    /// released too (AFTER the offload's full device sync, so no in-flight
    /// kernel can still reference the dropped buffers) — it may hold
    /// prefill-shaped scratch the student backward wants as headroom.
    fn offload_engine_weights(&mut self) -> Result<usize> {
        self.ensure_not_collective("offload_engine_weights")?;
        let freed = self.model.offload_engine_weights()?;
        self.workspace.release();
        // The batched-decode workspace holds `[*, B]` scratch (incl. MoE) —
        // release it for the same headroom reason. Its pointer TABLES survive:
        // they address per-slot state, which the weight offload leaves
        // resident (see `Qwen35BatchDecodeState::release`).
        if let Some(bd) = self.batch_decode.as_mut() {
            bd.release();
        }
        // Captured decode graphs bake the (now freed/placeholder) weight
        // addresses — drop them wholesale. Re-built + re-captured lazily after
        // reload (the gate flag stays armed).
        self.decode_graph = None;
        Ok(freed)
    }

    /// Release the inference forward scratch (24K-shaped workspace + batched-decode
    /// scratch + captured decode graphs) WITHOUT offloading weights or touching KV.
    /// The freed device blocks return to the shared CUmemAllocAsync caching pool the
    /// co-resident OPD writeback's autograd reuses (OPD `EngineOffloadMode::Off`
    /// rollout->writeback never offloads, so the scratch otherwise OOMs the writeback).
    /// This is `offload_engine_weights`'s scratch-teardown MINUS the weight offload —
    /// weights and per-slot KV / recurrent state stay resident.
    fn release_inference_scratch(&mut self) -> Result<()> {
        self.workspace.release();
        if let Some(bd) = self.batch_decode.as_mut() {
            bd.release();
        }
        self.decode_graph = None;
        Ok(())
    }

    /// Reload the model's device weights from the host snapshot.
    fn reload_engine_weights(&mut self) -> Result<()> {
        self.ensure_not_collective("reload_engine_weights")?;
        self.model.reload_engine_weights()
    }

    /// OPD surfaces are rank-0 control-seam calls: under multi-rank TP they
    /// would run on one rank only, desyncing the per-step NCCL collective
    /// sequence (and, for LoRA/offload, diverging the resident weights across
    /// ranks). Bail loudly instead of silently corrupting the lockstep.
    fn ensure_not_collective(&self, what: &str) -> Result<()> {
        ensure!(
            !self.model.tp.is_collective(),
            "{what} is single-GPU only: the Qwen3.5/3.6 OPD surfaces are not \
             wired for multi-rank tensor parallelism (world_size={})",
            self.model.tp.config().world_size
        );
        Ok(())
    }

    fn submit(&mut self, plan: &ForwardPlan) -> Result<StepOutput> {
        let rows = plan.decode_rows.len() + plan.prefill_rows.len();
        if rows == 0 {
            return Ok(StepOutput { tokens: Vec::new() });
        }

        // rows == 1 keeps the existing single-row path byte-identical
        // (including the whole-step B=1 decode-graph lane, which is gated to
        // rows==1 PLANS — batched/mixed steps never capture or replay).
        if rows == 1 {
            let (slot, token) = if let Some(row) = plan.prefill_rows.first() {
                (row.slot, self.submit_prefill_row(row)?)
            } else {
                let row = &plan.decode_rows[0];
                if self.dspark.is_some() {
                    return Ok(StepOutput {
                        tokens: self.dspark_decode_row(row)?,
                    });
                }
                (
                    row.slot,
                    self.submit_decode_row(row, /* allow_graph = */ true)?,
                )
            };
            return Ok(StepOutput {
                tokens: vec![SlotToken {
                    slot,
                    token,
                    logprob: None,
                    finish: None,
                }],
            });
        }

        // Multi-row plans (DSv4 executor pattern): per-prefill single-row
        // sub-steps, then ONE decode sub-batch. Plan rows always address
        // disjoint slots (a request is either Prefilling or Decoding), so the
        // sequential sub-steps are math-identical to consecutive single-mode
        // ticks.
        let mut seen_slots = std::collections::BTreeSet::new();
        for slot in plan
            .prefill_rows
            .iter()
            .map(|row| row.slot)
            .chain(plan.decode_rows.iter().map(|row| row.slot))
        {
            ensure!(
                seen_slots.insert(slot),
                "Qwen3.5 plan schedules slot {slot} more than once per tick"
            );
        }

        let mut tokens = Vec::with_capacity(rows);
        for row in &plan.prefill_rows {
            let token = self.submit_prefill_row(row)?;
            tokens.push(SlotToken {
                slot: row.slot,
                token,
                logprob: None,
                finish: None,
            });
        }
        if self.dspark.is_some() {
            // DSpark decode is per-row (each row runs its own draft + block
            // verify); batched spec verify is a later increment.
            for row in &plan.decode_rows {
                tokens.extend(self.dspark_decode_row(row)?);
            }
            return Ok(StepOutput { tokens });
        }
        match plan.decode_rows.len() {
            0 => {}
            1 => {
                // Mixed tick with a single decode row: eager (the B=1 graph
                // lane stays gated to rows==1 plans in stage 1).
                let row = &plan.decode_rows[0];
                let token = self.submit_decode_row(row, /* allow_graph = */ false)?;
                tokens.push(SlotToken {
                    slot: row.slot,
                    token,
                    logprob: None,
                    finish: None,
                });
            }
            _ => tokens.extend(self.submit_decode_batch(&plan.decode_rows)?),
        }
        Ok(StepOutput { tokens })
    }

    /// One prefill row as its own single-row sub-step (the pre-batching
    /// single-row prefill arm, factored).
    fn submit_prefill_row(&mut self, row: &infer_plan::PrefillRow) -> Result<u32> {
        ensure!(
            row.slot < self.num_slots,
            "prefill slot {} outside Qwen3.5 executor slots {}",
            row.slot,
            self.num_slots
        );
        ensure!(!row.tokens.is_empty(), "prefill row must carry tokens");
        // A fresh prefill (start_pos == 0) rewinds this slot's recurrent +
        // conv state and cache cursor before appending. Request-boundary
        // rearm discipline (the e95e11b6 lesson): the slot's captured
        // decode graph stays valid (state buffers are memset, not
        // re-allocated), but its FIRST decode of the new occupant runs one
        // eager warm step so any per-request host work executes without
        // dropping the capture — capture cost stays once per slot, not
        // once per request.
        if row.start_pos == 0 {
            // Request boundary: the prior occupant is finished (the scheduler
            // only reassigns a slot after its request completes — no in-flight
            // forward references the slot), so return its recurrent block to the
            // free-list, then acquire (pop the same block back, or alloc) and
            // zero it for the new occupant. This replaces the old in-place
            // `reset()` zeroing; the block MUST be resident before the forward.
            self.slots[row.slot].release_recurrent(&mut self.recurrent_pool);
            // Drop any prior occupant's pending boundary snapshot: this slot is a
            // new request, so a stale snapshot must never key a future sidecar.
            self.prefill_boundary_snapshot[row.slot] = None;
            // New occupant: the prior request's draft ctx cache is dead.
            if let Some(df) = self
                .dspark
                .as_mut()
                .and_then(|ds| ds.slots[row.slot].as_mut())
            {
                df.reset();
            }
            let (num_linear, gdr_len, conv_len) = self.model.recurrent_dims();
            self.slots[row.slot].acquire_recurrent(
                &self.model.ctx,
                num_linear,
                gdr_len,
                conv_len,
                &mut self.recurrent_pool,
            )?;
            // The new block's `gdr_states`/`conv_states` addresses differ from the
            // prior occupant's, so the batched-decode pointer-table cache (keyed
            // on slot_indices only) must restage on the next decode batch.
            if let Some(bd) = self.batch_decode.as_mut() {
                bd.invalidate_staged_pointers();
            }
            // The captured decode graph (legacy contiguous lane) bakes this slot's
            // recurrent-block addresses; a different block invalidates it. Drop the
            // slot's capture (the `baked` staleness check only tracks workspace
            // ptrs, not recurrent ptrs) so it re-captures against the new block —
            // `rearm_warm` alone keeps the stale capture and would replay freed mem.
            if let Some(dg) = self.decode_graph.as_mut() {
                dg.graphs[row.slot] =
                    crate::graph::CudaGraphState::new(self.model.ctx.stream.clone());
                dg.baked[row.slot] = None;
            }
            // Default paged path: free the prior occupant's pages back to the
            // shared pool so a fresh prefill starts at logical page 0. (Under
            // recall the keepalive release below also runs.)
            if self.recall_active() {
                self.recall[row.slot].reset();
                // Release any pages still parked in the eviction keepalive from the
                // prior occupant's last decode step BEFORE freeing the slot, so the
                // detached physical pages rejoin the pool instead of leaking (they
                // are sentinels in the table, so `free_slot` alone would not recycle
                // them). The prior request finished, so its attention has long since
                // completed — no race with the one-step deferral.
                let parked = std::mem::take(&mut self.recall_keepalive[row.slot]);
                if let Some(pool) = self.full_attn_kv.as_mut() {
                    for (_logical, physical) in parked {
                        pool.release_evicted_page(physical);
                    }
                    pool.free_slot(row.slot);
                }
                // Stale L3 tier entries keyed by (slot, logical) from the prior
                // occupant are left to the store's LRU/capacity eviction: re-recall
                // only reads a block whose resident REP exists, and `reset()` just
                // cleared all reps, so a fresh occupant can never read a prior
                // occupant's tier block before overwriting that key — tenant-safe
                // without a per-session key registry (the dense arm's
                // `drop_tier_session` makes the same call, see its doc).
            } else if let Some(pool) = self.full_attn_kv.as_mut() {
                // Default paged (no recall): just recycle the slot's pages.
                pool.free_slot(row.slot);
            }
        }
        let position = (row.start_pos + row.tokens.len()) as u64;
        if self.recall_active() {
            return self.prefill_row_recall(row, position);
        }
        // Default paged prefill: full attention over all resident pages, no
        // eviction (the dense model applied to Qwen3.6 — Phase 2).
        //
        // Recurrent-sidecar boundary capture: on the FINAL prefill chunk, split
        // the forward at the last-page boundary `L* = align_down16(total - 1)`
        // and snapshot the recurrent state there, so the sidecar keyed at `L*`
        // matches a resend's restore target (popped for aligned, un-popped for
        // non-aligned) with snapshot-position == key-position. Without the split
        // the state is only observable at `P` (prompt end), and keying it at
        // `L* < P` bakes the residue `[L*..P]` — double-advanced on restore. For
        // a non-aligned prompt the planner already ends a chunk at `L*` (residue
        // split), so `head_len == 0` and we only snapshot; for an aligned prompt
        // the final chunk spans `L*`, so we forward `[start..L*]`, snapshot, then
        // forward the last page.
        let is_final = row.start_pos + row.tokens.len() == row.total_tokens;
        let lstar = row.total_tokens.saturating_sub(1) / SUPPORTED_PAGE_SIZE * SUPPORTED_PAGE_SIZE;
        if is_final && lstar > 0 && lstar >= row.start_pos && self.slots[row.slot].has_recurrent() {
            let head_len = lstar - row.start_pos;
            if head_len > 0 {
                let head = infer_plan::PrefillRow {
                    slot: row.slot,
                    tokens: row.tokens[..head_len].to_vec(),
                    start_pos: row.start_pos,
                    total_tokens: row.total_tokens,
                    params: row.params.clone(),
                };
                self.prefill_row_paged_default(&head, lstar as u64)?; // token discarded
            }
            // State is now materialized at exactly `L*`; snapshot before the tail.
            let snap = self.slots[row.slot].snapshot_recurrent(&self.model.ctx)?;
            self.prefill_boundary_snapshot[row.slot] = Some((lstar, snap));
            let tail = infer_plan::PrefillRow {
                slot: row.slot,
                tokens: row.tokens[head_len..].to_vec(),
                start_pos: lstar,
                total_tokens: row.total_tokens,
                params: row.params.clone(),
            };
            return self.prefill_row_paged_default(&tail, position);
        }
        self.prefill_row_paged_default(row, position)
    }

    /// One decode row as a single-row forward (the pre-batching single-row
    /// decode arm, factored). `allow_graph` admits the whole-step B=1
    /// decode-graph lane — true only for rows==1 plans.
    fn submit_decode_row(&mut self, row: &DecodeRow, allow_graph: bool) -> Result<u32> {
        ensure!(
            row.slot < self.num_slots,
            "decode slot {} outside Qwen3.5 executor slots {}",
            row.slot,
            self.num_slots
        );
        if std::env::var_os("ARLE_KVDRIFT_DEBUG").is_some()
            && self.slots[row.slot].seq_len() != row.kv_seq_len
        {
            let pool_len = self
                .full_attn_kv
                .as_ref()
                .map_or(usize::MAX, |p| p.seq_len(row.slot));
            eprintln!(
                "[kvdrift] DECODE-ASSERT slot={} slot.seq_len={} row.kv_seq_len={} pool.seq_len={} (Δslot-row={})",
                row.slot,
                self.slots[row.slot].seq_len(),
                row.kv_seq_len,
                pool_len,
                self.slots[row.slot].seq_len() as i64 - row.kv_seq_len as i64,
            );
        }
        ensure!(
            self.slots[row.slot].seq_len() == row.kv_seq_len,
            "Qwen3.5 materialized state len {} != DecodeRow.kv_seq_len {} for slot {}",
            self.slots[row.slot].seq_len(),
            row.kv_seq_len,
            row.slot
        );
        let position = row.kv_seq_len.saturating_add(1) as u64;
        // Session KV-recall (opt-in): decode reads the recall-restricted page
        // table (the fixed working set chosen at prefill). The seq_len invariant
        // above still holds — the slot's seq_len is advanced in lockstep inside
        // the recall forward.
        if self.recall_active() {
            return self.decode_row_recall(row, position);
        }
        // Default paged decode: append + attend the full resident page set. The
        // whole-step decode-graph lane bakes per-step device addresses (the page
        // table grows each step), so it is bypassed under the paged default —
        // the paged forward IS the correctness floor; the graph lane (kept below
        // for the legacy contiguous path) is a contiguous-cache optimization a
        // later phase re-enables over a stable paged table.
        if self.full_attn_paged() {
            return self.decode_row_paged_default(row, position);
        }
        // Legacy contiguous path (no paged pool — e.g. OPD weight offload dropped
        // it): the whole-step graph lane first (opt-in), with eager
        // `forward_tokens` as the correctness floor / graph-miss fallback.
        let graph_token = if allow_graph {
            self.try_graph_decode(row, position)?
        } else {
            None
        };
        match graph_token {
            Some(token) => Ok(token),
            None => self.model.forward_tokens(
                &mut self.slots[row.slot],
                &mut self.workspace,
                &[row.last_token],
                row.kv_seq_len,
                &row.params,
                position,
            ),
        }
    }

    /// A rows>1 pure-decode sub-batch: ONE batched forward over all rows
    /// ([`crate::qwen35::Qwen35Model::forward_decode_batch`] — stage 1
    /// re-port of the monolith batched decode; the c=4 amortization formula
    /// and the TP/all-reduce proof live on that method). With
    /// `--qwen35-batched-decode false`, runs the rows sequentially as
    /// single-row forwards instead (the honest A/B arm; the pre-batching
    /// behavior was a hard error, not a fallback).
    ///
    /// Decode-graph interaction (stage 1): batched steps NEVER capture or
    /// replay, and they cannot invalidate existing B=1 captures — the
    /// captured graphs bake (a) per-slot state addresses, which the batched
    /// path mutates strictly IN PLACE through pointer tables over the same
    /// allocations, and (b) the dedicated decode-graph workspace's buffer
    /// addresses, which the batched path never touches (it owns a third,
    /// batch-only workspace). The per-step position is read from a staged
    /// device scalar at replay, so a slot's seq_len advancing via a batched
    /// step replays correctly on its next single-row graphed step.
    fn submit_decode_batch(&mut self, rows: &[DecodeRow]) -> Result<Vec<SlotToken>> {
        debug_assert!(rows.len() > 1);
        // Per-row watermark/validation BEFORE any device mutation (dup-slot
        // ensure ran at plan level in `submit`).
        for row in rows {
            ensure!(
                row.slot < self.num_slots,
                "decode slot {} outside Qwen3.5 executor slots {}",
                row.slot,
                self.num_slots
            );
            ensure!(
                self.slots[row.slot].seq_len() == row.kv_seq_len,
                "Qwen3.5 materialized state len {} != DecodeRow.kv_seq_len {} for slot {}",
                self.slots[row.slot].seq_len(),
                row.kv_seq_len,
                row.slot
            );
        }

        // Under the shared-paged default (always-on), B>1 decode batches over the
        // shared `full_attn_kv` pool via the PAGED batched kernels (the contiguous
        // `k_caches`/`v_caches` the legacy batched path reads are never allocated).
        // BF16 single-GPU only: the batched-paged HD256 kernels are BF16, and a TP
        // collective per-rank lockstep is handled by `forward_decode_batch_paged`'s
        // all-reduces — but the correctness floor keeps quant-KV and `--kv-recall`
        // (per-row restricted page table) on the serial per-row path.
        if self.full_attn_paged() {
            let paged_bf16 = self
                .full_attn_kv
                .as_ref()
                .map(|p| p.format == KVFormat::BF16)
                .unwrap_or(false);
            if qwen35_batched_decode_enabled()
                && paged_bf16
                && !self.recall_active()
                && self.model.tp.is_single()
            {
                return self.submit_decode_batch_paged(rows);
            }
            // Correctness floor: recall / quant-KV / TP → serial per-row paged.
            let mut tokens = Vec::with_capacity(rows.len());
            for row in rows {
                let token = self.submit_decode_row(row, /* allow_graph = */ false)?;
                tokens.push(SlotToken {
                    slot: row.slot,
                    token,
                    logprob: None,
                    finish: None,
                });
            }
            return Ok(tokens);
        }

        // Legacy contiguous build (no paged pool — e.g. OPD weight offload): the
        // contiguous batched lane (env-gated) or its serial A/B arm.
        if !qwen35_batched_decode_enabled() || self.recall_active() {
            let mut tokens = Vec::with_capacity(rows.len());
            for row in rows {
                let token = self.submit_decode_row(row, /* allow_graph = */ false)?;
                tokens.push(SlotToken {
                    slot: row.slot,
                    token,
                    logprob: None,
                    finish: None,
                });
            }
            return Ok(tokens);
        }

        if self.batch_decode.is_none() {
            let num_full = self.model.config.num_full_attention_layers();
            let num_linear = self.model.config.num_hidden_layers - num_full;
            self.batch_decode = Some(crate::qwen35::Qwen35BatchDecodeState::new(
                &self.model.ctx,
                num_full,
                num_linear,
                self.num_slots,
            )?);
        }
        let bd = self
            .batch_decode
            .as_mut()
            .expect("batch_decode built above");

        let slot_indices: Vec<usize> = rows.iter().map(|r| r.slot).collect();
        let tokens_in: Vec<u32> = rows.iter().map(|r| r.last_token).collect();
        let kv_seq_lens: Vec<usize> = rows.iter().map(|r| r.kv_seq_len).collect();
        let params: Vec<SamplingParams> = rows.iter().map(|r| r.params.clone()).collect();
        let positions: Vec<u64> = rows
            .iter()
            .map(|r| r.kv_seq_len.saturating_add(1) as u64)
            .collect();
        let sampled = self.model.forward_decode_batch(
            &mut self.slots,
            bd,
            &slot_indices,
            &tokens_in,
            &kv_seq_lens,
            &params,
            &positions,
        )?;
        ensure!(
            sampled.len() == rows.len(),
            "Qwen3.5 batched decode returned {} tokens for {} rows",
            sampled.len(),
            rows.len()
        );
        Ok(slot_indices
            .into_iter()
            .zip(sampled)
            .map(|(slot, token)| SlotToken {
                slot,
                token,
                logprob: None,
                finish: None,
            })
            .collect())
    }

    /// A rows>1 pure-decode sub-batch over the SHARED-PAGED default lane: append
    /// each row's new token to its slot in `full_attn_kv`, build ONE B-row page
    /// table ([`PageMeta::for_decode_batch`]), and run a single batched-paged
    /// forward ([`Qwen35Model::forward_decode_batch_paged`]). This is the paged
    /// analogue of the contiguous [`Self::submit_decode_batch`] true-batch arm:
    /// the per-row append + page-table build mirrors the single-row
    /// [`Self::decode_row_paged_default`], so a B-row batch is byte-equivalent to
    /// B sequential single-row paged decodes (each row attends only its own
    /// slot's pages via its `kv_indptr` slice). BF16 single-GPU, no recall (gated
    /// by the caller in `submit_decode_batch`).
    fn submit_decode_batch_paged(&mut self, rows: &[DecodeRow]) -> Result<Vec<SlotToken>> {
        debug_assert!(rows.len() > 1);
        // Append this step's token to every row's slot BEFORE building the page
        // table (the pool must hold the POST-append length the meta encodes). The
        // pool seq_len must equal the engine's kv_seq_len pre-append, exactly as
        // `decode_row_paged_default` checks — a mismatch means an upstream prefill
        // left the slot inconsistent (e.g. leaked radix reuse).
        {
            let pool = self
                .full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (full_attn_paged)");
            for row in rows {
                ensure!(
                    pool.seq_len(row.slot) == row.kv_seq_len,
                    "Qwen3.6 paged batched decode: pool seq_len {} != kv_seq_len {} for slot {}",
                    pool.seq_len(row.slot),
                    row.kv_seq_len,
                    row.slot
                );
                pool.alloc_tokens(row.slot, 1)?;
            }
        }

        let slot_indices: Vec<usize> = rows.iter().map(|r| r.slot).collect();
        let tokens_in: Vec<u32> = rows.iter().map(|r| r.last_token).collect();
        let kv_seq_lens: Vec<usize> = rows.iter().map(|r| r.kv_seq_len).collect();
        let params: Vec<SamplingParams> = rows.iter().map(|r| r.params.clone()).collect();
        let sample_positions: Vec<u64> = rows
            .iter()
            .map(|r| r.kv_seq_len.saturating_add(1) as u64)
            .collect();
        // Page table over the POST-append lengths (one row per slot).
        let batch_rows: Vec<(usize, usize)> =
            rows.iter().map(|r| (r.slot, r.kv_seq_len + 1)).collect();

        if self.batch_decode.is_none() {
            let num_full = self.model.config.num_full_attention_layers();
            let num_linear = self.model.config.num_hidden_layers - num_full;
            self.batch_decode = Some(crate::qwen35::Qwen35BatchDecodeState::new(
                &self.model.ctx,
                num_full,
                num_linear,
                self.num_slots,
            )?);
        }

        let Self {
            model,
            slots,
            batch_decode,
            full_attn_kv,
            ..
        } = self;
        let bd = batch_decode.as_mut().expect("batch_decode built above");
        let pool = full_attn_kv.as_mut().expect("full_attn_kv present");
        let meta = PageMeta::for_decode_batch(&model.ctx, pool, &batch_rows)?;
        let sampled = model.forward_decode_batch_paged(
            slots,
            bd,
            pool,
            &meta,
            &slot_indices,
            &tokens_in,
            &kv_seq_lens,
            &params,
            &sample_positions,
        )?;
        ensure!(
            sampled.len() == rows.len(),
            "Qwen3.6 paged batched decode returned {} tokens for {} rows",
            sampled.len(),
            rows.len()
        );
        Ok(slot_indices
            .into_iter()
            .zip(sampled)
            .map(|(slot, token)| SlotToken {
                slot,
                token,
                logprob: None,
                finish: None,
            })
            .collect())
    }

    /// OPD teacher raw-logits forward: run the full hybrid forward over
    /// `(input_ids, positions)` on a FRESH transient slot state and return the
    /// FULL `[seq_len, vocab]` logits (every row) plus the model's device
    /// context, WITHOUT sampling.
    ///
    /// This does not touch the serving slots: the teacher scores a sequence
    /// in one shot (positions are contiguous from `positions[0]`), so a private
    /// slot state is allocated, advanced once, and dropped. `positions` must be
    /// the contiguous absolute positions of `input_ids` (the OPD teacher always
    /// scores a full prompt starting at `positions[0]`).
    pub(crate) fn forward_token_logits(
        &mut self,
        input_ids: &[u32],
        positions: &[u32],
    ) -> Result<(DeviceVec, [usize; 2])> {
        self.ensure_not_collective("forward_token_logits")?;
        ensure!(
            !input_ids.is_empty(),
            "forward_token_logits requires a non-empty token sequence"
        );
        ensure!(
            input_ids.len() == positions.len(),
            "forward_token_logits token/position length mismatch: tokens={} positions={}",
            input_ids.len(),
            positions.len()
        );
        let start_pos = positions[0] as usize;
        for (i, &p) in positions.iter().enumerate() {
            ensure!(
                p as usize == start_pos + i,
                "forward_token_logits requires contiguous positions; positions[{i}]={p} != {}",
                start_pos + i
            );
        }
        // Private transient slot: the teacher scores the whole sequence in one
        // forward, so it never shares the serving slots' KV/recurrent state.
        // The shared workspace IS reused (serial forwards; its buffers reshape
        // to the teacher's seq_len and back on the next serving call).
        // Recurrent state is acquired into a throwaway local pool (the block is
        // dropped with the slot, never pooled — this path is rare and transient).
        let mut slot = self.model.new_slot_state();
        let (num_linear, gdr_len, conv_len) = self.model.recurrent_dims();
        let mut scratch_pool = Vec::new();
        slot.acquire_recurrent(
            &self.model.ctx,
            num_linear,
            gdr_len,
            conv_len,
            &mut scratch_pool,
        )?;
        self.model
            .forward_token_logits_full(&mut slot, &mut self.workspace, input_ids, start_pos)
    }

    pub(crate) fn device(&self) -> &DeviceContext {
        &self.model.ctx
    }

    /// Fold a fresh student LoRA update into the resident projection weights
    /// (OPD per-step re-merge). Delegates to [`crate::qwen35::Qwen35Model`].
    pub(crate) fn remerge_student_lora(
        &mut self,
        update: crate::qwen35::StudentLoraUpdate,
    ) -> Result<()> {
        self.ensure_not_collective("remerge_student_lora")?;
        // The merge REPLACES `DeviceMatrix` buffers (new device addresses);
        // captured decode graphs bake the old ones — drop and recapture lazily.
        self.decode_graph = None;
        // Weight epoch changed: sidecar recurrent snapshots are as stale as the
        // radix the caller invalidates — drop every tracked sidecar blob from the
        // tier so a skipped capture never serves old-epoch state. (Blobs whose
        // tail page already LRU-evicted are unreachable — the radix is invalidated
        // too — and are reclaimed by the store's budget.)
        for (_, key) in self.sidecar_page_key.drain() {
            self.slot_tier
                .remove_chunked(NS_SIDECAR, NS_SIDECAR_CHUNK, key);
        }
        self.model.remerge_student_lora(update)
    }

    /// Read-only borrow of resident FP8 block-scaled base projection pointers
    /// (train-infer weight sharing, `--share-frozen-base`). Delegates to
    /// [`crate::qwen35::Qwen35Model`]. Read-only; does not mutate resident
    /// weights, so no decode-graph invalidation is needed.
    pub(crate) fn frozen_base_fp8_pointers(
        &self,
    ) -> Result<Vec<crate::qwen35::SharedFp8BaseProjection>> {
        self.ensure_not_collective("frozen_base_fp8_pointers")?;
        self.model.frozen_base_fp8_pointers()
    }
}

pub(crate) fn sample_cuda_token(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    params: &SamplingParams,
    position: u64,
) -> Result<u32> {
    maybe_dump_sample_topk(ctx, logits, position)?;
    if params.is_greedy() {
        let token = argmax(ctx, logits)?;
        probe_decode_entropy(ctx, logits, None, token, position)?;
        return Ok(token);
    }

    // TODO: repetition/frequency/presence penalties need the per-request
    // generated-token history threaded through the executor.
    let logits_host = logits.to_host(ctx)?;
    let token = infer_plan::sample_token(&logits_host, params, position);
    probe_decode_entropy(ctx, logits, Some(&logits_host), token, position)?;
    Ok(token)
}

/// [`sample_cuda_token`] with a caller-provided persistent argmax scratch (one
/// device i32): the Qwen3.5/3.6 steady-state decode sampler — greedy decode
/// performs ZERO device allocations per token (the last per-token
/// `alloc_zeros(1)` moved into the workspace `argmax_out` slot). Always runs
/// OUTSIDE any CUDA-graph capture (argmax syncs + reads D2H).
pub(crate) fn sample_cuda_token_scratched(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    params: &SamplingParams,
    position: u64,
    argmax_out: &mut cudarc::driver::CudaSlice<i32>,
) -> Result<u32> {
    maybe_dump_sample_topk(ctx, logits, position)?;
    if params.is_greedy() {
        let token = crate::ops::argmax_into(ctx, logits, argmax_out)?;
        probe_decode_entropy(ctx, logits, None, token, position)?;
        return Ok(token);
    }
    let logits_host = logits.to_host(ctx)?;
    let token = infer_plan::sample_token(&logits_host, params, position);
    probe_decode_entropy(ctx, logits, Some(&logits_host), token, position)?;
    Ok(token)
}

/// [`sample_cuda_token`] for the vocab-sharded lm_head
/// (`ARLE_DSV4_LM_HEAD_SHARD=1`, #99): `logits` holds this rank's contiguous
/// vocab slice, zero-padded to the uniform `shard.rows_per_rank`.
///
/// Greedy: local argmax over the real rows (pads excluded), then ONE 8-byte
/// host all-gather of `(max_value_f32, global_index_u32)` per rank; the merge
/// is exact vs the replicated device argmax (both resolve ties to the lowest
/// index — `sampling.cu` `warp_reduce_argmax` and
/// [`infer_plan::merge_vocab_shard_argmax`]). Non-greedy (or entropy-probe-on):
/// all-gather the bf16 slices to full vocab — rank slices are contiguous so
/// the gather is vocab-ordered with the pad tail strictly past `shard.vocab` —
/// then reuse the existing sampler on the first `vocab` entries. Every rank
/// merges the same gathered bytes, so all ranks emit the same token.
/// `INFER_DSV4_DUMP_TOPK*` is not supported on this path (the local slice
/// cannot rank full-vocab top-k); documented in `docs/environment.md`.
pub(crate) fn sample_cuda_token_vocab_sharded(
    ctx: &DeviceContext,
    tp: &crate::tp::TpRuntime,
    logits: &DeviceVec,
    shard: &crate::dsv4::Dsv4LmHeadShard,
    params: &SamplingParams,
    position: u64,
) -> Result<u32> {
    #[cfg(not(feature = "nccl"))]
    {
        // The DSv4 load gate requires a collective runtime for the shard knob,
        // so this arm is unreachable in practice; keep it loud, not silent.
        let _ = (ctx, tp, logits, shard, params, position);
        anyhow::bail!("ARLE_DSV4_LM_HEAD_SHARD sampling requires the nccl feature")
    }
    #[cfg(feature = "nccl")]
    {
        let world = tp.config().world_size;
        anyhow::ensure!(
            logits.len == shard.rows_per_rank && shard.local_rows <= shard.rows_per_rank,
            "sharded logits len {} != rows_per_rank {} (local_rows {})",
            logits.len,
            shard.rows_per_rank,
            shard.local_rows
        );
        if params.is_greedy() && !crate::probe::token_entropy() {
            let mut scratch = ctx
                .stream
                .alloc_zeros::<i32>(1)
                .map_err(|e| anyhow::anyhow!("sharded argmax scratch alloc failed: {e}"))?;
            let local = crate::ops::argmax_row_into(ctx, logits, 0, shard.local_rows, &mut scratch)?
                as usize;
            let value = ctx
                .stream
                .clone_dtoh(&logits.data.slice(local..local + 1))
                .map_err(|e| anyhow::anyhow!("sharded argmax value D2H failed: {e}"))?;
            ctx.sync()?;
            let global = (tp.config().rank * shard.rows_per_rank + local) as u32;
            let mut payload = [0u8; 8];
            payload[..4].copy_from_slice(&value[0].to_f32().to_le_bytes());
            payload[4..].copy_from_slice(&global.to_le_bytes());
            let gathered = tp.all_gather_bytes(ctx, &payload, 8)?;
            anyhow::ensure!(
                gathered.len() == 8 * world,
                "sharded argmax exchange returned {} bytes, expected {}",
                gathered.len(),
                8 * world
            );
            let token = infer_plan::merge_vocab_shard_argmax(gathered.chunks_exact(8).map(|c| {
                (
                    f32::from_le_bytes(c[..4].try_into().expect("4-byte value")),
                    u32::from_le_bytes(c[4..].try_into().expect("4-byte index")),
                )
            }))
            .ok_or_else(|| anyhow::anyhow!("sharded argmax merge over zero ranks"))?;
            anyhow::ensure!(
                (token as usize) < shard.vocab,
                "sharded argmax token {token} out of vocab {}",
                shard.vocab
            );
            return Ok(token);
        }

        // Full-vocab path: all-gather the padded slices (rank order = vocab
        // order, pad tail past `vocab`), then the standard sampler/probe.
        let mut gathered = DeviceVec::zeros(ctx, world * shard.rows_per_rank)?;
        {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (send, _sg) = logits.data.device_ptr(&ctx.stream);
            let (recv, _rg) = gathered.data.device_ptr_mut(&ctx.stream);
            // SAFETY: `send` is this rank's `rows_per_rank` contiguous bf16
            // logits; `recv` holds `world * rows_per_rank` bf16 on the same
            // device; every rank calls with the same sendcount.
            unsafe {
                tp.all_gather_bf16_raw(
                    ctx,
                    send as *const std::ffi::c_void,
                    shard.rows_per_rank,
                    recv as *mut std::ffi::c_void,
                )?;
            }
        }
        if params.is_greedy() {
            // Exact replicated semantics: the same device argmax kernel over
            // the first `vocab` gathered entries.
            let mut scratch = ctx.stream.alloc_zeros::<i32>(1).map_err(|e| {
                anyhow::anyhow!("sharded gathered argmax scratch alloc failed: {e}")
            })?;
            let token = crate::ops::argmax_row_into(ctx, &gathered, 0, shard.vocab, &mut scratch)?;
            if crate::probe::token_entropy() {
                let host = gathered.to_host(ctx)?;
                probe_decode_entropy(ctx, &gathered, Some(&host[..shard.vocab]), token, position)?;
            }
            return Ok(token);
        }
        let host_full = gathered.to_host(ctx)?;
        let host = &host_full[..shard.vocab];
        let token = infer_plan::sample_token(host, params, position);
        probe_decode_entropy(ctx, &gathered, Some(host), token, position)?;
        Ok(token)
    }
}

/// Per-token entropy probe over the raw (pre-penalty, T=1) logits at the
/// single-row sampling convergence point (all backends' eager + graph decode
/// plus the prefill last token; spec-decode/MTP verify paths are NOT
/// instrumented). Off = one `OnceLock` load. `host` reuses an already
/// materialized copy; `None` costs one D2H (probe-on only).
fn probe_decode_entropy(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    host: Option<&[f32]>,
    token: u32,
    position: u64,
) -> Result<()> {
    if !crate::probe::token_entropy() {
        return Ok(());
    }
    let owned;
    let host = match host {
        Some(host) => host,
        None => {
            owned = logits.to_host(ctx)?;
            &owned
        }
    };
    let (entropy, nll) = crate::probe::entropy_nll(host, Some(token));
    crate::probe::emit_token("decode", position, Some(token), nll, entropy);
    Ok(())
}

/// Positions (rank 0 only) at which to dump top-k logits, parsed once from
/// `INFER_DSV4_DUMP_TOPK_POSITIONS`. Empty = disabled, the production default —
/// so the per-token hot-path cost is one `OnceLock` load + a slice scan, never a
/// per-token env-lock across all TP ranks. A diagnostic for FP8-vs-bf16 parity
/// (e.g. the DIFF@122 margin investigation), not a serving path.
fn dump_topk_positions() -> &'static [u64] {
    static POSITIONS: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    POSITIONS.get_or_init(|| {
        if std::env::var("INFER_TP_RANK").ok().as_deref() != Some("0") {
            return Vec::new();
        }
        let Ok(raw) = std::env::var("INFER_DSV4_DUMP_TOPK_POSITIONS") else {
            return Vec::new();
        };
        raw.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse::<u64>().ok())
            .collect()
    })
}

fn maybe_dump_sample_topk(ctx: &DeviceContext, logits: &DeviceVec, position: u64) -> Result<()> {
    if !dump_topk_positions().contains(&position) {
        return Ok(());
    }

    let top_k = std::env::var("INFER_DSV4_DUMP_TOPK")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(8);
    let variant =
        std::env::var("INFER_DSV4_AB_CURRENT_VARIANT").unwrap_or_else(|_| "unknown".to_string());
    let logits_host = logits.to_host(ctx)?;
    let mut best: Vec<(u32, f32)> = Vec::with_capacity(top_k);
    let mut nan_count = 0usize;
    let mut pos_inf_count = 0usize;
    let mut neg_inf_count = 0usize;
    for (idx, &value) in logits_host.iter().enumerate() {
        if value.is_nan() {
            nan_count += 1;
            continue;
        }
        if value == f32::INFINITY {
            pos_inf_count += 1;
            continue;
        }
        if value == f32::NEG_INFINITY {
            neg_inf_count += 1;
            continue;
        }
        if best.len() < top_k {
            best.push((idx as u32, value));
            best.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            continue;
        }
        let Some(last) = best.last() else {
            continue;
        };
        if value > last.1 || (value == last.1 && (idx as u32) < last.0) {
            best.pop();
            best.push((idx as u32, value));
            best.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        }
    }
    let margin = match best.as_slice() {
        [first, second, ..] => first.1 - second.1,
        _ => 0.0,
    };
    println!(
        "sample_topk variant={variant} position={position} finite={} nan={} pos_inf={} neg_inf={} top={best:?} margin={margin:.6}",
        logits_host
            .len()
            .saturating_sub(nan_count + pos_inf_count + neg_inf_count),
        nan_count,
        pos_inf_count,
        neg_inf_count
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CudaKvCacheDtype, NS_SLOT, NS_SLOT_CHUNK};
    use crate::kv_tier::{chunk_sub, tier_key};
    use infer_seam::KvCacheDtype;

    #[test]
    fn tier_key_namespaces_never_collide() {
        // Same feature key across both namespaces → distinct store keys.
        let key = 7u64;
        let keys = [
            tier_key(NS_SLOT, key),
            tier_key(NS_SLOT_CHUNK, chunk_sub(key, 0)),
        ];
        for (i, a) in keys.iter().enumerate() {
            for b in keys.iter().skip(i + 1) {
                assert_ne!(a, b);
            }
        }
        // Chunk indices stay disjoint per key and across adjacent keys.
        assert_ne!(chunk_sub(7, 0), chunk_sub(7, 1));
        assert_ne!(chunk_sub(7, 65_535), chunk_sub(8, 0));
    }

    #[test]
    fn resolve_cuda_support_matrix() {
        // Auto + explicit Bf16 resolve to the default CUDA dtype.
        assert_eq!(
            CudaKvCacheDtype::resolve(KvCacheDtype::Auto).unwrap(),
            CudaKvCacheDtype::Bf16
        );
        assert_eq!(
            CudaKvCacheDtype::resolve(KvCacheDtype::Bf16).unwrap(),
            CudaKvCacheDtype::Bf16
        );
        // The paged quant-KV modes wired by #68 T3 resolve to their variants.
        assert_eq!(
            CudaKvCacheDtype::resolve(KvCacheDtype::Int8).unwrap(),
            CudaKvCacheDtype::Int8
        );
        assert_eq!(
            CudaKvCacheDtype::resolve(KvCacheDtype::Fp8).unwrap(),
            CudaKvCacheDtype::Fp8
        );
        // Tq4 stays an explicit deferral (no paged-prefill kernel path for
        // TurboQuant's page_size=1 pools), never a silent downgrade.
        let err = CudaKvCacheDtype::resolve(KvCacheDtype::Tq4)
            .expect_err("tq4 must bail with the explicit-deferral message");
        let msg = err.to_string();
        assert!(
            msg.contains("TurboQuant"),
            "error explains the TurboQuant deferral: {msg}"
        );
        assert!(
            msg.contains("page_size"),
            "error cites the page_size mismatch: {msg}"
        );
    }
}
