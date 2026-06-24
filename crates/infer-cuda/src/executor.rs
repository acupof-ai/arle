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
// Default DSv4 per-slot max KV sequence length when `INFER_DSV4_MAX_SEQ_LEN` is
// unset. 32768 admits realistic agentic prompts out of the box. NOT 262144: the
// O(N^2) DSA-indexer logits scratch OOMs above ~200K (16384 is a known-good
// floor). The executor clamps `num_slots` to HBM (`kv_budget_num_slots`), so a
// larger default degrades to fewer slots, never OOM.
const DSV4_DEFAULT_MAX_SEQ_LEN: usize = 32768;

/// Seq-len budget the captured decode graph's fixed `kv_indices` is sized to;
/// pages beyond it fall back to eager rather than replay a stale graph.
const DECODE_GRAPH_MAX_SEQ_LEN: usize = 32_768;

/// Decode-graph default when `INFER_CUDA_DECODE_GRAPH` is unset. Set once at load
/// from the `enable_cuda_graph` load flag (CLI `--cuda-graph`/`--no-cuda-graph`,
/// default on) via [`set_decode_graph_default`]; the env var, when present, always
/// overrides it. Single-GPU Qwen dense only — `warmup` still hard-disables the
/// graph under TP (NCCL not graph-capturable) and MoE (host routing per step).
static DECODE_GRAPH_DEFAULT_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Set the load-time decode-graph default (honored only when the env override is
/// unset). Wired from `LoadedInferenceEngine::load(.., enable_cuda_graph)` so the
/// CLI `--cuda-graph`/`--no-cuda-graph` flag actually controls the graph instead
/// of being discarded — single set at load, read once at `warmup`.
pub fn set_decode_graph_default(enabled: bool) {
    DECODE_GRAPH_DEFAULT_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Captured B=1 decode graph enabled? `INFER_CUDA_DECODE_GRAPH` is an explicit
/// override (`1`/`true`/`on` → on, `0`/`false`/`off` → off); when unset, falls
/// back to the load-time default ([`set_decode_graph_default`], CLI-driven,
/// default on). The eager path stays the correctness floor; `warmup` still gates
/// TP and MoE off regardless of this.
fn decode_graph_enabled() -> bool {
    match std::env::var("INFER_CUDA_DECODE_GRAPH").as_deref() {
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON") => true,
        Ok("0" | "false" | "FALSE" | "no" | "off" | "OFF") => false,
        _ => DECODE_GRAPH_DEFAULT_ENABLED.load(std::sync::atomic::Ordering::Relaxed),
    }
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

    pub(crate) fn from_qwen35_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
        mem_fraction_static: f64,
    ) -> Result<Self> {
        Ok(Self::Qwen35(Box::new(
            Qwen35CudaExecutor::from_qwen35_safetensors(
                model_path,
                num_slots,
                total_pages,
                mem_fraction_static,
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
            // ARLE_QWEN35_DECODE_GRAPH=1) — warmup logs the gate verdict;
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
            Self::Qwen35(_) | Self::Dsv4(_) => 0,
        }
    }

    pub(crate) fn kv_tier_page_bytes(&self) -> usize {
        match self {
            Self::Qwen(q) => q.kv_tier_page_bytes(),
            Self::Qwen35(_) | Self::Dsv4(_) => 0,
        }
    }

    pub(crate) fn kv_tier_host_demoted_pages(&self) -> usize {
        match self {
            Self::Qwen(q) => q.kv_tier_host_demoted_pages(),
            Self::Qwen35(_) | Self::Dsv4(_) => 0,
        }
    }

    pub(crate) fn kv_tier_disk_pages(&self) -> usize {
        match self {
            Self::Qwen(q) => q.kv_tier_disk_pages(),
            Self::Qwen35(_) | Self::Dsv4(_) => 0,
        }
    }

    pub(crate) fn kv_tier_location(&self, key: u64) -> Option<infer_seam::KvTierLocation> {
        match self {
            Self::Qwen(q) => q.kv_tier_location(key),
            Self::Qwen35(_) | Self::Dsv4(_) => None,
        }
    }

    pub(crate) fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
        match self {
            Self::Qwen(q) => q.reusable_prefix_blocks(blocks),
            Self::Qwen35(q) => q.reusable_prefix_blocks(blocks),
            // DSv4 reuses prefixes via the position-0 whole-slot store, not page-radix
            // (its MLA KV is per-slot, not yet paged — Phase 5).
            Self::Dsv4(_) => 0,
        }
    }

    pub(crate) fn demote_prefix_pages(&mut self, entries: &[(u32, u64)]) -> Result<usize> {
        match self {
            Self::Qwen(q) => q.demote_prefix_pages(entries),
            Self::Qwen35(_) | Self::Dsv4(_) => Ok(0),
        }
    }

    pub(crate) fn promote_prefix_pages(&mut self, entries: &[(u64, u32)]) -> Result<()> {
        match self {
            Self::Qwen(q) => q.promote_prefix_pages(entries),
            Self::Qwen35(_) | Self::Dsv4(_) => {
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

    /// Whole-slot KV tier hooks. CUDA currently implements these only for the
    /// DSv4 executor arm; other arms report no slot tier.
    pub(crate) fn kv_slot_tier_enabled(&self) -> bool {
        match self {
            Self::Dsv4(d) => d.kv_slot_tier_enabled(),
            Self::Qwen(_) | Self::Qwen35(_) => false,
        }
    }

    pub(crate) fn demote_slot(&mut self, slot: usize, key: u64) -> Result<bool> {
        match self {
            Self::Dsv4(d) => d.demote_slot(slot, key),
            Self::Qwen(_) | Self::Qwen35(_) => Ok(false),
        }
    }

    pub(crate) fn promote_slot(&mut self, key: u64, slot: usize) -> Result<()> {
        match self {
            Self::Dsv4(d) => d.promote_slot(key, slot),
            Self::Qwen(_) | Self::Qwen35(_) => {
                anyhow::bail!("whole-slot KV tier store is implemented only for DSv4 CUDA")
            }
        }
    }

    pub(crate) fn drop_kv_slot_entries(&mut self, keys: &[u64]) {
        if let Self::Dsv4(d) = self {
            d.drop_kv_slot_entries(keys);
        }
    }

    /// Cross-request position-0 prefix reuse. Only the DSv4 arm holds a store;
    /// page-radix-reusing arms (dense Qwen) report no match here.
    pub(crate) fn cached_prefix_match_len(&self, tokens: &[u32]) -> usize {
        match self {
            Self::Dsv4(d) => d.cached_prefix_match_len(tokens),
            Self::Qwen(_) | Self::Qwen35(_) => 0,
        }
    }

    pub(crate) fn capture_cached_prefix(&mut self, slot: usize, tokens: &[u32]) -> Result<()> {
        match self {
            Self::Dsv4(d) => d.capture_cached_prefix(slot, tokens),
            Self::Qwen(_) | Self::Qwen35(_) => Ok(()),
        }
    }

    pub(crate) fn restore_cached_prefix(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
    ) -> Result<()> {
        match self {
            Self::Dsv4(d) => d.restore_cached_prefix(slot, tokens, matched_len),
            Self::Qwen(_) | Self::Qwen35(_) => {
                anyhow::bail!("position-0 prefix store is implemented only for DSv4 CUDA")
            }
        }
    }

    /// Re-budget the host-demoted tier store (`0` disables; pre-serve only). No-op on
    /// arms without a tier store.
    pub(crate) fn set_kv_tier_budget_bytes(&mut self, bytes: usize) {
        if let Self::Qwen(q) = self {
            q.set_kv_tier_budget_bytes(bytes);
        }
    }

    /// Apply the `--dram-fraction` knob (default 0.5): re-budget the L2 host-DRAM
    /// KV tier from measured DRAM at this fraction. Pre-serve only. The dense arm
    /// re-budgets its eager prefix tier immediately; the Qwen3.6 arm stores it for
    /// the lazily-built recall tier; DSv4 owns per-slot KV and has no tier (no-op).
    /// Run BEFORE any `--kv-t1-budget-bytes` override (that explicit cap wins).
    pub(crate) fn set_dram_fraction(&mut self, fraction: f64) {
        match self {
            Self::Qwen(q) => q.set_dram_fraction(fraction),
            Self::Qwen35(q) => q.set_dram_fraction(fraction),
            Self::Dsv4(_) => {}
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
    /// dropping an explicit `--kv-ssd-path` request.
    pub(crate) fn set_kv_tier_disk(
        &mut self,
        root: std::path::PathBuf,
        budget_bytes: usize,
    ) -> bool {
        match self {
            Self::Qwen(q) => q.set_kv_tier_disk(root, budget_bytes),
            Self::Qwen35(_) | Self::Dsv4(_) => false,
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
    /// Qwen3 + Qwen3.6). Both profile their pool from measured free VRAM at
    /// construction (`profile_kv_pool_tokens`), so this is the page count the
    /// host admission pool MUST mirror 1:1 — not the requested `total_pages`.
    /// `None` for DSv4 (slot MLA arena, no shared page pool); its admission stays
    /// per-slot this phase.
    pub(crate) fn effective_total_pages(&self) -> Option<usize> {
        match self {
            Self::Qwen(q) => Some(q.kv.max_total_pages),
            Self::Qwen35(q) => q.full_attn_pool_pages(),
            Self::Dsv4(_) => None,
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

use crate::kv_tier::{CudaKvTierStore, default_t1_budget_bytes};

/// Default fraction of available host DRAM the L2 KV tier may claim when
/// `--dram-fraction` is unset — the shared-box-safe 0.5 (the store is pageable
/// host memory; see `infer_seam::DramTierPolicy`). The constructor sizes the
/// tier at this default; `--dram-fraction` re-budgets it pre-serve.
pub(crate) const DEFAULT_DRAM_FRACTION: f64 = 0.5;

pub(crate) struct QwenCudaExecutor {
    model: CudaModel,
    kv: PagedKVPool,
    tier: CudaKvTierStore,
    /// Fraction of available host DRAM the L2 prefix/recall tier may claim
    /// (`--dram-fraction`, default 0.5). Set post-construction via
    /// [`Self::set_dram_fraction`]; the construction-time tier uses the default.
    dram_fraction: f64,
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
    /// `mem_fraction_static` (default 0.9): the dense shared paged pool is sized
    /// from MEASURED free VRAM after weights load (`infer_seam::profile_kv_pool_tokens`,
    /// SGLang-style), NOT the requested `total_pages`. `total_pages` becomes a
    /// minimum-capacity floor: the profiled pool is the larger of the two so an
    /// explicit `--total-pages` never shrinks below the request, but a large card
    /// gets the extra capacity for more concurrency.
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
            dram_fraction: DEFAULT_DRAM_FRACTION,
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

    /// Apply the `--dram-fraction` knob: store it and re-budget the (eager) L2
    /// prefix tier from measured DRAM at that fraction. Pre-serve only (drops any
    /// existing entries, like `set_kv_tier_budget_bytes`). The explicit
    /// `--kv-t1-budget-bytes` override, if set, runs AFTER this and wins.
    pub(crate) fn set_dram_fraction(&mut self, fraction: f64) {
        self.dram_fraction = fraction;
        self.tier = CudaKvTierStore::with_budget(
            default_t1_budget_bytes(fraction),
            self.kv.storage_bytes_per_page(),
        );
    }

    /// Opt into session KV-recall (`--kv-recall`, default off). Mirrors the Metal
    /// `set_kv_recall`: a post-construction setter so the constructor signature
    /// stays stable. With recall off the decode hot path is unchanged
    /// (byte-identical baseline — CUDA is the Stable backend).
    pub(crate) fn set_kv_recall(&mut self, enabled: bool) {
        self.kv_recall = enabled;
    }

    /// Attach the opt-in disk spill level (`--kv-ssd-path`). Pre-serve only.
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
        for &(key, page) in entries {
            // Disjoint-field borrows: payload borrows self.tier; the copy
            // borrows self.kv + self.model.
            let payload = self
                .tier
                .read(key)
                .map_err(|err| anyhow::anyhow!("KV tier promote: {err}"))?;
            self.kv
                .copy_pages_from_host(&self.model.ctx, &[page], &payload)?;
        }
        self.model.ctx.sync()?;
        Ok(())
    }

    pub(crate) fn drop_kv_tier_entries(&mut self, keys: &[u64]) {
        self.tier.remove(keys);
    }

    /// **Write-through** (`infer_seam::KvTier::write_through`): mirror a filled
    /// device `page` into the host tier under `key`, so a later evict-drop of that
    /// page is free (the tier keeps the source of truth). Reuses the same
    /// `CudaKvTierStore` as the prefix tier — there is ONE session-keyed store
    /// (R5), not a parallel one. `key` is the `(session, block)` `TierBlockKey`
    /// flattened to the store's `u64` namespace by [`tier_block_u64`].
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

    /// **Prefetch** (`infer_seam::KvTier::prefetch`): load blocks from the host
    /// tier back into freshly allocated device pages (`(key, page)`), complete on
    /// return. Identical transport to `promote_prefix_pages`; the difference is the
    /// entry point (relevance-prefetch at prefill vs prefix-hit promote), per R5.
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
        ensure!(
            rows == 1,
            "R6 clean CUDA forward is single-row only, got {} prefill + {} decode rows",
            plan.prefill_rows.len(),
            plan.decode_rows.len()
        );
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

    /// Warmup the B=1 decode graph before serving.
    ///
    /// Captures the smallest shape (`num_pages = 1`) so the machinery is proven
    /// before the first request; later page counts capture lazily on first decode
    /// (capture-once, replay-after, per key). Opt-in via
    /// `INFER_CUDA_DECODE_GRAPH=1`; any capture failure downgrades to eager-only
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
            info!("CUDA decode graph disabled (set INFER_CUDA_DECODE_GRAPH=1 to enable)");
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
                continue; // tier full → keep the page resident (no KV loss)
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

/// Write-through tiered KV memory contract for the dense-Qwen3 paged pool.
///
/// The CUDA paged pool is the natural write-through grain (page == mirror/evict/
/// prefetch unit). These verbs delegate to the SAME `CudaKvTierStore` the prefix
/// tier uses (R5 — one session-keyed store, not a parallel one), flattening the
/// device-neutral `(session, block)` `TierBlockKey` into the store's `u64`
/// namespace via [`tier_block_u64`] so write-through keys never alias prefix-tier
/// keys.
///
/// Status (per `docs/plans/2026-06-23-writethrough-tiered-kv-memory.md`):
/// - `write_through` / `prefetch` — **real** (D2H/H2D over the existing store).
/// - `evict_drop` — the **real page free now happens** in the decode path
///   ([`QwenCudaExecutor::evict_drop_recall_pages`]), which holds the slot + the
///   host `&mut dyn KvPool` and frees the physical page from BOTH the device pool
///   ([`PagedKVPool::evict_slot_page`]) and the host single-allocator
///   ([`infer_seam::KvAllocator::evict_slot_page`]). The deferred blocker — that
///   `mirror_slot` re-publishes a contiguous page table every step, so freeing a
///   live slot's middle page would break the `SlotProgress` contiguity guard — is
///   resolved by decoupling physical residency from the logical page table: an
///   evicted page leaves an `EVICTED_PAGE` sentinel in its logical slot, so the
///   logical page count (and `seq_len`) is unchanged and both contracts hold. This
///   seam `evict_drop(page)` verb (no slot/host context) is the device-side
///   mirror-drop hook; with no per-page device sidecar to release it is a no-op,
///   and the host allocator performs the page free via the decode-path call above.
impl infer_seam::KvTier for QwenCudaExecutor {
    fn tier_capacity_pages(&self) -> usize {
        self.tier.capacity_pages()
    }

    fn tier_page_bytes(&self) -> usize {
        self.tier.page_bytes()
    }

    fn tier_location(&self, key: infer_seam::TierBlockKey) -> Option<infer_seam::KvTierLocation> {
        self.tier.location(tier_block_u64(key.session, key.block))
    }

    fn write_through(&mut self, key: infer_seam::TierBlockKey, page: u32) -> Result<bool> {
        QwenCudaExecutor::write_through(self, tier_block_u64(key.session, key.block), page)
    }

    fn evict_drop(&mut self, _page: u32) {
        // No backend-side device mirror to release; the host CudaKvPool owns the
        // page free. Mid-decode device-page free is the documented blocker (see
        // the impl-level doc) — left a no-op so the contract is satisfiable
        // without a half-broken page-lifecycle path. The page's tier copy (from
        // `write_through`) remains the source of truth.
    }

    fn prefetch(&mut self, entries: &[(infer_seam::TierBlockKey, u32)]) -> Result<()> {
        let u64_entries: Vec<(u64, u32)> = entries
            .iter()
            .map(|&(key, page)| (tier_block_u64(key.session, key.block), page))
            .collect();
        self.prefetch_pages(&u64_entries)
    }

    fn drop_tier_session(&mut self, session: u64) {
        // The store has no session index (keys are opaque u64), so a precise
        // per-session sweep would need a key registry. Until session_id is
        // threaded into the engine (the prefetch-policy plumbing), sessions are
        // reclaimed lazily via the tier's own LRU/capacity eviction; this hook is
        // the seam-level entry point for the future precise sweep.
        let _ = session;
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
    /// See `mtp_should_speculate`. Opt-in via `ARLE_DSV4_MTP_ADAPTIVE` (bring-up;
    /// promote to a `--mtp-adaptive` CLI flag once pod-calibrated).
    mtp_accept_ema: f32,
    mtp_skip_streak: usize,
    /// Host images of demoted DSv4 slots, keyed by the engine-minted swap key.
    /// The count cap bounds host RAM; beyond it, the engine falls back to
    /// recompute instead of accumulating swap images.
    slot_swap_store: std::collections::BTreeMap<u64, Dsv4SlotSwapEntry>,
    /// Cross-request position-0 prefix store. Maps a full captured prompt to its
    /// whole-slot KV image; a new request whose leading tokens exactly equal a
    /// stored prompt restores that prefix and re-prefills only the tail. LRU over
    /// a host-byte budget (see `Dsv4PrefixCache`). Default-on (pod-verified:
    /// correct + 11.7x prefill speedup); size knob `ARLE_DSV4_PREFIX_CACHE_BYTES`.
    prefix_cache: Dsv4PrefixCache,
}

/// Concrete DSv4 prefix store: the LRU keyed by full prompt, payload = the
/// whole-slot KV image captured at absolute positions `[0, tokens.len())`.
type Dsv4PrefixCache = PrefixImageStore<crate::dsv4::Dsv4SlotImage>;

/// One position-0-anchored cached prefix: the full prompt tokens that produced
/// the image plus the payload `image`. The store keeps `host_bytes` separately
/// so the LRU/budget logic never inspects the payload (host-only testable).
struct PrefixStoreEntry<P> {
    tokens: Vec<u32>,
    image: P,
    host_bytes: usize,
}

/// LRU host-byte-bounded store of position-0 prefix images, generic over the
/// payload `P` so the key/LRU/budget logic is unit-testable without a device.
///
/// Keyed by a token hash for candidate lookup, with an exact token-vector
/// compare to reject hash collisions. `order` is the LRU recency list (front =
/// coldest). Match returns the LONGEST stored prompt that is an exact leading
/// prefix of the query — the longest skip-able prefill.
struct PrefixImageStore<P> {
    /// `hash(prompt) -> entries that hashed there` (collision chains rare).
    by_hash: std::collections::HashMap<u64, Vec<PrefixStoreEntry<P>>>,
    /// LRU recency: prompt hashes, coldest at front, hottest at back.
    order: std::collections::VecDeque<u64>,
    budget_bytes: usize,
    used_bytes: usize,
}

impl<P> PrefixImageStore<P> {
    /// Position-0 prefix store with an explicit host-byte budget. The store is
    /// ALWAYS active — cross-request prefix KV reuse is default-on (verified
    /// correct + 11.7x prefill speedup on pod; the on/off env tag was removed).
    fn new(budget_bytes: usize) -> Self {
        Self {
            by_hash: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
            budget_bytes,
            used_bytes: 0,
        }
    }

    /// Default 4 GiB budget, overridable via `ARLE_DSV4_PREFIX_CACHE_BYTES`
    /// (a sizing knob for pod sweeps — NOT an on/off switch).
    fn from_env() -> Self {
        let budget_bytes = std::env::var("ARLE_DSV4_PREFIX_CACHE_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4 * 1024 * 1024 * 1024);
        Self::new(budget_bytes)
    }

    fn hash_tokens(tokens: &[u32]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        tokens.hash(&mut hasher);
        hasher.finish()
    }

    /// Longest stored prompt that is an exact leading prefix of `tokens`. The
    /// match must be a STRICT prefix or full equality: stored `prompt` with
    /// `prompt.len() <= tokens.len()` and `tokens[..prompt.len()] == prompt`.
    /// Returns that length, or 0 on no match.
    fn match_len(&self, tokens: &[u32]) -> usize {
        if tokens.is_empty() {
            return 0;
        }
        let mut best = 0usize;
        // Exact prompts hash on their full content; a query that is a longer
        // prompt sharing a stored prefix hashes differently, so we probe every
        // stored prompt prefix length present. Since prompts are full keys, we
        // scan candidate hashes for each prefix boundary the query could match.
        // Bounded by the number of distinct stored prompt lengths, which is the
        // store size — acceptable for the bring-up store; pod sweeps cap size by
        // the byte budget.
        for entries in self.by_hash.values() {
            for entry in entries {
                let len = entry.tokens.len();
                if len > best && len <= tokens.len() && tokens[..len] == entry.tokens[..] {
                    best = len;
                }
            }
        }
        best
    }

    fn touch(&mut self, hash: u64) {
        if let Some(pos) = self.order.iter().position(|&h| h == hash) {
            self.order.remove(pos);
        }
        self.order.push_back(hash);
    }

    /// Remove and return the entry for the exact stored prompt `tokens[..len]`,
    /// so the caller can borrow `&entry.image` without aliasing the rest of the
    /// executor. The caller MUST re-insert it via [`Self::reinsert`] after the
    /// restore to keep it hot. `None` when no exact entry exists.
    fn take(&mut self, tokens: &[u32], len: usize) -> Option<PrefixStoreEntry<P>> {
        if len == 0 || len > tokens.len() {
            return None;
        }
        let key = &tokens[..len];
        let hash = Self::hash_tokens(key);
        let entries = self.by_hash.get_mut(&hash)?;
        let pos = entries.iter().position(|e| e.tokens == key)?;
        let entry = entries.swap_remove(pos);
        if entries.is_empty() {
            self.by_hash.remove(&hash);
            if let Some(order_pos) = self.order.iter().position(|&h| h == hash) {
                self.order.remove(order_pos);
            }
        }
        self.used_bytes = self.used_bytes.saturating_sub(entry.host_bytes);
        Some(entry)
    }

    /// Re-insert an entry pulled out by [`Self::take`], marking it hottest.
    fn reinsert(&mut self, entry: PrefixStoreEntry<P>) {
        let hash = Self::hash_tokens(&entry.tokens);
        self.used_bytes = self.used_bytes.saturating_add(entry.host_bytes);
        self.by_hash.entry(hash).or_default().push(entry);
        self.touch(hash);
        self.evict_to_budget();
    }

    /// Insert (or replace) a position-0 prefix image keyed by its full prompt.
    /// `host_bytes` is the payload's host RAM (the caller computes it from the
    /// concrete image), keeping the LRU logic payload-agnostic. Evicts coldest
    /// entries until the new image fits the byte budget; a prompt larger than the
    /// whole budget is dropped.
    fn insert(&mut self, tokens: Vec<u32>, image: P, host_bytes: usize) {
        if host_bytes == 0 || host_bytes > self.budget_bytes {
            return;
        }
        let hash = Self::hash_tokens(&tokens);
        // Replace an existing identical-prompt entry in place.
        if let Some(entries) = self.by_hash.get_mut(&hash) {
            if let Some(existing) = entries.iter_mut().find(|e| e.tokens == tokens) {
                self.used_bytes = self.used_bytes.saturating_sub(existing.host_bytes);
                existing.image = image;
                existing.host_bytes = host_bytes;
                self.used_bytes = self.used_bytes.saturating_add(host_bytes);
                self.touch(hash);
                self.evict_to_budget();
                return;
            }
        }
        // Make room before inserting the new entry.
        while self.used_bytes.saturating_add(host_bytes) > self.budget_bytes {
            if !self.evict_one_coldest() {
                break;
            }
        }
        if self.used_bytes.saturating_add(host_bytes) > self.budget_bytes {
            return;
        }
        self.by_hash
            .entry(hash)
            .or_default()
            .push(PrefixStoreEntry {
                tokens,
                image,
                host_bytes,
            });
        self.used_bytes = self.used_bytes.saturating_add(host_bytes);
        self.touch(hash);
    }

    fn evict_to_budget(&mut self) {
        while self.used_bytes > self.budget_bytes {
            if !self.evict_one_coldest() {
                break;
            }
        }
    }

    /// Drop the coldest hash bucket entirely. Returns false when nothing remains.
    fn evict_one_coldest(&mut self) -> bool {
        let Some(hash) = self.order.pop_front() else {
            return false;
        };
        if let Some(entries) = self.by_hash.remove(&hash) {
            for entry in entries {
                self.used_bytes = self.used_bytes.saturating_sub(entry.host_bytes);
            }
            true
        } else {
            // Stale order entry (hash already removed); keep draining.
            self.evict_one_coldest()
        }
    }
}

/// One demoted slot: the device-state image plus the executor-level MTP spec
/// chain. `spec_pending`/`spec_hidden` MUST ride along (not reset): under
/// `--spec-type mtp` the resumed decode hard-requires the pending token and
/// the previous MTP stream (`forward_decode_tokens` errors on a missing
/// pending), and the slot's spec state is overwritten by whichever request
/// occupies the slot while this one is demoted.
struct Dsv4SlotSwapEntry {
    image: crate::dsv4::Dsv4SlotImage,
    spec_pending: Option<u32>,
    spec_hidden: Option<Vec<half::bf16>>,
}

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
        // Dynamic KV mem budget: clamp num_slots to what GPU free mem affords (was: fixed
        // num_slots → c=32 OOM crash at long max_seq_len). Deterministic ⇒ TP-consistent.
        let num_slots = model.kv_budget_num_slots(num_slots, max_seq_len)?;
        let kv_adapter = model.new_kv_adapter(max_seq_len, num_slots)?;
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
            slot_swap_store: std::collections::BTreeMap::new(),
            prefix_cache: Dsv4PrefixCache::from_env(),
        })
    }

    /// Whole-slot swap is single-rank today.
    pub(crate) fn kv_slot_tier_enabled(&self) -> bool {
        let world_size = self.model.tp.config().world_size;
        if world_size > 1 {
            // Multi-rank demote/promote must execute on EVERY rank in lockstep
            // (the seam hooks fire on the coordinator only), or the
            // deterministic planner diverges and NCCL deadlocks.
            static MULTI_RANK_LOGGED: std::sync::Once = std::sync::Once::new();
            MULTI_RANK_LOGGED.call_once(|| {
                info!(
                    "DSv4 whole-slot KV tier disabled at world_size={world_size}: \
                     multi-rank lockstep swap is not implemented"
                );
            });
            return false;
        }
        true
    }

    /// Demote `slot`'s entire device state into the host store under `key`.
    /// Contract (see `infer_seam::BackendExecutor::demote_slot`): the copy is
    /// complete before returning — `swap_out_image` ends in `ctx.sync()` — so
    /// the engine may free the slot immediately. Returns `Ok(false)` when the
    /// store is at its v1 count cap (engine falls back to plain recompute).
    pub(crate) fn demote_slot(&mut self, slot: usize, key: u64) -> Result<bool> {
        ensure!(
            slot < self.num_slots,
            "DSv4 demote slot {slot} outside executor slots {}",
            self.num_slots
        );
        if !self.slot_swap_store.contains_key(&key)
            && self.slot_swap_store.len() >= self.num_slots.saturating_mul(2)
        {
            return Ok(false);
        }
        // Spec chain D2H first: the copies are stream-ordered, so the trailing
        // sync inside `swap_out_image` covers them too.
        let spec_pending = self.spec_slots[slot].pending;
        let spec_hidden = match self.spec_slots[slot].hidden.as_ref() {
            Some(hidden) => Some(
                self.model
                    .ctx
                    .stream
                    .clone_dtoh(&hidden.data)
                    .map_err(|e| anyhow::anyhow!("DSv4 swap spec hidden D2H failed: {e}"))?,
            ),
            None => None,
        };
        let image = self.slots[slot].swap_out_image(&self.model.ctx, &self.kv_adapter)?;
        self.slot_swap_store.insert(
            key,
            Dsv4SlotSwapEntry {
                image,
                spec_pending,
                spec_hidden,
            },
        );
        Ok(true)
    }

    /// Restore the whole-slot image stored under `key` into `slot`. The engine
    /// resumes decode at the demoted position right after this returns, and
    /// drops the entry via [`Self::drop_kv_slot_entries`] — the entry
    /// intentionally stays in the store here. `swap_in_image` ends in
    /// `ctx.sync()`, so both the device restore and the spec-hidden H2D (same
    /// stream, ordered before it) are complete before the host image can be
    /// dropped.
    pub(crate) fn promote_slot(&mut self, key: u64, slot: usize) -> Result<()> {
        ensure!(
            slot < self.num_slots,
            "DSv4 promote slot {slot} outside executor slots {}",
            self.num_slots
        );
        let entry = self.slot_swap_store.get(&key).ok_or_else(|| {
            anyhow::anyhow!("DSv4 whole-slot KV store has no image for key {key}")
        })?;
        self.spec_slots[slot] = Dsv4SpecSlotState {
            pending: entry.spec_pending,
            hidden: match entry.spec_hidden.as_ref() {
                Some(host) => Some(DeviceVec::from_host(&self.model.ctx, host)?),
                None => None,
            },
        };
        self.slots[slot].swap_in_image(&self.model.ctx, &mut self.kv_adapter, &entry.image)
    }

    pub(crate) fn drop_kv_slot_entries(&mut self, keys: &[u64]) {
        for key in keys {
            self.slot_swap_store.remove(key);
        }
    }

    /// Length of the longest stored position-0 prompt that is an exact leading
    /// prefix of `tokens`. `0` when the store is disabled or has no match.
    pub(crate) fn cached_prefix_match_len(&self, tokens: &[u32]) -> usize {
        self.prefix_cache.match_len(tokens)
    }

    /// Capture `slot`'s whole-slot KV image into the position-0 prefix store,
    /// keyed by `tokens` (the full prompt).
    ///
    /// CORRECTNESS INVARIANT: the request must have prefilled from absolute
    /// position 0, so the slot's materialized KV is exactly `tokens` at
    /// positions `[0, tokens.len())`. We assert the slot's materialized length
    /// equals the prompt length (start_pos==0 ⇒ every prompt token's KV is
    /// resident and no generated token's KV is included — the engine calls this
    /// before the next decode write). A mismatch means the request did not start
    /// at 0 (already had a reattached prefix) or generated past the prompt; we
    /// refuse to cache rather than store a misaligned image.
    pub(crate) fn capture_cached_prefix(&mut self, slot: usize, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        ensure!(
            slot < self.num_slots,
            "DSv4 capture_cached_prefix slot {slot} outside executor slots {}",
            self.num_slots
        );
        let materialized = self.slots[slot].seq_len();
        // start_pos==0 anchor: the slot must hold exactly the prompt prefix.
        // Caching is best-effort, so a misaligned slot is skipped (not fatal):
        // the engine only loses the reuse opportunity for this request.
        if materialized != tokens.len() {
            return Ok(());
        }
        let image = self.slots[slot].swap_out_image(&self.model.ctx, &self.kv_adapter)?;
        debug_assert_eq!(
            image.seq_len(),
            tokens.len(),
            "DSv4 position-0 prefix image seq_len must equal prompt length"
        );
        let host_bytes = image.host_bytes();
        self.prefix_cache.insert(tokens.to_vec(), image, host_bytes);
        Ok(())
    }

    /// Restore the cached position-0 prefix image for `tokens[..matched_len]`
    /// into `slot`. The engine has already allocated `matched_len` tokens of
    /// host KV pages on `slot` and resumes prefill from absolute position
    /// `matched_len` right after.
    ///
    /// CORRECTNESS INVARIANT: the cached image was captured at start_pos==0, so
    /// its KV lands at the SAME absolute positions `[0, matched_len)` here —
    /// `swap_in_image` re-resolves the target slot's page table but never
    /// re-rotates positions, so the RoPE-rotated K, the SW ring slot
    /// (`abs_pos % window`), and the DSA indexer keys are all position-identical.
    /// The spec (MTP) draft state is reset to empty: the tail prefill re-seeds it
    /// (it is per-request, not part of the prefix KV).
    pub(crate) fn restore_cached_prefix(
        &mut self,
        slot: usize,
        tokens: &[u32],
        matched_len: usize,
    ) -> Result<()> {
        ensure!(
            slot < self.num_slots,
            "DSv4 restore_cached_prefix slot {slot} outside executor slots {}",
            self.num_slots
        );
        ensure!(
            matched_len > 0 && matched_len <= tokens.len(),
            "DSv4 restore_cached_prefix matched_len {matched_len} invalid for prompt len {}",
            tokens.len()
        );
        // Take the image out of the store so the restore can mutably borrow
        // `self.slots`/`self.kv_adapter` without aliasing `self.prefix_cache`;
        // re-insert it after to keep the prefix hot for the next request.
        let entry = self.prefix_cache.take(tokens, matched_len).ok_or_else(|| {
            anyhow::anyhow!(
                "DSv4 position-0 prefix store has no image for prompt prefix len {matched_len}"
            )
        })?;
        ensure!(
            entry.image.seq_len() == matched_len,
            "DSv4 cached prefix image seq_len {} != requested matched_len {matched_len}",
            entry.image.seq_len()
        );
        // Reset the slot's spec (MTP) draft state: the tail prefill re-seeds it.
        self.spec_slots[slot] = Dsv4SpecSlotState::default();
        let result =
            self.slots[slot].swap_in_image(&self.model.ctx, &mut self.kv_adapter, &entry.image);
        // Keep the entry hot regardless of restore outcome (the image is intact;
        // a failure is a slot/device error, not a corrupt image).
        self.prefix_cache.reinsert(entry);
        result
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
        let pending = self.spec_slots[slot_idx]
            .pending
            .ok_or_else(|| anyhow::anyhow!("DSv4 MTP decode missing pending token"))?;
        ensure!(
            pending == last_token,
            "DSv4 MTP pending token {pending} != DecodeRow.last_token {last_token}"
        );
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

    fn forward_decode_batch(
        &mut self,
        rows: &[DecodeRow],
        kv_batch: &KvBatchDescriptor,
    ) -> Result<Vec<SlotToken>> {
        validate_dsv4_decode_kv_batch(rows, kv_batch)?;
        let kv_view = self.kv_adapter.prepare_kv_batch(kv_batch)?;
        validate_dsv4_decode_kv_view(rows, &kv_view)?;
        let batch = Dsv4DecodeBatch::from_rows(rows, &self.slots, self.num_slots)?;
        ensure!(
            batch.slot_ids.len() == batch.rows.len()
                && batch.tokens.len() == batch.rows.len()
                && batch.start_positions.len() == batch.rows.len()
                && batch.positions.len() == batch.rows.len(),
            "DSv4 decode batch surface length mismatch"
        );
        if batch.rows.len() == 1 {
            return self.forward_decode_row(&batch.rows[0]);
        }

        // Cross-slot batched MTP decode (batched-MTP Stage 1). B=1 already took
        // the single-row path above; B>1 drives all N chains through one batched
        // verify (MoE grouped over the verify rows, attention per-slot) instead
        // of the per-row `spec_step` loop.
        let spec_on = self.spec_requested();
        let all_greedy = batch.rows.iter().all(|row| row.params.is_greedy());
        if spec_on && all_greedy {
            for row in &batch.rows {
                ensure!(
                    row.params.is_greedy(),
                    "DSv4 MTP greedy verify currently supports greedy sampling only"
                );
                let pending = self.spec_slots[row.slot].pending.ok_or_else(|| {
                    anyhow::anyhow!("DSv4 MTP batched decode missing pending token")
                })?;
                ensure!(
                    pending == row.last_token,
                    "DSv4 MTP pending token {pending} != DecodeRow.last_token {} (slot {})",
                    row.last_token,
                    row.slot
                );
            }
            let committed = self.spec_step_batched(&batch.slot_ids, &batch.start_positions)?;
            ensure!(
                committed.len() == batch.rows.len(),
                "DSv4 batched MTP returned {} chains for {} rows",
                committed.len(),
                batch.rows.len()
            );
            let mut tokens = Vec::new();
            for (&slot, chain) in batch.slot_ids.iter().zip(committed) {
                for token in chain {
                    tokens.push(SlotToken {
                        slot,
                        token,
                        logprob: None,
                        finish: None,
                    });
                }
            }
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
        let kv_view = self.kv_adapter.prepare_kv_batch(kv_batch)?;
        validate_dsv4_prefill_kv_view(row, &kv_view)?;
        ensure!(
            row.slot < self.num_slots,
            "prefill slot {} outside DSv4 executor slots {}",
            row.slot,
            self.num_slots
        );
        ensure!(!row.tokens.is_empty(), "prefill row must carry tokens");
        if row.start_pos == 0 {
            self.slots[row.slot].reset(&self.model.ctx, &mut self.kv_adapter)?;
            self.spec_slots[row.slot] = Dsv4SpecSlotState::default();
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

/// The DSv4 executor's configured max KV sequence length (`INFER_DSV4_MAX_SEQ_LEN`,
/// default [`DSV4_DEFAULT_MAX_SEQ_LEN`]). Exposed so harnesses size their host KV
/// pool to the design max (length-agnostic — any prompt up to this works) instead
/// of the specific test prompt length.
pub fn dsv4_max_seq_len() -> usize {
    std::env::var("INFER_DSV4_MAX_SEQ_LEN")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DSV4_DEFAULT_MAX_SEQ_LEN)
}

/// Whole-step Qwen3.5/3.6 decode graph enabled? `ARLE_QWEN35_DECODE_GRAPH=1`
/// opt-in, default OFF until the pod license (≥ +10% tok/s + needle gate +
/// replay-reuse evidence per the bench spec). The eager path stays the
/// correctness floor; `Qwen35CudaExecutor::warmup` additionally gates TP and
/// host-routed MoE off regardless of this.
fn qwen35_decode_graph_enabled() -> bool {
    matches!(
        std::env::var("ARLE_QWEN35_DECODE_GRAPH").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
    )
}

/// Batched rows>1 decode for Qwen3.5/3.6 (stage 1, contiguous KV) — re-port
/// of the deleted monolith's `decode_batch` design
/// ([`crate::qwen35::Qwen35BatchDecodeState`]). DEFAULT ON: before this path
/// existed, a rows>1 plan was a hard executor error (`rows == 1` ensure →
/// engine death), so even a conservative batched path strictly dominates the
/// status quo. `ARLE_QWEN35_BATCHED_DECODE=0` is the escape hatch AND the
/// honest same-binary A/B arm: it processes rows>1 decode plans as sequential
/// per-row single-row forwards instead (a NEW loop — the old behavior was
/// death, not a fallback). rows==1 plans never consult this gate (the
/// single-row path is byte-identical either way).
fn qwen35_batched_decode_enabled() -> bool {
    !matches!(
        std::env::var("ARLE_QWEN35_BATCHED_DECODE").as_deref(),
        Ok("0" | "false" | "FALSE" | "no" | "off" | "OFF")
    )
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
/// The whole-step decode graph is opt-in (`ARLE_QWEN35_DECODE_GRAPH=1`,
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
    slots: Vec<crate::qwen35::Qwen35SlotState>,
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
    /// Fraction of available host DRAM the lazily-built L3 recall tier may claim
    /// (`--dram-fraction`, default 0.5). Stored at construction; consumed when
    /// `set_kv_recall(true)` builds `recall_tier`.
    dram_fraction: f64,
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
    /// Radix prefix reuse over the shared paged full-attn pool — same uniform
    /// mechanism as dense Qwen3 (`pages_only_reusable_prefix_blocks` + the
    /// infer-core radix cache). Resident pages only: Qwen3.6 has no prefix-tier
    /// demote/promote yet (`demote/promote_prefix_pages` no-op for Qwen35), so
    /// demoted keys are never restorable and the count stops at the first one.
    ///
    /// SOUNDNESS GATE — full-attention-only models. The radix cache keys on
    /// full-attn KV *pages*, but Qwen3.6 is a HYBRID: its gated-delta linear
    /// layers carry per-slot recurrent + conv state that is content-based,
    /// position-free, and NOT page-addressable, so a reused prefix has no way to
    /// restore those layers' state (they never processed the prefix). Reusing a
    /// prefix would attend correct full-attn KV but feed the linear layers a
    /// slot whose recurrent state never saw the prefix → silently wrong output,
    /// and the page accounting itself diverges (`pool.seq_len` self-allocated for
    /// the tail vs the engine's `start_pos`-credited `kv_seq_len`, the
    /// `decode_graph.rs` ensure that surfaced this). Until a prefix-keyed
    /// recurrent-state snapshot/restore exists, reuse is sound ONLY for a pure
    /// full-attention checkpoint (no linear layers). Otherwise report 0 so the
    /// scheduler never sets `start_pos > 0` and every prefill starts fresh at
    /// position 0 (byte-identical to the no-reuse path).
    pub(crate) fn reusable_prefix_blocks(&self, blocks: &[PrefixBlock]) -> usize {
        let cfg = &self.model.config;
        let pure_full_attn = cfg.num_full_attention_layers() == cfg.num_hidden_layers;
        if pure_full_attn {
            pages_only_reusable_prefix_blocks(blocks, |_| false)
        } else {
            0
        }
    }

    pub(crate) fn from_qwen35_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
        mem_fraction_static: f64,
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
        // `None`: MTP spec-decode flag wiring lands in a later increment; the
        // baseline load is byte-identical (no draft head loaded).
        let model =
            crate::qwen35::Qwen35Model::from_safetensors(model_path.as_ref(), max_seq_len, None)?;
        cuda_startup_log(
            "qwen35_model_load",
            model_t0,
            format_args!(
                "requested_slots={num_slots} total_pages={total_pages} max_seq_len={max_seq_len}"
            ),
        );
        // Dynamic KV mem budget (unified with DSv4 via the infer-seam kernel):
        // clamp num_slots to what post-weights free VRAM affords. Qwen3.5/3.6
        // previously admitted the requested count as-is → OOM at large
        // max_seq_len. Deterministic + NCCL min-reduced ⇒ TP-consistent.
        let budget_t0 = Instant::now();
        let num_slots = model.kv_budget_num_slots(num_slots)?;
        cuda_startup_log(
            "qwen35_kv_budget",
            budget_t0,
            format_args!("effective_slots={num_slots}"),
        );
        let slots_t0 = Instant::now();
        let mut slots = Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            slots.push(model.new_slot_state()?);
        }
        cuda_startup_log(
            "qwen35_slot_alloc",
            slots_t0,
            format_args!("slots={num_slots} max_seq_len={max_seq_len}"),
        );

        // Shared paged full-attn KV pool — the DEFAULT substrate (Phase 2 of the
        // shared-paged-KV migration). Built EAGERLY here, profile-sized from
        // MEASURED free VRAM after weights load (SGLang's mem_fraction_static),
        // NOT `num_slots × max_seq_len` (the per-slot contiguous waste). The
        // per-token cell cost is the pool's own charge for one token across all
        // full-attn layers (storage + work bytes), exactly the dense recipe.
        let pool_t0 = Instant::now();
        let num_full = model.config.num_full_attention_layers();
        let local_kv_heads = model.local_kv_heads();
        let head_dim = model.config.head_dim;
        let cell_bytes_per_token = PagedKVPool::budget_bytes_for_tokens(
            num_full,
            local_kv_heads,
            head_dim,
            1,
            KVFormat::BF16,
        ) as u64;
        // Host-admission span (the floor): every slot must be addressable up to
        // `max_seq_len`, so the device pool is at least `num_slots × total_pages`
        // pages — the same span the host CudaKvPool admits. The profiled budget
        // raises this on an ample card for more concurrency / longer context.
        let admission_pages = num_slots.saturating_mul(total_pages).max(1);
        let total_pool_pages = match model.ctx.mem_info_bytes() {
            Ok((free, total)) => {
                let profiled_tokens = infer_seam::profile_kv_pool_tokens(
                    free as u64,
                    total as u64,
                    cell_bytes_per_token,
                    mem_fraction_static,
                );
                let profiled_pages = (profiled_tokens / SUPPORTED_PAGE_SIZE as u64) as usize;
                let sized = profiled_pages.max(admission_pages).max(1);
                log::info!(
                    "CUDA Qwen3.6 full-attn KV pool profiled from measured VRAM: free {}MB / \
                     total {}MB, mem_fraction_static {mem_fraction_static}, cell \
                     {cell_bytes_per_token}B/tok ({num_full} full-attn layers × {local_kv_heads} \
                     kv-heads × {head_dim} hd) -> max_total_tokens {profiled_tokens} \
                     ({profiled_pages} pages); admission floor {admission_pages} pages \
                     (num_slots {num_slots} × total_pages {total_pages}) -> sizing {sized} pages",
                    free >> 20,
                    total >> 20,
                );
                sized
            }
            Err(e) => {
                log::warn!(
                    "CUDA Qwen3.6 full-attn KV pool: free-VRAM probe failed ({e}); falling back \
                     to admission floor {admission_pages} pages"
                );
                admission_pages
            }
        };
        let pool_token_budget = total_pool_pages * SUPPORTED_PAGE_SIZE;
        let pool_budget_bytes = PagedKVPool::budget_bytes_for_tokens(
            num_full,
            local_kv_heads,
            head_dim,
            pool_token_budget,
            KVFormat::BF16,
        );
        let full_attn_kv = PagedKVPool::with_format(
            &model.ctx,
            num_full,
            local_kv_heads,
            head_dim,
            num_slots,
            pool_budget_bytes,
            KVFormat::BF16,
        )?;
        ensure!(
            full_attn_kv.page_size == SUPPORTED_PAGE_SIZE,
            "Qwen3.6 full-attn paged pool page_size={} != {SUPPORTED_PAGE_SIZE}",
            full_attn_kv.page_size
        );
        cuda_startup_log(
            "qwen35_paged_pool_alloc",
            pool_t0,
            format_args!("pages={total_pool_pages} tokens={pool_token_budget}"),
        );

        // Whole-step decode graph: env opt-in ∧ single-GPU (NCCL all-reduce is
        // not graph-capturable on this stack — TP≥2 stays eager, same as
        // dense) ∧ every layer's decode step is a pure device-kernel sequence.
        let decode_graph_armed = qwen35_decode_graph_enabled()
            && model.tp.is_single()
            && model.decode_graph_unsupported_reason().is_none();
        let executor = Self {
            model,
            slots,
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
            recall_tier: None,
            recall_keepalive: (0..num_slots).map(|_| Vec::new()).collect(),
            dram_fraction: DEFAULT_DRAM_FRACTION,
        };
        cuda_startup_log(
            "qwen35_executor_total",
            total_t0,
            format_args!("slots={num_slots} max_seq_len={max_seq_len}"),
        );
        Ok(executor)
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
                 (set ARLE_QWEN35_DECODE_GRAPH=1 to enable)"
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

    /// Apply the `--dram-fraction` knob: store it so the lazily-built L3 recall
    /// tier sizes its host-DRAM budget at that fraction. Pre-serve only (the tier
    /// is not yet built — it allocates on the first `set_kv_recall(true)`).
    pub(crate) fn set_dram_fraction(&mut self, fraction: f64) {
        self.dram_fraction = fraction;
    }

    /// Opt into session KV-recall (`--kv-recall`, default off). The paged
    /// full-attn pool (`full_attn_kv`) is ALWAYS resident since the shared-paged
    /// migration — the DEFAULT forward already attends it (full resident set, no
    /// eviction). Enabling recall layers the eviction/scoring *cycle* on the
    /// SAME pool (working-set restriction) and lazily builds its L3 tier on the
    /// first enable. Default-off keeps the full resident attention.
    pub(crate) fn set_kv_recall(&mut self, enabled: bool) -> Result<()> {
        self.kv_recall = enabled;
        if enabled && self.recall_tier.is_none() {
            // L3 write-through tier: source of truth for evict-dropped middle
            // blocks. One entry == one pool page image (all `num_full` layers,
            // K+V). Host-DRAM budget is dram_fraction-profiled (same policy as the
            // prefix tier); NVMe spill is opt-in via the prefix-tier
            // `--kv-ssd-path` wiring. Reuses the SAME `CudaKvTierStore` transport
            // the dense arm's prefix/write-through tier uses (R5 — one store kind).
            let page_bytes = self
                .full_attn_kv
                .as_ref()
                .map(|p| p.storage_bytes_per_page())
                .ok_or_else(|| {
                    anyhow::anyhow!("--kv-recall: full-attn paged pool not allocated")
                })?;
            let tier = CudaKvTierStore::with_budget(
                default_t1_budget_bytes(self.dram_fraction),
                page_bytes,
            );
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
        {
            let pool = self
                .full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (full_attn_paged)");
            // This self-allocating path materializes the slot from a fresh start
            // (`free_slot` ran at `start_pos == 0` in `submit_prefill_row`), so
            // the pool must hold exactly `start_pos` tokens before appending the
            // tail. A `start_pos > 0` here means radix reuse leaked through the
            // `reusable_prefix_blocks` soundness gate — fail loudly rather than
            // double-count (`pool.seq_len` would end at the tail length, not
            // `start_pos + tail`, surfacing as a confusing `decode_graph` mismatch
            // many steps later). The hybrid recurrent state is unrestorable for a
            // reused prefix regardless (see `reusable_prefix_blocks`).
            ensure!(
                pool.seq_len(slot) == row.start_pos,
                "Qwen3.6 default-paged prefill: pool seq_len {} != start_pos {} for slot {} \
                 (radix prefix reuse is not supported for the hybrid recurrent state)",
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
            &row.tokens,
            row.start_pos,
            &row.params,
            position,
            &mut rc,
        )
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

    /// One prefill row over the paged recall pool (`--kv-recall`), and the ONE
    /// place the whole recall cycle runs (the write-through model's non-negotiable
    /// rule: "decode 不召回; prefetch 只在 prefill; 其他时机不交互").
    ///
    /// 1. Self-allocate the prompt tokens, build the full-resident page table, run
    ///    the paged forward — this writes the new KV and reads back the layer-0
    ///    prefill query (`rc.layer0_query`, the mean of the last `m` prompt tokens'
    ///    post-RoPE queries; see `full_attention_paged`).
    /// 2. Score the whole history against that query (`recompute_recall_plan`,
    ///    `allow_prefetch=true`): pick the working set (sink + local + top-k
    ///    relevant blocks), choosing `evict_pages` (cold middle to drop) and
    ///    `prefetch_pages` (chosen blocks that are tier-resident sentinels).
    /// 3. Batched-H2D prefetch the chosen tier-resident blocks back into HBM
    ///    (`reinstate_slot_page` + `copy_pages_from_host`), then resolve the FIXED
    ///    `recall_pages` working set the decode steps will attend.
    /// 4. Write-back-evict the cold middle (`copy_pages_to_host` → `tier.insert`)
    ///    and free its physical pages IMMEDIATELY (`evict_slot_page`): prefill's
    ///    forward + sampling already drained the compute stream, so there is no
    ///    in-flight attention to race — no decode-step keepalive needed.
    ///
    /// After this returns, `self.recall[slot].recall_pages()` holds the immutable
    /// working set; decode does nothing but append + attend it (see
    /// [`Self::decode_row_recall`]).
    fn prefill_row_recall(&mut self, row: &infer_plan::PrefillRow, position: u64) -> Result<u32> {
        let slot = row.slot;
        let cfg = self.recall_cfg;
        // Self-allocate the new tokens (extends the page table + pool seq_len).
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
        // (1) Forward (paged prefill). Borrow split: forward needs &model +
        // &mut slots[slot] + &mut workspace + &mut pool (through rc). `rc` carries
        // back the layer-0 prefill query for the recall score below.
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

        // (2) Score the whole history against the prefill query and plan the FIXED
        // working set (sink + local + top-k relevant). `allow_prefetch=true` lets a
        // tier-resident (previously evicted) block re-enter the set; page-list
        // resolution is deferred until after the prefetch patches the table.
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

        // (3) Prefetch the chosen tier-resident blocks back into HBM (alloc a fresh
        // page, H2D from the tier, patch the sentinel). Decode never does this; it
        // is the one batched prefill sync point. If the pool is out of free pages or
        // the tier lost the entry, the block stays evicted (no crash; it simply
        // won't be in the working set).
        for logical in prefetch_pages {
            let key = tier_block_u64(slot as u64, logical as u64);
            let Some(tier) = self.recall_tier.as_mut() else {
                break;
            };
            let payload = match tier.read(key) {
                Ok(p) => p.into_owned(),
                Err(_) => continue, // tier dropped it (LRU/capacity) → stays evicted
            };
            if let Some(pool) = self.full_attn_kv.as_mut() {
                if let Some(new_page) = pool.reinstate_slot_page(slot, logical) {
                    pool.copy_pages_from_host(&self.model.ctx, &[new_page], &payload)?;
                }
            }
        }

        // (3b) Resolve the FIXED working-set page list now that prefetch has patched
        // any reinstated sentinels to real ids. Decode reads this and never mutates it.
        if let Some(state) = self.recall.get_mut(slot) {
            if let Some(pool) = self.full_attn_kv.as_ref() {
                state.resolve_recall_pages(pool, slot);
            }
        }

        // (4) Write-back-evict the cold middle pages: mirror each to the L3 tier,
        // then free its physical page IMMEDIATELY (`evict_slot_page`). Prefill's
        // forward + sampling already drained the compute stream, so no in-flight
        // attention can be reading these pages — the deferred-keepalive dance that
        // the per-decode-step path needed is unnecessary here. A tier-full
        // write_through keeps the page resident (no KV loss).
        for logical in evict_pages {
            let physical = {
                let pool = self.full_attn_kv.as_ref().expect("full_attn_kv");
                pool.page_indices(slot)
                    .get(logical)
                    .copied()
                    .filter(|&p| p != cuda_kernels::prelude::EVICTED_PAGE)
            };
            let Some(physical) = physical else {
                continue; // already evicted
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
                continue; // tier full → keep the page resident (no KV loss)
            }
            if let Some(pool) = self.full_attn_kv.as_mut() {
                pool.evict_slot_page(slot, logical);
            }
        }
        Ok(token)
    }

    /// One decode row over the paged recall pool (`--kv-recall`): **append +
    /// attend, ZERO tier I/O** (the write-through model's non-negotiable rule —
    /// "decode 不召回; prefetch 只在 prefill; 其他时机不交互"). The whole recall
    /// cycle (score → evict → H2D prefetch) ran ONCE at prefill
    /// ([`Self::prefill_row_recall`]) and fixed the working set; decode only:
    ///
    /// 1. Alloc this step's token (extends the tail page).
    /// 2. Read the FIXED `recall_pages` working set (immutable for this decode
    ///    run; never mutated here) — or the full contiguous list if the session
    ///    still fits the budget (prefill chose no restriction).
    /// 3. Forward over that page table (paged decode) + sample.
    ///
    /// There is NO `recompute_recall_plan`, NO `copy_pages_to_host`/`_from_host`,
    /// NO `write_through`, NO `reinstate_slot_page`, NO keepalive release — none of
    /// the tier machinery is reachable from here (that is the whole point: a
    /// 6000-token recall request must not pay the per-step cycle that took >10 min).
    /// A single ultra-long generation with no re-prefill keeps whatever working set
    /// prefill chose (accepted boundary, ckl's design).
    fn decode_row_recall(&mut self, row: &DecodeRow, position: u64) -> Result<u32> {
        let slot = row.slot;
        // (1) Alloc this step's token so the tail page exists.
        {
            let pool = self
                .full_attn_kv
                .as_mut()
                .expect("full_attn_kv present (recall_active)");
            pool.alloc_tokens(slot, 1)?;
        }
        let cache_len = row.kv_seq_len + 1; // incl. this step's token
        // (2) The FIXED working set chosen at prefill; else the full resident list
        // (session under budget → prefill left no restriction). Read-only here.
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
        // (3) Forward (paged decode) + sample. Borrow split. `rc.layer0_query` is
        // collected by the forward but UNUSED on decode — no recall re-scoring,
        // no eviction, no prefetch happens here.
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
    /// `Ok(None)` for eager fallback (gate off, out-of-budget position, or a
    /// capture/replay failure — which permanently downgrades to eager with a
    /// warn, never fatal). On `Some`, the sampled token is final and the
    /// slot's seq_len has been advanced.
    ///
    /// See [`crate::qwen35::Qwen35Model::forward_decode_step_captured`] for
    /// the captured-kernel capture-safety table and the perf formula.
    fn try_graph_decode(&mut self, row: &DecodeRow, position: u64) -> Result<Option<u32>> {
        if !self.decode_graph_armed {
            return Ok(None);
        }
        // Replay-time invariant: host `ensure!`s inside the captured closure
        // run only on warm/capture steps, so the budget check must live here,
        // on EVERY step. Out-of-budget falls back to eager for the canonical
        // error message.
        if row.kv_seq_len + 1 > self.model.max_seq_len() {
            return Ok(None);
        }
        if self.decode_graph.is_none() {
            self.decode_graph = Some(Qwen35DecodeGraph::new(
                self.num_slots,
                &self.model.ctx.stream,
            ));
        }
        // Split borrows: the capture closure needs &model + &mut slot state +
        // &mut decode workspace while the graph entry itself is borrowed.
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

        // Stage the per-step device scalars OUTSIDE the graph (dense
        // stage1_write pattern): token id + position into fixed device
        // buffers the captured kernels read.
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
                // Decode-workspace addresses drifted since capture (release →
                // re-alloc). Replaying would launch against freed memory —
                // drop the capture and re-capture against the new addresses.
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
            // Any capture/replay failure downgrades to eager permanently —
            // never fatal (dense pattern). A mid-CAPTURE error recorded (not
            // executed) its kernels, so device state is untouched and the
            // eager re-run of this step is clean.
            warn!(
                "Qwen3.5 whole-step decode graph failed (slot {slot_idx}), \
                 downgrading to eager forward: {e}"
            );
            self.decode_graph_armed = false;
            self.decode_graph = None;
            return Ok(None);
        }
        // Host-side state advance happens exactly once per step HERE — the
        // captured closure is host-state-free (replay skips host code).
        slots[slot_idx].advance_seq_len(1);

        // Reuse-evidence probes (license needs replay counts, not
        // capture-exists). Key cardinality must stay ≤ num_slots.
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
            self.slots[row.slot].reset(&self.model.ctx)?;
            if let Some(dg) = self.decode_graph.as_mut() {
                dg.graphs[row.slot].rearm_warm(1);
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
            self.prefill_row_recall(row, position)
        } else {
            // Default paged prefill: full attention over all resident pages, no
            // eviction (the dense model applied to Qwen3.6 — Phase 2).
            self.prefill_row_paged_default(row, position)
        }
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
    /// `ARLE_QWEN35_BATCHED_DECODE=0`, runs the rows sequentially as
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

        // The contiguous batched-decode kernel reads each slot's per-slot
        // `k_caches`/`v_caches`, which the shared-paged default no longer
        // allocates. Under the paged default (always-on) and under `--kv-recall`
        // (serial per-slot rescore/evict), decode routes per-row through
        // `submit_decode_row` (paged forward). The batched lane only runs in the
        // legacy contiguous build (no paged pool) with the env gate on.
        if !qwen35_batched_decode_enabled() || self.recall_active() || self.full_attn_paged() {
            // Sequential per-row fallback (A/B arm / escape hatch / paged / recall).
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
        let mut slot = self.model.new_slot_state()?;
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
        return argmax(ctx, logits);
    }

    // TODO: repetition/frequency/presence penalties need the per-request
    // generated-token history threaded through the executor.
    let logits_host = logits.to_host(ctx)?;
    Ok(infer_plan::sample_token(&logits_host, params, position))
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
        return crate::ops::argmax_into(ctx, logits, argmax_out);
    }
    let logits_host = logits.to_host(ctx)?;
    Ok(infer_plan::sample_token(&logits_host, params, position))
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
    for (idx, &value) in logits_host.iter().enumerate() {
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
    println!("sample_topk variant={variant} position={position} top={best:?} margin={margin:.6}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CudaKvCacheDtype, PrefixImageStore};
    use infer_seam::KvCacheDtype;

    // The payload is opaque to the store, so tests use a `u32` tag as the image
    // and pass an explicit host-byte size — exactly how the executor passes the
    // real `Dsv4SlotImage::host_bytes()`.
    fn store(budget_bytes: usize) -> PrefixImageStore<u32> {
        PrefixImageStore::new(budget_bytes)
    }

    #[test]
    fn exact_leading_prefix_matches_longest() {
        let mut s = store(1 << 20);
        s.insert(vec![10, 20], 1, 100);
        s.insert(vec![10, 20, 30, 40], 2, 100);
        // A query extending the longer stored prompt matches the longer one.
        assert_eq!(s.match_len(&[10, 20, 30, 40, 50]), 4);
        // A query that only extends the shorter prompt matches the shorter one.
        assert_eq!(s.match_len(&[10, 20, 99]), 2);
        // An exact full match returns the full length.
        assert_eq!(s.match_len(&[10, 20]), 2);
        // A divergent query matches nothing.
        assert_eq!(s.match_len(&[10, 99]), 0);
        // A query SHORTER than every stored prompt matches nothing (a stored
        // prompt may only be reused as a leading prefix, never truncated).
        assert_eq!(s.match_len(&[10]), 0);
    }

    #[test]
    fn take_returns_image_and_reinsert_keeps_it_hot() {
        let mut s = store(1 << 20);
        s.insert(vec![5, 6, 7], 42, 100);
        let entry = s
            .take(&[5, 6, 7, 8], 3)
            .expect("exact prefix entry present");
        assert_eq!(entry.image, 42, "take returns the stored payload");
        assert_eq!(s.used_bytes, 0, "take debits the byte accounting");
        assert_eq!(
            s.match_len(&[5, 6, 7]),
            0,
            "taken entry is gone until reinserted"
        );
        s.reinsert(entry);
        assert_eq!(s.match_len(&[5, 6, 7]), 3, "reinsert restores the entry");
        assert_eq!(s.used_bytes, 100, "reinsert re-credits the bytes");
    }

    #[test]
    fn lru_evicts_coldest_when_over_budget() {
        // Budget holds two 100-byte entries; a third evicts the coldest.
        let mut s = store(250);
        s.insert(vec![1], 1, 100);
        s.insert(vec![2], 2, 100);
        // Touch entry 1 so entry 2 becomes the coldest.
        assert_eq!(s.match_len(&[1]), 1);
        let _ = s.take(&[1], 1).map(|e| s.reinsert(e)); // mark 1 hottest
        s.insert(vec![3], 3, 100); // over budget → evict coldest (entry 2)
        assert_eq!(s.match_len(&[1]), 1, "hot entry survives");
        assert_eq!(s.match_len(&[3]), 1, "new entry present");
        assert_eq!(s.match_len(&[2]), 0, "coldest entry evicted");
        assert!(s.used_bytes <= 250, "store stays within budget");
    }

    #[test]
    fn oversized_image_is_rejected() {
        let mut s = store(100);
        s.insert(vec![1, 2, 3], 7, 200); // larger than the whole budget
        assert_eq!(s.match_len(&[1, 2, 3]), 0, "oversized image is not stored");
        assert_eq!(s.used_bytes, 0);
    }

    #[test]
    fn reinsert_of_identical_prompt_replaces_in_place() {
        let mut s = store(1 << 20);
        s.insert(vec![1, 2], 1, 100);
        s.insert(vec![1, 2], 2, 150); // same key, new image + size
        assert_eq!(s.used_bytes, 150, "in-place replace re-accounts bytes");
        let entry = s.take(&[1, 2], 2).expect("present");
        assert_eq!(entry.image, 2, "replaced payload is the latest");
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
