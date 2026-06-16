//! Real CUDA executor: the engine-facing step driver and sampling tail.
//!
//! Wraps the loaded [`CudaModel`] + device [`PagedKVPool`], validates the
//! single-row plan, mirrors host→device page allocation, runs the forward, and
//! samples the next token (`sample_cuda_token`: greedy argmax / host sampling).

use std::path::Path;

use anyhow::{Result, ensure};
use cuda_kernels::KVFormat;
use cuda_kernels::prelude::{DeviceContext, DeviceVec, PagedKVPool};
use infer_plan::{DecodeRow, ForwardPlan, SamplingParams, SlotToken, StepOutput};
use infer_seam::{KvBatchDescriptor, KvBatchRowKind, KvPool};
use log::{info, warn};

use crate::attention::ModelKvAdapter;
use crate::decode_graph::DecodeGraphContext;
use crate::decode_graph_key::{DECODE_GRAPH_BATCH, DecodeGraphKey};
use crate::graph::GraphBucket;
use crate::model::CudaModel;
use crate::ops::argmax;

#[path = "executor/spec_decode.rs"]
mod spec_decode;

const SUPPORTED_PAGE_SIZE: usize = 16;
const DSV4_DEFAULT_MAX_SEQ_LEN: usize = 4096;

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

/// The real cuda-kernels executor. Dense Qwen3 runs the paged continuous-batching
/// path ([`QwenCudaExecutor`]); Qwen3.5/3.6 HYBRID MoE runs the gated-delta +
/// periodic-full-attention forward ([`Qwen35CudaExecutor`]), which owns its KV
/// state (per-slot full-attn caches + recurrent state, no `PagedKVPool`);
/// DSv4-Flash runs the MLA + hyper-connection + FP8 DeepGEMM MoE forward
/// ([`Dsv4CudaExecutor`]), which also owns its own MLA KV state. Both
/// state-owning executors disable the decode graph.
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
    ) -> Result<Self> {
        Ok(Self::Qwen(Box::new(
            QwenCudaExecutor::from_qwen3_bf16_safetensors(
                model_path,
                num_slots,
                total_pages,
                kv_dtype,
            )?,
        )))
    }

    pub(crate) fn from_qwen35_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
    ) -> Result<Self> {
        Ok(Self::Qwen35(Box::new(
            Qwen35CudaExecutor::from_qwen35_safetensors(model_path, num_slots, total_pages)?,
        )))
    }

    /// Build the DSv4-Flash executor (MLA + HC + FP8 MoE, multi-GPU TP/EP).
    /// `mtp_draft_tokens`: `Some(n)` = config-driven MTP spec decode on (draft
    /// depth `n`); `None` falls back to the `ARLE_DSV4_SPEC_DECODE` env gate.
    pub(crate) fn from_dsv4_fp8_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        max_seq_len: usize,
        mtp_draft_tokens: Option<usize>,
    ) -> Result<Self> {
        Ok(Self::Dsv4(Box::new(
            Dsv4CudaExecutor::from_dsv4_fp8_safetensors(
                model_path,
                num_slots,
                max_seq_len,
                mtp_draft_tokens,
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

    /// T1 host-tier hooks. Only dense Qwen3 has a page-addressable device
    /// pool with cross-request prefix reuse, so it is the only arm with a
    /// tier store; the hybrid/DSv4 arms (prefix cache disabled — recurrent /
    /// ring sidecar state, see infer-api's carve-out) report capacity 0 and
    /// the engine never calls the other hooks.
    pub(crate) fn kv_tier_capacity_pages(&self) -> usize {
        match self {
            Self::Qwen(q) => q.kv_tier_capacity_pages(),
            Self::Qwen35(_) | Self::Dsv4(_) => 0,
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
                anyhow::bail!("KV tier store is only wired for the dense Qwen3 arm")
            }
        }
    }

    pub(crate) fn drop_kv_tier_entries(&mut self, keys: &[u64]) {
        if let Self::Qwen(q) = self {
            q.drop_kv_tier_entries(keys);
        }
    }

    /// Whole-slot KV tier hooks (#84/#85 Route B). Only the DSv4 arm owns
    /// page-less per-slot state it can demote/promote as one image; the Qwen
    /// arms keep the page-granular tier above and report no slot tier, so the
    /// engine never routes their preemptions here.
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
                anyhow::bail!("whole-slot KV tier store is only wired for the DSv4 arm")
            }
        }
    }

    pub(crate) fn drop_kv_slot_entries(&mut self, keys: &[u64]) {
        if let Self::Dsv4(d) = self {
            d.drop_kv_slot_entries(keys);
        }
    }

    /// Re-budget the T1 tier store (`0` disables; pre-serve only). No-op on
    /// arms without a tier store.
    pub(crate) fn set_kv_tier_budget_bytes(&mut self, bytes: usize) {
        if let Self::Qwen(q) = self {
            q.set_kv_tier_budget_bytes(bytes);
        }
    }

    /// Attach the opt-in T2 disk spill level (pre-serve only). Returns whether
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
}

use crate::kv_tier::{CudaKvTierStore, default_t1_budget_bytes};

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
    pub(crate) fn from_qwen3_bf16_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
        kv_dtype: CudaKvCacheDtype,
    ) -> Result<Self> {
        ensure!(num_slots > 0, "CudaExecutor requires at least one slot");
        ensure!(
            total_pages > 0,
            "CudaExecutor requires at least one KV page"
        );

        let model = CudaModel::from_safetensors(model_path.as_ref())?;
        let kv_format = kv_dtype.kv_format();
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
        let tier =
            CudaKvTierStore::with_budget(default_t1_budget_bytes(), kv.storage_bytes_per_page());
        Ok(Self {
            model,
            kv,
            tier,
            num_slots,
            slot_progress,
            decode_ctx: None,
            graphs: None,
        })
    }

    pub(crate) fn kv_tier_capacity_pages(&self) -> usize {
        self.tier.capacity_pages()
    }

    /// Re-budget the T1 tier store (`0` disables). Pre-serve only: any
    /// existing entries are dropped, so callers configure this right after
    /// construction, before the engine demotes anything.
    pub(crate) fn set_kv_tier_budget_bytes(&mut self, bytes: usize) {
        self.tier = CudaKvTierStore::with_budget(bytes, self.kv.storage_bytes_per_page());
    }

    /// Attach the opt-in T2 disk spill level (`--kv-ssd-path`). Pre-serve only.
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
        }
        *progress = SlotProgress { epoch, len: end };
        Ok(())
    }
}

/// DSv4-Flash executor: drives [`crate::dsv4::Dsv4Model::forward_tokens`].
/// Prefill/mixed still run one scheduled row; pure decode can accept multiple
/// rows and currently loops over the existing single-row forward as the
/// correctness foundation for later true batched kernels. DSv4 owns its MLA KV
/// state inside the forward (bf16 SW rings + compressor pending/compressed
/// pools), so it does NOT use a [`PagedKVPool`]. The decode graph is disabled
/// (MLA host-routing per step).
pub(crate) struct Dsv4CudaExecutor {
    model: crate::dsv4::Dsv4Model,
    slots: Vec<crate::dsv4::Dsv4SlotState>,
    kv_adapter: crate::attention::Dsv4KvAdapter,
    spec_slots: Vec<Dsv4SpecSlotState>,
    /// `Some(n)` = config-driven MTP spec decode on (draft depth `n`, from the
    /// serve path's `--spec-type mtp`); `None` falls back to the
    /// `ARLE_DSV4_SPEC_DECODE` env gate at each spec branch.
    spec_draft_tokens: Option<usize>,
    num_slots: usize,
    mtp_accepts: usize,
    mtp_rejects: usize,
    /// Whole-slot KV tier store (#84/#85 Route B): host images of demoted
    /// slots, keyed by the engine-minted swap key. Capacity is a v1 COUNT cap
    /// of `2 * num_slots` images (each image ≈ one slot's device KV footprint,
    /// so host RAM is bounded at ~2× the device arena; preemption churn beyond
    /// that signals thrash where plain recompute is the better fallback). A
    /// byte-budget cap (CudaKvTierStore reuse) is the follow-up tracked in
    /// docs/plans/2026-06-11-dsv4-whole-slot-kv-swap.md.
    slot_swap_store: std::collections::BTreeMap<u64, Dsv4SlotSwapEntry>,
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

/// Gate for the layer-major batched DSv4 decode path. Default ON since the
/// 2026-06-15 N=4 flip (see the in-body note), so the `arle serve
/// --dsv4-batched-decode` CLI flag is now a no-op — the lane is on by default.
/// Force the per-row byte-identical reference with `INFER_DSV4_BATCHED_DECODE=0`.
fn dsv4_batched_decode_enabled() -> bool {
    // Default ON (2026-06-15 N=4 flip): the no-MTP decode path auto-batches at
    // `rows >= dsv4_batched_decode_min_rows()`. `--spec-type` defaults to None, so
    // the DEFAULT serve takes the batched lane at c>=4 (the +58% @c=8 win) with no
    // env var and no MTP-state risk (the gate also requires `!spec`; batched+MTP
    // reconciliation stays deferred). Force the per-row byte-identical reference
    // with `INFER_DSV4_BATCHED_DECODE=0`.
    match std::env::var("INFER_DSV4_BATCHED_DECODE") {
        Ok(v) => !matches!(
            v.as_str(),
            "0" | "false" | "FALSE" | "no" | "off" | "OFF" | ""
        ),
        Err(_) => true,
    }
}

/// Minimum concurrent decode rows for the batched lane (default 4 = the N=4
/// crossover ckl licensed: batched amortization overtakes MTP's c-independent
/// ~2×/row at c>=3 prod / c>=4 short, [c-sweep]
/// (docs/experience/wins/2026-06-14-dsv4-batched-decode-csweep-threshold-n4.md)).
/// Below this the per-row path runs. Clamped to >=2 (single-row never batches).
/// `INFER_DSV4_BATCHED_DECODE_MIN_ROWS`.
fn dsv4_batched_decode_min_rows() -> usize {
    std::env::var("INFER_DSV4_BATCHED_DECODE_MIN_ROWS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&v| v >= 2)
        .unwrap_or(4)
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
    ) -> Result<Self> {
        ensure!(num_slots > 0, "Dsv4CudaExecutor requires at least one slot");
        ensure!(max_seq_len > 0, "Dsv4CudaExecutor requires max_seq_len > 0");
        let model = crate::dsv4::Dsv4Model::from_dsv4_fp8_safetensors(
            model_path.as_ref(),
            mtp_draft_tokens,
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
            spec_draft_tokens: mtp_draft_tokens,
            num_slots,
            mtp_accepts: 0,
            mtp_rejects: 0,
            slot_swap_store: std::collections::BTreeMap::new(),
        })
    }

    /// Whole-slot KV tier gate (#84/#85 Route B): single-rank only for v1.
    pub(crate) fn kv_slot_tier_enabled(&self) -> bool {
        let world_size = self.model.tp.config().world_size;
        if world_size > 1 {
            // Multi-rank demote/promote must execute on EVERY rank in lockstep
            // (the seam hooks fire on the coordinator only), or the
            // deterministic planner diverges and NCCL deadlocks. The
            // multiproc-relay SwapOut/SwapIn envelopes are the tracked
            // follow-up in docs/plans/2026-06-11-dsv4-whole-slot-kv-swap.md.
            static MULTI_RANK_LOGGED: std::sync::Once = std::sync::Once::new();
            MULTI_RANK_LOGGED.call_once(|| {
                info!(
                    "DSv4 whole-slot KV tier disabled at world_size={world_size}: \
                     multi-rank lockstep swap envelopes are the tracked follow-up \
                     (docs/plans/2026-06-11-dsv4-whole-slot-kv-swap.md)"
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

    fn forward_prefill_tokens(
        &mut self,
        slot_idx: usize,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
        final_prefill: bool,
    ) -> Result<Vec<u32>> {
        if self.spec_draft_tokens.is_some() || crate::dsv4::dsv4_spec_decode_enabled() {
            let (token, hidden) = self.model.forward_tokens_with_hidden(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                tokens,
                start_pos,
                params,
                position,
            )?;
            if final_prefill {
                self.spec_slots[slot_idx].pending = Some(token);
                self.spec_slots[slot_idx].hidden = Some(hidden);
            } else {
                self.spec_slots[slot_idx] = Dsv4SpecSlotState::default();
            }
            Ok(vec![token])
        } else {
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
        if !(self.spec_draft_tokens.is_some() || crate::dsv4::dsv4_spec_decode_enabled()) {
            let token = self.model.forward_tokens(
                &mut self.slots[slot_idx],
                &mut self.kv_adapter,
                &[last_token],
                start_pos,
                params,
                position,
            )?;
            self.model.dump_mtp_rollback_state(
                &self.slots[slot_idx],
                "nonspec_after_forward",
                start_pos + 1,
            )?;
            return Ok(vec![token]);
        }

        ensure!(
            params.is_greedy(),
            "DSv4 MTP greedy verify currently supports greedy sampling only"
        );
        let pending = self.spec_slots[slot_idx]
            .pending
            .ok_or_else(|| anyhow::anyhow!("DSv4 MTP decode missing pending token"))?;
        ensure!(
            pending == last_token,
            "DSv4 MTP pending token {pending} != DecodeRow.last_token {last_token}"
        );
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
        // Cross-slot batched MTP decode (batched-MTP Stage 1). Default OFF
        // (`ARLE_DSV4_BATCHED_MTP`): when ON, spec is on, and the batch has
        // `>= dsv4_batched_decode_min_rows()` rows, drive all N chains through
        // ONE batched verify (MoE grouped over the verify rows, attention
        // per-slot) instead of the per-row `spec_step` loop. With the env unset
        // the per-row spec loop below runs (byte-identical reference / the
        // B=1 / c<N path).
        let spec_on = self.spec_draft_tokens.is_some() || crate::dsv4::dsv4_spec_decode_enabled();
        if spec_on
            && crate::dsv4::dsv4_batched_mtp_enabled()
            && batch.rows.len() >= dsv4_batched_decode_min_rows()
        {
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
            let committed =
                self.spec_step_batched(&batch.slot_ids, &batch.start_positions, &batch.positions)?;
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
        // MoE). Default ON at `rows >= dsv4_batched_decode_min_rows()` (N=4); below
        // the threshold and under `--spec-type mtp` the per-row loop below runs
        // (the byte-identical reference / the MTP spec path). Only the multi-row,
        // non-spec path batches.
        if dsv4_batched_decode_enabled()
            && batch.rows.len() >= dsv4_batched_decode_min_rows()
            && !spec_on
        {
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
            return Ok(batch
                .slot_ids
                .iter()
                .zip(out)
                .map(|(&slot, token)| SlotToken {
                    slot,
                    token,
                    logprob: None,
                    finish: None,
                })
                .collect());
        }

        let mut tokens = Vec::with_capacity(batch.rows.len());
        for (idx, row) in batch.rows.iter().enumerate() {
            debug_assert_eq!(batch.slot_ids[idx], row.slot);
            debug_assert_eq!(batch.tokens[idx], row.last_token);
            debug_assert_eq!(batch.start_positions[idx], row.start_pos);
            debug_assert_eq!(batch.positions[idx], row.position);
            tokens.extend(self.forward_decode_row(row)?);
        }
        Ok(tokens)
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
        let (verify_one, _) = self.model.forward_tokens_verify(
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
            verify_one.first().copied() == Some(normal_one),
            "DSv4 verify selftest one-token mismatch: verify={verify_one:?} normal={normal_one}"
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
        let (verify_one, _) = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &[token_a],
            start_pos,
            (start_pos + 1) as u64,
        )?;
        let token_b = verify_one[0];
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
        let (verify_two, _) = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &mut self.kv_adapter,
            &[token_a, wrong_b],
            start_pos,
            (start_pos + 1) as u64,
        )?;
        ensure!(
            verify_two.first() == verify_one.first(),
            "DSv4 verify selftest two-token row0 mismatch: one={verify_one:?} two={verify_two:?}"
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
                "[dsv4-mtp-selftest] PASS token_a={token_a} token_b={token_b} wrong_b={wrong_b} verify_two={verify_two:?}"
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
    pub(crate) fn from_qwen35_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
    ) -> Result<Self> {
        ensure!(
            num_slots > 0,
            "Qwen35CudaExecutor requires at least one slot"
        );
        ensure!(
            total_pages > 0,
            "Qwen35CudaExecutor requires at least one KV page"
        );
        // The host CudaKvPool pages the logical seq budget; size each slot's
        // contiguous full-attn cache to the same token budget.
        let max_seq_len = total_pages * SUPPORTED_PAGE_SIZE;
        let model = crate::qwen35::Qwen35Model::from_safetensors(model_path.as_ref(), max_seq_len)?;
        // Dynamic KV mem budget (unified with DSv4 via the infer-seam kernel):
        // clamp num_slots to what post-weights free VRAM affords. Qwen3.5/3.6
        // previously admitted the requested count as-is → OOM at large
        // max_seq_len. Deterministic + NCCL min-reduced ⇒ TP-consistent.
        let num_slots = model.kv_budget_num_slots(num_slots)?;
        let mut slots = Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            slots.push(model.new_slot_state()?);
        }
        // Whole-step decode graph: env opt-in ∧ single-GPU (NCCL all-reduce is
        // not graph-capturable on this stack — TP≥2 stays eager, same as
        // dense) ∧ every layer's decode step is a pure device-kernel sequence.
        let decode_graph_armed = qwen35_decode_graph_enabled()
            && model.tp.is_single()
            && model.decode_graph_unsupported_reason().is_none();
        Ok(Self {
            model,
            slots,
            workspace: crate::qwen35::Qwen35Workspace::new(),
            num_slots,
            decode_graph_armed,
            decode_graph: None,
            batch_decode: None,
        })
    }

    /// Boot-time decode-graph verdict log (mirrors the dense `warmup` info
    /// messages). Capture itself is lazy — one whole-step capture per slot on
    /// its first gated decode (after `CudaGraphState`'s universal eager warm
    /// run), so unused slots never pay capture/instantiation memory.
    pub(crate) fn warmup(&mut self) -> Result<()> {
        let (warmed_shapes, warm_m) = self.model.warm_fp8_deepgemm_dense_prefill()?;
        if warmed_shapes > 0 {
            info!(
                "Qwen3.5 FP8 dense DeepGEMM warmed {warmed_shapes} projection shape(s) at M={warm_m}"
            );
        }
        let (grouped_shapes, grouped_tokens, grouped_min_rows, grouped_max_rows) =
            self.model.warm_fp8_deepgemm_grouped_prefill()?;
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
            return Ok(());
        }
        if !self.model.tp.is_single() {
            info!(
                "Qwen3.5 whole-step decode graph disabled under tensor parallelism \
                 (world_size>1, NCCL collectives are not graph-capturable); \
                 using eager forward"
            );
            return Ok(());
        }
        if let Some(reason) = self.model.decode_graph_unsupported_reason() {
            info!("Qwen3.5 whole-step decode graph disabled: {reason}; using eager forward");
            return Ok(());
        }
        debug_assert!(self.decode_graph_armed);
        info!(
            "Qwen3.5 whole-step decode graph ARMED ({} slots; lazy capture per slot, \
             one eager warm run before each first capture; eager fallback on any failure)",
            self.num_slots
        );
        Ok(())
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
        }
        let position = (row.start_pos + row.tokens.len()) as u64;
        self.model.forward_tokens(
            &mut self.slots[row.slot],
            &mut self.workspace,
            &row.tokens,
            row.start_pos,
            &row.params,
            position,
        )
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
        // Whole-step graph lane first (opt-in); eager forward is the
        // correctness floor and the fallback on any graph miss/failure.
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

        if !qwen35_batched_decode_enabled() {
            // Sequential per-row fallback (A/B arm / escape hatch).
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

    /// Fold a fresh student LoRA update into the resident q/v projection weights
    /// (OPD per-step re-merge). Delegates to [`crate::qwen35::Qwen35Model`].
    pub(crate) fn remerge_student_lora(
        &mut self,
        update: crate::qwen35::StudentLoraUpdate,
    ) -> Result<()> {
        self.ensure_not_collective("remerge_student_lora")?;
        // The merge REPLACES q/v `DeviceMatrix` buffers (new device
        // addresses); captured decode graphs bake the old ones — drop and
        // recapture lazily.
        self.decode_graph = None;
        self.model.remerge_student_lora(update)
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
    use super::CudaKvCacheDtype;
    use infer_seam::KvCacheDtype;

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
