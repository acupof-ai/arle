//! Real CUDA executor: the engine-facing step driver and sampling tail.
//!
//! Wraps the loaded [`CudaModel`] + device [`PagedKVPool`], validates the
//! single-row plan, mirrors host→device page allocation, runs the forward, and
//! samples the next token (`sample_cuda_token`: greedy argmax / host sampling).

use std::path::Path;

use anyhow::{Result, ensure};
use cuda_kernels::KVFormat;
use cuda_kernels::prelude::{DeviceContext, DeviceVec, PagedKVPool};
use infer_plan::{ForwardPlan, SamplingParams, SlotToken, StepOutput};
use infer_seam::KvPool;
use log::{info, warn};

use crate::decode_graph::DecodeGraphContext;
use crate::decode_graph_key::{DECODE_GRAPH_BATCH, DecodeGraphKey};
use crate::graph::GraphBucket;
use crate::model::CudaModel;
use crate::ops::argmax;

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
    ) -> Result<Self> {
        Ok(Self::Qwen(Box::new(
            QwenCudaExecutor::from_qwen3_bf16_safetensors(model_path, num_slots, total_pages)?,
        )))
    }

    pub(crate) fn from_qwen35_moe_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
        total_pages: usize,
    ) -> Result<Self> {
        Ok(Self::Qwen35(Box::new(
            Qwen35CudaExecutor::from_qwen35_moe_safetensors(model_path, num_slots, total_pages)?,
        )))
    }

    /// Build the DSv4-Flash executor (MLA + HC + FP8 MoE, multi-GPU TP/EP).
    pub(crate) fn from_dsv4_fp8_safetensors(
        model_path: impl AsRef<Path>,
        num_slots: usize,
    ) -> Result<Self> {
        Ok(Self::Dsv4(Box::new(
            Dsv4CudaExecutor::from_dsv4_fp8_safetensors(model_path, num_slots)?,
        )))
    }

    pub(crate) fn submit(
        &mut self,
        plan: &ForwardPlan,
        host_kv: &mut dyn KvPool,
    ) -> Result<StepOutput> {
        match self {
            Self::Qwen(q) => q.submit(plan, host_kv),
            Self::Qwen35(q) => q.submit(plan),
            Self::Dsv4(d) => d.submit(plan),
        }
    }

    pub(crate) fn warmup(&mut self) -> Result<()> {
        match self {
            Self::Qwen(q) => q.warmup(),
            // Qwen3.5 hybrid / DSv4 have no captured decode graph (MoE host-routing
            // + recurrent/recompute state are not graph-capturable).
            Self::Qwen35(_) | Self::Dsv4(_) => Ok(()),
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

pub(crate) struct QwenCudaExecutor {
    model: CudaModel,
    kv: PagedKVPool,
    num_slots: usize,
    /// Fixed device buffers for the B=1 captured decode path. Built lazily at
    /// warmup; `None` until then / on capture failure (capture is never
    /// load-bearing for correctness — eager is the floor).
    decode_ctx: Option<DecodeGraphContext>,
    /// Per-shape captured decode graphs, keyed by page-table length: batch is
    /// fixed at [`DECODE_GRAPH_BATCH`], so `num_pages` is the only varying capture
    /// scalar. A new page count recaptures.
    graphs: Option<GraphBucket>,
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
    ) -> Result<Self> {
        ensure!(num_slots > 0, "CudaExecutor requires at least one slot");
        ensure!(
            total_pages > 0,
            "CudaExecutor requires at least one KV page"
        );

        let model = CudaModel::from_safetensors(model_path.as_ref())?;
        let token_budget = total_pages * SUPPORTED_PAGE_SIZE;
        let budget_bytes = PagedKVPool::budget_bytes_for_tokens(
            model.config.num_hidden_layers,
            model.config.num_key_value_heads,
            model.config.head_dim,
            token_budget,
            KVFormat::BF16,
        );
        let kv = PagedKVPool::with_format(
            &model.ctx,
            model.config.num_hidden_layers,
            model.config.num_key_value_heads,
            model.config.head_dim,
            num_slots,
            budget_bytes,
            KVFormat::BF16,
        )?;
        ensure!(
            kv.page_size == SUPPORTED_PAGE_SIZE,
            "R6 BF16 Qwen3 expects cuda-kernels page_size={SUPPORTED_PAGE_SIZE}, got {}",
            kv.page_size
        );

        Ok(Self {
            model,
            kv,
            num_slots,
            decode_ctx: None,
            graphs: None,
        })
    }

    pub(crate) fn submit(
        &mut self,
        plan: &ForwardPlan,
        host_kv: &mut dyn KvPool,
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
                host_kv.seq_len(row.slot) >= expected_len,
                "host KV length {} is behind prefill materialization end {} for slot {}",
                host_kv.seq_len(row.slot),
                expected_len,
                row.slot
            );
            self.ensure_slot_ready_for_prefill(row.slot, row.start_pos)?;
            self.kv.alloc_tokens(row.slot, row.tokens.len())?;
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
                self.kv.seq_len(row.slot) == row.kv_seq_len,
                "CUDA materialized cache_len {} != DecodeRow.kv_seq_len {} for slot {}",
                self.kv.seq_len(row.slot),
                row.kv_seq_len,
                row.slot
            );
            ensure!(
                host_kv.seq_len(row.slot) > row.kv_seq_len,
                "host KV length {} is behind decode materialization end {} for slot {}",
                host_kv.seq_len(row.slot),
                row.kv_seq_len + 1,
                row.slot
            );
            self.kv.alloc_tokens(row.slot, 1)?;
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

        // Reserve dummy slot 0 with a single token so the page table is valid for
        // num_pages = 1, capture, then release it so serving starts clean.
        ensure!(self.num_slots > 0, "warmup needs at least one slot");
        let dummy_slot = 0usize;
        self.kv.free_slot(dummy_slot);
        self.kv.alloc_tokens(dummy_slot, 1)?;
        let capture_result = self.capture_decode_for_current_state(dummy_slot, 0, 0);
        self.kv.free_slot(dummy_slot);
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

    fn ensure_slot_ready_for_prefill(&mut self, slot: usize, start_pos: usize) -> Result<()> {
        let materialized = self.kv.seq_len(slot);
        if start_pos == 0 {
            if materialized != 0 {
                self.kv.free_slot(slot);
            }
            return Ok(());
        }
        ensure!(
            materialized == start_pos,
            "chunked prefill requires materialized CUDA cache_len == start_pos; got cache_len={materialized}, start_pos={start_pos}"
        );
        Ok(())
    }
}

/// DSv4-Flash executor: drives [`crate::dsv4::Dsv4Model::forward_tokens`] over a
/// single scheduled row. DSv4 owns its MLA KV state inside the forward (bf16 SW
/// rings + compressor pending/compressed pools), so it does NOT use a
/// [`PagedKVPool`]. The decode graph is disabled (MLA host-routing per step).
pub(crate) struct Dsv4CudaExecutor {
    model: crate::dsv4::Dsv4Model,
    slots: Vec<crate::dsv4::Dsv4SlotState>,
    spec_slots: Vec<Dsv4SpecSlotState>,
    num_slots: usize,
    mtp_accepts: usize,
    mtp_rejects: usize,
}

#[derive(Default)]
struct Dsv4SpecSlotState {
    pending: Option<u32>,
    hidden: Option<DeviceVec>,
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
    ) -> Result<Self> {
        ensure!(num_slots > 0, "Dsv4CudaExecutor requires at least one slot");
        let model = crate::dsv4::Dsv4Model::from_dsv4_fp8_safetensors(model_path.as_ref())?;
        let max_seq_len = dsv4_max_seq_len();
        let mut slots = Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            slots.push(model.new_slot_state(max_seq_len)?);
        }
        let spec_slots = (0..num_slots)
            .map(|_| Dsv4SpecSlotState::default())
            .collect();
        Ok(Self {
            model,
            slots,
            spec_slots,
            num_slots,
            mtp_accepts: 0,
            mtp_rejects: 0,
        })
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
        if crate::dsv4::dsv4_spec_decode_enabled() {
            let (token, hidden) = self.model.forward_tokens_with_hidden(
                &mut self.slots[slot_idx],
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
        if !crate::dsv4::dsv4_spec_decode_enabled() {
            let token = self.model.forward_tokens(
                &mut self.slots[slot_idx],
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
        let spec = &mut self.spec_slots[slot_idx];
        let pending = spec
            .pending
            .ok_or_else(|| anyhow::anyhow!("DSv4 MTP decode missing pending token"))?;
        ensure!(
            pending == last_token,
            "DSv4 MTP pending token {pending} != DecodeRow.last_token {last_token}"
        );
        let hidden = spec
            .hidden
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DSv4 MTP decode missing previous hidden"))?
            .clone();

        let draft_position = start_pos.saturating_add(1) as u64;
        let draft = self.model.mtp_forward(&hidden, pending, draft_position)?;
        let (argmax, mut hiddens) = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &[pending, draft],
            start_pos,
            position,
        )?;
        ensure!(
            argmax.len() == 2 && hiddens.len() == 2,
            "DSv4 MTP depth-1 verify expected 2 rows, got argmax={} hidden={}",
            argmax.len(),
            hiddens.len()
        );
        let base_next = argmax[0];
        let bonus = argmax[1];
        let matched = base_next == draft;
        if std::env::var("ARLE_DSV4_MTP_DRAFT_DUMP").as_deref() == Ok("1")
            && self.model.tp.config().rank == 0
        {
            let total_before = self.mtp_accepts + self.mtp_rejects;
            let matches_before = self.mtp_accepts;
            let total_after = total_before + 1;
            let matches_after = matches_before + usize::from(matched);
            let accuracy = (matches_after as f64) / (total_after as f64);
            eprintln!(
                "[dsv4-mtp-draft] step={} start_pos={} pending={} draft={} actual={} match={} depth1_match_total={} depth1_total={} depth1_accuracy={:.6}",
                total_after,
                start_pos,
                pending,
                draft,
                base_next,
                matched,
                matches_after,
                total_after,
                accuracy
            );
        }
        if base_next == draft {
            let hidden_for_draft = hiddens.remove(1);
            spec.pending = Some(bonus);
            spec.hidden = Some(hidden_for_draft);
            self.mtp_accepts += 1;
            if self.model.tp.config().rank == 0 {
                eprintln!(
                    "[dsv4-mtp] accept_total={} reject_total={} pending={} draft={} bonus={}",
                    self.mtp_accepts, self.mtp_rejects, pending, draft, bonus
                );
            }
            Ok(vec![draft, bonus])
        } else {
            let keep_len = start_pos + 1;
            self.model
                .truncate_slot(&mut self.slots[slot_idx], keep_len)?;
            self.model.dump_mtp_rollback_state(
                &self.slots[slot_idx],
                "spec_after_reject_truncate",
                keep_len,
            )?;
            self.slots[slot_idx].restore_spec_rollback(&self.model.ctx, keep_len)?;
            self.model.dump_mtp_rollback_state(
                &self.slots[slot_idx],
                "spec_after_reject_restore",
                keep_len,
            )?;
            let hidden_for_pending = hiddens.remove(0);
            spec.pending = Some(base_next);
            spec.hidden = Some(hidden_for_pending);
            self.mtp_rejects += 1;
            if self.model.tp.config().rank == 0 {
                eprintln!(
                    "[dsv4-mtp] accept_total={} reject_total={} pending={} draft={} base_next={}",
                    self.mtp_accepts, self.mtp_rejects, pending, draft, base_next
                );
            }
            Ok(vec![base_next])
        }
    }

    fn submit(&mut self, plan: &ForwardPlan) -> Result<StepOutput> {
        let rows = plan.decode_rows.len() + plan.prefill_rows.len();
        if rows == 0 {
            return Ok(StepOutput { tokens: Vec::new() });
        }
        ensure!(
            rows == 1,
            "DSv4 CUDA forward is single-row only, got {} prefill + {} decode rows",
            plan.prefill_rows.len(),
            plan.decode_rows.len()
        );

        let (slot, token) = if let Some(row) = plan.prefill_rows.first() {
            ensure!(
                row.slot < self.num_slots,
                "prefill slot {} outside DSv4 executor slots {}",
                row.slot,
                self.num_slots
            );
            ensure!(!row.tokens.is_empty(), "prefill row must carry tokens");
            if row.start_pos == 0 {
                self.slots[row.slot].reset(&self.model.ctx)?;
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
            (row.slot, tokens)
        } else {
            let row = &plan.decode_rows[0];
            ensure!(
                row.slot < self.num_slots,
                "decode slot {} outside DSv4 executor slots {}",
                row.slot,
                self.num_slots
            );
            ensure!(
                self.slots[row.slot].seq_len() == row.kv_seq_len,
                "DSv4 materialized state len {} != DecodeRow.kv_seq_len {} for slot {}",
                self.slots[row.slot].seq_len(),
                row.kv_seq_len,
                row.slot
            );
            let position = row.kv_seq_len.saturating_add(1) as u64;
            let tokens = self.forward_decode_tokens(
                row.slot,
                row.last_token,
                row.kv_seq_len,
                &row.params,
                position,
            )?;
            (row.slot, tokens)
        };

        Ok(StepOutput {
            tokens: token
                .into_iter()
                .map(|token| SlotToken {
                    slot,
                    token,
                    logprob: None,
                    finish: None,
                })
                .collect(),
        })
    }

    pub(crate) fn verify_forward_selftest(&mut self, prompt: &[u32]) -> Result<()> {
        ensure!(
            !prompt.is_empty(),
            "DSv4 verify-forward selftest requires a non-empty prompt"
        );
        let slot_idx = 0;
        let params = SamplingParams::default();
        let start_pos = prompt.len();

        self.slots[slot_idx].reset(&self.model.ctx)?;
        let token_a = self.model.forward_tokens(
            &mut self.slots[slot_idx],
            prompt,
            0,
            &params,
            start_pos as u64,
        )?;
        let (verify_one, _) = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &[token_a],
            start_pos,
            (start_pos + 1) as u64,
        )?;

        self.slots[slot_idx].reset(&self.model.ctx)?;
        let token_a_again = self.model.forward_tokens(
            &mut self.slots[slot_idx],
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
            &[token_a],
            start_pos,
            &params,
            (start_pos + 1) as u64,
        )?;
        ensure!(
            verify_one.first().copied() == Some(normal_one),
            "DSv4 verify selftest one-token mismatch: verify={verify_one:?} normal={normal_one}"
        );

        self.slots[slot_idx].reset(&self.model.ctx)?;
        let token_a = self.model.forward_tokens(
            &mut self.slots[slot_idx],
            prompt,
            0,
            &params,
            start_pos as u64,
        )?;
        let (verify_one, _) = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &[token_a],
            start_pos,
            (start_pos + 1) as u64,
        )?;
        let token_b = verify_one[0];
        let mut wrong_b = token_b.wrapping_add(2);
        if wrong_b == token_b {
            wrong_b = token_b.wrapping_add(3);
        }

        self.slots[slot_idx].reset(&self.model.ctx)?;
        let token_a = self.model.forward_tokens(
            &mut self.slots[slot_idx],
            prompt,
            0,
            &params,
            start_pos as u64,
        )?;
        let (verify_two, _) = self.model.forward_tokens_verify(
            &mut self.slots[slot_idx],
            &[token_a, wrong_b],
            start_pos,
            (start_pos + 1) as u64,
        )?;
        ensure!(
            verify_two.first() == verify_one.first(),
            "DSv4 verify selftest two-token row0 mismatch: one={verify_one:?} two={verify_two:?}"
        );

        self.slots[slot_idx].reset(&self.model.ctx)?;
        self.spec_slots[slot_idx] = Dsv4SpecSlotState::default();
        if self.model.tp.config().rank == 0 {
            eprintln!(
                "[dsv4-mtp-selftest] PASS token_a={token_a} token_b={token_b} wrong_b={wrong_b} verify_two={verify_two:?}"
            );
        }
        Ok(())
    }
}

fn dsv4_max_seq_len() -> usize {
    std::env::var("INFER_DSV4_MAX_SEQ_LEN")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DSV4_DEFAULT_MAX_SEQ_LEN)
}

/// Qwen3.5 / Qwen3.6 HYBRID executor: drives
/// [`crate::qwen35::Qwen35Model::forward_tokens`] over a single scheduled row.
/// Owns per-slot KV state inside the model (full-attn contiguous caches +
/// gated-delta recurrent state), so it does NOT use a [`PagedKVPool`]; it relies
/// on the host [`KvPool`] only for the slot's logical `seq_len` to derive
/// `start_pos`. Decode graph disabled (MoE host-routing + recurrent state).
///
/// First-runnable scope: single-row prefill/decode, uncached full-prefix (each
/// full-attn layer recomputes over its contiguous cache; each linear-attn layer
/// advances the recurrent state in place). A continuous-batching paged +
/// packed-batch path is the perf follow-up (legacy `infer/src/model/qwen35`).
pub(crate) struct Qwen35CudaExecutor {
    model: crate::qwen35::Qwen35Model,
    /// Per-slot KV + recurrent state (one [`crate::qwen35::Qwen35SlotState`] per slot).
    slots: Vec<crate::qwen35::Qwen35SlotState>,
    num_slots: usize,
}

impl std::fmt::Debug for Qwen35CudaExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen35CudaExecutor")
            .field("model", &self.model)
            .field("num_slots", &self.num_slots)
            .finish()
    }
}

impl Qwen35CudaExecutor {
    pub(crate) fn from_qwen35_moe_safetensors(
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
        let model = crate::qwen35::Qwen35Model::from_qwen35_moe_safetensors(
            model_path.as_ref(),
            max_seq_len,
        )?;
        let mut slots = Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            slots.push(model.new_slot_state()?);
        }
        Ok(Self {
            model,
            slots,
            num_slots,
        })
    }

    /// Offload the model's device weights to host RAM (OPD teacher time-share),
    /// returning the device bytes freed. Per-slot KV / recurrent state is left
    /// resident — only the shared model weights move.
    fn offload_engine_weights(&mut self) -> Result<usize> {
        self.model.offload_engine_weights()
    }

    /// Reload the model's device weights from the host snapshot.
    fn reload_engine_weights(&mut self) -> Result<()> {
        self.model.reload_engine_weights()
    }

    fn submit(&mut self, plan: &ForwardPlan) -> Result<StepOutput> {
        let rows = plan.decode_rows.len() + plan.prefill_rows.len();
        if rows == 0 {
            return Ok(StepOutput { tokens: Vec::new() });
        }
        ensure!(
            rows == 1,
            "Qwen3.5 hybrid CUDA forward is single-row only, got {} prefill + {} decode rows",
            plan.prefill_rows.len(),
            plan.decode_rows.len()
        );

        let (slot, token) = if let Some(row) = plan.prefill_rows.first() {
            ensure!(
                row.slot < self.num_slots,
                "prefill slot {} outside Qwen3.5 executor slots {}",
                row.slot,
                self.num_slots
            );
            ensure!(!row.tokens.is_empty(), "prefill row must carry tokens");
            // A fresh prefill (start_pos == 0) rewinds this slot's recurrent +
            // conv state and cache cursor before appending.
            if row.start_pos == 0 {
                self.slots[row.slot].reset(&self.model.ctx)?;
            }
            let position = (row.start_pos + row.tokens.len()) as u64;
            let token = self.model.forward_tokens(
                &mut self.slots[row.slot],
                &row.tokens,
                row.start_pos,
                &row.params,
                position,
            )?;
            (row.slot, token)
        } else {
            let row = &plan.decode_rows[0];
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
            let token = self.model.forward_tokens(
                &mut self.slots[row.slot],
                &[row.last_token],
                row.kv_seq_len,
                &row.params,
                position,
            )?;
            (row.slot, token)
        };

        Ok(StepOutput {
            tokens: vec![SlotToken {
                slot,
                token,
                logprob: None,
                finish: None,
            }],
        })
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
        let mut slot = self.model.new_slot_state()?;
        self.model
            .forward_token_logits_full(&mut slot, input_ids, start_pos)
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
