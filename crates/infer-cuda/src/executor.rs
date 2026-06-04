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

/// Captured B=1 decode graph enabled? Opt-in via `INFER_CUDA_DECODE_GRAPH=1`;
/// the eager path (correctness floor) is the default.
fn decode_graph_enabled() -> bool {
    matches!(
        std::env::var("INFER_CUDA_DECODE_GRAPH").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
    )
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
    num_slots: usize,
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
        Ok(Self {
            model,
            slots,
            num_slots,
        })
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
    if params.is_greedy() {
        return argmax(ctx, logits);
    }

    // TODO: repetition/frequency/presence penalties need the per-request
    // generated-token history threaded through the executor.
    let logits_host = logits.to_host(ctx)?;
    Ok(infer_plan::sample_token(&logits_host, params, position))
}
