//! Real CUDA executor: the engine-facing step driver and sampling tail.
//!
//! Wraps the loaded [`CudaModel`] + device [`PagedKVPool`], validates the
//! single-row R6 plan, mirrors host→device page allocation, and runs the forward
//! that emits the next token. `sample_cuda_token` is the greedy/host-sampled
//! decision applied to the final logits. Pure relocation from `model.rs` —
//! identical numerics, with the Fix 0 sampling wiring preserved.

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

/// Sequence-length budget the captured decode graph's fixed `kv_indices` buffer is
/// sized to. Bounds the page-table length so the buffer never reallocates mid-serve
/// (design §5). Pages beyond this force eager fallback rather than a stale replay.
const DECODE_GRAPH_MAX_SEQ_LEN: usize = 32_768;

/// Whether the captured B=1 decode graph is enabled. Opt-in so the eager path
/// (the numerically-verified Phase-0 correctness floor) stays the default until
/// the lead runs the H20 eager-vs-replay parity + perf A/B. Flip with
/// `INFER_CUDA_DECODE_GRAPH=1`.
fn decode_graph_enabled() -> bool {
    matches!(
        std::env::var("INFER_CUDA_DECODE_GRAPH").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
    )
}

pub(crate) struct RealCudaExecutor {
    model: CudaModel,
    kv: PagedKVPool,
    num_slots: usize,
    /// Fixed device buffers for the B=1 captured decode path (CG-2). Lazily built
    /// at warmup; `None` until then (and on capture failure, which falls back to
    /// the eager path — capture is never load-bearing for correctness).
    decode_ctx: Option<DecodeGraphContext>,
    /// Per-shape captured decode graphs (CG-1/CG-4). Keyed by page-table length
    /// (`num_pages`) for this B=1 landing: batch is always
    /// [`DECODE_GRAPH_BATCH`], so the page count is the only shape that varies the
    /// captured TileLang `total_pages` launch scalar. A new page count recaptures.
    graphs: Option<GraphBucket>,
}

impl std::fmt::Debug for RealCudaExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealCudaExecutor")
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

impl RealCudaExecutor {
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
            // CG-3: try the captured B=1 decode graph first; on any miss (graphs
            // disabled, page count over budget, capture failure) fall back to the
            // verified eager path below. Eager is the correctness floor.
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

    /// CG-4: warmup the B=1 decode graph before serving.
    ///
    /// Builds the fixed [`DecodeGraphContext`], reserves a dummy slot, writes dummy
    /// metadata, then captures the decode graph for the smallest shape
    /// (`num_pages = 1`) so the capture machinery is proven before the first real
    /// request. Subsequent page counts capture lazily on first decode at that shape
    /// (legacy `run_or_capture` semantics — capture-once, replay-after, per key),
    /// which adapts the design's "warmup captures every bucket" to the page-count
    /// key this B=1 landing uses (see [`DecodeGraphContext`] docs).
    ///
    /// Capture is opt-in via `INFER_CUDA_DECODE_GRAPH=1` so the lead can run a
    /// matched eager-vs-replay A/B on H20 with one env flip. Any failure during
    /// warmup capture is logged and downgrades to eager-only (graphs left `None`);
    /// it is never fatal, because the eager path is the correctness floor.
    pub(crate) fn warmup(&mut self) -> Result<()> {
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

    /// Allocate the fixed buffers + capture the `num_pages = 1` decode graph using a
    /// dummy slot. Isolated so a capture failure leaves the executor eager-only.
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

    /// CG-3 fast path: write Stage-1 metadata into the fixed buffers and replay (or
    /// lazily capture-then-replay) the decode graph for this step's page count.
    ///
    /// Returns `Ok(Some(()))` when the captured graph wrote `decode_ctx.logits`
    /// (caller samples from there), or `Ok(None)` to signal eager fallback (graphs
    /// disabled, or page count exceeds the fixed buffer budget). The KV token for
    /// this step must already be allocated in `self.kv` before calling.
    fn try_captured_decode(
        &mut self,
        slot: usize,
        token: u32,
        kv_seq_len: usize,
    ) -> Result<Option<()>> {
        if self.decode_ctx.is_none() || self.graphs.is_none() {
            return Ok(None);
        }
        // Page count over the fixed buffer budget → eager (never a stale replay).
        let total_len = kv_seq_len + 1;
        let num_pages = total_len.div_ceil(self.kv.page_size);
        if num_pages > DECODE_GRAPH_MAX_SEQ_LEN.div_ceil(self.kv.page_size) {
            return Ok(None);
        }
        self.capture_decode_for_current_state(slot, token, kv_seq_len)?;
        Ok(Some(()))
    }

    /// Stage-1 write + `run_or_capture` for the current decode state.
    ///
    /// Captures the graph the first time a `num_pages` shape is seen and replays it
    /// on every later call at that shape. The page-table length is the capture key
    /// (B=1, so batch is fixed); a new page count gets its own graph entry.
    fn capture_decode_for_current_state(
        &mut self,
        slot: usize,
        token: u32,
        kv_seq_len: usize,
    ) -> Result<()> {
        // Split borrows: stage1 needs &self.kv + &mut decode_ctx; replay needs
        // &self.model + &mut self.kv + &mut decode_ctx + &mut graphs.
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
    ///
    /// Sampling stays OUTSIDE the graph (design §7): the replay ends at
    /// `decode_ctx.logits`; greedy argmax / host sampling runs here, eager.
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

pub(crate) fn sample_cuda_token(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    params: &SamplingParams,
    position: u64,
) -> Result<u32> {
    if params.is_greedy() {
        return argmax(ctx, logits);
    }

    // TODO(Fix 0 follow-up): repetition/frequency/presence penalties need the
    // per-request generated-token history threaded through the executor.
    let logits_host = logits.to_host(ctx)?;
    Ok(infer_plan::sample_token(&logits_host, params, position))
}
