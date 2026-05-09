use anyhow::{Context, Result};
use cudarc::driver::{CudaEvent, CudaSlice};
use std::cell::RefCell;

use super::forward::Qwen3State;
use super::weights::{Qwen3Model, TransformerBlock};
use crate::model::cuda_graph::CudaGraphState;
use crate::model::kv_cache::{KVCache, KVFormat};
use crate::ops::{self, OpsBackend};
use cuda_kernels::TokenKVPool;
use cuda_kernels::kv_quant;
use cuda_kernels::prelude::{DeviceContext, DeviceVec, HiddenStates};

/// Pre-allocated scratch buffers for one prefill forward pass.
/// Created once per prefill in `process_all_layers_batch`, eliminating
/// per-layer `cuMemAllocAsync` overhead (~11k calls / 88ms at seq=2048).
///
/// Buffer reuse across steps (all kernels serialized on a single stream):
///   `normed`  reused for `normed2`  (steps 1-4 done before step 8)
///   `o_buf`   reused for `mlp_out`  (step 7 done before step 12)
struct PrefillBuffers {
    /// Output ping-pong: layer writes result here; caller swaps with the incoming hidden.
    hidden_out: HiddenStates, // hidden_dim × seq_len
    /// fp32 shadow of the residual stream. Maintained across layers so that
    /// per-layer bf16 outputs accumulate into fp32 without compounding
    /// ~0.4% bf16 rounding noise at each residual add. Norm reads from here
    /// directly to avoid a further bf16 round-trip on the hidden state.
    /// `None` unless `INFER_QWEN3_FP32_RESIDUAL=1` is set.
    residual_f32: Option<CudaSlice<f32>>,
    normed: HiddenStates,      // hidden_dim × seq_len (reused for normed2)
    q_batch: HiddenStates,     // q_dim × seq_len
    k_batch: HiddenStates,     // kv_dim × seq_len
    v_batch: HiddenStates,     // kv_dim × seq_len
    o_buf: HiddenStates,       // hidden_dim × seq_len (reused for mlp_out)
    gate_out: HiddenStates,    // inter_dim × seq_len, or fused 2×inter_dim × seq_len
    up_out: HiddenStates,      // inter_dim × seq_len; unused when fused gate/up is active
    act_out: HiddenStates,     // inter_dim × seq_len
    attn_output: HiddenStates, // q_dim × seq_len
    last_hidden: DeviceVec,    // hidden_dim
    last_normed: DeviceVec,    // hidden_dim
    marlin_prefill_scratch: Option<RefCell<ops::MarlinPrefillScratch>>,
}

impl PrefillBuffers {
    fn new(
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        seq_len: usize,
        fused_gate_up: bool,
        marlin_scratch_config: Option<ops::MarlinPrefillScratchConfig>,
    ) -> Result<Self> {
        let residual_f32 = if std::env::var("INFER_QWEN3_FP32_RESIDUAL").is_ok() {
            Some(
                ctx.stream
                    .alloc_zeros::<f32>(hidden_dim * seq_len)
                    .map_err(|e| anyhow::anyhow!("alloc residual_f32: {e}"))?,
            )
        } else {
            None
        };
        let gate_out_dim = if fused_gate_up {
            inter_dim * 2
        } else {
            inter_dim
        };
        let max_k = hidden_dim.max(q_dim).max(inter_dim);
        let max_n = hidden_dim.max(q_dim).max(kv_dim).max(gate_out_dim);
        let marlin_prefill_scratch = match marlin_scratch_config {
            Some(config) if config.any() => Some(RefCell::new(ops::MarlinPrefillScratch::new(
                ctx, seq_len, max_k, max_n, config,
            )?)),
            _ => None,
        };
        Ok(Self {
            hidden_out: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            residual_f32,
            normed: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, seq_len)?,
            k_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            v_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            o_buf: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            gate_out: HiddenStates::zeros(ctx, gate_out_dim, seq_len)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, seq_len)?,
            last_hidden: DeviceVec::zeros(ctx, hidden_dim)?,
            last_normed: DeviceVec::zeros(ctx, hidden_dim)?,
            marlin_prefill_scratch,
        })
    }
}

pub(super) struct Qwen3PagedPrefillRequest<'a> {
    pub tokens: &'a [u32],
    pub slot: usize,
    pub state_idx: usize,
}

pub(super) struct Qwen3PagedPrefillLayout {
    sequences: Vec<ops::PagedPrefillSequence>,
    page_indices: Vec<i32>,
    prefix_token_rows: Vec<i32>,
    prefill_token_rows: Vec<i32>,
}

struct OwnedPendingPagedPrefillBuffers {
    _hidden: HiddenStates,
    _bufs: PrefillBuffers,
    _page_indices_dev: CudaSlice<i32>,
    _page_table_offsets_dev: CudaSlice<i32>,
    _start_positions_dev: CudaSlice<i32>,
    _prefix_token_rows_dev: CudaSlice<i32>,
    _prefill_token_rows_dev: CudaSlice<i32>,
    _fwd: crate::ops::PagedPrefillForward,
}

struct PendingPagedPrefill {
    _owned: Option<OwnedPendingPagedPrefillBuffers>,
    owners: Vec<PendingPagedPrefillOwner>,
    completion_error: Option<anyhow::Error>,
}

#[derive(Clone, Debug)]
struct PendingPagedPrefillOwner {
    slot: usize,
    state_idx: usize,
    token_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Qwen3PrefillGraphKey {
    total_tokens: usize,
    page_size: usize,
    /// Bucketed allocation capacity, not the exact request page count.
    page_indices_len: usize,
    /// Bucketed allocation capacity, not the exact prefix row count.
    prefix_token_rows_len: usize,
    batch_size: usize,
    seq_lens: Vec<usize>,
}

impl Qwen3PrefillGraphKey {
    fn new(layout: &Qwen3PagedPrefillLayout, page_size: usize) -> Self {
        Self {
            total_tokens: layout.prefill_token_rows.len(),
            page_size,
            page_indices_len: round_nonzero_up(
                layout.page_indices.len(),
                QWEN3_PREFILL_GRAPH_PAGE_INDICES_BUCKET,
            ),
            prefix_token_rows_len: round_nonzero_up(
                layout.prefix_token_rows.len(),
                QWEN3_PREFILL_GRAPH_PREFIX_ROWS_BUCKET,
            ),
            batch_size: layout.sequences.len(),
            seq_lens: layout.sequences.iter().map(|seq| seq.seq_len).collect(),
        }
    }
}

struct Qwen3PrefillGraphMetadata {
    page_table_offsets_dev: CudaSlice<i32>,
    start_positions_dev: CudaSlice<i32>,
    seq_lens_dev: CudaSlice<i32>,
}

impl Qwen3PrefillGraphMetadata {
    fn new(ctx: &DeviceContext, batch_size: usize) -> Result<Self> {
        let capacity = batch_size.max(1);
        Ok(Self {
            page_table_offsets_dev: ctx
                .stream
                .alloc_zeros(capacity)
                .map_err(|e| anyhow::anyhow!("alloc prefill graph page-table offsets: {e}"))?,
            start_positions_dev: ctx
                .stream
                .alloc_zeros(capacity)
                .map_err(|e| anyhow::anyhow!("alloc prefill graph start positions: {e}"))?,
            seq_lens_dev: ctx
                .stream
                .alloc_zeros(capacity)
                .map_err(|e| anyhow::anyhow!("alloc prefill graph seq lens: {e}"))?,
        })
    }

    fn refresh(&mut self, ctx: &DeviceContext, layout: &Qwen3PagedPrefillLayout) -> Result<()> {
        let start_positions: Vec<i32> = layout
            .sequences
            .iter()
            .map(|seq| seq.start_pos as i32)
            .collect();
        let page_table_offsets: Vec<i32> = layout
            .sequences
            .iter()
            .map(|seq| seq.page_table_offset as i32)
            .collect();
        let seq_lens: Vec<i32> = layout
            .sequences
            .iter()
            .map(|seq| seq.seq_len as i32)
            .collect();
        let mut start_positions_dev = self.start_positions_dev.slice_mut(..start_positions.len());
        ctx.stream
            .memcpy_htod(&start_positions, &mut start_positions_dev)
            .map_err(|e| anyhow::anyhow!("prefill graph start positions H2D failed: {e}"))?;
        let mut page_table_offsets_dev = self
            .page_table_offsets_dev
            .slice_mut(..page_table_offsets.len());
        ctx.stream
            .memcpy_htod(&page_table_offsets, &mut page_table_offsets_dev)
            .map_err(|e| anyhow::anyhow!("prefill graph page-table offsets H2D failed: {e}"))?;
        let mut seq_lens_dev = self.seq_lens_dev.slice_mut(..seq_lens.len());
        ctx.stream
            .memcpy_htod(&seq_lens, &mut seq_lens_dev)
            .map_err(|e| anyhow::anyhow!("prefill graph seq lens H2D failed: {e}"))?;
        Ok(())
    }
}

struct Qwen3PagedPrefillGraphResources {
    key: Qwen3PrefillGraphKey,
    token_ids_gpu: CudaSlice<i32>,
    hidden: HiddenStates,
    bufs: PrefillBuffers,
    page_indices_dev: CudaSlice<i32>,
    metadata: Qwen3PrefillGraphMetadata,
    prefix_token_rows_dev: CudaSlice<i32>,
    prefill_token_rows_dev: CudaSlice<i32>,
    fwd: crate::ops::PagedPrefillForward,
    graph_state: CudaGraphState,
}

pub struct Qwen3PrefillContext {
    pending: Option<PendingPagedPrefill>,
    completion_event: CudaEvent,
    pending_ready_without_event: bool,
    graph_resources: Vec<Qwen3PagedPrefillGraphResources>,
}

impl Qwen3PrefillContext {
    pub(super) fn new(ctx: &DeviceContext) -> Result<Self> {
        Ok(Self {
            pending: None,
            completion_event: ctx
                .ctx
                .new_event(None)
                .map_err(|e| anyhow::anyhow!("Alloc async prefill completion event failed: {e}"))?,
            pending_ready_without_event: false,
            graph_resources: Vec::new(),
        })
    }

    fn set_pending(&mut self, ctx: &DeviceContext, pending: PendingPagedPrefill) -> Result<()> {
        anyhow::ensure!(
            self.pending.is_none(),
            "qwen3 prefill context already has a pending batch"
        );
        self.completion_event
            .record(&ctx.stream)
            .map_err(|e| anyhow::anyhow!("record async prefill completion event: {e}"))?;
        self.pending = Some(pending);
        Ok(())
    }

    fn set_completed_pending(&mut self, pending: PendingPagedPrefill) -> Result<()> {
        anyhow::ensure!(
            self.pending.is_none(),
            "qwen3 prefill context already has a pending batch"
        );
        self.pending = Some(pending);
        self.pending_ready_without_event = true;
        Ok(())
    }

    pub(super) fn complete(&mut self, expected_slots: &[usize]) -> Result<bool> {
        let Some(pending) = self.pending.as_ref() else {
            return Ok(true);
        };
        pending.validate_slots(expected_slots)?;
        if !self.pending_ready_without_event {
            match unsafe { cudarc::driver::result::event::query(self.completion_event.cu_event()) }
            {
                Ok(()) => {}
                Err(err) if err.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY => {
                    return Ok(false);
                }
                Err(err) => {
                    self.pending = None;
                    return Err(anyhow::anyhow!(
                        "async prefill completion event failed: {err}"
                    ));
                }
            }
        }
        self.pending_ready_without_event = false;
        let pending = self
            .pending
            .take()
            .expect("pending prefill existed before completion event");
        if let Some(err) = pending.completion_error {
            return Err(err);
        }
        Ok(true)
    }

    pub(super) fn wait(&mut self, expected_slots: &[usize]) -> Result<()> {
        while !self.complete(expected_slots)? {
            std::thread::yield_now();
        }
        Ok(())
    }
}

impl PendingPagedPrefill {
    fn validate_slots(&self, expected_slots: &[usize]) -> Result<()> {
        if expected_slots.is_empty() {
            return Ok(());
        }
        anyhow::ensure!(
            self.owners.len() == expected_slots.len(),
            "async prefill owner count mismatch: pending={} expected={}",
            self.owners.len(),
            expected_slots.len()
        );
        for (row, (owner, &expected_slot)) in self.owners.iter().zip(expected_slots).enumerate() {
            anyhow::ensure!(
                owner.slot == expected_slot,
                "async prefill owner mismatch at row {row}: pending slot {} expected slot {}",
                owner.slot,
                expected_slot
            );
            anyhow::ensure!(
                owner.token_count > 0,
                "async prefill owner for slot {} has empty chunk",
                owner.slot
            );
            anyhow::ensure!(
                owner.state_idx == owner.slot,
                "async scheduler prefill owner slot/state mismatch: slot {} state {}",
                owner.slot,
                owner.state_idx
            );
        }
        Ok(())
    }
}

fn prefill_owners(requests: &[Qwen3PagedPrefillRequest<'_>]) -> Vec<PendingPagedPrefillOwner> {
    requests
        .iter()
        .map(|request| PendingPagedPrefillOwner {
            slot: request.slot,
            state_idx: request.state_idx,
            token_count: request.tokens.len(),
        })
        .collect()
}

pub(super) fn qwen3_prefill_graph_requested() -> bool {
    matches!(
        std::env::var("INFER_PREFILL_GRAPH").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on" | "ON")
    )
}

const QWEN3_PREFILL_GRAPH_CACHE_MAX_KEYS: usize = 8;
const QWEN3_PREFILL_GRAPH_PAGE_INDICES_BUCKET: usize = 64;
const QWEN3_PREFILL_GRAPH_PREFIX_ROWS_BUCKET: usize = 128;

fn round_nonzero_up(value: usize, bucket: usize) -> usize {
    debug_assert!(bucket > 0);
    if value == 0 {
        0
    } else {
        value.div_ceil(bucket) * bucket
    }
}

impl Qwen3Model {
    #[fastrace::trace(name = "get_embeddings_batch")]
    pub(super) fn get_embeddings_batch(&self, token_ids: &[u32]) -> Result<HiddenStates> {
        crate::model::common::get_embeddings_batch(
            &self.ctx,
            &self.embed_tokens,
            token_ids,
            self.config.hidden_size,
        )
    }

    #[fastrace::trace(name = "process_all_layers_batch")]
    pub(super) fn process_all_layers_batch(
        &self,
        mut hidden: HiddenStates,
        start_pos: usize,
        kv_cache: &mut KVCache,
    ) -> Result<HiddenStates> {
        let seq_len = hidden.seq_len;
        let mut bufs = self.prefill_buffers(seq_len)?;

        // If fp32 residual shadow is enabled, seed it from the bf16 embedding.
        if let Some(ref mut r) = bufs.residual_f32 {
            ops::cast_bf16_to_f32(&self.ctx, &hidden, r)?;
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_layer_batch(
                layer_idx,
                layer,
                &mut hidden,
                start_pos,
                kv_cache,
                &mut bufs,
            )?;
        }

        // If fp32 residual shadow was active, convert back to bf16 for the
        // final norm + LM head which still consume bf16.
        if let Some(ref r) = bufs.residual_f32 {
            ops::cast_f32_to_bf16(&self.ctx, r, &mut hidden)?;
        }

        // Increment sequence length AFTER all layers processed
        for _ in 0..seq_len {
            kv_cache.increment_seq_len();
        }

        Ok(hidden)
    }

    fn prefill_buffers(&self, seq_len: usize) -> Result<PrefillBuffers> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        PrefillBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            q_dim,
            kv_dim,
            self.config.intermediate_size,
            seq_len,
            self.uses_fused_gate_up(),
            None,
        )
    }

    fn prefill_buffers_with_marlin_scratch(&self, seq_len: usize) -> Result<PrefillBuffers> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        PrefillBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            q_dim,
            kv_dim,
            self.config.intermediate_size,
            seq_len,
            self.uses_fused_gate_up(),
            Some(self.marlin_prefill_scratch_config()),
        )
    }

    fn ensure_paged_prefill_graph_resources<'a>(
        &self,
        prefill_ctx: &'a mut Qwen3PrefillContext,
        key: Qwen3PrefillGraphKey,
        layout: &Qwen3PagedPrefillLayout,
    ) -> Result<&'a mut Qwen3PagedPrefillGraphResources> {
        if let Some(pos) = prefill_ctx
            .graph_resources
            .iter()
            .position(|resources| resources.key == key)
        {
            if pos + 1 != prefill_ctx.graph_resources.len() {
                let resources = prefill_ctx.graph_resources.remove(pos);
                prefill_ctx.graph_resources.push(resources);
            }
            return prefill_ctx
                .graph_resources
                .last_mut()
                .context("prefill graph resources must exist after cache hit");
        }

        if prefill_ctx.graph_resources.len() >= QWEN3_PREFILL_GRAPH_CACHE_MAX_KEYS {
            let evicted = prefill_ctx.graph_resources.remove(0);
            log::warn!(
                "Qwen3 prefill graph cache evicting key: tokens={} batch={} pages={} prefix_rows={}",
                evicted.key.total_tokens,
                evicted.key.batch_size,
                evicted.key.page_indices_len,
                evicted.key.prefix_token_rows_len
            );
        }

        log::info!(
            "Qwen3 prefill graph capture key: tokens={} batch={} pages={} prefix_rows={} marlin_scratch={}",
            key.total_tokens,
            key.batch_size,
            key.page_indices_len,
            key.prefix_token_rows_len,
            self.marlin_prefill_scratch_config().any()
        );
        prefill_ctx
            .graph_resources
            .push(Qwen3PagedPrefillGraphResources {
                token_ids_gpu: self
                    .ctx
                    .stream
                    .alloc_zeros(key.total_tokens.max(1))
                    .map_err(|e| anyhow::anyhow!("alloc prefill graph token ids: {e}"))?,
                hidden: HiddenStates::zeros(&self.ctx, self.config.hidden_size, key.total_tokens)?,
                bufs: self.prefill_buffers_with_marlin_scratch(key.total_tokens)?,
                page_indices_dev: self
                    .ctx
                    .stream
                    .alloc_zeros(key.page_indices_len.max(1))
                    .map_err(|e| anyhow::anyhow!("alloc prefill graph page indices: {e}"))?,
                metadata: Qwen3PrefillGraphMetadata::new(&self.ctx, key.batch_size)?,
                prefix_token_rows_dev: self
                    .ctx
                    .stream
                    .alloc_zeros(key.prefix_token_rows_len.max(1))
                    .map_err(|e| anyhow::anyhow!("alloc prefill graph prefix rows: {e}"))?,
                prefill_token_rows_dev: self
                    .ctx
                    .stream
                    .alloc_zeros(key.total_tokens.max(1))
                    .map_err(|e| anyhow::anyhow!("alloc prefill graph token rows: {e}"))?,
                fwd: crate::ops::PagedPrefillForward::new_hd128(
                    &self.ctx,
                    &layout.sequences,
                    key.page_size,
                )?,
                key,
                graph_state: CudaGraphState::new(),
            });

        prefill_ctx
            .graph_resources
            .last_mut()
            .context("prefill graph resources must exist after cache insert")
    }

    fn upload_paged_prefill_graph_inputs(
        &self,
        resources: &mut Qwen3PagedPrefillGraphResources,
        packed_tokens: &[u32],
        layout: &Qwen3PagedPrefillLayout,
    ) -> Result<()> {
        let token_ids: Vec<i32> = packed_tokens.iter().map(|&token| token as i32).collect();
        self.ctx
            .stream
            .memcpy_htod(&token_ids, &mut resources.token_ids_gpu)
            .map_err(|e| anyhow::anyhow!("prefill graph token ids H2D failed: {e}"))?;
        anyhow::ensure!(
            layout.page_indices.len() <= resources.key.page_indices_len,
            "prefill graph page index bucket too small: actual {} capacity {}",
            layout.page_indices.len(),
            resources.key.page_indices_len
        );
        let mut page_indices = vec![0i32; resources.key.page_indices_len.max(1)];
        page_indices[..layout.page_indices.len()].copy_from_slice(&layout.page_indices);
        self.ctx
            .stream
            .memcpy_htod(&page_indices, &mut resources.page_indices_dev)
            .map_err(|e| anyhow::anyhow!("prefill graph page indices H2D failed: {e}"))?;
        resources.metadata.refresh(&self.ctx, layout)?;
        resources
            .fwd
            .refresh_hd128(&self.ctx, &layout.sequences, resources.key.page_size)?;
        anyhow::ensure!(
            layout.prefix_token_rows.len() <= resources.key.prefix_token_rows_len,
            "prefill graph prefix row bucket too small: actual {} capacity {}",
            layout.prefix_token_rows.len(),
            resources.key.prefix_token_rows_len
        );
        let prefix_rows = if resources.key.prefix_token_rows_len == 0 {
            vec![0]
        } else {
            let mut rows = vec![0i32; resources.key.prefix_token_rows_len];
            rows[..layout.prefix_token_rows.len()].copy_from_slice(&layout.prefix_token_rows);
            rows
        };
        let mut prefix_view = resources
            .prefix_token_rows_dev
            .slice_mut(..prefix_rows.len());
        self.ctx
            .stream
            .memcpy_htod(&prefix_rows, &mut prefix_view)
            .map_err(|e| anyhow::anyhow!("prefill graph prefix rows H2D failed: {e}"))?;
        self.ctx
            .stream
            .memcpy_htod(
                &layout.prefill_token_rows,
                &mut resources.prefill_token_rows_dev,
            )
            .map_err(|e| anyhow::anyhow!("prefill graph token rows H2D failed: {e}"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_mlp_batch_into(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        normed: &HiddenStates,
        gate_out: &mut HiddenStates,
        up_out: &mut HiddenStates,
        act_out: &mut HiddenStates,
        out: &mut HiddenStates,
        ops_backend: ops::CudaOpsBackend<'_, '_>,
    ) -> Result<()> {
        if let Some(gate_up_proj) = layer.mlp.fused_gate_up() {
            if let Some(ll) = self.layer_lora(layer_idx) {
                anyhow::ensure!(
                    ll.gate_proj.is_none() && ll.up_proj.is_none(),
                    "Qwen3 fused gate_up MLP cannot apply gate/up LoRA; \
                     set INFER_QWEN3_FUSED_GATE_UP=0 before loading the model"
                );
            }
            ops_backend.linear_batch_into(gate_up_proj, normed, gate_out)?;
            ops::silu_mul_split_batch_into(&self.ctx, gate_out, act_out)?;
        } else {
            let (gate_proj, up_proj) = layer
                .mlp
                .separate_gate_up()
                .expect("separate Qwen3 MLP must carry gate/up weights");
            ops_backend.linear_batch_into(gate_proj, normed, gate_out)?;
            ops_backend.linear_batch_into(up_proj, normed, up_out)?;
            if let Some(ll) = self.layer_lora(layer_idx) {
                if let Some(ad) = ll.gate_proj.as_ref() {
                    ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, normed, gate_out)?;
                }
                if let Some(ad) = ll.up_proj.as_ref() {
                    ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, normed, up_out)?;
                }
            }
            ops_backend.silu_mul_batch_into(gate_out, up_out, act_out)?;
        }

        ops_backend.linear_batch_into(&layer.mlp.down_proj, act_out, out)?;
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.down_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, act_out, out)?;
            }
        }
        Ok(())
    }

    pub(super) fn build_paged_prefill_sequences(
        &self,
        requests: &[Qwen3PagedPrefillRequest<'_>],
        pool: &TokenKVPool,
    ) -> Result<Qwen3PagedPrefillLayout> {
        anyhow::ensure!(
            !requests.is_empty(),
            "paged prefill batch requires at least one request"
        );

        let mut token_offset = 0usize;
        let mut page_table_offset = 0usize;
        let mut sequences = Vec::with_capacity(requests.len());
        let mut page_indices = Vec::new();
        let mut prefix_token_rows = Vec::new();
        let mut prefill_token_rows =
            Vec::with_capacity(requests.iter().map(|req| req.tokens.len()).sum());

        for req in requests {
            let seq_len = req.tokens.len();
            anyhow::ensure!(
                seq_len > 0,
                "paged prefill request for slot {} must not be empty",
                req.slot
            );

            let pool_seq_len = pool.seq_len(req.slot);
            anyhow::ensure!(
                pool_seq_len >= seq_len,
                "paged prefill: pool seq_len {pool_seq_len} < chunk len {seq_len} for slot {}",
                req.slot
            );
            let start_pos = pool_seq_len - seq_len;
            let num_pages = (start_pos + seq_len).div_ceil(pool.page_size);
            let all_pages = pool.page_indices(req.slot);
            anyhow::ensure!(
                all_pages.len() >= num_pages,
                "paged prefill: slot {} has {} pages, expected at least {num_pages}",
                req.slot,
                all_pages.len()
            );

            page_indices.extend(all_pages[..num_pages].iter().map(|&page| page as i32));
            for pos in 0..start_pos {
                let page = all_pages[pos / pool.page_size] as usize;
                let in_page = pos % pool.page_size;
                prefix_token_rows.push((page * pool.page_size + in_page) as i32);
            }
            for pos in start_pos..start_pos + seq_len {
                let page = all_pages[pos / pool.page_size] as usize;
                let in_page = pos % pool.page_size;
                prefill_token_rows.push((page * pool.page_size + in_page) as i32);
            }
            sequences.push(ops::PagedPrefillSequence {
                token_offset,
                seq_len,
                start_pos,
                page_table_offset,
                num_pages,
            });
            token_offset += seq_len;
            page_table_offset += num_pages;
        }

        Ok(Qwen3PagedPrefillLayout {
            sequences,
            page_indices,
            prefix_token_rows,
            prefill_token_rows,
        })
    }

    fn compute_logits_batch_packed(
        &self,
        hidden: &HiddenStates,
        requests: &[Qwen3PagedPrefillRequest<'_>],
        states: &mut [Qwen3State],
        sequences: &[ops::PagedPrefillSequence],
        bufs: &mut PrefillBuffers,
    ) -> Result<()> {
        anyhow::ensure!(
            requests.len() == sequences.len(),
            "paged prefill request/sequence count mismatch: requests={} sequences={}",
            requests.len(),
            sequences.len()
        );
        let ops_backend = ops::CudaOpsBackend::prefill(&self.ctx);
        for (request, seq) in requests.iter().zip(sequences) {
            let last_token = seq.token_offset + seq.seq_len - 1;
            ops_backend.extract_vec_into(hidden, last_token, &mut bufs.last_hidden)?;
            ops_backend.rms_norm_into(
                &bufs.last_hidden,
                &self.norm,
                self.config.rms_norm_eps,
                &mut bufs.last_normed,
            )?;
            let state = states.get_mut(request.state_idx).ok_or_else(|| {
                anyhow::anyhow!("invalid paged prefill state {}", request.state_idx)
            })?;
            let needs_alloc = state
                .base
                .prefill_logits
                .as_ref()
                .is_none_or(|logits| logits.len != self.config.vocab_size);
            if needs_alloc {
                state.base.prefill_logits = Some(
                    DeviceVec::zeros(&self.ctx, self.config.vocab_size)?
                        .with_label("qwen3_paged_prefill_logits"),
                );
            }
            let seq_logits = state
                .base
                .prefill_logits
                .as_mut()
                .expect("prefill logits allocated");
            ops_backend.linear_vec_into(self.output_projection(), &bufs.last_normed, seq_logits)?;
        }
        Ok(())
    }

    fn supports_paged_prefill_graph(&self) -> bool {
        self.enable_cuda_graph
            && qwen3_prefill_graph_requested()
            && self.lora.is_none()
            && self.layers.iter().all(|layer| {
                Self::graphsafe_batched_weight(&layer.attention.q_proj)
                    && Self::graphsafe_batched_weight(&layer.attention.k_proj)
                    && Self::graphsafe_batched_weight(&layer.attention.v_proj)
                    && Self::graphsafe_batched_weight(&layer.attention.o_proj)
                    && Self::graphsafe_batched_weight(&layer.mlp.down_proj)
                    && match layer.mlp.fused_gate_up() {
                        Some(gate_up) => Self::graphsafe_batched_weight(gate_up),
                        None => layer.mlp.separate_gate_up().is_some_and(|(gate, up)| {
                            Self::graphsafe_batched_weight(gate)
                                && Self::graphsafe_batched_weight(up)
                        }),
                    }
            })
    }

    fn graphsafe_batched_weight(weight: &cuda_kernels::prelude::DeviceMatrix) -> bool {
        ops::graphsafe_batched_weight(weight)
    }

    #[fastrace::trace(name = "launch_prefill_paged_batch")]
    pub(super) fn launch_prefill_paged_batch(
        &self,
        requests: &[Qwen3PagedPrefillRequest<'_>],
        states: &mut [Qwen3State],
        pool: &TokenKVPool,
        prefill_ctx: &mut Qwen3PrefillContext,
    ) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }

        let total_tokens = requests.iter().map(|req| req.tokens.len()).sum();
        let mut packed_tokens = Vec::with_capacity(total_tokens);
        for req in requests {
            packed_tokens.extend_from_slice(req.tokens);
        }

        let layout = self.build_paged_prefill_sequences(requests, pool)?;
        anyhow::ensure!(
            layout.prefill_token_rows.len() == packed_tokens.len(),
            "paged prefill token rows cover {} rows, expected {}",
            layout.prefill_token_rows.len(),
            packed_tokens.len()
        );
        if self.supports_paged_prefill_graph() {
            let key = Qwen3PrefillGraphKey::new(&layout, pool.page_size);
            let completion_error = {
                let resources =
                    self.ensure_paged_prefill_graph_resources(prefill_ctx, key, &layout)?;
                self.upload_paged_prefill_graph_inputs(resources, &packed_tokens, &layout)?;
                let mut graph_state =
                    std::mem::replace(&mut resources.graph_state, CudaGraphState::new());
                let graph_result = graph_state.run_or_capture(&self.ctx, || {
                    ops::embedding_batch(
                        &self.ctx,
                        &self.embed_tokens,
                        &resources.token_ids_gpu,
                        &mut resources.hidden,
                    )?;
                    self.process_all_layers_batch_paged(
                        &mut resources.hidden,
                        &mut resources.bufs,
                        pool,
                        &layout.sequences,
                        &resources.page_indices_dev,
                        &resources.metadata.page_table_offsets_dev,
                        &resources.metadata.start_positions_dev,
                        resources.key.page_indices_len,
                        &resources.prefix_token_rows_dev,
                        resources.key.prefix_token_rows_len,
                        &resources.prefill_token_rows_dev,
                        &mut resources.fwd,
                    )
                });
                resources.graph_state = graph_state;
                match graph_result {
                    Ok(()) => self
                        .compute_logits_batch_packed(
                            &resources.hidden,
                            requests,
                            states,
                            &layout.sequences,
                            &mut resources.bufs,
                        )
                        .err(),
                    Err(err) => Some(err),
                }
            };
            prefill_ctx.set_completed_pending(PendingPagedPrefill {
                _owned: None,
                owners: prefill_owners(requests),
                completion_error,
            })?;
            return Ok(());
        }
        let mut bufs = self.prefill_buffers(packed_tokens.len())?;
        let page_indices_dev: CudaSlice<i32> = self
            .ctx
            .stream
            .clone_htod(&layout.page_indices)
            .map_err(|e| anyhow::anyhow!("page_indices H2D failed: {e}"))?;
        let prefix_token_rows_upload: &[i32] = if layout.prefix_token_rows.is_empty() {
            &[0]
        } else {
            &layout.prefix_token_rows
        };
        let prefix_token_rows_dev: CudaSlice<i32> = self
            .ctx
            .stream
            .clone_htod(prefix_token_rows_upload)
            .map_err(|e| anyhow::anyhow!("prefix token rows H2D failed: {e}"))?;
        let start_positions: Vec<i32> = layout
            .sequences
            .iter()
            .map(|seq| seq.start_pos as i32)
            .collect();
        let page_table_offsets: Vec<i32> = layout
            .sequences
            .iter()
            .map(|seq| seq.page_table_offset as i32)
            .collect();
        let page_table_offsets_dev: CudaSlice<i32> = self
            .ctx
            .stream
            .clone_htod(&page_table_offsets)
            .map_err(|e| anyhow::anyhow!("page-table offsets H2D failed: {e}"))?;
        let start_positions_dev: CudaSlice<i32> = self
            .ctx
            .stream
            .clone_htod(&start_positions)
            .map_err(|e| anyhow::anyhow!("start positions H2D failed: {e}"))?;
        let prefill_token_rows_dev: CudaSlice<i32> = self
            .ctx
            .stream
            .clone_htod(&layout.prefill_token_rows)
            .map_err(|e| anyhow::anyhow!("prefill token rows H2D failed: {e}"))?;
        let mut fwd = crate::ops::PagedPrefillForward::new_hd128(
            &self.ctx,
            &layout.sequences,
            pool.page_size,
        )?;
        let mut hidden = self.get_embeddings_batch(&packed_tokens)?;
        let completion_error = match self.process_all_layers_batch_paged(
            &mut hidden,
            &mut bufs,
            pool,
            &layout.sequences,
            &page_indices_dev,
            &page_table_offsets_dev,
            &start_positions_dev,
            layout.page_indices.len(),
            &prefix_token_rows_dev,
            layout.prefix_token_rows.len(),
            &prefill_token_rows_dev,
            &mut fwd,
        ) {
            Ok(()) => self
                .compute_logits_batch_packed(
                    &hidden,
                    requests,
                    states,
                    &layout.sequences,
                    &mut bufs,
                )
                .err(),
            Err(err) => Some(err),
        };
        prefill_ctx.set_pending(
            &self.ctx,
            PendingPagedPrefill {
                _owned: Some(OwnedPendingPagedPrefillBuffers {
                    _hidden: hidden,
                    _bufs: bufs,
                    _page_indices_dev: page_indices_dev,
                    _page_table_offsets_dev: page_table_offsets_dev,
                    _start_positions_dev: start_positions_dev,
                    _prefix_token_rows_dev: prefix_token_rows_dev,
                    _prefill_token_rows_dev: prefill_token_rows_dev,
                    _fwd: fwd,
                }),
                owners: prefill_owners(requests),
                completion_error,
            },
        )?;
        Ok(())
    }

    pub(super) fn run_prefill_paged_batch_sync(
        &self,
        requests: &[Qwen3PagedPrefillRequest<'_>],
        states: &mut [Qwen3State],
        pool: &TokenKVPool,
    ) -> Result<()> {
        let mut prefill_ctx = Qwen3PrefillContext::new(&self.ctx)?;
        self.launch_prefill_paged_batch(requests, states, pool, &mut prefill_ctx)?;
        prefill_ctx.wait(&[])
    }

    /// Paged-KV prefill over one or more packed requests. Writes K/V directly
    /// to the paged pool via page-table indirection and runs one paged-prefill
    /// attention call per layer over the packed varlen batch. No contiguous KV
    /// cache is touched; the scheduler must skip
    /// `migrate_kv_range_to_paged` for this forward.
    #[fastrace::trace(name = "process_all_layers_batch_paged")]
    fn process_all_layers_batch_paged(
        &self,
        hidden: &mut HiddenStates,
        bufs: &mut PrefillBuffers,
        pool: &TokenKVPool,
        sequences: &[ops::PagedPrefillSequence],
        page_indices: &CudaSlice<i32>,
        page_table_offsets: &CudaSlice<i32>,
        start_positions: &CudaSlice<i32>,
        page_indices_len: usize,
        prefix_token_rows: &CudaSlice<i32>,
        prefix_token_count: usize,
        prefill_token_rows: &CudaSlice<i32>,
        fwd: &mut crate::ops::PagedPrefillForward,
    ) -> Result<()> {
        let seq_len = hidden.seq_len;

        anyhow::ensure!(
            !sequences.is_empty(),
            "paged prefill forward requires at least one sequence"
        );
        anyhow::ensure!(
            sequences
                .last()
                .map_or(0, |seq| seq.token_offset + seq.seq_len)
                == seq_len,
            "paged prefill token packing does not cover all hidden rows"
        );

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_layer_batch_paged(
                layer_idx,
                layer,
                hidden,
                bufs,
                pool,
                sequences,
                page_indices,
                page_table_offsets,
                start_positions,
                page_indices_len,
                prefix_token_rows,
                prefix_token_count,
                prefill_token_rows,
                fwd,
            )?;
        }

        Ok(())
    }

    /// Paged-KV variant of `forward_layer_batch`. Differences vs the contiguous
    /// path:
    ///  - No `kv_cache.init_if_needed` / `prepare_layer` / `commit_layer`.
    ///  - Attention call writes K/V directly into the paged pool through the
    ///    page-table indirection kernel + paged-prefill attention.
    ///  - No `scatter_write_kv` dual-write step.
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_batch_paged(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        bufs: &mut PrefillBuffers,
        pool: &TokenKVPool,
        sequences: &[ops::PagedPrefillSequence],
        page_indices: &CudaSlice<i32>,
        page_table_offsets: &CudaSlice<i32>,
        start_positions: &CudaSlice<i32>,
        page_indices_len: usize,
        prefix_token_rows: &CudaSlice<i32>,
        prefix_token_count: usize,
        prefill_token_rows: &CudaSlice<i32>,
        fwd: &mut crate::ops::PagedPrefillForward,
    ) -> Result<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} pre-norm hidden"),
            self.config.hidden_size,
        );
        let ops_backend = bufs.marlin_prefill_scratch.as_ref().map_or_else(
            || ops::CudaOpsBackend::prefill(&self.ctx),
            |scratch| ops::CudaOpsBackend::prefill_with_marlin_scratch(&self.ctx, scratch),
        );
        // 1. RMSNorm
        ops_backend.rms_norm_batch_into(
            hidden,
            &layer.input_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        )?;

        // 2. QKV projections
        ops_backend.linear_batch_into(&layer.attention.q_proj, &bufs.normed, &mut bufs.q_batch)?;
        ops_backend.linear_batch_into(&layer.attention.k_proj, &bufs.normed, &mut bufs.k_batch)?;
        ops_backend.linear_batch_into(&layer.attention.v_proj, &bufs.normed, &mut bufs.v_batch)?;
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.q_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.q_batch)?;
            }
            if let Some(ad) = ll.k_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.k_batch)?;
            }
            if let Some(ad) = ll.v_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.v_batch)?;
            }
        }

        // 3. Paged-KV attention: QK norm + RoPE + paged K/V write (page-table
        //    indirection) + TileLang paged prefill.
        let nrp = ops::NormRopeParams {
            q_norm: &layer.attention.q_norm,
            k_norm: &layer.attention.k_norm,
            cos_cache: &self.cos_cache,
            sin_cache: &self.sin_cache,
            rms_eps: self.config.rms_norm_eps,
        };
        let heads = ops::HeadConfig {
            num_q_heads: num_heads,
            num_kv_heads,
            head_dim,
        };
        let meta = ops::PagedPrefillMeta {
            pool,
            layer_idx,
            page_indices,
            page_table_offsets,
            start_positions,
            sequences,
            page_indices_len,
            page_size: pool.page_size,
        };
        self.refill_paged_prefill_prefix_if_needed(
            pool,
            layer_idx,
            prefix_token_rows,
            prefix_token_count,
            num_kv_heads,
            head_dim,
        )?;
        ops::prefill_attention_paged_batch(
            &self.ctx,
            &mut bufs.q_batch,
            &mut bufs.k_batch,
            &bufs.v_batch,
            &nrp,
            &meta,
            fwd,
            &mut bufs.attn_output,
            &heads,
        )?;

        self.finalize_paged_prefill_kv_layer(
            pool,
            layer_idx,
            prefill_token_rows,
            hidden.seq_len,
            num_kv_heads,
            head_dim,
        )?;

        // 4-8: Same as forward_layer_batch (O proj, residual, MLP)
        ops_backend.linear_batch_into(
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        )?;
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.o_proj.as_ref() {
                ops::apply_lora_gemm_add(
                    &self.ctx,
                    &ad.a,
                    &ad.b,
                    &bufs.attn_output,
                    &mut bufs.o_buf,
                )?;
            }
        }
        self.layer_communicator
            .post_attn_all_reduce_hidden_states(&mut bufs.o_buf)?;

        ops_backend.add_batch_into(hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} after-attn+residual"),
            self.config.hidden_size,
        );

        ops_backend.rms_norm_batch_into(
            hidden,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        )?;

        self.forward_mlp_batch_into(
            layer_idx,
            layer,
            &bufs.normed,
            &mut bufs.gate_out,
            &mut bufs.up_out,
            &mut bufs.act_out,
            &mut bufs.o_buf,
            ops_backend,
        )?;
        self.layer_communicator
            .post_mlp_all_reduce_hidden_states(&mut bufs.o_buf)?;

        ops_backend.add_batch_into(hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} layer-end"),
            self.config.hidden_size,
        );

        Ok(())
    }

    fn refill_paged_prefill_prefix_if_needed(
        &self,
        pool: &TokenKVPool,
        layer_idx: usize,
        prefix_token_rows: &CudaSlice<i32>,
        prefix_token_count: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        if prefix_token_count == 0 {
            return Ok(());
        }
        match pool.format {
            KVFormat::BF16 => Ok(()),
            KVFormat::FP8E4M3 => {
                let stream = &self.ctx.stream;
                kv_quant::dequantize_paged_kv_fp8_to_hnd(
                    &self.ctx,
                    pool.k_data_ptr(layer_idx, stream),
                    pool.k_scales_ptr(layer_idx, stream),
                    pool.k_work_ptr(stream),
                    prefix_token_rows,
                    num_kv_heads,
                    head_dim,
                    pool.kv_dim,
                    prefix_token_count,
                )?;
                kv_quant::dequantize_paged_kv_fp8_to_hnd(
                    &self.ctx,
                    pool.v_data_ptr(layer_idx, stream),
                    pool.v_scales_ptr(layer_idx, stream),
                    pool.v_work_ptr(stream),
                    prefix_token_rows,
                    num_kv_heads,
                    head_dim,
                    pool.kv_dim,
                    prefix_token_count,
                )
            }
            KVFormat::INT8 => {
                let stream = &self.ctx.stream;
                kv_quant::dequantize_paged_kv_int8_to_hnd(
                    &self.ctx,
                    pool.k_data_ptr(layer_idx, stream),
                    pool.k_scales_ptr(layer_idx, stream),
                    pool.k_work_ptr(stream),
                    prefix_token_rows,
                    num_kv_heads,
                    head_dim,
                    pool.kv_dim,
                    prefix_token_count,
                )?;
                kv_quant::dequantize_paged_kv_int8_to_hnd(
                    &self.ctx,
                    pool.v_data_ptr(layer_idx, stream),
                    pool.v_scales_ptr(layer_idx, stream),
                    pool.v_work_ptr(stream),
                    prefix_token_rows,
                    num_kv_heads,
                    head_dim,
                    pool.kv_dim,
                    prefix_token_count,
                )
            }
            KVFormat::TurboQuant { .. } => {
                anyhow::bail!("Qwen3 paged prefill does not support TurboQuant KV prefix refill")
            }
        }
    }

    fn finalize_paged_prefill_kv_layer(
        &self,
        pool: &TokenKVPool,
        layer_idx: usize,
        prefill_token_rows: &CudaSlice<i32>,
        token_count: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        match pool.format {
            KVFormat::BF16 => Ok(()),
            KVFormat::FP8E4M3 => {
                let stream = &self.ctx.stream;
                kv_quant::quantize_paged_kv_fp8(
                    &self.ctx,
                    pool.k_work_ptr(stream),
                    pool.k_data_ptr(layer_idx, stream),
                    pool.k_scales_ptr(layer_idx, stream),
                    prefill_token_rows,
                    num_kv_heads,
                    head_dim,
                    pool.kv_dim,
                    token_count,
                )?;
                kv_quant::quantize_paged_kv_fp8(
                    &self.ctx,
                    pool.v_work_ptr(stream),
                    pool.v_data_ptr(layer_idx, stream),
                    pool.v_scales_ptr(layer_idx, stream),
                    prefill_token_rows,
                    num_kv_heads,
                    head_dim,
                    pool.kv_dim,
                    token_count,
                )
            }
            KVFormat::INT8 => {
                let stream = &self.ctx.stream;
                kv_quant::quantize_paged_kv_single(
                    &self.ctx,
                    pool.k_work_ptr(stream),
                    pool.k_data_ptr(layer_idx, stream),
                    pool.k_scales_ptr(layer_idx, stream),
                    prefill_token_rows,
                    num_kv_heads,
                    head_dim,
                    pool.kv_dim,
                    token_count,
                )?;
                kv_quant::quantize_paged_kv_single(
                    &self.ctx,
                    pool.v_work_ptr(stream),
                    pool.v_data_ptr(layer_idx, stream),
                    pool.v_scales_ptr(layer_idx, stream),
                    prefill_token_rows,
                    num_kv_heads,
                    head_dim,
                    pool.kv_dim,
                    token_count,
                )
            }
            KVFormat::TurboQuant { .. } => {
                anyhow::bail!("Qwen3 paged prefill does not support TurboQuant KV finalization")
            }
        }
    }

    pub(super) fn compute_logits_batch(&self, hidden: &HiddenStates) -> Result<DeviceVec> {
        crate::model::common::compute_logits_batch(
            &self.ctx,
            hidden,
            &self.norm,
            self.output_projection(),
            self.config.rms_norm_eps,
            false, // standard RMSNorm (not offset)
        )
    }

    fn forward_layer_batch(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        start_pos: usize,
        kv_cache: &mut KVCache,
        bufs: &mut PrefillBuffers,
    ) -> Result<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        kv_cache.init_if_needed(&self.ctx, self.config.head_dim)?;
        let ops_backend = ops::CudaOpsBackend::prefill(&self.ctx);

        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} pre-norm hidden"),
            self.config.hidden_size,
        );
        // 1. RMSNorm → bufs.normed. When the fp32 residual shadow is active,
        //    read from it directly — skipping the bf16 rounding in `hidden`.
        if let Some(ref r) = bufs.residual_f32 {
            ops::rms_norm_batch_f32_in_into(
                &self.ctx,
                r,
                &layer.input_layernorm,
                &mut bufs.normed,
                hidden.seq_len,
                self.config.rms_norm_eps,
            )?;
        } else {
            ops_backend.rms_norm_batch_into(
                hidden,
                &layer.input_layernorm,
                self.config.rms_norm_eps,
                &mut bufs.normed,
            )?;
        }
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.normed,
            &format!("L{layer_idx} after-input-norm"),
            self.config.hidden_size,
        );

        // 2. QKV projections → bufs.q_batch, bufs.k_batch, bufs.v_batch
        ops_backend.linear_batch_into(&layer.attention.q_proj, &bufs.normed, &mut bufs.q_batch)?;
        ops_backend.linear_batch_into(&layer.attention.k_proj, &bufs.normed, &mut bufs.k_batch)?;
        ops_backend.linear_batch_into(&layer.attention.v_proj, &bufs.normed, &mut bufs.v_batch)?;
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.q_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.q_batch)?;
            }
            if let Some(ad) = ll.k_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.k_batch)?;
            }
            if let Some(ad) = ll.v_proj.as_ref() {
                ops::apply_lora_gemm_add(&self.ctx, &ad.a, &ad.b, &bufs.normed, &mut bufs.v_batch)?;
            }
        }
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.q_batch,
            &format!("L{layer_idx} q_proj_out (pre-norm-rope)"),
            bufs.q_batch.hidden_dim,
        );
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.k_batch,
            &format!("L{layer_idx} k_proj_out (pre-norm-rope)"),
            bufs.k_batch.hidden_dim,
        );
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.v_batch,
            &format!("L{layer_idx} v_proj_out"),
            bufs.v_batch.hidden_dim,
        );

        // 3. Prefill attention -> bufs.attn_output
        let (k_cache_layer, v_cache_layer) = kv_cache.prepare_layer(&self.ctx, layer_idx)?;
        let nrp = ops::NormRopeParams {
            q_norm: &layer.attention.q_norm,
            k_norm: &layer.attention.k_norm,
            cos_cache: &self.cos_cache,
            sin_cache: &self.sin_cache,
            rms_eps: self.config.rms_norm_eps,
        };
        let heads = ops::HeadConfig {
            num_q_heads: num_heads,
            num_kv_heads,
            head_dim,
        };
        ops::prefill_attention_batch(
            &self.ctx,
            &mut bufs.q_batch,
            &mut bufs.k_batch,
            &bufs.v_batch,
            &nrp,
            k_cache_layer,
            v_cache_layer,
            &mut bufs.attn_output,
            &heads,
            start_pos,
        )?;
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.q_batch,
            &format!("L{layer_idx} q (post-norm-rope)"),
            bufs.q_batch.hidden_dim,
        );
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.attn_output,
            &format!("L{layer_idx} attn_output (pre-o-proj)"),
            bufs.attn_output.hidden_dim,
        );
        // Quantize newly written KV tokens → INT8 storage (no-op for BF16)
        kv_cache.commit_layer(&self.ctx, layer_idx, start_pos, hidden.seq_len)?;

        // 4. O projection → bufs.o_buf (as o_batch)
        ops_backend.linear_batch_into(
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        )?;
        if let Some(ll) = self.layer_lora(layer_idx) {
            if let Some(ad) = ll.o_proj.as_ref() {
                ops::apply_lora_gemm_add(
                    &self.ctx,
                    &ad.a,
                    &ad.b,
                    &bufs.attn_output,
                    &mut bufs.o_buf,
                )?;
            }
        }
        self.layer_communicator
            .post_attn_all_reduce_hidden_states(&mut bufs.o_buf)?;
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            &bufs.o_buf,
            &format!("L{layer_idx} o_proj_out"),
            bufs.o_buf.hidden_dim,
        );

        // 5. Residual add: hidden + o_buf.
        //    With fp32 shadow: accumulate into residual_f32 (fp32 precision),
        //    then sync hidden for downstream bf16 consumers / debug dumps.
        //    Without shadow: use the classic bf16 add + swap path.
        if let Some(ref mut r) = bufs.residual_f32 {
            ops::add_bf16_into_f32(&self.ctx, r, &bufs.o_buf)?;
            ops::cast_f32_to_bf16(&self.ctx, r, hidden)?;
        } else {
            ops_backend.add_batch_into(hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
            std::mem::swap(hidden, &mut bufs.hidden_out);
        }
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} after-attn+residual"),
            self.config.hidden_size,
        );

        // 6. MLP RMSNorm → bufs.normed.
        if let Some(ref r) = bufs.residual_f32 {
            ops::rms_norm_batch_f32_in_into(
                &self.ctx,
                r,
                &layer.post_attention_layernorm,
                &mut bufs.normed,
                hidden.seq_len,
                self.config.rms_norm_eps,
            )?;
        } else {
            ops_backend.rms_norm_batch_into(
                hidden,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
                &mut bufs.normed,
            )?;
        }

        // 7. MLP: gate/up → act → down → bufs.o_buf (reused for mlp_out; step 5 is done)
        self.forward_mlp_batch_into(
            layer_idx,
            layer,
            &bufs.normed,
            &mut bufs.gate_out,
            &mut bufs.up_out,
            &mut bufs.act_out,
            &mut bufs.o_buf,
            ops_backend,
        )?;
        self.layer_communicator
            .post_mlp_all_reduce_hidden_states(&mut bufs.o_buf)?;

        // 8. Residual add: attn_residual + mlp_out.
        if let Some(ref mut r) = bufs.residual_f32 {
            ops::add_bf16_into_f32(&self.ctx, r, &bufs.o_buf)?;
            ops::cast_f32_to_bf16(&self.ctx, r, hidden)?;
        } else {
            ops_backend.add_batch_into(hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
            std::mem::swap(hidden, &mut bufs.hidden_out);
        }
        crate::model::common::debug_dump_hidden(
            &self.ctx,
            hidden,
            &format!("L{layer_idx} layer-end"),
            self.config.hidden_size,
        );

        Ok(())
    }
}
