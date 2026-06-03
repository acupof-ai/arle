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

use crate::model::CudaModel;
use crate::ops::argmax;

const SUPPORTED_PAGE_SIZE: usize = 16;

pub(crate) struct RealCudaExecutor {
    model: CudaModel,
    kv: PagedKVPool,
    num_slots: usize,
}

impl std::fmt::Debug for RealCudaExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RealCudaExecutor")
            .field("model", &self.model)
            .field("num_slots", &self.num_slots)
            .field("page_size", &self.kv.page_size)
            .field("max_total_pages", &self.kv.max_total_pages)
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
            self.model.forward_tokens(
                row.slot,
                &[row.last_token],
                row.kv_seq_len,
                &mut self.kv,
                &row.params,
                position,
            )?
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
