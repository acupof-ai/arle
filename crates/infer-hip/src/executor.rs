//! HIP DSv4 `BackendExecutor` + GGUF assembly.
//!
//! Mirrors the SIMPLER `infer-metal/src/executor.rs` shape: one prefill or
//! decode row per plan, synchronous submit (`poll` only materializes the
//! already-sampled token), host `infer_plan::sample_token` over f32 logits.
//! Chunked prefill runs the decode path token-by-token (correctness-first
//! MVP per plan §2.2; the batched mmq path is the later optimization).
//!
//! Off-box: without the `hip` feature every load path returns a clear
//! error (never panics); the executor type itself stays constructible so
//! plan-shape handling is unit-testable anywhere.

use anyhow::{Result, anyhow, bail, ensure};
use infer_plan::{ForwardPlan, SamplingParams, SlotToken, StepOutput};
use infer_seam::{BackendExecutor, KvPool, PollResult};

use crate::kv_pool::HipKvPool;

/// In-flight handle — the MVP resolves synchronously at submit.
#[derive(Debug)]
pub enum HipInflight {
    Ready(StepOutput),
}

/// HIP DSv4 executor. `model` is populated only by [`load_dsv4_gguf`] on a
/// ROCm box; the empty executor errors loud on any non-idle submit.
#[derive(Default)]
pub struct HipDsv4Executor {
    #[cfg(feature = "hip")]
    model: Option<crate::model::HipDsv4Model>,
    stop_tokens: Vec<u32>,
}

impl HipDsv4Executor {
    /// An executor with no model — every non-idle submit errors loud.
    #[must_use]
    pub fn unloaded() -> Self {
        Self::default()
    }

    fn forward_tokens(
        &mut self,
        slot: usize,
        epoch: u64,
        tokens: &[u32],
        start_pos: usize,
        params: &SamplingParams,
        position: u64,
    ) -> Result<u32> {
        ensure!(
            !tokens.is_empty(),
            "HIP forward requires at least one token"
        );
        #[cfg(feature = "hip")]
        if let Some(model) = self.model.as_mut() {
            let logits = tokens
                .iter()
                .enumerate()
                .try_fold(Vec::new(), |_, (i, &token)| {
                    model.forward_token(slot, epoch, token, start_pos + i)
                })?;
            return Ok(infer_plan::sample_token(&logits, params, position));
        }
        let _ = (slot, epoch, start_pos, params, position);
        bail!(
            "HIP DSv4 executor has no model loaded{}",
            if cfg!(feature = "hip") {
                " (load via load_dsv4_gguf on a ROCm box)"
            } else {
                " (built without the `hip` feature)"
            }
        )
    }
}

impl BackendExecutor for HipDsv4Executor {
    type Inflight = HipInflight;

    fn prefix_reuse(&mut self) -> Option<&mut dyn infer_seam::PrefixReuse> {
        // Written opt-out: HIP DSv4 stores the real restore state in per-slot
        // arenas (`slot.sw_window_ring`, compressor/indexer buffers, and
        // seq_len). Host pages are bookkeeping only, so page-prefix attach is
        // never a complete restore boundary. Add whole-slot restore when HIP
        // needs preemption reuse.
        None
    }

    fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> Result<HipInflight> {
        if plan.is_idle() {
            return Ok(HipInflight::Ready(StepOutput { tokens: Vec::new() }));
        }
        let row_count = plan.prefill_rows.len() + plan.decode_rows.len();
        ensure!(
            row_count == 1,
            "HIP DSv4 executor supports exactly one prefill or decode row, got {row_count}"
        );

        if let Some(row) = plan.prefill_rows.first() {
            ensure!(
                !row.tokens.is_empty(),
                "HIP prefill row must contain at least one token"
            );
            let epoch = kv.slot_epoch(row.slot);
            let position = (row.start_pos + row.tokens.len()) as u64;
            let token = self.forward_tokens(
                row.slot,
                epoch,
                &row.tokens,
                row.start_pos,
                &row.params,
                position,
            )?;
            return Ok(HipInflight::Ready(StepOutput {
                tokens: vec![SlotToken {
                    slot: row.slot,
                    token,
                    logprob: None,
                    top_logprobs: Vec::new(),
                    finish: None,
                }],
            }));
        }
        if let Some(row) = plan.decode_rows.first() {
            let epoch = kv.slot_epoch(row.slot);
            let position = (row.kv_seq_len + 1) as u64;
            let token = self.forward_tokens(
                row.slot,
                epoch,
                &[row.last_token],
                row.kv_seq_len,
                &row.params,
                position,
            )?;
            return Ok(HipInflight::Ready(StepOutput {
                tokens: vec![SlotToken {
                    slot: row.slot,
                    token,
                    logprob: None,
                    top_logprobs: Vec::new(),
                    finish: None,
                }],
            }));
        }
        bail!("HIP DSv4 executor received a non-idle plan with no rows")
    }

    fn poll(&mut self, inflight: HipInflight) -> Result<PollResult<HipInflight>> {
        match inflight {
            HipInflight::Ready(output) => Ok(PollResult::Ready(output)),
        }
    }

    fn model_stop_token_ids(&self) -> Vec<u32> {
        self.stop_tokens.clone()
    }
}

/// GGUF → (executor, host KV pool): open, map config, plan residency,
/// upload (device), build the model. Page size pins the host bookkeeping
/// granularity only — DSv4 slot state is per-slot device arenas.
pub const DEFAULT_PAGE_SIZE: usize = 64;

pub fn load_dsv4_gguf(
    path: impl AsRef<std::path::Path>,
    num_slots: usize,
    max_seq_len: usize,
) -> Result<(HipDsv4Executor, HipKvPool)> {
    ensure!(num_slots > 0, "HIP load requires at least one slot");
    ensure!(max_seq_len > 0, "HIP load requires max_seq_len > 0");
    let gguf = infer_gguf::gguf::GgufFile::open(&path)?;
    let config = infer_gguf::deepseek4::dsv4_config_from_gguf(&gguf)?;
    let plan = crate::loader::plan_model(&gguf, config.num_hidden_layers)?;
    crate::model::validate_matmul_residency(
        plan.tensors.iter().map(|t| {
            let ndims = gguf.tensor(&t.name).map_or(2, |i| i.dims.len());
            (t.name.as_str(), t.kind, t.residency, ndims)
        }),
        config.tie_word_embeddings,
    )?;
    #[cfg(feature = "hip")]
    {
        // ONE shared dynamic-draw page pool (mirrors CUDA dense): size from a ROCm
        // free/total probe → `profile_kv_pool_tokens`, never `× num_slots`.
        // TODO(#77, pending-remote): the GGUF DSv4 lane zeroes `kv_lora_rank`, so
        // the latent cell is approximated as `head_dim × layers × 2 (bf16)`; refine
        // once the GGUF MLA split is parsed and the device pool goes dynamic.
        let cell = config
            .head_dim
            .saturating_mul(config.num_hidden_layers)
            .saturating_mul(2) as u64;
        let requested_pages = max_seq_len.div_ceil(DEFAULT_PAGE_SIZE).max(1);
        let total_pages = match hip_sys::mem_get_info() {
            Ok((free, total)) => {
                let tokens =
                    infer_seam::profile_kv_pool_tokens(free as u64, total as u64, cell, 0.9);
                ((tokens as usize) / DEFAULT_PAGE_SIZE).max(requested_pages)
            }
            Err(_) => requested_pages, // probe miss → pure requested floor, no num_slots multiply
        };
        let pool = HipKvPool::new(
            &config,
            num_slots,
            total_pages,
            DEFAULT_PAGE_SIZE,
            max_seq_len,
        );
        let tensors = crate::loader::upload::upload_plan(&gguf, &plan)
            .map_err(|e| anyhow!("DSv4 GGUF upload failed: {e}"))?;
        let model = crate::model::HipDsv4Model::new(
            &gguf,
            config.clone(),
            &plan,
            tensors,
            num_slots,
            max_seq_len,
        )?;
        let stop_tokens = config.eos_token_id.into_iter().collect();
        Ok((
            HipDsv4Executor {
                model: Some(model),
                stop_tokens,
            },
            pool,
        ))
    }
    #[cfg(not(feature = "hip"))]
    {
        let _ = (plan, num_slots, max_seq_len, config);
        Err(anyhow!(
            "HIP backend not compiled: rebuild with --features hip on a ROCm box \
             (host stages validated: GGUF parsed, config mapped, residency planned)"
        ))
    }
}
