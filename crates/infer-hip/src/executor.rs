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
use infer_seam::{BackendExecutor, KvPool, PollResult, PrefixBlock};

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

    fn reusable_prefix_blocks(&self, _blocks: &[PrefixBlock]) -> usize {
        // HIP DSv4 stores the real restore state in per-slot arenas
        // (`slot.sw_window_ring`, compressor/indexer buffers, and seq_len).
        // Host pages are bookkeeping only, so page-prefix attach is never a
        // complete restore boundary. Add whole-slot restore when HIP needs
        // preemption reuse.
        0
    }

    fn kv_tier_transfer_is_zero_copy(&self) -> bool {
        true
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

#[cfg(test)]
mod tests {
    use super::*;
    use infer_plan::{DecodeRow, ForwardMode, PrefillRow};

    fn one_row_plan(decode: bool) -> ForwardPlan {
        ForwardPlan {
            mode: if decode {
                ForwardMode::Decode
            } else {
                ForwardMode::Prefill
            },
            decode_rows: if decode {
                vec![DecodeRow {
                    slot: 0,
                    last_token: 7,
                    kv_seq_len: 3,
                    params: SamplingParams::default(),
                }]
            } else {
                Vec::new()
            },
            prefill_rows: if decode {
                Vec::new()
            } else {
                vec![PrefillRow {
                    slot: 0,
                    tokens: vec![1, 2, 3],
                    start_pos: 0,
                    total_tokens: 3,
                    params: SamplingParams::default(),
                }]
            },
            microbatch: None,
            spec: None,
        }
    }

    fn pool() -> HipKvPool {
        HipKvPool::new(&crate::kv_pool::test_config(), 2, 8, 16, 256)
    }

    #[test]
    fn idle_plan_returns_empty_output() {
        let mut exec = HipDsv4Executor::unloaded();
        let mut pool = pool();
        let inflight = exec.submit(&ForwardPlan::idle(), &mut pool).unwrap();
        match exec.poll(inflight).unwrap() {
            PollResult::Ready(out) => assert!(out.tokens.is_empty()),
            PollResult::NotReady(_) => panic!("MVP resolves synchronously"),
        }
    }

    #[test]
    fn unloaded_executor_errors_clearly_not_panics() {
        let mut exec = HipDsv4Executor::unloaded();
        let mut pool = pool();
        for plan in [one_row_plan(true), one_row_plan(false)] {
            let err = exec.submit(&plan, &mut pool).unwrap_err().to_string();
            assert!(err.contains("no model loaded"), "{err}");
        }
    }

    #[test]
    fn multi_row_plans_rejected_loud() {
        let mut exec = HipDsv4Executor::unloaded();
        let mut pool = pool();
        let mut plan = one_row_plan(true);
        plan.decode_rows.push(plan.decode_rows[0].clone());
        let err = exec.submit(&plan, &mut pool).unwrap_err().to_string();
        assert!(err.contains("exactly one"), "{err}");
        let mut mixed = one_row_plan(true);
        mixed.prefill_rows = one_row_plan(false).prefill_rows;
        mixed.mode = ForwardMode::Mixed;
        assert!(exec.submit(&mixed, &mut pool).is_err());
    }

    #[test]
    fn page_prefix_reuse_is_fail_closed() {
        let exec = HipDsv4Executor::unloaded();
        assert_eq!(
            exec.reusable_prefix_blocks(&[
                PrefixBlock::ResidentPage(1),
                PrefixBlock::DemotedKey(2)
            ]),
            0
        );
    }

    #[cfg(not(feature = "hip"))]
    #[test]
    fn load_without_hip_feature_errors_clean() {
        // Synthetic GGUF exercises the full host pipeline (parse → config →
        // residency plan) before the gated upload fails with a clear error.
        let path = synthetic_loadable_gguf();
        let err = match load_dsv4_gguf(&path, 1, 64) {
            Ok(_) => panic!("load must fail without the hip feature"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("not compiled"), "{err}");
        std::fs::remove_file(path).ok();
    }

    #[cfg(not(feature = "hip"))]
    fn synthetic_loadable_gguf() -> std::path::PathBuf {
        use infer_gguf::gguf::test_writer::{T, V, write_to_temp};
        let kvs = vec![
            ("general.architecture", V::Str("deepseek4")),
            ("deepseek4.block_count", V::U32(1)),
            ("deepseek4.context_length", V::U32(256)),
            ("deepseek4.embedding_length", V::U32(256)),
            ("deepseek4.vocab_size", V::U32(512)),
            ("deepseek4.attention.head_count", V::U32(4)),
            ("deepseek4.attention.head_count_kv", V::U32(1)),
            ("deepseek4.attention.key_length", V::U32(64)),
            ("deepseek4.attention.layer_norm_rms_epsilon", V::F32(1e-6)),
            ("deepseek4.attention.q_lora_rank", V::U32(256)),
            ("deepseek4.attention.sliding_window", V::U32(32)),
            ("deepseek4.rope.dimension_count", V::U32(32)),
            ("deepseek4.rope.freq_base", V::F32(10000.0)),
            ("deepseek4.expert_count", V::U32(4)),
            ("deepseek4.expert_used_count", V::U32(2)),
            ("deepseek4.expert_feed_forward_length", V::U32(256)),
        ];
        // Q4_K everywhere keeps every matmul residency on the GEMV surface.
        let q4k = |name: &str, dims: Vec<u64>| {
            let n: u64 = dims.iter().product();
            T {
                name: name.into(),
                dims,
                type_id: 12,
                data: vec![0u8; n as usize / 256 * 144],
            }
        };
        write_to_temp(
            &kvs,
            &[q4k("token_embd.weight", vec![256, 512])],
            "executor-load",
        )
    }
}
