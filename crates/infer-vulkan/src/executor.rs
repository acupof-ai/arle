//! Vulkan `BackendExecutor` skeleton.
//!
//! P2 keeps submit/poll behavior identical to `infer-hip`: one row per plan,
//! synchronous completion, and host sampling once numeric logits exist. Until
//! a model is loaded, every non-idle plan errors loud.

use anyhow::{Result, bail, ensure};
use infer_plan::{ForwardPlan, SamplingParams, SlotToken, StepOutput};
use infer_seam::{BackendExecutor, KvPool, PollResult};

use crate::kv_pool::VulkanKvPool;

pub const DEFAULT_PAGE_SIZE: usize = 64;

#[derive(Debug)]
pub enum VulkanInflight {
    Ready(StepOutput),
}

#[derive(Default)]
pub struct VulkanExecutor {
    #[cfg(feature = "vulkan")]
    model: Option<crate::model_qwen3::VulkanQwen3Model>,
    stop_tokens: Vec<u32>,
}

impl VulkanExecutor {
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
            "Vulkan forward requires at least one token"
        );
        #[cfg(feature = "vulkan")]
        if let Some(model) = self.model.as_mut() {
            let mut logits = Vec::new();
            for (i, &token) in tokens.iter().enumerate() {
                logits = model.forward_token(slot, epoch, token, start_pos + i)?;
            }
            return Ok(infer_plan::sample_token(&logits, params, position));
        }
        let _ = (slot, epoch, start_pos, params, position);
        bail!(
            "Vulkan executor has no model loaded{}",
            if cfg!(feature = "vulkan") {
                " (Qwen3 numeric forward is pending shader/residency bring-up)"
            } else {
                " (built without the `vulkan` feature)"
            }
        )
    }
}

impl BackendExecutor for VulkanExecutor {
    type Inflight = VulkanInflight;

    fn submit(&mut self, plan: &ForwardPlan, kv: &mut dyn KvPool) -> Result<VulkanInflight> {
        if plan.is_idle() {
            return Ok(VulkanInflight::Ready(StepOutput { tokens: Vec::new() }));
        }
        let row_count = plan.prefill_rows.len() + plan.decode_rows.len();
        ensure!(
            row_count == 1,
            "Vulkan executor supports exactly one prefill or decode row, got {row_count}"
        );

        if let Some(row) = plan.prefill_rows.first() {
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
            return Ok(VulkanInflight::Ready(StepOutput {
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
            return Ok(VulkanInflight::Ready(StepOutput {
                tokens: vec![SlotToken {
                    slot: row.slot,
                    token,
                    logprob: None,
                    finish: None,
                }],
            }));
        }
        bail!("Vulkan executor received a non-idle plan with no rows")
    }

    fn poll(&mut self, inflight: VulkanInflight) -> Result<PollResult<VulkanInflight>> {
        match inflight {
            VulkanInflight::Ready(output) => Ok(PollResult::Ready(output)),
        }
    }

    fn model_stop_token_ids(&self) -> Vec<u32> {
        self.stop_tokens.clone()
    }
}

pub fn load_qwen3_gguf(
    path: impl AsRef<std::path::Path>,
    num_slots: usize,
    max_seq_len: usize,
) -> Result<(VulkanExecutor, VulkanKvPool)> {
    ensure!(num_slots > 0, "Vulkan load requires at least one slot");
    ensure!(max_seq_len > 0, "Vulkan load requires max_seq_len > 0");
    let gguf = infer_hip::gguf::GgufFile::open(&path)?;
    let _ = gguf.metadata();
    #[cfg(feature = "vulkan")]
    {
        let _ = gguf;
        bail!(
            "Vulkan Qwen3 GGUF load is pending Qwen3 config mapping and residency upload; \
             host GGUF parse succeeded"
        )
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (num_slots, max_seq_len);
        Err(anyhow::anyhow!(
            "Vulkan backend not compiled: rebuild with --features vulkan \
             (host stage validated: GGUF parsed)"
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

    fn pool() -> VulkanKvPool {
        VulkanKvPool::new(2, 8, DEFAULT_PAGE_SIZE, 256)
    }

    #[test]
    fn idle_plan_returns_empty_output() {
        let mut exec = VulkanExecutor::unloaded();
        let mut pool = pool();
        let inflight = exec.submit(&ForwardPlan::idle(), &mut pool).unwrap();
        match exec.poll(inflight).unwrap() {
            PollResult::Ready(out) => assert!(out.tokens.is_empty()),
            PollResult::NotReady(_) => panic!("P2 resolves synchronously"),
        }
    }

    #[test]
    fn unloaded_executor_errors_clearly() {
        let mut exec = VulkanExecutor::unloaded();
        let mut pool = pool();
        let err = exec.submit(&one_row_plan(false), &mut pool).unwrap_err();
        assert!(err.to_string().contains("no model loaded"), "{err}");
        let err = exec.submit(&one_row_plan(true), &mut pool).unwrap_err();
        assert!(err.to_string().contains("no model loaded"), "{err}");
    }
}
