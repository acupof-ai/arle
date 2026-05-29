//! Infer-runtime student rollout (OPD Phase P1 bring-up).
//!
//! Mirrors [`crate::teacher_infer::InferTeacher`]: holds an in-process
//! `LoadedInferenceEngine` and drives it via `forward_token_logits`. The
//! student differs from the teacher only in that its LoRA weights update every
//! training step — but that per-step sync is **P2**. This module is the
//! zero-LoRA (step-0 == base) bring-up + measurement path only; it contains no
//! LoRA-sync machinery.
//!
//! `decode_next_token` greedily picks the argmax over the **last position**'s
//! logits. Host argmax is sufficient for the bring-up smoke; the device/D2D
//! bridge used by the teacher's KL path is not needed because the rollout loop
//! consumes only the token sequence (see plan
//! `docs/plans/2026-05-29-opd-student-rollout-via-infer.md`).

#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use anyhow::{Result, anyhow, bail};
#[cfg(feature = "cuda")]
use autograd::Backend;
#[cfg(feature = "cuda")]
use infer::server_engine::LoadedInferenceEngine;

#[cfg(feature = "cuda")]
pub struct InferStudent {
    engine: Arc<Mutex<LoadedInferenceEngine>>,
    train_backend: Arc<dyn Backend>,
    vocab_size: usize,
}

#[cfg(feature = "cuda")]
impl InferStudent {
    pub fn new(
        engine: Arc<Mutex<LoadedInferenceEngine>>,
        train_backend: Arc<dyn Backend>,
        vocab_size: usize,
    ) -> Self {
        Self {
            engine,
            train_backend,
            vocab_size,
        }
    }

    pub fn engine(&self) -> &Arc<Mutex<LoadedInferenceEngine>> {
        &self.engine
    }

    pub fn train_backend(&self) -> &Arc<dyn Backend> {
        &self.train_backend
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Run a single greedy forward over `input_ids` (with absolute `positions`)
    /// and return the argmax token over the **last** position's logits.
    ///
    /// The infer engine's `forward_token_logits` is stateless from the caller's
    /// POV: it accepts the full token sequence + contiguous positions each call
    /// (matching how `InferTeacher` drives it). The rollout loop therefore
    /// passes the growing full sequence with `positions = 0..len` each step.
    pub fn decode_next_token(&self, input_ids: &[u32], positions: &[u32]) -> Result<u32> {
        if input_ids.is_empty() {
            bail!("InferStudent requires a non-empty token sequence");
        }
        if input_ids.len() != positions.len() {
            bail!(
                "InferStudent token/position length mismatch: tokens={} positions={}",
                input_ids.len(),
                positions.len()
            );
        }

        let raw_logits = {
            let engine = self
                .engine
                .lock()
                .map_err(|err| anyhow!("LoadedInferenceEngine lock poisoned: {err}"))?;
            engine.forward_token_logits(input_ids, positions)?
        };

        if raw_logits.vocab_size() != self.vocab_size {
            bail!(
                "InferStudent vocab mismatch: raw logits vocab={}, configured vocab={}",
                raw_logits.vocab_size(),
                self.vocab_size
            );
        }
        if raw_logits.seq_len() != input_ids.len() {
            bail!(
                "InferStudent seq_len mismatch: raw logits seq_len={}, input token len={}",
                raw_logits.seq_len(),
                input_ids.len()
            );
        }

        // Host argmax over the last position. `to_host_f32` materializes the
        // full [seq_len, vocab_size] block (mirrors teacher BF16->F32 handling
        // via the engine's own conversion); we only scan the last row.
        let host = raw_logits.to_host_f32()?;
        let vocab = raw_logits.vocab_size();
        let seq_len = raw_logits.seq_len();
        let last_row = &host[(seq_len - 1) * vocab..seq_len * vocab];
        let token = argmax(last_row)?;
        Ok(token as u32)
    }
}

#[cfg(feature = "cuda")]
fn argmax(logits: &[f32]) -> Result<usize> {
    if logits.is_empty() {
        bail!("argmax over empty logits");
    }
    let mut best_idx = 0usize;
    let mut best_val = logits[0];
    for (idx, &val) in logits.iter().enumerate().skip(1) {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    Ok(best_idx)
}
