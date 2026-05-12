//! DeepSeek V4 prefill scaffold.
//!
//! Phase 2A.1 keeps the same CUDA smoke contract as decode: validate the
//! scheduler-visible surface, expose device-resident logits with the expected
//! vocab shape, and advance sequence state. Real V4 embedding, hybrid
//! attention, routed MoE, and LM-head projection remain behind the Phase 2A
//! kernel work.

#[cfg(feature = "cuda")]
use anyhow::{Result, ensure};

#[cfg(feature = "cuda")]
use super::state::DeepseekState;
#[cfg(feature = "cuda")]
use super::weights::DeepseekModel;
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::DeviceVec;

/// Pre-allocated scratch for a batched prefill launch. Empty until the kernel
/// surface is real; the scheduler wires `Qwen3PrefillContext` for the same
/// purpose on the Qwen3 path.
///
/// Public so the `ModelForward::PrefillContext` associated type (a `pub`
/// surface on the trait) does not leak a private name.
#[cfg(feature = "cuda")]
pub struct DeepseekPrefillContext;

#[cfg(feature = "cuda")]
impl DeepseekPrefillContext {
    pub(super) fn new() -> Self {
        Self
    }
}

#[cfg(feature = "cuda")]
impl DeepseekModel {
    /// Run prefill for a single sequence into the contiguous KV cache, then
    /// expose the resulting logits via `state.base.prefill_logits`.
    pub(super) fn prefill_one(&self, tokens: &[u32], state: &mut DeepseekState) -> Result<()> {
        self.validate_phase0_sw_decode_scope()?;
        ensure!(
            !tokens.is_empty(),
            "DeepSeek V4 prefill requires at least one token"
        );
        ensure!(
            state.base.kv_cache.len().saturating_add(tokens.len())
                <= state.base.kv_cache.max_seq_len(),
            "DeepSeek V4 prefill would exceed max_seq_len: current={} incoming={} max={}",
            state.base.kv_cache.len(),
            tokens.len(),
            state.base.kv_cache.max_seq_len()
        );
        for &token in tokens {
            ensure!(
                (token as usize) < self.config.vocab_size,
                "DeepSeek V4 token id {token} exceeds vocab_size {}",
                self.config.vocab_size
            );
        }

        // Phase 2A.1 only promises a scheduler-safe CUDA prefill surface. The
        // logits vector is intentionally all-zero until the V4 weight loader and
        // real attention/MoE kernels are wired.
        state.base.prefill_logits = Some(
            DeviceVec::zeros(&self.ctx, self.config.vocab_size)?
                .with_label("dsv4_phase2a1_prefill_logits"),
        );
        state.base.kv_cache.advance_seq_len(tokens.len());
        Ok(())
    }
}
