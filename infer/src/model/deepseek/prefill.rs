//! DeepSeek prefill scaffold.
//!
//! Mirrors `qwen3::prefill` but every entrypoint is a `todo!()` stub. The
//! prefill path will read packed token activations, run MLA per layer (latent
//! KV with decoupled RoPE), then dense MLP, and finally project to logits.

#[cfg(feature = "cuda")]
use anyhow::Result;

#[cfg(feature = "cuda")]
use super::state::DeepseekState;
#[cfg(feature = "cuda")]
use super::weights::DeepseekModel;

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
    pub(super) fn prefill_one(&self, _tokens: &[u32], _state: &mut DeepseekState) -> Result<()> {
        todo!("MLA kernel — see docs/plans/2026-05-01-mla-kernel-design.md")
    }
}
