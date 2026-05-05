//! DeepSeek Multi-head Latent Attention (MLA) scaffold.
//!
//! Per [`docs/plans/2026-05-01-mla-kernel-design.md`](../../../../../docs/plans/2026-05-01-mla-kernel-design.md),
//! MLA replaces standard MHA / GQA with a low-rank latent KV plus decoupled
//! RoPE per the DeepSeek-V3 paper. The on-disk projection layout is:
//!
//! - `q_a_proj` (+ `q_a_layernorm`, `q_b_proj`) when `q_lora_rank` is set, OR
//!   `q_proj` directly when `q_lora_rank` is `None` (nano / SKU-A path).
//! - `kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj` for the latent KV.
//! - `o_proj` for the output projection.
//!
//! Kernels (CUDA + Metal) land in `infer/src/ops/attention/mla.rs`. This file
//! holds the weight container and the forward stub used by `forward.rs` /
//! `prefill.rs`. Until the kernel ships every forward path returns
//! `todo!("MLA kernel — see docs/plans/2026-05-01-mla-kernel-design.md")`.

#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use cuda_kernels::prelude::{DeviceMatrix, DeviceVec, HiddenStates};

/// Weights for one MLA attention block.
///
/// `q_proj` is `Some` when `config.q_lora_rank` is `None` (the direct-Q path
/// used by the nano fixture); `q_a_proj` / `q_a_layernorm` / `q_b_proj` are
/// `Some` when `q_lora_rank` is set (the standard DeepSeek-V3 layout). The
/// loader enforces "exactly one of those two paths" — there is no silent
/// fallback. See substrate plan §6.1.
#[cfg(feature = "cuda")]
pub(super) struct MlaAttention {
    /// Direct Q projection (used when `q_lora_rank` is `None`).
    pub(super) q_proj: Option<DeviceMatrix>,
    /// Down-projection into the q latent (used when `q_lora_rank` is `Some`).
    pub(super) q_a_proj: Option<DeviceMatrix>,
    /// RMSNorm gain on the q latent (paired with `q_a_proj`).
    pub(super) q_a_layernorm: Option<DeviceVec>,
    /// Up-projection from the q latent into per-head Q (paired with `q_a_proj`).
    pub(super) q_b_proj: Option<DeviceMatrix>,
    /// Down-projection into the kv latent + MQA RoPE (always present).
    pub(super) kv_a_proj_with_mqa: DeviceMatrix,
    /// RMSNorm gain on the kv latent.
    pub(super) kv_a_layernorm: DeviceVec,
    /// Up-projection from the kv latent into per-head K-nope and V.
    pub(super) kv_b_proj: DeviceMatrix,
    /// Output projection back to `hidden_size`.
    pub(super) o_proj: DeviceMatrix,
}

#[cfg(feature = "cuda")]
impl MlaAttention {
    /// True when the loader chose the direct-Q path (`q_lora_rank == None`).
    pub(super) fn uses_direct_q(&self) -> bool {
        self.q_proj.is_some()
    }

    /// True when the loader chose the LoRA-Q path (`q_lora_rank == Some(_)`).
    pub(super) fn uses_lora_q(&self) -> bool {
        self.q_a_proj.is_some()
    }

    /// Run MLA prefill for a packed `[seq, hidden]` row block.
    ///
    /// Stub until the MLA prefill kernel lands. The `_` prefixes silence
    /// unused-arg warnings while the body is `todo!()`.
    pub(super) fn forward_prefill(
        &self,
        _hidden: &HiddenStates,
        _start_pos: usize,
    ) -> Result<HiddenStates> {
        todo!("MLA kernel — see docs/plans/2026-05-01-mla-kernel-design.md")
    }

    /// Run MLA decode for a single token using the latent KV cache.
    pub(super) fn forward_decode(&self, _token_pos: usize) -> Result<()> {
        todo!("MLA kernel — see docs/plans/2026-05-01-mla-kernel-design.md")
    }
}
