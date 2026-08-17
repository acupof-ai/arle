//! DeepSeek-V4-Flash FP8 model: weight structs, MLA KV arena, EP-aware loader.
//!
//! Multi-GPU only (256 FP8 experts + MLA sharding don't fit one GPU);
//! `ExpertSplit::single` is the dev/typecheck fallback.

use anyhow::{Result, anyhow, ensure};
use cuda_kernels::ffi;
use cuda_kernels::prelude::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};
use cuda_kernels::tensor::Dsv4Fp8DeepGemmWeightCache;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use deepseek_spec::{DeepSeekV4AttentionMode, DeepSeekV4Config, DeepSeekV4MoeRoutingKind};
use infer_moe::MoeConfig;
use infer_plan::SamplingParams;

use crate::moe_config::ExpertSplit;

#[path = "dsv4/budget.rs"]
mod budget;
#[path = "dsv4/decode_batch.rs"]
mod decode_batch;
#[path = "dsv4/dspark.rs"]
pub(crate) mod dspark;
#[path = "dsv4/forward_state.rs"]
mod forward_state;
#[path = "dsv4/head.rs"]
mod head;
#[path = "dsv4/layer_block.rs"]
mod layer_block;
#[path = "dsv4/load.rs"]
mod load;
#[path = "dsv4/mega_moe.rs"]
mod mega_moe;
#[path = "dsv4/mtp.rs"]
mod mtp;
#[path = "dsv4/prefill.rs"]
mod prefill;
#[path = "dsv4/slot.rs"]
mod slot;
#[path = "dsv4/slot_image.rs"]
mod slot_image;
#[path = "dsv4/spec_verify.rs"]
mod spec_verify;
#[path = "dsv4/weights.rs"]
mod weights;

pub(crate) use budget::Dsv4MlaKvArena;
pub(crate) use forward_state::*;
pub(crate) use load::load_dspark_draft;
#[cfg(all(feature = "cuda", feature = "nccl"))]
pub(crate) use mega_moe::Dsv4MegaMoeTransport;
pub(crate) use mtp::*;
pub(crate) use slot::*;
pub(crate) use slot_image::*;
pub(crate) use spec_verify::*;
pub(crate) use weights::*;

/// Loaded DSv4-Flash model for one TP/EP rank.
pub(crate) struct Dsv4Model {
    pub ctx: DeviceContext,
    pub config: DeepSeekV4Config,
    pub moe_config: MoeConfig,
    pub split: ExpertSplit,
    pub kv_arena: Dsv4MlaKvArena,
    pub embed_tokens: DeviceMatrix,
    pub lm_head: DeviceMatrix,
    pub layers: Vec<Dsv4Layer>,
    pub norm: DeviceVec,
    /// Head hyper-connection: folds the wide residual stream back to one hidden
    /// row before the final RMSNorm + lm_head projection.
    pub head_hc: Dsv4HyperConnection,
    pub mtp: Option<Dsv4MtpLayer>,
    /// Resolved at construction from `--spec-type mtp` / `dspark`. Per-slot
    /// construction reads this so the MTP-head load and the per-slot rollback
    /// snapshots agree on one decision.
    pub spec_decode_on: bool,
    pub tp: crate::tp::TpRuntime,
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    pub(crate) mega_moe: Option<Dsv4MegaMoeTransport>,
    #[cfg(feature = "deepep")]
    pub deepep: Option<crate::deepep::DeepEpTransport>,
}

impl std::fmt::Debug for Dsv4Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dsv4Model")
            .field("layers", &self.layers.len())
            .field("hidden_size", &self.config.hidden_size)
            .field("heads", &self.config.num_attention_heads)
            .field("experts", &self.config.n_routed_experts)
            .field("experts_per_rank", &self.split.experts_per_rank)
            .field("kv_bytes_per_token", &self.kv_arena.bytes_per_token)
            .field("mtp_loaded", &self.mtp.is_some())
            .finish()
    }
}
