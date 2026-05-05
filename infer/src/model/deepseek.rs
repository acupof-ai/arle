//! DeepSeek model: MLA latent attention + (optional) DeepSeekMoE.
//!
//! Scaffold landed 2026-05-05; kernels still pending. See
//! [`docs/plans/2026-05-05-deepseek-v4-small-substrate.md`](../../../../docs/plans/2026-05-05-deepseek-v4-small-substrate.md)
//! for the project plan and
//! [`docs/plans/2026-05-01-mla-kernel-design.md`](../../../../docs/plans/2026-05-01-mla-kernel-design.md)
//! for the MLA kernel design these stubs are waiting on.

#[cfg(feature = "cuda")]
#[path = "deepseek/batch_decode.rs"]
mod batch_decode;
#[path = "deepseek/config.rs"]
mod config;
#[path = "deepseek/forward.rs"]
mod forward;
#[path = "deepseek/mla.rs"]
mod mla;
#[path = "deepseek/mlp.rs"]
mod mlp;
#[path = "deepseek/prefill.rs"]
mod prefill;
#[path = "deepseek/state.rs"]
mod state;
#[path = "deepseek/weights.rs"]
mod weights;

pub use config::DeepseekRuntimeConfig;
pub use state::DeepseekState;
pub use weights::DeepseekModel;
