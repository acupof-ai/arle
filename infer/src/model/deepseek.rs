//! DeepSeek V4 model scaffold.
//!
//! Phase 0.5 aligns infer-side DeepSeek runtime truth to the actual
//! `DeepseekV4ForCausalLM` checkpoint at `infer/models/dsv4-mini-1B-init/`.
//! Forward kernels remain pending Phase 2A.

#[cfg(feature = "cuda")]
#[path = "deepseek/batch_decode.rs"]
mod batch_decode;
#[path = "deepseek/config.rs"]
mod config;
#[path = "deepseek/forward.rs"]
mod forward;
#[cfg(feature = "cuda")]
#[path = "deepseek/load.rs"]
mod load;
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

pub use crate::deepseek_v4_manifest::DeepseekV4CheckpointManifest;
pub use config::DeepseekRuntimeConfig;
pub use state::DeepseekState;
pub use weights::DeepseekModel;
