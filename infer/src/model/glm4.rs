//! GLM-4 model: standard dense transformer with GQA, RoPE, and SwiGLU MLP.

#[cfg(feature = "cuda")]
mod batch_decode;
mod config;
mod decode;
mod decode_buffers;
mod forward;
mod prefill;
mod weights;

pub use config::Config;
pub use forward::GLM4State;
pub use weights::GLM4Model;
