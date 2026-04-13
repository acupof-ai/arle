//! GLM-4 model: standard dense transformer with GQA, RoPE, and SwiGLU MLP.

#[cfg(feature = "cuda")]
#[path = "glm4/batch_decode.rs"]
mod batch_decode;
#[path = "glm4/config.rs"]
mod config;
#[path = "glm4/decode.rs"]
mod decode;
#[path = "glm4/decode_buffers.rs"]
mod decode_buffers;
#[path = "glm4/forward.rs"]
mod forward;
#[path = "glm4/prefill.rs"]
mod prefill;
#[path = "glm4/weights.rs"]
mod weights;

pub use config::Config;
pub use forward::GLM4State;
pub use weights::GLM4Model;
