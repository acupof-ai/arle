//! Qwen3.5 model: mixed full attention + linear attention transformer.

#[cfg(feature = "cuda")]
#[path = "qwen35/batch_decode.rs"]
mod batch_decode;
#[path = "qwen35/config.rs"]
pub(crate) mod config;
#[path = "qwen35/decode_buffers.rs"]
mod decode_buffers;
#[path = "qwen35/forward.rs"]
mod forward;
#[path = "qwen35/prefill.rs"]
mod prefill;
#[path = "qwen35/prefill_buffers.rs"]
pub mod prefill_buffers;
#[path = "qwen35/recurrent_state.rs"]
mod recurrent_state;
#[path = "qwen35/single_token_buffers.rs"]
mod single_token_buffers;
#[path = "qwen35/weights.rs"]
mod weights;

pub use forward::Qwen35State;
pub use weights::Qwen35Model;
