//! Offline speculative-decoding draft training.
//!
//! Separate from `train` (OPD-only) because the two share only the autograd
//! substrate: data recipe, loss, block-mask construction and eval all differ.
//!
//! Built: block construction, the attention plan, the objective, and the
//! artifact layer. Not built: the draft backbone forward/backward and the
//! masked-tile attention kernel [`mask::Plan::partial`] calls for — both GPU
//! work, neither verifiable without CUDA.

#[path = "block.rs"]
pub mod block;
#[path = "iso_spectrum.rs"]
pub mod iso_spectrum;
#[path = "loss.rs"]
pub mod loss;
#[path = "markov_head.rs"]
pub mod markov_head;
#[path = "mask.rs"]
pub mod mask;
