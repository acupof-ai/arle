//! Offline speculative-decoding draft training.
//!
//! Separate from `train` (OPD-only) because the two share only the autograd
//! substrate: data recipe, loss, block-mask construction and eval all differ.
//!
//! Block construction, the attention mask, the draft backbone, and the DSpark
//! objective. The backbone is the eager autograd op chain, so its backward is
//! the tape's and it runs on any backend.

#[path = "backbone.rs"]
pub mod backbone;
#[path = "block.rs"]
pub mod block;
#[path = "data.rs"]
pub mod data;
#[path = "loss.rs"]
pub mod loss;
#[path = "markov_head.rs"]
pub mod markov_head;
#[path = "mask.rs"]
pub mod mask;
#[path = "trainer.rs"]
pub mod trainer;
