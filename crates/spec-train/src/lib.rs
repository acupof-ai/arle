//! Offline speculative-decoding draft training.
//!
//! Separate from `train` (OPD-only) because the two share only the autograd
//! substrate: data recipe, loss, block-mask construction and eval all differ.
//!
//! Currently the artifact layer alone — the DSpark Markov head and the
//! fixed-spectrum (ISO) frames. The trainer is not built.

#[path = "iso_spectrum.rs"]
pub mod iso_spectrum;
#[path = "markov_head.rs"]
pub mod markov_head;
