//! GGUF host substrate — neutral leaf shared by GGUF-consuming backends.
//!
//! Mappers may depend on `*-spec` crates; `*-spec` crates stay dependency-pure
//! and never depend on this crate. Model *forward* code stays in the backends.

pub mod deepseek4;
pub mod dequant;
pub mod gguf;
pub mod safetensors;
pub mod tokenizer;
