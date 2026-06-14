//! GGUF host substrate — neutral leaf shared by GGUF-consuming backends
//! (extracted from `infer-hip`, refactor roadmap tranche R1).
//!
//! Owns GGUF container reading (`gguf` — v2/v3 memmap reader), CPU
//! dequantizers (`dequant` — llama.cpp ports), and per-arch GGUF →
//! spec-config mappers (`deepseek4`). Mappers may depend on `*-spec`
//! crates; `*-spec` crates stay dependency-pure and never depend on this
//! crate. Model *forward* code stays in the backends.

pub mod deepseek4;
pub mod dequant;
pub mod gguf;
pub mod tokenizer;
