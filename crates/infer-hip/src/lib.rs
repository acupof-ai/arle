//! HIP backend host substrate for the AIPC DSv4-Flash 2-bit lane (#76/#77,
//! `docs/plans/2026-06-10-hip-backend-mvp.md` §4 H2/H3).
//!
//! Stage A — GGUF v3 weight reader, CPU dequantizers, GGUF→DSv4 config
//! mapping, residency planner, and DSv4 slot/KV bookkeeping. Everything
//! host-side compiles without the `hip` feature; only device upload and
//! device pool buffers gate on it. No forward / `BackendExecutor` yet
//! (stage B).

pub mod config;
pub mod dequant;
pub mod gguf;
pub mod kv_pool;
pub mod loader;
