//! HIP backend host substrate for the AIPC DSv4-Flash 2-bit lane (#76/#77,
//! `docs/plans/2026-06-10-hip-backend-mvp.md` §4 H2/H3).
//!
//! Stage A — GGUF v3 weight reader, CPU dequantizers, GGUF→DSv4 config
//! mapping, residency planner, and DSv4 slot/KV bookkeeping. Stage B —
//! DSv4 forward orchestration (`model`) + `BackendExecutor` seam
//! (`executor`). Everything host-side compiles without the `hip` feature;
//! device buffers, kernel launches, and the loaded model gate on it
//! (correctness on the AMD box is pending-remote).

pub mod config;
pub mod dequant;
pub mod executor;
pub mod gguf;
pub mod kv_pool;
pub mod loader;
pub mod model;

pub use executor::{HipDsv4Executor, load_dsv4_gguf};
pub use kv_pool::HipKvPool;
