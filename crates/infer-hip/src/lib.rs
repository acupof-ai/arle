//! HIP backend host substrate for the AIPC DSv4-Flash 2-bit lane (#76/#77,
//! §4 H2/H3).
//!
//! Stage A — residency planner and DSv4 slot/KV bookkeeping over the
//! `infer-gguf` host substrate (GGUF reader, CPU dequantizers, GGUF→DSv4
//! config mapping). Stage B — DSv4 forward orchestration (`model`) +
//! `BackendExecutor` seam (`executor`). Everything host-side compiles
//! without the `hip` feature; device buffers, kernel launches, and the
//! loaded model gate on it (correctness on the AMD box is pending-remote).

pub mod executor;
pub mod kv_pool;
pub mod loader;
pub mod model;

pub use executor::{HipDsv4Executor, load_dsv4_gguf};
pub use kv_pool::HipKvPool;
