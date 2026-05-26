//! ARLE native DeepEP sidecar — Rust host side.
//!
//! Spawns one C++ sidecar child per rank, exchanges CUDA IPC handles via
//! pipes, and dispatches commands (boot → sync → round_trip / dispatch /
//! combine → shutdown) over a binary wire protocol shared with
//! `crates/cuda-kernels/csrc/deepep_sidecar/protocol.hpp`.
//!
//! This module is opt-in: nothing here is reachable until
//! `ARLE_DSV4_MOE_BACKEND=native-deepep` is set AND the sidecar binary
//! was built (requires `ARLE_DEEPEP_DIR` at build time). See
//! `docs/plans/2026-05-26-dsv4-deepep-process-per-rank.md` for the design.

#[path = "deepep_sidecar/protocol.rs"]
pub mod protocol;

#[path = "deepep_sidecar/pool.rs"]
pub mod pool;

pub use pool::{SidecarPool, SidecarPoolConfig};
pub use protocol::{
    BootResponse, CommandId, KMAX_NVL_PEERS, MessageHeader, PROTOCOL_VERSION, RoundTripRequest,
    RoundTripResponse, Status,
};

/// Path to the sidecar binary baked in at build time when `ARLE_DEEPEP_DIR`
/// is set. `None` indicates the sidecar wasn't built — the native-deepep
/// backend cannot be used in this binary.
pub fn baked_binary_path() -> Option<&'static str> {
    option_env!("ARLE_DEEPEP_SIDECAR_PATH")
}
