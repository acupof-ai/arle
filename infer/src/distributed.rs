//! Distributed coordination primitives for multi-GPU / multi-host execution.
//!
//! F0 foundation: TCP rendezvous (`init_method`). NCCL FFI, group coordinators,
//! and parallel-state wiring land in later phases — see
//! [`docs/plans/2026-04-28-single-node-multi-gpu.md`](../../docs/plans/2026-04-28-single-node-multi-gpu.md).

pub mod init_method;

pub use init_method::{RendezvousClient, RendezvousServer, UNIQUE_ID_BYTES};
