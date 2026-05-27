//! Native DeepEP boot path — phase B-3 of the multiproc-serve pivot.
//!
//! Wraps `deepep_sys::Buffer` with the cross-rank handle exchange that
//! `Buffer::sync` needs. The handle exchange runs over the existing
//! EP NCCL group (created during model construction); same comm group
//! used for forward TP/EP collectives, so no extra rendezvous required.
//!
//! Lifetime model: one `NativeDeepEp` per scheduler / per process. Held
//! behind `Arc<Mutex<>>` so the forward path (commit B-3.2) can grab the
//! `Buffer` briefly per dispatch / combine call.
//!
//! Only available when `deepep-sys` was built natively (env
//! `ARLE_DEEPEP_DIR` set at build time). When the deep-ep crate is in
//! stub mode, `NativeDeepEp::boot` returns an error pointing at the
//! build-time flag.

#![cfg(all(feature = "cuda", feature = "nccl"))]

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow};

use crate::distributed::nccl::NcclGroup;
pub use deepep_sys::{Buffer, CombineParams, DispatchParams};

pub struct NativeDeepEp {
    pub rank: u32,
    pub world_size: u32,
    pub buffer: Arc<Mutex<Buffer>>,
}

impl std::fmt::Debug for NativeDeepEp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeDeepEp")
            .field("rank", &self.rank)
            .field("world_size", &self.world_size)
            .finish_non_exhaustive()
    }
}

impl NativeDeepEp {
    /// Boot a `Buffer` at this rank, exchange IPC handles with peers
    /// via the provided NCCL group's all_gather_bytes, then `Buffer::
    /// sync` to populate peer pointer arrays + run the cross-rank
    /// `intranode::barrier`.
    ///
    /// The NCCL group must already be initialized with the same rank /
    /// world_size as is being passed here; this constructor does not
    /// re-do the TCP rendezvous.
    pub fn boot(rank: u32, world_size: u32, nccl: &NcclGroup) -> Result<Arc<Self>> {
        if !deepep_sys::is_native() {
            return Err(anyhow!(
                "deepep-sys was built in stub mode — set ARLE_DEEPEP_DIR \
                 at build time to enable native DeepEP. See \
                 docs/plans/2026-05-27-multiproc-serve-pivot.md §B-2."
            ));
        }
        if nccl.rank != rank as usize || nccl.world_size != world_size as usize {
            return Err(anyhow!(
                "NCCL group (rank={}, ws={}) does not match NativeDeepEp request \
                 ({rank}, {world_size})",
                nccl.rank,
                nccl.world_size
            ));
        }

        let mut buffer = Buffer::new(rank, world_size).context("Buffer::new")?;
        let (local_handle, local_device_id) = buffer
            .local_ipc_handle()
            .context("Buffer::local_ipc_handle")?;

        // 1. All-gather IPC handles (64 bytes per rank).
        let gathered_handles = nccl
            .all_gather_bytes(&local_handle, deepep_sys::IPC_HANDLE_BYTES)
            .context("all_gather_bytes (deepep IPC handles)")?;
        // 2. All-gather device ids (4 bytes per rank, little-endian u32).
        let gathered_ids = nccl
            .all_gather_bytes(&local_device_id.to_le_bytes(), 4)
            .context("all_gather_bytes (deepep device ids)")?;
        let mut peers: Vec<([u8; deepep_sys::IPC_HANDLE_BYTES], u32)> =
            Vec::with_capacity(world_size as usize);
        for r in 0..world_size as usize {
            let h_start = r * deepep_sys::IPC_HANDLE_BYTES;
            let mut h = [0u8; deepep_sys::IPC_HANDLE_BYTES];
            h.copy_from_slice(&gathered_handles[h_start..h_start + deepep_sys::IPC_HANDLE_BYTES]);
            let id_start = r * 4;
            let did = u32::from_le_bytes(
                gathered_ids[id_start..id_start + 4]
                    .try_into()
                    .expect("4-byte slice"),
            );
            peers.push((h, did));
        }

        buffer.sync(&peers).context("Buffer::sync")?;
        log::info!(
            "[native-deepep] rank {rank}/{world_size} booted (device_id={local_device_id}, \
             peer_handles={})",
            peers.len()
        );

        Ok(Arc::new(Self {
            rank,
            world_size,
            buffer: Arc::new(Mutex::new(buffer)),
        }))
    }
}
