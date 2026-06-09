//! DSv4 native DeepEP transport glue.
//!
//! This module owns the torch-free DeepEP `Buffer` lifecycle plus the small
//! NCCL byte all-gather used to exchange CUDA IPC handles. The MoE math stays in
//! `moe.rs`; this layer only dispatches token rows to expert-owner ranks and
//! combines local expert outputs back to the original token rows.

#![cfg(all(feature = "cuda", feature = "deepep"))]

use std::sync::Mutex;

use anyhow::{Result, anyhow, ensure};
use cuda_kernels::prelude::DeviceContext;

use crate::tp::TpRuntime;

// `world_size` and `buffer` are written by `maybe_boot` and held for the
// upcoming SGLang low-latency DeepEP path (which will reintroduce dispatch /
// combine readers); the old normal-mode forward that read them was deleted.
#[allow(dead_code)]
pub(crate) struct DeepEpTransport {
    world_size: u32,
    buffer: Mutex<deepep_sys::Buffer>,
}

impl DeepEpTransport {
    pub(crate) fn should_enable_from_env() -> bool {
        matches!(
            std::env::var("ARLE_DSV4_MOE_TRANSPORT").as_deref(),
            Ok("deepep" | "native-deepep" | "native_deepep")
        ) || matches!(
            std::env::var("ARLE_DSV4_MOE_BACKEND").as_deref(),
            Ok("deepep" | "native-deepep" | "native_deepep")
        )
    }

    pub(crate) fn maybe_boot(ctx: &DeviceContext, tp: &TpRuntime) -> Result<Option<Self>> {
        if !Self::should_enable_from_env() {
            return Ok(None);
        }
        ensure!(deepep_sys::is_native(), "deepep-sys was built as a stub");
        let cfg = tp.config();
        ensure!(
            cfg.world_size >= 2,
            "native DeepEP requires world_size >= 2, got {}",
            cfg.world_size
        );
        let rank = u32::try_from(cfg.rank)?;
        let world_size = u32::try_from(cfg.world_size)?;
        let mut buffer = deepep_sys::Buffer::new(rank, world_size)
            .map_err(|e| anyhow!("DeepEP Buffer::new failed: {e}"))?;
        let (local_handle, local_device_id) = buffer
            .local_ipc_handle()
            .map_err(|e| anyhow!("DeepEP local_ipc_handle failed: {e}"))?;

        let gathered_handles = tp
            .all_gather_bytes(ctx, &local_handle, deepep_sys::IPC_HANDLE_BYTES)
            .map_err(|e| anyhow!("DeepEP IPC handle all_gather failed: {e}"))?;
        let gathered_ids = tp
            .all_gather_bytes(ctx, &local_device_id.to_ne_bytes(), 4)
            .map_err(|e| anyhow!("DeepEP device-id all_gather failed: {e}"))?;

        let mut peers = Vec::with_capacity(cfg.world_size);
        for peer in 0..cfg.world_size {
            let h0 = peer * deepep_sys::IPC_HANDLE_BYTES;
            let mut handle = [0u8; deepep_sys::IPC_HANDLE_BYTES];
            handle.copy_from_slice(&gathered_handles[h0..h0 + deepep_sys::IPC_HANDLE_BYTES]);
            let id0 = peer * 4;
            let device_id = u32::from_ne_bytes(gathered_ids[id0..id0 + 4].try_into()?);
            peers.push((handle, device_id));
        }
        buffer
            .sync(&peers)
            .map_err(|e| anyhow!("DeepEP Buffer::sync failed: {e}"))?;
        Ok(Some(Self {
            world_size,
            buffer: Mutex::new(buffer),
        }))
    }

    // Kept for the upcoming SGLang low-latency DeepEP path; the old normal-mode
    // forward that called it was deleted.
    #[allow(dead_code)]
    pub(crate) fn num_sms() -> Result<u32> {
        let value = std::env::var("ARLE_DSV4_DEEPEP_NUM_SMS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(20);
        ensure!(
            value > 0 && value.is_multiple_of(2),
            "DeepEP num_sms must be positive and even, got {value}"
        );
        Ok(value)
    }
}
