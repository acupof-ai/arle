//! Context-parallel training launcher: SPMD coordinator that spawns one worker
//! process per CP rank. Sequence-parallel training shards the sequence across N
//! GPUs (per-card activation memory O(seq/N)) — the only way to fit the 256K OPD
//! writeback, which peaks past one card at ~seq 49152.
//!
//! Simpler than `serve_multiproc`: no engine-ready barrier, no lockstep relay —
//! NCCL collectives are the only cross-rank synchronization. The coordinator mints
//! the NCCL unique id, re-execs `current_exe()` once per rank with per-rank env
//! (rank/size/device/uid), waits for the first exit, and kills survivors so one
//! rank's crash tears the group down instead of wedging NCCL.

#![cfg(all(unix, feature = "cuda"))]

#[cfg(feature = "nccl")]
use std::time::Duration;

#[cfg(feature = "nccl")]
use anyhow::Context;
use anyhow::Result;

/// If this process is a spawned CP worker (`ARLE_TRAIN_CP_RANK` set), install a
/// `[cpN]` stderr prefix and return true. Unlike serve's worker_entry this does
/// NOT short-circuit dispatch — the child flows through clap into the normal
/// agent-opd handler as its rank; `build_opd_store` reads the CP env to build the
/// NCCL backend.
pub(crate) fn install_cp_worker_logger() -> bool {
    let Ok(rank) = std::env::var("ARLE_TRAIN_CP_RANK") else {
        return false;
    };
    infer_util::logging::init_stderr_with_prefix("info", &format!("[cp{rank}] "));
    true
}

/// Coordinator side: if `cp_size > 1` and this is NOT already a worker, mint the
/// NCCL unique id, spawn `cp_size` rank children (per-rank env: rank/size/device +
/// the minted uid), wait for the first to exit, and return `Ok(true)` so the
/// caller returns without running training itself. `Ok(false)` = run in-process
/// (single-card, or this process IS a spawned worker).
pub(crate) fn maybe_spawn_cp_ranks_and_wait(cp_size: usize, cp_devices: &[usize]) -> Result<bool> {
    if cp_size <= 1 || std::env::var("ARLE_TRAIN_CP_RANK").is_ok() {
        return Ok(false);
    }

    // Mint + publish the NCCL unique id once; children inherit it via env.
    #[cfg(feature = "nccl")]
    let uid_hex = infer_api::mint_nccl_unique_id_hex().context("mint NCCL unique id")?;
    #[cfg(not(feature = "nccl"))]
    {
        let _ = cp_devices;
        anyhow::bail!(
            "context parallelism (--cp-size {cp_size}) requires the nccl feature; \
             rebuild with --features cuda,nccl"
        );
    }

    #[cfg(feature = "nccl")]
    {
        let exe = std::env::current_exe().context("current_exe")?;
        let mut children: Vec<(usize, std::process::Child)> = Vec::with_capacity(cp_size);
        for rank in 0..cp_size {
            let device = cp_devices.get(rank).copied().unwrap_or(rank);
            let mut cmd = std::process::Command::new(&exe);
            for arg in std::env::args().skip(1) {
                cmd.arg(arg);
            }
            cmd.env("ARLE_TRAIN_CP_RANK", rank.to_string());
            cmd.env("ARLE_TRAIN_CP_SIZE", cp_size.to_string());
            cmd.env("INFER_CUDA_DEVICE", device.to_string());
            cmd.env("INFER_NCCL_UNIQUE_ID", &uid_hex);
            let child = cmd
                .spawn()
                .with_context(|| format!("spawn CP rank {rank}"))?;
            log::info!("spawned CP rank {rank} pid={} device={device}", child.id());
            children.push((rank, child));
        }

        // Wait for the first exit; on any exit (success or crash) kill the rest so
        // a single rank's failure can't leave the others blocked in a collective.
        let (exit_rank, code) = loop {
            if let Some(hit) = children
                .iter_mut()
                .find_map(|(r, c)| c.try_wait().ok().flatten().map(|s| (*r, s.code())))
            {
                break hit;
            }
            std::thread::sleep(Duration::from_millis(100));
        };
        log::info!("CP rank {exit_rank} exited (code={code:?}); tearing down group");
        for (rank, child) in &mut children {
            if *rank != exit_rank {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
        match code {
            Some(0) => Ok(true),
            other => anyhow::bail!("CP rank {exit_rank} exited with {other:?}"),
        }
    }
}
