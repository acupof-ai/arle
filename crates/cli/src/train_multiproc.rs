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
use std::io::{BufRead, BufReader};
#[cfg(feature = "nccl")]
use std::sync::Arc;
#[cfg(feature = "nccl")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "nccl")]
use std::time::Duration;

#[cfg(feature = "nccl")]
use anyhow::Context;
use anyhow::Result;

/// If this process is a spawned CP/DP worker (`ARLE_TRAIN_CP_RANK` or
/// `ARLE_TRAIN_DP_RANK` set), install a `[cpN]`/`[dpN]` stderr prefix and return
/// true. Unlike serve's worker_entry this does NOT short-circuit dispatch — the
/// child flows through clap into the normal agent-opd handler as its rank;
/// `build_opd_store` reads the CP/DP env to build the NCCL backend.
pub(crate) fn install_cp_worker_logger() -> bool {
    for (env, tag) in [("ARLE_TRAIN_CP_RANK", "cp"), ("ARLE_TRAIN_DP_RANK", "dp")] {
        if let Ok(rank) = std::env::var(env) {
            infer_util::logging::init_stderr_with_prefix("info", &format!("[{tag}{rank}] "));
            return true;
        }
    }
    false
}

/// Coordinator side: if `size > 1` on the `axis` ("CP"/"DP") and this is NOT
/// already a worker, mint the NCCL unique id, spawn `size` rank children (per-rank
/// env `ARLE_TRAIN_{axis}_RANK`/`_SIZE` + device + the minted uid), wait for the
/// first to exit, and return `Ok(true)` so the caller returns without training
/// itself. `Ok(false)` = run in-process (single-card, or this IS a spawned worker).
/// CP and DP are mutually exclusive here (combined needs ncclCommSplit subgroups,
/// rejected in `build_opd_store`).
pub(crate) fn maybe_spawn_ranks_and_wait(
    axis: &str,
    size: usize,
    devices: &[usize],
) -> Result<bool> {
    if size <= 1
        || std::env::var("ARLE_TRAIN_CP_RANK").is_ok()
        || std::env::var("ARLE_TRAIN_DP_RANK").is_ok()
    {
        return Ok(false);
    }

    // Mint + publish the NCCL unique id once; children inherit it via env.
    #[cfg(feature = "nccl")]
    let uid_hex = infer_api::mint_nccl_unique_id_hex().context("mint NCCL unique id")?;
    #[cfg(not(feature = "nccl"))]
    {
        let _ = devices;
        anyhow::bail!(
            "{axis} parallelism (--{}-size {size}) requires the nccl feature; \
             rebuild with --features cuda,nccl",
            axis.to_lowercase()
        );
    }

    #[cfg(feature = "nccl")]
    {
        let rank_env = format!("ARLE_TRAIN_{axis}_RANK");
        let size_env = format!("ARLE_TRAIN_{axis}_SIZE");
        let exe = std::env::current_exe().context("current_exe")?;
        let mut children: Vec<(usize, std::process::Child)> = Vec::with_capacity(size);
        // Monotonic "last time any rank emitted a log line", in ms since `start`.
        // A wedged collective goes silent on every rank (all block on the hung one),
        // so any child stderr line is a group-liveness signal. Forwarder threads bump
        // this; the wait loop measures IDLE time against it, not total wall-clock.
        let last_progress_ms = Arc::new(AtomicU64::new(0));
        let start = std::time::Instant::now();
        let mut forwarders: Vec<std::thread::JoinHandle<()>> = Vec::with_capacity(size);
        for rank in 0..size {
            let device = devices.get(rank).copied().unwrap_or(rank);
            let mut cmd = std::process::Command::new(&exe);
            for arg in std::env::args().skip(1) {
                cmd.arg(arg);
            }
            cmd.env(&rank_env, rank.to_string());
            cmd.env(&size_env, size.to_string());
            cmd.env("INFER_CUDA_DEVICE", device.to_string());
            cmd.env("INFER_NCCL_UNIQUE_ID", &uid_hex);
            // Pipe stderr so a forwarder thread can watch for liveness; the worker's
            // own `[cpN]` prefix (install_cp_worker_logger) is preserved on re-print.
            cmd.stderr(std::process::Stdio::piped());
            // Each rank's rollout engine is single-GPU (this launcher owns the
            // per-rank device via INFER_CUDA_DEVICE). Strip any inherited TP-serving
            // env so resolve_tp_config_from_env doesn't build a spurious rollout TP
            // communicator on the training uid (else rank>0 dies with NCCL RemoteError).
            for k in [
                "INFER_TP_SIZE",
                "INFER_TP_RANK",
                "INFER_CUDA_DEVICES",
                "ARLE_TP_SIZE",
                "ARLE_TP_RANK",
            ] {
                cmd.env_remove(k);
            }
            let mut child = cmd
                .spawn()
                .with_context(|| format!("spawn {axis} rank {rank}"))?;
            log::info!(
                "spawned {axis} rank {rank} pid={} device={device}",
                child.id()
            );
            if let Some(stderr) = child.stderr.take() {
                let progress = Arc::clone(&last_progress_ms);
                forwarders.push(std::thread::spawn(move || {
                    for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                        progress.store(start.elapsed().as_millis() as u64, Ordering::Relaxed);
                        eprintln!("{line}");
                    }
                }));
            }
            children.push((rank, child));
        }

        // Wait for the first exit; on any exit (success or crash) kill the rest so
        // a single rank's failure can't leave the others blocked in a collective.
        // A rank that HANGS inside a collective never exits, so try_wait alone can't
        // see it — an opt-in NO-PROGRESS deadline (ARLE_TRAIN_STEP_TIMEOUT_SECS)
        // tears the group down only after the whole group has been SILENT that long.
        // Measuring idle-since-last-log (not total wall-clock) lets a legit slow-but-
        // progressing step run forever while still catching a true wedge. Off by
        // default; set it to bound a ladder/OOM probe.
        let hang_timeout = std::env::var("ARLE_TRAIN_STEP_TIMEOUT_SECS")
            .or_else(|_| std::env::var("ARLE_TRAIN_CP_STEP_TIMEOUT_SECS"))
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&s| s > 0)
            .map(Duration::from_secs);
        let (exit_rank, code) = loop {
            if let Some(hit) = children
                .iter_mut()
                .find_map(|(r, c)| c.try_wait().ok().flatten().map(|s| (*r, s.code())))
            {
                break hit;
            }
            if let Some(limit) = hang_timeout {
                let idle = start.elapsed().saturating_sub(Duration::from_millis(
                    last_progress_ms.load(Ordering::Relaxed),
                ));
                if idle > limit {
                    log::error!(
                        "no {axis} rank logged for {idle:?} (> {limit:?}); assuming a hung collective, tearing down"
                    );
                    for (rank, child) in &mut children {
                        let _ = child.kill();
                        let _ = child.wait();
                        log::info!("killed hung {axis} rank {rank}");
                    }
                    anyhow::bail!("{axis} group hung: no rank progressed for {idle:?}");
                }
            }
            std::thread::sleep(Duration::from_millis(100));
        };
        log::info!("{axis} rank {exit_rank} exited (code={code:?}); tearing down group");
        for (rank, child) in &mut children {
            if *rank != exit_rank {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
        // Forwarders end when their child's stderr closes (post-kill/exit); join so
        // buffered final lines flush before we return.
        for f in forwarders {
            let _ = f.join();
        }
        match code {
            Some(0) => Ok(true),
            other => anyhow::bail!("{axis} rank {exit_rank} exited with {other:?}"),
        }
    }
}
