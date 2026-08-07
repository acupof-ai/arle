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

/// Spawned mesh worker (`ARLE_TRAIN_WORLD_RANK` set) → install a `[cpNdpM]`-style
/// stderr prefix. Does NOT short-circuit dispatch — the child flows through clap
/// into the normal agent-opd handler as its rank.
pub(crate) fn install_mesh_worker_logger() -> bool {
    let Ok(rank) = std::env::var("ARLE_TRAIN_WORLD_RANK") else {
        return false;
    };
    let cp = train::context_parallel::CpContext::from_env();
    let dp = train::context_parallel::DpContext::from_env();
    let tag = match (cp.is_enabled(), dp.is_enabled()) {
        (true, false) => format!("cp{}", cp.rank),
        (false, true) => format!("dp{}", dp.rank),
        (true, true) => format!("cp{}dp{}", cp.rank, dp.rank),
        (false, false) => format!("r{rank}"),
    };
    infer_util::logging::init_stderr_with_prefix("info", &format!("[{tag}] "));
    true
}

/// Coordinator: spawn one child per world rank (`cp*dp`, CP inner) with the mesh
/// env + minted NCCL uid, wait for the first exit. `Ok(true)` = caller returns
/// without training; `Ok(false)` = run in-process (single card, or IS a worker).
pub(crate) fn maybe_spawn_mesh_and_wait(
    cp_size: usize,
    dp_size: usize,
    devices: &[usize],
) -> Result<bool> {
    let (cp_size, dp_size) = (cp_size.max(1), dp_size.max(1));
    let size = cp_size * dp_size;
    if size <= 1 || std::env::var("ARLE_TRAIN_WORLD_RANK").is_ok() {
        return Ok(false);
    }
    let axis = match (cp_size > 1, dp_size > 1) {
        (true, true) => "CP×DP",
        (true, false) => "CP",
        (false, true) => "DP",
        (false, false) => unreachable!("size > 1 above"),
    };

    // Mint + publish the NCCL unique id once; children inherit it via env.
    #[cfg(feature = "nccl")]
    let uid_hex = infer_api::mint_nccl_unique_id_hex().context("mint NCCL unique id")?;
    #[cfg(not(feature = "nccl"))]
    {
        let _ = devices;
        anyhow::bail!(
            "{axis} parallelism ({size} ranks) requires the nccl feature; \
             rebuild with --features cuda,nccl"
        );
    }

    #[cfg(feature = "nccl")]
    {
        let exe = std::env::current_exe().context("current_exe")?;
        // Shared dir for the rank-0 → follower update stream (MeshUpdateChannel).
        struct MeshDirGuard(std::path::PathBuf);
        impl Drop for MeshDirGuard {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let mesh_dir = std::env::temp_dir().join(format!("arle-mesh-{}", std::process::id()));
        let _mesh_guard = MeshDirGuard(mesh_dir.clone());
        std::fs::create_dir_all(&mesh_dir)
            .with_context(|| format!("create mesh dir {}", mesh_dir.display()))?;
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
            cmd.env("ARLE_TRAIN_WORLD_RANK", rank.to_string());
            cmd.env("ARLE_TRAIN_CP_SIZE", cp_size.to_string());
            cmd.env("ARLE_TRAIN_DP_SIZE", dp_size.to_string());
            cmd.env("INFER_CUDA_DEVICE", device.to_string());
            cmd.env("INFER_NCCL_UNIQUE_ID", &uid_hex);
            cmd.env("ARLE_TRAIN_MESH_DIR", &mesh_dir);
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
        // Followers exit cleanly at the leader's end-of-stream marker while rank 0
        // still runs eval/save, so only a rank-0 exit (any code) or a nonzero exit
        // tears the group down; cleanly-exited followers leave the poll set.
        let (exit_rank, code) = loop {
            let hit = (0..children.len()).find_map(|i| {
                let (r, c) = &mut children[i];
                c.try_wait().ok().flatten().map(|s| (i, *r, s.code()))
            });
            if let Some((i, rank, code)) = hit {
                if rank == 0 || code != Some(0) {
                    break (rank, code);
                }
                children.swap_remove(i);
                log::info!("{axis} rank {rank} exited cleanly; waiting for rank 0");
                continue;
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
