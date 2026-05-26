//! Phase B-0 license-or-kill — 2-process NCCL EnvBootstrap smoke.
//!
//! Proves the foundational assumption of `docs/plans/2026-05-27-multiproc-
//! serve-pivot.md`: NCCL EnvBootstrap is process-agnostic. ARLE's current
//! single-process-N-thread NCCL setup uses `MASTER_ADDR/MASTER_PORT/WORLD_
//! SIZE` env vars for TCP rendezvous (`infer/src/main.rs:407-415`,
//! `infer/src/distributed/nccl.rs:38-49`). The Option-B pivot relies on
//! that pattern surviving when each rank runs in its own process.
//!
//! Mode:
//!   - If ARLE_WORKER_RANK is unset → coordinator. Sets MASTER_ADDR/PORT,
//!     spawns N-1 children via `std::process::Command::current_exe()`,
//!     becomes rank 0 itself, runs broadcast+all_reduce, waits for children.
//!   - If ARLE_WORKER_RANK is set → worker. Joins the existing NCCL group
//!     at the rank specified by the env var, runs the same broadcast+
//!     all_reduce, exits.
//!
//! Usage:
//!   ARLE_NCCL_SMOKE_WORLD_SIZE=2 cargo run --release --features cuda,nccl \
//!     --bin multiproc_nccl_smoke
//!   ARLE_NCCL_SMOKE_WORLD_SIZE=8 cargo run --release --features cuda,nccl \
//!     --bin multiproc_nccl_smoke
//!
//! PASS: all N processes exit 0; rank-0's broadcast is observed by every
//! worker; all_reduce produces sum-of-ranks = N*(N-1)/2 as the gathered
//! value; runs 10 cycles in a row without hang.
//! KILL: any process hangs > 60 s, exit code != 0, all_reduce mismatch, or
//! NCCL init fails with "no peer" / "rendezvous timeout" — would falsify
//! the cross-process EnvBootstrap assumption and force going back to
//! Option A.

#![cfg(all(feature = "cuda", feature = "nccl"))]

use std::env;
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result, bail};

use infer::distributed::nccl::{NcclGroup, NcclInitMethod};

const CYCLES: usize = 10;

fn main() -> Result<()> {
    // Worker mode: ARLE_WORKER_RANK is set by the coordinator before spawn.
    if let Ok(rank_str) = env::var("ARLE_WORKER_RANK") {
        let rank: usize = rank_str
            .parse()
            .context("ARLE_WORKER_RANK must be a non-negative integer")?;
        let world_size: usize = env::var("ARLE_NCCL_SMOKE_WORLD_SIZE")
            .context("worker missing ARLE_NCCL_SMOKE_WORLD_SIZE")?
            .parse()
            .context("ARLE_NCCL_SMOKE_WORLD_SIZE not an integer")?;
        return run_rank(rank, world_size).context(format!("rank {rank} failed"));
    }

    // Coordinator mode. World size from env (default 2 for the minimal gate).
    let world_size: usize = env::var("ARLE_NCCL_SMOKE_WORLD_SIZE")
        .ok()
        .map(|s| s.parse().expect("ARLE_NCCL_SMOKE_WORLD_SIZE invalid"))
        .unwrap_or(2);
    if world_size < 2 {
        bail!("world_size must be >= 2 for the 2-process smoke (got {world_size})");
    }

    let master_addr = env::var("MASTER_ADDR").unwrap_or_else(|_| "127.0.0.1".to_string());
    let master_port = env::var("MASTER_PORT").unwrap_or_else(|_| pick_port().to_string());

    eprintln!(
        "[coordinator pid={}] world_size={world_size} master={master_addr}:{master_port}",
        std::process::id()
    );

    // SAFETY: env writes happen before we spawn any child. Single-threaded.
    unsafe {
        env::set_var("MASTER_ADDR", &master_addr);
        env::set_var("MASTER_PORT", &master_port);
        env::set_var("WORLD_SIZE", world_size.to_string());
        env::set_var("ARLE_NCCL_SMOKE_WORLD_SIZE", world_size.to_string());
    }

    // Spawn N-1 children (rank 1..world_size). Coordinator is rank 0.
    let exe = std::env::current_exe().context("current_exe")?;
    let mut children = Vec::with_capacity(world_size - 1);
    for rank in 1..world_size {
        let child = Command::new(&exe)
            .env("ARLE_WORKER_RANK", rank.to_string())
            .env("ARLE_NCCL_SMOKE_WORLD_SIZE", world_size.to_string())
            .env("MASTER_ADDR", &master_addr)
            .env("MASTER_PORT", &master_port)
            .env("WORLD_SIZE", world_size.to_string())
            // Mirror RANK convention that NCCL EnvBootstrap may inspect.
            .env("RANK", rank.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawning worker rank {rank}"))?;
        children.push((rank, child));
    }

    // Coordinator becomes rank 0.
    let result = run_rank(0, world_size);

    // Drain children regardless of rank-0 outcome.
    let mut fail = result.is_err();
    for (rank, mut child) in children {
        let deadline = std::time::Instant::now() + Duration::from_secs(90);
        loop {
            match child.try_wait()? {
                Some(status) => {
                    if !status.success() {
                        eprintln!("[coordinator] rank {rank} exited {:?}", status.code());
                        fail = true;
                    } else {
                        eprintln!("[coordinator] rank {rank} exited 0");
                    }
                    break;
                }
                None if std::time::Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(100));
                }
                _ => {
                    eprintln!("[coordinator] rank {rank} timed out — killing");
                    let _ = child.kill();
                    let _ = child.wait();
                    fail = true;
                    break;
                }
            }
        }
    }

    if fail {
        if let Err(err) = result {
            eprintln!("[coordinator] rank 0 error: {err:#}");
        }
        bail!("multiproc_nccl_smoke FAILED");
    }
    eprintln!("[coordinator] multiproc_nccl_smoke PASS ({CYCLES} cycles, world={world_size})");
    Ok(())
}

fn run_rank(rank: usize, world_size: usize) -> Result<()> {
    eprintln!(
        "[rank {rank} pid={}] joining NCCL EnvBootstrap world={world_size}",
        std::process::id()
    );

    let group = NcclGroup::new(rank, world_size, NcclInitMethod::EnvBootstrap)
        .with_context(|| format!("rank {rank} NcclGroup::new"))?;
    eprintln!("[rank {rank}] NCCL group ready");

    for cycle in 0..CYCLES {
        // Broadcast a single f32 from rank 0. Rank 0 sends the cycle counter
        // as a sentinel; others receive zeros and overwrite locally.
        let bcast_in = if rank == 0 {
            vec![cycle as f32 + 1.0]
        } else {
            vec![0.0f32]
        };
        let bcast_out = group
            .broadcast_f32(&bcast_in, 1, 0)
            .with_context(|| format!("rank {rank} cycle {cycle} broadcast"))?;
        let expected = cycle as f32 + 1.0;
        if (bcast_out[0] - expected).abs() > 1e-6 {
            bail!(
                "rank {rank} cycle {cycle} broadcast mismatch: got {} expected {expected}",
                bcast_out[0]
            );
        }

        // All-reduce sum: every rank contributes (rank+1). Sum = N*(N+1)/2.
        let reduce_out = group
            .all_reduce_f32(&[(rank + 1) as f32])
            .with_context(|| format!("rank {rank} cycle {cycle} all_reduce"))?;
        let expected_sum = (1..=world_size).sum::<usize>() as f32;
        if (reduce_out[0] - expected_sum).abs() > 1e-6 {
            bail!(
                "rank {rank} cycle {cycle} all_reduce mismatch: got {} expected {expected_sum}",
                reduce_out[0]
            );
        }
    }

    eprintln!("[rank {rank}] {CYCLES} cycles PASS");
    Ok(())
}

fn pick_port() -> u16 {
    // Bind 127.0.0.1:0 to grab a free port from the OS, then drop the
    // listener so the port is available for NCCL's TcpStore. Inherent race
    // window is tiny in practice on a single host.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("pick_port bind");
    let port = listener.local_addr().expect("pick_port addr").port();
    drop(listener);
    port
}
