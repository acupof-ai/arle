//! Phase B-1 commit C.1 — multiproc_relay end-to-end smoke.
//!
//! Pure TCP transport check; no NCCL, no CUDA, no scheduler. Coordinator
//! mode forks N-1 workers via Command::current_exe() with
//! ARLE_RELAY_SMOKE_RANK=R and ARLE_RELAY_SMOKE_PORT=<port>. Each worker
//! connects to the coordinator and receives 5 Request envelopes + a
//! coordinator EOF on shutdown. PASS = every worker exits 0 with all
//! envelopes received in order.
//!
//! Independent of the cuda/nccl features — no GPU needed. Useful to
//! validate the protocol module on any host.
//!
//! Usage:
//!   ARLE_RELAY_SMOKE_WORLD_SIZE=2 cargo run --release --bin multiproc_relay_smoke
//!   ARLE_RELAY_SMOKE_WORLD_SIZE=8 cargo run --release --bin multiproc_relay_smoke

use std::env;
use std::net::SocketAddr;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

#[allow(unused_imports)]
use infer::multiproc_relay::PendingRelayCoordinator;
use infer::multiproc_relay::{RelayCoordinator, RelayEnvelope, RelayWorker};

const CYCLES: usize = 5;

fn main() -> Result<()> {
    // Worker mode: coordinator passes our rank + port via env.
    if let Ok(rank_str) = env::var("ARLE_RELAY_SMOKE_RANK") {
        let rank: usize = rank_str.parse().context("ARLE_RELAY_SMOKE_RANK parse")?;
        let port: u16 = env::var("ARLE_RELAY_SMOKE_PORT")
            .context("worker missing ARLE_RELAY_SMOKE_PORT")?
            .parse()
            .context("ARLE_RELAY_SMOKE_PORT parse")?;
        return run_worker(rank, port);
    }

    let world_size: usize = env::var("ARLE_RELAY_SMOKE_WORLD_SIZE")
        .ok()
        .map(|s| s.parse().expect("ARLE_RELAY_SMOKE_WORLD_SIZE parse"))
        .unwrap_or(2);
    if world_size < 2 {
        bail!("relay smoke needs world_size >= 2 (got {world_size})");
    }

    // Bind FIRST so the listener exists before workers connect.
    let pending = RelayCoordinator::bind()?;
    let port = pending.port();
    eprintln!(
        "[coordinator pid={}] world_size={world_size} port={port}",
        std::process::id()
    );

    // Spawn N-1 workers; they connect to the bound port.
    let exe = env::current_exe().context("current_exe")?;
    let mut children = Vec::with_capacity(world_size - 1);
    for rank in 1..world_size {
        let child = Command::new(&exe)
            .env("ARLE_RELAY_SMOKE_RANK", rank.to_string())
            .env("ARLE_RELAY_SMOKE_PORT", port.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawn worker rank {rank}"))?;
        children.push((rank, child));
    }

    let mut coord = pending.accept(world_size, Duration::from_secs(10))?;
    eprintln!(
        "[coordinator] all {} workers connected",
        coord.worker_count()
    );

    for cycle in 0..CYCLES {
        coord.broadcast(&RelayEnvelope::Request {
            request_id: cycle as u64 + 1,
            prompt_tokens: vec![100 + cycle as u32, 200 + cycle as u32],
            max_new_tokens: 1,
            sampling: serde_json::json!({"cycle": cycle}),
        })?;
    }
    eprintln!(
        "[coordinator] broadcast {} cycles, dropping coord to signal EOF",
        CYCLES
    );
    drop(coord);

    let mut fail = false;
    for (rank, mut child) in children {
        let deadline = Instant::now() + Duration::from_secs(10);
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
                None if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(50));
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
        bail!("multiproc_relay_smoke FAILED");
    }
    eprintln!("[coordinator] multiproc_relay_smoke PASS ({CYCLES} cycles, world={world_size})");
    Ok(())
}

fn run_worker(rank: usize, port: u16) -> Result<()> {
    let addr: SocketAddr = format!("127.0.0.1:{port}")
        .parse()
        .context("worker parse coordinator addr")?;
    eprintln!(
        "[worker pid={} rank={rank}] connecting to {addr}",
        std::process::id()
    );
    let mut worker = RelayWorker::connect(addr, Duration::from_secs(10))
        .with_context(|| format!("worker rank {rank} connect"))?;

    let mut received = 0usize;
    loop {
        match worker.recv()? {
            Some(env) => {
                received += 1;
                if let RelayEnvelope::Request {
                    request_id,
                    prompt_tokens,
                    ..
                } = env
                {
                    let expected_first = 100 + (request_id as u32 - 1);
                    if prompt_tokens.first().copied() != Some(expected_first) {
                        bail!(
                            "rank {rank} envelope {request_id} prompt_tokens mismatch: \
                             got {prompt_tokens:?} expected first={expected_first}"
                        );
                    }
                }
            }
            None => {
                eprintln!("[worker rank={rank}] coordinator EOF after {received} envelopes");
                if received != CYCLES {
                    bail!("rank {rank} expected {CYCLES} envelopes, received {received}");
                }
                return Ok(());
            }
        }
    }
}
