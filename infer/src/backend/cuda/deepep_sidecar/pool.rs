//! Sidecar process pool — spawns N child processes (one per rank),
//! handshakes IPC + sync, dispatches commands.
//!
//! Lifetime: the pool owns each child's stdin/stdout pipes (mapped to the
//! child's fd 10/11 = `protocol::CHILD_P2C_FD` / `CHILD_C2P_FD`). On drop,
//! sends `Shutdown` to each rank and waits for the process to exit. SIGKILL
//! fallback if a child doesn't honor shutdown within 2 s.

use super::protocol::{
    self, BootRequest, BootResponse, CHILD_C2P_FD, CHILD_P2C_FD, CommandId, KMAX_NVL_PEERS,
    MessageHeader, PROTOCOL_VERSION, RoundTripRequest, RoundTripResponse, Status,
};
use anyhow::{Context, Result, anyhow, bail};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Construction parameters for a `SidecarPool`.
pub struct SidecarPoolConfig<'a> {
    /// Path to the sidecar binary. Use
    /// `crate::backend::cuda::deepep_sidecar::baked_binary_path` when running
    /// from a build that bundled the sidecar; or an explicit path when
    /// driving from a test.
    pub binary: &'a Path,

    /// Number of CUDA ranks. Each rank gets one child process bound to a
    /// matching CUDA device (device id == rank for now).
    pub world_size: usize,
}

/// One forked sidecar child — pipes + child handle.
struct Rank {
    rank: usize,
    /// Writer for the parent → sidecar pipe. Sidecar reads on fd
    /// `CHILD_P2C_FD`.
    p2c_tx: Mutex<std::fs::File>,
    /// Reader for the sidecar → parent pipe. Sidecar writes on fd
    /// `CHILD_C2P_FD`.
    c2p_rx: Mutex<std::fs::File>,
    child: Mutex<Option<Child>>,
}

pub struct SidecarPool {
    world_size: usize,
    ranks: Vec<Rank>,
}

impl SidecarPool {
    /// Spawn `world_size` sidecar children, run the boot+sync handshake,
    /// and return a ready-to-dispatch pool.
    ///
    /// On any error, all already-spawned children are killed before
    /// returning.
    pub fn spawn(cfg: SidecarPoolConfig<'_>) -> Result<Self> {
        if cfg.world_size == 0 || cfg.world_size > KMAX_NVL_PEERS {
            bail!(
                "SidecarPool world_size must be 1..={} (got {})",
                KMAX_NVL_PEERS,
                cfg.world_size
            );
        }
        if !cfg.binary.exists() {
            bail!("sidecar binary not found at {}", cfg.binary.display());
        }

        let mut ranks: Vec<Rank> = Vec::with_capacity(cfg.world_size);
        for rank in 0..cfg.world_size {
            match spawn_one(cfg.binary, rank, cfg.world_size) {
                Ok(r) => ranks.push(r),
                Err(e) => {
                    // Best-effort kill of already-spawned siblings.
                    for r in &ranks {
                        if let Ok(mut guard) = r.child.lock() {
                            if let Some(c) = guard.as_mut() {
                                let _ = c.kill();
                                let _ = c.wait();
                            }
                        }
                    }
                    return Err(e.context(format!("spawning sidecar rank {rank}")));
                }
            }
        }

        let pool = Self {
            world_size: cfg.world_size,
            ranks,
        };
        pool.boot_and_sync()?;
        Ok(pool)
    }

    fn boot_and_sync(&self) -> Result<()> {
        // Step 1: send BootRequest to every rank, collect BootResponse.
        let mut peers: Vec<BootResponse> = Vec::with_capacity(self.world_size);
        for (rank, r) in self.ranks.iter().enumerate() {
            let req = BootRequest {
                protocol_version: PROTOCOL_VERSION,
                rank: rank as u32,
                world_size: self.world_size as u32,
                reserved: 0,
            };
            send_message(r, CommandId::Boot, &req.to_le_bytes())
                .with_context(|| format!("boot send to rank {rank}"))?;
            let (status, payload) =
                recv_message(r).with_context(|| format!("boot recv from rank {rank}"))?;
            if status != Status::Ok {
                bail!("sidecar rank {rank} BootRequest returned {:?}", status);
            }
            let resp = BootResponse::from_le_bytes(&payload)
                .with_context(|| format!("decode BootResponse from rank {rank}"))?;
            peers.push(resp);
        }

        // Step 2: broadcast peer list to every rank via Sync, await Ok.
        let sync_payload = protocol::encode_sync_payload(&peers)?;
        for (rank, r) in self.ranks.iter().enumerate() {
            send_message(r, CommandId::Sync, &sync_payload)
                .with_context(|| format!("sync send to rank {rank}"))?;
        }
        // Sync triggers intranode::barrier, which blocks until ALL ranks
        // call it — so we must send to all before reading from any.
        for (rank, r) in self.ranks.iter().enumerate() {
            let (status, _payload) =
                recv_message(r).with_context(|| format!("sync recv from rank {rank}"))?;
            if status != Status::Ok {
                bail!("sidecar rank {rank} Sync returned {:?}", status);
            }
        }
        Ok(())
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Dispatch the smoke-test `RoundTrip` command on every rank in
    /// parallel. Returns one response per rank in rank order.
    ///
    /// All ranks call into `intranode::barrier` during boot, then dispatch
    /// + combine in parallel. We must send the command to every rank
    /// before reading any response because the kernels block on
    /// cross-rank handshake.
    pub fn round_trip_all(&self, req: RoundTripRequest) -> Result<Vec<RoundTripResponse>> {
        let payload = req.to_le_bytes();
        for (rank, r) in self.ranks.iter().enumerate() {
            send_message(r, CommandId::RoundTrip, &payload)
                .with_context(|| format!("round_trip send to rank {rank}"))?;
        }
        let mut out = Vec::with_capacity(self.world_size);
        for (rank, r) in self.ranks.iter().enumerate() {
            let (status, payload) =
                recv_message(r).with_context(|| format!("round_trip recv from rank {rank}"))?;
            if status != Status::Ok {
                bail!("sidecar rank {rank} RoundTrip returned {:?}", status);
            }
            out.push(
                RoundTripResponse::from_le_bytes(&payload)
                    .with_context(|| format!("decode RoundTripResponse from rank {rank}"))?,
            );
        }
        Ok(out)
    }
}

impl Drop for SidecarPool {
    fn drop(&mut self) {
        for r in &self.ranks {
            // Best-effort shutdown — if any of these fail we still want to
            // continue and reap remaining children.
            let _ = send_message(r, CommandId::Shutdown, &[]);
        }
        let deadline = Instant::now() + Duration::from_secs(2);
        for r in &self.ranks {
            let mut guard = match r.child.lock() {
                Ok(g) => g,
                Err(_) => continue,
            };
            if let Some(mut child) = guard.take() {
                loop {
                    match child.try_wait() {
                        Ok(Some(_)) => break,
                        Ok(None) if Instant::now() < deadline => {
                            std::thread::sleep(Duration::from_millis(50));
                        }
                        _ => {
                            let _ = child.kill();
                            let _ = child.wait();
                            break;
                        }
                    }
                }
            }
        }
    }
}

fn spawn_one(binary: &Path, rank: usize, world_size: usize) -> Result<Rank> {
    // Parent ↔ child pipes. Child reads commands on p2c[0] (becomes fd 10),
    // writes responses on c2p[1] (becomes fd 11). Parent keeps p2c[1] (tx)
    // and c2p[0] (rx).
    let (p2c_r, p2c_w) = mk_pipe().context("create p2c pipe")?;
    let (c2p_r, c2p_w) = mk_pipe().context("create c2p pipe")?;

    let p2c_r_raw = p2c_r.as_raw_fd();
    let c2p_w_raw = c2p_w.as_raw_fd();

    let mut cmd = Command::new(binary);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    // Cargo / supervisor envs are inherited; the binary needs nothing
    // extra. Argv is unused by the sidecar today (rank comes from BootRequest).
    cmd.arg("--rank").arg(rank.to_string());
    cmd.arg("--world-size").arg(world_size.to_string());

    // pre_exec runs in the forked child between fork() and execvp(). We
    // dup the read end of p2c onto fd 10 and the write end of c2p onto
    // fd 11, mirroring the phase 1.0a-iii spike fork layout. dup2 clears
    // CLOEXEC on the destination, so the duped fds survive exec.
    unsafe {
        cmd.pre_exec(move || {
            if libc::dup2(p2c_r_raw, CHILD_P2C_FD) < 0 || libc::dup2(c2p_w_raw, CHILD_C2P_FD) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    let child = cmd.spawn().context("spawn sidecar binary")?;

    // Close the child-side fds in the parent — only the duped 10/11
    // need to live in the child, and only p2c_w / c2p_r need to live
    // here.
    drop(p2c_r);
    drop(c2p_w);

    Ok(Rank {
        rank,
        p2c_tx: Mutex::new(p2c_w),
        c2p_rx: Mutex::new(c2p_r),
        child: Mutex::new(Some(child)),
    })
}

fn mk_pipe() -> std::io::Result<(std::fs::File, std::fs::File)> {
    let mut fds = [0i32; 2];
    // Safety: pipe(2) writes two fds into the passed array.
    let ret = unsafe { libc::pipe(fds.as_mut_ptr()) };
    if ret != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // Safety: pipe(2) returns valid owned fds.
    unsafe {
        use std::os::fd::FromRawFd;
        Ok((
            std::fs::File::from_raw_fd(fds[0]),
            std::fs::File::from_raw_fd(fds[1]),
        ))
    }
}

fn send_message(r: &Rank, cmd: CommandId, payload: &[u8]) -> Result<()> {
    let hdr = MessageHeader {
        cmd_or_status: cmd as u32,
        payload_bytes: payload.len() as u32,
    };
    let mut tx = r
        .p2c_tx
        .lock()
        .map_err(|_| anyhow!("p2c_tx poisoned on rank {}", r.rank))?;
    tx.write_all(&hdr.to_le_bytes())?;
    if !payload.is_empty() {
        tx.write_all(payload)?;
    }
    tx.flush()?;
    Ok(())
}

fn recv_message(r: &Rank) -> Result<(Status, Vec<u8>)> {
    let mut rx = r
        .c2p_rx
        .lock()
        .map_err(|_| anyhow!("c2p_rx poisoned on rank {}", r.rank))?;
    let mut hdr_buf = [0u8; MessageHeader::SIZE];
    rx.read_exact(&mut hdr_buf)?;
    let hdr = MessageHeader::from_le_bytes(&hdr_buf);
    let mut payload = vec![0u8; hdr.payload_bytes as usize];
    if !payload.is_empty() {
        rx.read_exact(&mut payload)?;
    }
    let status = Status::from_u32(hdr.cmd_or_status)?;
    Ok((status, payload))
}
