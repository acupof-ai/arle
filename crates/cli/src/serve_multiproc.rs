//! DSv4 multiproc-serve coordinator + worker scaffold (Stage 1 re-wire).
//!
//! Ported from `git show e81b98fb^:infer/src/main.rs` (fn `main` worker-entry,
//! `run_worker_mode`, `spawn_cuda_worker_processes`, and the coordinator-side
//! relay bind/spawn/accept in `async_main`), adapted to the rewrite crate stack.
//!
//! Control flow this reproduces:
//!   - `arle serve` on a DSv4 + CUDA model becomes the COORDINATOR (rank 0):
//!     [`bind_relay_and_spawn_workers`] binds the relay listener
//!     ([`RelayCoordinator::bind`]), publishes the port via
//!     `ARLE_COORDINATOR_RELAY_PORT`, spawns N-1 worker processes via
//!     `current_exe()` with `ARLE_WORKER_RANK=R` + `WORLD_SIZE` +
//!     `INFER_CUDA_DEVICE` + a parent-fd pipe (`ARLE_WORKER_PARENT_FD`), accepts
//!     the N-1 relay connects, boot-pings, then the caller proceeds to its
//!     normal in-process serve. The returned [`CoordinatorGuard`] holds the
//!     relay + worker pipes; dropping it EOFs the workers so they exit.
//!   - On process start, BEFORE clap parsing, [`worker_entry`] detects
//!     `ARLE_WORKER_RANK>0` and runs worker mode: pre-connect the relay
//!     ([`RelayWorker::connect_with_rank`]), build the rank-R DSv4 engine (which
//!     bootstraps NCCL as rank R from env during construction), then run a
//!     relay-receiver loop.
//!
//! Stage 2 (this change) wires the FUNCTIONAL lockstep forward: the coordinator
//! installs an admission broadcaster (`infer_api::set_admission_broadcaster`) so
//! every request the rank-0 engine admits is fanned out as a `RelayEnvelope::
//! Request` to ranks 1..N-1; each worker's relay-receiver loop submits the
//! reconstructed request into its own rank-R engine
//! (`LoadedInferenceEngine::submit_replicated`), whose background loop drives the
//! executor's NCCL collective `forward` in lockstep with rank 0. Worker output is
//! discarded — only rank 0 returns over HTTP.
//!
//! `// STAGE 3:` owner-routed visible output (`RelayEnvelope::Completion` back to
//! a non-rank-0 owner), chunked-prefill scratch chunk-bounding, and the
//! long-decode path beyond a single short prompt remain later stages.

#![cfg(all(unix, feature = "cuda"))]

use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use infer_api::{
    EngineLoadConfig, LoadedInferenceEngine, RelayCoordinator, RelayEnvelope, RelayWorker,
    WireRequest,
};

/// Relay accept / worker-connect timeout. Generous: a cold DSv4 rank-0 build can
/// take tens of seconds before it reaches the accept point.
const RELAY_TIMEOUT: Duration = Duration::from_secs(120);

/// Whether the checkpoint at `model_path` is a DSv4-Flash model (so serve should
/// take the multiproc path). Reads `architectures` / `model_type` from
/// `config.json`; returns `false` on any read/parse failure (the single-process
/// path then errors with its normal message).
#[must_use]
pub(crate) fn is_dsv4_model(model_path: &str) -> bool {
    let Ok(raw) = std::fs::read_to_string(std::path::Path::new(model_path).join("config.json"))
    else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return false;
    };
    let model_type = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
    if model_type == "deepseek_v4" {
        return true;
    }
    v.get("architectures")
        .and_then(|a| a.as_array())
        .is_some_and(|a| {
            a.iter()
                .any(|s| s.as_str().is_some_and(|s| s.contains("DeepseekV4")))
        })
}

/// Resolve the multiproc world size from the environment. `INFER_TP_SIZE` wins,
/// else the comma-separated `INFER_CUDA_DEVICES` count, else 1 (single GPU — no
/// worker spawn, the coordinator serves alone).
#[must_use]
pub(crate) fn world_size_from_env() -> usize {
    if let Some(n) = std::env::var("INFER_TP_SIZE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return n;
    }
    std::env::var("INFER_CUDA_DEVICES")
        .ok()
        .map(|list| list.split(',').filter(|s| !s.trim().is_empty()).count())
        .filter(|&n| n > 0)
        .unwrap_or(1)
}

/// The comma-separated CUDA device ordinals from `INFER_CUDA_DEVICES`, or
/// `0..world_size` when unset. Used to assign one GPU per worker rank.
fn cuda_ordinals(world_size: usize) -> Vec<usize> {
    std::env::var("INFER_CUDA_DEVICES")
        .ok()
        .map(|list| {
            list.split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| (0..world_size).collect())
}

/// RAII guard for a running coordinator: holds the accepted relay plus the
/// worker-process pipe write-ends. Dropping it closes the pipes (workers EOF on
/// their parent-pipe read and exit) and lets the relay drop.
///
/// The relay is `Arc<Mutex<_>>` so it is shared between the process-global
/// admission broadcaster (installed in [`bind_relay_and_spawn_workers`], invoked
/// from the rank-0 engine thread per request) and this guard, which keeps it (and
/// the worker pipes) alive for the serve loop. This mirrors the old coordinator
/// wrapping its `RelayCoordinator` in `Arc<Mutex<_>>` to share it with
/// `DistributedSchedulerGroup` (`e81b98fb^:infer/src/main.rs:1949`).
pub(crate) struct CoordinatorGuard {
    _relay: Arc<Mutex<RelayCoordinator>>,
    _children: WorkerChildren,
}

/// COORDINATOR (rank 0): bind the relay, publish its port, spawn N-1 workers,
/// accept their relay connects, boot-ping, then install the process-global
/// admission broadcaster so each request the rank-0 engine admits is fanned out
/// to workers in lockstep. Returns a guard the caller holds for the lifetime of
/// the serve loop; on `world_size <= 1` returns `None` (single GPU — serve alone,
/// no workers, no broadcaster — the default serve path is byte-identical).
///
/// Stage-2 wiring (the admission-broadcast hook) is ported from the old
/// `DistributedSchedulerGroup::submit` path (`e81b98fb^:infer/src/request_handle.rs:909`):
/// the coordinator's relay is shared (`Arc<Mutex<_>>`) between this guard and a
/// closure installed via [`infer_api::set_admission_broadcaster`]; the rank-0
/// engine thread invokes that closure (`infer_server::broadcast_admission`) at the
/// single deterministic admission point, broadcasting a [`WireRequest`] to every
/// worker before submitting the request locally.
pub(crate) fn bind_relay_and_spawn_workers(model_path: &str) -> Result<Option<CoordinatorGuard>> {
    let world_size = world_size_from_env();
    if world_size <= 1 {
        log::info!(
            "[multiproc-coord] world_size={world_size}; serving DSv4 single-process (no workers)"
        );
        return Ok(None);
    }

    // 0. Mint the NCCL rendezvous id ONCE (rank 0) and publish it via env so every
    //    spawned worker inherits the SAME handle. The DSv4 executor reads
    //    `INFER_NCCL_UNIQUE_ID` during construction (loader::nccl_unique_id_from_env);
    //    minting here is the multiproc analogue of the parity launcher's
    //    file-rendezvous. Must happen BEFORE spawn_workers so children inherit it.
    //    Skip if already set (an external launcher provided the rendezvous).
    #[cfg(feature = "nccl")]
    if std::env::var("INFER_NCCL_UNIQUE_ID").is_err() {
        let hex = infer_api::mint_nccl_unique_id_hex().context("mint NCCL unique id")?;
        // SAFETY: single CLI thread, pre-spawn, pre-tokio (same as the relay-port write).
        unsafe {
            std::env::set_var("INFER_NCCL_UNIQUE_ID", &hex);
        }
        log::info!("[multiproc-coord] minted NCCL unique id (published via INFER_NCCL_UNIQUE_ID)");
    }
    #[cfg(not(feature = "nccl"))]
    anyhow::bail!(
        "DSv4 multi-rank serve (world_size={world_size}) requires the `nccl` feature; \
         rebuild with --features cuda,nccl"
    );

    // 1. Bind relay BEFORE spawning workers so the port can be exported via env.
    let pending = RelayCoordinator::bind().context("multiproc relay bind")?;
    // SAFETY: env write happens before child spawn, on the single CLI thread
    // (clap parsing is done, no tokio runtime built yet for the serve path).
    unsafe {
        std::env::set_var("ARLE_COORDINATOR_RELAY_PORT", pending.port().to_string());
    }
    log::info!(
        "[multiproc-coord] relay bound at 127.0.0.1:{} (published via ARLE_COORDINATOR_RELAY_PORT)",
        pending.port()
    );

    // 2. Spawn N-1 worker processes (rank 1..world_size).
    let children = spawn_workers(model_path, world_size)?;

    // 3. Accept N-1 worker relay connects.
    let mut relay = pending
        .accept(world_size, RELAY_TIMEOUT)
        .context("multiproc relay accept")?;
    log::info!(
        "[multiproc-coord] relay accepted {} worker connects (ranks {:?})",
        relay.worker_count(),
        relay.worker_ranks()
    );

    // 4. Boot ping — proves every worker's relay-receiver thread is alive before
    //    the coordinator opens HTTP.
    relay
        .broadcast(&RelayEnvelope::BootPing { request_id: 0 })
        .context("multiproc relay boot-ping")?;

    // 5. Install the admission broadcaster. Share the relay (`Arc<Mutex<_>>`)
    //    between this guard and the broadcast closure the rank-0 engine thread
    //    invokes per request. The closure assigns each request a monotonic id and
    //    broadcasts a `WireRequest` to every worker, matching the old
    //    `DistributedSchedulerGroup::submit` fanout. Broadcast failure here is
    //    logged (not fatal) — a dead worker is surfaced by the relay EOF path; the
    //    request still runs on rank 0, which is the only rank returning HTTP output.
    let relay = Arc::new(Mutex::new(relay));
    let broadcast_relay = Arc::clone(&relay);
    let next_request_id = std::sync::atomic::AtomicU64::new(1);
    infer_api::set_admission_broadcaster(Box::new(move |prompt, max_tokens, sampling| {
        let request_id = next_request_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let wire = WireRequest {
            request_id,
            prompt_tokens: prompt.to_vec(),
            max_tokens,
            sampling: sampling.clone(),
        };
        let mut coord = broadcast_relay
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Err(err) = coord.broadcast(&RelayEnvelope::Request { wire }) {
            log::warn!(
                "[multiproc-coord] admission broadcast for req#{request_id} failed: {err:#}"
            );
        }
    }))
    .context("install multiproc admission broadcaster")?;

    Ok(Some(CoordinatorGuard {
        _relay: relay,
        _children: children,
    }))
}

/// WORKER entry, hooked at the very top of `cli::run()` before clap parsing.
///
/// Returns `Some(ExitCode)` when this process is a worker (rank > 0) and ran
/// worker mode to completion; `None` to fall through to the normal CLI path
/// (rank 0 / not a worker).
#[must_use]
pub(crate) fn worker_entry() -> Option<ExitCode> {
    let rank: usize = std::env::var("ARLE_WORKER_RANK").ok()?.parse().ok()?;
    if rank == 0 {
        // rank 0 is the coordinator; it falls through to the normal serve path.
        return None;
    }
    infer_util::logging::init_stderr("info");
    match run_worker_mode(rank) {
        Ok(()) => Some(ExitCode::SUCCESS),
        Err(err) => {
            eprintln!("[arle-worker rank={rank}] failed: {err:#}");
            Some(ExitCode::FAILURE)
        }
    }
}

/// Worker-mode body (rank R > 0). Pre-connects the relay, builds the rank-R DSv4
/// engine (NCCL bootstrap as rank R happens inside construction from env), runs
/// the relay-receiver loop, then blocks on the parent pipe until the coordinator
/// dies.
fn run_worker_mode(rank: usize) -> Result<()> {
    let world_size: usize = std::env::var("WORLD_SIZE")
        .context("worker missing WORLD_SIZE env (set by coordinator)")?
        .parse()
        .context("worker WORLD_SIZE parse")?;
    let model_path = std::env::var("ARLE_WORKER_MODEL_PATH")
        .context("worker missing ARLE_WORKER_MODEL_PATH env (set by coordinator)")?;

    log::info!(
        "[arle-worker pid={} rank={rank}/{world_size}] starting (model={model_path})",
        std::process::id()
    );

    // 1. Pre-connect the relay BEFORE building the engine. The coordinator's flow
    //    binds the relay listener first, then spawns workers, then ACCEPTS relay
    //    connects (blocking), and only after that proceeds. If workers built the
    //    engine (NCCL rendezvous) first they could deadlock against the
    //    coordinator still stuck in accept. Connect relay first so the
    //    coordinator unblocks accept -> proceeds -> NCCL listener up.
    let mut relay = if let Ok(port_str) = std::env::var("ARLE_COORDINATOR_RELAY_PORT") {
        let port: u16 = port_str
            .parse()
            .with_context(|| format!("worker rank {rank} ARLE_COORDINATOR_RELAY_PORT parse"))?;
        let addr: std::net::SocketAddr = format!("127.0.0.1:{port}")
            .parse()
            .with_context(|| format!("worker rank {rank} relay addr parse"))?;
        let relay = RelayWorker::connect_with_rank(addr, RELAY_TIMEOUT, rank, world_size)
            .with_context(|| format!("worker rank {rank} relay connect"))?;
        log::info!("[arle-worker rank={rank}] relay pre-connected to {addr}");
        Some(relay)
    } else {
        log::warn!(
            "[arle-worker rank={rank}] ARLE_COORDINATOR_RELAY_PORT unset; \
             skipping relay-receiver loop"
        );
        None
    };

    // 2. Build the rank-R DSv4 engine. `LoadedInferenceEngine::load` dispatches
    //    (CUDA build) to the DSv4 executor builder, which resolves TP rank R /
    //    world-size + EP split + NCCL communicator from the env that the
    //    coordinator set (`INFER_TP_RANK`, `INFER_TP_SIZE` / `INFER_CUDA_DEVICES`,
    //    `INFER_NCCL_UNIQUE_ID` / `INFER_NCCL_ID_FILE`).
    //
    //    The engine is built so the NCCL group forms AND so the relay-receiver
    //    loop below can submit each relayed request into it: the engine's own
    //    background loop then drives the executor's `forward` (the NCCL collective)
    //    in lockstep with rank 0.
    log::info!("[arle-worker rank={rank}] building rank-{rank} DSv4 engine");
    let engine: LoadedInferenceEngine =
        LoadedInferenceEngine::load_with_config(&model_path, false, EngineLoadConfig::default())
            .with_context(|| format!("worker rank {rank} engine build"))?;
    log::info!("[arle-worker rank={rank}] engine built; entering relay-receiver loop");

    // 3. Relay-receiver loop: submit each received request into the local engine
    //    so its loop steps the executor in lockstep with rank 0.
    if let Some(relay) = relay.as_mut() {
        run_relay_receiver(rank, relay, &engine)?;
    }

    // 4. Block on the parent pipe — the coordinator closes the write end on
    //    shutdown, the read() returns EOF, and the worker exits.
    block_on_parent_pipe(rank);
    Ok(())
}

/// Drain the relay until EOF / shutdown, submitting each received request into
/// the worker's local engine so its loop steps the executor in lockstep with
/// rank 0.
///
/// Ported from the old worker relay-receiver thread
/// (`e81b98fb^:infer/src/main.rs:486-540`): on each `RelayEnvelope::Request`, the
/// worker reconstructed the `IncomingRequest` from the wire and called
/// `permit.submit(req)` into its local scheduler, discarding the deltas (worker
/// output never reaches the user). Here, the rewrite engine's `ServeHandle`
/// background loop already owns the step/poll cycle, so the worker only needs to
/// submit the reconstructed request and drop the ticket; the NCCL collective
/// inside each `forward` is the implicit per-step barrier that keeps the worker's
/// step cadence aligned with rank 0's.
///
/// `// STAGE 2 (GPU-verify):` lockstep alignment — this assumes the worker's
/// engine admits the same requests in the same order as rank 0 (guaranteed by the
/// FIFO relay + rank-0's single ordered admission broadcast) so its deterministic
/// planner builds the same per-step batches, and that the NCCL collective barriers
/// any rank that races ahead. Confirm on hardware that no rank deadlocks or drifts.
fn run_relay_receiver(
    rank: usize,
    relay: &mut RelayWorker,
    engine: &LoadedInferenceEngine,
) -> Result<()> {
    let mut count: usize = 0;
    loop {
        match relay.recv()? {
            Some(RelayEnvelope::Request { wire }) => {
                count += 1;
                let request_id = wire.request_id;
                let prompt_len = wire.prompt_tokens.len();
                let (prompt_tokens, max_tokens, sampling) = wire.submit_args();
                // Submit into the local engine; its background loop drives the
                // executor's NCCL `forward` in lockstep with rank 0. Worker output
                // is discarded — only rank 0 returns over HTTP.
                if let Err(err) = engine.submit_replicated(prompt_tokens, max_tokens, sampling) {
                    log::warn!(
                        "[arle-worker rank={rank}] submit for req#{request_id} failed: {err:#}"
                    );
                } else {
                    log::debug!(
                        "[arle-worker rank={rank}] submitted req#{request_id} (prompt_tokens={prompt_len}, max_tokens={max_tokens})"
                    );
                }
            }
            Some(RelayEnvelope::BootPing { request_id }) => {
                log::debug!("[arle-worker rank={rank}] boot-ping request_id={request_id}");
            }
            Some(RelayEnvelope::WorkerHello { .. }) => {
                log::warn!("[arle-worker rank={rank}] unexpected worker hello after relay accept");
            }
            Some(RelayEnvelope::Completion { request_id, .. }) => {
                log::warn!(
                    "[arle-worker rank={rank}] unexpected completion envelope request_id={request_id}"
                );
            }
            Some(RelayEnvelope::Shutdown) => {
                log::info!("[arle-worker rank={rank}] relay shutdown envelope received");
                return Ok(());
            }
            None => {
                log::info!("[arle-worker rank={rank}] relay EOF after {count} envelopes");
                return Ok(());
            }
        }
    }
}

/// Block on the parent-fd pipe until EOF (coordinator closed it / died).
fn block_on_parent_pipe(rank: usize) {
    use std::io::Read;
    use std::os::fd::FromRawFd;

    if let Ok(fd_str) = std::env::var("ARLE_WORKER_PARENT_FD") {
        match fd_str.parse::<i32>() {
            Ok(fd) => {
                // SAFETY: fd was inherited from the coordinator; the File
                // destructor closes it on exit.
                let mut parent_pipe = unsafe { std::fs::File::from_raw_fd(fd) };
                let mut buf = [0u8; 1];
                match parent_pipe.read(&mut buf) {
                    Ok(0) => {
                        log::info!("[arle-worker rank={rank}] parent pipe closed, shutting down");
                    }
                    Ok(_) => log::info!("[arle-worker rank={rank}] parent shutdown byte, exiting"),
                    Err(err) => {
                        log::info!("[arle-worker rank={rank}] parent pipe error {err}, exiting");
                    }
                }
            }
            Err(err) => log::warn!("[arle-worker rank={rank}] ARLE_WORKER_PARENT_FD parse: {err}"),
        }
    } else {
        log::info!("[arle-worker rank={rank}] no ARLE_WORKER_PARENT_FD; sleeping until killed");
        loop {
            std::thread::sleep(Duration::from_secs(3600));
        }
    }
}

/// Coordinator-side child spawn: fork one child per rank ≥ 1 via
/// `current_exe()` with `ARLE_WORKER_RANK=R` + a parent pipe so the child can
/// detect coordinator death (read EOF -> exit). Returns the children + parent
/// pipe write-ends; dropping [`WorkerChildren`] closes them so workers exit.
struct WorkerChildren {
    children: Vec<(usize, std::process::Child, std::fs::File)>,
}

impl Drop for WorkerChildren {
    fn drop(&mut self) {
        // Close parent pipe write-ends -> children's read() returns EOF -> they
        // exit on their own. Then wait up to 5s per child; kill stragglers.
        let _writes_dropped: Vec<_> = self
            .children
            .iter_mut()
            .map(|(_, _, pipe)| std::mem::replace(pipe, dummy_file()))
            .collect();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        for (rank, child, _) in &mut self.children {
            loop {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        if !status.success() {
                            log::warn!("worker rank {rank} exited {:?}", status.code());
                        }
                        break;
                    }
                    Ok(None) if std::time::Instant::now() < deadline => {
                        std::thread::sleep(Duration::from_millis(50));
                    }
                    _ => {
                        log::warn!("worker rank {rank} timed out on shutdown — killing");
                        let _ = child.kill();
                        let _ = child.wait();
                        break;
                    }
                }
            }
        }
    }
}

fn dummy_file() -> std::fs::File {
    // /dev/null wrapped in a File so std::mem::replace has something to swap in.
    // Used by Drop only — never read or written.
    std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/null")
        .expect("/dev/null open for dummy_file")
}

fn spawn_workers(model_path: &str, world_size: usize) -> Result<WorkerChildren> {
    use std::os::fd::AsRawFd;

    let exe = std::env::current_exe().context("current_exe")?;
    let ordinals = cuda_ordinals(world_size);
    let mut children = Vec::with_capacity(world_size - 1);

    for rank in 1..world_size {
        // Parent -> child pipe. Child holds the read end, parent the write end.
        // When parent drops the write end (or dies), child's read() returns EOF
        // and the worker exits.
        let mut fds = [0i32; 2];
        // SAFETY: pipe(2) writes two valid owned fds into `fds`.
        let ret = unsafe { libc::pipe(fds.as_mut_ptr()) };
        if ret != 0 {
            anyhow::bail!(
                "pipe(2) for worker rank {rank} failed: {}",
                std::io::Error::last_os_error()
            );
        }
        // SAFETY: pipe(2) returned two valid owned fds.
        use std::os::fd::FromRawFd;
        let child_read_end = unsafe { std::fs::File::from_raw_fd(fds[0]) };
        let parent_write_end = unsafe { std::fs::File::from_raw_fd(fds[1]) };
        let child_read_raw = child_read_end.as_raw_fd();

        let cuda_ordinal = ordinals.get(rank).copied().unwrap_or(rank);

        let mut cmd = std::process::Command::new(&exe);
        // Forward every CLI arg so the child sees identical args; it short-
        // circuits to worker_entry() because ARLE_WORKER_RANK is set, so the
        // other args (model path, etc.) are inert but the parser still accepts
        // them if it ever reaches clap.
        for arg in std::env::args().skip(1) {
            cmd.arg(arg);
        }
        cmd.env("ARLE_WORKER_RANK", rank.to_string());
        cmd.env("ARLE_WORKER_PARENT_FD", child_read_raw.to_string());
        cmd.env("ARLE_WORKER_MODEL_PATH", model_path);
        cmd.env("WORLD_SIZE", world_size.to_string());
        cmd.env("INFER_CUDA_DEVICE", cuda_ordinal.to_string());
        // The DSv4 executor reads its TP rank from INFER_TP_RANK; bind it to this
        // worker's rank so the per-rank EP split + NCCL rank match.
        cmd.env("INFER_TP_RANK", rank.to_string());
        cmd.env("INFER_TP_SIZE", world_size.to_string());
        // MASTER_ADDR/PORT + INFER_NCCL_ID_FILE/INFER_NCCL_UNIQUE_ID are
        // inherited from the coordinator's env (set before spawning workers).

        let child = cmd
            .spawn()
            .with_context(|| format!("spawn worker rank {rank}"))?;

        // After spawn the parent doesn't need the child-side read fd; the child
        // inherited it. Closing it here would EOF the child immediately.
        drop(child_read_end);
        let pid = child.id();
        children.push((rank, child, parent_write_end));
        log::info!("spawned worker rank {rank} pid={pid} cuda_ordinal={cuda_ordinal}");
    }

    Ok(WorkerChildren { children })
}
