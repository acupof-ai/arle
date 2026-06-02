//! Multiproc-serve control-plane relay (phase B-1 commit C scaffolding).
//!
//! TCP transport for shipping `IncomingRequest`-equivalent payloads from
//! the coordinator process (rank 0) to worker processes (rank 1..N-1).
//! NCCL handles the data-plane forward sync; this module handles the
//! per-request control-plane fanout.
//!
//! Wire format: length-prefixed JSON envelopes.
//!   [u32 LE: payload_len][payload_len bytes JSON].
//!
//! Why JSON not bincode: serde_json is already a workspace dep; the
//! per-request volume is small (~1 KB) and the latency hit vs binary
//! is negligible (~10 µs encode + ~10 µs decode at SLO scale).
//!
//! Lifetime: coordinator binds at boot, waits for N-1 worker connects,
//! then becomes write-only. Workers connect at boot, then read in a loop.
//! On coordinator drop, all worker streams EOF and workers exit.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use crate::server_engine::CompletionStreamDelta;

/// Free-port picker mirroring `infer/src/distributed.rs`. Binds 127.0.0.1:0,
/// reads the assigned port, drops the listener. Caller races to use the
/// port (negligible window on single host).
pub fn pick_free_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").context("relay port reservation")?;
    let port = listener.local_addr().context("relay port read")?.port();
    drop(listener);
    Ok(port)
}

/// Wire envelope for relay messages. Tagged enum so future variants
/// (Heartbeat, Stats, Abort) compose cleanly without protocol breaks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RelayEnvelope {
    /// Worker identity handshake. The coordinator consumes this during
    /// `PendingRelayCoordinator::accept` and indexes streams by rank. It is
    /// never forwarded to the scheduler loop.
    WorkerHello { rank: usize, world_size: usize },
    /// (Phase B-1 commit C.1 boot-ping variant.) Lightweight envelope
    /// used by C.3 boot validation; retained for backward compat.
    Request {
        request_id: u64,
        prompt_tokens: Vec<u32>,
        max_new_tokens: u32,
        /// Opaque JSON-encoded SamplingParams — kept as a Value so this
        /// module doesn't depend on `crate::sampler`.
        sampling: serde_json::Value,
    },
    /// (Phase B-1 commit C.4.) Full per-request fanout payload — the
    /// coordinator's HTTP submission path serializes one of these per
    /// incoming chat completion; every worker reconstructs an
    /// `IncomingRequest` from it on receive.
    Request2 { wire: WireRequest },
    /// Token-owned request payload. The coordinator sends this only to the
    /// selected DP-owner ranks, not to every worker rank.
    RequestOwned {
        wire: WireRequest,
        shard_rank: usize,
        shard_world_size: usize,
        emits_visible_output: bool,
    },
    /// Worker-to-coordinator output for a remote visible-output owner rank.
    Completion {
        request_id: u64,
        delta: CompletionStreamDelta,
    },
    /// Graceful shutdown notice; workers should drain in-flight then
    /// exit. (Coordinator can also just drop streams; this is the
    /// nicer path for telemetry/log capture.)
    Shutdown,
}

/// Serializable counterpart of `crate::scheduler::IncomingRequest` (minus
/// the channel + DistributedRequestCoordination + tracing fields, which
/// are reconstructed worker-side). Captures the minimum data needed for
/// a worker rank to schedule the same forward path as rank 0.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WireRequest {
    pub request_id: u64,
    pub prompt: String,
    pub prompt_tokens: Option<Vec<u32>>,
    pub max_tokens: usize,
    pub sampling: WireSamplingParams,
    pub stop: Option<Vec<String>>,
    /// `RequestPriority` discriminant (0=Low, 1=Normal, 2=High).
    pub priority: u8,
    /// `SessionId` as a string (it wraps `Arc<str>` internally).
    pub session_id: Option<String>,
}

/// Serializable counterpart of `crate::sampler::SamplingParams`. Field-
/// by-field mirror; conversion helpers in
/// `crate::request_handle::wire_conversion` (C.4.3+) translate the two.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WireSamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub ignore_eos: bool,
    pub stop_token_ids: Vec<u32>,
    pub seed: Option<u64>,
    pub max_new_tokens: Option<usize>,
}

/// Coordinator-side TCP relay. Binds a port, accepts N-1 worker
/// connections at boot, then provides `broadcast()` to send envelopes
/// to every worker stream.
pub struct RelayCoordinator {
    port: u16,
    workers: BTreeMap<usize, TcpStream>,
    completion_sinks:
        Arc<Mutex<HashMap<u64, tokio::sync::mpsc::UnboundedSender<CompletionStreamDelta>>>>,
    completion_shutdown: Arc<AtomicBool>,
}

/// Pending coordinator state — listener is bound and port is known but
/// workers have not yet connected. Caller uses this to publish the port
/// (env var, file, etc.) so children can connect, then calls
/// `accept(world_size, timeout)` to finalize a `RelayCoordinator`.
pub struct PendingRelayCoordinator {
    port: u16,
    listener: TcpListener,
}

impl PendingRelayCoordinator {
    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn accept(self, world_size: usize, accept_timeout: Duration) -> Result<RelayCoordinator> {
        if world_size < 2 {
            bail!("RelayCoordinator needs world_size >= 2 (got {world_size})");
        }
        let expected = world_size - 1;
        let mut workers = BTreeMap::new();
        let deadline = Instant::now() + accept_timeout;

        while workers.len() < expected {
            match self.listener.accept() {
                Ok((mut stream, addr)) => {
                    stream
                        .set_nonblocking(false)
                        .context("worker stream set_nonblocking(false)")?;
                    let hello = read_envelope(&mut stream)
                        .with_context(|| format!("RelayCoordinator read hello from {addr}"))?;
                    let Some(RelayEnvelope::WorkerHello {
                        rank,
                        world_size: worker_world_size,
                    }) = hello
                    else {
                        bail!("RelayCoordinator expected worker hello from {addr}");
                    };
                    if worker_world_size != world_size {
                        bail!(
                            "RelayCoordinator worker rank {rank} reported world_size={worker_world_size}, expected {world_size}"
                        );
                    }
                    if rank == 0 || rank >= world_size {
                        bail!("RelayCoordinator worker rank {rank} out of range [1, {world_size})");
                    }
                    if workers.insert(rank, stream).is_some() {
                        bail!("RelayCoordinator duplicate worker rank {rank}");
                    }
                    log::info!(
                        "[relay-coordinator] worker rank {} ({}/{}) connected from {}",
                        rank,
                        workers.len(),
                        expected,
                        addr
                    );
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        bail!(
                            "RelayCoordinator timed out after {accept_timeout:?} waiting for \
                             worker connects ({}/{expected} so far)",
                            workers.len()
                        );
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(err) => return Err(err).context("RelayCoordinator accept"),
            }
        }
        let completion_sinks = Arc::new(Mutex::new(HashMap::new()));
        let completion_shutdown = Arc::new(AtomicBool::new(false));
        for (&rank, stream) in &workers {
            let stream = stream
                .try_clone()
                .with_context(|| format!("RelayCoordinator clone worker rank {rank} stream"))?;
            spawn_completion_reader(
                rank,
                stream,
                Arc::clone(&completion_sinks),
                Arc::clone(&completion_shutdown),
            );
        }
        Ok(RelayCoordinator {
            port: self.port,
            workers,
            completion_sinks,
            completion_shutdown,
        })
    }
}

impl Drop for RelayCoordinator {
    fn drop(&mut self) {
        self.completion_shutdown.store(true, Ordering::Relaxed);
    }
}

impl RelayCoordinator {
    /// Bind to a free port WITHOUT yet accepting any worker connections.
    /// Returns a `PendingRelayCoordinator` whose port can be published to
    /// children (env var, file). Call `pending.accept(world_size, timeout)`
    /// after spawning workers to finalize.
    ///
    /// Two-phase API exists because coordinator must `bind` before spawning
    /// workers (so child env carries the port) but `accept` must happen
    /// after spawn (children need to connect first).
    pub fn bind() -> Result<PendingRelayCoordinator> {
        let port = pick_free_port()?;
        let listener = TcpListener::bind(("127.0.0.1", port))
            .with_context(|| format!("RelayCoordinator bind port {port}"))?;
        listener
            .set_nonblocking(true)
            .context("RelayCoordinator set_nonblocking on listener")?;
        Ok(PendingRelayCoordinator { port, listener })
    }

    /// Convenience: bind + accept in one call. Only useful when the
    /// children are already running (e.g. in tests where both sides race
    /// at startup).
    pub fn bind_and_accept(world_size: usize, accept_timeout: Duration) -> Result<Self> {
        Self::bind()?.accept(world_size, accept_timeout)
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    pub fn worker_ranks(&self) -> Vec<usize> {
        self.workers.keys().copied().collect()
    }

    pub fn register_completion_sink(
        &mut self,
        request_id: u64,
        sink: tokio::sync::mpsc::UnboundedSender<CompletionStreamDelta>,
    ) -> Result<()> {
        let mut sinks = self
            .completion_sinks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if sinks.insert(request_id, sink).is_some() {
            bail!("RelayCoordinator duplicate completion sink for request_id={request_id}");
        }
        Ok(())
    }

    pub fn unregister_completion_sink(&mut self, request_id: u64) {
        let mut sinks = self
            .completion_sinks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        sinks.remove(&request_id);
    }

    /// Broadcast an envelope to every connected worker. On the first
    /// write error, returns immediately — caller decides whether to
    /// drop the coordinator (workers exit) or retry.
    pub fn broadcast(&mut self, envelope: &RelayEnvelope) -> Result<()> {
        for (&rank, stream) in self.workers.iter_mut() {
            write_envelope(stream, envelope).with_context(|| {
                format!("RelayCoordinator write envelope to worker rank {rank}")
            })?;
        }
        Ok(())
    }

    /// Send an envelope to a selected global-rank set. Rank 0 is local to the
    /// coordinator process and is intentionally skipped.
    pub fn send_to_ranks(&mut self, ranks: &[usize], envelope: &RelayEnvelope) -> Result<()> {
        for &rank in ranks {
            if rank == 0 {
                continue;
            }
            let stream = self
                .workers
                .get_mut(&rank)
                .with_context(|| format!("RelayCoordinator missing worker rank {rank}"))?;
            write_envelope(stream, envelope).with_context(|| {
                format!("RelayCoordinator write envelope to worker rank {rank}")
            })?;
        }
        Ok(())
    }
}

/// Worker-side TCP relay. Connects to the coordinator at boot, then
/// provides `recv()` to read one envelope at a time.
pub struct RelayWorker {
    stream: TcpStream,
}

impl RelayWorker {
    /// Connect to the coordinator's relay port. Retries up to
    /// `connect_timeout` since the coordinator may not have called
    /// `bind_and_accept` yet at the moment the worker fires off.
    pub fn connect(coordinator: SocketAddr, connect_timeout: Duration) -> Result<Self> {
        Self::connect_with_rank(coordinator, connect_timeout, 1, 2)
    }

    pub fn connect_with_rank(
        coordinator: SocketAddr,
        connect_timeout: Duration,
        rank: usize,
        world_size: usize,
    ) -> Result<Self> {
        let deadline = Instant::now() + connect_timeout;
        let mut last_err = None;
        while Instant::now() < deadline {
            match TcpStream::connect_timeout(&coordinator, Duration::from_millis(200)) {
                Ok(mut stream) => {
                    write_envelope(
                        &mut stream,
                        &RelayEnvelope::WorkerHello { rank, world_size },
                    )
                    .with_context(|| {
                        format!("RelayWorker rank {rank}/{world_size} write hello to {coordinator}")
                    })?;
                    log::info!(
                        "[relay-worker] rank {rank}/{world_size} connected to coordinator at {coordinator}"
                    );
                    return Ok(Self { stream });
                }
                Err(err) => {
                    last_err = Some(err);
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }
        Err(last_err
            .map(std::convert::Into::into)
            .unwrap_or_else(|| anyhow::anyhow!("RelayWorker connect timeout")))
    }

    /// Read one envelope. Returns `Ok(None)` on coordinator EOF (clean
    /// shutdown); `Err` on protocol violation or transport failure.
    pub fn recv(&mut self) -> Result<Option<RelayEnvelope>> {
        read_envelope(&mut self.stream)
    }

    pub fn send(&mut self, envelope: &RelayEnvelope) -> Result<()> {
        write_envelope(&mut self.stream, envelope)
    }

    pub fn try_clone(&self) -> Result<Self> {
        let stream = self
            .stream
            .try_clone()
            .context("RelayWorker clone stream")?;
        Ok(Self { stream })
    }
}

fn spawn_completion_reader(
    rank: usize,
    mut stream: TcpStream,
    completion_sinks: Arc<
        Mutex<HashMap<u64, tokio::sync::mpsc::UnboundedSender<CompletionStreamDelta>>>,
    >,
    shutdown: Arc<AtomicBool>,
) {
    let _ = stream.set_read_timeout(Some(Duration::from_millis(100)));
    let _ = std::thread::Builder::new()
        .name(format!("arle-relay-rank{rank}-completion-reader"))
        .spawn(move || loop {
            match read_envelope(&mut stream) {
                Ok(Some(RelayEnvelope::Completion { request_id, delta })) => {
                    let done = delta.finish_reason.is_some() || delta.error.is_some();
                    let sink = {
                        let sinks = completion_sinks
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        sinks.get(&request_id).cloned()
                    };
                    let remove = match sink {
                        Some(tx) => tx.send(delta).is_err() || done,
                        None => {
                            log::warn!(
                                "[relay-coordinator] completion for unknown request_id={request_id} from rank {rank}"
                            );
                            done
                        }
                    };
                    if remove {
                        let mut sinks = completion_sinks
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        sinks.remove(&request_id);
                    }
                }
                Ok(Some(other)) => {
                    log::warn!(
                        "[relay-coordinator] unexpected worker->coordinator envelope from rank {rank}: {other:?}"
                    );
                }
                Ok(None) => {
                    log::info!("[relay-coordinator] worker rank {rank} completion stream EOF");
                    break;
                }
                Err(err) if is_timeout_error(&err) => {
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }
                }
                Err(err) => {
                    log::warn!(
                        "[relay-coordinator] worker rank {rank} completion reader failed: {err:#}"
                    );
                    break;
                }
            }
        });
}

fn is_timeout_error(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .map(|io| {
                matches!(
                    io.kind(),
                    std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                )
            })
            .unwrap_or(false)
    })
}

fn write_envelope(stream: &mut TcpStream, envelope: &RelayEnvelope) -> Result<()> {
    let payload = serde_json::to_vec(envelope).context("relay serialize envelope")?;
    let header = (payload.len() as u32).to_le_bytes();
    stream.write_all(&header).context("relay write header")?;
    stream.write_all(&payload).context("relay write payload")?;
    Ok(())
}

fn read_envelope(stream: &mut TcpStream) -> Result<Option<RelayEnvelope>> {
    let mut header = [0u8; 4];
    match stream.read_exact(&mut header) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Ok(None);
        }
        Err(err) => return Err(err).context("relay read header"),
    }
    let len = u32::from_le_bytes(header) as usize;
    if len == 0 {
        bail!("relay received empty envelope (corrupt stream?)");
    }
    if len > 64 * 1024 * 1024 {
        bail!(
            "relay envelope length {len} exceeds 64 MiB sanity cap — likely corrupted stream or version mismatch"
        );
    }
    let mut payload = vec![0u8; len];
    stream
        .read_exact(&mut payload)
        .context("relay read payload")?;
    let envelope: RelayEnvelope =
        serde_json::from_slice(&payload).context("relay deserialize envelope")?;
    Ok(Some(envelope))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn envelope_round_trip() {
        let env = RelayEnvelope::Request {
            request_id: 42,
            prompt_tokens: vec![1, 2, 3, 4],
            max_new_tokens: 16,
            sampling: serde_json::json!({"temperature": 0.7}),
        };
        let bytes = serde_json::to_vec(&env).unwrap();
        let decoded: RelayEnvelope = serde_json::from_slice(&bytes).unwrap();
        match decoded {
            RelayEnvelope::Request {
                request_id,
                prompt_tokens,
                max_new_tokens,
                sampling,
            } => {
                assert_eq!(request_id, 42);
                assert_eq!(prompt_tokens, vec![1, 2, 3, 4]);
                assert_eq!(max_new_tokens, 16);
                assert_eq!(sampling["temperature"], 0.7);
            }
            other => panic!("expected Request, got {other:?}"),
        }
    }

    #[test]
    fn coordinator_worker_round_trip() {
        let world_size = 2;
        let pending = RelayCoordinator::bind().unwrap();
        let coord_addr: SocketAddr = format!("127.0.0.1:{}", pending.port()).parse().unwrap();

        let worker_thread = thread::spawn(move || {
            let mut worker =
                RelayWorker::connect_with_rank(coord_addr, Duration::from_secs(5), 1, world_size)
                    .unwrap();
            // Receive one envelope, then EOF on shutdown.
            let env = worker.recv().unwrap().expect("envelope");
            match env {
                RelayEnvelope::Request {
                    request_id,
                    prompt_tokens,
                    ..
                } => {
                    assert_eq!(request_id, 7);
                    assert_eq!(prompt_tokens, vec![100, 200]);
                }
                _ => panic!("expected Request"),
            }
            let next = worker.recv().unwrap();
            assert!(next.is_none(), "expected EOF after coordinator drop");
        });

        let mut coord = pending
            .accept(world_size, Duration::from_secs(5))
            .expect("worker connected within 5s");
        assert_eq!(coord.worker_ranks(), vec![1]);
        coord
            .broadcast(&RelayEnvelope::Request {
                request_id: 7,
                prompt_tokens: vec![100, 200],
                max_new_tokens: 1,
                sampling: serde_json::json!({}),
            })
            .unwrap();
        drop(coord);

        worker_thread.join().unwrap();
    }

    #[test]
    fn coordinator_targeted_send_reaches_only_selected_rank() {
        let world_size = 3;
        let pending = RelayCoordinator::bind().unwrap();
        let coord_addr: SocketAddr = format!("127.0.0.1:{}", pending.port()).parse().unwrap();

        let rank1_thread = thread::spawn(move || {
            let mut worker =
                RelayWorker::connect_with_rank(coord_addr, Duration::from_secs(5), 1, world_size)
                    .unwrap();
            assert!(worker.recv().unwrap().is_none());
        });

        let coord_addr: SocketAddr = format!("127.0.0.1:{}", pending.port()).parse().unwrap();
        let rank2_thread = thread::spawn(move || {
            let mut worker =
                RelayWorker::connect_with_rank(coord_addr, Duration::from_secs(5), 2, world_size)
                    .unwrap();
            let env = worker.recv().unwrap().expect("targeted envelope");
            match env {
                RelayEnvelope::Request {
                    request_id,
                    prompt_tokens,
                    ..
                } => {
                    assert_eq!(request_id, 9);
                    assert_eq!(prompt_tokens, vec![9]);
                }
                other => panic!("expected Request, got {other:?}"),
            }
            assert!(worker.recv().unwrap().is_none());
        });

        let mut coord = pending
            .accept(world_size, Duration::from_secs(5))
            .expect("workers connected within 5s");
        assert_eq!(coord.worker_ranks(), vec![1, 2]);
        coord
            .send_to_ranks(
                &[2],
                &RelayEnvelope::Request {
                    request_id: 9,
                    prompt_tokens: vec![9],
                    max_new_tokens: 1,
                    sampling: serde_json::json!({}),
                },
            )
            .unwrap();
        drop(coord);

        rank1_thread.join().unwrap();
        rank2_thread.join().unwrap();
    }

    #[test]
    fn coordinator_dispatches_worker_completion_to_registered_sink() {
        let world_size = 2;
        let pending = RelayCoordinator::bind().unwrap();
        let coord_addr: SocketAddr = format!("127.0.0.1:{}", pending.port()).parse().unwrap();

        let worker_thread = thread::spawn(move || {
            let mut worker =
                RelayWorker::connect_with_rank(coord_addr, Duration::from_secs(5), 1, world_size)
                    .unwrap();
            match worker.recv().unwrap().expect("request signal") {
                RelayEnvelope::Request { request_id, .. } => assert_eq!(request_id, 11),
                other => panic!("expected request signal, got {other:?}"),
            }
            worker
                .send(&RelayEnvelope::Completion {
                    request_id: 11,
                    delta: CompletionStreamDelta::text("remote".to_string()),
                })
                .unwrap();
            worker
                .send(&RelayEnvelope::Completion {
                    request_id: 11,
                    delta: CompletionStreamDelta {
                        text_delta: String::new(),
                        finish_reason: Some(crate::server_engine::FinishReason::Length),
                        usage: None,
                        logprob: None,
                        token_ids: Vec::new(),
                        error: None,
                    },
                })
                .unwrap();
        });

        let mut coord = pending
            .accept(world_size, Duration::from_secs(5))
            .expect("worker connected within 5s");
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        coord.register_completion_sink(11, tx).unwrap();
        coord
            .send_to_ranks(
                &[1],
                &RelayEnvelope::Request {
                    request_id: 11,
                    prompt_tokens: vec![11],
                    max_new_tokens: 1,
                    sampling: serde_json::json!({}),
                },
            )
            .unwrap();
        let first = rx.blocking_recv().expect("text completion");
        assert_eq!(first.text_delta, "remote");
        let second = rx.blocking_recv().expect("finish completion");
        assert_eq!(
            second.finish_reason,
            Some(crate::server_engine::FinishReason::Length)
        );

        worker_thread.join().unwrap();
    }
}
