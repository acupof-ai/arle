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
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

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
    /// Submit a new chat/completion request to every rank's scheduler.
    /// `prompt_tokens` are the post-tokenized prompt; sampling params
    /// already serialized in `sampling`.
    Request {
        request_id: u64,
        prompt_tokens: Vec<u32>,
        max_new_tokens: u32,
        /// Opaque JSON-encoded SamplingParams — kept as a Value so this
        /// module doesn't depend on `crate::sampler`.
        sampling: serde_json::Value,
    },
    /// Graceful shutdown notice; workers should drain in-flight then
    /// exit. (Coordinator can also just drop streams; this is the
    /// nicer path for telemetry/log capture.)
    Shutdown,
}

/// Coordinator-side TCP relay. Binds a port, accepts N-1 worker
/// connections at boot, then provides `broadcast()` to send envelopes
/// to every worker stream.
pub struct RelayCoordinator {
    port: u16,
    workers: Vec<TcpStream>,
}

impl RelayCoordinator {
    /// Bind to a free port and accept exactly `world_size - 1` worker
    /// connections. Blocks until all workers connect or `accept_timeout`
    /// elapses (returns error on timeout).
    ///
    /// `world_size` is the FULL distributed world (coordinator + workers);
    /// the coordinator itself doesn't connect to its own listener.
    pub fn bind_and_accept(world_size: usize, accept_timeout: Duration) -> Result<Self> {
        if world_size < 2 {
            bail!("RelayCoordinator needs world_size >= 2 (got {world_size})");
        }
        let port = pick_free_port()?;
        let listener = TcpListener::bind(("127.0.0.1", port))
            .with_context(|| format!("RelayCoordinator bind port {port}"))?;
        listener
            .set_nonblocking(true)
            .context("RelayCoordinator set_nonblocking on listener")?;

        let expected = world_size - 1;
        let mut workers = Vec::with_capacity(expected);
        let deadline = Instant::now() + accept_timeout;

        while workers.len() < expected {
            match listener.accept() {
                Ok((stream, addr)) => {
                    stream
                        .set_nonblocking(false)
                        .context("worker stream set_nonblocking(false)")?;
                    log::info!(
                        "[relay-coordinator] worker {}/{} connected from {}",
                        workers.len() + 1,
                        expected,
                        addr
                    );
                    workers.push(stream);
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
        Ok(Self { port, workers })
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Broadcast an envelope to every connected worker. On the first
    /// write error, returns immediately — caller decides whether to
    /// drop the coordinator (workers exit) or retry.
    pub fn broadcast(&mut self, envelope: &RelayEnvelope) -> Result<()> {
        let payload =
            serde_json::to_vec(envelope).context("RelayCoordinator serialize envelope")?;
        let header = (payload.len() as u32).to_le_bytes();
        for (idx, stream) in self.workers.iter_mut().enumerate() {
            stream
                .write_all(&header)
                .with_context(|| format!("RelayCoordinator write header to worker {idx}"))?;
            stream
                .write_all(&payload)
                .with_context(|| format!("RelayCoordinator write payload to worker {idx}"))?;
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
        let deadline = Instant::now() + connect_timeout;
        let mut last_err = None;
        while Instant::now() < deadline {
            match TcpStream::connect_timeout(&coordinator, Duration::from_millis(200)) {
                Ok(stream) => {
                    log::info!("[relay-worker] connected to coordinator at {coordinator}");
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
        let mut header = [0u8; 4];
        match self.stream.read_exact(&mut header) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            }
            Err(err) => return Err(err).context("RelayWorker read header"),
        }
        let len = u32::from_le_bytes(header) as usize;
        if len == 0 {
            bail!("RelayWorker received empty envelope (corrupt stream?)");
        }
        if len > 64 * 1024 * 1024 {
            bail!(
                "RelayWorker envelope length {len} exceeds 64 MiB sanity cap — likely corrupted \
                 stream or version mismatch"
            );
        }
        let mut payload = vec![0u8; len];
        self.stream
            .read_exact(&mut payload)
            .context("RelayWorker read payload")?;
        let envelope: RelayEnvelope =
            serde_json::from_slice(&payload).context("RelayWorker deserialize envelope")?;
        Ok(Some(envelope))
    }
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
        let port = pick_free_port().unwrap();
        let coord_addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();

        // Bind first, then spawn worker. Bind doesn't block on accept until
        // we call accept(); the worker can connect immediately.
        let listener = TcpListener::bind(("127.0.0.1", port)).unwrap();
        listener.set_nonblocking(true).unwrap();

        let worker_thread = thread::spawn(move || {
            let mut worker = RelayWorker::connect(coord_addr, Duration::from_secs(5)).unwrap();
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

        // Coordinator side: accept the worker (poll the listener since we set non-blocking).
        let mut accepted = None;
        for _ in 0..100 {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(false).unwrap();
                    accepted = Some(stream);
                    break;
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(20));
                }
                Err(err) => panic!("accept failed: {err}"),
            }
        }
        let stream = accepted.expect("worker connected within 2s");
        let mut coord = RelayCoordinator {
            port,
            workers: vec![stream],
        };
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
        let _ = world_size; // silence unused var
    }
}
