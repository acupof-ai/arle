//! TCP rendezvous protocol — single-host (and later multi-host) bootstrap for
//! N worker ranks.
//!
//! Wire protocol (rank 0 = server, ranks 1..N-1 = clients):
//!
//! ```text
//!   rank 0  (server)                     rank k  (client, k >= 1)
//!   --------------                       --------------------------
//!   bind(addr)                           connect(addr)  [retry loop]
//!   accept until N-1 sockets             ─── TCP ESTABLISHED ───
//!   write 128B unique_id  ──────────►    read 128B unique_id
//!                                        write 1B 0xAA  ─────────►
//!   read 1B per peer (must be 0xAA)
//!   return Ok(())                        return Ok(unique_id)
//! ```
//!
//! All sockets carry a 30s read/write timeout. Errors include the affected
//! rank index (server side) so a hung peer is identifiable from logs.
//!
//! This is the local-network analogue of SGLang's `_create_global_tcp_store`
//! (`parallel_state.py:1591`), distilled to the minimum F0 needs. F1 will
//! feed the NCCL `ncclUniqueId` (also 128 bytes) through here.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

/// Size of the rendezvous payload in bytes. Matches `ncclUniqueId` so F1 can
/// pass the NCCL handle without an additional copy or framing layer.
pub const UNIQUE_ID_BYTES: usize = 128;

const BARRIER_ACK: u8 = 0xAA;
const SOCKET_TIMEOUT: Duration = Duration::from_secs(30);
const CLIENT_RETRY_INTERVAL: Duration = Duration::from_secs(1);
const CLIENT_RETRY_ATTEMPTS: u32 = 10;

/// Rank-0 side of the rendezvous.
pub struct RendezvousServer {
    listener: TcpListener,
    world_size: usize,
    peers: Vec<TcpStream>,
}

/// Rank-k (k >= 1) side of the rendezvous.
pub struct RendezvousClient {
    stream: TcpStream,
}

impl RendezvousServer {
    /// Bind on `addr`. Use `127.0.0.1:0` to request an ephemeral local port,
    /// then publish [`local_addr`](Self::local_addr) to peers out-of-band.
    pub fn bind(addr: impl ToSocketAddrs, world_size: usize) -> Result<Self> {
        if world_size == 0 {
            bail!("rendezvous: world_size must be >= 1");
        }
        let listener =
            TcpListener::bind(addr).context("rendezvous server: bind TCP listener failed")?;
        Ok(Self {
            listener,
            world_size,
            peers: Vec::with_capacity(world_size.saturating_sub(1)),
        })
    }

    /// Address the server is bound to (useful when `bind` was called with
    /// port `0`).
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.listener
            .local_addr()
            .context("rendezvous server: local_addr lookup failed")
    }

    /// Block until N-1 clients connect, then broadcast `unique_id` and wait
    /// for a barrier ack from each. Uses [`SOCKET_TIMEOUT`] as the deadline
    /// for both accept and the post-accept read/write phases.
    pub fn rendezvous(&mut self, unique_id: &[u8; UNIQUE_ID_BYTES]) -> Result<()> {
        self.rendezvous_with_timeout(unique_id, SOCKET_TIMEOUT)
    }

    /// `rendezvous` with an explicit deadline. F1 (NCCL init) and tests use
    /// this when the default 30s isn't appropriate.
    pub fn rendezvous_with_timeout(
        &mut self,
        unique_id: &[u8; UNIQUE_ID_BYTES],
        timeout: Duration,
    ) -> Result<()> {
        let expected_peers = self.world_size - 1;
        self.listener
            .set_nonblocking(true)
            .context("rendezvous server: set_nonblocking failed")?;
        let deadline = Instant::now() + timeout;
        for peer_idx in 0..expected_peers {
            let (stream, _addr) = loop {
                match self.listener.accept() {
                    Ok(pair) => break pair,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        if Instant::now() >= deadline {
                            bail!(
                                "rendezvous server: accept rank {} timed out after {:?} ({} of {} peers connected)",
                                peer_idx + 1,
                                timeout,
                                self.peers.len(),
                                expected_peers
                            );
                        }
                        thread::sleep(Duration::from_millis(20));
                    }
                    Err(e) => {
                        return Err(e).with_context(|| {
                            format!("rendezvous server: accept rank {} failed", peer_idx + 1)
                        });
                    }
                }
            };
            stream.set_nonblocking(false).with_context(|| {
                format!(
                    "rendezvous server: clear nonblocking for rank {}",
                    peer_idx + 1
                )
            })?;
            stream.set_read_timeout(Some(timeout)).with_context(|| {
                format!(
                    "rendezvous server: set_read_timeout for rank {}",
                    peer_idx + 1
                )
            })?;
            stream.set_write_timeout(Some(timeout)).with_context(|| {
                format!(
                    "rendezvous server: set_write_timeout for rank {}",
                    peer_idx + 1
                )
            })?;
            self.peers.push(stream);
        }

        for (peer_idx, stream) in self.peers.iter_mut().enumerate() {
            let rank = peer_idx + 1;
            stream
                .write_all(unique_id)
                .with_context(|| format!("rendezvous server: write unique_id to rank {rank} failed (timeout or peer disconnect)"))?;
        }

        let mut ack = [0u8; 1];
        for (peer_idx, stream) in self.peers.iter_mut().enumerate() {
            let rank = peer_idx + 1;
            stream
                .read_exact(&mut ack)
                .with_context(|| format!("rendezvous server: read barrier ack from rank {rank} failed (timeout or peer disconnect)"))?;
            if ack[0] != BARRIER_ACK {
                bail!(
                    "rendezvous server: rank {rank} sent barrier byte 0x{:02X}, expected 0x{:02X}",
                    ack[0],
                    BARRIER_ACK
                );
            }
        }
        Ok(())
    }
}

impl RendezvousClient {
    /// Connect to rank 0. Retries up to 10 times with a 1s interval to
    /// tolerate spawn-order races where a peer wakes before rank 0 binds.
    pub fn connect(addr: impl ToSocketAddrs) -> Result<Self> {
        let addrs: Vec<SocketAddr> = addr
            .to_socket_addrs()
            .context("rendezvous client: resolve server address failed")?
            .collect();
        if addrs.is_empty() {
            bail!("rendezvous client: server address resolved to zero socket addrs");
        }

        let mut last_err: Option<std::io::Error> = None;
        let started = Instant::now();
        for attempt in 0..CLIENT_RETRY_ATTEMPTS {
            for sa in &addrs {
                match TcpStream::connect_timeout(sa, SOCKET_TIMEOUT) {
                    Ok(stream) => {
                        stream
                            .set_read_timeout(Some(SOCKET_TIMEOUT))
                            .context("rendezvous client: set_read_timeout failed")?;
                        stream
                            .set_write_timeout(Some(SOCKET_TIMEOUT))
                            .context("rendezvous client: set_write_timeout failed")?;
                        return Ok(Self { stream });
                    }
                    Err(err) => last_err = Some(err),
                }
            }
            if attempt + 1 < CLIENT_RETRY_ATTEMPTS {
                thread::sleep(CLIENT_RETRY_INTERVAL);
            }
        }
        let waited = started.elapsed();
        let err = last_err.expect("retry loop must record at least one error");
        Err(anyhow::Error::new(err).context(format!(
            "rendezvous client: failed to connect to {:?} after {} attempts ({:.1?} elapsed)",
            addrs, CLIENT_RETRY_ATTEMPTS, waited
        )))
    }

    /// Read the rendezvous payload, then send the barrier ack.
    pub fn rendezvous(&mut self) -> Result<[u8; UNIQUE_ID_BYTES]> {
        let mut id = [0u8; UNIQUE_ID_BYTES];
        self.stream
            .read_exact(&mut id)
            .context("rendezvous client: read unique_id failed (timeout or rank-0 disconnect)")?;
        self.stream
            .write_all(&[BARRIER_ACK])
            .context("rendezvous client: write barrier ack failed")?;
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::mpsc::sync_channel;
    use std::time::Duration;

    fn run_with_timeout<T: Send + 'static>(
        label: &'static str,
        timeout: Duration,
        f: impl FnOnce() -> T + Send + 'static,
    ) -> T {
        let (tx, rx) = sync_channel::<T>(1);
        let handle = thread::spawn(move || {
            let value = f();
            let _ = tx.send(value);
        });
        let value = rx
            .recv_timeout(timeout)
            .unwrap_or_else(|_| panic!("{label} did not finish within {timeout:?}"));
        handle.join().expect("thread panicked");
        value
    }

    #[test]
    fn unique_id_size_constant() {
        assert_eq!(UNIQUE_ID_BYTES, 128);
    }

    #[test]
    fn rendezvous_world_size_2() {
        let (addr_tx, addr_rx) = sync_channel::<SocketAddr>(1);
        let unique_id = [0x42u8; UNIQUE_ID_BYTES];

        let server = thread::spawn(move || -> Result<()> {
            let mut server = RendezvousServer::bind("127.0.0.1:0", 2)?;
            addr_tx.send(server.local_addr()?).expect("send local_addr");
            server.rendezvous(&unique_id)
        });

        let client = thread::spawn(move || -> Result<[u8; UNIQUE_ID_BYTES]> {
            let addr = addr_rx.recv().expect("recv local_addr");
            let mut client = RendezvousClient::connect(addr)?;
            client.rendezvous()
        });

        let (server_res, client_res) =
            run_with_timeout("world_size_2", Duration::from_secs(5), move || {
                (
                    server.join().expect("server panic"),
                    client.join().expect("client panic"),
                )
            });
        server_res.expect("server rendezvous");
        let received = client_res.expect("client rendezvous");
        assert_eq!(received, unique_id);
    }

    #[test]
    fn rendezvous_world_size_4() {
        let (addr_tx, addr_rx) = sync_channel::<SocketAddr>(3);
        let unique_id = [0x37u8; UNIQUE_ID_BYTES];

        let server = thread::spawn(move || -> Result<()> {
            let mut server = RendezvousServer::bind("127.0.0.1:0", 4)?;
            let addr = server.local_addr()?;
            for _ in 0..3 {
                addr_tx.send(addr).expect("send local_addr");
            }
            server.rendezvous(&unique_id)
        });

        let mut clients = Vec::with_capacity(3);
        for _ in 0..3 {
            let rx_clone = addr_rx.recv().expect("recv local_addr");
            clients.push(thread::spawn(move || -> Result<[u8; UNIQUE_ID_BYTES]> {
                let mut client = RendezvousClient::connect(rx_clone)?;
                client.rendezvous()
            }));
        }

        let server_handle_res =
            run_with_timeout("world_size_4_server", Duration::from_secs(5), move || {
                server.join().expect("server panic")
            });
        server_handle_res.expect("server rendezvous");

        for (idx, c) in clients.into_iter().enumerate() {
            let received = c
                .join()
                .expect("client panic")
                .unwrap_or_else(|e| panic!("client {} rendezvous: {e:?}", idx + 1));
            assert_eq!(received, unique_id, "client {} mismatched id", idx + 1);
        }
    }

    #[test]
    fn client_retries_when_server_late() {
        // Pre-bind a listener to reserve a port, then drop it just before the
        // server thread starts. This avoids guessing a free port while still
        // letting the client attempt connect before the server is up.
        let probe = TcpListener::bind("127.0.0.1:0").expect("probe bind");
        let addr = probe.local_addr().expect("probe addr");
        drop(probe);

        let unique_id = [0x5Au8; UNIQUE_ID_BYTES];

        let client = thread::spawn(move || -> Result<[u8; UNIQUE_ID_BYTES]> {
            let mut client = RendezvousClient::connect(addr)?;
            client.rendezvous()
        });

        // Sleep < retry interval so the client is mid-retry when we bind.
        thread::sleep(Duration::from_millis(500));

        let server = thread::spawn(move || -> Result<()> {
            let mut server = RendezvousServer::bind(addr, 2)?;
            server.rendezvous(&unique_id)
        });

        let server_res = run_with_timeout(
            "client_retries_server",
            Duration::from_secs(15),
            move || server.join().expect("server panic"),
        );
        server_res.expect("server rendezvous");

        let received = client
            .join()
            .expect("client panic")
            .expect("client rendezvous");
        assert_eq!(received, unique_id);
    }

    #[test]
    fn server_errors_on_bad_barrier_byte() {
        let (addr_tx, addr_rx) = sync_channel::<SocketAddr>(1);
        let unique_id = [0x99u8; UNIQUE_ID_BYTES];

        let server = thread::spawn(move || -> Result<()> {
            let mut server = RendezvousServer::bind("127.0.0.1:0", 2)?;
            addr_tx.send(server.local_addr()?).expect("send local_addr");
            server.rendezvous(&unique_id)
        });

        let bad_client = thread::spawn(move || -> Result<()> {
            let addr = addr_rx.recv().expect("recv local_addr");
            let mut stream =
                TcpStream::connect_timeout(&addr, SOCKET_TIMEOUT).context("bad client connect")?;
            stream.set_read_timeout(Some(SOCKET_TIMEOUT))?;
            stream.set_write_timeout(Some(SOCKET_TIMEOUT))?;
            let mut buf = [0u8; UNIQUE_ID_BYTES];
            stream.read_exact(&mut buf).context("bad client read")?;
            stream.write_all(&[0x00]).context("bad client write")?;
            // Hold the stream open until the server read completes.
            thread::sleep(Duration::from_millis(100));
            drop(stream);
            Ok(())
        });

        let server_res =
            run_with_timeout("bad_barrier_server", Duration::from_secs(5), move || {
                server.join().expect("server panic")
            });
        let err = server_res.expect_err("server should reject bad barrier");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("barrier"),
            "error message should mention 'barrier', got: {msg}"
        );

        bad_client
            .join()
            .expect("bad client panic")
            .expect("bad client ok");
    }

    #[test]
    fn server_accept_times_out_when_peer_never_connects() {
        let mut server = RendezvousServer::bind("127.0.0.1:0", 4).expect("bind ephemeral listener");
        let unique_id = [0x01u8; UNIQUE_ID_BYTES];
        let started = Instant::now();
        let result = run_with_timeout(
            "server_accept_deadline",
            Duration::from_secs(2),
            move || server.rendezvous_with_timeout(&unique_id, Duration::from_millis(200)),
        );
        let elapsed = started.elapsed();
        let err = result.expect_err("rendezvous must fail when peers never connect");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("timed out"),
            "error should mention 'timed out', got: {msg}"
        );
        assert!(
            elapsed < Duration::from_secs(1),
            "deadline should fire well under 1s, took {elapsed:?}"
        );
    }
}
