//! Multiproc-serve control-plane relay (Stage 1 re-wire scaffold).
//!
//! TCP transport for shipping serve-request payloads from the coordinator
//! process (rank 0) to worker processes (rank 1..N-1). NCCL handles the
//! data-plane forward sync (Stage 2); this module handles the per-request
//! control-plane fanout.
//!
//! Ported from `git show e81b98fb^:infer/src/multiproc_relay.rs` and adapted to
//! the rewrite crate stack:
//!   - `WireSamplingParams` -> the rewrite [`infer_plan::SamplingParams`] (the
//!     legacy field-by-field mirror was identical), with [`WireRequest::submit_args`]
//!     reconstructing the `(prompt, max_tokens, sampling)` triple the rewrite
//!     [`crate::ServeHandle::submit`] takes.
//!   - The legacy `WireRequest` carried `priority`/`session_id`/`stop`; the
//!     rewrite `submit` does not surface those yet, so they are dropped here
//!     rather than carried inert. Re-add when the rewrite scheduler grows the
//!     fields. (`// STAGE 2:`)
//!   - The legacy `Completion` envelope coupled to `infer::server_engine::
//!     CompletionStreamDelta`. That type now lives in `infer-api` (which depends
//!     on this crate), so it can't be referenced here. A self-contained
//!     [`RelayCompletionDelta`] (serde, length-prefixed JSON) carries the
//!     Stage-1 worker->coordinator output; Stage 2 maps it to the public
//!     `CompletionStreamDelta` at the `infer-api` boundary. (`// STAGE 2:`)
//!
//! Wire format: length-prefixed JSON envelopes.
//!   [u32 LE: payload_len][payload_len bytes JSON].
//!
//! Lifetime: coordinator binds at boot, waits for N-1 worker connects, then
//! becomes write-only. Workers connect at boot, then read in a loop. On
//! coordinator drop, all worker streams EOF and workers exit.

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use infer_plan::SamplingParams;
use serde::{Deserialize, Serialize};

/// Process-global per-tick admission broadcaster: the rank-0 multiproc
/// coordinator installs a closure here that ships a
/// [`RelayEnvelope::TickAdmissions`] to every worker rank. The rank-0 engine
/// loop ([`crate::execution`]) calls [`broadcast_tick`] exactly once before
/// every engine step (with the submissions drained for that tick — possibly
/// none), so admission is PART of the lockstep instead of racing it.
///
/// Why per-tick and not per-request: with a free-running async relay, a request
/// lands in rank 0's queue at T and in a worker's at T+δ; whichever rank's
/// tick-top drain straddles T plans it one forward earlier than the others →
/// divergent NCCL collective sequences → deadlock (measured 2026-06-10: tick-9
/// fingerprints, `errors/2026-06-10-dsv4-c2-layer2-lockstep-admission-deadlock.md`).
/// Per-tick broadcast is the SGLang `recv_requests` + `broadcast_pyobj` shape:
/// every rank admits the same requests at the same step index by construction.
/// Soundness preconditions (all hold on the CUDA path): executor `submit` is
/// synchronous and `poll` always `Ready` (one forward per `Engine::step`),
/// sampling is `(seed, position)`-deterministic, and plan-building is a pure
/// function of engine state.
///
/// Single-process serving (`world_size == 1`) never installs a broadcaster, so
/// [`tick_broadcaster_installed`] is false and the default serve path is
/// byte-identical to before.
type TickBroadcaster = Box<dyn Fn(u64, Vec<WireRequest>) -> Result<()> + Send + Sync>;

static TICK_BROADCASTER: OnceLock<TickBroadcaster> = OnceLock::new();

/// Install the process-global per-tick admission broadcaster (rank-0
/// coordinator only). Must happen at boot, before `serve_http` builds the
/// rank-0 engine, so the engine loop observes it from its first tick.
///
/// Returns an error if a broadcaster is already installed.
pub fn set_tick_broadcaster(broadcaster: TickBroadcaster) -> Result<()> {
    TICK_BROADCASTER
        .set(broadcaster)
        .map_err(|_| anyhow::anyhow!("tick broadcaster already installed"))
}

/// Whether a tick broadcaster is installed (multiproc rank-0 coordinator).
#[must_use]
pub fn tick_broadcaster_installed() -> bool {
    TICK_BROADCASTER.get().is_some()
}

/// Invoke the installed tick broadcaster. Called by the rank-0 engine loop
/// exactly once per engine step, BEFORE the drained submissions are admitted
/// locally. `seq` is the loop's monotonic tick counter; workers verify
/// contiguity and treat a gap as a fatal protocol violation.
pub fn broadcast_tick(seq: u64, requests: Vec<WireRequest>) -> Result<()> {
    if let Some(broadcaster) = TICK_BROADCASTER.get() {
        broadcaster(seq, requests)?;
    }
    Ok(())
}

/// Bound on one blocking `send()` syscall on a relay socket. Shorter than the
/// coordinator's `ACK_STALL_TIMEOUT` (120s, tolerates a legitimately slow
/// compute step) — a write that doesn't drain means the peer's TCP receive
/// buffer is full, a more acute symptom than "still computing," so it fails
/// fast and lets the caller's existing broadcast-error teardown path run
/// instead of blocking the lockstep thread (and the mutex it holds) forever.
const BROADCAST_WRITE_TIMEOUT: Duration = Duration::from_secs(30);

/// Transport seam for length-prefixed JSON [`RelayEnvelope`]s between coordinator
/// and worker ranks. Only impl today is [`TcpChannel`]; the trait exists so a
/// shm ring can plug in later without touching the loops.
pub trait RelayChannel: Send {
    /// Write one envelope to the peer.
    fn send(&mut self, envelope: &RelayEnvelope) -> Result<()>;
    /// Read one envelope; `Ok(None)` on clean peer EOF.
    fn recv(&mut self) -> Result<Option<RelayEnvelope>>;
    /// Clone an independent handle over the same underlying connection.
    fn try_clone_channel(&self) -> Result<Box<dyn RelayChannel>>;
    /// Set a read timeout (used by the completion reader to poll for shutdown);
    /// `None` blocks indefinitely. Default is a no-op for transports without one.
    fn set_read_timeout(&mut self, _timeout: Option<Duration>) -> Result<()> {
        Ok(())
    }
    /// Set a write timeout. `send()` otherwise blocks indefinitely on a full
    /// socket buffer (the peer not draining) — the 2026-07-05 livelock had the
    /// coordinator's lockstep thread wedged inside exactly this write, holding
    /// the `RelayCoordinator` mutex, one call before `wait_for_ack_window` (whose
    /// own timeout never got a chance to run). Default is a no-op for
    /// transports without one.
    fn set_write_timeout(&mut self, _timeout: Option<Duration>) -> Result<()> {
        Ok(())
    }
}

/// Localhost-TCP [`RelayChannel`] — the only transport; a thin newtype over
/// [`TcpStream`].
pub struct TcpChannel(TcpStream);

impl TcpChannel {
    #[must_use]
    pub fn new(stream: TcpStream) -> Self {
        Self(stream)
    }
}

impl RelayChannel for TcpChannel {
    fn send(&mut self, envelope: &RelayEnvelope) -> Result<()> {
        write_envelope(&mut self.0, envelope)
    }

    fn recv(&mut self) -> Result<Option<RelayEnvelope>> {
        read_envelope(&mut self.0)
    }

    fn try_clone_channel(&self) -> Result<Box<dyn RelayChannel>> {
        let stream = self.0.try_clone().context("TcpChannel clone stream")?;
        Ok(Box::new(TcpChannel(stream)))
    }

    fn set_read_timeout(&mut self, timeout: Option<Duration>) -> Result<()> {
        self.0
            .set_read_timeout(timeout)
            .context("TcpChannel set_read_timeout")
    }

    fn set_write_timeout(&mut self, timeout: Option<Duration>) -> Result<()> {
        self.0
            .set_write_timeout(timeout)
            .context("TcpChannel set_write_timeout")
    }
}

/// Free-port picker. Binds 127.0.0.1:0, reads the assigned port, drops the
/// listener. Caller races to use the port (negligible window on single host).
pub fn pick_free_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").context("relay port reservation")?;
    let port = listener.local_addr().context("relay port read")?.port();
    drop(listener);
    Ok(port)
}

/// In-process LocalChannel: coordinator→engine direction (send only).
/// The SyncSender is Clone so the relay driver can hand copies to spawned threads.
pub struct LocalChannelSend(pub(crate) std::sync::mpsc::SyncSender<RelayEnvelope>);

/// In-process LocalChannel: engine←coordinator OR coordinator←engine (recv only).
pub struct LocalChannelRecv(std::sync::mpsc::Receiver<RelayEnvelope>);

impl RelayChannel for LocalChannelSend {
    fn send(&mut self, e: &RelayEnvelope) -> Result<()> {
        self.0
            .send(e.clone())
            .map_err(|_| anyhow::anyhow!("local channel: peer disconnected"))
    }

    fn recv(&mut self) -> Result<Option<RelayEnvelope>> {
        Ok(None) // send-only side; never called
    }

    fn try_clone_channel(&self) -> Result<Box<dyn RelayChannel>> {
        anyhow::bail!("LocalChannelSend is send-only; use clone_tx() instead")
    }
}

impl LocalChannelSend {
    /// Clone the underlying sender for use in spawned threads.
    #[allow(dead_code)]
    pub(crate) fn clone_tx(&self) -> std::sync::mpsc::SyncSender<RelayEnvelope> {
        self.0.clone()
    }
}

impl RelayChannel for LocalChannelRecv {
    fn send(&mut self, _: &RelayEnvelope) -> Result<()> {
        anyhow::bail!("LocalChannelRecv is recv-only")
    }

    fn recv(&mut self) -> Result<Option<RelayEnvelope>> {
        match self.0.recv() {
            Ok(e) => Ok(Some(e)),
            Err(_) => Ok(None), // disconnected = EOF
        }
    }

    fn try_clone_channel(&self) -> Result<Box<dyn RelayChannel>> {
        anyhow::bail!("LocalChannelRecv is not cloneable")
    }
}

/// Create a bidirectional in-process relay channel pair.
///
/// Returns:
/// - `coord_send`: coordinator → engine (send TickAdmissions/StatsQuery)
/// - `coord_recv`: coordinator ← engine (receive Completion/StatsResponse)
/// - `engine_recv`: engine ← coordinator (receive TickAdmissions/StatsQuery)
/// - `engine_tx`: engine → coordinator (send Completion/StatsResponse; Clone-able)
pub fn local_relay_pair() -> (
    LocalChannelSend,
    LocalChannelRecv,
    LocalChannelRecv,
    std::sync::mpsc::SyncSender<RelayEnvelope>,
) {
    let (coord_to_engine_tx, engine_from_coord_rx) = std::sync::mpsc::sync_channel(64);
    let (engine_to_coord_tx, coord_from_engine_rx) = std::sync::mpsc::sync_channel(64);
    (
        LocalChannelSend(coord_to_engine_tx),
        LocalChannelRecv(coord_from_engine_rx),
        LocalChannelRecv(engine_from_coord_rx),
        engine_to_coord_tx,
    )
}

/// Stats payload for the coordinator `/v1/stats` relay round-trip.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WireStats {
    #[serde(default)]
    pub build_identity: crate::BuildIdentity,
    pub active_requests: usize,
    pub queue_depth: usize,
    pub kv_free_pages: usize,
    pub throughput_steps: u64,
    pub throughput_prefill_tokens: u64,
    pub throughput_generated_tokens: u64,
    pub throughput_requests_completed: u64,
    #[serde(default)]
    pub throughput_forward_busy_micros: u64,
    #[serde(default)]
    pub throughput_prefill_forward_steps: u64,
    #[serde(default)]
    pub throughput_prefill_forward_busy_micros: u64,
    #[serde(default)]
    pub throughput_decode_forward_steps: u64,
    #[serde(default)]
    pub throughput_decode_forward_busy_micros: u64,
    #[serde(default)]
    pub throughput_mixed_forward_steps: u64,
    #[serde(default)]
    pub throughput_mixed_forward_busy_micros: u64,
    #[serde(default)]
    pub throughput_decode_phase_steps: u64,
    #[serde(default)]
    pub throughput_decode_phase_poll_micros: u64,
    #[serde(default)]
    pub throughput_decode_phase_apply_output_micros: u64,
    #[serde(default)]
    pub throughput_decode_phase_poll_background_micros: u64,
    #[serde(default)]
    pub throughput_decode_phase_admit_micros: u64,
    #[serde(default)]
    pub throughput_decode_phase_plan_micros: u64,
    #[serde(default)]
    pub throughput_decode_phase_submit_micros: u64,
    pub prefix_lookups: u64,
    pub prefix_hits: u64,
    pub prefix_hit_tokens: u64,
    pub prefix_hit_pages: u64,
    pub prefix_published_pages: u64,
    pub prefix_cached_pages: usize,
    pub kv_tier_demoted_pages: u64,
    pub kv_tier_promoted_pages: u64,
    pub kv_tier_promote_failures: u64,
    pub kv_tier_resident_blocks: usize,
    pub kv_tier_demoted_slots: u64,
    pub kv_tier_promoted_slots: u64,
    pub kv_tier_slot_promote_failures: u64,
    pub kv_system_resident_pages: usize,
    pub kv_system_resident_evictable_pages: usize,
    pub kv_system_host_demoted_pages: usize,
    pub kv_system_host_demoted_pending_inflight: usize,
    pub kv_system_disk_pages: usize,
    pub kv_system_reuse_hit_resident: u64,
    pub kv_system_reuse_hit_host_demoted: u64,
    pub kv_system_reuse_hit_disk: u64,
    pub kv_system_reuse_miss: u64,
    pub kv_system_demote_mset_count: u64,
    pub kv_system_demote_mset_copy_bytes: u64,
    pub kv_system_demote_mset_copy_ms: u64,
    pub kv_system_promote_mget_count: u64,
    pub kv_system_promote_mget_copy_bytes: u64,
    pub kv_system_promote_mget_copy_ms: u64,
    pub kv_system_fetch_wait_ms: u64,
    pub kv_system_fallback_recompute: u64,
    pub kv_system_prefix_match_full_blocks: u64,
    pub kv_system_prefix_match_clamped_blocks: u64,
    #[serde(default)]
    pub kv_system_tier_io_mode: infer_seam::KvTierIoMode,
    #[serde(default)]
    pub kv_system_tier_io_useful_read_bytes: u64,
    #[serde(default)]
    pub kv_system_tier_io_useful_write_bytes: u64,
    #[serde(default)]
    pub kv_system_tier_io_submitted_read_bytes: u64,
    #[serde(default)]
    pub kv_system_tier_io_submitted_write_bytes: u64,
    #[serde(default)]
    pub kv_system_tier_io_metadata_write_bytes: u64,
    #[serde(default)]
    pub kv_system_tier_io_failures: u64,
    #[serde(default)]
    pub kv_system_tier_io_completion_wait_ns: u64,
    #[serde(default)]
    pub spec_chains: u64,
    #[serde(default)]
    pub spec_drafted: u64,
    #[serde(default)]
    pub spec_accepted: u64,
    #[serde(default)]
    pub spec_rejected: u64,
    #[serde(default)]
    pub spec_partial_ctx_chains: u64,
    #[serde(default)]
    pub operator_dispatch: infer_seam::OperatorDispatchStats,
}

impl WireStats {
    /// Reconstruct the per-tick counters; request-boundary operator stats remain
    /// in the wire payload for `/v1/stats` only.
    pub fn into_counter_snapshot(self) -> crate::CounterSnapshot {
        crate::CounterSnapshot {
            active_requests: self.active_requests,
            queue_depth: self.queue_depth,
            kv_free_pages: self.kv_free_pages,
            throughput: infer_core::ThroughputStats {
                steps: self.throughput_steps,
                prefill_tokens: self.throughput_prefill_tokens,
                generated_tokens: self.throughput_generated_tokens,
                requests_completed: self.throughput_requests_completed,
                forward_busy_micros: self.throughput_forward_busy_micros,
                prefill_forward_steps: self.throughput_prefill_forward_steps,
                prefill_forward_busy_micros: self.throughput_prefill_forward_busy_micros,
                decode_forward_steps: self.throughput_decode_forward_steps,
                decode_forward_busy_micros: self.throughput_decode_forward_busy_micros,
                mixed_forward_steps: self.throughput_mixed_forward_steps,
                mixed_forward_busy_micros: self.throughput_mixed_forward_busy_micros,
                decode_step_phase: infer_core::StepPhaseStats {
                    steps: self.throughput_decode_phase_steps,
                    poll_micros: self.throughput_decode_phase_poll_micros,
                    apply_output_micros: self.throughput_decode_phase_apply_output_micros,
                    poll_background_micros: self.throughput_decode_phase_poll_background_micros,
                    admit_micros: self.throughput_decode_phase_admit_micros,
                    plan_micros: self.throughput_decode_phase_plan_micros,
                    submit_micros: self.throughput_decode_phase_submit_micros,
                },
            },
            prefix_cache: infer_core::PrefixCacheStats {
                lookups: self.prefix_lookups,
                hits: self.prefix_hits,
                hit_tokens: self.prefix_hit_tokens,
                hit_pages: self.prefix_hit_pages,
                published_pages: self.prefix_published_pages,
                cached_pages: self.prefix_cached_pages,
            },
            kv_tier: infer_core::KvTierStats {
                demoted_pages: self.kv_tier_demoted_pages,
                promoted_pages: self.kv_tier_promoted_pages,
                promote_failures: self.kv_tier_promote_failures,
                resident_blocks: self.kv_tier_resident_blocks,
                demoted_slots: self.kv_tier_demoted_slots,
                promoted_slots: self.kv_tier_promoted_slots,
                slot_promote_failures: self.kv_tier_slot_promote_failures,
            },
            kv_system: infer_core::KvSystemMetrics {
                resident_pages: self.kv_system_resident_pages,
                resident_evictable_pages: self.kv_system_resident_evictable_pages,
                host_demoted_pages: self.kv_system_host_demoted_pages,
                host_demoted_pending_inflight: self.kv_system_host_demoted_pending_inflight,
                disk_pages: self.kv_system_disk_pages,
                reuse_hit_resident: self.kv_system_reuse_hit_resident,
                reuse_hit_host_demoted: self.kv_system_reuse_hit_host_demoted,
                reuse_hit_disk: self.kv_system_reuse_hit_disk,
                reuse_miss: self.kv_system_reuse_miss,
                demote_mset_count: self.kv_system_demote_mset_count,
                demote_mset_copy_bytes: self.kv_system_demote_mset_copy_bytes,
                demote_mset_copy_ms: self.kv_system_demote_mset_copy_ms,
                promote_mget_count: self.kv_system_promote_mget_count,
                promote_mget_copy_bytes: self.kv_system_promote_mget_copy_bytes,
                promote_mget_copy_ms: self.kv_system_promote_mget_copy_ms,
                fetch_wait_ms: self.kv_system_fetch_wait_ms,
                fallback_recompute: self.kv_system_fallback_recompute,
                prefix_match_full_blocks: self.kv_system_prefix_match_full_blocks,
                prefix_match_clamped_blocks: self.kv_system_prefix_match_clamped_blocks,
                tier_io_mode: self.kv_system_tier_io_mode,
                tier_io_useful_read_bytes: self.kv_system_tier_io_useful_read_bytes,
                tier_io_useful_write_bytes: self.kv_system_tier_io_useful_write_bytes,
                tier_io_submitted_read_bytes: self.kv_system_tier_io_submitted_read_bytes,
                tier_io_submitted_write_bytes: self.kv_system_tier_io_submitted_write_bytes,
                tier_io_metadata_write_bytes: self.kv_system_tier_io_metadata_write_bytes,
                tier_io_failures: self.kv_system_tier_io_failures,
                tier_io_completion_wait_ns: self.kv_system_tier_io_completion_wait_ns,
            },
            spec_decode: infer_seam::SpecDecodeStats {
                chains: self.spec_chains,
                drafted: self.spec_drafted,
                accepted: self.spec_accepted,
                rejected: self.spec_rejected,
                partial_ctx_chains: self.spec_partial_ctx_chains,
            },
        }
    }

    /// Own the per-tick counters plus request-boundary operator stats for relay.
    pub fn from_counters(
        c: &crate::CounterSnapshot,
        build_identity: crate::BuildIdentity,
        operator_dispatch: infer_seam::OperatorDispatchStats,
    ) -> Self {
        Self {
            build_identity,
            active_requests: c.active_requests,
            queue_depth: c.queue_depth,
            kv_free_pages: c.kv_free_pages,
            throughput_steps: c.throughput.steps,
            throughput_prefill_tokens: c.throughput.prefill_tokens,
            throughput_generated_tokens: c.throughput.generated_tokens,
            throughput_requests_completed: c.throughput.requests_completed,
            throughput_forward_busy_micros: c.throughput.forward_busy_micros,
            throughput_prefill_forward_steps: c.throughput.prefill_forward_steps,
            throughput_prefill_forward_busy_micros: c.throughput.prefill_forward_busy_micros,
            throughput_decode_forward_steps: c.throughput.decode_forward_steps,
            throughput_decode_forward_busy_micros: c.throughput.decode_forward_busy_micros,
            throughput_mixed_forward_steps: c.throughput.mixed_forward_steps,
            throughput_mixed_forward_busy_micros: c.throughput.mixed_forward_busy_micros,
            throughput_decode_phase_steps: c.throughput.decode_step_phase.steps,
            throughput_decode_phase_poll_micros: c.throughput.decode_step_phase.poll_micros,
            throughput_decode_phase_apply_output_micros: c
                .throughput
                .decode_step_phase
                .apply_output_micros,
            throughput_decode_phase_poll_background_micros: c
                .throughput
                .decode_step_phase
                .poll_background_micros,
            throughput_decode_phase_admit_micros: c.throughput.decode_step_phase.admit_micros,
            throughput_decode_phase_plan_micros: c.throughput.decode_step_phase.plan_micros,
            throughput_decode_phase_submit_micros: c.throughput.decode_step_phase.submit_micros,
            prefix_lookups: c.prefix_cache.lookups,
            prefix_hits: c.prefix_cache.hits,
            prefix_hit_tokens: c.prefix_cache.hit_tokens,
            prefix_hit_pages: c.prefix_cache.hit_pages,
            prefix_published_pages: c.prefix_cache.published_pages,
            prefix_cached_pages: c.prefix_cache.cached_pages,
            kv_tier_demoted_pages: c.kv_tier.demoted_pages,
            kv_tier_promoted_pages: c.kv_tier.promoted_pages,
            kv_tier_promote_failures: c.kv_tier.promote_failures,
            kv_tier_resident_blocks: c.kv_tier.resident_blocks,
            kv_tier_demoted_slots: c.kv_tier.demoted_slots,
            kv_tier_promoted_slots: c.kv_tier.promoted_slots,
            kv_tier_slot_promote_failures: c.kv_tier.slot_promote_failures,
            kv_system_resident_pages: c.kv_system.resident_pages,
            kv_system_resident_evictable_pages: c.kv_system.resident_evictable_pages,
            kv_system_host_demoted_pages: c.kv_system.host_demoted_pages,
            kv_system_host_demoted_pending_inflight: c.kv_system.host_demoted_pending_inflight,
            kv_system_disk_pages: c.kv_system.disk_pages,
            kv_system_reuse_hit_resident: c.kv_system.reuse_hit_resident,
            kv_system_reuse_hit_host_demoted: c.kv_system.reuse_hit_host_demoted,
            kv_system_reuse_hit_disk: c.kv_system.reuse_hit_disk,
            kv_system_reuse_miss: c.kv_system.reuse_miss,
            kv_system_demote_mset_count: c.kv_system.demote_mset_count,
            kv_system_demote_mset_copy_bytes: c.kv_system.demote_mset_copy_bytes,
            kv_system_demote_mset_copy_ms: c.kv_system.demote_mset_copy_ms,
            kv_system_promote_mget_count: c.kv_system.promote_mget_count,
            kv_system_promote_mget_copy_bytes: c.kv_system.promote_mget_copy_bytes,
            kv_system_promote_mget_copy_ms: c.kv_system.promote_mget_copy_ms,
            kv_system_fetch_wait_ms: c.kv_system.fetch_wait_ms,
            kv_system_fallback_recompute: c.kv_system.fallback_recompute,
            kv_system_prefix_match_full_blocks: c.kv_system.prefix_match_full_blocks,
            kv_system_prefix_match_clamped_blocks: c.kv_system.prefix_match_clamped_blocks,
            kv_system_tier_io_mode: c.kv_system.tier_io_mode,
            kv_system_tier_io_useful_read_bytes: c.kv_system.tier_io_useful_read_bytes,
            kv_system_tier_io_useful_write_bytes: c.kv_system.tier_io_useful_write_bytes,
            kv_system_tier_io_submitted_read_bytes: c.kv_system.tier_io_submitted_read_bytes,
            kv_system_tier_io_submitted_write_bytes: c.kv_system.tier_io_submitted_write_bytes,
            kv_system_tier_io_metadata_write_bytes: c.kv_system.tier_io_metadata_write_bytes,
            kv_system_tier_io_failures: c.kv_system.tier_io_failures,
            kv_system_tier_io_completion_wait_ns: c.kv_system.tier_io_completion_wait_ns,
            spec_chains: c.spec_decode.chains,
            spec_drafted: c.spec_decode.drafted,
            spec_accepted: c.spec_decode.accepted,
            spec_rejected: c.spec_decode.rejected,
            spec_partial_ctx_chains: c.spec_decode.partial_ctx_chains,
            operator_dispatch,
        }
    }
}

/// Self-contained worker->coordinator completion delta (Stage 1).
///
/// Mirrors the shape of the public `infer_api::CompletionStreamDelta` minus the
/// crate coupling, so the relay can ship worker output without depending on
/// `infer-api`. `finish` marks the terminal delta. `error` carries a terminal
/// failure. `// STAGE 2:` map this to/from `infer_api::CompletionStreamDelta`
/// at the boundary once the lockstep forward emits real tokens.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RelayCompletionDelta {
    /// Newly decoded text for this delta.
    pub text_delta: String,
    /// Token ids newly emitted in this delta.
    pub token_ids: Vec<u32>,
    /// Generation-time behavior logprobs, one per `token_ids` entry — or empty
    /// when the executor did not capture them (greedy, non-CUDA arms). Serde
    /// default so pre-P6 workers interoperate.
    #[serde(default)]
    pub logprobs: Vec<f32>,
    /// Whether this is the terminal delta for the request.
    pub finish: bool,
    /// Finish reason on the terminal delta, so the coordinator reports the real
    /// OpenAI `finish_reason` rather than always "stop".
    pub finish_reason: Option<infer_plan::FinishReason>,
    /// Terminal failure message, if the request failed before a normal finish.
    pub error: Option<String>,
}

impl RelayCompletionDelta {
    /// A text-only, non-terminal delta.
    #[must_use]
    pub fn text(s: String) -> Self {
        Self {
            text_delta: s,
            ..Self::default()
        }
    }

    /// Whether this delta closes the request (terminal finish or error).
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.finish || self.error.is_some()
    }
}

/// Wire envelope for relay messages. Tagged enum so future variants
/// (Heartbeat, Stats, Abort) compose cleanly without protocol breaks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RelayEnvelope {
    /// Worker identity handshake. The coordinator consumes this during
    /// [`PendingRelayCoordinator::accept`] and indexes streams by rank. It is
    /// never forwarded to the scheduler loop.
    WorkerHello { rank: usize, world_size: usize },
    /// Lightweight boot-ping / liveness envelope used at coordinator boot to
    /// validate every worker opened its relay end.
    BootPing { request_id: u64 },
    /// One engine tick's admissions, sent by rank 0 exactly once before every
    /// engine step (empty `requests` for pure-decode ticks). Workers inject the
    /// requests into their local engine and run exactly one step per envelope,
    /// keeping admission-at-step-index identical on every rank. `seq` is
    /// contiguous from 0; a gap on the worker side is a fatal protocol bug.
    TickAdmissions {
        seq: u64,
        requests: Vec<WireRequest>,
    },
    /// A request's HTTP client disconnected/timed out — cancel it in every
    /// rank's engine (`InFlightGuard::drop`, coordinator.rs). Broadcast
    /// BEFORE the tick's `TickAdmissions` so every rank applies it at the
    /// identical point relative to that tick's step, same discipline as
    /// admissions (2026-07-05 multiproc hang investigation —
    /// docs/experience/errors/2026-07-05-multiproc-lockstep-ack-hang-no-timeout.md).
    /// A no-op on a rank where the request already finished naturally.
    CancelRequest { request_id: u64 },
    /// Worker→coordinator flow control: rank consumed the tick with this seq
    /// (sent whether or not the engine stepped — an idle skip still consumes).
    /// The coordinator caps unacked ticks at a fixed window so the tick stream
    /// paces to engine speed instead of flooding the FIFO (admission latency
    /// was queue-depth × step time; now ≤ window × step time).
    TickAck { rank: usize, seq: u64 },
    /// Worker readiness ack, sent once a rank's engine (incl. NCCL rendezvous) is
    /// built. The coordinator blocks HTTP bind until all `world_size` readies
    /// arrive, so no request is broadcast before the collective has formed.
    EngineReady { rank: usize },
    /// Worker-to-coordinator output for a remote visible-output owner rank.
    /// `// STAGE 2:` the visible-output owner routing is not wired yet; today
    /// every worker is a replicated-token follower with no visible output.
    Completion {
        request_id: u64,
        delta: RelayCompletionDelta,
    },
    /// Graceful shutdown notice; workers should drain in-flight then exit.
    /// (Coordinator can also just drop streams; this is the nicer path for
    /// telemetry/log capture.)
    Shutdown,
    /// Coordinator-to-rank-0 stats probe. Bypasses lockstep tick sequencing;
    /// the worker replies immediately with its current engine counters.
    StatsQuery { request_id: u64 },
    /// Rank-0-to-coordinator response to a `StatsQuery`. `data` is boxed so
    /// the enum's other variants stay small (WireStats is ~320 B).
    StatsResponse {
        request_id: u64,
        data: Box<WireStats>,
    },
}

/// Serializable counterpart of the rewrite serve request. Captures the minimum
/// data a worker rank needs to schedule the same forward path as rank 0: the
/// prompt tokens, the new-token budget, and the sampling params.
///
/// `// STAGE 2:` the legacy `WireRequest` also carried `prompt: String`,
/// `stop`, `priority`, and `session_id`. The rewrite `ServeHandle::submit` takes
/// only `(prompt_tokens, max_tokens, SamplingParams)`, so those are dropped
/// until the rewrite scheduler surfaces them.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WireRequest {
    /// Coordinator-assigned logical request id (for completion routing).
    pub request_id: u64,
    /// Prompt token ids (rank 0 tokenizes; workers receive the ids).
    pub prompt_tokens: Vec<u32>,
    /// Newly generated token budget.
    pub max_tokens: usize,
    /// Sampling parameters (rewrite [`SamplingParams`]).
    pub sampling: SamplingParams,
    /// Structured-output constraint, resolved into a matcher engine-side.
    #[serde(default)]
    pub response_format: Option<crate::grammar::ResponseFormat>,
}

impl WireRequest {
    /// Reconstruct the `(prompt, max_tokens, sampling)` triple the rewrite
    /// [`crate::ServeHandle::submit`] consumes.
    #[must_use]
    pub fn submit_args(
        self,
    ) -> (
        Vec<u32>,
        usize,
        SamplingParams,
        Option<crate::grammar::ResponseFormat>,
    ) {
        (
            self.prompt_tokens,
            self.max_tokens,
            self.sampling,
            self.response_format,
        )
    }
}

/// Per-rank [`RelayEnvelope::TickAck`] bookkeeping for lockstep flow control.
/// Reader threads record each rank's acks; the coordinator lockstep loop reads
/// [`Self::min_acked`] to cap unacked ticks at a fixed window.
#[derive(Debug)]
pub struct TickAckLedger {
    /// `(rank, acked tick count)` per connected rank; the count is
    /// `last_acked_seq + 1`, so 0 means "never acked".
    per_rank: Vec<(usize, AtomicU64)>,
    /// Ranks whose reader observed EOF/failure. A dead rank never acks again,
    /// so the ack-window wait must stop waiting on it and let the next
    /// broadcast surface the dead socket (-> `fail_all`), or the coordinator
    /// wedges forever on rate-limited warns while HTTP requests hang.
    dead_ranks: AtomicUsize,
}

impl TickAckLedger {
    fn new(ranks: impl IntoIterator<Item = usize>) -> Self {
        Self {
            per_rank: ranks
                .into_iter()
                .map(|rank| (rank, AtomicU64::new(0)))
                .collect(),
            dead_ranks: AtomicUsize::new(0),
        }
    }

    /// A reader thread lost its rank (EOF / hard error).
    pub fn mark_dead(&self) {
        self.dead_ranks.fetch_add(1, Ordering::AcqRel);
    }

    /// Whether any rank's reader has died since boot.
    #[must_use]
    pub fn any_dead(&self) -> bool {
        self.dead_ranks.load(Ordering::Acquire) > 0
    }

    /// Record that `rank` consumed the tick with `seq`. Acks are monotonic per
    /// rank; a stale/duplicate ack is a no-op (`fetch_max`).
    pub fn ack(&self, rank: usize, seq: u64) {
        match self.per_rank.iter().find(|(r, _)| *r == rank) {
            Some((_, count)) => {
                count.fetch_max(seq + 1, Ordering::AcqRel);
            }
            None => log::warn!("[relay-coordinator] tick ack from unknown rank {rank} (seq {seq})"),
        }
    }

    /// Number of ticks fully acked by ALL ranks (min over per-rank acked
    /// counts). A rank that never acked holds this at 0.
    #[must_use]
    pub fn min_acked(&self) -> u64 {
        self.per_rank
            .iter()
            .map(|(_, count)| count.load(Ordering::Acquire))
            .min()
            .unwrap_or(0)
    }
}

/// Coordinator-side TCP relay. Binds a port, accepts N-1 worker connections at
/// boot, then provides `broadcast()` to send envelopes to every worker stream.
pub struct RelayCoordinator {
    port: u16,
    workers: BTreeMap<usize, Box<dyn RelayChannel>>,
    completion_sinks:
        Arc<Mutex<HashMap<u64, tokio::sync::mpsc::UnboundedSender<RelayCompletionDelta>>>>,
    stats_sinks: Arc<Mutex<HashMap<u64, tokio::sync::oneshot::Sender<WireStats>>>>,
    completion_shutdown: Arc<AtomicBool>,
    /// Count of ranks that sent [`RelayEnvelope::EngineReady`]; the coordinator
    /// polls it via [`Self::ready_count`] before opening HTTP.
    ready_count: Arc<AtomicUsize>,
    /// Per-rank tick-ack ledger, fed by the reader threads; the lockstep loop
    /// polls [`Self::min_acked_ticks`] to pace the tick stream to engine speed.
    tick_acks: Arc<TickAckLedger>,
}

/// Pending coordinator state — listener is bound and port is known but workers
/// have not yet connected. Caller publishes the port (env var, file, etc.) so
/// children can connect, then calls `accept(world_size, timeout)` to finalize a
/// [`RelayCoordinator`].
pub struct PendingRelayCoordinator {
    port: u16,
    listener: TcpListener,
}

impl PendingRelayCoordinator {
    #[must_use]
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Legacy accept: wait for ranks 1..N-1 (rank 0 is the in-process engine).
    /// Kept for the relay tests; the SPMD coordinator uses [`Self::accept_symmetric`].
    pub fn accept(self, world_size: usize, accept_timeout: Duration) -> Result<RelayCoordinator> {
        if world_size < 2 {
            bail!("RelayCoordinator needs world_size >= 2 (got {world_size})");
        }
        self.accept_n(world_size, world_size - 1, accept_timeout)
    }

    fn accept_n(
        self,
        world_size: usize,
        expected: usize,
        accept_timeout: Duration,
    ) -> Result<RelayCoordinator> {
        if world_size < 2 {
            bail!("RelayCoordinator needs world_size >= 2 (got {world_size})");
        }
        let mut workers = BTreeMap::new();
        let deadline = Instant::now() + accept_timeout;

        while workers.len() < expected {
            match self.listener.accept() {
                Ok((mut stream, addr)) => {
                    stream
                        .set_nonblocking(false)
                        .context("worker stream set_nonblocking(false)")?;
                    // Bound the hello read by the remaining accept budget: a peer
                    // that completes the handshake but sends nothing must not wedge
                    // this single-threaded accept loop. Timeout/short read → drop
                    // the connection and keep accepting.
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        bail!(
                            "RelayCoordinator timed out after {accept_timeout:?} waiting for \
                             worker connects ({}/{expected} so far)",
                            workers.len()
                        );
                    }
                    stream
                        .set_read_timeout(Some(remaining))
                        .context("worker stream set_read_timeout")?;
                    let hello = match read_envelope(&mut stream) {
                        Ok(hello) => hello,
                        Err(err) => {
                            log::warn!(
                                "[relay-coordinator] read hello from {addr} failed ({err:#}); dropping connection"
                            );
                            continue;
                        }
                    };
                    let Some(RelayEnvelope::WorkerHello {
                        rank,
                        world_size: worker_world_size,
                    }) = hello
                    else {
                        log::warn!(
                            "[relay-coordinator] {addr} did not send a worker hello; dropping connection"
                        );
                        continue;
                    };
                    if worker_world_size != world_size {
                        bail!(
                            "RelayCoordinator worker rank {rank} reported world_size={worker_world_size}, expected {world_size}"
                        );
                    }
                    if rank >= world_size {
                        bail!("RelayCoordinator worker rank {rank} out of range [0, {world_size})");
                    }
                    // Clear the hello-read timeout before the stream enters the
                    // steady-state relay: completion reads must block indefinitely
                    // (tokens can be far apart), so a leftover timeout would fire
                    // mid-envelope → desynced framing → serve teardown (seen at c8).
                    stream
                        .set_read_timeout(None)
                        .context("worker stream clear read timeout after hello")?;
                    let mut channel: Box<dyn RelayChannel> = Box::new(TcpChannel::new(stream));
                    channel
                        .set_write_timeout(Some(BROADCAST_WRITE_TIMEOUT))
                        .with_context(|| {
                            format!("RelayCoordinator worker rank {rank} set_write_timeout")
                        })?;
                    if workers.insert(rank, channel).is_some() {
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
        let stats_sinks = Arc::new(Mutex::new(HashMap::<
            u64,
            tokio::sync::oneshot::Sender<WireStats>,
        >::new()));
        let completion_shutdown = Arc::new(AtomicBool::new(false));
        let ready_count = Arc::new(AtomicUsize::new(0));
        let tick_acks = Arc::new(TickAckLedger::new(workers.keys().copied()));
        for (&rank, channel) in &workers {
            let reader = channel
                .try_clone_channel()
                .with_context(|| format!("RelayCoordinator clone worker rank {rank} channel"))?;
            spawn_completion_reader(
                rank,
                reader,
                Arc::clone(&completion_sinks),
                Arc::clone(&stats_sinks),
                Arc::clone(&completion_shutdown),
                Arc::clone(&ready_count),
                Arc::clone(&tick_acks),
            );
        }
        Ok(RelayCoordinator {
            port: self.port,
            workers,
            completion_sinks,
            stats_sinks,
            completion_shutdown,
            ready_count,
            tick_acks,
        })
    }

    /// SPMD (B) accept: wait for ALL `world_size` ranks (0..N-1), including rank 0
    /// (a child worker / the visible-output owner). Like [`Self::accept`] but
    /// admits rank 0 into the worker map for tick broadcast + completion routing.
    pub fn accept_symmetric(
        self,
        world_size: usize,
        accept_timeout: Duration,
    ) -> Result<RelayCoordinator> {
        self.accept_n(world_size, world_size, accept_timeout)
    }
}

impl Drop for RelayCoordinator {
    fn drop(&mut self) {
        self.completion_shutdown.store(true, Ordering::Relaxed);
    }
}

impl RelayCoordinator {
    /// Bind to a free port WITHOUT yet accepting any worker connections.
    /// Returns a [`PendingRelayCoordinator`] whose port can be published to
    /// children (env var, file). Call `pending.accept(world_size, timeout)`
    /// after spawning workers to finalize.
    ///
    /// Two-phase API: the coordinator must `bind` before spawning workers (so
    /// child env carries the port) but `accept` must happen after spawn
    /// (children need to connect first).
    pub fn bind() -> Result<PendingRelayCoordinator> {
        let port = pick_free_port()?;
        let listener = TcpListener::bind(("127.0.0.1", port))
            .with_context(|| format!("RelayCoordinator bind port {port}"))?;
        listener
            .set_nonblocking(true)
            .context("RelayCoordinator set_nonblocking on listener")?;
        Ok(PendingRelayCoordinator { port, listener })
    }

    /// Convenience: bind + accept in one call. Only useful when the children are
    /// already running (e.g. in tests where both sides race at startup).
    pub fn bind_and_accept(world_size: usize, accept_timeout: Duration) -> Result<Self> {
        Self::bind()?.accept(world_size, accept_timeout)
    }

    /// Build a `RelayCoordinator` backed by a single in-process local "worker" (rank 0).
    /// Returns the relay + the engine-side channels:
    /// - `engine_recv`: engine reads from this (TickAdmissions / StatsQuery from coordinator)
    /// - `engine_tx`: engine writes to this (Completion / StatsResponse to coordinator)
    pub fn new_local() -> (
        Self,
        LocalChannelRecv,
        std::sync::mpsc::SyncSender<RelayEnvelope>,
    ) {
        let (coord_send, coord_recv, engine_recv, engine_tx) = local_relay_pair();

        let completion_sinks = Arc::new(Mutex::new(HashMap::new()));
        let stats_sinks = Arc::new(Mutex::new(HashMap::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        // Mark rank-0 as ready immediately — no boot handshake for local workers.
        let ready_count = Arc::new(AtomicUsize::new(1));
        let tick_acks = Arc::new(TickAckLedger::new([0]));

        let mut workers = BTreeMap::new();
        workers.insert(0usize, Box::new(coord_send) as Box<dyn RelayChannel>);

        spawn_completion_reader(
            0,
            Box::new(coord_recv),
            Arc::clone(&completion_sinks),
            Arc::clone(&stats_sinks),
            Arc::clone(&shutdown),
            Arc::clone(&ready_count),
            Arc::clone(&tick_acks),
        );

        let relay = Self {
            port: 0,
            workers,
            completion_sinks,
            stats_sinks,
            completion_shutdown: shutdown,
            ready_count,
            tick_acks,
        };

        (relay, engine_recv, engine_tx)
    }

    #[must_use]
    pub fn port(&self) -> u16 {
        self.port
    }

    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    #[must_use]
    pub fn worker_ranks(&self) -> Vec<usize> {
        self.workers.keys().copied().collect()
    }

    /// Ranks that have sent [`RelayEnvelope::EngineReady`] so far; polled to block
    /// HTTP bind until the whole collective has rendezvoused.
    #[must_use]
    pub fn ready_count(&self) -> usize {
        self.ready_count.load(Ordering::Acquire)
    }

    /// Number of ticks fully acked by ALL ranks (min over per-rank
    /// [`RelayEnvelope::TickAck`] counts); ranks that never acked → 0. The
    /// lockstep loop caps `seq` at `min_acked_ticks() + window` so the tick
    /// stream paces to engine speed instead of flooding the worker FIFOs.
    #[must_use]
    pub fn min_acked_ticks(&self) -> u64 {
        self.tick_acks.min_acked()
    }

    /// Whether any rank's completion reader has died (EOF / hard error). The
    /// ack-window wait consults this: a dead rank never acks again, so waiting
    /// on it would wedge the coordinator; instead the wait returns and the
    /// next broadcast surfaces the dead socket (-> `fail_all`).
    #[must_use]
    pub fn any_worker_dead(&self) -> bool {
        self.tick_acks.any_dead()
    }

    /// Clone the shared completion-sink registry so the async HTTP path can
    /// (un)register sinks directly, without contending for the `RelayCoordinator`
    /// mutex the lockstep loop holds across its blocking broadcast.
    #[must_use]
    pub fn completion_sinks(&self) -> CompletionSinks {
        CompletionSinks {
            inner: Arc::clone(&self.completion_sinks),
        }
    }

    pub fn register_completion_sink(
        &mut self,
        request_id: u64,
        sink: tokio::sync::mpsc::UnboundedSender<RelayCompletionDelta>,
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

    /// Broadcast an envelope to every connected worker. On the first write
    /// error, returns immediately — caller decides whether to drop the
    /// coordinator (workers exit) or retry.
    pub fn broadcast(&mut self, envelope: &RelayEnvelope) -> Result<()> {
        for (&rank, channel) in self.workers.iter_mut() {
            channel.send(envelope).with_context(|| {
                format!("RelayCoordinator write envelope to worker rank {rank}")
            })?;
        }
        Ok(())
    }

    /// Send an envelope to a selected global-rank set. A rank absent from the
    /// worker map (e.g. rank 0 in the legacy in-process layout) is skipped.
    pub fn send_to_ranks(&mut self, ranks: &[usize], envelope: &RelayEnvelope) -> Result<()> {
        for &rank in ranks {
            let Some(channel) = self.workers.get_mut(&rank) else {
                // Legacy layout: rank 0 is the in-process engine, not a relay peer.
                if rank == 0 {
                    continue;
                }
                bail!("RelayCoordinator missing worker rank {rank}");
            };
            channel.send(envelope).with_context(|| {
                format!("RelayCoordinator write envelope to worker rank {rank}")
            })?;
        }
        Ok(())
    }

    /// Register a oneshot that will receive the stats response for `request_id`.
    /// Must be called BEFORE `send_stats_query` to avoid a race.
    pub fn register_stats_awaiter(
        &self,
        request_id: u64,
    ) -> tokio::sync::oneshot::Receiver<WireStats> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.stats_sinks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(request_id, tx);
        rx
    }

    /// Drop a stats awaiter registered by [`Self::register_stats_awaiter`] when
    /// the query send fails — otherwise the never-answered entry leaks.
    pub fn unregister_stats_awaiter(&self, request_id: u64) {
        self.stats_sinks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&request_id);
    }

    /// Send a stats query to rank-0 (does NOT broadcast to all ranks).
    pub fn send_stats_query(&mut self, request_id: u64) -> Result<()> {
        let rank0 = self
            .workers
            .get_mut(&0)
            .ok_or_else(|| anyhow::anyhow!("rank-0 worker not connected"))?;
        rank0.send(&RelayEnvelope::StatsQuery { request_id })
    }
}

/// Shared handle to the per-request completion-sink registry, cloned so the async
/// HTTP path can (un)register without taking the `RelayCoordinator` mutex the
/// lockstep loop holds across its blocking broadcast. Reader threads use the same `Arc`.
#[derive(Clone)]
pub struct CompletionSinks {
    inner: Arc<Mutex<HashMap<u64, tokio::sync::mpsc::UnboundedSender<RelayCompletionDelta>>>>,
}

impl CompletionSinks {
    /// Register a per-request sink. Errors on a duplicate id (a coordinator
    /// request-id allocation bug).
    pub fn register(
        &self,
        request_id: u64,
        sink: tokio::sync::mpsc::UnboundedSender<RelayCompletionDelta>,
    ) -> Result<()> {
        let mut sinks = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if sinks.insert(request_id, sink).is_some() {
            bail!("duplicate completion sink for request_id={request_id}");
        }
        Ok(())
    }

    /// Remove a per-request sink (idempotent). Called from the HTTP path's RAII
    /// guard on completion OR cancellation.
    pub fn unregister(&self, request_id: u64) {
        let mut sinks = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        sinks.remove(&request_id);
    }

    /// Fail every registered sink with a terminal error delta and clear the
    /// registry, so awaiting handlers return an error when the worker group dies.
    pub fn fail_all(&self, message: &str) {
        let mut sinks = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (_request_id, sink) in sinks.drain() {
            let _ = sink.send(RelayCompletionDelta {
                finish: true,
                error: Some(message.to_string()),
                ..Default::default()
            });
        }
    }
}

/// Worker-side relay: connects at boot, then `recv()`s one envelope at a time
/// over a [`RelayChannel`] trait object (transport-agnostic, TCP today).
pub struct RelayWorker {
    channel: Box<dyn RelayChannel>,
}

impl RelayWorker {
    /// Connect to the coordinator's relay port. Retries up to `connect_timeout`
    /// since the coordinator may not have called `accept` yet at the moment the
    /// worker fires off.
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
                    let mut channel: Box<dyn RelayChannel> = Box::new(TcpChannel::new(stream));
                    channel
                        .set_write_timeout(Some(BROADCAST_WRITE_TIMEOUT))
                        .with_context(|| {
                            format!("RelayWorker rank {rank}/{world_size} set_write_timeout")
                        })?;
                    return Ok(Self { channel });
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
        self.channel.recv()
    }

    pub fn send(&mut self, envelope: &RelayEnvelope) -> Result<()> {
        self.channel.send(envelope)
    }

    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            channel: self.channel.try_clone_channel()?,
        })
    }
}

fn spawn_completion_reader(
    rank: usize,
    mut channel: Box<dyn RelayChannel>,
    completion_sinks: Arc<
        Mutex<HashMap<u64, tokio::sync::mpsc::UnboundedSender<RelayCompletionDelta>>>,
    >,
    stats_sinks: Arc<Mutex<HashMap<u64, tokio::sync::oneshot::Sender<WireStats>>>>,
    shutdown: Arc<AtomicBool>,
    ready_count: Arc<AtomicUsize>,
    tick_acks: Arc<TickAckLedger>,
) {
    let _ = channel.set_read_timeout(Some(Duration::from_millis(100)));
    let _ = std::thread::Builder::new()
        .name(format!("arle-relay-rank{rank}-completion-reader"))
        .spawn(move || {
            loop {
                match channel.recv() {
                    Ok(Some(RelayEnvelope::StatsResponse { request_id, data })) => {
                        if let Some(tx) = stats_sinks
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .remove(&request_id)
                        {
                            let _ = tx.send(*data);
                        }
                    }
                    Ok(Some(RelayEnvelope::Completion { request_id, delta })) => {
                        let done = delta.is_done();
                        let sink = {
                            let sinks = completion_sinks
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            sinks.get(&request_id).cloned()
                        };
                        let remove = match sink {
                            Some(tx) => tx.send(delta).is_err() || done,
                            None => {
                                log::debug!(
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
                    Ok(Some(RelayEnvelope::TickAck { rank: ack_rank, seq })) => {
                        // The connection identifies the rank; the payload rank is
                        // only cross-checked (a mis-ranked ack must never re-open
                        // the window for a stalled peer).
                        if ack_rank != rank {
                            log::warn!(
                                "[relay-coordinator] tick ack rank mismatch: envelope says {ack_rank}, connection is {rank}"
                            );
                        }
                        tick_acks.ack(rank, seq);
                    }
                    Ok(Some(RelayEnvelope::EngineReady { rank: ready_rank })) => {
                        ready_count.fetch_add(1, Ordering::AcqRel);
                        log::info!(
                            "[relay-coordinator] worker rank {ready_rank} engine ready (via rank {rank} reader)"
                        );
                    }
                    Ok(Some(other)) => {
                        log::warn!(
                            "[relay-coordinator] unexpected worker->coordinator envelope from rank {rank}: {other:?}"
                        );
                    }
                    Ok(None) => {
                        log::info!("[relay-coordinator] worker rank {rank} completion stream EOF");
                        tick_acks.mark_dead();
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
                        tick_acks.mark_dead();
                        break;
                    }
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

    /// The 2026-07-05 livelock's actual freeze point: the coordinator wedged
    /// inside `send()` on a peer that stopped reading. A non-reading peer
    /// fills the socket buffer after enough writes; `set_write_timeout` must
    /// turn that block into a bounded `Err`, not a permanent hang.
    #[test]
    fn write_timeout_bounds_a_send_to_a_non_reading_peer() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        // Accept and hold the connection open, but never read from it —
        // simulates the stuck-peer symptom without needing a real second process.
        let _peer = thread::spawn(move || listener.accept().unwrap().0);
        let mut client = TcpChannel::new(TcpStream::connect(addr).unwrap());
        client
            .set_write_timeout(Some(Duration::from_millis(50)))
            .unwrap();
        let envelope = RelayEnvelope::TickAdmissions {
            seq: 0,
            requests: vec![],
        };
        let started = Instant::now();
        let err = loop {
            if let Err(err) = client.send(&envelope) {
                break err;
            }
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "socket buffer never filled after 5s of sends"
            );
        };
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "send() only errored after {:?}, longer than any reasonable write timeout",
            started.elapsed()
        );
        drop(err); // just needs to be an Err; message is OS/timing-dependent.
    }

    #[test]
    fn envelope_round_trip() {
        let env = RelayEnvelope::TickAdmissions {
            seq: 5,
            requests: vec![WireRequest {
                request_id: 42,
                prompt_tokens: vec![1, 2, 3, 4],
                max_tokens: 16,
                sampling: SamplingParams::default(),
                response_format: None,
            }],
        };
        let bytes = serde_json::to_vec(&env).unwrap();
        let decoded: RelayEnvelope = serde_json::from_slice(&bytes).unwrap();
        match decoded {
            RelayEnvelope::TickAdmissions { seq, requests } => {
                assert_eq!(seq, 5);
                assert_eq!(requests.len(), 1);
                assert_eq!(requests[0].request_id, 42);
                assert_eq!(requests[0].prompt_tokens, vec![1, 2, 3, 4]);
                assert_eq!(requests[0].max_tokens, 16);
            }
            other => panic!("expected TickAdmissions, got {other:?}"),
        }
    }

    #[test]
    fn tick_ack_round_trip() {
        let env = RelayEnvelope::TickAck { rank: 3, seq: 17 };
        let bytes = serde_json::to_vec(&env).unwrap();
        match serde_json::from_slice(&bytes).unwrap() {
            RelayEnvelope::TickAck { rank, seq } => {
                assert_eq!(rank, 3);
                assert_eq!(seq, 17);
            }
            other => panic!("expected TickAck, got {other:?}"),
        }
    }

    #[test]
    fn tick_ack_ledger_min_over_all_ranks() {
        let ledger = TickAckLedger::new([0, 1]);
        assert_eq!(ledger.min_acked(), 0, "no acks yet");
        ledger.ack(0, 0);
        assert_eq!(ledger.min_acked(), 0, "rank 1 never acked");
        ledger.ack(1, 2);
        assert_eq!(ledger.min_acked(), 1, "rank 0 only acked seq 0");
        ledger.ack(0, 2);
        assert_eq!(ledger.min_acked(), 3, "both ranks acked through seq 2");
        ledger.ack(0, 1);
        assert_eq!(ledger.min_acked(), 3, "stale ack is a no-op");
    }

    #[test]
    fn coordinator_tracks_min_acked_ticks_from_worker_acks() {
        let world_size = 2;
        let pending = RelayCoordinator::bind().unwrap();
        let coord_addr: SocketAddr = format!("127.0.0.1:{}", pending.port()).parse().unwrap();

        let worker_thread = thread::spawn(move || {
            let mut worker =
                RelayWorker::connect_with_rank(coord_addr, Duration::from_secs(5), 1, world_size)
                    .unwrap();
            for seq in 0..3 {
                worker
                    .send(&RelayEnvelope::TickAck { rank: 1, seq })
                    .unwrap();
            }
            assert!(worker.recv().unwrap().is_none(), "EOF on coordinator drop");
        });

        let coord = pending
            .accept(world_size, Duration::from_secs(5))
            .expect("worker connected within 5s");
        // No 0-before-ack assert here: the reader thread races the accept
        // return (the pure-ledger test covers the unacked state).
        let deadline = Instant::now() + Duration::from_secs(5);
        while coord.min_acked_ticks() < 3 {
            assert!(Instant::now() < deadline, "acks not observed within 5s");
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(coord.min_acked_ticks(), 3);
        drop(coord);

        worker_thread.join().unwrap();
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
                RelayEnvelope::TickAdmissions { seq, requests } => {
                    assert_eq!(seq, 0);
                    assert_eq!(requests[0].request_id, 7);
                    assert_eq!(requests[0].prompt_tokens, vec![100, 200]);
                }
                _ => panic!("expected TickAdmissions"),
            }
            let next = worker.recv().unwrap();
            assert!(next.is_none(), "expected EOF after coordinator drop");
        });

        let mut coord = pending
            .accept(world_size, Duration::from_secs(5))
            .expect("worker connected within 5s");
        assert_eq!(coord.worker_ranks(), vec![1]);
        coord
            .broadcast(&RelayEnvelope::TickAdmissions {
                seq: 0,
                requests: vec![WireRequest {
                    request_id: 7,
                    prompt_tokens: vec![100, 200],
                    max_tokens: 1,
                    sampling: SamplingParams::default(),
                    response_format: None,
                }],
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
                RelayEnvelope::TickAdmissions { seq, requests } => {
                    assert_eq!(seq, 3);
                    assert_eq!(requests[0].request_id, 9);
                    assert_eq!(requests[0].prompt_tokens, vec![9]);
                }
                other => panic!("expected TickAdmissions, got {other:?}"),
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
                &RelayEnvelope::TickAdmissions {
                    seq: 3,
                    requests: vec![WireRequest {
                        request_id: 9,
                        prompt_tokens: vec![9],
                        max_tokens: 1,
                        sampling: SamplingParams::default(),
                        response_format: None,
                    }],
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
                RelayEnvelope::TickAdmissions { requests, .. } => {
                    assert_eq!(requests[0].request_id, 11);
                }
                other => panic!("expected request signal, got {other:?}"),
            }
            worker
                .send(&RelayEnvelope::Completion {
                    request_id: 11,
                    delta: RelayCompletionDelta::text("remote".to_string()),
                })
                .unwrap();
            worker
                .send(&RelayEnvelope::Completion {
                    request_id: 11,
                    delta: RelayCompletionDelta {
                        finish: true,
                        ..Default::default()
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
                &RelayEnvelope::TickAdmissions {
                    seq: 0,
                    requests: vec![WireRequest {
                        request_id: 11,
                        prompt_tokens: vec![11],
                        max_tokens: 1,
                        sampling: SamplingParams::default(),
                        response_format: None,
                    }],
                },
            )
            .unwrap();
        let first = rx.blocking_recv().expect("text completion");
        assert_eq!(first.text_delta, "remote");
        let second = rx.blocking_recv().expect("finish completion");
        assert!(second.finish);

        worker_thread.join().unwrap();
    }
}
