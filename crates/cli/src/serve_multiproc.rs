//! Multi-rank TP multiproc-serve: SPMD coordinator + symmetric workers (DSv4 +
//! Qwen3.5/3.6 MoE). The engine-less parent ([`bind_relay_and_spawn_workers`])
//! spawns all N workers and runs the coordinator HTTP loop; [`worker_entry`]
//! detects `ARLE_WORKER_RANK` (set on every rank) and runs worker mode + the
//! lockstep driver. Each worker steps SYNCHRONOUSLY, one `inject + step` per
//! relayed tick, so the per-step NCCL collective sequences stay aligned —
//! free-running per-request relays desync (`errors/2026-06-10-dsv4-c2-layer2-
//! lockstep-admission-deadlock.md`). Worker planners must use the SAME config
//! (serialized via `ARLE_WORKER_ENGINE_CONFIG`): a divergent knob = NCCL deadlock.

#![cfg(all(unix, feature = "cuda"))]

use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::process::ExitCode;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use infer_api::{
    CudaWorkerEngine, EngineLoadConfig, RelayCoordinator, RelayEnvelope, RelayWorker, WireStats,
};

/// Relay accept / worker-connect timeout. This covers only the socket handshake;
/// model build has its own wider barrier below.
const RELAY_TIMEOUT: Duration = Duration::from_secs(120);
/// Engine-ready barrier floor: NCCL rendezvous + DeepGEMM setup + a warm-cache
/// checkpoint load fit comfortably.
const ENGINE_READY_FLOOR: Duration = Duration::from_secs(600);
/// Conservative cold-read floor for scaling the barrier with checkpoint size
/// (a 274 GB FP8 load off virtio needed ~31 min — the old fixed
/// 600 s killed every cold boot).
const ENGINE_READY_COLD_READ_BPS: u64 = 100 << 20;
const ENGINE_READY_CAP: Duration = Duration::from_secs(2 * 3600);

/// Ready barrier scaled to the checkpoint: floor + dir bytes at a cold-read
/// rate, capped. A missing/opaque path falls back to the floor (the crashed-
/// child fail-fast below stays the real guard against a wedged build).
fn engine_ready_timeout(model_path: &str) -> Duration {
    let bytes: u64 = std::fs::read_dir(model_path)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok()?.metadata().ok())
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0);
    (ENGINE_READY_FLOOR + Duration::from_secs(bytes / ENGINE_READY_COLD_READ_BPS))
        .min(ENGINE_READY_CAP)
}

// (Model-kind gating lives in `infer_api::cuda_model_takes_multiproc_serve` —
// one classifier shared with `build_cuda_engine`, covering DSv4 AND the
// Qwen3.5/3.6 MoE TpRuntime consumer.)

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

/// RAII guard for the worker children: holds their pipe write-ends, so dropping
/// it EOFs the workers. Must outlive the serve loop to keep the workers up.
pub(crate) struct CoordinatorGuard {
    _children: WorkerChildren,
}

/// What [`bind_relay_and_spawn_workers`] hands back: the accepted relay (drives
/// the coordinator HTTP loop) + the child guard (keeps workers alive).
pub(crate) struct MultiprocCoordinator {
    pub(crate) relay: RelayCoordinator,
    pub(crate) guard: CoordinatorGuard,
}

/// COORDINATOR: bind the relay, publish its port, spawn ALL N workers (rank
/// 0..N-1; rank 0 owns the visible output), accept their connects, and boot-ping.
/// Returns the accepted relay + child guard for the coordinator HTTP loop; on
/// `world_size <= 1` returns `None` (single GPU, byte-identical serve). The parent
/// owns no TP rank and joins no collective, so it is never the missing rank.
pub(crate) fn bind_relay_and_spawn_workers(
    model_path: &str,
    engine_config: &EngineLoadConfig,
) -> Result<Option<MultiprocCoordinator>> {
    let world_size = world_size_from_env();
    if world_size <= 1 {
        log::info!(
            "[multiproc-coord] world_size={world_size}; serving single-process (no workers)"
        );
        return Ok(None);
    }

    // Config parity (load-bearing): workers must build from the SAME config — any
    // scheduler-knob divergence diverges the planner across ranks → NCCL deadlock.
    // Published via env before spawn so children inherit it.
    let config_json =
        serde_json::to_string(engine_config).context("serialize ARLE_WORKER_ENGINE_CONFIG")?;
    // SAFETY: env write happens before child spawn, on the single CLI thread.
    unsafe {
        std::env::set_var("ARLE_WORKER_ENGINE_CONFIG", &config_json);
    }

    // 0. Mint the NCCL rendezvous id ONCE and publish via env before spawn so every
    //    worker inherits the SAME handle (executors read `INFER_NCCL_UNIQUE_ID` at
    //    construction). Skip if already set (an external launcher provided it).
    #[cfg(feature = "nccl")]
    if std::env::var("INFER_NCCL_UNIQUE_ID").is_err() {
        let hex = infer_api::mint_nccl_unique_id_hex().context("mint NCCL unique id")?;
        // SAFETY: single CLI thread, pre-spawn, pre-tokio (same as the relay-port write).
        unsafe {
            std::env::set_var("INFER_NCCL_UNIQUE_ID", &hex);
        }
        log::info!("[multiproc-coord] minted NCCL unique id (published via INFER_NCCL_UNIQUE_ID)");
    }
    // cfg!() (not #[cfg]) so the relay/spawn code below stays reachable in
    // non-nccl builds — an attribute-cfg bail makes the rest of the function
    // unreachable and trips -D warnings on the cuda,no-cuda typecheck lane.
    if !cfg!(feature = "nccl") {
        anyhow::bail!(
            "Multi-rank TP serve (world_size={world_size}) requires the `nccl` feature; \
             rebuild with --features cuda,nccl"
        );
    }

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

    // 2. Spawn ALL N worker processes (rank 0..world_size); rank 0 is a child too.
    let mut children = spawn_workers(model_path, world_size)?;

    // 3. Accept ALL N worker relay connects (rank 0 included).
    let mut relay = pending
        .accept_symmetric(world_size, RELAY_TIMEOUT)
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

    // 5. Engine-ready barrier: block until all N ranks report `EngineReady` before
    //    binding HTTP, else requests broadcast into socket buffers while workers are
    //    still in NCCL rendezvous. Fails fast if a child exits during the build.
    let ready_timeout = engine_ready_timeout(model_path);
    log::info!("[multiproc-coord] engine-ready barrier: {ready_timeout:?} (checkpoint-scaled)");
    wait_all_engines_ready(&relay, &mut children, world_size, ready_timeout)?;
    log::info!("[multiproc-coord] all {world_size} worker engines ready; opening HTTP");

    Ok(Some(MultiprocCoordinator {
        relay,
        guard: CoordinatorGuard {
            _children: children,
        },
    }))
}

/// Block until all `world_size` ranks report [`RelayEnvelope::EngineReady`], or
/// fail fast if a child exits first (a crashed rank never acks). Bounded by
/// the checkpoint-scaled [`engine_ready_timeout`].
fn wait_all_engines_ready(
    relay: &RelayCoordinator,
    children: &mut WorkerChildren,
    world_size: usize,
    timeout: Duration,
) -> Result<()> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let ready = relay.ready_count();
        if ready >= world_size {
            return Ok(());
        }
        // A child that exited during the build never acks → survivors would hang.
        if let Some((rank, code)) = children.first_exited() {
            anyhow::bail!(
                "worker rank {rank} exited (code {code:?}) during engine build \
                 ({ready}/{world_size} ready); aborting coordinator"
            );
        }
        if std::time::Instant::now() >= deadline {
            anyhow::bail!(
                "timed out after {timeout:?} waiting for worker engine-ready \
                 ({ready}/{world_size} ready)"
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// WORKER entry, hooked at the top of `cli::run()` before clap parsing. Returns
/// `Some(ExitCode)` for a worker (any rank with `ARLE_WORKER_RANK` set, including
/// rank 0); `None` for the parent coordinator (which never sets it on itself).
#[must_use]
pub(crate) fn worker_entry() -> Option<ExitCode> {
    let rank: usize = std::env::var("ARLE_WORKER_RANK").ok()?.parse().ok()?;
    infer_util::logging::init_stderr_with_prefix("info", &format!("[rank{rank}] "));
    match run_worker_mode(rank) {
        Ok(()) => Some(ExitCode::SUCCESS),
        Err(err) => {
            eprintln!("[arle-worker rank={rank}] failed: {err:#}");
            Some(ExitCode::FAILURE)
        }
    }
}

/// Worker-mode body (rank R, 0..N-1). Pre-connects the relay, builds the rank-R
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

    // 1. Connect the relay BEFORE building the engine (load-bearing order): the
    //    coordinator blocks in accept until all workers connect, so building NCCL
    //    first would deadlock against an un-accepted coordinator.
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

    // 2. Build the rank-R engine from the coordinator's config
    //    (`ARLE_WORKER_ENGINE_CONFIG` — config parity is a lockstep invariant);
    //    TP rank / world / EP / NCCL come from env. Built on THIS thread (no
    //    background loop); the driver below steps it once per relayed tick.
    let engine_config: EngineLoadConfig = match std::env::var("ARLE_WORKER_ENGINE_CONFIG") {
        Ok(json) => serde_json::from_str(&json)
            .with_context(|| format!("worker rank {rank} ARLE_WORKER_ENGINE_CONFIG parse"))?,
        Err(_) => {
            anyhow::bail!(
                "worker rank {rank} missing ARLE_WORKER_ENGINE_CONFIG (set by coordinator); \
                 refusing to guess a config — divergent scheduler knobs deadlock the lockstep"
            )
        }
    };
    // Rank 0 owns the visible output (tracks + emits completions); followers discard.
    let owns_output = rank == 0;
    log::info!(
        "[arle-worker rank={rank}] building rank-{rank} engine (owns_output={owns_output}) ({engine_config:?})"
    );
    let engine = CudaWorkerEngine::load(&model_path, &engine_config, owns_output)
        .with_context(|| format!("worker rank {rank} engine build"))?;
    log::info!("[arle-worker rank={rank}] engine built; entering lockstep driver");

    // 2b. Ack readiness (engine + NCCL rendezvous built) before the driver, so the
    //     coordinator unblocks its HTTP bind. A send failure means it is gone.
    if let Some(relay) = relay.as_mut() {
        relay
            .send(&RelayEnvelope::EngineReady { rank })
            .with_context(|| format!("worker rank {rank} engine-ready ack"))?;
        log::info!("[arle-worker rank={rank}] engine-ready ack sent");
    }

    // 3. Lockstep driver: one engine step per relayed `TickAdmissions`.
    if let Some(relay) = relay.as_mut() {
        run_lockstep_driver(rank, relay, engine)?;
    }

    // 4. Block on the parent pipe; the coordinator closing it EOFs us into exit.
    block_on_parent_pipe(rank);
    Ok(())
}

/// Drive the worker engine in lockstep: inject one tick's admissions, then step
/// once, so every rank admits the same requests at the same step index and the
/// per-step NCCL collective sequences stay aligned. A `seq` gap means a dropped
/// tick — fatal (a missing tick deadlocks NCCL), so die loudly rather than spin.
/// After each step the rank-0 driver also ships terminal completions back to the
/// coordinator (`drain_completions` is a no-op on followers).
fn run_lockstep_driver(
    rank: usize,
    relay: &mut RelayWorker,
    mut engine: CudaWorkerEngine,
) -> Result<()> {
    let mut next_seq: u64 = 0;
    loop {
        match relay.recv()? {
            Some(RelayEnvelope::TickAdmissions { seq, requests }) => {
                ensure!(
                    seq == next_seq,
                    "worker rank {rank} lockstep tick gap: got seq {seq}, expected {next_seq}"
                );
                next_seq += 1;
                for wire in requests {
                    log::debug!(
                        "[arle-worker rank={rank}] tick #{seq} inject req#{} (prompt_tokens={}, max_tokens={})",
                        wire.request_id,
                        wire.prompt_tokens.len(),
                        wire.max_tokens
                    );
                    let request_id = wire.request_id;
                    // Lockstep workers mirror rank-0 tokens; the grammar
                    // constraint is applied once, on the rank that owns the matcher.
                    let (prompt_tokens, max_tokens, sampling, _) = wire.submit_args();
                    engine.inject(request_id, prompt_tokens, max_tokens, sampling);
                }
                if !engine.is_idle() {
                    engine
                        .step()
                        .with_context(|| format!("worker rank {rank} step (tick #{seq})"))?;
                }
                // Emit terminal completions (rank 0 only); a send failure means the
                // coordinator is gone, so stop and let the parent pipe reap us.
                for (request_id, delta) in engine.drain_completions() {
                    relay
                        .send(&RelayEnvelope::Completion { request_id, delta })
                        .with_context(|| {
                            format!(
                                "worker rank {rank} completion send (req#{request_id}, tick #{seq})"
                            )
                        })?;
                }
                // Followers never drain_completions (owns_output-gated), so
                // without this their request_id->handle map (kept for
                // CancelRequest lookups) would grow for the process lifetime.
                engine.prune_finished();
                // Flow-control ack, sent unconditionally (an idle skip still
                // consumes the tick — the coordinator may legitimately over-send
                // decode-only ticks). Paces the coordinator's tick window.
                relay
                    .send(&RelayEnvelope::TickAck { rank, seq })
                    .with_context(|| format!("worker rank {rank} tick ack (tick #{seq})"))?;
            }
            Some(RelayEnvelope::CancelRequest { request_id }) => {
                log::debug!("[arle-worker rank={rank}] cancel req#{request_id}");
                engine.cancel(request_id);
            }
            Some(RelayEnvelope::StatsQuery { request_id }) => {
                let prefix = engine.prefix_cache_stats();
                let throughput = engine.throughput_stats();
                let tier = engine.kv_tier_stats();
                let system = engine.kv_system_metrics();
                let spec = engine.spec_decode_stats();
                let operator_dispatch = engine.operator_dispatch_stats();
                let op_timing = engine.op_timing_stats();
                let artifact = engine.artifact_identity();
                let data = WireStats {
                    build_identity: infer_api::build_identity(artifact),
                    active_requests: engine.active_count(),
                    queue_depth: engine.waiting_count(),
                    kv_free_pages: engine.kv_free_pages(),
                    throughput_steps: throughput.steps,
                    throughput_prefill_tokens: throughput.prefill_tokens,
                    throughput_generated_tokens: throughput.generated_tokens,
                    throughput_requests_completed: throughput.requests_completed,
                    throughput_forward_busy_micros: throughput.forward_busy_micros,
                    throughput_prefill_forward_steps: throughput.prefill_forward_steps,
                    throughput_prefill_forward_busy_micros: throughput.prefill_forward_busy_micros,
                    throughput_decode_forward_steps: throughput.decode_forward_steps,
                    throughput_decode_forward_busy_micros: throughput.decode_forward_busy_micros,
                    throughput_mixed_forward_steps: throughput.mixed_forward_steps,
                    throughput_mixed_forward_busy_micros: throughput.mixed_forward_busy_micros,
                    throughput_decode_phase_steps: throughput.decode_step_phase.steps,
                    throughput_decode_phase_poll_micros: throughput.decode_step_phase.poll_micros,
                    throughput_decode_phase_apply_output_micros: throughput
                        .decode_step_phase
                        .apply_output_micros,
                    throughput_decode_phase_poll_background_micros: throughput
                        .decode_step_phase
                        .poll_background_micros,
                    throughput_decode_phase_admit_micros: throughput.decode_step_phase.admit_micros,
                    throughput_decode_phase_plan_micros: throughput.decode_step_phase.plan_micros,
                    throughput_decode_phase_submit_micros: throughput
                        .decode_step_phase
                        .submit_micros,
                    prefix_lookups: prefix.lookups,
                    prefix_hits: prefix.hits,
                    prefix_hit_tokens: prefix.hit_tokens,
                    prefix_hit_pages: prefix.hit_pages,
                    prefix_published_pages: prefix.published_pages,
                    prefix_cached_pages: prefix.cached_pages,
                    kv_tier_demoted_pages: tier.demoted_pages,
                    kv_tier_promoted_pages: tier.promoted_pages,
                    kv_tier_promote_failures: tier.promote_failures,
                    kv_tier_resident_blocks: tier.resident_blocks,
                    kv_tier_demoted_slots: tier.demoted_slots,
                    kv_tier_promoted_slots: tier.promoted_slots,
                    kv_tier_slot_promote_failures: tier.slot_promote_failures,
                    kv_system_resident_pages: system.resident_pages,
                    kv_system_resident_evictable_pages: system.resident_evictable_pages,
                    kv_system_host_demoted_pages: system.host_demoted_pages,
                    kv_system_host_demoted_pending_inflight: system.host_demoted_pending_inflight,
                    kv_system_disk_pages: system.disk_pages,
                    kv_system_reuse_hit_resident: system.reuse_hit_resident,
                    kv_system_reuse_hit_host_demoted: system.reuse_hit_host_demoted,
                    kv_system_reuse_hit_disk: system.reuse_hit_disk,
                    kv_system_reuse_miss: system.reuse_miss,
                    kv_system_demote_mset_count: system.demote_mset_count,
                    kv_system_demote_mset_copy_bytes: system.demote_mset_copy_bytes,
                    kv_system_demote_mset_copy_ms: system.demote_mset_copy_ms,
                    kv_system_promote_mget_count: system.promote_mget_count,
                    kv_system_promote_mget_copy_bytes: system.promote_mget_copy_bytes,
                    kv_system_promote_mget_copy_ms: system.promote_mget_copy_ms,
                    kv_system_fetch_wait_ms: system.fetch_wait_ms,
                    kv_system_fallback_recompute: system.fallback_recompute,
                    kv_system_prefix_match_full_blocks: system.prefix_match_full_blocks,
                    kv_system_prefix_match_clamped_blocks: system.prefix_match_clamped_blocks,
                    kv_system_tier_io_mode: system.tier_io_mode,
                    kv_system_tier_io_useful_read_bytes: system.tier_io_useful_read_bytes,
                    kv_system_tier_io_useful_write_bytes: system.tier_io_useful_write_bytes,
                    kv_system_tier_io_submitted_read_bytes: system.tier_io_submitted_read_bytes,
                    kv_system_tier_io_submitted_write_bytes: system.tier_io_submitted_write_bytes,
                    kv_system_tier_io_metadata_write_bytes: system.tier_io_metadata_write_bytes,
                    kv_system_tier_io_failures: system.tier_io_failures,
                    kv_system_tier_io_completion_wait_ns: system.tier_io_completion_wait_ns,
                    spec_chains: spec.chains,
                    spec_drafted: spec.drafted,
                    spec_accepted: spec.accepted,
                    spec_rejected: spec.rejected,
                    spec_partial_ctx_chains: spec.partial_ctx_chains,
                    operator_dispatch,
                    op_timing,
                };
                if let Err(e) = relay.send(&RelayEnvelope::StatsResponse {
                    request_id,
                    data: Box::new(data),
                }) {
                    log::warn!("[arle-worker rank={rank}] stats response send failed: {e:#}");
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
            Some(RelayEnvelope::TickAck {
                rank: ack_rank,
                seq,
            }) => {
                log::warn!(
                    "[arle-worker rank={rank}] unexpected tick-ack envelope (rank {ack_rank}, seq {seq})"
                );
            }
            Some(RelayEnvelope::EngineReady { rank: ready_rank }) => {
                log::warn!(
                    "[arle-worker rank={rank}] unexpected engine-ready envelope (rank {ready_rank})"
                );
            }
            Some(RelayEnvelope::StatsResponse { request_id, .. }) => {
                log::warn!(
                    "[arle-worker rank={rank}] unexpected stats response envelope request_id={request_id}"
                );
            }
            Some(RelayEnvelope::Shutdown) => {
                log::info!("[arle-worker rank={rank}] relay shutdown envelope received");
                return Ok(());
            }
            None => {
                log::info!("[arle-worker rank={rank}] relay EOF after {next_seq} lockstep ticks");
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

impl WorkerChildren {
    /// `(rank, exit_code)` of the first already-exited child, else `None`. Lets the
    /// engine-ready barrier fail fast on a worker that crashed during the build.
    fn first_exited(&mut self) -> Option<(usize, Option<i32>)> {
        for (rank, child, _) in &mut self.children {
            if let Ok(Some(status)) = child.try_wait() {
                return Some((*rank, status.code()));
            }
        }
        None
    }
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
    let exe = std::env::current_exe().context("current_exe")?;
    let ordinals = cuda_ordinals(world_size);
    let mut children = Vec::with_capacity(world_size);

    // SPMD (B): spawn ALL ranks 0..world_size as identical child workers (rank 0 is
    // the output owner); the engine-less parent never sets `ARLE_WORKER_RANK`.
    for rank in 0..world_size {
        // Parent -> child pipe. Only the read end is inherited by the exec'd
        // worker; the parent write end stays close-on-exec so the worker cannot
        // keep its own writer alive and mask EOF after coordinator death.
        let (child_read_end, parent_write_end) = worker_parent_pipe(rank)?;
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
        // The TP executors read their TP rank from INFER_TP_RANK; bind it to this
        // worker's rank so the per-rank EP split + NCCL rank match.
        cmd.env("INFER_TP_RANK", rank.to_string());
        cmd.env("INFER_TP_SIZE", world_size.to_string());
        // MASTER_ADDR/PORT + INFER_NCCL_ID_FILE/INFER_NCCL_UNIQUE_ID are
        // inherited from the coordinator's env (set before spawning workers).

        let child = cmd
            .spawn()
            .with_context(|| format!("spawn worker rank {rank}"))?;

        // After spawn the parent doesn't need its copy of the child-side read fd;
        // the worker inherited the same fd number across exec.
        drop(child_read_end);
        let pid = child.id();
        children.push((rank, child, parent_write_end));
        log::info!("spawned worker rank {rank} pid={pid} cuda_ordinal={cuda_ordinal}");
    }

    Ok(WorkerChildren { children })
}

fn worker_parent_pipe(rank: usize) -> Result<(std::fs::File, std::fs::File)> {
    let mut fds = [0i32; 2];
    make_cloexec_pipe(&mut fds)
        .with_context(|| format!("create parent pipe for worker rank {rank}"))?;
    // Child must inherit exactly this fd across exec; the write end remains
    // FD_CLOEXEC and is held only by the coordinator process.
    if let Err(err) = set_fd_cloexec(fds[0], false)
        .with_context(|| format!("clear CLOEXEC on worker rank {rank} read pipe"))
    {
        // SAFETY: make_cloexec_pipe succeeded, so both fds are owned here.
        unsafe {
            libc::close(fds[0]);
            libc::close(fds[1]);
        }
        return Err(err);
    }
    // SAFETY: make_cloexec_pipe initialized both fds and ownership moves into
    // File. On error before this point, the process exits setup with no child.
    let child_read_end = unsafe { std::fs::File::from_raw_fd(fds[0]) };
    let parent_write_end = unsafe { std::fs::File::from_raw_fd(fds[1]) };
    Ok((child_read_end, parent_write_end))
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn make_cloexec_pipe(fds: &mut [i32; 2]) -> Result<()> {
    // SAFETY: pipe2(2) writes two owned fds into `fds` on success.
    let ret = unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) };
    if ret != 0 {
        anyhow::bail!(
            "pipe2(O_CLOEXEC) failed: {}",
            std::io::Error::last_os_error()
        );
    }
    Ok(())
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn make_cloexec_pipe(fds: &mut [i32; 2]) -> Result<()> {
    // SAFETY: pipe(2) writes two owned fds into `fds` on success.
    let ret = unsafe { libc::pipe(fds.as_mut_ptr()) };
    if ret != 0 {
        anyhow::bail!("pipe(2) failed: {}", std::io::Error::last_os_error());
    }
    if let Err(err) = set_fd_cloexec(fds[0], true).and_then(|_| set_fd_cloexec(fds[1], true)) {
        // SAFETY: pipe(2) succeeded, so both fds are owned and must be closed on
        // setup failure to avoid leaking them in the coordinator.
        unsafe {
            libc::close(fds[0]);
            libc::close(fds[1]);
        }
        return Err(err);
    }
    Ok(())
}

fn set_fd_cloexec(fd: RawFd, cloexec: bool) -> Result<()> {
    // SAFETY: fcntl(F_GETFD/F_SETFD) does not take ownership of `fd`.
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags < 0 {
        anyhow::bail!(
            "fcntl(F_GETFD) failed for fd {fd}: {}",
            std::io::Error::last_os_error()
        );
    }
    let next = if cloexec {
        flags | libc::FD_CLOEXEC
    } else {
        flags & !libc::FD_CLOEXEC
    };
    let ret = unsafe { libc::fcntl(fd, libc::F_SETFD, next) };
    if ret != 0 {
        anyhow::bail!(
            "fcntl(F_SETFD) failed for fd {fd}: {}",
            std::io::Error::last_os_error()
        );
    }
    Ok(())
}
