# DSv4 multi-rank lockstep admission — kill the c≥2 plan-divergence deadlock

> Status: **LANDED `23b69249`** (ck ack 2026-06-10 "你review再然后开工吧";
> review upgraded the design — see §Design revision). Verified on 8×H20:
> c=2/4/8 burst+stagger all complete, 1150/1150 ticks rank-uniform incl. 140
> Mixed ticks, B=1 −0.3% (noise), batched-decode arm +57% agg @ c=8. Results:
> [`wins/2026-06-10-dsv4-lockstep-admission-c-sweep-lane-exists.md`](../experience/wins/2026-06-10-dsv4-lockstep-admission-c-sweep-lane-exists.md).
>
> **Design revision (review, pre-implementation):** the sketch below
> ("rank 0 sends one TickAdmissions per *forward-executing* tick"; an earlier
> variant stamped admit-ticks with a margin K) had two holes — empty plans run
> no collectives so any forward/tick counter drifts across ranks, and an
> idle-system stamp can never be reached (self-deadlock). What landed is
> **message-per-step**: rank 0 sends exactly one `TickAdmissions{seq,requests}`
> before EVERY `Engine::step` (empty list allowed); workers own their Engine
> directly (`CudaWorkerEngine`, no background loop) and step exactly once per
> envelope. Sound because the CUDA executor's `submit` is synchronous with
> `poll` always `Ready` (one forward per step), sampling is
> `(seed,position)`-deterministic, and plan-building is pure in engine state.

## Problem — two layers behind "c≥2 crashes"

The 2026-06-10 deepep error entry blamed the single-row executor `ensure!`.
That was only **Layer 1** (engine thread dies on Mixed plans). With Layer 1
fixed (`cd421794`: per-prefill sub-steps + decode sub-batch via
`KvBatchDescriptor::subset`), c≥2 advances to **Layer 2**: the engine no longer
dies — the 8-rank serve **deadlocks** (GPUs 100% util at ~120 W = NCCL
collective spin; engine makes no progress). This is the 2026-06-08 multirank
entry's known follow-up #3, now reproducible at will.

## Root cause (source-confirmed race; runtime fingerprint evidence §Evidence)

The multi-rank serve runs a **full symmetric Engine per rank**, fed by an
async request relay:

- rank 0: HTTP → `Submission` mpsc → `engine_loop` drains via `try_recv` at
  the top of each tick (`infer-server/src/execution.rs:132`), broadcasting each
  admission to workers (`admit_submission` → `broadcast_admission`,
  `execution.rs:211`) before local submit.
- workers: a relay-receiver **thread** submits relayed requests into the
  worker's own engine mpsc (`cli/src/serve_multiproc.rs`,
  `run_relay_receiver`), drained by that worker's `engine_loop` at *its* tick
  top.

Lockstep therefore assumes "identical request order ⇒ identical batches". But
admission is **timing-coupled**: a request lands in rank 0's queue at T and in
a worker's at T+δ (TCP). If T falls between rank A's tick-top drain and rank
B's, A plans it into forward #k, B into #k+1 → different row sets → different
collective sequences (per-layer all-reduces count differs between prefill
sub-steps and decode batches) → NCCL deadlock. The race window is the whole
step duration (25–160 ms); at c≥2 arrival rates it is hit almost immediately.
B=1 cannot hit it (one request in flight; every plan is forced identical).

**Second divergence source (config):** workers build engines with
`EngineLoadConfig::default()` (`serve_multiproc.rs` worker path) while rank 0
uses the CLI-resolved config. Any non-default `num_slots` / chunk size /
prefill budget / prefix-cache flag diverges the planner deterministically —
this is the likely mechanism of multirank follow-up #2 (multi-chunk prefill
crash at c=1: `slot seq_len != start_pos`).

## Evidence (2026-06-10 pod run, binary with `cd421794`+`78553406`)

- Hang repro (`desync_repro.py`): B=1 sanity 5.8 s `" Paris…"` OK; c=2 burst →
  **both requests 90 s timeout**, 8 GPUs 100% util @ ~120 W (collective spin),
  no engine error in the log (vs Layer 1's instant `ensure!` death on the same
  lane pre-`cd421794`).
- Plan fingerprints (`78553406`, `RUST_LOG=infer_cuda=debug`): all 7 workers
  byte-uniform through tick 8 — sanity prefill `(0,0,5)` + 7 decodes
  `(0,5)…(0,11)` at ~25 ms/tick, then c2-request-A's prefill `(0,0,5)` (slot 0
  reused) at tick 8. **Tick 9 on every worker = `decode=[(0,5)]` only — no
  prefill row for request B — and no tick 10 ever appears** (all workers stuck
  inside tick 9's forward). rank 0 alone had drained request B from its
  HTTP-local queue, so its tick 9 was Mixed (prefill B sub-step + decode A) —
  a different collective sequence/shape than the workers' decode-only tick.
  Had tick 9 been uniform it would have completed in ~25 ms like ticks 1–7.
  Divergence-at-admission confirmed; the race window is the full step
  duration, exactly as modeled.
- Known gap: rank 0 emits no `[dsv4-plan]` DEBUG lines (coordinator's logger
  filter differs from workers') — fix alongside this change so the uniformity
  check covers all 8 ranks.

## Fix — per-tick admission broadcast (SGLang shape)

SGLang solves exactly this: every TP rank runs a symmetric scheduler, and
**`recv_requests` is collectively synchronized per scheduling iteration**
(rank 0 drains its ZMQ queue, then `broadcast_pyobj` to all TP ranks inside
the loop). Admission becomes part of the lockstep instead of racing it.

ARLE equivalent — keep symmetric Engines, move admission into the tick:

1. **Relay message** `TickAdmissions { seq: u64, requests: Vec<WireRequest> }`
   (`infer-server/src/multiproc_relay.rs`). rank 0 sends exactly one per
   *forward-executing* tick (empty `requests` allowed); `seq` increments per
   message.
2. **rank 0 `engine_loop`** (`infer-server/src/execution.rs`): replace the
   free-running `try_recv` drain with a tick-top drain that (a) collects newly
   arrived submissions into `L`, (b) sends `TickAdmissions{seq, L}` to all
   workers, (c) admits `L` locally, (d) steps. Idle path (no active work, `L`
   empty): park as today, send nothing — workers park on recv symmetric-ly.
3. **worker loop** (`cli/src/serve_multiproc.rs`): delete the fire-and-forget
   relay-receiver→mpsc path. Drive the engine *synchronously*: block on
   `TickAdmissions`, verify `seq` is contiguous, admit, `engine.step()`, loop.
   (Worker engines never see HTTP submissions; their only admission source is
   the relay, so the blocking recv IS the lockstep barrier on the host side —
   the NCCL forward remains the device barrier.)
4. **Config parity**: serialize rank 0's resolved `EngineConfig` into the
   worker boot message (or env), replacing `EngineLoadConfig::default()` on
   workers. Closes the second divergence source (and likely follow-up #2).
5. **Determinism invariant** stays the planner's: BTreeMap iteration +
   deterministic budgets/clamps. The fingerprint log (`78553406`) is the
   permanent regression surface: a uniform-fingerprint check at c=2/4/8 rides
   every multirank change.

Cost: one localhost TCP message per tick (~50 µs) against 25–160 ms steps —
noise. No engine-core/planner changes; `infer-core` stays single-rank-pure.

### Files

| File | Change |
|---|---|
| `crates/infer-server/src/multiproc_relay.rs` | `TickAdmissions` wire message + seq; coordinator send-all; worker blocking recv |
| `crates/infer-server/src/execution.rs` | tick-top admission collect + broadcast hook (engine_loop) |
| `crates/infer-server/src/lib.rs` | broadcaster install surface carries the per-tick hook |
| `crates/cli/src/serve_multiproc.rs` | worker: synchronous step-driver loop replacing relay-receiver thread; config parity plumb |
| `crates/infer-api/src/loaded.rs` | accept serialized `EngineConfig` for worker builds |

### Gates

- c=1 byte-identical (B=1 lane unchanged vs pre-change run, same binary).
- c=2/4/8 burst + 1s-stagger lanes complete; per-request expected-substring
  probes all HIT (`dsv4_c_sweep.py`); fingerprint streams uniform across ranks.
- Multi-chunk prefill (≥2 chunks) at c=1 no longer crashes (follow-up #2).
- c-sweep TTFT/ITL/agg tok/s recorded vs the per-row and
  `INFER_DSV4_BATCHED_DECODE=1` grouped paths (the Phase 6a harness numbers:
  per-row 36.7 → grouped 51.75 agg tok/s @ c=8 were executor-direct; this
  yields the first END-TO-END serving numbers).

## Sequencing vs the deepep_ll question

This plan IS the critical path of "task #8 batched decode": grouped MoE
(Phase 6a/6b) already landed executor-side; without lockstep admission there
is no serving lane to run it in. Only after this lands is the
deepep_ll-vs-allreduce batched A/B runnable (deepep additionally needs the
batched-decode MoE transport hook — currently refused in
`forward_decode_batch_stream_impl`).

Refs: `docs/experience/errors/2026-06-10-dsv4-deepep-ll-b1-regression-no-batch-lane.md`,
`docs/experience/wins/2026-06-08-dsv4-multirank-serve-rewire.md`,
`docs/plans/2026-06-07-unified-batched-kvpool-abstraction.md`,
`docs/experience/wins/2026-06-07-dsv4-batched-decode-grouped-moe-throughput.md`.
