# B (SPMD) — line-level implementation spec

Companion to `2026-06-24-multiproc-control-data-plane-redesign.md` (read that for
the why). This is the executable spec: every change site, mapped. Goal — one
cohesive diff, no half-states, `cargo check` green for both `cpu,no-cuda` and
`cuda,no-cuda`, single-process (TP=1) path BYTE-UNCHANGED.

## Invariant to preserve
- **TP=1 / single-process serve is unchanged.** All B logic is gated on multiproc
  mode (`world_size > 1`). The in-process fast path stays for TP=1.
- Backend-neutral: edits live in `cli` / `infer-server` / `infer-api` only. No
  `infer-cuda` types leak up.

## Change sites (file:line at HEAD; re-grep, lines drift)

### 1. RelayTransport trait — `crates/infer-server/src/multiproc_relay.rs`
- Today: `Coordinator { workers: BTreeMap<usize, TcpStream> }` (~:209), `RelayWorker
  { stream: TcpStream }` (~:411), `write_envelope(&mut TcpStream,..)` (~:553),
  `read_envelope(&mut TcpStream)` (~:561), broadcast iterates streams (~:381).
- Do: define `trait RelayChannel { fn send(&mut self, &RelayEnvelope)->Result<()>;
  fn recv(&mut self)->Result<RelayEnvelope>; }`. Impl it for the current TCP stream
  (`TcpChannel(TcpStream)`). Change `write_envelope`/`read_envelope` to take `&mut
  dyn RelayChannel` (or generic). Coordinator/worker hold the trait object. **No
  behavior change** — TCP is the only impl this phase. This is the seam shmipc would
  later plug into; do NOT implement shm now (killed per the design's measurement).

### 2. Symmetric spawn — `crates/cli/src/serve_multiproc.rs`
- `spawn_workers` (~:474-521): spawn `0..world_size` (was `1..world_size`). Rank 0
  becomes a child too.
- Config parity: rank 0 must also receive `ARLE_WORKER_ENGINE_CONFIG` (the
  serialized config at ~:136-145) — make the config flow symmetric for all ranks.
- `worker_entry` (~:233-247): remove the `rank==0 → return None` special-case; ALL
  ranks (0..N-1) run `run_worker_mode`. `run_worker_mode` (~:253-323) is already
  rank-agnostic (pre-connect relay, build engine, run lockstep driver) — rank 0
  now runs it identically.

### 3. Thin coordinator (no in-process engine) — `crates/cli/src/serve.rs` + `serve_multiproc.rs`
- `serve.rs:160-166`: in multiproc mode, do NOT build the in-process rank-0 engine.
  The parent process, after `bind_relay_and_spawn_workers`, runs the **coordinator
  loop** instead of `serve_http`'s engine path.
- The coordinator owns: the HTTP server (ingress), request-id allocation, per-tick
  admission broadcast to ALL N workers, completion collection from the output-owner
  rank, response routing. It holds NO executor and joins NO collective.
- `bind_relay_and_spawn_workers` (~:110-225): now accepts N worker connects (was
  N-1). The tick broadcaster (~:203-219) broadcasts to all N.

### 4. Stage-3 completion return — `crates/infer-server/src/multiproc_relay.rs` + execution
- Today only rank-0's in-process engine feeds HTTP; worker output is discarded
  (`RelayEnvelope::Completion`, ~:168; `RelayCompletionDelta { token_ids }`, ~:116;
  sink registry `register/unregister_completion_sink` ~:355-377 exists but unused by
  HTTP).
- Do: the **output-owner rank** (rank 0 today — the visible-output owner) emits its
  per-step token deltas as `RelayEnvelope::Completion` back to the coordinator over
  its RelayChannel. The coordinator's completion reader (~:476-537) routes each
  delta to the per-request sink. Wire the HTTP layer to **register a sink per
  request, await deltas, unregister on finish** (this replaces reading the
  in-process engine).
- `execution.rs:221-246` (TickAdmissions build/broadcast) + `:124-137` (tick
  broadcaster install): the coordinator drives ticks now; ensure request-id space is
  coordinator-allocated and echoed in completions so the sink match works.

### 5. async(HTTP)/sync(relay) bridge — `crates/infer-api/src/serve.rs`
- HTTP is tokio (`serve_http` ~:193-260). The relay is sync (blocking TCP). Bridge:
  a dedicated blocking thread drains the relay completion reader → a `tokio::sync::
  mpsc` per-request (or a shared completion router) the async handlers await. Do NOT
  async-rewrite the relay.

## Gate (this workflow): `cargo check` green
- `cargo check -p agent-infer --no-default-features --features cpu,no-cuda,cli`
- `cargo check -p infer-api --no-default-features --features cuda,no-cuda --lib`
- `cargo test -p infer-server --no-default-features` (relay unit tests still pass).
- TP=1 path visibly unchanged (gated on world_size>1).

## Gate (pod, done by Claude after): EP=4 deepep_ll boots to ready deterministically
(no rank-0 stall) + needle exact + c跑 ok>0; then EP=8.
