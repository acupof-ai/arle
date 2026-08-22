# Multiproc lockstep coordinator hangs forever when a worker silently stops acking — no timeout, no error

## Context

Found while pod-verifying the DSv4 KV three-layer refactor
(plan,
[wins entry](../wins/2026-07-05-dsv4-p1-p2-p4-needle-gate.md)): serving
DeepSeek-V4-Flash-FP8 TP=4/EP=4 on 4×H20, a request with `prompt_tokens≈8106`
hangs the entire server indefinitely (`/v1/stats` frozen at `steps=0`), while
`prompt_tokens=7661` completes normally in ~5s. Reproduced identically on two
separate commits (current HEAD and a pre-DSv4-refactor commit), so it is
**not caused by the KV-storage refactor** — this entry documents the
orthogonal bug, root-caused independently.

The first pass at this (bisection + A/B control only) guessed "VRAM squeeze
from TP=4 halving rank count vs the validated TP=8 baseline" — **that guess
was never measured and turned out wrong.** This entry supersedes it.

## Root Cause

`crates/infer-server/src/coordinator.rs:88-109` (`wait_for_ack_window`) is
the rate-limiter for the multiproc lockstep tick-broadcast protocol between
the HTTP coordinator process and the N TP worker processes
(`TICK_WINDOW=4` ticks). Its only exit conditions are `any_dead` (a worker's
relay reader marked it dead — crash/socket-close) or `seq < min_acked +
TICK_WINDOW`. **There is no deadline.** The function's own doc comment
explains the crash case is handled ("a CRASHED worker's reader marks it
dead... the next broadcast hits its dead socket and takes the `fail_all`
path") but has no equivalent for a worker that stops acking **without**
erroring or crashing — it just silently never sends another `TickAck`.

At `prompt_tokens=8106` (not 7661), gdb backtraces on a symbol build during a
live repro show:
- All 4 TP worker main threads parked in `infer_server::multiproc_relay::
  read_envelope` → `TcpChannel::recv` → `serve_multiproc::
  run_lockstep_driver`, waiting for a tick broadcast that never arrives.
- The coordinator's dedicated `arle-coordinator` thread parked in
  `coordinator::lockstep_loop` → `wait_for_ack_window` → `nanosleep`,
  spinning forever because `min_acked` never advances.
- On a second repro attempt (different binary build), the exact freeze point
  differed: rank2 was in `send()` while the other 3 ranks + coordinator relay
  readers were in `recv()` — same subsystem, same failure class (relay
  send/recv desync), different exact line — consistent with a genuine
  cross-rank timing race, not one static deadlock site.

Since every HTTP request on the server routes through this same coordinator
lockstep, once one rank silently stops acking the **entire server** hangs
forever, for every request, with zero error/log signal beyond a rate-limited
`"[coordinator] lockstep stalled"` warn every 10s (which reads, misleadingly,
identically for "a slow rank, wait it out" and "a rank that will never ack
again").

## Ruled out (measured, not inferred)

- **Not OOM/VRAM.** GPU memory across all 4 serving GPUs plateaus at
  96999/97871 MiB and stays flat for the entire hang — no growth, no
  OOM-killer, no Xid in `dmesg` during the test window (the one Xid entry
  present was 6 days stale, unrelated).
- **Not a `libnccl` collective hang.** The stuck frames are inside this
  codebase's own TCP relay (`multiproc_relay.rs`), not inside NCCL.

## Not yet pinned down (explicitly deferred, not guessed)

- **Why** a rank stops acking specifically between 7661 and 8106 prompt
  tokens — which rank, and what it is doing instead of acking (chunked-
  prefill step timing? the KV-tier host-demote path — the 7661 control run
  already showed `host_demoted_pages:13`, so write-through KV is active near
  this boundary). Needs per-rank tick-sequence instrumentation or CUDA-side
  tracing on the stalled rank, not done here.
- **TP=8 at the same prompt length is unverified** (this box only has 4 free
  GPUs; GPU1 is held by a foreign tenant). The lockstep subsystem is
  TP>1-specific by construction (`TP=1 never reaches here`), so it cannot
  reproduce at TP=1, but whether TP=8 also breaks at ~8106 tokens is open.

## First fix attempt was incomplete — retested, found the real blocking site

`b595b0e95` added `ACK_STALL_TIMEOUT` (120s) to `wait_for_ack_window`,
bounding the "poll `min_acked`, sleep, repeat" loop. Pod retest (same
TP=4/EP=4, `prompt_tokens=8106`) showed **the request still hung forever** —
two independent repro attempts, one waited out a 600s client timeout with
zero response. gdb on a symbol build caught the actual freeze:

```
#0  send() [libc]
#3  infer_server::multiproc_relay::write_envelope
#4  infer_server::multiproc_relay::RelayCoordinator::broadcast
#5  infer_server::coordinator::lockstep_loop
```

The coordinator's lockstep thread was wedged **one call earlier** than the
fix reaches — inside the raw blocking `send()` that `broadcast()` does to
push a tick, while holding the `RelayCoordinator` mutex. `TcpStream` has no
write timeout by default, so a peer not draining its receive buffer blocks
this write indefinitely; `wait_for_ack_window` never gets a chance to run
(and its new timeout never fires) because the loop never reaches it for that
tick. `grep -c "lockstep stalled\|lockstep ack wait exceeded"` on the retest
logs was `0` — direct confirmation the new code path was never reached. This
is exactly the "rank2 was in `send()`" alternate freeze point the original
investigation had already flagged as a second, distinct freeze location.

**Second fix**: `set_write_timeout` (30s, shorter than `ACK_STALL_TIMEOUT` —
a full send buffer is a more acute symptom than "still computing") on every
`TcpChannel`, coordinator and worker side, at connection setup. A timed-out
write now surfaces as an `Err` through the existing `write_all`/`?` chain,
which `lockstep_loop`'s pre-existing broadcast-error branch already tears
down on — no new plumbing needed for that half.

## Second fix (`57606c63c`) is sound but doesn't fix the reported hang — the freeze is one layer deeper

Retest 2 (same repro, `prompt_tokens=8106`, fresh build with both fixes):
**still hangs forever** — 590s+ client timeout, zero bytes, zero HTTP
response (not even an error). `grep` for every fix's log signature
(`"lockstep stalled"`, `"lockstep ack wait exceeded"`, any broadcast-error
line) on the server log: **zero matches, neither fired.** gdb this time
shows a clean relay layer — all 4 workers parked normally in
`TcpChannel::recv` waiting for a tick, the coordinator's lockstep thread
parked in the fully-idle branch (`submit_rx.recv_timeout(IDLE_PARK)`,
`in_flight == 0`). One snapshot caught rank0 inside `Engine::step →
try_admit_front_waiter → cached_prefix_match_len →
TpRuntime::all_reduce_min_scalar_i32` (an NCCL collective in the admission
path) — real engine-side work, not a relay block.

**Both TCP fixes are validated correct and kept** (their own unit/repro tests
pass, and this run's regression check confirms the relay layer itself is
healthy — sockets clean, `/v1/stats` round-trips throughout). They were
just never the actual mechanism for *this* specific hang.

**New mechanism, inferred from the evidence (not yet file:line pinned):** the
HTTP client's own timeout fires and cancels the request client-side;
`InFlightGuard::drop` (`coordinator.rs`) decrements `in_flight` and
unregisters the sink, but does **not** cancel anything inside the engine —
the request is still sitting half-admitted in each worker's internal state.
Once `in_flight` hits 0, `lockstep_loop` stops broadcasting ticks entirely
(idle), so the abandoned request never gets popped, retried, or torn down —
a permanent head-of-line zombie. The control request (`prompt_tokens=7661`,
independently verified healthy standalone: 4.97s, HTTP 200) queued behind
this zombie in the same per-rank FIFO and **also hung forever** — reproducing
the original "hangs the entire server" symptom through a third, different
mechanism than either landed fix.

Two separate open questions this surfaces, not one:
1. Why does `try_admit_front_waiter`'s admission (the `all_reduce_min_scalar_i32`
   cross-rank collective) never complete or error for this specific
   8106-token shape — an actual stuck collective, or just legitimately very
   slow and never given long enough to finish? Not distinguished yet.
2. **`InFlightGuard::drop` only touches coordinator-side bookkeeping — it
   never propagates a cancellation into the engine.** This looks like a real,
   separate gap regardless of (1): a client that disconnects/times out should
   free the engine-side resources its abandoned request holds, not leave a
   zombie that head-of-line-blocks every later request on the same rank.

## Rule

- **A retry/wait loop gating shared server state (here: every HTTP request)
  needs an explicit deadline**, not just a dead-worker check — "the worker
  will error before it silently stops responding" is an assumption, not a
  guarantee, and when it's wrong the failure mode is a permanent, silent,
  whole-server hang instead of a bounded error.
- **A guessed root cause ("VRAM squeeze") from bisection + A/B alone, without
  a measured mechanism, is not licensed** — the A/B correctly ruled out "this
  diff caused it" but said nothing about *why*. Root-cause hypotheses need
  their own verification (here: `nvidia-smi` during the hang + gdb
  backtraces), same as any other claim.
- **A fix for "the retry loop can hang" must bound every blocking call in the
  call chain, not just the one with the obvious retry loop** — the ack-wait
  loop was the visible poll-and-sleep pattern, but the actual freeze was one
  synchronous, unrelated-looking `write_all()` call earlier in the same
  function that nothing in the loop protects. Retest on the real repro (not
  just a unit test of the fixed function in isolation) is what caught this —
  the unit test for `wait_for_ack_window_impl` passed and proved nothing
  about the call site the fix didn't reach.
- **"Same symptom" does not mean "same mechanism" — retest after every fix,
  don't assume the second hang is the first hang unresolved.** Two fixes into
  the relay layer, gdb showed a THIRD, unrelated freeze point (engine
  admission, not the TCP relay at all) producing an identical-looking
  permanent hang. Each attempt's unit tests were real and each fix was
  individually correct for the mechanism it targeted; none of that licenses
  "therefore the user-facing bug is fixed" without a fresh gdb/log check on
  the actual repro, every time. It happened AGAIN between rounds 4 and 5 —
  the correct, pod-verified TP fix still left the identical repro hanging,
  via a plain capacity check with no reject path.
- **Any "wait and retry" admission/scheduling loop needs its own "can this
  ever succeed" check, separate from "can it succeed on THIS tick."**
  `Throttled` (retry later, capacity may free up) and "structurally
  impossible" (capacity will never exist) look identical from inside the
  retry loop unless something explicitly distinguishes them — the loop
  cannot tell the difference on its own, so it must be told.

## Round 4 — SPMD admission livelock, fixed (`5fd6a8984`)

Design doc: plans/2026-07-05-spmd-admission-page-sync.md.
Live simultaneous 5-process gdb snapshots (coordinator + all 4 workers, 3
timepoints ~15-40s apart) proved the stuck rank ROTATES (rank3 →
{rank1,rank2} → rank1) while others idle — proof against a static 4-way NCCL
deadlock, proof for a moving cross-rank mismatch. Pinned to
`crates/infer-core/src/lib.rs:1087` `admit_waiting`'s `remaining_pages`
starting from `self.kv.free_pages()` (rank-local, unsynced) while the
collective it gates (`cached_prefix_match_len`) requires every rank to call
it symmetrically every tick — a rank that Admits stops calling the
collective forever, a rank that Throttles keeps calling it, and NCCL matches
calls by order not content, so they can never realign.

Fix: new `BackendExecutor::tp_sync_min` seam method (default identity for
single-rank/no-TP backends), syncing `remaining_pages`' starting value once
per `admit_waiting()` call via the same min-reduce pattern
`cached_prefix_match_len` already uses.

**Pod-verified correct for its target**: 4 ranks now cycle symmetrically
through the collective, never diverging. This closes the SPMD-livelock CLASS
of bug.

## Round 5 — the SAME repro still hung, different mechanism, fixed (`eeac3d2b9`)

With round 4's fix confirmed working, the identical `prompt_tokens=8106`
repro **still hung** — this time with all 4 ranks staying symmetric (no
divergence). A temporary diagnostic pinned it exactly, identical every tick,
every rank: `pages_needed=127 > remaining_pages=121`. This pod's TP=4 /
`mem_fraction_static=0.97` config has a hard ceiling of 121 KV pages: an
8106-token prompt (+ 1 min decode token) needs 127. With
`--max-running-requests=1`, nothing else can ever finish to free pages, so
`try_admit_front_waiter` returned `AdmitOutcome::Throttled` every tick,
forever — no error, no timeout surfaced anywhere.

This is NOT the TP livelock recurring: it reproduced identically on a
completely fresh server (`host_demoted_pages:0`, no preceding control
request) — a plain, deterministic capacity shortfall, not a cross-request or
cross-rank divergence. The `host_demoted_pages:13` residual that seemed
load-bearing in earlier rounds' hypothesis turned out not to matter for this
mechanism.

Fix: new `AdmitOutcome::Rejected` — when the pool is completely idle
(`self.active.is_empty()`, so nothing else could ever free more pages) and
the candidate still doesn't fit after maximal eviction, complete it with
`FinishReason::Abort` (the same path `submit_request_with_options` already
uses for `prompt_tokens.len() > max_prompt_tokens`) instead of throttling
forever. Deliberately conservative: with other requests still active,
behavior is unchanged (Throttle — they may free enough pages on finish), so
there is no false-positive-rejection path.

**Pod-verified** (two independently launched fresh servers, 4 repeats): the
`prompt_tokens=8106` repro now returns HTTP 200 in **0.017s** with
`{"finish_reason":"abort", "usage":{"prompt_tokens":8106,"completion_tokens":0}}`
instead of hanging — deterministic across every run. Log line, all 4 ranks,
every run: `admission reject: request needs 127 KV pages, pool has 121 free
with no other request active (prompt_len=8106)` — matches the diagnostic's
127-vs-121 numbers exactly. The `prompt_tokens=7661` control still completes
normally (HTTP 200, 4.97s cold / 0.66s prefix-cached, `finish_reason:length`)
on both servers — no false-positive rejection. Server health after
rejection: control → repro → repro → repro → control, all correct, no
degradation, no zombie state.

## Round 6 — `InFlightGuard` cancellation propagation, landed (`d9222a29a`, `b7ec09160`)

The one gap rounds 1-5 left open, deliberately deferred pending a scope
check-in: `InFlightGuard::drop` only decremented coordinator-side `in_flight`
and unregistered the sink — a client that disconnects/times out left its
request a permanent zombie in every rank's engine (`waiting` or an active
slot), never told to stop.

Fix, two commits:
- `d9222a29a` (`infer-core`, self-contained, unit-tested): new
  `Engine::cancel_request(handle)` — drops a queued request from `waiting`
  (same `complete_immediately(Abort)` path as the `max_prompt_tokens`
  rejection), or frees an active request's slot through the exact same
  `finish_slot` release path a natural completion uses. Safe no-op for an
  already-finished or unknown handle. Same MULTIPROC INVARIANT as admission:
  must be called identically, same tick, on every SPMD rank.
- `b7ec09160` (multiproc wiring): new `RelayEnvelope::CancelRequest`,
  broadcast BEFORE the tick's `TickAdmissions` (same locked `coord` scope) so
  every rank applies it at the identical point relative to that tick's step.
  `CoordSubmission::Cancel(u64)` carries it from `InFlightGuard::drop`
  (sent unconditionally — cancelling an already-finished handle is a no-op,
  so `Drop` doesn't need to know whether the stream ended naturally or was
  abandoned) into the same channel `lockstep_loop` already drains for new
  submissions. `CudaWorkerEngine.tracked` (request_id -> handle) is now
  populated on every rank, not just the output owner, with a
  `prune_finished()` cleanup step for followers (who never `drain_completions`).

Tests: 3 new `infer-core` tests (cancel while waiting / cancel while active /
no-op on finished-or-unknown), 1 new `infer-server` test proving the wire
ordering (`CancelRequest` arrives before its paired `TickAdmissions`). Full
downstream typecheck clean (cuda+no-cuda, cpu+no-cuda, metal+no-cuda), 0 new
clippy warnings.

**Pod-verified** (TP=4/EP=4, `RUST_LOG=debug`): 3 client-disconnect cycles
(client timeout + 2× `kill -9` mid-decode), each restored `kv_free_pages` to
the full 121-page pool immediately (no zombie slot) and left the server fully
healthy — a follow-up normal request completed in 1.44-1.59s every time (vs.
1.63s baseline), no hang, no desync, no `lockstep stalled`/panic/error lines
anywhere. Per-rank log evidence across the whole session: `grep 'cancel
req#'` → exactly 40 lines = 10 request IDs × 4 ranks, e.g. all 4 ranks
logging `cancel req#7` within ~20μs of each other — same-tick delivery,
exactly the invariant this fix exists to guarantee. Round 5's capacity
rejection (0.017s abort) and the 7661-token control (4.78s, matches
baseline) both unregressed.

## Status — CLOSED. All 6 rounds pod-verified; the multiproc investigation is done.

Rounds 1-3: relay-layer hardening (kept). Round 4: SPMD admission-collective
divergence (pod-verified). Round 5: capacity-shortfall reject path
(pod-verified). Round 6: `InFlightGuard` cancellation propagation
(pod-verified). No further gaps identified.

Deliberately out of scope: the single-process/local relay driver
(`infer-server/src/lib.rs`'s "local-relay-driver" loop, used for single-GPU
serve) does not handle `CancelRequest` — it falls into that loop's existing
`Ok(Some(other)) => log::debug!(...)` catch-all, a silent no-op. This
investigation was scoped to the multiproc (TP>1) hang that was actually
reported; single-GPU serving never had this gap surfaced as a real issue, and
wiring it would need its own exploration of `execution.rs`'s engine-thread
command channel (a different structure than the multiproc `RequestHandle`
tracking this fix uses). Own follow-up if it's ever needed.
