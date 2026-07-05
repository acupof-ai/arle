# Multiproc lockstep coordinator hangs forever when a worker silently stops acking — no timeout, no error

## Context

Found while pod-verifying the DSv4 KV three-layer refactor
([plan](../../plans/2026-07-04-dsv4-dsa-kv-three-layer.md),
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
  the actual repro, every time.

## Status — paused here, not closed

The relay layer (`coordinator.rs`, `multiproc_relay.rs`) is done: both fixes
landed, unit-tested, and this retest independently confirms it's no longer
the stuck component. **The user-facing hang is NOT fixed** — it now lives in
`infer-core`, a bigger/different area than this doc's original scope
(multiproc relay). Paused for a scope check-in rather than continuing
unilaterally into scheduler/engine-cancellation changes. Next-session
starting points, decomposed to the implementation level so a fresh session
doesn't have to re-derive them:

1. **Is `all_reduce_min_scalar_i32` actually stuck, or just slow?** Add
   per-tick timing/instrumentation around `try_admit_front_waiter` /
   `cached_prefix_match_len` (`crates/infer-core/src/lib.rs:1128`,
   `crates/infer-core/src/prefix.rs:19`) for the specific 8106-token shape at
   TP=4/EP=4, and let it run far longer than the ~10-16 min already observed
   before concluding either way — a genuinely stuck NCCL collective and a
   legitimately-very-slow one look identical from `/v1/stats` alone.
2. **`InFlightGuard::drop` (`coordinator.rs`) needs to propagate cancellation
   into the engine**, not just decrement `in_flight` and unregister the sink.
   Whatever `try_admit_front_waiter`/the per-rank engine queue is currently
   waiting on for an abandoned request needs an explicit cancel path — this
   is a real gap independent of (1)'s answer, since even a "just slow"
   admission still permanently head-of-line-blocks every later request once
   its owning HTTP client has given up.
3. Both are `infer-core`/scheduler-level changes (bigger blast radius than
   the relay), not `infer-server` — re-scope before touching code, per the
   project's own >3-files/architectural-decision rule.
