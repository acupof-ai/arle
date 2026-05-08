# M_ttft-p99 — TTFT p99 tail-latency investigation

**Status:** P2 plan(per `5364612` codex pickup queue)— scaffolded by Claude
**Owner:** TBD(codex impl,Claude planning)
**Trigger:** `f5cf829` W4 c=8 admission-fix LICENSED noted
"TTFT p99 still very poor" — separate from liveness,deferred from
substrate fix landing
**Effort estimate:** 0.5d Claude reconnaissance + 1-2d codex impl

## §1 Empirical signal

Post-substrate-fix W4 c=8 bench(`f5cf829` 256/256 turns OK):
- TTFT p50:11768 ms
- TTFT p99:**72515 ms**(6.2× spread)
- engine_ttft_us(server-side):2000 ms — last token(steady-state TTFT post-warm)

`c4fae17` W4A16 c=4 8k bench:
- TTFT p50:5570 ms
- TTFT std:43.4 ms — c=4 is healthy(σ < 1% of mean)

`c4fae17` W4A8 c=4 8k bench:
- TTFT p50:4079 ms
- TTFT std:58.4 ms — c=4 is healthy(σ < 1.5% of mean)

**Pattern**:c=4 has tight TTFT distribution(σ < 2%);c=8 burst loads
develop heavy p99 tail。

## §2 Hypothesis space

### H1:Sequential prefill chunk admission(strongest)

ARLE chunked prefill processes 1 prefill candidate's chunk per step
when no decode is active。Burst-of-8 sessions at t=0:
- t=0:admit 8 sessions to active,start prefill_queue
- step 1:session 0's first 4K chunk(out of 8K)
- step 2:session 0's second 4K chunk OR session 1's first chunk?

If **HOL(head-of-line)scheduling**:session 0 finishes all chunks
before session 1 starts。Then session 7 waits 7 × per-session prefill
time before its first chunk fires。

Empirical TTFT p99 72.5s ÷ p50 11.7s = 6.2× spread,close to 7-8 sessions
sequential pattern。

**Verify**:check `infer/src/scheduler/cuda/prefill.rs` for prefill
candidate selection logic — round-robin vs HOL?

### H2:KV pool admission rejection cascade

If KV slot reservation logic rejects late-arriving sessions until
earlier sessions complete some prefill,backlogged session may hit
multiple admission failures with retry-backoff(per harness retry-backoff
`e7b4765`)before getting into queue。

**Verify**:check `/v1/stats` `engine_queue_depth` over time during burst — does it spike to 8 then drain monotonically?

### H3:Prefill chunk-size auto-clamp at high concurrency

`max_num_batched_tokens` envelope at high conc may force chunk size to
shrink → more steps per session → linearly more wait for queued sessions。

**Verify**:check chunk-size in `/v1/stats step_phase_us` traces — fixed
or dynamic with admission pressure?

### H4:CUDA Graph warmup deferred for non-canonical batch sizes

ARLE pre-captures batch sizes 1-4(per `--num-slots` config)。At c=8
bursts hitting batch=5-8,first encounter triggers eager kernel launch
instead of replayed graph → ITL/TTFT spike for first session in
each new batch size。

**Verify**:check graph capture log lines `Capturing CUDA Graph for
batched decode B=N` during initial warmup → does C=5-8 capture happen
during bench or only post-admission?

### H5:Single-stream prefill blocking decode

If prefill and decode share the same CUDA stream,decode(c=8 sessions
that already finished prefill)blocks waiting for prefill chunk to
complete。Latest session's prefill blocks earlier sessions' decode →
artificially inflates the LATER session's TTFT(measured from request
to first decode token,which now includes wait for stream sync).

**Verify**:CUDA stream architecture in `infer/src/scheduler/cuda/` —
single stream or per-phase?

## §3 Phase 0 — Reconnaissance(Claude,0.5d)

- [ ] Read `infer/src/scheduler/cuda/prefill.rs` candidate selection logic
- [ ] Read `infer/src/scheduler/cuda/runtime/scheduler_loop.rs` step orchestration
- [ ] Trace `/v1/stats step_phase_us` over time during burst — capture
      empirical chunk size / dispatch pattern
- [ ] Identify which hypothesis(H1-H5)matches empirical signal
- [ ] Produce wins/errors entry per skill v1.4.0 methodology

## §4 Phase 1 — Implementation candidates(codex,1-2d)

Depending on Phase 0 findings:

### Fix-A(if H1):Continuous prefill admission(round-robin)

Replace HOL with round-robin chunk processing across queued sessions:
- Each step processes 1 chunk per active session(if budget allows)
- 8 sessions get first chunk done at step 1-2 instead of session 0 alone
- TTFT p99 reduces from 8× to 1-2× per-chunk-time

Effort:~150 LOC scheduler refactor。Risk:Medium(scheduler hot path)。

### Fix-B(if H3):Larger chunk envelope at high conc

Increase `max_num_batched_tokens` from 16K → 64K to handle 8 × 8K =
64K total prefill。Verifies with W4 c=8 production-shape bench。

Effort:~10 LOC config change。Risk:Low(envelope bump)。But may OOM
GPU at very high conc。

### Fix-C(if H4):Pre-capture all batch sizes 1-N at startup

Currently `--num-slots K` captures only batch sizes 1..K。Production
bursts may exceed K transiently。Pre-capture larger range up to
admission-cap。

Effort:~30 LOC warmup loop。Risk:Low(startup time only)。

### Fix-D(if H5):Per-phase CUDA stream

Separate prefill stream from decode stream so they run concurrently
on GPU。

Effort:~200 LOC stream coordination。Risk:High(complex sync)。

## §5 KILL criteria

- **Phase 0**:if hypothesis cannot be empirically validated within
  0.5d → write up "no clear root cause" research entry,defer
- **Phase 1**:if any fix produces ITL regression > 5% as side effect
  → revert,re-evaluate
- **Phase 1**:if Fix-A produces TTFT p99 reduction < 30% → not worth
  scheduler complexity,revert
- **Phase 1**:if Fix-B causes OOM at production c=16 → revert and
  pursue Fix-A or Fix-C path

## §6 ROI estimate

- Current:c=8 W4 admission TTFT p99 72.5s — most workload owners would
  reject "10s+ tail"
- Target:TTFT p99 ≤ 3× p50(20-30s for c=8 W4)— acceptable production tail
- ROI:**unblocks production deployment at c≥8**
- Side benefit:reduced p99 likely improves p50 marginally too(less
  serialization within step)

## §7 Tradeoffs

| Axis | Status | Note |
|---|---|---|
| LOC complexity | varies by Fix | A=150 / B=10 / C=30 / D=200 |
| Hardware specificity | likely none | scheduler-only fix |
| Numerical correctness | ✅ no kernel changes | scheduler ordering only |
| Memory budget | varies | Fix-B raises envelope,may interact with KV |
| **Multi-shape** | needs verification | TTFT p99 measured at one shape only |

## §8 Phase 8 license-or-kill

| Outcome | Action |
|---|---|
| TTFT p99 ≤ 30s for W4 c=8 production-shape | LICENSE,land wins entry |
| Any ITL regression > 5% | KILL,revert fix |
| Fix complexity > 200 LOC and gain < 30% | KILL,not worth substrate burden |

## Cross-references

- Trigger:`f5cf829`(TTFT p99 72515 ms noted in W4 c=8 wins entry)
- Substrate baseline:`b708e00`(W3+W4 admission deadlock fix)
- 3-shape grid wins:`c4fae17`(c=4 healthy,c=8 needs investigation)
- Codex pickup queue:`5364612`
- Skill v1.4.0:`6c627c4`
- Scheduler:`infer/src/scheduler/cuda/`(prefill.rs,scheduler_loop.rs,execution.rs)

## Phase 0 deliverable

Claude can EXECUTE Phase 0 in next tick(0.5d)— produce a
`docs/research/2026-05-08-ttft-p99-rootcause-investigation.md` entry
identifying which H1-H5 hypothesis matches empirical signal。Then
codex picks the matching Fix path。

## Rule

**TTFT p99 tail latency at burst load is its own optimization axis**,
separate from substrate liveness and from per-session decode performance。
Empirical signal:if c=4 is tight σ<2% but c=8 is >5×p99/p50 spread,
the bug is in admission/scheduling under burst,not in kernel speed。

For ARLE specifically:burst-of-8 8K-prompt is the binding shape per
master §2.1。c=8 TTFT p99 is the production blocker for 8-conc deployment。
