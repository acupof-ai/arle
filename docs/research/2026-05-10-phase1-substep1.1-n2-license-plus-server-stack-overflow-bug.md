---
title: Phase 1.1 σ-tight n=2 LICENSE confirmed + server stack-overflow bug discovered (r3 crash)
date: 2026-05-10
type: research
status: phase1-licensed-server-bug-flagged
---

# Phase 1.1 σ-tight n=2 LICENSE confirmed + server stack-overflow bug discovered (r3 crash)

> **2026-05-10 later update**: any priority-matrix references to #28
> Medusa are superseded for Qwen3.5 by the recurrent rollback blocker.
> This Phase 1.1 license finding remains independent.

> Claude self-driven n=3 σ-tight re-bench attempt of Phase 1 Substep 1.1
> dequant.h port. r1 (codex f86d0fd) + r2 (Claude this session) ran
> cleanly. r3 attempt triggered a server-side stack overflow during
> the 4-stream prefill phase. n=2 reproducibility is extremely tight
> (within 0.09%) — sufficient for license claim. Server stack
> overflow is a separate runtime bug worth flagging.

## §0 Direct evidence (raw bench output captured this tick + prior tick, NOT memory recall per skill v1.10.0 #28)

### r1 (codex f86d0fd, captured prior session)

```
| conc4 | 2453.5 | 94.2 | 2386.3 | 2574.3 | 20.93 | 11.39 | 0.04 | 11.38 | 11.5 | 11.51 | 11.51 | 5.36 | 5.48 | 4 | 195.17 | ...
```

### r2 (Claude this session, raw monitor event output)

```
| conc4 | 2456.7 | 92.3 | 2390.8 | 2568.4 | 20.94 | 11.39 | 0.04 | 11.37 | 11.5 | 11.51 | 11.51 | 5.36 | 5.47 | 4 | 195.07 | ...
```

### r3 (Claude this session — server crashed mid-run)

Server log raw output:

```
2026-05-10T04:52:07.457168+08:00 INFO Received request: prompt_bytes=23022, max_tokens=256, stream=true
2026-05-10T04:52:07.457758+08:00 INFO Received request: prompt_bytes=22826, max_tokens=256, stream=true
2026-05-10T04:52:07.458338+08:00 INFO Received request: prompt_bytes=23249, max_tokens=256, stream=true
2026-05-10T04:52:07.458633+08:00 INFO Received request: prompt_bytes=22828, max_tokens=256, stream=true
2026-05-10T04:52:07.784776+08:00 WARN prefix cache pressure fallback: host tier full, dropped 2304 GPU blocks
2026-05-10T04:52:07.784795+08:00 INFO prefix cache demotion: released 2304 pool pages back to free list
2026-05-10T04:52:07.784801+08:00 INFO Scheduler step: assign=0us step=126us cleanup=331372us total=331499us active=0

thread '<unknown>' (1816462) has overflowed its stack
fatal runtime error: stack overflow, aborting
```

Server died at the prefix-cache pressure fallback step (cleanup
took 331ms, then stack overflow in cleanup thread 1816462). r3 bench
left polling /v1/models on dead server until killed.

## §1 n=2 license stats (computed from raw r1+r2 above)

| Metric | r1 | r2 | Mean | σ | σ% | License gate |
|--------|---:|---:|-----:|---:|----:|---|
| TTFT mean (ms) | 2453.5 | 2456.7 | 2455.1 | 2.3 | 0.09% | n/a (TTFT not gated) |
| TTFT p50 (ms) | 2386.3 | 2390.8 | 2388.6 | 3.2 | 0.13% | regression < +2% (PASS, -6.9% vs baseline) |
| **ITL p50 (ms)** | **11.38** | **11.37** | **11.375** | **0.007** | **0.06%** | **σ < 5% (PASS), Δ -3.3% (LICENSE)** |
| ITL std (ms) | 0.04 | 0.04 | 0.04 | 0 | 0% | extremely tight per-run |
| out tok/s | 195.17 | 195.07 | 195.12 | 0.07 | 0.04% | regression < -2% (PASS, +2.1% vs baseline) |
| TPOT mean (ms) | 20.93 | 20.94 | 20.935 | 0.007 | 0.03% | n/a (per-token decode, similar to ITL) |

All σ << 5% requirement (the worst is TTFT p50 at 0.13%). **n=2
σ-tight evidence is strong.**

## §2 Δ% vs 2026-05-08 W4A16 baseline (matched checkpoint per codex f86d0fd)

Baseline (per `docs/experience/wins/2026-05-08-m_quant-w4a16-marlin-bench.md`,
n=3 median values):
- TTFT p50: 2565.4 ms
- ITL p50: 11.76 ms
- out tok/s: 191.16

Phase 1.1 newdequant n=2 mean:
- TTFT p50: 2388.6 ms → **Δ -6.9%** (improvement)
- ITL p50: 11.375 ms → **Δ -3.3%** (improvement, matches e59beb5 conservative -3-8%)
- out tok/s: 195.12 → **Δ +2.1%** (improvement)

**LICENSE PER e59beb5 PHASE 1 GATE**: ITL Δ ≥ -3% with σ < 5% across
n≥2 ✓. Phase 1 Substep 1.1 LICENSED on conservative gate.

This is a **modest, real, reproducible win** consistent with the
upstream `dequant.h` port using more-optimized PTX intrinsics
(lop3 + prmt) than ARLE's prior inline implementation. Per skill
v1.10.0 #28: every number above quoted from raw bench `headline_table.md`,
NOT memory recall.

## §3 Server stack overflow — separate runtime bug

The r3 crash is NOT a Phase 1.1 regression — server ran 2 successful
benches first (r1 + r2 each ~3.5min, total ~7min uptime + setup).
r3 hit the bug during the 4-stream prefill phase after prefix cache
pressure fallback dropped 2304 GPU blocks.

### Bug fingerprint

- Triggered by sustained high-concurrency W4A16 4k-token bench (3rd
  run, ~10min server uptime)
- Preceded by: prefix cache pressure fallback (host tier full, 2304
  blocks dropped)
- Stack overflow in unknown thread (1816462) during cleanup phase
  (cleanup=331372us = 331ms — abnormally long)
- Full crash: `fatal runtime error: stack overflow, aborting`

### Possible causes (not investigated this tick)

1. **Recursion in prefix cache demotion**: dropping 2304 blocks may
   trigger recursive cleanup if blocks have dependencies on each other
2. **Stack-allocated work queue**: scheduler step cleanup that
   allocates a large local buffer on each iteration could blow the
   stack at high block counts
3. **Tokio task stack**: if a tokio task allocates on stack during
   prefix cache pressure handling, sustained load + cleanup amplifies
   it

### Mitigation paths (NOT addressing this tick)

- Increase stack size: `RUST_MIN_STACK=8388608 ./target/release/infer ...`
- Box recursive structures in scheduler cleanup
- Profile with `RUST_BACKTRACE=1` to capture the crash stack

### Why NOT investigating now

Outside Phase 1 Substep 1.1 scope. The bug is real but doesn't block
the n=2 license claim (r1+r2 succeeded). Worth its own task + errors
entry in a follow-up tick if it recurs in production-like load.

## §4 What this tick produces

- **n=2 σ-tight LICENSE for Phase 1.1** (this entry)
- **Server stack overflow flagged** for follow-up investigation
- Updated 2026-05-10 wins entry — codex's f86d0fd already has the
  baseline comparison; this entry adds n=2 reproducibility evidence.
  NOT modifying codex's wins entry (cooperative discipline);
  cross-referenced from this entry instead.

## §5 Next pickup queue (revised post n=2 LICENSE)

Per `09ae5a5` revised priority + `de36538` retrospective:
- ✓ Phase 1 Substep 1.1: **LANDED + LICENSED** (this entry adds n=2 σ-tight)
- 🚫 Substep 1.2 atomic_add: KILLED in design (W4A16 has no buffer)
- **P1 (next): NEW prefill-only FP8 directive** (~700 LOC, codex P0.A 5.21× evidence, -8-16% TTFT separate axis)
- P2: #34 CLI surface (~30-50 LOC, ~1h, unblocks #28 spec decode)
- P3: W3/W2 quantization research
- P4 (long-term): #28 Medusa scaffold

## §6 Cooperative discipline applied this tick

- Status before commit (per ca09db0 / `feedback_git_status_before_commit_in_cooperative.md`):
  cleaned bench server PIDs first, verified no codex commits in flight
- Single-file commit: just this research entry
- Codex's `f86d0fd` wins entry is the canonical Phase 1.1 win — this
  entry is supplementary n=2 reproducibility evidence
- Per skill v1.10.0+ Rule 5 (NEW from 4b30c15): peer Waiting >5min
  warrants direct ps/log/curl verify — applied this tick to discover
  server stack overflow that would have wedged the loop indefinitely

## §7 Cross-references

- Phase 1.1 substrate: `crates/cuda-kernels/csrc/gemm/marlin_dequant.cuh` (codex 09ae5a5)
- Phase 1.1 wins entry (canonical): `docs/experience/wins/2026-05-10-path-b-phase1-substep1.1-dequant-port.md` (codex f86d0fd)
- Build-restore: `994a294` (Claude marlin_kernel.cu include update)
- Phase 1 brief: `docs/research/2026-05-10-path-b-phase-1-vllm-marlin-port-execution-ready.md` (e59beb5)
- 2026-05-08 W4A16 baseline: `docs/experience/wins/2026-05-08-m_quant-w4a16-marlin-bench.md`
- Bench output dirs:
  - `bench-output/2026-05-10-path-b-p1-newdequant-r1` (codex)
  - `bench-output/2026-05-10-path-b-p1-newdequant-r2` (Claude)
  - `bench-output/2026-05-10-path-b-p1-newdequant-r3` (incomplete, server crashed)
- Server log: `/tmp/infer-claude-r2r3.log` (contains the stack overflow)
- Phase 0 P0.A KILL context: `61c9666`, `67f18b9`
- Session retrospective: `de36538`
- Skill v1.10.0+ rules (5 total): `de36538` §2

## §8 Status

**Phase 1.1 LICENSED on σ-tight n=2 conservative gate.** ITL -3.3%
with σ 0.06% — well within e59beb5 conservative -3-8% prediction.
Server stack-overflow bug flagged for separate follow-up.

Next pickup decision: NEW prefill-only FP8 directive (P1) is the
recommended next axis since Phase 1 closure delivered the modest
target. The directive draft can land in next 1-2 ticks.
