---
title: DSv4 native DeepEP — process-per-rank transport design
date: 2026-05-26
type: design plan
status: draft — phase 0 spike licensed for implementation
owner: ckl
related:
  - docs/experience/errors/2026-05-26-dsv4-native-deepep-process-model-gate.md
  - docs/experience/errors/2026-05-26-dsv4-native-deepep-ll-sameprocess-timeout.md
  - docs/experience/wins/2026-05-26-dsv4-default-deepep-deepgemm.md
  - docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md
---

# DSv4 native DeepEP — process-per-rank transport design

## Why this doc

`docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md` A1 is the highest
expected-leverage DSv4 axis (reduce-scatter ~20 ms / rank-range). Codex's
2026-05-26 gate run killed the **same-process drop-in** for native DeepEP:

| Gate (8xH20 pod) | Result |
|---|---|
| Official DeepEP LL multi-process DSv4 shape | PASS, dispatch+combine ~48.7 us/rank |
| Official DeepEP intranode multi-process DSv4 decode shape | PASS, dispatch 42.05 us / combine 36.34 us |
| ARLE same-process 8-thread DeepEP LL init/clean | FAIL — 180 s timeout |
| ARLE same-process 8-thread DeepEP intranode init | FAIL — `cudaIpcOpenMemHandle invalid device context` |
| Same retry with per-thread CUDA context reset before sync | FAIL — identical error |

Evidence:
[`../experience/errors/2026-05-26-dsv4-native-deepep-process-model-gate.md`](../experience/errors/2026-05-26-dsv4-native-deepep-process-model-gate.md).

The same-process gate is not a recoverable bug — DeepEP's NVSHMEM /
`cudaIpcOpenMemHandle` lifecycle assumes one process per rank, and ARLE's
current scheduler is one process with N worker threads bound to N CUDA devices.

**This doc decides which process model to land before the next attempt.** No
further same-process retries until one of the options below has a written
license-or-kill gate that passes.

## Non-goals

- Not deciding A1 Mega-MoE kernel fusion (that's a separate axis once native
  DeepEP transport lands).
- Not redesigning the HTTP / tokenizer / weight-loading surface. Those stay
  in the host scheduler process.
- Not introducing `torch.distributed` or a Python broadcaster.
- Not changing the public `ARLE_DSV4_MOE_BACKEND=deepep` default — that
  continues to mean "DeepEP-style NCCL fallback" until phase 2 lands.

## Current shape (the thing we have to change)

```
                          one process (Rust)
                          per ARLE serve invocation
        HTTP server   ─┐
        scheduler ─────┤   N worker threads (N = world_size)
        weight cache ──┤        ├─ thread 0 → CUDA device 0  ─┐
        tokenizer     ─┘        ├─ thread 1 → CUDA device 1  ─┤  one
                                ├─ ...                        ├─ NCCL
                                └─ thread N-1 → device N-1   ─┘  group
                                       │
                                       └─ LayerCommunicator (NCCL handle wrapper)
                                       └─ DeepEP-style fallback via NCCL
                                          (forward_deepep_routed_gpu in mlp.rs)
```

Key code surfaces:

- `infer/src/scheduler/cuda/core.rs` — scheduler worker boot, one per device.
- `infer/src/backend/cuda/layer_communicator.rs` — NCCL handle, lives in host
  worker thread.
- `infer/src/model/deepseek/mlp.rs:3088` — `forward_deepep_routed_gpu` is the
  DeepEP-style dispatch/combine entry that we want to back with native DeepEP.

The mismatch is structural, not a bug: DeepEP wants one CUDA process per
rank; ARLE wants one process per node.

## Decision space

Three implementation options. License-or-kill per phase listed in §License gates.

### Option A — Sidecar EP child process (recommended for phase 1)

```
        host process (one per node)         per-rank child processes
        ┌──────────────────────────┐        ┌─────────────────────┐
        │ scheduler + N workers    │        │ rank-0 EP sidecar   │
        │ model forward, attn, GEMM│  CUDA  │  DeepEP buffer      │
        │ DeepEP-style fallback ───┼─ IPC ──┼─ dispatch / combine │
        │ (current)                │  + ctl │  NVSHMEM lifecycle  │
        └──────────────────────────┘  pipe  └─────────────────────┘
                                              ... 7 more children on 8xH20
```

- Host fork()s one child per rank at scheduler boot; child binds the
  matching CUDA device and constructs the DeepEP buffer in the child's CUDA
  context — satisfying the process-per-rank lifecycle.
- Host worker prepares dispatch input (`tokens, topk_ids, topk_weights`) in
  device memory; passes a CUDA IPC handle + control message to child.
- Child runs DeepEP dispatch → expert GEMM → DeepEP combine using device
  memory aliased via CUDA IPC; writes combined output back to a host-owned
  buffer over the same IPC channel.
- Host worker continues with downstream ops (residual add, layernorm,
  attention of the next layer) once the child posts a "combine done"
  CUDA event over IPC.

**Pros**
- Smallest scope: scheduler / HTTP / tokenizer / weight loading stay in host.
- Only `LayerCommunicator` grows a `NativeDeepEPSidecar` transport variant
  next to the existing NCCL transport. `forward_deepep_routed_gpu` calls
  the same shape it already calls.
- Sidecar can be left dormant until `ARLE_DSV4_MOE_BACKEND=native-deepep`
  is set, so it does not regress the NCCL-default path.
- Child crash is recoverable: host monitors child PID, restarts on exit,
  falls back to NCCL DeepEP-style if restart fails.

**Cons**
- IPC overhead lives on the hot path. Per layer the host must (a) signal
  dispatch, (b) wait for combine. If the round-trip is > 20 us per layer
  the wins shrink fast.
- 9 processes per node instead of 1 — slightly more visible in monitoring,
  shared host memory mapping needs hardening.
- CUDA IPC handle stability across child lifetime — must verify that the
  same allocated tensor pool can be aliased for the full server lifetime.

**Risks to verify before phase 2**
- IPC round-trip < 5 us per dispatch in the phase 1 bench. If it isn't,
  re-evaluate Option B before committing.
- Child fault recovery doesn't drop in-flight requests.
- DeepEP buffer lifetime across model warm-up / shape change.

### Option B — Full multi-process scheduler (one process per rank)

```
        rank-0 process               rank-1 process       ... rank-7
        ┌────────────────┐           ┌────────────────┐
        │ scheduler      │  side     │ scheduler      │
        │ model fwd      │  channel  │ model fwd      │
        │ HTTP server    │←─────────→│ HTTP shadow    │
        │ tokenizer      │  broker   │ tokenizer      │
        │ weight loader  │           │ weight loader  │
        │ DeepEP buffer  │  NVSHMEM  │ DeepEP buffer  │
        └────────────────┘←─────────→└────────────────┘
```

- Full SGLang / vLLM-style topology. One rank is the request entry point;
  the rest run a `forward-only` mode that receives broadcast batches.
- DeepEP works out of the box because each rank is its own process.

**Pros**
- Matches the upstream DeepEP / NVSHMEM design exactly. No IPC bridge.
- Aligns ARLE with the industry baseline that already proved this works.

**Cons**
- Large blast radius: scheduler, HTTP, weight loader, KV pool, RadixCache
  all need a "this is rank R of W" mode. Easily a multi-week refactor.
- Forces ARLE to ship `mpirun` / process orchestrator alongside the
  binary, plus a way to broadcast HTTP request batches rank-0 → others.
- Breaks the "single process, single binary" deployment story that
  `arle serve` currently has.
- Has to keep working alongside single-rank `metal_serve` on Mac, single-GPU
  CUDA, and the `no-cuda` build feature.

**Risks to verify before adopting**
- Cost of broadcasting batch metadata / token ids each step vs benefit.
- Whether `crates/cuda-kernels` and `crates/mlx-sys` initialization still
  composes safely under multi-process.

### Option C — In-process NVSHMEM + custom EP runtime (not DeepEP)

Skip DeepEP, build our own EP runtime that does work inside one process
using NVSHMEM symmetric memory primitives directly.

**Pros**
- Keeps the current process model.

**Cons**
- We give up DeepEP's tuned dispatch/combine kernels and have to write
  equivalent kernels ourselves. This is the work A1 Mega-MoE was supposed
  to follow, not precede.
- NVSHMEM in-process across multiple CUDA contexts is itself unproven —
  the same lifecycle assumption that broke same-process DeepEP applies.
- We trade a known process problem for an unknown kernel problem with the
  same root constraint.

**Verdict**: defer until Option A and Option B are both ruled out.

### Recommendation

**Phase 1 lands Option A (sidecar).** It is the smallest reversible change
that produces real wall-clock evidence. If Option A's IPC round-trip
exceeds the budget, we have data in hand to license Option B without
guessing.

## License gates

Each phase must finish before the next starts. **All gates use wall-clock
framing per CLAUDE.md §0** — narrow-window NVTX percentages do not pass.

### Phase 0 — process-per-rank spike, evidence only

**Question**: Can ARLE's Rust runtime fork a child process, bind a CUDA
device in the child, initialize a DeepEP buffer in the child, and run a
single dispatch+combine pair driven by data from the host process?

**Implementation**:
- `crates/cuda-kernels/csrc/comm/deepep_sidecar/`: minimal C++ binary
  that links DeepEP / NVSHMEM, accepts a control pipe, opens a CUDA IPC
  handle posted by host, runs `Buffer.dispatch(...) / Buffer.combine(...)`
  on a fixed-shape input, posts a "done" event.
- `infer/src/backend/cuda/deepep_sidecar.rs`: host-side spawn helper,
  posts handles + control messages, awaits done event.
- Test surface: `cargo test --features cuda --test deepep_sidecar_spike`
  on the remote 8xH20 pod. Test runs N=8 children for one layer-shape
  dispatch+combine, verifies output matches the reference NCCL DeepEP-style
  result byte-for-byte (greedy).

**PASS**:
- All 8 children initialize and complete one dispatch+combine.
- Combined output is byte-identical to NCCL DeepEP-style baseline for
  the same input.
- Child PID survives 10 consecutive dispatch+combine pairs without crash.

**KILL**:
- Same-process equivalent error reappears in the child (suggests the
  problem is not actually about process model — re-examine root cause).
- Child can't open a host-side CUDA IPC handle (suggests our IPC plan
  is structurally wrong, not just unimplemented).
- Output does not match NCCL baseline (cannot ship as transparent
  fallback; would require user-visible behavior change).

**Wall-clock framing note**: phase 0 is connectivity, not perf. No
TTFT / TPOT thresholds. The only timing recorded is dispatch+combine
total per layer for sanity vs the official 48.7 us/rank multi-process
PASS — if our sidecar is 10× slower we still pass phase 0 (it's the
gate) but phase 1 has to recover the gap.

**Phase 0 outcome (2026-05-26)**: PASS. 8xH20 pod, 8 children × 5
dispatch+combine cycles, all exit 0; steady-state per-layer min
~87us dispatch / ~52us combine vs official tuned ~42us / ~36us
(within phase 0 tolerance). Same-process `cudaIpcOpenMemHandle`
error does not reappear under child-process shape. Evidence:
[`../experience/wins/2026-05-26-dsv4-deepep-child-process-spike.md`](../experience/wins/2026-05-26-dsv4-deepep-child-process-spike.md).
Phase 1 (LayerCommunicator `NativeDeepEPTransport` variant +
SLO wall-clock bench) is licensed.

### Phase 1 — sidecar transport landed behind a flag

**Question**: With `ARLE_DSV4_MOE_BACKEND=native-deepep`, does ARLE produce
wall-clock wins over the current `deepep` NCCL fallback on the canonical
DSv4 SLO workload (32K input / 1.5K output, c=8, qps=8, H20)?

**Implementation**:
- `LayerCommunicator` grows a `NativeDeepEPTransport` variant alongside
  the NCCL transport.
- `forward_deepep_routed_gpu` calls the same `dispatch / combine` shape;
  the variant routes to either NCCL (default) or sidecar (opt-in).
- Sidecar pool reused across all layers — buffer constructed once at
  scheduler boot, not per layer.
- Bench harness: `scripts/bench_guidellm.sh dsv4-native-deepep-phase1`
  vs `dsv4-deepep-baseline-phase1`, same env, same commit, only the
  `ARLE_DSV4_MOE_BACKEND` flag changes (single-variable rule).

**PASS** (all four must hold):
- p50 TTFT delta ≥ +5% improvement at SLO workload (≥ 240 ms off 4800 ms
  target) vs NCCL DeepEP-style fallback.
- p50 TPOT delta ≥ +5% improvement (≥ 0.9 ms off 18 ms target).
- Tail p99 TTFT does not regress > 3%.
- Greedy output byte-identical to NCCL fallback for a 32-prompt fixed
  set, `max_tokens >= 32`.

**KILL**:
- Wall-clock TTFT or TPOT regresses at SLO workload.
- IPC round-trip per layer > 5 us measured by NVTX inside the sidecar.
- Tail p99 regresses > 3% (latency bimodality from child scheduling).
- Output diverges from baseline (sampling-level reproducibility break).

If KILL: file an errors entry, re-evaluate Option B with the round-trip
number from this bench in hand. Do not silently fall back to "small
launch axes" — `docs/projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md`
explicitly orders against that.

### Phase 2 — flip default

Only after phase 1 PASS and ≥ 1 week of sidecar uptime on the H20 pod
under bench load:

- Flip `ARLE_DSV4_MOE_BACKEND` default from `deepep` (NCCL fallback) to
  `native-deepep` (sidecar).
- Keep NCCL fallback selectable via `ARLE_DSV4_MOE_BACKEND=nccl-deepep`
  for one release window.
- Run the full DSv4 SLO bench at default to confirm phase 1 numbers
  are not session-dependent.

## Implementation order

| # | Task | Owner | Estimate |
|---|---|---|---|
| 0.1 | sidecar binary skeleton (links DeepEP, opens IPC pipe, runs one dispatch+combine on fixed shape) | claude direct | small |
| 0.2 | host-side spawn helper (`deepep_sidecar.rs`) + test that posts CUDA IPC handles | claude direct | small |
| 0.3 | spike test on remote 8xH20: 8 children, 1 layer, byte-identical vs NCCL baseline | tn exec on pod | small |
| 0.4 | phase 0 wins or errors entry; license phase 1 | claude direct | trivial |
| 1.1 | `LayerCommunicator` `NativeDeepEPTransport` variant | general-purpose subagent | medium |
| 1.2 | `forward_deepep_routed_gpu` route to sidecar when flag set | general-purpose subagent | medium |
| 1.3 | sidecar pool: one buffer constructed at boot, all layers reuse | general-purpose subagent | medium |
| 1.4 | bench A/B at SLO workload, fill in §License gates phase 1 | tn exec + claude direct | medium |
| 1.5 | wins or errors entry, license phase 2 (or kill) | claude direct | trivial |

Phase 2 not estimated until phase 1 lands.

## Cross-refs

- backlog entry that triggered this work:
  [`../projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md`](../projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md) §A1
- same-process LL timeout:
  [`../experience/errors/2026-05-26-dsv4-native-deepep-ll-sameprocess-timeout.md`](../experience/errors/2026-05-26-dsv4-native-deepep-ll-sameprocess-timeout.md)
- process-model gate roll-up:
  [`../experience/errors/2026-05-26-dsv4-native-deepep-process-model-gate.md`](../experience/errors/2026-05-26-dsv4-native-deepep-process-model-gate.md)
- DSv4 SLO baseline (32K/1.5K, c=8, qps=8, H20, target TTFT 4800 ms,
  target TPOT 18 ms):
  [`../projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md`](../projects/2026-05-25-dsv4-next-axis-from-sglang-v4.md) §Serving SLO Baseline
- binding constraints reference:
  [`../research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md`](../research/2026-05-15-dsv4-decode-memaccess-binding-constraints.md)
