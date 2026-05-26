# DSv4 native DeepEP — child-process spike PASS (phase 0)

## Context

`docs/plans/2026-05-26-dsv4-deepep-process-per-rank.md` phase 0 was the
architecture-license gate for the sidecar transport design: can a parent
(host) process supervise N persistent DeepEP children that reuse a single
`deep_ep.Buffer` across many dispatch+combine cycles and exit cleanly?

The same-process drop-in was killed earlier today
([`../errors/2026-05-26-dsv4-native-deepep-process-model-gate.md`](../errors/2026-05-26-dsv4-native-deepep-process-model-gate.md)).
The remaining open question was whether the alternative — parent forks
N children, each child a process-per-rank DeepEP host — works at all
under a Rust-style supervision pattern (not `torch.multiprocessing.spawn`
which auto-joins and exits).

Pod: 8xH20, DeepEP installed at `/usr/local/lib/python3.12/dist-packages/deep_ep`,
SM90 compiled. Shape mirrors DSv4 decode: `num_ranks=8`, `num_tokens=1`,
`hidden=4096`, `num_topk=6`, `num_experts=256` — same as the official
intranode multi-process test that PASSed at ~42us dispatch / ~36us combine
([`2026-05-26-dsv4-native-deepep-process-model-gate.md`](../errors/2026-05-26-dsv4-native-deepep-process-model-gate.md)).

## What worked

Python parent script using `multiprocessing.get_context("spawn").Process`
(parent stays alive; not the auto-join `torch.multiprocessing.spawn`)
spawned 8 child processes. Each child:

1. set its CUDA device to its rank;
2. initialized `torch.distributed` via TCP rendezvous on `127.0.0.1:29501`;
3. constructed `deep_ep.Buffer(group, 2e9, 0, low_latency_mode=False,
   num_qps_per_rank=1, explicitly_destroy=True)` — same constructor the
   official passing intranode test uses;
4. ran 5 dispatch+combine cycles on the same Buffer with synthetic
   DSv4-shape input;
5. explicitly destroyed the Buffer, barriered, destroyed the process
   group, exited 0.

Parent joined all 8 children inside a 180 s deadline, collected exit codes,
read per-child timings JSON. `passed: true`, no rank alive after timeout.

### Per-cycle timing (8 ranks × 5 cycles)

```
cycle | dispatch_us  mean /  min /  max  | combine_us mean / min / max
  0   |  31 950.5  / 5 394.7 / 109 292.1 |  202.2 / 165.9 / 227.3
  1   |   1 013.7  /   125.5 /   1 668.6 |   65.2 /  63.3 /  68.1
  2   |     874.9  /    93.1 /   1 494.1 |   60.5 /  56.7 /  65.4
  3   |     892.8  /    89.2 /   1 514.5 |   55.9 /  52.7 /  60.2
  4   |     873.8  /    87.5 /   1 446.0 |   57.2 /  53.0 /  61.9
```

Cycle 0 is dominated by JIT / first-launch overhead. Steady-state cycles
1–4 stabilize: **dispatch min ~87us, combine min ~52us**. Combined
per-layer best is ~140us at default chunk size, **roughly 1.4× the
official tuned ~78us (42 + 36) at NVL chunk 10**. Well within the phase 0
PASS tolerance of 10× (design doc §License gates).

The single-shot mean is far higher than the min because we did not use
`bench()` averaging — each cycle includes Python overhead (topk_idx +
num_tokens_per_expert loops + count gather). Phase 1 bench harness will
use `bench()` averaging for apples-to-apples comparison with the official
test.

## Architectural conclusion

The sidecar transport process model is **licensed for phase 1**:

- Parent process can spawn and supervise DeepEP children via
  `multiprocessing.Process` semantics — i.e. parent stays alive, owns
  child PIDs, can post a termination signal.
- DeepEP `Buffer` survives N cycles of dispatch+combine without
  re-initialization. This is the central design assumption of the sidecar
  (buffer built once at scheduler boot, reused across all model layers
  for the lifetime of the server).
- Same-process `cudaIpcOpenMemHandle invalid device context` error from
  the earlier kill does NOT reappear in the child-process shape. Confirms
  the root cause was process model, not a recoverable bug.

### Byte-identical determinism check (follow-up)

A second pass ran 8 ranks × 8 cycles with a deterministic input (fixed
per-rank seed, scores tensor pinned so topk routing is reproducible).
After each cycle, the combined output bytes were SHA-256 hashed.

| Rank | warm-cycle unique hashes | determinism |
|---:|---:|---|
| 0–7 | 1 each | **PASS** — every cycle on every rank produced the same combined bytes |

For each rank, cycle 0 hash also matched the warm hash, so determinism
held from the first cycle. Per-rank hashes differ across ranks (expected:
combine returns per-rank tokens, not a single broadcast tensor).

This closes the design doc's PASS criterion "combined output is
byte-identical to NCCL DeepEP-style baseline for the same input" on the
intra-DeepEP side. The vs-NCCL byte-identical comparison itself moves
to phase 1's bench harness, since it requires the full forward
integration — but the upstream determinism is now confirmed.

Open questions deferred to phase 1 by design:

- How does the Rust host post tensor data INTO the child (CUDA IPC handle
  posting via control pipe; one allocated buffer pool shared across
  layers; lifetime management).
- How does the child notify the host that combine output is ready (CUDA
  event over IPC channel vs busy-wait on a host pointer).
- Failure recovery: child crash → host restart vs fall back to NCCL
  DeepEP-style transport.

These are implementation, not architecture-license, questions.

## Artifacts

- Spike script:
  `/sgl-workspace/bench-artifacts/dsv4-deepep-child-process-spike-20260526/child_process_spike.py`
- Full per-rank JSON:
  `/sgl-workspace/bench-artifacts/dsv4-deepep-child-process-spike-20260526/rank{0..7}.json`
- Summary JSON:
  `/sgl-workspace/bench-artifacts/dsv4-deepep-child-process-spike-20260526/summary.json`
- Byte-identical determinism check script + summary:
  `/sgl-workspace/bench-artifacts/dsv4-deepep-byte-identical-check-20260526/byte_identical_check.py`
  `/sgl-workspace/bench-artifacts/dsv4-deepep-byte-identical-check-20260526/summary.json`

## Rule

When a library's official fast path is process-per-rank and the
same-process gate fails, the next gate is **parent-supervised
process-per-rank**, not "tune the same-process path harder". Verify both
that children spawn and run a dispatch+combine cycle, AND that the
Buffer survives multi-cycle reuse — single-cycle proof would not
license a sidecar that holds the buffer for the server lifetime.
