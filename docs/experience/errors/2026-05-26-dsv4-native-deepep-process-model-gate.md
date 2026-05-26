# DSv4 native DeepEP process-model gate

## Context

After the DSv4 default path was moved to the DeepEP-style MoE route, the next
high-leverage question was whether native DeepEP should be integrated before
continuing launch-churn micro-optimizations.

The current ARLE server starts one process with one CUDA scheduler worker
thread per rank. DeepEP's official tests use one process per rank. This entry
adds a process-model gate for both native DeepEP low-latency and intranode
paths before touching the model hot path.

Model path and host identifiers are intentionally omitted.

## Root Cause

Native DeepEP cannot be treated as a drop-in replacement for the current
threaded `LayerCommunicator` path yet. The official multi-process DeepEP paths
pass on the 8xH20 pod, but same-process multi-thread gates do not.

Evidence:

- Official low-latency multi-process DSv4-shape gate passed: dispatch+combine
  about 48.7 us/rank.
- Same-process 8-thread low-latency gate timed out at 180 s before usable
  post-sync timing.
- Same-process 8-thread intranode gate failed at DeepEP `deep_ep.cpp:200`
  during `cudaIpcOpenMemHandle` with `invalid device context`; resetting the
  CUDA device immediately before `runtime.sync(...)` did not fix it.
- Official intranode multi-process DSv4 decode-shape gate passed. For
  `tokens=1`, `hidden=4096`, `topk=6`, `experts=256`, it reported best BF16
  dispatch around 42.05 us and best combine around 36.34 us.

This points at a process-model mismatch: DeepEP's native transport expects the
process-per-rank CUDA IPC / NVSHMEM lifecycle. ARLE's same-process
multi-worker shape is not licensed by evidence for native DeepEP.

## Fix

No runtime transport swap landed in this entry. The correct implementation
direction is:

1. Treat native DeepEP as the highest-priority DSv4 communication axis.
2. Do not force native DeepEP into the existing one-process worker shape.
3. Add a process-per-rank DeepEP transport design before replacing the current
   NCCL-backed DeepEP-style dispatch/combine path.
4. Keep `ARLE_DSV4_MOE_BACKEND=deepep` meaning the validated DeepEP-style
   fallback until native DeepEP passes request-level A/B.

## Results

| Gate | Result |
|---|---|
| Official DeepEP LL multi-process DSv4 shape | PASS, dispatch+combine about 48.7 us/rank |
| Same-process DeepEP LL 8-thread init/clean | FAIL, 180 s timeout |
| Same-process DeepEP intranode 8-thread init | FAIL, `cudaIpcOpenMemHandle` invalid device context |
| Same-process DeepEP intranode with per-thread context reset | FAIL, same error |
| Official DeepEP intranode multi-process DSv4 decode shape | PASS, best BF16 dispatch 42.05 us, best combine 36.34 us |

## Artifacts

- Same-process native DeepEP LL gate:
  `/sgl-workspace/bench-artifacts/dsv4-deepep-ll-sameprocess-gate-20260526`
- Official DeepEP low-latency DSv4-shape gate:
  `/sgl-workspace/bench-artifacts/dsv4-deepep-ll-standalone-t1-h4096-topk8-e256-20260526/test_low_latency.log`
- Same-process DeepEP intranode gate:
  `/sgl-workspace/bench-artifacts/dsv4-deepep-intranode-sameprocess-gate-20260526`
- Same-process DeepEP intranode context-reset gate:
  `/sgl-workspace/bench-artifacts/dsv4-deepep-intranode-sameprocess-gate2-20260526`
- Official DeepEP intranode DSv4-shape gate:
  `/sgl-workspace/bench-artifacts/dsv4-deepep-intranode-standalone-t1-h4096-topk6-e256-20260526/test_intranode.log`

## Rule

When a library's official fast path is process-per-rank, first verify ARLE's
actual process model before integrating it into the hot path. A failed
same-process gate is a design blocker, not a reason to continue lower-leverage
launch micro-optimizations.
