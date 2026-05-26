# DSv4 native DeepEP LL same-process gate timeout

## Context

DSv4 already defaults to `ARLE_DSV4_MOE_BACKEND=deepep` for the validated
DeepEP-style dispatch/combine path. The next proposed step was to make that a
native DeepEP low-latency `internode_ll` transport instead of the current
NCCL-backed transport.

The strategic gate was intentionally smaller than a Rust integration: prove
that DeepEP low-latency runtime init works in ARLE's one-process, multi-worker
shape. The remote pod has 8 H20 GPUs, DeepEP imports successfully, SM90 support
is compiled, and the official multi-process DeepEP low-latency test passes for
the DSv4 decode shape.

Model path and host identifiers are intentionally omitted.

## Root Cause

The native DeepEP low-latency runtime is not yet licensed as the default ARLE
transport because the same-process gate timed out.

The gate created 8 Python threads in one process, bound one CUDA device per
thread, constructed `deep_ep_cpp.Buffer(rank, 8, 0, rdma_bytes, true, true)`,
exchanged local device IDs, CUDA IPC handles, and rank-0 NVSHMEM unique ID in
process memory, then called `runtime.sync(...)` and
`clean_low_latency_buffer(1, 4096, 256)`.

It printed the expected DSv4 low-latency shape and environment:

```json
{"cuda_devices": 8, "experts": 256, "gate": "same-process-deepep-ll-init", "hidden": 4096, "max_dispatch_tokens_per_rank": 1, "ranks": 8, "rdma_bytes": 8521856, "sm90_compiled": true}
```

No worker reached the post-sync timing output before the 180 second hard
timeout. This is a process-model blocker, not a performance result. The
upstream DeepEP low-latency test uses one process per rank and still passes on
the same machine, so the current evidence says ARLE needs either a native
C/C++ process-model design or a proven same-process NVSHMEM initialization path
before replacing the NCCL-backed default.

## Fix

No runtime fix landed in this entry. The shipped default remains:

- `ARLE_DSV4_MOE_BACKEND=deepep` when unset.
- `ARLE_DSV4_EXPERT_BACKEND=deepgemm-auto` when the MoE backend is DeepEP.
- `scripts/dsv4_toolchain.sh` validation defaults to
  `ARLE_DSV4_MOE_BACKEND=deepep` and required `ARLE_DSV4_EXPERT_BACKEND=deepgemm`.

The native DeepEP LL path must stay behind a future explicit axis until the
same-process or multi-process runtime contract is designed and verified.

## Results

| Gate | Result |
|---|---|
| DeepEP import in pod | PASS |
| `Buffer.is_sm90_compiled()` / SM90 support | PASS |
| RDMA size hint for `max=1, hidden=4096, ranks=8, experts=256` | PASS, `8521856` bytes |
| Official DeepEP low-latency multi-process DSv4-shape test | PASS, dispatch+combine about 48.7 us/rank |
| Same-process 8-thread DeepEP LL init/clean gate | FAIL, 180 s timeout |
| Default ARLE toolchain env-check after user default request | PASS, prints `ARLE_DSV4_MOE_BACKEND=deepep` |

## Artifacts

- Same-process native DeepEP LL gate:
  `dsv4-deepep-ll-sameprocess-gate-20260526`
- Official DeepEP low-latency DSv4-shape gate:
  `dsv4-deepep-ll-standalone-t1-h4096-topk8-e256-20260526/test_low_latency.log`
- Default env-check:
  `dsv4-default-deepep-envcheck-after-default-request-20260526.log`

## Rule

Keep `deepep` as the DSv4 default, but do not silently reinterpret that default
as native DeepEP low-latency until the process model has evidence. Official
multi-process DeepEP latency is a useful lower bound; it is not proof that
ARLE's same-process scheduler can initialize NVSHMEM low-latency buffers.
