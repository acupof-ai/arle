# bf16 teacher-logits bridge: event-ordered D2D replaces context sync — CUDA, 2026-08-14

> Status: Shipped

## Goal

Remove the per-round full-context sync in the OPD teacher→student bf16 logits
bridge. Metric: `d2d_bridge_import_seconds` from `InferTeacherProfile`, measured
over one agent-OPD round on the H20 pod.

## Hypothesis

The bridge (`copy_bf16_device_ptr_to_local`) copied teacher logits with
`cuMemcpyDtoD_v2` on the default stream and then drained the whole context so
the foreign owner's `cuMemFreeAsync` could not race the copy. Ordering the copy
on the source stream and gating the student stream on a completion event
removes the host sync; the source's free is stream-ordered after the copy.
Expected: `d2d_bridge_import_seconds` drops to the copy time plus two
microsecond-class driver calls, with no correctness change.

## Parameters

A/B in one test process, same device state (H20, GPU 0, idle):

- `src_stream = 0` (legacy sync path: `cuMemcpyDtoD_v2` + `context.synchronize`)
- `src_stream = alt_stream` (event-ordered async path)
- Buffer: `[512, 151936]` bf16 = 155.6 MB, 5 calls per arm, p50 of 5.
- Test: `autograd::backend_cuda::tests::bf16_bridge_timing_realistic_logits`

## Environment

- Host / GPU: H20 pod (sm_90), single device, GPU 0 idle.
- Driver / CUDA: pod CUDA 12.6.
- Model / dtype: test buffer bf16 (no model load for the microbench).
- TP / EP / slots / KV: N/A (microbench).

## Results

| arm | ms/call | delta |
|---|---:|---:|
| legacy-sync (context sync) | 2.005 | — |
| event-ordered (this change) | 0.897 | **-55.3% (2.24x)** |

The legacy path also pays `alloc_zeros` (memset 155.6 MB); the event path uses
uninitialized `alloc` (the copy fully overwrites). On a busy device (teacher +
student co-resident), the legacy context sync drains more queued work, so the
production gap is likely wider than this idle-device microbench.

Correctness: `bf16_device_import_roundtrip_preserves_d2d_bytes_and_widens`
passes (cross-stream event handshake, second stream, byte-exact copy + bf16→f32
widening). Full autograd lib suite: 2/2 pass. One pre-existing unrelated
failure in `test_cuda_bf16_frozen_ops` (bf16 matmul tolerance, does not touch
the bridge).

## Problems

None.

## Learnings

PASS. The event-ordered bridge is 2.24x faster than the legacy sync path on an
idle H20. The `src_stream == 0` fallback retains the old behavior for callers
without a source stream. Site 2 (`release_kv_pool`) deferred — its syncs are
entangled with the OPD phase machine and VRAM accounting, and need a pod A/B of
their own.

