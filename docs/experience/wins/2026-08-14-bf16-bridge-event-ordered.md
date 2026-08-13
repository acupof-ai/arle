# bf16 teacher-logits bridge: event-ordered D2D replaces context sync — CUDA, 2026-08-14

> Status: pending-remote

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

A/B over one OPD round (teacher 27B bf16, student on the same device):

- Baseline: `git stash` (context-sync bridge), one round, read
  `d2d_bridge_import_seconds` from the round profile.
- Treatment: this commit, same round, same profile field.
- Correctness gate: `bf16_device_import_roundtrip_preserves_d2d_bytes_and_widens`
  unit test (cross-stream event handshake, second stream) plus one full OPD
  round with `--share-frozen-base` to exercise the production path.
- Trials: 3 rounds per arm, report p50.

## Environment

- Host / GPU: H20 pod (sm_90), single device, teacher + student co-resident.
- Driver / CUDA: pod CUDA 12.6.
- Model / dtype: teacher Qwen3.6-27B bf16, student bf16, `--share-frozen-base`.
- TP / EP / slots / KV: teacher TP=1, one transient OPD slot.

## Results

| arm | d2d_bridge_import_seconds p50 | delta |
|---|---:|---:|
| baseline (context sync) | | — |
| treatment (event handshake) | | |

Raw artifacts: `<round profile log>`.

## Problems

None yet — pending the pod run.

## Learnings

pending-remote. The unit test exercises the cross-stream handshake on a second
stream of the same primary context; the production teacher/student stream pair
is the same shape. The `src_stream == 0` fallback retains the legacy sync path
for callers without a source stream.
