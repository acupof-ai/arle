# ARLE CUDA OPD Rollout Graph Capture KILL

## Context

Axis: capture one OPD rollout decode iteration as a CUDA Graph and replay it
for subsequent decode iterations. The target was to reduce Qwen3-0.6B OPD
step time from the post-grad-clip `~0.232 s` region toward `<= 0.18 s`.

No CUDA Graph code is retained. The probe was useful, but the high-level
rollout forward is not graph-replay-correct yet.

## Probe

Temporary probe file:

```text
crates/train/examples/opd_step_cuda_rollout_graph_probe.rs
```

The probe used `CudaContext::disable_event_tracking()` on a dedicated
nonblocking stream. Without that, cudarc's safe-launch event dependencies
created capture isolation failures unrelated to OPD.

Controls:

| Probe | Result |
|---|---|
| empty stream capture | captured |
| preallocated raw `mul_scalar_f32` kernel | captured, output `[2, 4, 6, 8]` |
| backend `mul_scalar` with output allocation inside capture | captured, output `[2, 4, 6, 8]` |
| backend cuBLAS matmul inside capture | captured, output `[19, 22, 43, 50]` |

This isolates the CUDA Graph API and cuBLAS as viable. The failure is in the
current high-level rollout decode graph.

## Evidence

Before the device-resident RoPE fix, capture failed during the first decode
layer's RoPE:

```text
probe_prefill next_token=888 decode_input=[888] decode_positions=[4]
cuda_readback_during_capture len=2048
probe_result status=forward_error error=cuda readback called during graph capture
```

The `2048` element readback is `[batch=1, heads=16, q_len=1, head_dim=128]`.
Root cause: CUDA implemented `rope_forward` for host slices but did not
override `Backend::rope`, so the default device method performed
`readback -> host rope -> upload`.

After adding the device-resident RoPE path, the full decode forward did
capture, but replay was numerically wrong:

```text
probe_prefill next_token=888 normal_decode_next=536 decode_input=[888] decode_positions=[4]
probe_result status=captured normal_next=536 captured_next=0 match=false
```

The controls still passed in the same run, including cuBLAS matmul capture.
That makes "CUDA Graph unsupported" and "cuBLAS cannot be captured" unlikely.

## Root Cause

The current rollout decode forward captures transient HtoD copies whose source
buffers are stack/local Rust values, not stable preallocated device inputs.
Examples on the captured path include:

- embedding token id uploads;
- RoPE `cos` / `sin` slices cloned into temporary host `Vec`s;
- slice/reshape metadata uploads such as starts and shape arrays.

CUDA Graph replay records the host source pointers for those memcpy nodes.
Once the temporary Rust buffers are dropped or reused, replay can still launch
successfully but no longer reproduces the normal decode output. The observed
`normal_next=536` vs `captured_next=0` mismatch trips the correctness gate.

## Fix

Killed the CUDA Graph implementation path for this tranche. No graph-capture
API, stream-constructor, or probe example is shipped.

The RoPE host fallback discovered by the probe was a separate valid axis and
is landed independently in
`docs/experience/wins/2026-05-21-arle-cuda-opd-device-rope.md`.

## Rule

Do not graph-capture the existing high-level rollout decode forward as-is.
Passing capture is not enough; replay must match normal greedy tokens. For
this code path, CUDA Graph capture needs a lower-level decode runner with:

- stable preallocated input buffers for token ids, position ids, and metadata;
- device-resident RoPE cache slices or persistent host-pinned memcpy sources;
- preallocated KV/output buffers, ideally the static max-rollout layout from
  option 2a;
- a replay correctness test that compares rollout token sequences before any
  wall-clock license decision.

Next graph attempt kill criterion: if a preallocated decode graph replay does
not match the non-graph token sequence for the first 7 decode iterations, stop
before benchmarking. If replay matches but Qwen3-0.6B remains above
`0.205 s/step`, kill graph capture as insufficient for this rollout length.

## Artefacts

- `bench-output/2026-05-21-arle-cuda-opd-rollout-graph-capture-kill/readback-failure-before-device-rope.txt`
- `bench-output/2026-05-21-arle-cuda-opd-rollout-graph-capture-kill/capture-controls-and-mismatch.txt`
