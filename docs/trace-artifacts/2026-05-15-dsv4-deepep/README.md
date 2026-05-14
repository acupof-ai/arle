# DSv4 DeepEP Trace Artifacts

Date: 2026-05-15

These artifacts were captured on the remote 8xH20 host against the real
`/root/DeepSeek-V4-Flash` checkpoint. The run uses the default DSv4 DeepEP MoE
path with FP8 KV cache and incremental KV enabled.

Current trace set:

- [`nsys-one-token-current/`](nsys-one-token-current/) isolates one generated
  decode token with Nsight Systems.
- [`shared-expert-scratch/`](shared-expert-scratch/) records the shared expert
  scratch cleanup, including trace-off math/writing smoke and a single-token
  Nsight comparison.
- [`nsys-single-token-live/`](nsys-single-token-live/) is the current live
  single-token rerun from a clean remote source tree. It shows a 146.448 ms
  decode wave with the remaining cost concentrated in allocator/runtime churn,
  D2H routing readback, NCCL exchange, and per-expert GEMV.
- [`nsys-single-token-segment-input/`](nsys-single-token-segment-input/)
  validates the local expert packed-input segment path. It keeps the same
  `霓虹` output, trims decode-only `cuMemcpyDtoDAsync_v2` from 871 calls /
  1.795 ms per rank range to 613 calls / 1.240 ms, and leaves the main
  bottleneck concentrated in allocator/runtime churn, D2H readback, NCCL
  exchange, and per-expert GEMV.
