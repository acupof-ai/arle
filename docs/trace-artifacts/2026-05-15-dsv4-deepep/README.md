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
- [`nsys-single-token-hidden-scratch/`](nsys-single-token-hidden-scratch/)
  validates per-layer hidden scratch reuse for incremental HC pre-projection
  and RMSNorm temporaries. The same streaming output remains `霓虹`, decode
  wave wall time drops from 145.104 ms to 135.390 ms, and decode-only
  alloc/free/memset calls each drop by 1,376. Remaining cost is still launch
  overhead, D2H route readback, NCCL SendRecv/AllReduce, and local expert
  FP8/FP4 GEMV.
- [`nsys-single-token-allgather-counts/`](nsys-single-token-allgather-counts/)
  removes the default AllGather path's redundant 32-byte send-count D2H
  readback by deriving send and receive counts from the same all-rank count
  matrix. The same `霓虹` output now measures a 129.768 ms decode wave, and
  decode-only D2H calls drop from 887 to 543. The remaining count readback is
  the 256-byte all-rank matrix.
