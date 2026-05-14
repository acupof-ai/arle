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
- [`nsys-single-token-padded-dispatch/`](nsys-single-token-padded-dispatch/)
  records the first fixed-top-k padded dispatch experiment. It removes the
  256-byte all-rank matrix readback but still runs the now-unused send-count
  kernel, so decode regresses to 136.908 ms. This is kept as a negative trace.
- [`nsys-single-token-padded-dispatch-skip-count/`](nsys-single-token-padded-dispatch-skip-count/)
  is the shipped B=1 decode path: fixed `ep_world * topk` padded dispatch plus
  skipping the unused send-rank zero/count kernel. The `霓彩` streaming output
  is normal, the decode wave drops to 123.955 ms, and decode-only D2H calls
  fall from 543 to 344 by deleting the 256-byte all-rank count readback. The
  remaining slow stack is NCCL SendRecv/AllReduce, launch/runtime churn,
  allocator/memset/free overhead, the local-count D2H, and FP8/FP4 expert GEMV.
- [`nsys-single-token-padded-peer-combine/`](nsys-single-token-padded-peer-combine/)
  adds the B=1 padded return-side combine optimization. Expert ranks now sum
  padded `topk` route outputs into one row per origin peer before the return
  exchange, shrinking combine payload from `ep_world * topk` rows to `ep_world`
  rows. The same `霓彩` output measures a 112.133 ms decode wave; `SendRecv`
  time drops from 25.211 ms to 23.329 ms per rank range, while local expert
  FP8/FP4 GEMV remains essentially unchanged.
- [`nsys-single-decode-token-current-b48/`](nsys-single-decode-token-current-b48/)
  reruns a current commit `b48a363d` single decode-token Nsight capture with
  streaming `max_tokens=2`, because `max_tokens=1` exits from prefill and does
  not create a decode launch. The `霓彩` output is normal. The single decode
  wave measures 125.497 ms, with the slow stack concentrated in NCCL
  SendRecv/AllReduce, alloc/free/memset/launch runtime overhead, and local
  FP8/FP4 expert GEMV. Actual D2H copy payload is only 44 KiB total; the
  visible `cuMemcpyDtoHAsync_v2` cost is call/synchronization overhead.
- [`nsys-single-decode-token-pair-gemv/`](nsys-single-decode-token-pair-gemv/)
  records a negative single-expert `w1`/`w3` pair GEMV experiment. The output
  remains `霓彩`, but the decode wave is 127.412 ms and the new
  `dsv4_fp4_gemv_pair_batch_kernel` costs 23.207 ms per rank range. The
  experiment is therefore gated behind `ARLE_DSV4_PAIR_EXPERT_GEMV=1` and
  default-off; simple gate/up fusion is not a substitute for real grouped
  GEMM/DeepGEMM plus DeepEP overlap.
- [`nsys-single-decode-token-route-grouped/`](nsys-single-decode-token-route-grouped/)
  records a negative route-wise grouped expert experiment behind
  `ARLE_DSV4_ROUTE_GROUPED_EXPERTS=1`. It removes the local-count D2H readback
  from the top runtime list, but the fixed padded route shape makes
  `dsv4_fp4_route_gemv_batch_kernel` cost 35.895 ms per rank range and regresses
  the single decode wave to 145.669 ms. This stays default-off; the compute
  target remains real grouped GEMM/DeepGEMM, not route-wise GEMV.
