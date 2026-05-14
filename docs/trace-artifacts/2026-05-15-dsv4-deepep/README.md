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
- [`bench-decode-pair-gemv-626477b1/`](bench-decode-pair-gemv-626477b1/)
  records a clean decode-only HTTP comparison for the default split expert GEMV
  path versus `ARLE_DSV4_PAIR_EXPERT_GEMV=1` on commit `626477b1`. Both paths
  return normal decode text and the arithmetic check returns `410`, but pair
  GEMV regresses `decode64` post-first throughput from 11.79 tok/s to
  7.70 tok/s, so it remains default-off.
- [`nsys-single-decode-token-uninit/`](nsys-single-decode-token-uninit/)
  validates uninitialized allocation for selected full-write temporary hidden
  buffers. The `霓彩` output remains normal, `cuMemsetD8Async` drops from 8,789
  calls / 11.855 ms per rank range to 2,957 calls / 4.180 ms, and the isolated
  single decode wave moves from 125.497 ms to 112.724 ms. Remaining top costs
  are still NCCL SendRecv/AllReduce, launch overhead, async allocation/free,
  and local expert FP8/FP4 GEMV.
- [`nsys-single-decode-token-fused-dispatch-payload/`](nsys-single-decode-token-fused-dispatch-payload/)
  validates the default BF16 fused dispatch payload for B=1 DeepEP decode. It
  appends route metadata as raw 16-bit words behind each hidden row and sends
  hidden+metadata through one BF16 grouped exchange, reducing decode-window
  SendRecv launches from 1,032 to 688. The `霓彩` output remains normal and the
  latest isolated single decode wave is 118.985 ms, still dominated by NCCL,
  launch/runtime overhead, allocator churn, D2H, and local expert GEMV.
- [`nsys-single-decode-token-route-pair-gemv/`](nsys-single-decode-token-route-pair-gemv/)
  records the route-wise grouped expert follow-up that pairs the route-local
  `w1` and `w3` GEMV launches. The `max_tokens=2` request returns `霓彩` and
  measures a 117.894 ms decode wave, but the trace shows the slow stack is
  still `ncclDevKernel_SendRecv` at 50.338 ms per rank range, the FP4 route
  pair GEMV at 19.616 ms, the FP4 route `w2` GEMV at 10.487 ms, FP8 GEMV at
  9.408 ms, plus allocator and launch overhead. The route-grouped path remains
  opt-in; this is evidence for grouped GEMM/DeepGEMM plus DeepEP overlap, not a
  default-path replacement.
- [`nsys-single-decode-token-default-warm-decode/`](nsys-single-decode-token-default-warm-decode/)
  reruns the default fused-dispatch DeepEP path after a real `max_tokens=2`
  decode warmup, then profiles a second `max_tokens=2` request. The `霓彩`
  output remains normal and the profiled single decode wave is 128.130 ms.
  Warmup does not remove allocator/free churn: decode-window runtime still has
  8,453 `cuMemAllocAsync` calls and 6,048 `cuMemFreeAsync` calls, while actual
  D2H payload is only 44 KiB total. The steady-state bottleneck is NCCL
  SendRecv/AllReduce, local expert FP8/FP4 GEMV, CUDA launch overhead,
  allocator/free overhead, and route-count D2H synchronization.
- [`nsys-single-decode-token-expert-grouped/`](nsys-single-decode-token-expert-grouped/)
  records the opt-in `ARLE_DSV4_GROUPED_EXPERTS=1` expert-wise grouped GEMV
  path after the same real decode warmup. The output remains `霓彩`, but the
  single decode wave regresses to 145.693 ms. `ncclDevKernel_SendRecv` rises to
  58.049 ms per rank range, the FP4 grouped gate/up GEMV costs 23.162 ms, and
  the FP4 grouped `w2` GEMV costs 11.428 ms. This stays default-off; the trace
  confirms that the current grouped GEMV path is not the target grouped
  GEMM/DeepGEMM implementation.
- [`bench-fused-dispatch-payload-local/`](bench-fused-dispatch-payload-local/)
  records the matching trace-off HTTP smoke. `decode64` returns normal English
  content at 12.22 post-first tok/s and the arithmetic case returns `410`.
