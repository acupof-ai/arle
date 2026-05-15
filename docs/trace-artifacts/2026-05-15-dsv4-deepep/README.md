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
- [`nsys-single-decode-token-route-grouped-current/`](nsys-single-decode-token-route-grouped-current/)
  reruns the current opt-in route-grouped path before persistent grouped
  pointer caches. It returns exact arithmetic `406` and removes decode-window
  D2H, but still measures a 105.808 ms decode wave. The trace exposes 1,918 H2D
  memcpy activity calls / 374,752 bytes from small pointer/control copies plus
  heavy route-wise FP4/FP8 GEMV and reduce-scatter timing.
- [`nsys-single-decode-token-route-grouped-persistent-ptrs/`](nsys-single-decode-token-route-grouped-persistent-ptrs/)
  validates moving grouped expert weight/scale pointer tables into
  `DeepseekV4MoeBlock` load-time caches. The same opt-in route-grouped request
  returns `406`, H2D activity drops to 440 calls / 7,808 bytes, H2D runtime
  drops from 5.490 ms to 1.380 ms, and the single decode wave moves from
  105.808 ms to 94.828 ms. The path remains default-off because reduce-scatter
  combine and route-wise FP4/FP8 GEMV still dominate; this is a prerequisite
  cleanup for true grouped GEMM/DeepGEMM, not the final compute path.
- [`bench-persistent-grouped-ptrs-default-smoke/`](bench-persistent-grouped-ptrs-default-smoke/)
  keeps `ARLE_DSV4_ROUTE_GROUPED_EXPERTS=0` after the persistent pointer-cache
  change and verifies the default DeepEP path still loads the true checkpoint:
  math returns `410`, Chinese writing is normal, and a 16-token English decode
  produces normal text instead of repeated digits.
- [`bench-decode-pair-gemv-626477b1/`](bench-decode-pair-gemv-626477b1/)
  records a clean decode-only HTTP comparison for the default split expert GEMV
  path versus `ARLE_DSV4_PAIR_EXPERT_GEMV=1` on commit `626477b1`. Both paths
  return normal decode text and the arithmetic check returns `410`, but pair
  GEMV regresses `decode64` post-first throughput from 11.79 tok/s to
  7.70 tok/s, so it remains default-off.
- [`bench-route-grouped-pair-vs-default/`](bench-route-grouped-pair-vs-default/)
  records a trace-off HTTP comparison for the default fused-dispatch path
  versus `ARLE_DSV4_ROUTE_GROUPED_EXPERTS=1`. Both paths return normal writing
  output and exact arithmetic `410`, but route-grouped pair regresses `decode64`
  server-side completion throughput from 11.47 tok/s to 6.54 tok/s. The
  route-grouped path remains default-off.
- [`bench-stream-recycle/`](bench-stream-recycle/) records the matching
  trace-off HTTP smoke after incremental stream scratch recycling. The outputs
  remain normal and the arithmetic case returns `410`; `decode64` is still
  11.48 e2e requested tok/s, effectively unchanged from the 11.47 tok/s
  default baseline.
- [`bench-compressor-projection-scratch/`](bench-compressor-projection-scratch/)
  records the trace-off HTTP smoke after reusing GPU compressor projection
  buffers. The output checks remain normal and arithmetic returns `410`;
  `decode64` stays flat at 11.47 e2e requested tok/s.
- [`bench-attention-scratch/`](bench-attention-scratch/) records the matching
  trace-off HTTP smoke after reusing incremental attention scratch. It returns
  normal multi-token Chinese/English output and exact arithmetic `410`;
  `decode64` reaches 12.08 post-first tok/s.
- [`nsys-single-decode-token-attention-scratch/`](nsys-single-decode-token-attention-scratch/)
  reuses prepared Q/K, local-attention, and `wo_a` latent buffers in the
  B=1 incremental attention path without retaining prompt-sized prefill
  buffers. The profiled request returns `406`, alloc/free calls move from
  6,765/4,360 to 6,760/3,048, and the isolated single-token decode wave is
  97.042 ms. The remaining top costs are NCCL
  SendRecv/AllReduce, D2H route-count synchronization, launch/runtime overhead,
  local expert FP8/FP4 GEMV, and attention/MHC kernels; sampler is not visible
  in the top stack.
- [`nsys-single-decode-token-direct-20260515-0829/`](nsys-single-decode-token-direct-20260515-0829/)
  is the direct single-token nsys request used to answer where one generated
  token is slow on the current default path. It returns `406` and measures a
  97.071 ms decode wave. The largest buckets are NCCL SendRecv at 23.163 ms,
  local expert FP8/FP4 GEMV at 11.476/10.851 ms, AllReduce at 7.505 ms,
  attention at 7.399 ms, and large runtime overhead from kernel launches,
  async allocation/free, D2H readbacks, memsets, and event waits.
- [`bench-reduce-scatter-combine/`](bench-reduce-scatter-combine/) and
  [`nsys-single-decode-token-reduce-scatter-combine/`](nsys-single-decode-token-reduce-scatter-combine/)
  validate the default-on `ARLE_DSV4_COMBINE_REDUCE_SCATTER=1` path for B=1
  padded BF16 combine. The focused HTTP smoke keeps normal Chinese/English
  output and exact arithmetic `410`; `decode64` reaches 12.05 post-first
  tok/s. The single-token nsys wave moves from 97.071 ms to 94.923 ms, with
  return-side combine visible as `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL`
  at 20.443 ms and residual `SendRecv` down to 3.259 ms per rank range. The
  gain is limited; grouped GEMM/DeepGEMM plus DeepEP overlap remains the main
  performance target.
- [`nsys-single-decode-token-current-user/`](nsys-single-decode-token-current-user/)
  is a fresh user-requested single-token `nsys` rerun of the current
  default-on reduce-scatter path. The profiled arithmetic request returns
  `406`, and the isolated decode wave measures 94.841 ms. The trace shows the
  current slow stack directly: reduce-scatter combine at 20.549 ms per
  rank-range, local FP8/FP4 expert GEMV at 11.471/11.103 ms, attention at
  7.396 ms, all-reduce at 6.184 ms, route/MHC at 5.661/5.501 ms, plus large
  runtime overhead from 16,176 kernel launches, 6,760 async allocations, 3,048
  async frees, 3,640 memsets, and 344 D2H copies inside the decode range.
- [`nsys-single-decode-token-nccl-ll128/`](nsys-single-decode-token-nccl-ll128/)
  records the matching `NCCL_PROTO=LL128` protocol experiment. It also returns
  exact arithmetic `406`, but the isolated decode wave is 94.936 ms versus
  94.841 ms on the current default reference, with reduce-scatter combine
  slightly worse at 21.371 ms per rank-range. This stays a negative trace:
  changing NCCL protocol alone does not address the main decode bottleneck.
- [`nsys-single-decode-token-combine-overlap/`](nsys-single-decode-token-combine-overlap/)
  records the opt-in `ARLE_DSV4_COMBINE_OVERLAP=1` experiment. It returns
  exact arithmetic `406`; reduce-scatter improves from 20.549 ms to
  18.918 ms per rank-range, but the decode wave regresses to 104.359 ms
  because all-reduce timing and cross-stream event overhead dominate.
  The overlap path therefore stays default-off.
- [`nsys-single-decode-token-combine-overlap-disabled/`](nsys-single-decode-token-combine-overlap-disabled/)
  is the same binary with `ARLE_DSV4_COMBINE_OVERLAP=0`. It also returns exact
  arithmetic `406` and is kept as a control trace. The matching
  [`bench-combine-overlap-disabled/`](bench-combine-overlap-disabled/) smoke
  confirms default-off decode throughput remains at the current 12.05
  post-first tok/s baseline, while `prefill4k` still OOMs under the low
  `mem_fraction_static=0.10` profile.
- [`nsys-single-decode-token-attn-proj-scratch/`](nsys-single-decode-token-attn-proj-scratch/)
  validates per-layer incremental attention projection scratch reuse for
  `c_q`, `c_q_normed`, `q_raw`, `kv_raw`, and `kv_normed`. It returns exact
  arithmetic `406` and moves the single-token decode wave from 94.841 ms to
  90.946 ms, with `cuMemAllocAsync` calls down from 6,760 to 5,040 and
  `cuMemFreeAsync` calls down from 3,048 to 1,328. The matching
  [`bench-attn-proj-scratch/`](bench-attn-proj-scratch/) smoke keeps normal
  math and writing output, with `decode64` at 11.89 post-first tok/s.
- [`nsys-single-decode-token-current-breakdown/`](nsys-single-decode-token-current-breakdown/)
  is the latest direct user-requested single-token Nsight rerun of the current
  default DeepEP path. The profiled arithmetic request returns exact `406`; the
  isolated second-token decode wave measures 105.205 ms. The trace shows the
  concrete slow stack: 16,177 CUDA kernel launches at 34.159 ms per rank range,
  reduce-scatter combine at 20.122 ms, local FP8/FP4 expert GEMV at
  11.474/11.109 ms, all-reduce at 8.978 ms, attention at 7.394 ms, route/MHC at
  5.660/5.500 ms, and 347 D2H calls at 7.306 ms. The D2H activity payload is
  only 44,044 bytes, so the cost is synchronization/call overhead rather than
  bandwidth. This is not a sampler issue and not a missing-KV/full-prefill
  failure; the next target remains removing local-count host sync, real grouped
  GEMM/DeepGEMM, launch/runtime consolidation, and DeepEP combine overlap/fusion.
- [`nsys-single-decode-token-expanded-uninit/`](nsys-single-decode-token-expanded-uninit/)
  validates switching additional full-write DSv4 runtime scratch buffers
  (`HiddenStates`, MHC parameter buffers, and route logits) from zeroed
  allocation to uninitialized allocation. The same arithmetic request returns
  exact `406`; the isolated decode wave moves from 105.205 ms to 88.554 ms,
  and `cuMemsetD8Async` drops from 3,640 calls / 6.932 ms per rank range to
  1,920 calls / 2.839 ms. The remaining top costs are still reduce-scatter
  combine at 20.342 ms, local FP8/FP4 expert GEMV at 11.477/11.108 ms,
  attention/MHC/route kernels, and 16,177 CUDA launches, so the next large
  target remains real grouped GEMM/DeepGEMM plus DeepEP combine fusion/overlap.
- [`bench-expanded-uninit-smoke/`](bench-expanded-uninit-smoke/) records the
  matching trace-off HTTP smoke for the expanded uninit change. Multi-token
  Chinese and English output remains normal, arithmetic returns exact `410`,
  and `decode64` reaches 11.94 post-first tok/s.
- [`nsys-single-decode-token-moe-scratch-uninit-rerun/`](nsys-single-decode-token-moe-scratch-uninit-rerun/)
  extends the uninitialized allocation cleanup to MoE dispatch, payload,
  recv/local-route, active grouped, and combine scratch buffers. The profiled
  arithmetic request returns exact `406`; after rerunning to avoid a noisy NCCL
  sample, the isolated decode wave moves from 88.554 ms to 87.667 ms and
  `cuMemsetD8Async` drops from 1,920 calls / 2.839 ms per rank range to 1,232
  calls / 1.558 ms. Remaining dominant work is unchanged: reduce-scatter
  combine, FP8/FP4 expert GEMV, attention/MHC/route kernels, and 16,177 CUDA
  launches.
- [`bench-moe-scratch-uninit-smoke/`](bench-moe-scratch-uninit-smoke/) records
  the matching trace-off HTTP smoke. Multi-token Chinese and English output
  remains normal, arithmetic returns exact `410`, and `decode64` reaches 12.06
  post-first tok/s.
- [`nsys-single-decode-token-small-local-pack-prepare/`](nsys-single-decode-token-small-local-pack-prepare/)
  validates fusing the B=1 padded DeepEP local expert prepare step. The new
  small CUDA kernel clears local counts, counts received route metadata, writes
  device-side local offsets, and clears pack cursors. The arithmetic request
  returns exact `406`; H2D runtime calls drop from 1,040 to 696 and
  `cuMemsetD8Async` drops from 1,232 calls / 1.558 ms per rank range to 544
  calls / 0.728 ms. The single captured wave is 92.602 ms because D2H and
  AllReduce timing were noisier than the prior rerun, so this is recorded as
  small-call cleanup rather than a wall-time win.
- [`bench-small-local-pack-prepare-smoke/`](bench-small-local-pack-prepare-smoke/)
  records the matching trace-off HTTP smoke. Multi-token Chinese/English output
  remains normal, arithmetic returns exact `410`, and `decode64` reaches 12.05
  post-first tok/s.
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
- [`nsys-single-decode-token-stream-recycle/`](nsys-single-decode-token-stream-recycle/)
  validates incremental stream scratch recycling on the default fused-dispatch
  DeepEP path. The `霓彩` output remains normal and the single decode wave drops
  from 128.130 ms to 111.798 ms. Alloc/free overhead improves but remains
  visible: `cuMemAllocAsync` falls from 8,453 calls / 16.802 ms to 7,757 calls /
  12.574 ms, and `cuMemFreeAsync` falls from 6,048 calls / 13.801 ms to
  5,352 calls / 11.096 ms. HTTP throughput remains essentially unchanged, so
  the dominant target remains NCCL plus local expert GEMV.
- [`nsys-single-decode-token-compressor-projection-scratch/`](nsys-single-decode-token-compressor-projection-scratch/)
  reuses GPU compressor update `kv_raw` and `score_raw` projection buffers. It
  removes another 992 alloc/free pairs from the warmed decode window
  (`cuMemAllocAsync` 7,757 -> 6,765 and `cuMemFreeAsync` 5,352 -> 4,360), but
  the single captured wave regresses to 121.550 ms due to D2H/NCCL timing
  variance. This is recorded as allocator-pressure cleanup, not a throughput
  win.
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
