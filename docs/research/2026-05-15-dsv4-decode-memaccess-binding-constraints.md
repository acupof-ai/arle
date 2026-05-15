# DSv4 decode memory-access binding constraints

Date: 2026-05-15

Scope: Phase 0 read-only audit for DSv4 decode memory-access optimization.
No code changes, no new benchmark runs.

## Sources

[SOLID] Primary trace and request frame:

- `docs/experience/errors/2026-05-14-dsv4-decode-nccl-bottleneck.md`
- `docs/trace-artifacts/2026-05-14-dsv4-decode/README.md`
- `docs/trace-artifacts/2026-05-14-dsv4-decode/arle-dsv4-decode-nsys-stats.txt`
- `docs/trace-artifacts/2026-05-14-dsv4-decode/arle-dsv4-decode-nsys-client.json`
- `docs/trace-artifacts/2026-05-14-dsv4-decode/arle-dsv4-default-after.json`

[SOLID] Recent post-trace commits reviewed:

- `277ca10a` Accelerate DeepSeek V4 GPU MoE path
- `664d0103` Tune DeepSeek V4 batch GEMV tile
- `ee7bd085` Optimize gated DSv4 grouped experts
- `bd2fc01d` Reuse DSv4 route logits scratch
- `2b397db2` Reuse DSv4 shared expert scratch
- `c6906b07` Reuse DSv4 incremental hidden scratch
- `f314c787` Reuse DSv4 incremental stream scratch
- `53bb10e1` Reuse DSv4 compressor projection scratch
- `bce43d3b` Reuse DSv4 decode attention scratch
- `a4bbbf8` Reuse DSv4 attention projection scratch
- `8545882b` Reduce DSv4 decode scratch zeroing
- `b1e9ac6e` Reduce DSv4 MoE scratch memset overhead
- `7d6739f9` Cache DSv4 grouped expert pointer tables
- `521b8a10` Fuse DSv4 local expert prepare

[SOLID] Current-state post-trace context:

- `docs/trace-artifacts/2026-05-15-dsv4-deepep/README.md`
- Current single-token traces under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-*`

## Wall-clock frame

[SOLID] The 2026-05-14 nsys request returned HTTP 200, generated 32 completion
tokens, and took `6.464 s` total with `0.4226 s` TTFT. That is `4.95`
completion tok/s end-to-end. The nsys NVTX table reports 248
`step_decode_kernel_launch` rank ranges with `194.169 ms` average range time,
which is consistent with roughly 31 decode waves across 8 ranks:
`248 / 8 = 31`.

[SOLID] The non-nsys follow-up in `arle-dsv4-default-after.json` measured
32 tokens in `4.2694 s` (`7.50` e2e tok/s) and 64 tokens in `8.3799 s`
(`7.64` e2e tok/s). The error entry records
`infer_scheduler_step_phase_decode_microseconds=117646` after the 64-token run.

[SOLID] The 2026-05-15 current single-token traces show that post-trace work has
changed the starting point materially. Representative current default traces
report single decode waves around `94.841 ms`, `105.205 ms`, `88.554 ms`,
`87.667 ms`, and `92.602 ms` depending on the exact experiment and NCCL/D2H
noise sample.

[Hypothesis] Therefore the 2026-05-14 trace is still the binding historical
source, but Phase 1 fixes must begin with current-main caller-count evidence.
The old `522k` allocation count cannot be used as a direct patch target after
the scratch-reuse chain.

## Binding constraints table

| Layer | Symptom | Cost | Examples | Status |
|---|---|---:|---|---|
| L1 Allocator storm | [SOLID] 2026-05-14 nsys shows `522,765` `cuMemAllocAsync` and `522,765` `cuMemFreeAsync` calls in the 32-token trace. | `31.1%` + `32.5%` of CUDA API time in the 2026-05-14 trace. Current single-token traces still show thousands of alloc/free calls, but much fewer than the old trace. | `cuMemAllocAsync`, `cuMemFreeAsync`; DSv4 per-token/per-layer scratch paths. | License Phase 1 only as caller-counted current-main audit plus single-site A/B. |
| L2 Memset churn | [SOLID] 2026-05-14 nsys shows `505,347` `cuMemsetD8Async` calls. CUDA activity table shows `505,347` CUDA memset operations. | `9.4%` CUDA API time; `90.6%` of CUDA memory-operation activity time. Current single-token traces after `8545882`, `b1e9ac6`, and `521b8a1` reduce this axis heavily. | Scratch zeroing before DSv4 hidden, MHC, route, MoE dispatch, local-route, and combine buffers. | Keep as Phase 2, but require fresh top-20 memset caller table before deleting more zeroing. |
| L3 In-kernel memory access | [SOLID] Top compute kernels in 2026-05-14 are GEMV, route, and MHC parameter kernels. | GPU time: FP8 batch GEMV `8.6%`, FP4 batch GEMV `4.5%`, route `4.4%`, MHC params `4.3%`, FP8 tiled `1.6%`, FP4 tiled `1.3%`. | `dsv4_fp8_gemv_batch_kernel`, `dsv4_fp4_gemv_batch_kernel`, `dsv4_route_kernel`, `dsv4_mhc_params_kernel`, `dsv4_fp8/fp4_gemv_batch_tiled_kernel`. | License Phase 3 after L1/L2 audit unless fresh traces rank it higher. |
| L4 Attention KV access | [SOLID] 2026-05-14 nsys shows hybrid attention and CSA selection as material GPU kernels. | `dsv4_hybrid_attention_kernel` `6.4%` plus `dsv4_csa_select_kernel` `3.9%` GPU time in 2026-05-14 trace. | Hybrid attention, CSA select, SWA attention, compressor update, KV window/cache update. | Secondary to NCCL/GEMV/runtime, but real; keep separate from GEMV work. |
| L5 DtoH readback | [SOLID] 2026-05-14 nsys API table shows `11,264` `cuMemcpyDtoHAsync_v2` calls, avg `589.939 us`, max `4.123 ms`. CUDA activity shows only `27.223 ms` total DtoH device-copy activity. | `14.3%` CUDA API time in 2026-05-14 trace. Current single-token traces show `344`-`347` DtoH calls and only about `44 KiB` payload, so synchronization/call overhead dominates bandwidth. | Route/local-count readbacks, sampler/control-plane pulls, scheduler-side offsets/counts. | License Phase 4 after caller attribution; do not treat it as bulk bandwidth. |
| L6 NCCL collective | [SOLID] 2026-05-14 nsys shows `22,016` `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` instances. | `19.989 s` of GPU time, `60.7%` of GPU kernel time. Current DeepEP reduce-scatter path still spends about `20 ms` per rank-range in return-side combine. | Attention all-reduce, MoE output reduce/combine, later reduce-scatter/sendrecv traces. | Deferred by brief; requires DeepEP/DeepGEMM style plan, not this memory-access tranche. |

## Cross-check against recent commits

[SOLID] The scratch-reuse chain changed the active baseline:

- `bd2fc01d`, `2b397db2`, `c6906b07`, `f314c787`, `53bb10e1`,
  `bce43d3b`, and `a4bbbf8` added reuse for route logits, shared expert,
  incremental hidden, stream, compressor projection, attention, and attention
  projection scratch.
- `8545882b` and `b1e9ac6e` switched additional full-write DSv4 scratch
  buffers away from pessimistic zeroing.
- `521b8a10` fused local expert prepare work and reduced H2D/memset pressure.
- `664d0103` and the current upstream tree changed batch GEMV dispatch by
  adding B>1 tiled kernels. The Prelude commit `8c8b90ab` only preserves the
  legacy FP4 batch fallback hoist; it is not evidence for the active tiled B>1
  path.

[SOLID] Current single-token trace snippets from
`docs/trace-artifacts/2026-05-15-dsv4-deepep/README.md`:

- `nsys-single-decode-token-current-breakdown`: `105.205 ms` wave,
  `16,177` launches, `5,040` allocs, `1,328` frees, `3,640` memsets,
  `347` DtoH calls, `44,044 B` DtoH activity.
- `nsys-single-decode-token-attn-proj-scratch`: attention projection scratch
  reuse moved the wave from `94.841 ms` to `90.946 ms`, allocs `6,760` to
  `5,040`, frees `3,048` to `1,328`.
- `nsys-single-decode-token-expanded-uninit`: memset calls moved from `3,640`
  to `1,920`, decode wave from `105.205 ms` to `88.554 ms`.
- `nsys-single-decode-token-moe-scratch-uninit-rerun`: memsets moved from
  `1,920` to `1,232`, wave `87.667 ms`.
- `nsys-single-decode-token-small-local-pack-prepare`: memsets moved from
  `1,232` to `544`; wave `92.602 ms`, explicitly not recorded as a wall-time
  win because D2H/AllReduce were noisier.

[Hypothesis] Residual L1 alloc/free remains large enough to audit first:
current traces still show `5,040` allocs and `1,328` frees per isolated decode
wave, with per-rank-range API time typically around `5-8 ms` for alloc and
`1-2 ms` for free. Against a `90-105 ms` decode wave, a fully licensed site
could still clear the Phase 1 `>=3%` gate, but only if caller attribution finds
a concentrated residual source.

## Phase ordering decision

[SOLID] I do not find a contradiction with the 2026-05-14 table. The original
trace really was:

- NCCL dominated on GPU time.
- Alloc/free dominated CUDA API time.
- Memsets dominated CUDA memory-operation activity.
- DtoH was expensive as API synchronization, not payload bandwidth.
- GEMV/route/MHC/attention kernels were the main non-NCCL GPU compute stack.

[SOLID] The current-main traces mean the old L1/L2 raw counts are stale as
patch targets. That does not invalidate Phase 1; it strengthens the brief's
requirement to start Phase 1 with NVTX/caller-count instrumentation on current
main before any preallocation patch.

## License-or-kill outcome

[SOLID] License Phase 1 as an evidence-gathering tranche:

1. Re-collect or derive current-main allocation caller counts inside
   `step_decode_kernel_launch`.
2. Rank top alloc/free callers by call count and bytes.
3. Patch at most one residual caller per commit.
4. Run the required matched wall-clock A/B before claiming any win.

[SOLID] No Phase 0 framing-trap error entry is required. The table is correct
for the 2026-05-14 trace, and the document explicitly defers current-main
fix selection until caller-count evidence exists.

