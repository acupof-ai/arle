# DSv4 native DeepEP all-to-all transport wired + verified (axis ⑤ complete)

**Status:** PASS — native DeepEP dispatch/combine replaces `tp.all_reduce_sum` in the
DSv4 MoE forward; transport verified on H20; opt-in, all_reduce stays default.
**Track:** R6 clean-CUDA DSv4 (`crates/infer-cuda`), branch `arch/ideal-inference-engine`.
**SKU:** H20 8×sm_90a, CUDA 12.9, DeepSeek-V4-Flash.

## Context

The DSv4 MoE forward combined EP-sharded routed experts via `tp.all_reduce_sum`
(correct + 16/16, but not the efficient DeepEP all-to-all transport). The goal
("deepep 好好的") needed native DeepEP dispatch/combine wired in and verified.

## What Worked

New `crates/infer-cuda/src/deepep.rs` (`DeepEpTransport`: NCCL handle-exchange boot +
`sync` + dispatch/combine wrappers that own the worst-case scratch and the two
silent-deadlock gotcha params). The DSv4 MoE half (`dsv4.rs`) gains an opt-in branch
(`ARLE_DSV4_MOE_TRANSPORT=deepep`): dispatch → per-rank local-expert GEMM over
RECEIVED tokens (reusing the floored `deepgemm_grouped_experts`) → combine, skipping
`all_reduce_sum`; the replicated shared expert is added after combine, unchanged.
Supporting: `NcclBackend::all_gather_bytes` (`cuda-kernels/collective.rs`) for the
512-MiB IPC-handle exchange; a launcher fix (DeepEP `Buffer::new` needs all 8 devices
visible → `CUDA_VISIBLE_DEVICES=0..7` + `INFER_CUDA_DEVICE=$r`, not the per-rank mask).

**Two gotchas handled** (the deepep-sys C++ wrapper encodes them; the Rust caller
feeds them right): combine's channel-prefix arg = the dispatch-output
`recv_channel_prefix` (recv-side exclusive), and `num_input_tokens = R_r` /
`num_output_tokens = T` (inverted naming). **Topk-weight:** the local scatter
pre-weights via `packed_weight`; combine is still passed `recv_topk_weights` (legacy
contract) — licensed by the layer-0 tensor parity below.

**Verified (H20 TP=8/EP=8, hash prompt):**
- Layer-0 `moe_out` (before shared-expert add), allreduce vs DeepEP, **all 8 ranks**:
  max_abs `0.00390625`, mean_abs `1.8e-4`, rms `4.4e-4`, scale-relative `0.0039` vs
  tensor-max `0.99` → **BF16 summation-order noise, not a transport bug** (DeepEP
  combine sums in a different order than NCCL all-reduce).
- Token-1 end-to-end: allreduce `[260]` == DeepEP `[260]`.

Mirrored to the repo (semantic apply — patch base diverged): both `cuda,no-cuda` and
`cuda,no-cuda,deepep` typecheck (deepep-sys stub exposes the full API on Mac), clippy
`-D warnings` + fmt clean. All-reduce remains the default; non-deepep/single-GPU
builds are byte-identical (all new code feature-gated).

## Rule

- **DeepEP combine vs NCCL all-reduce differ by bf16 summation ORDER** — verify the
  transport by `moe_out` tensor parity within a float-order tolerance + a token match,
  NOT bit-exact equality.
- **DeepEP `Buffer::new` requires ALL ranks' devices visible** (it `cudaSetDevice(rank)`
  + asserts `device_count >= world_size`) — a per-rank `CUDA_VISIBLE_DEVICES=$r` mask
  that works for the compute path breaks DeepEP boot. Use all-visible + `INFER_CUDA_DEVICE=$r`.
