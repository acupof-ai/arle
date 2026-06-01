# DSv4 SGLang path alignment plan

## Context

The current DSv4 optimization queue was framed around the ARLE default path:

```text
FlashMLA attention + replicated-token TP/EP + local routed experts + FFN all-reduce
```

That path is correct enough to serve tokens, but it is not SGLang's DSv4 path.
Operator-level fixes on this route can improve ARLE, but they cannot license
an "ARLE > SGLang" claim because the data distribution and runtime contract are
different.

This plan is the controlling plan for the SGLang-gap campaign. The older
roofline document remains useful only as an operator backlog after the path
contract below is satisfied.

## Target framing

The user-supplied SGLang reference is about 18 ms/token. If that number is the
accepted apples-to-apples target-step TPOT, then "ARLE exceeds SGLang by 20%"
requires:

```text
ARLE TPOT <= 14.4 ms/token
```

If the 18 ms number includes EAGLE/MTP accepted-output-token accounting, it must
not be compared to raw target-model steps. In that case ARLE must report both:

- raw target-model TPOT
- effective output-token TPOT after accepted drafts

The prior 60-64 ms target is not valid for the current claim. It came from a
different community datapoint and is now deprecated for this campaign.

## Evidence status

Confirmed ARLE evidence:

- p2048/o8 warm default path: about 112 ms/token before the CSA fast path.
- p2047/o8 warm after top-k-cover CSA fast path: about 91 ms/token.
- p2047/o32 warm after top-k-cover CSA fast path: about 117 ms/token.
- native DeepEP on the current replicated-token path over-transports by about
  4.4x and loses to all-reduce.

Confirmed SGLang-source path from remote source survey:

- Runtime source tree is `/workspace/sglang @ 0d51db3`. The older
  `/sgl-workspace/sglang @ 232982a` tree is not the Python import path for DSv4.
- DSv4 hook defaults: attention backend `dsv4`, page size 256, FP8 KV cache.
- Attention fuses Q/KV A projections, norm/RoPE, paged FlashMLA cache write,
  and FlashMLA decode with sparse/recent indices.
- MoE uses DeepEP low-latency dispatch/combine or MegaMoE fused expert paths.
- Tested high-end path uses TP/DP style decomposition, DP attention, native
  DeepEP, CUDA graphs, and optional EAGLE/MTP.
- H200 FP8 cookbook lanes use `sgl-project/DeepSeek-V4-Flash-FP8`,
  `SGLANG_DSV4_FP4_EXPERTS=0`, and for Balanced/MaxThroughput use
  `--tp 4 --dp 4 --enable-dp-attention --moe-a2a-backend deepep`.

Invalid or stale SGLang control artifacts:

- `/sgl-workspace/bench-artifacts/dsv4-analysis-20260531-sglang-8k/server.log`
  fails on `KeyError: 'deepseek_v4'`.
- `/sgl-workspace/bench-artifacts/dsv4-longseq-20260525/sglang_server.log`
  fails on stale `sglang-kernel`.

Conclusion: the 18 ms target can be used as the product target requested by the
user, but the pod still needs a fresh SGLang control run before final
percentage claims.

## User-supplied vLLM/SGLang trace reference

The latest comparative trace reference changes the operator priority after path
alignment. It says the major gap between vLLM and SGLang is MoE MLP plus EP
transport, not the attention main kernel:

- `swiglu_limit_func` / `triton_poi_fused_clamp_copy__mul_silu_slice_0`: vLLM
  about 788 ms vs SGLang `per_token_group_quant_8bit_kernel` about 1.3 ms or
  `silu_mul_quant_varlen_kernel` about 15 ms.
- expert GEMM: vLLM `marlin_moe_wna16::Marlin` about 133 ms vs SGLang
  `deep_gemm::sm90_fp8_gemm_1d2d_impl` about 30 ms.
- EP transport: vLLM DeepEP combine about 114 ms vs SGLang about 25 ms;
  dispatch about 181 ms vs about 60 ms.
- buffer materialization: vLLM `FillFunctor<BFloat16>` about 24 ms; SGLang
  shows lighter copy/init cost.
- attention main kernel `flash_fwd_splitkv_mla_fp8_sparse_kernel` is close
  between the two paths, and vLLM can even be slightly lower.

This is a user-supplied reference, not yet independently reproduced in the pod.
It is still useful for priority: after PC0-PC2 remove replicated-token
confounders, the first roofline target is MoE MLP + EP dispatch/combine +
materialization. ARLE's current attention/CSA cost remains a real
current-route blocker, but it is not the SGLang-path P0 unless a fresh
path-aligned trace proves it again.

## Current ARLE chain

Decode request:

```text
HTTP scheduler
  -> DistributedSchedulerGroup submits the same logical request to every rank
  -> each rank owns the same token rows
  -> DSv4 batched decode batches FFN rows, but attention loops per row
  -> FlashMLA decode per row
  -> routed local experts on every rank
  -> post-MoE EP all-reduce sums hidden states
  -> rank 0 token selection is broadcast to follower ranks
```

Important current code boundaries:

- `infer/src/main.rs::deepseek_parallel_config_for_rank` only accepts TP and EP
  sizes of 1 or total worker count.
- `infer/src/model/deepseek/config.rs::DeepseekRuntimeConfig` stores only
  `tp` and `ep`.
- `infer/src/model/deepseek/weights.rs::layer_communicator_from_config` passes
  DP and CP as `(rank=0, world=1)`.
- `infer/src/model/deepseek/weights.rs::forward_decode_batch` still runs the
  attention core per row.
- `infer/src/model/deepseek/weights.rs::forward_ffn_layer_stream_with_scratch_into`
  correctly blocks native DeepEP on the replicated-token path by default.

## Why the phase costs are unreasonable

Treating the prior per-token breakdown as true, each high-cost item points to a
path mismatch, not just a slow kernel.

| Stage | Observed order | Why unreasonable | Required path fix |
|---|---:|---|---|
| `ffn_routed_local` | about 25 ms/token | every rank runs routed local expert work for replicated tokens; user-supplied SGLang trace also says MoE MLP dominates cross-runtime gaps | token-owned MoE rows plus native DeepEP/MegaMoE-style fused expert path |
| `ffn_all_reduce` | about 19 ms/token | all-reduce is the wrong combine primitive once rows are token-owned | remove post-FFN all-reduce from the SGLang path; use DeepEP combine |
| `attn_hybrid_kernel` | about 17 ms/token | attention is launched per row and carries ARLE metadata overhead | batched FlashMLA decode with persistent paged KV and SGLang index contract |
| `attn_csa_select_kernel` | about 11 ms/token avg, worse in long output | selector is recomputed per layer/row/token instead of prepared in the SGLang metadata path | in-graph/in-kernel metadata prep and batched sparse index build |
| `ffn_expert_loop` | about 6 ms/token | local loop is still visible after DeepGEMM-auto because transport/topology is wrong | fused dispatch + grouped expert GEMM on received rows |
| `attn_all_reduce` | about 5 ms/token | serial TP reduce on attention output | aligned TP/attention-DP groups plus overlap/graph capture |
| `ffn_shared` | about 5 ms/token | shared expert is not on the fused SGLang/MegaMoE-style path | tensor-core/DeepGEMM shared expert path |
| `attn_proj` | about 3 ms/token | projection remains an isolated block-scaled op | fuse or move to tensor-core path |

The sum of these items cannot be fixed by selecting one kernel. The first
systemic fix is to stop benchmarking the replicated-token route as though it
were SGLang-equivalent.

## Path contract

An ARLE run may be called "SGLang-path aligned" only when all of the following
are true:

1. Process model: one CUDA process per rank for native DeepEP process-local
   CUDA context semantics.
2. Rank layout: runtime has explicit TP, attention-DP/CP, MoE-EP/DP/TP
   coordinates, not only global TP and EP.
3. Request ownership: each DP/EP shard owns distinct token rows. It must not
   submit the full logical request to every rank.
4. MoE: routed expert transport uses native DeepEP low-latency
   dispatch/combine or a MegaMoE-equivalent fused path on token-owned rows.
5. Attention: decode uses batched FlashMLA over a persistent paged FP8 KV pool,
   with SGLang-compatible sparse/recent page indices.
6. Graph/metadata: recurrent decode metadata is prepared inside the captured
   path or in persistent device structures, not rebuilt by host-driven per-row
   launches.
7. Metrics: raw target TPOT and speculative/effective TPOT are reported
   separately whenever EAGLE/MTP is enabled.
8. Correctness: normal token output, token counts, finish reason, and at least
   one greedy A/B sanity probe pass before perf is accepted.

## Implementation order

This is one path-level milestone, not independent micro-optimizations. The
pieces below are ordered because later perf data is invalid without the earlier
contract.

### PC0 - fail closed

- Add a startup guard for explicit `ARLE_DSV4_SGLANG_PATH=1` claims.
- The guard must reject the current replicated-token/per-row-attention path
  with actionable missing contract items.
- Add runtime logging of the current DSv4 topology so traces say whether the
  run is `replicated-token` or `sglang-path-candidate`.

### PC1 - rank layout becomes runtime data

- Promote existing `MultiAxisConfig` parsing from test-only math into runtime
  diagnostics.
- Thread the axis coordinates into `DeepseekRuntimeConfig` and
  `LayerCommunicator`.
- Do not attach fake DP/CP groups. Missing communicators must fail closed.

### PC2 - token ownership

- Change distributed request fanout from "same request to every rank" to
  DP-owned request shards.
- Token synchronization remains rank-0-rooted for visible output, but hidden
  rows passed into MoE must be unique to the owning EP/DP rank.
- Native DeepEP default remains blocked until this is proven by counters:
  `sum(num_recv) / (ep * src_tokens)` must match owned-row fanout, not the old
  replicated-token 4.4x model.

### PC3 - MoE/EP fused path

Primary entry points:

- SwiGLU/clamp: `infer/src/model/deepseek/mlp.rs::DeepseekV4Expert::forward_scratch_input`
  and `infer/src/ops/elementwise.rs::dsv4_swiglu_clamped_batch_into`.
- Fused SwiGLU + quant: `forward_deepgemm_grouped_dsv4_experts_gpu` and
  `dsv4_deepgemm_swiglu_quantize_w13_cuda`.
- Expert GEMM: `dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked_cuda` and
  `crates/cuda-kernels/csrc/gemm/deepgemm_native.cu`.
- Native DeepEP: `forward_native_deepep_routed_gpu` plus
  `deepep_sys::Buffer::{dispatch,combine}`.
- Materialization risk: `ensure_deepgemm_scratch`, `ensure_native_deepep_scratch`,
  per-call `input_fp8/input_scales/act_fp8/act_scales` zeroing, and
  `dsv4_pack_dispatch_payload_cuda` / `dsv4_unpack_dispatch_payload_cuda`.

- Enable native DeepEP only after PC2.
- Remove post-FFN all-reduce from the SGLang candidate path.
- Use DeepGEMM/MegaMoE-class fused expert compute and shared expert compute.
- Keep the hot path on fused `silu_mul_quant` / DeepGEMM activation quant
  instead of BF16 SwiGLU followed by separate quantization.
- Eliminate DeepGEMM scratch zeroing and DeepEP payload materialization unless
  a trace proves the masked valid-count contract still reads those bytes.
  First landed tranche: FP8 input/activation scratch clears are skipped by
  default, with `ARLE_DSV4_DEEPGEMM_ZERO_FP8_SCRATCH=1` as the rollback/A-B
  switch; scale scratch remains zeroed for TMA padding safety.
- Keep route metadata on device; do not reintroduce local-count D2H, offset H2D,
  or active-expert H2D into the decode loop.

### PC4 - attention batch contract

- Shared FP8 paged KV must be long-context correct.
- Batched FlashMLA decode must avoid per-row staging and per-row selector
  rebuilds.
- The c=8/c=32 path must show flat or sublinear attention-step scaling before
  throughput is considered comparable.

### PC5 - metric lane

- Run a fresh SGLang control on the actual pod/image and workload.
- Run ARLE raw-target without speculation.
- Run ARLE with EAGLE/MTP only in a separate lane and report accepted-token
  accounting separately.

## Stop rules

- No operator-level win can close the SGLang campaign while PC0-PC3 are unmet.
- Native DeepEP remains non-default on replicated tokens.
- A trace narrow-window reduction is not accepted without same-server
  no-trace wall-clock improvement.
- Any "SGLang +20%" claim must name the exact SGLang artifact, ARLE artifact,
  workload, quantization, TP/DP/EP layout, and whether TPOT is raw or effective.
