# B-3.3 native-deepep forward path doesn't dispatch to DeepGEMM

## Context

After committing the GCC 8.3 `-lstdc++fs` link fix (12dbf410), the
DeepGEMM-enabled bin built clean on the pod. Re-ran the same
5×128-token A/B with one variable flipped:

| Backend                            | p50 tok/s | mean  |
|------------------------------------|-----------|-------|
| native-deepep + EXPERT=native      | 15.82     | 15.81 |
| native-deepep + EXPERT=**deepgemm** | **15.81** | 15.79 |

**Zero measurable delta**. Expected lift was +3-5× from grouped-FP8
GEMM consolidating 256 local experts × 4 GEMM/layer × 43 layers ≈
44k naïve cuBLAS launches per decode step into ~tens of grouped
calls.

Server logs confirm DeepGEMM IS active at *weight load* time:

```
INFO infer::model::deepseek::mlp: mlp.rs:681
  DeepSeek V4 DeepGEMM FP8 expert cache built:
    experts=32 w13=131072x4096 scales=1024x32
    w2=131072x2048 scales=1024x16 bytes=768.19 MiB
```

Logged 344 times = 43 layers × 8 ranks, all caches built and
resident (~6.1 GiB across ranks). But decode throughput is
unchanged from the EXPERT=native run.

## Root Cause

`forward_native_deepep_routed_gpu` (B-3.3.4, infer/src/model/
deepseek/mlp.rs:4789ff) has a post-dispatch FFN section that runs:

```rust
for (local_expert_idx, expert) in self.experts.iter().enumerate() {
    if count <= 0 { continue; }
    let mut expert_input = unsafe {
        HiddenStates::uninit(ctx, hidden.hidden_dim, count_usize)?
    };
    ctx.stream.memcpy_dtod(&packed_x_slice, &mut expert_input.data)?;
    let expert_out_slice = expert.forward(ctx, &expert_input, ...)?;
    // dsv4_scatter_packed_expert_cuda(...) into expert_out
}
```

This is the **per-expert native cuBLAS path**. It never branches
into `use_deepgemm_experts` like the existing `forward_deepep_
routed_gpu` does at line 3149:

```rust
let use_deepgemm_experts = has_moe_scratch
    && match expert_backend {
        Dsv4ExpertBackend::Native => false,
        Dsv4ExpertBackend::DeepGemmRequired => true,
        Dsv4ExpertBackend::DeepGemmAuto => {
            deepgemm_backend_usable && self.deepgemm_cache.is_some()
        }
    };
```

When `use_deepgemm_experts == true`, the legacy DeepEP-style path
routes through `forward_route_grouped_experts_with_deepgemm` (or
similar — uses the `deepgemm_cache` to issue grouped masked GEMMs
on the packed routes). My B-3.3.4 path skipped that entirely
because the focus was correctness of the new Buffer.dispatch /
Buffer.combine wire-up; the grouped-DeepGEMM expert dispatcher is
~200-300 LOC of separate code.

The end result: DeepGEMM builds, caches, holds GPU memory, but
never executes — wasted setup + memory cost.

## Fix

Mirror the `use_deepgemm_experts` branch from
`forward_deepep_routed_gpu` into `forward_native_deepep_routed_gpu`:

1. Detect `use_deepgemm_experts` after the dispatch returns
   `num_recv_tokens`.
2. When true, instead of the per-expert pack + native loop, build
   the grouped DeepGEMM call:
   - active_indices/counts/offsets on the recv batch
   - call `forward_deepgemm_grouped_experts(recv_x, recv_topk_idx,
     recv_topk_w, packed_offsets, deepgemm_cache, ...)`
   - output lands in `expert_out` for Buffer.combine
3. Keep the native loop as fallback for when
   `deepgemm_backend_usable == false`.

Sub-commits:
- B-3.3.5.1 (~80 LOC): scratch struct fields for active_indices /
  active_counts / active_offsets in
  `DeepseekNativeDeepEpRuntimeScratch` (mirror
  `DeepseekGroupedExpertRuntimeScratch::active`).
- B-3.3.5.2 (~150 LOC): port the grouped expert FFN path from
  `forward_deepep_routed_gpu` 4030-4250 ish into a
  `run_grouped_deepgemm_experts_on_recv` helper that takes the
  recv buffers + writes expert_out.
- B-3.3.5.3 (~30 LOC): branch in
  `forward_native_deepep_routed_gpu` between grouped and native.

## Rule

When wiring a new transport path that **inherits** from an existing
forward function, **audit the receiver-side variant branches** in
the original, not just the dispatch/combine endpoints. The B-3.3
chain landed Buffer.dispatch + Buffer.combine cleanly but skipped
the expert backend variant (DeepGEMM vs native), which is a
fork-point inside the forward function that an outside-in code
review can miss.

The smoke test for this gap: when a config flag changes the FFN
backend, run a perf A/B with one variable flipped. If the delta is
0, the new path isn't actually exercising that flag's downstream
code — even if the flag's startup log lines look healthy.

## Refs

- B-3.3.4 implementation: commit 0c2ba5c9 (forward_native_deepep_
  routed_gpu post-dispatch FFN — the per-expert native path).
- DeepGEMM dispatch in baseline: forward_deepep_routed_gpu
  use_deepgemm_experts branch ~line 3149-3170.
- Pod e2e + B-4 perf A/B (native vs NCCL): wins entries
  04938e85 + a2eb08f1.
- GCC 8.3 stdc++fs link fix (this session, unblocked the build
  path even though the dispatch wasn't wired): commit 12dbf410.
