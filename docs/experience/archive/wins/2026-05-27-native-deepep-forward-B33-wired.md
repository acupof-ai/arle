# Phase B-3.3 — Native DeepEP forward path wired end-to-end (Mac-only)

## Goal

Phase B-3.2 ([`./2026-05-27-native-deepep-boot-B32-pod-verified.md`](./2026-05-27-native-deepep-boot-B32-pod-verified.md))
landed the boot path — `NativeDeepEp::boot` allocates a 512 MiB NVL
Buffer + opens N-1 peer IPC handles via the EP NCCL group, but the
forward path still routed through the NCCL DeepEP-style fallback.

B-3.3 wires the actual `Buffer.dispatch` → local-expert FFN →
`Buffer.combine` path into the model's MoE forward, so
`ARLE_DSV4_MOE_BACKEND=native-deepep` now goes through DeepEP's
intranode kernels for real instead of the NCCL emulation.

## What landed (B-3.3 chain)

| Commit | Phase | What |
|---|---|---|
| `3025cd8b` | B-3.3.1 | `DeepseekNativeDeepEpRuntimeScratch` + `ensure_native_deepep_scratch` on `DeepseekMoeRuntimeCache`. Worst-case device buffers (capacity_recv = capacity_tokens × ep_world). |
| `4e3a32e2` | B-3.3.2 | `dsv4_cast_i32_to_i64_cuda` kernel + FFI — converts dsv4_route's i32 indices to int64_t for DeepEP intranode::dispatch. |
| `d15aad45` | B-3.3.3 | `forward_native_deepep_routed_gpu` dispatch skeleton: routing prep, i32→i64 cast, Buffer.dispatch via `nde.buffer.lock()`. After dispatch the function bail!()d with a B-3.3.4 marker (no production call site yet). |
| `0c2ba5c9` | B-3.3.4 | Post-dispatch FFN + Buffer.combine + activation from `weights.rs`. The dispatch-only `bail!()` is gone — the path is now fully reachable from production when `ARLE_DSV4_MOE_BACKEND=native-deepep`. |

## Forward path data flow

Replaces `forward_deepep_routed_gpu`'s NCCL-emulated all-to-all with
DeepEP intranode kernels. Per layer per forward call:

```
   gate_gemm(hidden, gate_weight) → logits
   dsv4_route_cuda(logits, ...)   → route_indices[seq×topk] i32
                                     route_weights[seq×topk] f32
   dsv4_cast_i32_to_i64_cuda      → topk_idx_i64[seq×topk]

   Buffer.dispatch(d_x=hidden, d_topk_idx=topk_idx_i64,
                   d_topk_weights=route_weights, ...)
     → recv_x[num_recv, hidden]                  (post-dispatch tokens)
     → recv_src_idx[num_recv]                    (original token index)
     → recv_topk_idx[num_recv, topk] i64         (this recv's topk experts)
     → recv_topk_w[num_recv, topk]               (this recv's topk weights)
     → rank_prefix[world×world], recv_channel_prefix[world×channels],
       send_head[seq×world]                       (combine inputs)
     → num_recv_tokens (host scalar, returned by host-poll)

   dsv4_cast_i64_to_i32_cuda(recv_topk_idx → recv_topk_idx_i32)
   dsv4_count_local_experts_cuda(recv_topk_idx_i32, ...) → local_counts[experts_per_rank]
   exclusive scan + D2H → offsets_host[experts_per_rank]
   dsv4_pack_local_experts_cuda(recv_x, recv_topk_idx_i32, recv_topk_w, ...)
                                    → packed_x[total_local_routes, hidden],
                                      packed_token[total],
                                      packed_weight[total]
   for each local expert with count>0:
       memcpy_dtod(packed_x[expert_slice]) → expert_input HiddenStates
       expert.forward(expert_input)        → expert_out_slice
       dsv4_scatter_packed_expert_cuda     → expert_out[recv_token] += w · expert_out_slice

   Buffer.combine(d_x=expert_out, d_topk_weights=recv_topk_w,
                  d_recv_src_idx, d_rank_prefix, d_recv_channel_prefix,
                  d_send_head, ...)
     → combined_x[seq_len, hidden]
   return DeepseekRoutedMoeOutput { hidden: combined_x, ready: None }
```

The semantics: the rank-side FFN multiplies expert outputs by the
routing weights (so combine sees pre-weighted outputs), then combine
sums across the dispatch tree back to the source token's row.

## Build verification (Mac only)

| Build | Result |
|---|---|
| `cargo check --no-default-features --features cuda,no-cuda,nccl --lib` | PASS 1.83 s (4 unrelated dead-code warnings) |
| `cargo check --no-default-features --features cuda,no-cuda --lib` (deepep_stub) | PASS — function is `#[cfg(feature = "nccl")]` gated |
| Pod `cargo build -p infer --release --features cuda,nccl --lib` w/ `ARLE_DEEPEP_DIR=/sgl-workspace/DeepEP` | **NOT YET** — committed for next session |

The pod e2e (env=native-deepep → end-to-end greedy completion, parity
with env=deepep) is deferred. This session built the wire; B-4 / e2e
verification spends the pod time.

## What's NOT done — open items for next session

1. **Pod end-to-end smoke**: `cargo build -p infer --release --features cuda,nccl
   --lib` on the 8 × H20 pod with `ARLE_DEEPEP_DIR=/sgl-workspace/DeepEP`
   set. Should compile clean given Mac+no-cuda passes, but nvcc may
   surface link errors against DeepEP's `intranode::dispatch` /
   `intranode::combine` symbols. The two new cast kernels need nvcc
   rebuild (touched files in `crates/cuda-kernels/csrc/moe/` per the
   `cargo:rerun-if-changed=<dir>` non-recursive caveat — confirm
   build.rs walks the dir).

2. **Numerical parity smoke**: greedy completion with
   `ARLE_DSV4_MOE_BACKEND=native-deepep` vs `=deepep` should produce
   byte-identical first-token logits on a fixed prompt + seed.
   If they diverge, the most likely culprits in order:
   - Pre-multiplied weight in FFN double-applied by combine kernel
     (need to verify intranode::combine's d_topk_weights semantics —
     re-read DeepEP source / spike test).
   - i64↔i32 cast wrapping a high-bit expert index (n_routed_experts
     384 < i32::MAX, safe, but cross-check).
   - Worst-case capacity_recv = seq_len × ep_world is loose; DeepEP
     internally caps recv at smaller bound for large seq.

3. **B-4 SLO bench**: `scripts/bench_guidellm.sh dsv4-native-deepep` vs
   `dsv4-nccl-deepep-fallback`. Gate per pivot doc: TTFT +5%, TPOT +5%,
   p99 not regressed >3%. Hours of pod time, blocked on (1)+(2).

## Rule

When wiring a new GPU collective library through a multi-layer model
forward path, split the implementation into **dispatch-only first,
combine + activation second**:

1. Land the scratch struct (additive, zero risk).
2. Land kernel-utility shims (cast / convert kernels — small, isolated).
3. Land the new forward function with `Buffer.dispatch` and bail!()
   immediately after — proves the routing prep + dispatch params type
   correctly and the function compiles clean.
4. Replace the bail!() with the post-dispatch path AND wire from the
   call site in the same commit — the function only becomes
   production-reachable once it's correctness-complete. Avoids the
   "function exists but is a half-state" anti-pattern.

Each sub-commit is independently revertible. Mac-side typecheck after
each step. Pod build + e2e smoke deferred to a single, focused
verification session rather than four interleaved with each
sub-commit.

This mirrors the B-2 + B-3.1 + B-3.2 cadence — the four B-3.3
sub-commits land in one Mac session (≈ 1 hour), pod verification
spends pod time once on the merged result.
