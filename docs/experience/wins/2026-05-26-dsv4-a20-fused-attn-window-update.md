# DSv4 A2.0 Fuses Decode Window Update Into Attention

## Context

DSv4 default CUDA decode now uses the DeepEP-style MoE path with the
DeepGEMM-auto local expert backend. The remaining single-request traces still
show heavy launch churn. A small, isolated A2.0 axis was to remove the
standalone sliding-window cache update launch from B=1 decode without changing
attention math, routing, or expert backend selection.

## What Worked

The SWA and hybrid DSv4 attention kernels now optionally write the current key
into the sliding-window cache from the `head == 0` block after attention output
is complete. Rust enables this only when a mutable cache exists and
`token_count == 1`; prefill and multi-token steps keep the previous standalone
update kernel. `ARLE_DSV4_FUSE_ATTN_WINDOW_UPDATE=0` remains an A/B escape
hatch.

Remote H20 validation artifacts:

| Run | Artifact | Result |
| --- | --- | --- |
| Build | `/sgl-workspace/bench-artifacts/dsv4-a20-fuse-window-build-20260526/build.log` | `cargo build --release -p infer --features cuda,nccl --bin infer` passed in 6m49s |
| Smoke baseline | `/sgl-workspace/bench-artifacts/dsv4-a20-window-baseline-smoke-max32-20260526` | `max_tokens=32`, elapsed 7.5398s, output `4062 0.0000 ...` |
| Smoke fused | `/sgl-workspace/bench-artifacts/dsv4-a20-window-fused-smoke-max32-20260526` | `max_tokens=32`, elapsed 7.5188s, byte-identical output |
| Nsys baseline | `/sgl-workspace/bench-artifacts/dsv4-a20-window-baseline-nsys-max32-20260526/nsys` | profile request 3.1348s; `dsv4_update_window_cache_kernel`: 9504 calls, 10.9566ms total GPU time |
| Nsys fused | `/sgl-workspace/bench-artifacts/dsv4-a20-window-fused-nsys-max32-20260526/nsys` | profile request 3.0272s; `dsv4_update_window_cache_kernel`: 0 calls |
| Default smoke | `/sgl-workspace/bench-artifacts/dsv4-a20-default-fuse-smoke-max32-20260526` | unset DSv4 backend/fuse env, env-check resolved `deepep` + `deepgemm`, elapsed 7.4941s, byte-identical output |

The hard pass condition is launch removal plus byte-identical greedy decode.
The single nsys profile-request wall-clock delta was -3.43%, but the trace also
showed wider top-kernel count movement than this one axis should explain. Treat
that wall-clock result as directional evidence, not a broad throughput claim.

## Rule

For launch-churn optimizations, document the exact removed kernel call count
and keep wall-clock claims scoped to the measured request. Do not extrapolate a
single nsys run into the 32K / 1.5K throughput target.
