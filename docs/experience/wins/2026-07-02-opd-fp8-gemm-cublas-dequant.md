# OPD writeback 8.5× — FP8 frozen-base GEMMs were a naive hand-rolled kernel

## Context

Agent-OPD masked-writeback forward on 27B Qwen3.6-FP8 (`--share-frozen-base`,
8×H20) took 122s; per-layer profile (`ARLE_OPD_PROFILE=1`, 137ffb28) showed
every layer uniformly slow — linear-attn ~1732ms, full-attn ~2423ms — pointing
at a shared per-op cost, not one hot op. MoE was a false lead (27B is dense).
Code reading alone found it: `CudaBackend::matmul_bt` routed any
`CudaFp8BlockScaled` weight to `fp8_block_scaled_matmul_bt_f32` — one output
element per block, software `ldexpf` FP8 decode per weight element, no tiling,
no tensor cores (~0.4 TFLOPS vs cuBLAS bf16 ~150 TFLOPS ≈ the observed ~290×).
Every projection + MLP GEMM in all 64 layers took this path; backward
grad-input took the sibling `fp8_block_scaled_matmul_f32`.

## What Worked

Dequantize the FP8 block-scaled weight to bf16 on device (one memory-bound
elementwise kernel, ~0.1ms per 27B weight) and delegate to the existing
`matmul[_bt]_device_f32_bf16` cuBLAS tensor-core path. Both naive GEMM kernels
and the orphaned `launch_2d` helper deleted (net −181 LOC); no shape/model
guards — any FP8 block-scaled weight on any model qualifies. Commit `270a509e`.

Same toy config A/B (run-fp8dq-toy1r vs run-profile3-toy1r, same pod, same
task, GPU 1, RUN_EXIT=0, zero NVRTC/runtime errors):

| metric | 137ffb28 baseline | 270a509e | Δ |
|---|---|---|---|
| forward_hidden_states | 122.119s | **14.416s** | **8.5×** |
| forward layers sum | 121.807s | 13.496s | 9.0× |
| linear-attn layer wall | ~1732ms | **25–27ms** | ~68× |
| full-attn layer wall | ~2423–2453ms | 757–838ms | 3.2× |
| backward | 149.100s | **38.003s** | 3.9× |
| loss (toy round) | 0.327482 (targets=120) | 0.279333 (targets=122) | in 0.24–0.33 band |
| whole run wall | 5m35s | 2m41s | 2.1× |
| VRAM post_backward | 36,079 MiB | 39,535 MiB | +3.5 GiB (transient bf16 copies; trims to 36,047 post-round) |

Next wall (hypothesis, unmeasured): full-attn layers still ~760ms vs linear
~26ms — 16 full-attn layers ≈ 12.2s of the remaining 13.5s forward; the
full-attention op itself is the residual target. VRAM headroom worth watching
at larger writeback windows.

## Rule

- A uniform per-layer slowdown means a shared substrate op, not a model op —
  audit the backend dispatch (`matmul` handle-type branches) before any layer
  math. The 290× "roofline gap" was one `if let DeviceHandle::… = b` branch.
- Quantized-weight GEMM in autograd: dequant-to-bf16 + vendored GEMM beats any
  hand-rolled fused-dequant kernel until a measured A/B licenses otherwise
  (先用最好的再自己写).
