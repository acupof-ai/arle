# DSv4 FlashMLA Decode Uses Graph-Safe Start Position Pointers

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44s, TPOT ~4.85ms, E2E ~7.7s, output throughput ~196 tok/s.

ARLE is not at that target yet. The high-performance route still fails closed
on full decode CUDA graph, MTP graph replay, and DeepEP/NCCL capture/replay.

## What Worked

FlashMLA decode already used a device `start_pos` pointer for SW pack, sparse
indices, and window update. Q/K prep and FlashMLA output inverse-rope still
took a host scalar `start_pos`, which would bake the capture-time position into
future graph replays.

This tranche keeps the scalar ABI for existing prefill and legacy paths, and
adds pointer ABI variants for:

- `dsv4_prepare_qk_start_pos_ptr_cuda`
- `dsv4_prepare_qk_fused_start_pos_ptr_cuda`
- `arle_dsv4_output_inverse_rope_start_pos_ptr_cuda`

The Rust FlashMLA decode gate now stages `start_pos` into the stable
per-layer `fm_decode_start_pos` device slot before Q/K prep, then reuses the
same pointer for Q/K prep, SW pack, sparse indices, output inverse-rope, and
window update.

Local verification:

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `git diff --check`

Remote nvcc build and DSv4 startup/correctness probe are pending for the next
step because this change adds CUDA symbols and intentionally invalidates older
prebuilt archives.

## Rule

For DSv4 CUDA graph work, any per-step decode scalar consumed by a captured
kernel must either be graph-invariant or come from a stable device pointer that
is refreshed before replay. Do not mark DSv4 decode as `full_decode` while host
scalar metadata is still baked into captured kernels.
