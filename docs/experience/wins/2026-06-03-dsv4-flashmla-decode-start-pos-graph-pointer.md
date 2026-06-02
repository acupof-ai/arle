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

Follow-up tranche: batched decode now owns a stable `start_pos_gpu` scratch
buffer and uploads the current per-row decode positions before the decode
piece. The per-row attention loop passes each row's device pointer through the
attention path, so cached FlashMLA decode Q/K prep no longer has to restage a
host scalar per layer. Pure-SWA Q/K prep can also consume the pointer ABI.

This is a graph-enablement fix, not a performance win by itself. The route
still needs full batch metadata replay, DeepEP/NCCL capture/replay, and
frozen-KV EAGLE/MTP graph replay before it can target the 4.85ms TPOT baseline.

Local verification:

- `cargo fmt --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `git diff --check`

Remote verification on `/data01/build/arle` at commit
`232c41761173dc9fd1100fbaedf738624ee50009`:

- release-fast CUDA build passed in 7m01s, rebuilt the CUDA tree, and harvested
  fresh prebuilt artifacts.
- `libkernels_cuda.a` exports `dsv4_prepare_qk_start_pos_ptr_cuda`,
  `dsv4_prepare_qk_fused_start_pos_ptr_cuda`, and
  `arle_dsv4_output_inverse_rope_start_pos_ptr_cuda`.
- TP8 + EAGLE SGLang-best-practice startup probe still fails closed, as
  expected, on remaining full graph blockers: decode is still `piecewise`,
  DeepEP/NCCL capture/replay is missing, frozen-KV EAGLE/MTP graph replay is
  missing, FlashMLA/SWA/C4/C128 metadata replay is missing, and attention still
  has per-row host-side planning.

Artifacts:

- `/tmp/dsv4_graph_start_pos_20260603/build.log`
- `/tmp/dsv4_graph_start_pos_20260603/startup_tp8_eagle.log`

## Rule

For DSv4 CUDA graph work, any per-step decode scalar consumed by a captured
kernel must either be graph-invariant or come from a stable device pointer that
is refreshed before replay. Do not mark DSv4 decode as `full_decode` while host
scalar metadata is still baked into captured kernels.
