# DSv4 Prefill D2H Bottleneck Diagram Notes

Output:

- `/path/to/code/agent-infer/diagrams/dsv4-prefill-d2h-bottleneck.png`
- `/path/to/code/agent-infer/diagrams/dsv4-prefill-d2h-bottleneck.svg`

Key numbers shown in the diagram:

- `step_prefill_kernel_launch`: 8 NVTX instances, each about `35.7s`, for about `286s` rank aggregate.
- Local MoE fallback reads `local_counts[32]` from GPU to CPU with `clone_dtoh(local_counts)`.
- `local_counts[32]` is `32 x i32 = 128B`.
- `43` DSv4 layers across `8` ranks gives `43 x 8 = 344` tiny count readbacks.
- nsys CUDA API accounting inside prefill showed about `269s` rank aggregate in `cuMemcpyDtoHAsync_v2`, while actual GPU D2H memcpy device work was only about `25ms` in the full trace.
- This identifies synchronization/serialization as the bottleneck, not D2H bandwidth.

Recommended direction:

- Keep counts, offsets, dispatch metadata, and compact buffers on the GPU side.
- Prefer DeepEP-style all-to-all dispatch/combine, route-grouped expert execution, or device-side compact/prefix-sum.
- Avoid per-layer host readback in the steady-state prefill path.
