# DSv4 Small Local Pack Prepare Nsight Trace

This artifact profiles one generated decode token on the real 8xH20
`/root/DeepSeek-V4-Flash` checkpoint after fusing the B=1 padded DeepEP local
expert preparation step. The fused CUDA kernel clears local expert counts,
counts received route metadata, writes device-side local offsets, and clears
pack cursors in one small-kernel launch.

The request was:

```text
Compute 137 + 269. Answer with the number only.
```

The HTTP response was `406`.

Compared with
[`../nsys-single-decode-token-moe-scratch-uninit-rerun/`](../nsys-single-decode-token-moe-scratch-uninit-rerun/):

- Host-to-device runtime calls drop from 1,040 to 696 by removing the per-layer
  local-offset H2D copy.
- `cuMemsetD8Async` drops from 1,232 calls / 1.558 ms per rank range to 544
  calls / 0.728 ms by folding the local count/cursor clears into the prepare
  kernel.
- H2D activity drops to 696 calls / 12,416 bytes.
- The single captured decode wave is 92.602 ms. This capture is not recorded
  as a wall-time win because D2H synchronization and AllReduce timing were
  noisier than the prior rerun.

The remaining slow stack is unchanged:

- `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL`: 20.386 ms per rank range.
- Local expert GEMV: FP8 11.474 ms plus FP4 11.107 ms per rank range.
- D2H synchronization: 344 calls / 44,032 bytes of activity, but
  19.480 ms of CUDA runtime time in this sample.
- Attention/MHC/route kernels: 7.393 ms, 5.503 ms, and 5.659 ms.

This is a small-call cleanup. It reduces per-layer host-side launch/copy
pressure, but the next material decode target remains eliminating the local
count readback and replacing the local expert GEMV loop with real grouped
GEMM/DeepGEMM plus better DeepEP combine behavior.
