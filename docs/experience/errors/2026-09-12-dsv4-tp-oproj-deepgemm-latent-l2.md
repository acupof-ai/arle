# DSv4 TP oproj parity: DeepGEMM wo_a latent l2=1.0 (pending-remote)

Date: 2026-09-12. Gate: `dsv4_tp_oproj_parity` (registry parity GPU batch,
2026-09-11 first run). Status: **pending-remote** — code-read triage
complete; the deciding GPU run waits for a released H20 (V4.1 holds
GPUs 0-5).

## Context

First-run symptom (results row): head o-slice exact and
`[tp=1 s=1 gemv] PASS latent+partial-sum in bound (l2=0.000e0)`, then
`[tp=1 s=1 deepgemm] rank=0 latent l2=1.000e0` and
`Error: tp=1 s=1 deepgemm: wo_a latent FAILED`. The negative-control arm
was not reached. l2=1.0 means the DeepGEMM latent is orthogonal to the
oracle (zero matching output), which per-128 e4m3 quantization error
cannot produce — it indicates a layout or construction mismatch, not
precision.

The gate geometry (`crates/infer-cuda/examples/dsv4_tp_oproj_parity.rs`):
W=32768, H=4096, G=8 groups, C=1024 o-rank, WG=H=4096, BLK=128. The wo_a
weight is per-128-block e4m3 with e8m0 power-of-two scales. For TP=1 the
DeepGEMM lane runs the grouped path (G>1): for each group
`dsv4_oproj_group_gather_cuda` gathers the group's [s,WG] columns,
`dsv4_deepgemm_pack_quantize_bf16_to_fp8` quantizes the activation,
`dsv4_deepgemm_fp8_gemm_nt` runs against the per-group resident
`Dsv4Fp8DeepGemmWeightCache`, and `dsv4_oproj_group_scatter_cuda` writes
the group's [s,C] block into the latent. The f64 oracle models the
direct e4m3·decode(e8m0) weight, per element.

## Root cause

Cause unknown; code reading does not pin a side. The layout audit
matches the oracle and the passing scalar GEMV lane at every step:

- Group gather/scatter byte addressing
  (`crates/cuda-kernels/csrc/attention/dsv4_oproj.cu:15,31`) maps group
  `g` to offset `(row*groups+g)*cols + col`, the same [s,G,C] ordering
  the oracle uses (`dsv4_tp_oproj_parity.rs:257`).
- Activation scales SFA are emitted column-major
  (`expert*stride*k_blocks + k_block*stride + row`,
  `dsv4_deepgemm_ops.cu:53`) and the dense TMA descriptor declares the
  same 2-D scale layout with `outer_stride=sfa_aligned_m`
  (`deepgemm_native.cu:921-935`).
- Weight scales SFB: the dense kernel is instantiated with
  `kMajorSFB = cute::UMMA::Major::K` (`deepgemm_native.cu:1275`), so it
  indexes `sfb[(n/128)*k_blocks + k_block]`, exactly the resident
  cache's row-major `[ceil(n/128), ceil(k/128)]` FP32 layout
  (`dsv4_fp8_cache.cu:135`).
- The conversion kernel `dsv4_block_scaled_to_fp8_deepgemm_cuda`
  decodes e8m0 as `bits<<23` (`dsv4_fp8_cache.cu:34-37`), takes a
  per-128×128 amax/448 re-quant scale, and writes weight rows
  row-major, matching the oracle decode and the SFB index above.
- Scratch strides (`DG_STRIDE=128`, scales `128*k_max/128`,
  `dsv4_tp_oproj_parity.rs:337-346`) match the pack kernel's
  `max_m`/`scale_stride_m` and the dense gemm's `sfa_aligned_m >= m`.

The one structural difference between the failing path and the only
passing dense-NT coverage (`fp8_smallm_gemm_probe.rs`): the probe feeds
native e4m3 weights plus FP32 scales straight into
`dsv4_deepgemm_fp8_gemm_nt`; the oproj gate is the only dense-NT
consumer that builds its cache through the re-quantization kernel
`dsv4_block_scaled_to_fp8_deepgemm_cuda` (`from_dsv4_weight`). The
gate's own f64 oracle models the direct representation, not the
re-quantized cache.

Suspected: the re-quantization resident-cache path produces the l2=1.0
output. Refuted if the direct-cache arm below still returns l2≈1.0; in
that case the fault is in the dense DeepGEMM plumbing at this
m=1, n=1024, k=4096 grouped-staging shape rather than in the conversion
kernel.

## Fix (pending-remote)

A diagnostic arm, not a production change: `--direct-cache` on
`dsv4_tp_oproj_parity` builds the wo_a resident cache from the native
e4m3 bytes with e8m0 decoded to FP32 on the host
(`DeviceMatrix::from_fp8_block_scaled` +
`Dsv4Fp8DeepGemmWeightCache::from_fp8_block_scaled_weight`, a D2D copy),
bypassing `dsv4_block_scaled_to_fp8_deepgemm_cuda`. It runs only the
clean DeepGEMM lane against the existing f64 oracle. Interpretation:

- direct-cache PASS, default FAIL → fix the conversion kernel.
- both FAIL → fix the dense DeepGEMM grouped-staging plumbing; add a
  smaller probe before changing vendor GEMM.

Lane: `tpoproj-direct-cache-arm` (one example-only commit). No runtime
code under `crates/{infer-*,cuda-kernels}/src` or `csrc/` changes, so
no before/after kernel measurement is possible until the arm identifies
a kernel side; this entry records the gate failure and the pending
decision run, per the runtime-change bench rule.

### Rerun command (once a GPU is released)

```
INFER_CUDA_DEVICE=<free-h20> \
  target/release/examples/dsv4_tp_oproj_parity --direct-cache
# then the default lane for the control:
INFER_CUDA_DEVICE=<free-h20> \
  target/release/examples/dsv4_tp_oproj_parity
```

Needle (first failing cell, from the 2026-09-11 batch):
`[tp=1 s=1 deepgemm] rank=0 latent l2=1.000e0`.

Lever: the only changed bytes between the two runs are the wo_a cache
construction (`from_fp8_block_scaled_weight` vs `from_dsv4_weight`);
everything else (activation pack-quantize, gather/scatter, gemm launch,
oracle) is identical.

## Rule

When every layout in a vendor-GEMM wrapper matches by inspection and the
only passing gate bypasses one specific transform, add a diagnostic arm
that bypasses that same transform and let one GPU run decide the side
instead of guessing between gate and kernel.
