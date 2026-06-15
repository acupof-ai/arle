# DSv4 pre-P4 quantized GEMV baseline — byte parity captured

## Context

Before changing `crates/cuda-kernels/csrc/gemm/quantized_gemv.cu` for the CUDA
quant subsystem P4, capture the DSv4 E8M0 path reference. This is the guard that
the later scale-ABI parameterization does not perturb the existing DSv4
instantiations.

Remote source snapshot: local `main` at `557611d9`, synchronized to
`/sgl-workspace/arle` in the H20 pod.

Build note: the pod's offline cargo index lacked `image 0.25.10`, so the remote
validation-only `Cargo.lock` was temporarily downgraded to cached `image 0.25.9`
and related image transitive crates. The built target is `infer-cuda` only; those
`infer-server` image crates are not compiled for the DSv4 parity example.

## What Worked

Build:

```bash
ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1 \
ARLE_CUDA_KERNEL_SET=dsv4_flash \
ARLE_DSV4_MODEL_PATH=/data01/models/DeepSeek-V4-Flash \
ARLE_DEEPGEMM_ROOT=/tmp/DeepGEMM \
ARLE_DEEPGEMM_LIBRARY_ROOT=/tmp/DeepGEMM/deep_gemm \
ARLE_DEEPGEMM_CUTLASS_INCLUDE=/tmp/DeepGEMM/third-party/cutlass/include \
CUDARC_CUDA_VERSION=12060 \
cargo build --offline --release -p infer-cuda --features cuda,nccl,deepep \
  --example dsv4_parity
```

Run:

```bash
INFER_DSV4_MODEL_PATH=/data01/models/DeepSeek-V4-Flash \
INFER_DSV4_BATCH_DECODE_VALIDATE=2,4 \
INFER_DSV4_MAX_NEW=8 \
scripts/dsv4_multigpu_parity.sh
```

Rank-0 reference:

```text
batch_decode_reference batch=1 max_new=8 clean_tokens=[11111, 603, 671, 6102, 294, 8760, 344, 11111]
batch_decode_reference_rerun clean_tokens=[11111, 603, 671, 6102, 294, 8760, 344, 11111] ref_self_parity=true ref_self_first_div=None
batch_decode_validate batch=2 byte_parity=true
  row0_first_div_vs_ref=None clean_tokens=[11111, 603, 671, 6102, 294, 8760, 344, 11111]
  row1_first_div_vs_ref=None clean_tokens=[11111, 603, 671, 6102, 294, 8760, 344, 11111]
batch_decode_validate batch=4 byte_parity=true
  row0_first_div_vs_ref=None clean_tokens=[11111, 603, 671, 6102, 294, 8760, 344, 11111]
  row1_first_div_vs_ref=None clean_tokens=[11111, 603, 671, 6102, 294, 8760, 344, 11111]
  row2_first_div_vs_ref=None clean_tokens=[11111, 603, 671, 6102, 294, 8760, 344, 11111]
  row3_first_div_vs_ref=None clean_tokens=[11111, 603, 671, 6102, 294, 8760, 344, 11111]
```

## Rule

Any P4 edit that parameterizes `quantized_gemv.cu` scale decoding must preserve
this DSv4 reference exactly for batch=1/2/4. If the post-P4 run differs, the
scale-ABI refactor is rejected before Qwen FP8/NVFP4 wiring proceeds.
