# DSv4 A2.1 Fuses Q/K Prep Launches

## Context

After A2.0 removed the standalone decode window-cache update launch, the
current DSv4 DeepEP + DeepGEMM-auto decode trace still showed separate Q and K
prepare launches for each attention layer. This A2.1 substep keeps the same
math and expert path, but fuses Q RMSNorm+RoPE prep and K RoPE prep into one
CUDA kernel launch. It does not claim to complete the full FlashMLA / CSA
fusion axis.

## What Worked

`dsv4_prepare_qk_fused_cuda` now launches one heterogeneous row kernel for Q
rows plus one K row per token. The older two-launch path is still available via
`ARLE_DSV4_FUSE_QK_PREP=0`; unset defaults to the fused path.

Remote H20 validation artifacts:

| Run | Artifact | Result |
| --- | --- | --- |
| Build | `/sgl-workspace/bench-artifacts/dsv4-a21-fuse-qk-build-20260526/build.log` | `cargo build --release -p infer --features cuda,nccl --bin infer` passed in 6m52s |
| Smoke split | `/sgl-workspace/bench-artifacts/dsv4-a21-qk-split-smoke-max32-20260526` | `max_tokens=32`, elapsed 7.5552s, output `4062 0.0000 ...` |
| Smoke fused | `/sgl-workspace/bench-artifacts/dsv4-a21-qk-fused-smoke-max32-20260526` | `max_tokens=32`, elapsed 7.5016s, byte-identical output |
| Smoke default | `/sgl-workspace/bench-artifacts/dsv4-a21-qk-default-smoke-max32-20260526` | unset DSv4 backend/fuse env, env-check resolved `deepep` + `deepgemm`, elapsed 7.5169s, byte-identical output |
| Nsys split | `/sgl-workspace/bench-artifacts/dsv4-a21-qk-split-nsys-max32-20260526/nsys` | profile request 3.1086s; `dsv4_prepare_q_kernel`: 7352 calls / 23.8502ms; `dsv4_prepare_k_kernel`: 7342 calls / 19.3829ms |
| Nsys fused | `/sgl-workspace/bench-artifacts/dsv4-a21-qk-fused-nsys-max32-20260526/nsys` | profile request 3.0231s; `dsv4_prepare_qk_fused_kernel`: 7290 calls / 24.2723ms |

`cudaLaunchKernel` runtime calls dropped from 490244 to 479574 in the filtered
decode trace. The single profile request improved by 2.75%. The steady-wave
average excluding the first wave was 32.9829ms split vs 33.1648ms fused, so the
wall-clock claim stays scoped to this profile request and launch-churn evidence.

## Rule

Launch fusion needs both correctness and direct kernel-count evidence. If the
steady-wave framing is mixed, keep the claim local and continue searching for
larger wall-clock levers instead of overstating a substep.
