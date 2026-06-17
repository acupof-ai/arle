# Qwen3.6 BF16 small-N GEMM SIGFPE fix

## Context

While rechecking Qwen3.6-35B-A3B BF16 inference on `.62`, the run path crashed
after the initial prompt prefill, before a coherence verdict could be trusted.
The last runtime marker before the crash was the first layer-0 dense BF16 GEMM
of a short continuation chunk:

```text
[qwen-gemm-profile] format=dense_bf16 M=8192 N=14 K=2048
Floating point exception
```

The earlier linear-attention hypothesis was killed by an env-gated layer-0 dump:
`in_proj_qkv`, `conv1d_silu_qkv`, `gdr_out`, `gated_norm_out`, and `out_proj`
were finite with sane magnitudes. The crash was therefore below the model math,
in the dense BF16 GEMM call.

## Root Cause

`INFER_DETERMINISTIC=1` did not avoid the crash, so this was not only the
cuBLASLt fast path. A gdb run on the same shape showed the host SIGFPE inside
CUDA 12.9 cuBLAS/Lt heuristic code reached through the cuBLAS fallback:

```text
log=/data01/arle-f2-probes/qwen36_bf16_gdb_fpe.log

Thread 2 "infer-engine" received signal SIGFPE, Arithmetic exception.
#0  libcublasLt.so.12
#6  cublasLtTSTMatmulAlgoGetHeuristic
#13 cublasGemmEx
```

The measured failing shape was `M=8192,N=14,K=2048`. Larger prefill chunks
(`N=64`) ran normally; the failure is a small/odd-N host-side cuBLAS heuristic
bug on this CUDA stack, not a Qwen linear-attention or MoE math bug.

## Fix

`gemm_cublaslt_impl` now routes small dense BF16 GEMM batches through the
existing handwritten BF16 GEMV kernel, one column at a time, before touching any
cuBLAS handle:

```text
N > 0 && N <= 16 && K*sizeof(bf16) <= 48 KiB -> handwritten GEMV loop
otherwise -> existing cuBLAS/Lt path
```

This keeps the large-N prefill path unchanged and gives small continuation
chunks a simple cuBLAS-free fallback.

## Evidence

Local typecheck:

```text
CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib
PASS
```

Remote `.62` build:

```text
cd /data01/arle-build
CUDARC_CUDA_VERSION=12090 CUDA_HOME=/usr/local/cuda \
ARLE_CUDA_DISABLE_FLASHMLA=1 INFER_TILELANG_PYTHON=/root/tl-venv/bin/python \
cargo build --release --features cuda -p agent-infer --bin arle
PASS
```

Remote `.62` after-fix repro, same model and same prompt:

```text
model=/data01/models/Qwen3.6-35B-A3B
CUDA_VISIBLE_DEVICES=2
INFER_DETERMINISTIC=1
ARLE_CUDA_DISABLE_FLASHMLA=1
ARLE_QWEN35_DEEPGEMM=0
log=/data01/arle-f2-probes/qwen36_bf16_after_smalln.log

STATUS=0
```

The after-fix log reached the formerly crashing `N=14` dense BF16 GEMMs across
the layer stack, then continued into real `seq=1` decode and exited normally:

```text
[qwen-gemm-profile] format=dense_bf16 M=8192 N=14 K=2048
[qwen-gemm-profile] format=dense_bf16 M=4096 N=14 K=2048
...
[qwen-layer-profile] qwen/forward_hidden layer=na seq=1 cuda_ms=25.566 host_ms=25.565
[qwen-layer-profile] qwen/sample layer=na seq=1 cuda_ms=0.042 host_ms=0.041
```

After removing the temporary diagnostic dump source and rebuilding the clean
binary, the same repro still passed:

```text
log=/data01/arle-f2-probes/qwen36_bf16_after_smalln_clean.log
STATUS=0
grep 'Floating point exception|SIGFPE|received signal' -> no matches

[qwen-gemm-profile] format=dense_bf16 M=8192 N=14 K=2048
...
[qwen-layer-profile] qwen/forward_hidden layer=na seq=1 cuda_ms=25.635 host_ms=25.635
[qwen-layer-profile] qwen/sample layer=na seq=1 cuda_ms=0.041 host_ms=0.040
```

Follow-up HTTP chat gate showed the boundary matters: the first chat request
crashed at exactly `N=16` with the same libcublasLt divide-error class. The
fallback threshold was therefore tightened to include `N=16`, still leaving the
normal `N=64` chunked-prefill path on cuBLAS/Lt:

```text
log=/data01/arle-f2-probes/f2_bf16_profile_serve.log

[qwen-layer-profile] qwen/embedding layer=na seq=16 cuda_ms=0.195 host_ms=0.192
[qwen-layer-profile] qwen/input_norm layer=0 seq=16 cuda_ms=0.168 host_ms=0.163
[qwen-gemm-profile] format=dense_bf16 M=8192 N=16 K=2048
```

## Rule

Do not assume `cublasGemmEx` is a safe fallback for every small batched GEMM
shape: on CUDA 12.9 it can still enter cuBLASLt heuristic code and SIGFPE on
small/odd-N BF16 shapes. For tiny `N`, a simple handwritten GEMV loop is a
better correctness fallback than another cuBLAS API.
