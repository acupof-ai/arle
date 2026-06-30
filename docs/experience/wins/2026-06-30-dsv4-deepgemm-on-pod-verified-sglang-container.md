# DSv4 DeepGEMM-on serve verified on pod via the modern-glibc sglang container

## SLO-shape probed? — Partial (c=1 + c=4 steady-state tok/s on the real H20 pod; full guidellm sweep pending)

## Context

DSv4-Flash-FP8 TP=4 decode was running its FP8 projections on the scalar/BF16 path
(`gemv_handwritten` 53.4% of B=4 decode GPU time) because the DeepGEMM **runtime
JIT** could not compile in the serving environment. Root cause fully established
in [`errors/2026-06-30-dsv4-deepgemm-runtime-jit-gcc8-cxx20-fallback.md`](../errors/2026-06-30-dsv4-deepgemm-runtime-jit-gcc8-cxx20-fallback.md):
the `tn`-attached shell container is Debian-10 / glibc-2.28, too old for the
CUDA-12.9 + g++-13 JIT toolchain (nvcc cudafe++ mixes modern libstdc++ headers
with glibc-2.28 system headers → `pthread_cond_clockwait` undefined).

## What Worked

The node already runs a **modern-glibc container** as a k8s static pod:
`/etc/kubernetes/manifests/sglang-test.yaml` →
`iaas-gpu-cn-beijing.cr.volces.com/serving/sglang:v0.5.13.post1.iaas.nightly.202606171156-cu129`
(Ubuntu 24.04, **glibc 2.39**, **g++-13**, **nvcc 12.9**, privileged, all GPUs,
`hostPath /root` mounted at `/host`). Running arle **inside** it via
`crictl exec <sglang-container-id> bash -c '…'` gives DeepGEMM everything it needs
natively — no loader patching, and the binary's baked `library_root=/host/arle-build/...`
resolves correctly (because `/root` is mounted at `/host` there).

Launch (TP=4/EP=4, DeepGEMM expert backend, warm JIT cache):
```
crictl exec <id> bash -c 'CUDA_HOME=/usr/local/cuda \
  DG_JIT_CACHE_DIR=/host/deepgemm-warm \
  ARLE_DEEPGEMM_ROOT=/host/arle-build/crates/cuda-kernels/vendor/deepgemm \
  ARLE_DEEPGEMM_LIBRARY_ROOT=/host/arle-build/crates/cuda-kernels/vendor/deepgemm/deep_gemm \
  ARLE_DSV4_EXPERT_BACKEND=deepgemm ARLE_DSV4_MOE_BACKEND=allreduce \
  INFER_TP_SIZE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 INFER_DSV4_MAX_SEQ_LEN=16384 \
  /host/arle-build/target/release/arle serve --backend cuda \
    --model-path /host/DeepSeek-V4-Flash-FP8 --port 18195'
```

### Verified (H20 ×4, TP=4, DeepSeek-V4-Flash-FP8, DeepGEMM active)
- **Zero DeepGEMM fallback** in the serve log (no `disabled`/`preflight failed`/
  `CUDA_ERROR`/`fused wqkv failed`). The 113-cubin warm cache at
  `/host/deepgemm-warm` hit (0 fresh compiles), so even the JIT was bypassed.
- Inference correct: clean `/v1/chat/completions` responses, `/v1/stats` healthy.
- **c=1 steady-state: ~31 tok/s** (256 tok 30.7 tok/s; 512 tok 31.3 tok/s ≈ 32 ms/tok).
- **c=4 concurrent: ~53.8 tok/s aggregate** (256 tok × 4 in 19.0 s; per-stream ~14.9 tok/s),
  ≈ 74 ms/step — consistent with the commissioning B=4 profile (69.5 ms/step).
- VRAM ledger (TP=4): weights 73943 MB + adapter 20829 MB + 4 slots 668 MB =
  95441 MB on 97871 MB cards → only ~3.2 GB free, 4 slots. The adapter footprint is
  the binding constraint on slot count at TP=4 (matches the prior "TP=4 pressures
  the budget harder" note).

## Pending-remote / next

- This is the **DeepGEMM-ON baseline** for the kernel roadmap (P0 compressor/indexer
  proj → tensor-core, P1 RMS-norm+FP8-pack fusion). The runtime A/B vs the BF16
  fallback arm needs a build with a runtime `ARLE_DSV4_DECODE_PROJ_DEEPGEMM=0`
  override re-wired (the env opt-out is documented in code but no longer read —
  `dsv4_decode_proj_deepgemm_enabled` returns `has_deepgemm_native()` unconditionally).
- Full `guidellm` sweep (TTFT/ITL/throughput) not yet run; only c=1/c=4 chat-completions.

## Rule

- **DeepGEMM JIT needs a modern-glibc (≥2.30) + g++≥10 environment.** When the
  `tn` shell container is too old, run arle **inside the node's sglang static-pod
  container** via `crictl exec` — it already has CUDA 12.9 + g++-13 + glibc 2.39
  and mounts `/root` at `/host`. Don't fight the old container's toolchain.
- Confirm DeepGEMM is actually live (not silently BF16): serve log has no
  `DeepGEMM disabled`, and either fresh cubins land in `DG_JIT_CACHE_DIR` or the
  warm cache hits. "build log says enabled" ≠ "running".
- A real serve container is reproducible from `/etc/kubernetes/manifests/sglang-test.yaml`
  (static pod, kubelet-applied, no API server needed) — `/root/arle-mtp-pod.yaml`
  is a sibling debug-pod spec with the same image.
