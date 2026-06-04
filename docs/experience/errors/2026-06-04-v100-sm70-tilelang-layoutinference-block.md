# Rewrite TileLang paged-attention kernels fail LayoutInference on V100 / sm_70

**Status:** BLOCKED — the rewrite's current TileLang attention kernels do not build
on Volta (sm_70). V100 verification of the rewrite Qwen path is not possible until
fixed; Qwen verification is redirected to H20 (sm_90, where the kernels build).

## Context

V100 (Tesla V100-SXM2-32GB, sm_70) was offered as a parallel verification SKU for
the rewrite Qwen path. Target: `mlx-community`/HF `Qwen/Qwen3.5-4B` (HYBRID:
`Qwen3_5ForConditionalGeneration`, 32 layers, `full_attention_interval=4` →
full-attn HD256 q16/kv4 GQA every 4th layer, gated-delta linear elsewhere, DENSE
MLP — no MoE). A background agent recovered the full V100 build env and got the
build through cuda-kernels C compilation into TileLang AOT before hitting the wall.

## Root Cause

TileLang/TVM `LayoutInference` conflict in the GemmFMA-fallback online-softmax
rescale of the paged-attention kernels on sm_70:

```
Layout infer conflict between m_new and scale_i in T.Parallel loop:
  loop     Fragment((64,)->(64,), replicate:128, thread:128, forward_thread:_rep)
  fragment Fragment((64,)->(16,), replicate:32,  thread:128, forward_thread:_i%4*32+_rep)
```

A standalone read-only `gen_tilelang_aot.py` probe confirmed it hits **all three**
relevant shapes on sm_70: `hd128_prefill_q32_kv8` (aborts the cargo build first),
**and** `hd256_prefill_q16_kv4` + `hd256_decode_q16_kv4` — the exact kernels
Qwen3.5-4B's full-attention layers require. So it is not an unused-kernel problem;
dropping HD128 from the sm_70 allowlist just moves the failure to the HD256 q16/kv4
kernel the model needs. The same kernel `.py` lowers fine on H20 (sm_90), so it is
sm_70-specific to the GemmFMA fallback.

The prior V100 Qwen3.5 wins (2026-05-25..29) **predate** `1d6b7836 refactor(cuda):
migrate runtime kernels to tilelang` — i.e. the rewrite's *current* TileLang
attention kernels have never successfully built on Volta. This is genuinely new and
currently non-functional on sm_70.

## Fix (deferred — not yet done)

Out of scope for a verification pass; both options are `crates/cuda-kernels`
TileLang work:
1. Fix the sm_70 GemmFMA-fallback `LayoutInference` for the online-softmax rescale
   (`scale_i` fragment `replicate:32, forward_thread:_i%4*32+_rep` vs `m_new`
   `replicate:128`) — the real fix; covers every HD128/HD256 paged-attn shape on Volta.
2. Bump/re-patch TileLang past `14489d9d` if a newer sm_70 path resolves it.

**Decision:** V100/sm_70 is a legacy-Volta support tier; the rewrite's primary CUDA
target is H20/sm_90 (R6 + DSv4 verified there). Qwen3.5 hybrid greedy parity is
redirected to H20 via `agent-bench cuda_qwen35_greedy_parity`.

## Rule

- **A "parallel verification SKU" is only usable if the target builds there.**
  Confirm the kernel set builds on the SKU before counting it as a verification
  lane — sm_70 GemmFMA fallback ≠ sm_90 WGMMA path, and a kernel-migration commit
  can silently break the older arch.
- **Recovered V100 build env (for the future fix):** TileLang venv
  `~/tilelang/.venv` loading the sm_70-patched dev-root `~/tilelang-sm70-copy/build`
  (TileLang 0.1.10+cuda.git14489d9d), CUDA 12.4 at `/usr/local/cuda` (not the
  login-shell nvcc 11.8), `TORCH_CUDA_ARCH_LIST=7.0`, and **`ARLE_CUDA_DISABLE_FLASHMLA=1`
  is REQUIRED** (vendor/flashmla static_asserts SM80+ / sm_90 cluster launch-bounds).
