# GLM-5.2 → DSv4 V32 — Tranche D forward (pending-remote)

> **pending-remote**: runtime forward change in `crates/infer-cuda/src/` +
> `crates/cuda-kernels/csrc/` (CUDA-only). Cannot bench on a Mac (no nvcc/GPU).
> GPU validation is the E-tranche on the pod; this entry is the bench-rule stub.
> Cross-links: `docs/plans/2026-06-18-glm52-dsv4-port.md`.

## SLO-shape probed?  N — GPU-only, no local bench

GLM-5.2 (`glm_moe_dsa`) forward is CUDA + sm_90 (V32 FlashMLA) only; no Metal/CPU
lane. The Tranche-E pod validation (truncated GLM-5.2 on 8×H20, load→forward→
`needle_gate` x3 vs envelope, self-consistency NOT byte-identity) carries the
correctness + perf license. No default flag flips here; DSv4 path byte-unchanged.

## Goal

- Land Tranche D (forward) of the GLM-5.2→DSv4 V32 port: GLM's SparseIndexed MLA
  forward runs end-to-end structurally; the 5 `unimplemented!("Tranche D")` arms
  are replaced; the V32 FP8 pack kernel + bf16 MoE lane are wired; tree typechecks.

## What landed (D0–D5)

- **D0 — GLM MoE bf16 lane.** GLM `weight_scale_inv` blocks are general F32
  (non-pow2); C's E8M0 re-encode was lossy. Routed experts now dequant→bf16 and
  ride `deepgemm_m_grouped_bf16_gemm_nt_contiguous` (gate/up as separate grouped
  caches + `silu_mul` + w2, the proven Qwen3.6 bf16 MoE structure). Shared expert
  kept on E8M0 (1 expert; avoids an ~18-site `Option` ripple that would risk DSv4
  byte-identity — documented fallback). DSv4 FP8 DeepGEMM path byte-unchanged.
- **D1 — V32 FP8 pack kernel.** `dsv4_fp8_kv_pack_kernel` templatized on
  `HEAD_DIM_NOPE` (448 MODEL1 / 512 V32 → NUM_TILES 7/8, NUM_SCALES 8/16,
  TOKEN_BYTES 584/656). E4M3-max=448 scale math unchanged. New
  `arle_dsv4_v32_fp8_kv_pack_strided_cuda` FFI + Rust wrapper
  (`dsv4_v32_fp8_kv_pack_strided_raw`) + prebuilt-symbol allowlist entry. FFI
  signature ↔ `.cu` verified consistent by inspection (Mac no-cuda skips the `.cu`).
- **D2 — SparseIndexed (5 arms).** SparseIndexed = CompressedSparse MINUS the
  compressor (indexer over the full latent, ratio=1, capped by `index_topk=2048`):
  `flashmla_mode_int`→1 (CSA-style index build), decode/prefill `max_compressed_keys`
  →`index_topk`, prefill build-indices mirrors CSA at ratio=1, prefill mode_int→1.
- **D3 — runtime w_kc/w_vc absorption.** Per SGLang `forward_mla.py`: pre-decode
  `q_latent[h]=w_kc[h]·q_nope[h]` (concat q_rope→576), post-decode
  `v[h]=w_vc[h]·attn_out[h]` (512 latent→256). Loader emits w_kc/w_vc in
  `gemm_batch` orientation `[out,in]`; `glm_absorb_q`/`glm_absorb_v` run a per-head
  `gemm_cuda` loop. **Decode (token_count==1) wired exact**; prefill (>1) bails
  loudly (needs a batched-head GEMM). V32 decode arg-mapping: d_qk=head_dim(576),
  d_v=512 (shim hard-asserts d_v==512), strides + model_type/bytes match-bound.
- **D4 — plain-o.** `mla_oproj` early-returns a single `dsv4_linear(o_proj,…)` when
  `attention.o_proj.is_some()` (GLM), skipping wo_a/wo_b/group tables.
- **D5 — dense FFN + hc-bypass.** `per_layer_dense_mlp[i]` layers run a plain
  SwiGLU FFN (`dsv4_dense_mlp_forward`) instead of MoE. `hc_mult==1` (GLM) bypasses
  ALL hyper-connection machinery (`initial_stream_from_embeddings`/`hc_pre`/
  `hc_post`/`head_hidden_from_stream`) with plain residual + identity stream.
- **Blockers handled.** GLM `sliding_window==0` (pure SparseIndexed, no SW ring):
  SW-ring pack/attention gated on `sliding_window>0 || mode==SparseIndexed`.
  `local_heads` derived from `qk_head`(256) for GLM (wq_b is pre-absorption) vs
  `head_dim`(576) for DSv4. `DecodeShape::new` + the batched-decode lane accept/
  guard V32 (batched lane is MODEL1-only; V32 routes single-row).

## DSv4 byte-identity

Every GLM branch gates on a GLM marker (`plain_o_proj` / `hc_mult==1` /
`o_proj`/`w_kc`/`w_vc`.is_some() / `dense_mlp`.is_some() / `mode==SparseIndexed` /
`head_dim==576` / `GroupedCache.is_bf16`). DSv4 (MODEL1) arms unchanged; the V32
decode path collapses to byte-identical MODEL1 when `is_v32==false`.

## Verification (Mac, no-cuda)

```text
CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib      → PASS
CUDARC_CUDA_VERSION=12080 cargo clippy -p infer-cuda ...   → clean (0 warnings)
cargo test -p deepseek-spec                                → 9 passed, 3 ignored
```

The `.cu` (D1) compiles only on the GPU host (no-cuda skips it); FFI↔`.cu`
signature consistency checked by inspection.

## Honestly deferred (pod / follow-up)

- **GLM prefill Q/V absorption** (token_count>1): bails loudly — needs a
  batched-head bf16 GEMM (token-major rows are strided per head). Decode exact.
- **26 `// ponytail: pod-verify`** byte-layout/contraction points across the V32
  pack offsets, decode arg-mapping, SparseIndexed index build, absorption
  contraction, dense-FFN activation, and hc-bypass — each implemented faithfully,
  to be confirmed on a pod forward (Tranche E).
- Shared-expert E8M0 re-encode precision; FP8 grouped GEMM with F32 scales (perf).

## Rule

GLM-5.2 = DeepSeek-V3.2-DSA family on the vendored FlashMLA V32 path; the port is
runtime wiring + load-time absorption + structural branches, not a new kernel.
Decode is the wired hot path; prefix-absorption + all byte-layouts gate on pod
validation (Tranche E). Never fake correctness — bail loudly where a contraction
can't be GPU-verified.
