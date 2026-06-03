# DSv4 → rewrite (infer-cuda) port plan — MoE/EP/DeepEP/DeepGEMM verification route

**Date:** 2026-06-04. **Status:** scoped (file:line-grounded), not started.
**Why:** the only cached MoE model is `/data01/models/DeepSeek-V4-Flash` (FP8/FP4) — the
canonical EP/DeepEP/**DeepGEMM** target. Verifying MoE+EP+DeepEP+DeepGEMM on the rewrite
means porting the legacy DSv4 path into `infer-cuda`. **Kernels are SHARED** (`crates/cuda-kernels`:
FlashMLA, DeepGEMM, DeepEP) — reusable as-is; this is a **rewrite-side orchestration port**,
but MLA attention is a genuinely new subsystem (not a GEMM swap).

## Pieces (effort, dependency)

| # | Piece | Effort | Into | Reuses (shared) | Net-new |
|---|---|---|---|---|---|
| 1 | DSv4 loader/config/KV pool | **L** | `loader.rs` `from_dsv4_fp8_safetensors` | `deepseek-spec` v4 config/tensor-names; `Dsv4Fp8DeepGemmWeightCache` (`cuda-kernels/tensor.rs:1090`); `from_dsv4_fp8_block_scaled` | FP8/FP4 load, expert DeepGEMM cache pack, MLA KV pool (kv=1, latent 512 / FP8 584B/tok), new weight structs, `MoeConfig::dsv4()`, `deepseek-spec`+`deepep-sys` Cargo deps |
| 2 | MLA attention (SW-only min) | **L** (full CSA/HCA/indexer/HC/MTP = XL) | `attention.rs` `mla_attention` | FlashMLA FFI (`cuda-kernels/ffi/misc.rs:413/455`), FP8 KV pack (`attention.rs:34/120`), `arle_dsv4_*` Q/K-prep+inverse-rope | Q-LoRA(wq_a→q_norm→wq_b), wkv/kv_norm, mode dispatch, FlashMLA launch+sched-meta, O-LoRA, inverse-rope; HC pre/post if in-scope |
| 3 | FP8 DeepGEMM MoE forward | **M** | `moe.rs` `moe_forward` | route/pack/scatter/combine already wired in `moe.rs`; `infer-moe::route` (DSv4-complete); DeepGEMM FFI (`ffi/gemm.rs:568-612`) | swap 2 `moe_bf16_grouped_gemm_*` → 5-call FP8 pipeline (pack_quantize→masked GEMM w13→swiglu_quantize→masked GEMM w2→unpad); FP8 scratch; **thread real gate bias** (route.rs `&[]`→bias for noaux_tc); DSv4 shared expert + routed_scaling_factor |
| 4 | DeepEP / EP multi-rank | **L** | new communicator + `moe.rs` | `deepep-sys` Buffer dispatch/combine; `ExpertSplit` (EP-aware); `infer-topo::build_moe_ep_groups` | EP NCCL group, `NativeDeepEp::boot`, dispatch→recv→FP8-GEMM→combine; `--features nccl` |

Legacy source: `infer/src/model/deepseek/mlp.rs` (DeepGEMM MoE: route `:5930`, FP8 pipeline
`:2810`/`:2960`-`:3090`; DeepEP `:5355`/`:3664`), `weights.rs` (MLA: `:3852`/`:4387`/`:4428`,
SW `:3400`, FlashMLA decode `:886`/`:1041`/`:1355`), `crates/deepseek-spec/src/v4.rs`,
`infer/src/native_deepep.rs`, `infer/src/model/layer_communicator.rs`.

## DSv4 is MULTI-GPU ONLY — TP=8/EP=8 from the start (no single-GPU path)

The 256 FP8 experts + MLA sharding do not fit one GPU; DSv4 only runs multi-rank
(legacy runs it TP=8/EP=8 on the 8×H20). So:
- **DeepEP/EP (Piece 4) is MANDATORY**, concurrent with the MoE forward — not a later add-on.
- **Full MLA (Piece 2 = XL)** — SW-only is not viable (config: `hc_mult=4`, CSA/HCA, 3 hash layers).
- A **multi-process launcher** (NCCL TP+EP groups + `NativeDeepEp::boot` + per-rank GPU bind +
  `ncclGetUniqueId`→hex broadcast + per-rank logit compare) is a **shared early prerequisite**
  for both DSv4 and TP=8 Qwen — build it first.
- Per-rank EP-aware FP8 expert load (`ExpertSplit::new(256, 8, rank)` → 32 experts/rank).
- Verify: **TP=8/EP=8 DSv4 greedy parity vs the legacy DSv4 path** (the oracle, also multi-GPU)
  on the cached DeepSeek-V4-Flash.

## Build flags (H20 pod; both legacy + rewrite link the same kernels)
- DeepGEMM: build `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1` + `ARLE_DEEPGEMM_ROOT` (else stubbed/no-op — `build.rs:1761`). Runtime `dsv4_deepgemm_native_preflight_cuda` to fail loud.
- FlashMLA: `ARLE_CUDA_ENABLE_FLASHMLA` + `_DECODE` + `vendor/flashmla/`; sm_90a (`build.rs:1895`). Without tree → stub, decode OFF.
- DeepEP (Piece 4): `ARLE_DEEPEP_DIR` → native; else stub, `NativeDeepEp::boot` errors.

## Parallelism
Piece 1 first (foundation, serializes `loader.rs`). Then Pieces 2 (`attention.rs`) ∥ 3 (`moe.rs`+`moe_config.rs`) — disjoint files. Final `model.rs` layer-loop branch (MLA-vs-paged `:188`, MoE-vs-MLP `:221`) is the single integration edit, serialized at the end. Piece 4 strictly last.

## CRITICAL precondition (verify before committing to SW-only)
Read the real `DeepSeek-V4-Flash/config.json` `compress_ratios` / `hc_mult` / `num_hash_layers`:
if every layer is CSA/HCA/HC (no SW-mode `compress_ratio==0` layers) or `hc_mult>1`, the
SW-only minimal path is NOT viable and Piece 2 jumps to **XL** (full CSA/HCA + indexer +
hyperconnections). This gates the effort estimate.

## Cargo gaps (first concrete change)
`crates/infer-cuda/Cargo.toml` has no `deepseek-spec` / `deepep-sys` dep today — add them.
