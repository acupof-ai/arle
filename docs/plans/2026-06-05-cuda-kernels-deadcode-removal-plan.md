> **Status:** read-only plan (2026-06-05). EXECUTE only after the DSv4 prefill
> i32-overflow fix commits (Codex live-edits `crates/cuda-kernels`); the MoE/
> DeepGEMM files are out of scope here. Closes task #18 when executed.

# CUDA-Kernels Dead-Code Removal Plan (verified read-only — 2026-06-05)

**Verification basis:** every `extern "C"` symbol in each candidate `.cu` was grepped across `crates/` (`*.rs`/`*.cu`/`*.cuh`) for: (1) its Rust FFI binding, (2) its `cuda-kernels/src` wrapper, (3) any live call expression reaching it from `infer-cuda`. "Live" = a real call expression in `infer-cuda/src/` or a `cuda-kernels/src` wrapper that `infer-cuda` reaches. Doc-comments / `use`-only imports do **not** count. The runtime-reachability gate is decisive: **`infer-cuda` constructs only `KVFormat::BF16`** (`executor.rs:273,282`) — so every quantized-KV path is dead at the runtime boundary even where it is wired into live `paged_kv.rs` via a never-selected match arm.

**Build mechanics confirmed:** `build.rs::collect_cu_files` (lines 1393-1415) is a **recursive glob over `csrc/`** — deleting a `.cu` removes it from compilation automatically. There are **no explicit per-file csrc path lists** except the FlashMLA stub. `emit_rerun_recursive` (1423) auto-handles `rerun-if-changed`. So whole-file `csrc/` deletion needs **no build.rs source-list edit** unless the stem is named in a per-file nvcc-flag branch (only `marlin_*`, `deepgemm_native`, `arle_flashmla_*` stems are). **Caveat:** object files are named by `file_stem()` (1822-1823) — never delete a file whose stem collides with a kept file (none found among candidates).

**In-flight Codex edits (do NOT touch — defer until committed):** `csrc/gemm/dsv4_deepgemm_ops.cu`, `csrc/moe/dsv4_route.cu`, `src/ffi/moe.rs`, `src/moe.rs`, `crates/infer-cuda/src/moe.rs` (prefill i32-overflow fix). None of these are in the clean-delete set; the MoE/DeepGEMM files are all live regardless.

## Registry drift caught (candidates that are NOT dead — DO NOT delete)

| Inventory "dead" claim | Reality | Live caller |
|---|---|---|
| FP4 batch GEMV | **LIVE** | `dsv4_fp4_gemv_batch_cuda` ← `infer-cuda/attention.rs:945,1011` (`WeightFormat::Dsv4Fp4BlockScaled`, reachable in `dsv4.rs:1262,1298`) |
| `dsv4_tp_attention_repack.cu` ("0 caller") | **LIVE** | `dsv4_tp_q_repack_cuda` ← `attention.rs:1437`; `dsv4_tp_out_slice_cuda` ← `attention.rs:1542` (FlashMLA-path glue, matches inventory §2 measured table) |
| FlashMLA SM90 decode + FP8-KV pack + output inverse-RoPE + build-indices ("unwired, doc-only") | **LIVE (gated, in active dev)** | `arle_flashmla_sm90_sparse_decode_*` ← `attention.rs:202,1386,1489`; `arle_dsv4_output_inverse_rope_start_pos_ptr_cuda` ← `attention.rs:1561`. (`infer-cuda` grep for the prefill/kv-pack symbols showed NONE directly, but the decode path IS live; whole `arle_flashmla_*` family is the `ARLE_DSV4_FLASHMLA_DECODE` in-flight feature — **KEEP all of it**.) |
| `arle_bf16_to_f32_cuda` (already flagged) | **LIVE** | `infer-cuda/loader.rs:1094` |

These are the same failure mode as the `arle_bf16_to_f32_cuda` drift: the registry §Unwired table is stale relative to the live tree and contradicts the inventory's own §2 measured table. **The measured table wins.**

## Tier 1 — CLEAN whole-file deletes (0 live caller end-to-end, no build.rs edit, safe now)

| file:symbol(s) | 0-caller confirmed? | FFI binding to also remove | build.rs / registry / KernelSet impact | safe-to-delete |
|---|---|---|---|---|
| `attention/mla_decode.cu` : `mla_decode_paged_bf16_cuda` | YES (wrapper `mla_decode_paged_bf16` 0 callers; `ffi::mla` only via glob) | `src/ffi/mla.rs` (whole file, 2 items) **+** `src/ffi.rs` remove `#[path="ffi/mla.rs"] pub mod mla;` and `pub use mla::*;` | glob auto-drops .cu; registry: drop MISPLACED/attention `mla_decode` row + §Unwired attention row | **Y** |
| `attention/fused_attention.cu` : `fused_gqa_attention_decode`, `fused_gqa_attention_decode_batched` | YES | `src/ffi/attention.rs:98,134` (2 decls) | glob auto; registry §Unwired attention row | **Y** |
| `attention/prefill_attention.cu` : `prefill_attention_prep_cuda` | YES (distinct from live `prefill_attention_paged_prep.cu`) | `src/ffi/attention.rs:5` | glob auto; registry §Unwired attention row | **Y** |
| `attention/decode_prep_paged_hd256.cu` : `decode_prep_paged_hd256_cuda`, `attention_gate_paged_hd256_cuda` | YES (distinct from live `decode_prep_paged.cu` and live `prefill_attention_hd256.cu`) | `src/ffi/attention.rs:246,271` | glob auto; registry §Unwired attention row | **Y** |
| `misc/fused_mlp.cu` : `fused_mlp_cuda` | YES | `src/ffi/gemm.rs:34` | glob auto; registry misc-dead row | **Y** |
| `misc/split_qkv.cu` : `split_qkv_cuda`, `silu_mul_fused_cuda` | YES | `src/ffi/elementwise.rs:59,70` | glob auto; registry misc-dead row | **Y** |

## Tier 2 — Recurrent dead pair + batch variants (0 live caller; live path uses different files; no build.rs edit)

The live recurrent path uses exactly 3 symbols — `conv1d_prefill_cuda` (`conv1d.cu`), `gated_delta_rule_decode_cuda` + `gated_delta_rule_prefill_recurrent_cuda` (`gated_delta_rule.cu`) — all via `infer-cuda/qwen35.rs:1278,1307,1323`. **Keep `conv1d.cu` and `gated_delta_rule.cu`.** The batch/chunk variants are dead:

| file:symbol(s) | 0-caller confirmed? | FFI binding to also remove | impact | safe-to-delete |
|---|---|---|---|---|
| `misc/conv1d_decode_batch.cu` : `conv1d_decode_batch_cuda` | YES | `src/ffi/recurrent.rs:36` | glob auto; registry misc-dead | **Y** |
| `misc/conv1d_prefill_batch.cu` : `conv1d_prefill_packed_batch_cuda` | YES | `src/ffi/recurrent.rs:74` | glob auto | **Y** |
| `misc/gdr_decode_batch.cu` : `gdr_decode_batch_cuda` | YES | `src/ffi/recurrent.rs:47` | glob auto | **Y** |
| `misc/gdr_prefill_batch.cu` **+** `misc/gdr_prefill_solve.cu` (**delete together — interdependent**) : `gated_delta_rule_prefill_chunk_{prepare,cumsum,a,recompute,state,o,solve}_cuda`, `gated_delta_rule_prefill_chunkwise_batch_cuda` | YES (chunkwise umbrella + all chunk steps 0 external caller; `gdr_prefill_batch.cu:44` is only a **fwd-decl** of `gdr_prefill_solve.cu`'s real def — one cannot ship without the other) | `src/ffi/recurrent.rs` remove all `gated_delta_rule_prefill_chunk_*` + `chunkwise_batch` decls (lines ~122 + siblings) | **TileLang `gdr_specs` (build.rs:1258-1344) are SEPARATE `gdr_chunk_*` codegen kernels — NOT affected.** glob auto-drops the .cu | **Y (as a pair)** |

## Tier 3 — DEFER: mixed files (live + dead symbols intermixed — symbol-level surgery, riskier, not a file delete)

| file | why deferred |
|---|---|
| `gemm/quantized_gemv.cu` | **MIXED**: `dsv4_fp4_gemv_batch_cuda` + `dsv4_fp8_gemv_batch_cuda` LIVE; `_pair_batch`/`_grouped_gemv`/`_route_gemv`/non-batch `dsv4_{fp4,fp8}_gemv_cuda` + entire GGUF Qk family (`q3k/q4k/q5k/q6k/q8/qxk_*`, `w2a16/w4a16/w8a16_gemv_*`) dead. Intra-file deletion only; do **not** delete file. |
| `gemm/quantized_gemv_mma.cu` : `dsv4_fp8_gemv_batch_mma_launch` | 0 caller, but a sibling/variant of the live `quantized_gemv.cu` — verify no `#include` coupling before deleting; small, low-risk whole-file delete possible after Codex MoE settles. |
| `kv/kv_quant.cu` | **MIXED**: BF16-path symbols (`dequantize_paged_kv_cuda`, `quantize_scatter_kv_fp8_*`, `compute_k_per_channel_absmax`, etc.) are wired into live `paged_kv.rs` (`migrate_from_contiguous_fp8_range`, `decode_attention_int8_workspace_bytes`); the `decode_attention_*` kernels live in `decode_attention_quantized.cu` (see Tier 4). The whole `kv_quant.cu` file is reachable-but-never-selected (no `KVFormat::Int8/Int4/Fp8` constructed by `infer-cuda`) — deleting requires pruning the `paged_kv.rs` non-BF16 KVFormat arms first. **Defer to a dedicated KVFormat-pruning task.** |
| `misc/elementwise_basic.cu` (`fused_add_rms_norm_*` dead symbols) | Live file (`add_cuda`/`silu_mul_cuda`/`embedding_batched_cuda`/`dsv4_swiglu_clamped_*` all live) with dead `fused_add_rms_norm_*` symbols — intra-file prune only, do **not** delete file. |
| `gemm/marlin_kernel.cu`, `marlin_w4_fp8_kernel.cu`, `marlin_w4a8_kernel.cu` : `marlin_gemm_cuda`, `gemm_w4_fp8_marlin_cuda`, `gemm_w4a8_marlin_cuda` | 0 caller, BUT **build.rs names these stems** (`disable_marlin_w4_fp8`/`marlin_w4_fp8_kernel` @1835; `marlin_kernel`/`marlin_w4a8_kernel` @1850; `stem.starts_with("marlin_")` @1887) and they share `marlin_dequant.cuh` + `marlin_pf8/` headers with the **live** `marlin_repack.cu`/`marlin_int4_fp8_preprocess.cu` (`tensor.rs:2086,2625`). Deleting needs build.rs edits (remove the per-stem branches) + header-coupling audit. **Defer — separate Marlin-prune task.** |

## Tier 4 — DEFER: KVFormat-gated dead families (reachable-but-never-selected; need paged_kv.rs/KVFormat refactor)

All dead at the `infer-cuda` boundary (only `KVFormat::BF16` constructed), but wired into live `paged_kv.rs`/`kv_quant.rs` wrappers via never-selected match arms. Deleting the `.cu` requires first removing the dead `KVFormat`/`KVCacheDtype` arms and the `paged_kv.rs` fields/calls — a refactor touching live files, out of scope for a pure-deletion pass:

| family | files | dead symbols | what must be pruned first |
|---|---|---|---|
| KV-quant decode-attention | `attention/decode_attention_quantized.cu`, `attention/decode_attention_varlen_fp8.cu` | `decode_attention_{fp8,int8,int4}_per_channel_k_cuda`, `decode_attention_varlen_fp8_cuda` | wrappers in `kv_quant.rs:461,695,752,833` (0 external caller — confirmed) + FFI `attention.rs:290,314,337,375`. These wrappers are dead-end; **deletable once their FFI decls + wrappers go**, but file lives in the KV-quant subsystem — bundle with the KVFormat prune. |
| TurboQuant (full chain) | `attention/decode_attention_turboquant.cu`, `quant/turboquant.cu`, `quant/turboquant_fast.cu`, `gemm/turboquant_weight_gemv.cu` | `tq_decode_attention_cuda`, `tq_rotate_query_cuda`, `turboquant_{,fast_}{quantize,dequantize}_*`, `turboquant_lloyd_max`, `turboquant_generate_{rotation,signs}`, `turboquant_weight_{dequant,gemv}_cuda` | `KVFormat::TurboQuant` is **never constructed by `infer-cuda`** (0 matches), but `paged_kv.rs:490,1914,1926` + `kv_types.rs:110` reach `TurboQuantLayerState`/`turboquant_quantize_paged_single` under the `KVFormat::TurboQuant` arm. Must delete: `KVFormat::TurboQuant` enum arm + all its match arms in `kv_types.rs` + `paged_kv.rs` `tq_k_state`/`tq_v_state` fields/calls + `src/kv_turboquant.rs` + `src/turboquant_state.rs` modules (in `lib.rs`) + `src/ffi/quant.rs` turboquant decls + `kv_turboquant.rs`/`turboquant_state.rs`. **Whole-subsystem prune — defer to dedicated task.** `turboquant_weight_gemv.cu` (0 caller, not even wired) can go in the same task. |
| KV-quant pack/migrate dead variants | `kv/kv_cache_to_paged.cu`, `kv/scatter_kv.cu`, `kv/paged_kv_append.cu` (specific symbols) | `kv_cache_to_paged_cuda`, `paged_kv_append_cuda` (0 caller) — but the `*_range*`/`*_hnd*`/`*_int8*` variants in these files ARE live (`paged_kv.rs:1650,1748,1895`) | intra-file dead-symbol prune within otherwise-live files — defer with the KVFormat task. |

## Tier 5 — KEEP (verified live or gated-in-active-development)

`conv1d.cu`, `gated_delta_rule.cu`, `arle_dtype_convert.cu` (arle_bf16_to_f32), `dsv4_tp_attention_repack.cu`, all `arle_flashmla_*` + `dsv4_fp8_kv_pack.cu` + `dsv4_flashmla_decode_build_indices.cu` (FlashMLA decode in-flight), `marlin_repack.cu`/`marlin_int4_fp8_preprocess.cu`, `kvcacheio/transfer.cu` (live `transfer_kv_pages_layer_table_cuda` ← `paged_kv.rs:961`), all live GEMM/MoE/norm/sampling/embedding. **Do not touch the 5 Codex-in-flight files** regardless of tier.

## Deletion order (execute AFTER Codex's DSv4 prefill work commits)

1. **Tier 1** (6 files) — atomically delete each `.cu` + its FFI decl(s). Delete `ffi/mla.rs` whole + strip its `mod mla`/`pub use mla::*` from `ffi.rs`. One commit per logical group (attention-dead / misc-dead). Compile gate after each: `cargo check -p infer --no-default-features --features cuda,no-cuda` (Mac typecheck) or full CUDA build on H20.
2. **Tier 2** (5 files, gdr pair together) — delete + strip `recurrent.rs` decls. **Verify TileLang `gdr_specs` codegen still fires** (build.rs:1258-1344 unaffected). Compile gate.
3. After each commit: regenerate `docs/reviews/kernel-registry.md` §Unwired rows and the inventory §Excluded list to remove the deleted entries; correct the 4 registry-drift rows (FP4 GEMV, tp_repack, FlashMLA decode = LIVE; not dead).
4. **Tiers 3-4** are separate follow-up tasks (mixed-file symbol surgery; KVFormat/TurboQuant subsystem prune) — each needs its own plan because it edits live `paged_kv.rs`/`kv_types.rs`/`tensor.rs`/`elementwise_basic.cu`, not just deletions.

## build.rs / kernel-registry / KernelSet update summary

- **build.rs:** Tier 1+2 need **no source-list edit** (recursive glob). Only **Tier 3 Marlin** would touch build.rs (remove `marlin_w4_fp8_kernel` @1835, `marlin_kernel`/`marlin_w4a8_kernel` @1850 stem branches). No `KernelSet`/registry codegen struct references any Tier 1/2 stem (verified — no explicit per-file csrc paths exist).
- **kernel-registry.md:** drop the §Unwired rows for the deleted symbols; **fix the 4 drift rows** (move FP4 GEMV / `dsv4_tp_attention_repack` / FlashMLA-decode family from §Unwired into live tables — they have live callers today).
- **No `ffi.rs` glob breakage:** all deleted symbols are removed at their declaring `ffi/*.rs`; the `pub use <mod>::*` re-exports stay valid (only `mla` module is fully removed, so its `mod`+`pub use` lines go too).

**Bottom line:** 11 `.cu` files are clean/near-clean whole-file deletes (6 Tier-1 + 5 Tier-2), needing only FFI-decl strips and zero build.rs source-list edits. Everything else the inventory called "dead" is either **registry drift with a live caller today** (FP4 GEMV, tp_repack, FlashMLA decode — KEEP) or **reachable-but-never-selected KVFormat-gated code** (turboquant, quantized-KV decode-attention, mixed gemv/kv_quant files) that requires a live-file refactor and must wait for a dedicated KVFormat-prune task. Defer all MoE/DeepGEMM-adjacent work until Codex's prefill-overflow fix commits.