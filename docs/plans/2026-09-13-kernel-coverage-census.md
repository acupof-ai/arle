# Kernel parity coverage census — default single-card H20 (sm90, c=1) path

Date: 2026-09-13. Read-only census; **no code changed**. This is the
complement to `2026-09-13-gate-input-set-audit.md` (fd): that audit asks
whether each gate's input set contains the production thing it names;
this census starts from the kernels and asks which gate, if any,
exercises each one. The join is total: every production kernel reachable
on the default single-card serving path has either a named covering gate
or an explicit "none".

## Method and populations

- Kernel set: every FFI wrapper in `crates/cuda-kernels/src/{attention,
  recurrent,moe,tensor_ops,sampling,kv_quant,quant_linear}.rs` that has a
  production launch site in `crates/infer-cuda/src/` on the c=1 sm90
  path. Classification is from call-site code, not from
  `operators/registry.toml` labels or gate headers (the registry's
  `correctness_gate` field is declarative; several rows name gates whose
  input set excludes the target — see fd's audit).
- Default path: one H20, `rows.len()==1` dispatch
  (`executor/dsv4.rs:459`), canonical checkpoints DSv4-Flash FP8 and
  Qwen3.5/3.6 dense. Batched (`*_batched`, c≥2), TP/NCCL/DeepEP, and
  sm120-only kernels are excluded and listed once at the end.
- Populations:
  - **A** — gate launches the kernel at production geometry **and** has a
    recorded on-GPU positive run (negative arm also run).
  - **B** — gate launches the kernel at production geometry but has **no
    recorded GPU execution** (fixes merged, rerun blocked on the 72 h GPU
    reservation; or SKIP without model+cards). A B gate is coverage on
    paper only; per the wrap-up rule, a kernel whose *only* gate is B is
    counted as **not covered in practice** in the headline numbers.
  - **C** — reachable on the default path, **no gate launches the
    kernel**.
  - **D** — vendor-trusted kernel with a named end-to-end gate only.
- GPU evidence: the 2026-09-11 parity batch
  (`docs/experience/errors/2026-09-11-parity-gpu-batch-first-run.md`) and
  the P1 reruns recorded under the `parity-fix-*` agenda rows.
- Input-set verdicts in the Gate column come from fd's audit
  (`partial` = production geometry/dtype, synthetic weights; these
  kernel-arithmetic gates are weight-independent, so partial is sound
  for kernel coverage — it is the model-level claim that is unsupported).

## A — gated and executed on GPU (11 gate suites)

| Production kernel(s) | Gate | Input set | GPU evidence |
|---|---|---|---|
| `gated_delta_rule_decode_cuda`; `gated_delta_rule_decode_batch_cuda`, `conv1d_decode_batch_cuda` (`gdr_decode{,_batch}_raw`, `conv1d_decode_batch_raw`, qwen35_attention.rs:2054,2340,2521) | gdr_decode_parity | partial, geom sweep incl (16,48) | PASS 2026-09-11, pos+neg |
| gdr_fq AOT `gdr_fq_{cumsum,kkt,fwd}_*` + `gdr_fq_prep`; `conv1d_prefill_cuda`; `gated_delta_rule_prefill_recurrent_varlen_cuda`, `conv1d_prefill_varlen_cuda` (qwen35_attention.rs:2238,2435-2490,2724) | gdr_varlen_parity | yes (path-equivalence; RNG sufficient) | PASS 2026-09-11, pos+neg |
| `arle_fa3_fwd_hd256_{bf16,quant}_cuda` (qwen35_attention.rs:278,852,874) | fa3_hd256_shim_parity | partial, D256 matches | PASS 2026-09-11, pos+neg |
| `arle_paged_attention_quantized_fa3_cuda` (qwen35_attention.rs:675) | paged_quant_attn_parity | partial | PASS 2026-09-11; device tooth #434 merged after that run |
| `arle_flashmla_sm90_sparse_prefill_fwd_cuda`; `arle_flashmla_{csa,hca}_build_indices_cuda`; `arle_flashmla_csa_pack_kv_cuda` (attention.rs:1240,1343,1428,1445) | flashmla_prefill_parity | partial, tight rel ~7e-4 | card-verified #433/#439 |
| `dsv4_route` family (route, count, exclusive_scan, moe_exclusive_scan_aligned, pack_local_experts_with_slots, fill_m_indices, scatter_all_route_slots, combine_route_slot_outputs, `qwen36_renorm_topk_weights`) | moe_routing_parity | partial, EP256/TOPK6 match | card PASS #436 (dbfc04faa), 7-family neg sweep |
| `dsv4_fp8_grouped_swiglu_decode_cuda`, `_down_decode_cuda` (moe/dsv4.rs, c=1 FP8 decode) | dsv4_decode_moe_parity | partial | PASS 2026-09-11, pos+neg |
| `dsv4_deepgemm_m_grouped_fp8_gemm_nt_{masked,contiguous}_cuda` (moe/dsv4.rs:1880-2222) | deepgemm_grouped_prefill_parity | partial, one 128-row tile | card PASS #431 (8a5d53f5e), 174/174 bands |
| `dsv4_deepgemm_pack_quantize_bf16_to_fp8`, `dsv4_deepgemm_fp8_gemm_nt`, `dsv4_fp8_{,route_}gemv_batch_cuda`, o-proj gather/scatter, `tp_out_slice` (attention.rs:341,4951-5415) | dsv4_tp_oproj_parity | partial, TP1-8 sweep incl c=1 shapes | card PASS #443 (ceb873116), 12 shapes; open residual at m=8 token 6 |
| `nonpaged_prefill_attention_ring_varlen_cuda` (+batched; qwen35/dspark.rs drafter) | dspark_draft_attn_parity | no for real-weight model claim; yes for kernel arithmetic/dims 40q/8kv hd128 | PASS 2026-09-11, pos+neg |
| `marlin_fp4_gemm_cuda` (nvfp4 path, ops/quant_linear_fp4.rs:154) | marlin_fp4_correctness | — | PASS 2026-09-11, pos+neg |

## B — gated, never recorded executing on a GPU (12 gate suites)

All fixes from the first failed batch are merged; none of these has a
recorded positive run at the fixed head.

| Gate suite | Kernels it covers | State |
|---|---|---|
| argmax_parity | `argmax_cuda`, `argmax_batch_cuda` (hot greedy path) | #354 gate fix merged; no rerun |
| elementwise_parity | `rms_norm_cuda`, `rms_norm_batched_cuda`, `silu_mul_cuda`, `silu_mul_fused_cuda`, `split2_cuda`, `split_qkv_cuda`, `embedding_batched_cuda` | #354 split2 fix merged; no rerun |
| marlin_fp8_parity | `marlin_fp8_gemm_cuda`, `gemv_fp8_block_scaled_batch_cuda` | #354 planner fix; rerun queued; input set samples fixed 1024 columns (fd defect #5) |
| marlin_w8a16_parity | `marlin_w8a16_gemm_cuda`, `gemm_cuda` w8a16 lane, `dequantize_w8a16_to_bf16_cuda` | #354 fix; no rerun |
| dspark_sampler_parity | `dspark_filter_probs_cuda`, `dspark_draft_sample_cuda`, `dspark_chain_accept_cuda` | real kernel bug fixed #435; the contract is the on-card 61→0 rerun, not yet run |
| dspark_dsa_parity | `dsv4_dsa_fused_q_indexer_rope_hadamard_quant_cuda`, `dsv4_deepseek_v4_topk_transform_cuda`, `dsv4_dspark_draft_attention_cuda` | #357 gate fix; no rerun |
| flashmla_sparse_decode_parity | `arle_flashmla_sm90_sparse_decode_fwd_cuda` + get_meta/sched_meta, decode build_indices batched, `arle_dsv4_fp8_kv_pack_strided_cuda` (MODEL1 lane) | #382 oracle fixes; output negative tooth measured at 0.90× bound 2026-09-13 (washout, see negative-control audit) |
| flashmla_hca_decode_parity | HCA decode fwd + same pack/indices family | same; tooth 0.90× bound |
| `cuda-kernels` device tests (gemm_tests.rs and friends; **16 device-requiring functions**) | `quantize_paged_kv_{fp8,single}_cuda`, gemv fp8/fp4/w8a16/w4a16 family, dequantize family, `gemm_cuda` (cuBLAS) toy test | never executed under any runner; geometry toy (2 heads/hd64, n,k≈256) even once run — does not cover prod hd256/24q or prod GEMM M/N/K |
| fa2_sm70_parity | `arle_fa2_sm70_attention_cuda` | SKIP on sm80+; needs V100; kernel targets hardware the fleet does not serve (MTP spec head only on the c=1 path) |
| dspark_drafter_batch_invariance | drafter ring + sampler full step | #374 merged; never run; synthetic RNG/TP1 vs Qwen3.8-27B (fd: green cannot speak to production) |
| dsv4_parity (model gate) | first-token greedy over a real DSv4 checkpoint, TP=8 | never executed (no model+cards); gates token 1 of 16 per #445 scoping |

The two never-run **real-weight** gates (dsv4_parity above;
`qwen35_tq4_dense_parity.py`, hand-run torch diagnostic, no CI runner)
are not the sole coverage of any production kernel, so no C row is
reclassified because of them — but automated real-weight parity that has
ever run green remains **0** (fd cross-cutting finding 3).

## C — no gate at all

### C1 — HOT/WARM on the canonical c=1 path: 33 kernel entry points

DSv4-Flash single-row decode/prefill; no example, parity script, or
device test launches the symbol.

| # | Kernel entry point | Production call site (enclosing fn) | Reach |
|---|---|---|---|
| 1 | `arle_dsv4_compressor_update_start_pos_ptr_cuda` | attention.rs:6034 `compressor_forward` | every c=1 decode |
| 2 | `arle_dsv4_compressor_update_cuda` | attention.rs:6061 `compressor_forward`, host-pos lane | decode/commit |
| 3 | `arle_dsv4_compressor_fp32_prefill_probe_cuda` | attention.rs:5809 `compressor_fp32_probe` | every DSv4 prefill |
| 4 | `arle_dsv4_compressor_fp32_carry_reseed_cuda` | attention.rs:5731 same | prefill after bf16 carry |
| 5 | `gemm_bf16_f32_cuda` | attention.rs:5760,5769 (probe wkv/wgate projections) | DSv4 prefill probe only |
| 6 | `arle_dsv4_prepare_qk_start_pos_ptr_cuda` | attention.rs:3229,3884 `mla_attention_{prepare,decode}` | every c=1 decode |
| 7 | `arle_dsv4_prepare_qk_cuda` | attention.rs:217 `commit_layer_fold`, :3903 | prefill/commit |
| 8 | `arle_dsv4_prepare_qk_fused_batch_start_pos_cuda` | attention.rs:3864,4280; dsv4/dspark.rs:196,669 | prepass; c=1 n=1 reachability not fully traced — no prepare_qk variant is gated regardless |
| 9 | `arle_dsv4_output_inverse_rope_start_pos_ptr_cuda` | attention.rs:2142,2328 flashmla decode/prefill finish | every decode |
| 10 | `arle_dsv4_output_inverse_rope_cuda` | attention.rs:1748 flashmla prefill (non-chain) | prefill |
| 11 | `arle_dsv4_update_window_cache_start_pos_ptr_cuda` | attention.rs:1160 `update_bf16_sw_window` | every decode |
| 12 | `arle_dsv4_update_window_cache_cuda` | attention.rs:1174 | prefill tail |
| 13 | `arle_dsv4_fp8_kv_fill_one_sw_slot_from_start_pos_cuda` | attention.rs:963 `flashmla_pack_one_sw_token` | every decode |
| 14 | `arle_dsv4_fp8_kv_pack_completed_compressor_row_start_pos_cuda` | attention.rs:1055 `flashmla_pack_compressed_delta` | every decode single row |
| 15 | `arle_dsv4_flashmla_decode_build_indices_start_pos_ptr_cuda` | attention.rs:1920 `flashmla_decode_attention` c=1 twin | every decode; only the batched twin sits behind a B gate |
| 16 | `arle_dsv4_dsa_hadamard128_bf16_cuda` | attention.rs:7274 `csa_select_official` | CSA indexer cache write |
| 17 | `arle_dsv4_dsa_fused_store_index_k_cache_cuda` | attention.rs:7306 same | CSA select |
| 18 | `arle_dsv4_dsa_pack_index_row_start_pos_cuda` | attention.rs:7212 `csa_select` | CSA select |
| 19 | `arle_dsv4_dsa_fill_context_lens_positions_start_pos_cuda` | attention.rs:7401 `csa_select_official` | CSA select |
| 20 | `dsv4_deepgemm_paged_mqa_logits_metadata_cuda` | attention.rs:7462,7713 | CSA select, sm90 |
| 21 | `dsv4_deepgemm_fp8_paged_mqa_logits_fused_cache_cuda` | attention.rs:7499,7748 | CSA select, sm90 |
| 22 | `arle_flashmla_chain_verify_build_indices_cuda` | attention.rs:1397 `flashmla_prefill_attention` | DSpark chain-verify metadata; spec opt-in (DSpark is the production spec path) |
| 23 | `dsv4_mhc_expand_cuda` | dsv4/hc.rs:48 ← dsv4/{head,prefill,mtp,spec_verify} | every DSv4 layer |
| 24 | `dsv4_mhc_pre_rms_norm_cuda` | hc.rs:158 | every DSv4 layer |
| 25 | `dsv4_mhc_post_cuda` | hc.rs:207 | every DSv4 layer |
| 26 | `dsv4_mhc_head_pre_cuda` | hc.rs:335 | every DSv4 layer |
| 27 | `rms_norm_offset_cuda` | qwen35.rs:343 area; qwen35_forward.rs:465,530,792,847 routes via batched helper | every Qwen3.5/3.6 layer; elementwise_parity launches only the plain/batched norms, neither offset kernel |
| 28 | `rms_norm_batched_offset_cuda` | qwen35.rs:343 `rms_norm_offset` helper | every Qwen3.5/3.6 layer |
| 29 | `rms_norm_gated_cuda` | qwen35_attention.rs:2079 GDN output norm | every GDN layer |
| 30 | `add_cuda` | tensor_ops::add, 9 production call sites | residual stream; elementwise_parity excludes it |
| 31 | `decode_prep_paged_hd256_cuda` | qwen35_attention.rs:528 `full_attention_paged` | every Qwen dense c=1 paged decode |
| 32 | `prefill_attention_paged_prep_hd256_cuda` | qwen35_attention.rs:560 same | chunked prefill |
| 33 | `attention_gate_paged_hd256_cuda` | qwen35_attention.rs:1001 same | paged attention tail |

The FlashMLA attention cores themselves are A/B-gated; the gap is the
surrounding DSv4 memory/index/normalization machinery: compressor state
(1-5), qk/o-rope/window/pack glue (6-15), CSA selection (16-21), the
multi-head-combine quartet (23-26), and the Qwen dense paged prep/gate
trio (31-33).

### C2 — same single-card binary, alternate checkpoint quant family: 25 kernels

Reachable by loading a checkpoint that selects the path (no
recompile); none has a numeric gate. Production serves these
quantizations, so they are ungated production kernels but not on the
canonical FP8/DSv4-Flash default.

- V32/head_dim 576 pack: `arle_dsv4_v32_fp8_kv_pack_strided_cuda`
  (attention.rs:995; only MODEL1 512 is gated).
- W4AFP8 experts (moe/dsv4.rs:953,1179-1255): `w4a8_per_tensor_fp8_quant`,
  `w4a8_compute_problem_sizes`, `w4a8_moe_grouped_gemm`,
  `w4a8_swiglu_fused`, `w4afp8_grouped_swiglu_decode`.
- W4A16 grouped GEMV: `moe_w4a16_grouped_gemv_batch`,
  `_pair_batch` (moe/dsv4.rs:972,1000; moe/qwen.rs:1477,1601).
- DSv4 FP4 gemv: `dsv4_fp4_gemv_batch`, `dsv4_fp4_route_gemv_batch`
  (attention.rs:353,4951).
- Qwen3.6 grouped MoE (moe/qwen.rs): `moe_fp4_e2m1_grouped_gemv_batch` +
  `_pair_batch`; `deepgemm_m_grouped_bf16_gemm_nt_{masked,contiguous}`
  (BF16 experts; the deepgemm gate is FP8 only);
  `moe_bf16_grouped_gemm_swiglu_decode`, `_decode`, `_pair_batch`,
  `_batch`; `moe_fp8_block_scaled_grouped_gemv_batch` + `_pair_batch`;
  `qwen36_add_shared_expert_gated_cuda` (qwen.rs:683).
- Dense lanes: `gemv_fp4_e2m1_group_cuda`, `scale_columns_bf16_cuda`
  (per-channel FP8 post-scale, quant_linear_fp8.rs:566),
  `interleave_gate_up_fp8_rows_cuda` (load/init, moe/dsv4.rs:2331).
- Pre-fill MoE epilogue `dsv4_deepgemm_swiglu_quantize_w13_cuda`
  (moe/dsv4.rs:1895,2012,2209) is launched at production geometry by
  `fp8_grouped_prefill_probe`, but the probe compares an alternate GPU
  lane with timing only: no f64 oracle, no negative control, unregistered
  — counted as no numeric gate.

### C3 — opt-in features on one card: 11 kernels

Spec/MTP (default-off `--spec-type mtp`; DSpark spec is separately A-gated
for its ring attention only): `nonpaged_prefill_attention_raw`,
`_devpos_raw` (qwen35_attention.rs:217,296),
`prefill_attention_hd256_prep_raw` (:153),
`attention_gate_batch_hd256_raw` (:323),
`prefill_attention_hd256_prep_ring` (qwen35/dspark.rs:802,910,1302),
`dsv4_mhc_lane_mean_cuda` (dsv4/dspark.rs:569),
`dsv4_mtp_add_eproj_hproj_cuda` (dsv4/mtp.rs:99).
LoRA (qwen35_lora.rs): `quantize_bf16_to_fp8_block_scaled`,
fp8-marlin dequant, `add_scaled_row`.
Fallback: `gated_delta_rule_prefill_recurrent_cuda` single-row host ABI
(qwen35_attention.rs:2534) — reachable only when chunked FlashQLA is
unavailable (default is chunked); same math is A-gated through the
varlen ABI, this symbol itself is not.

## D — vendor-trusted, end-to-end gate only

None on the single-card default path: every vendored kernel it reaches
(FA3, FlashMLA SM90 sparse, DeepGEMM, Marlin FP4) has an executable
parity gate in A or B. The vendor code reachable **only** with multiple
cards — NCCL allreduce/allgather (comm.rs), DeepEP dispatch/combine
(deepep.rs, moe/dsv4_deepep.rs), `sm90_mega_moe_*` TP transport
(moe/dsv4.rs:421,446) — has only the model-level e2e gates
(scripts/needle_gate.py, scripts/lever_gate.sh), which require the
production model and are not in CI. cuBLAS `gemm_cuda` is vendor library
code but its only device test is toy-geometry B; it is not claimed D.

## Headline counts

| Quantity | Count | Scope |
|---|---|---|
| Production kernels with no gate at all (C1) | **33** | HOT/WARM entry points on the canonical c=1 sm90 path |
| Same, including alternate-quant single-card families (C1+C2) | 58 | +25 checkpoint-selectable kernels |
| Same, including opt-in spec/LoRA/fallback (C1+C2+C3) | 69 | +11 |
| Gate suites gated but never GPU-executed (B) | **12** | covering 16 never-run device test functions plus the example gates; kernels behind them are uncovered in practice |
| Gate suites with a recorded GPU pass (A) | 11 | |
| Vendor-only-with-e2e on c=1 (D) | 0 | |

## Positive controls

1. Known-gated symbol detector: `rg -l argmax crates/infer-cuda/examples/`
   returns argmax_parity.rs (plus dsv4_parity, dspark_sampler).
2. Real-launch vs header comment: `flashmla_csa_build_indices_raw`
   appears in flashmla_prefill_parity.rs comment at :11 and in real
   launches at :352,:629; csa_pack_kv at :293,:580.
3. Zero control: the same fixed-string detector returns no example for
   `dsv4_compressor_update` (C row 1/2) — a clean zero from a detector
   that returns hits in controls 1-2.
4. Wrapper→gate mapping built by matching all 192 wrapper names with
   fixed strings (`grep -F -f`) against every example, not regex guesses.
5. The W4A8 deletion trap was controlled against
   `docs/experience/wins/2026-09-13-w4a8-weight-format-deleted.md`: the
   deleted MarlinW4A8 *weight format* is distinct from the live W4A8
   CUTLASS kernels at moe/dsv4.rs:1179-1255, confirmed present.

## Unknowns (committed honest)

1. `dsv4_prepare_qk_fused_batch_start_pos` (row 8) c=1 n=1 reachability
   not fully traced; classified C regardless since no prepare_qk variant
   is gated.
2. Whether any current H20 production checkpoint selects the V32
   head_dim 576 branch (C2) was not confirmed from a committed config;
   GLM-5.2 is DSv4-family but its head dim was not read here.
3. A/B-lane perf probes (`fp8_{grouped_prefill,decode,smallm_gemm}_probe`)
   are counted as no numeric gate (no oracle, no negative control,
   unregistered); if the project convention counts cross-lane probes as
   parity, C2's quantize_w13 row and the grouped-GEMV rows move.
4. `paged_attention_v1` TileLang AOT (qwen35_attention.rs:961) is
   unreachable on sm90 (FA3 hd256 wins) and excluded from C; the sm<90
   prefill fallback was ruled unreachable for hd128 in the 2026-09-11
   audit.

## Excluded populations (verified guards)

All `*_batched` DSv4 kernels (executor n>1 only); sm90_mega_moe
(cfg nccl); `arle_fp8_moe_grouped_gemm_nt_sm120` (sm120 only);
ring/cross_cp/ring_prefill_* (attn_cp≥2); dsv4_cast_i32/i64 and
silu_mul_masked_quant (DeepEP transport); nvfp4_to_w4afp8 load path;
expert pointer-table builders; bf16_to_f32; batched_copy_uniform/
fill_bf16; host-only workspace/preflight/layout functions.
