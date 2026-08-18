# Global dead-code deletion, wave 2 — CUDA kernels + cross-crate, 2026-08-18

> Status: verified (local + CUDA build + parity gate + W8A16 Marlin smoke on pod)

## Goal

Second wave of the global dead-code deletion: remove non-main-path code
already validated as dead (zero callers repo-wide), with priority on the
CUDA C++/FFI surface that wave 1 did not reach. Zero runtime change is the
correctness criterion.

## Scope

12 commits, **+37 / −5,819 (net −5,782 lines)**. Every deletion verified as
zero-caller before removal (repo-root grep across rs/cu/cuh/cpp/h/sh/py/toml,
including cfg-gated and build.rs-generated callers).

| Chunk | Crate | Deleted | Lines |
|-------|-------|---------|------:|
| quantized_gemv.cu | cuda-kernels | 20 dead fns: dequantize_w4a16_to_fp16 kernel+export, fp8_wread_probe, q8_embedding batched/decode, w4a16_gemm_batch, w2a16_gemv×3, q3k_gemv/batch/dequant_chunk + exports, q3k/q8 embedding exports; orphan Q3K_* defines | ~554 |
| kv_quant.cu | cuda-kernels | dequantize_paged_kv_cuda, quantize_scatter_kv_fp8_cuda, finalize_k_per_channel_scales_int4_cuda, quantize_paged_kv_int4_per_channel_cuda, quantize_paged_kv_single_int4_cuda | ~170 |
| gemv.cu | cuda-kernels | autotune_all_cached_gemms_cuda, gemm_graphsafe_cuda, gemm_fp16_weight_cuda | ~120 |
| attention .cu dead exports | cuda-kernels | nonpaged_prefill_attention plain ring export (varlen variants live), decode_attention_quantized.cu whole file, arle_flashmla_csa_prep fill_pad_rows, dsv4_flashmla_decode_build_indices plain export (start_pos_ptr + batched live), dsv4_fp8_kv_pack fill_sw_slots | ~400 |
| 12 deleted .cu files | cuda-kernels | marlin_int4_fp8_preprocess, marlin_repack, fp16_gemm_wmma, turboquant_weight_gemv, kv_cache_to_paged, dtype_convert, bf16_to_fp8, gdr_prefill_solve, gdr_prefill_batch, decode_attention_turboquant, arle_q8kv8_prefill_shim, vendor/q8kv8_prefill/ (8 files) | ~2,400 |
| FFI decls | cuda-kernels | 4 attention decls + fill_pad_rows decl in ffi/misc.rs; doc-link retargets | ~60 |
| MoE / topo helpers | infer-moe, infer-topo | route_and_combine (CPU MoE reference), has_shared_expert, TpLinearConfig struct+impl, is_global_tp_ep_only | ~90 |
| plan/core/util | infer-plan, infer-core, infer-util | SpecPlan, ForwardPlan.spec/.microbatch fields (+8 external initializers), ForwardMode::TargetVerify/DraftExtend, merge_vocab_shard_argmax, diffusion_prediction_from_logits, generate_diffusion wrapper, predict_row/sample_gumbel orphans, RecallPlan::recalled_blocks, Engine::cancel_all_requests (pub→private), resolve_weighted_model_path, download_runtime_assets_from_hub | ~330 |
| autograd | autograd | module.rs whole file, ConstantLr/LinearWarmup/parse_lr_schedule (CosineWithWarmup + LrSchedule live), fused_linear_distill sparse leftovers, comm_world_rank, checkpoint_sequential group_size param collapsed to inline `li + 1` | ~520 |
| cli / train | cli, train | --teacher-topk ×2 (parsed-but-rejected stub; engine-side top-k never landed), reject_unimplemented_gkd_objectives, GkdLossConfig.teacher_topk field + validation/step rejection arms, checkpoint_policy group_size arg, unused KlDirection import | ~120 |
| infer-cuda | infer-cuda | tp.rs with_comm/oneshot_comm_active, dsv4_resident_ab.rs 3 dead env-var sets + env_flag helper (decode-graph lane deleted 9b12060fc), paged_kv_table.rs dsv4_pack_token_byte_base + dsv4_decode_route_index host mirrors | ~110 |
| Marlin kernel arg | cuda-kernels | unused `int max_shared_mem` arg from MARLIN_KERNEL_PARAMS macro + template + launch site (max_shared_mem_new stays as LaunchKernel dynamic-shm-size arg) | ~15 |
| stale docs/scripts | docs, scripts | eval_humaneval.py, eval_mbpp.py, scripts/README.md 6 stale rows, environment.md INFER_MARLIN_W4_FP8_PREFILL row, architecture/codebase-map/AGENTS kv_turboquant drift | ~120 |

## What was NOT deleted (live, not dead)

- q4k/q5k/q6k quantized_gemv exports — HIP backend callers
- shared qxk_embedding kernels, q3k_value/q3k_decode_scales — live dispatch case 3
- autotune_gemm_cuda — live (only the _all_cached variant was dead)
- varlen nonpaged_prefill exports, dsv4 build_indices start_pos_ptr + batched variants — live
- resolve_local_weighted_model_path — live (cli/ocr.rs, infer-metal)
- CosineWithWarmup + LrSchedule trait — live
- KVCacheDtype — live internally (paged_kv.rs legacy mapping); only the re-export was removed in wave 1
- agent-bench whole crate — deleted 2026-08-18 (user confirmed); 743-line harness crate, zero external dependents
- mlx-sys C++ dead wrappers, infer-metal pipeline_fast_path_hits — skipped, HOT crates with concurrent edits
- arle_monitor.py / arle_watchdog.sh / cp2_ttft_oneshot.sh cluster — deleted 2026-08-18 (user confirmed); zero external callers
- dsv4_route.cu 14 dead exports — blocked on user's in-flight ffi/moe.rs W4AFP8 work; delete after merge

## Verification

### Local (Mac)

```
cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib  → clean (after c9a01d538)
cargo check -p cuda-kernels -p infer-cuda --no-default-features --features cuda,no-cuda → clean
cargo check -p cli --no-default-features --features cuda,no-cuda                       → clean (after 477541e04)
cargo clippy -p <all changed crates> -- -D warnings                                     → zero warnings (changed crates)
cargo test -p cli --release --no-default-features --features metal,no-cuda              → pass
cargo test -p arle --profile release-fast --no-default-features --features cpu,no-cuda,cli → pass
```

Pre-existing warnings on HEAD (not from this wave): infer-cuda/src/gpu_sample.rs
spawn/parse_nvidia_smi/query_nvidia_smi never used — the sampler was disabled
in bbd422973. Left for the owner of that change.

### Remote (CUDA build + parity gate) — verified

- `cargo build --release --features cuda,nccl --bin arle` on pod (H20, sm_90)
  at 477541e04: clean (BUILD_EXIT=0). First attempt at 01e9c4822 failed on
  the CUDA-gated `agent_opd.rs` stale import — see Problems.
- `scripts/lever_gate.sh` with Qwen3.5-9B on GPU 1 (NEEDLE_MAX_TOKENS=2000,
  full ladder ×3):

| len | exact | partial | miss | wave-1 baseline |
|-----|-------|---------|------|-----------------|
| 115 | 0 | 0 | 3 | 0 / 0 / 3 |
| 300 | 3 | 0 | 0 | 3 / 0 / 0 |
| 446 | 3 | 0 | 0 | 3 / 0 / 0 |
| 2000 | 3 | 0 | 0 | 3 / 0 / 0 |
| 8000 | 3 | 0 | 0 | 3 / 0 / 0 |

PASS: exact match with the wave-1 same-config envelope at every length.
len=115 misses are the same model characteristic as wave 1 (reasoning model
consumed all 2000 tokens on reasoning_content, empty output).

- W8A16 Marlin 27B champion-row A/B (the one live-path touch — the unused
  `max_shared_mem` kernel arg removal): the Huihui abliterated checkpoint is
  not on this pod and hf-mirror.com is unreachable from the node, so the stock
  Qwen3.6-27B BF16 already on the node was quantized to W8A16 gs=128
  (491 tensors, `/root/w8a16work/w8a16_quant.py`). Same architecture, same
  shapes — abliteration changes weights only, not decode compute. Bench:
  `bench-agent-32k-64.jsonl`, c=1, 16 requests × 256 tokens, temp 0, seed
  20260416, GPU 6.

  | metric | champion row (2026-08-06) | wave-2 (2026-08-18) | delta |
  |--------|--------------------------:|--------------------:|------:|
  | ITL p50 | 16.70 ms | 16.72 ms | +0.1% |
  | ITL p99 | 20.50 ms | 20.69 ms | +0.9% |
  | TTFT p50 | 23.01 s | 22.93 s | −0.3% |
  | e2e p50 | 27.4 s | 27.39 s | −0.04% |

  All deltas inside the noise floor (champion row: reps agree to 0.02 ms ITL).
  The arg removal is zero-perf by construction — the kernel body never read
  the arg, launch config unchanged. Perf license granted.

## Problems

- **User's committed code broke the cuda,no-cuda lane (fixed in c9a01d538)**:
  64a922bbf added `if rc != 0 { bail!("... {rc}") }` on the FFI's `CUresult`
  enum in moe.rs:2318 — CUresult is neither comparable to 0 nor Display.
  Fixed to `rc != CUresult::CUDA_SUCCESS` + `{rc:?}`, matching the pattern in
  infer-cuda/src/tp.rs. Flagged to the user since they may be mid-edit on the
  W4AFP8 path.
- **External restore wiped tranche B mid-wave**: the user's W4AFP8 workflow
  restored all of csrc/ + ffi/{attention,misc}.rs to HEAD, deleting
  uncommitted wave-2 work. Re-applied and committed immediately (63e52f55d).
  Lesson reinforced: work is durable only once committed.
- **Brace-matching orphans (×2)**: TpLinearConfig and LinearWarmup cuts
  matched the struct's column-0 `}` instead of the impl's, leaving orphaned
  impl blocks. Fixed by cutting the impl separately. Lesson: for struct+impl,
  cut impl first or verify extent.
- **ForwardPlan field removal cascaded**: removing spec/.microbatch left 8
  external struct-literal initializers (E0560) across planner.rs/vulkan/metal/
  examples. Fixed with sed in the same commit.
- **Zero-ref grep from the wrong cwd**: an early verification pass ran from
  crates/cuda-kernels with repo-root-relative paths and silently searched only
  cuda-kernels/src/. Re-ran from the repo root; found 5 symbols still had Rust
  FFI decls. Lesson: zero-ref verification is only as wide as the cwd.
- **CUDA-gated caller invisible on Mac (fixed by user in 477541e04)**: the
  `reject_unimplemented_gkd_objectives` deletion left a stale import in
  `cli/train_cli/agent_opd.rs`, which is `#[cfg(feature = "cuda")]`-gated —
  invisible to the Mac `metal,no-cuda` test lane and to the
  `cuda,no-cuda` checks that covered only cuda-kernels/infer-cuda/infer-api.
  The pod `--features cuda,nccl` build failed with E0432; the user dropped the
  import. Lesson: local verification must include
  `cargo check -p cli --no-default-features --features cuda,no-cuda` — it runs
  on Mac and covers the cli CUDA-gated surface.

## Learnings

PASS (local + remote). The CUDA C++ surface was the richest vein in the
repo — 12 whole .cu files and ~40 dead exports, most predating the paged-KV
unification. Zero runtime change: every deletion was zero-caller code, and
the one live-path touch (Marlin kernel arg) is verified on a real W8A16
decode at both 0.8B (correctness) and 27B (champion-row perf, ITL p50
16.72 vs 16.70 ms). The needle-gate distribution matches wave 1 cell-for-cell.
