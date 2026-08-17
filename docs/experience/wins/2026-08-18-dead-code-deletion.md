# Global dead-code deletion — all backends, 2026-08-18

> Status: pending-remote (local verified; CUDA build + parity gate on pod)

## Goal

Delete non-main-path code verified as dead (zero callers) across the workspace,
then confirm zero runtime change via the correctness parity gate.

## Scope

12 commits, 65 files, **−6,818 net lines** (7,155 deleted − 337 restored). Every
deletion verified as zero-caller before removal; one cluster (GDR FFI) was
wrongly deleted and restored (see Problems).

| Chunk | Crate | Deleted | Lines |
|-------|-------|---------|------:|
| tilelang.rs metadata module | cuda-kernels | TileLangWorkspace/TileLangDecodeMetadata never constructed; setter wrote atomic read only by dead code | 1,135 |
| paged_kv migrate cluster | cuda-kernels | 8 migrate methods + 8 dead accessors + build_paged_kv_metadata + PagedKVBatchMeta + retained_count/free_count/cow_tail/evict_deferred | 552 |
| scattered dead code | cuda-kernels | kv_turboquant.rs (4 fns), 3 _raw attention fns, kv_types is_turboquant/pool_bytes_per_kv_head, moe device_vec_ptr, turboquant_state is_hadamard | 474 |
| FFI + .cu dead kernels | cuda-kernels | 3 dead .cu files (fused_attention, w4_fp8/w4a8 activation quant), dequant_paged_kv dead half, 2 test-only quant FFI + tests. **27 GDR FFI declarations deleted then restored** (see Problems) | 1,278 |
| collective_ep cluster | autograd | ops/collective_ep.rs (303 lines), EpExchangeCtx + 3 Ep* BackwardOp variants, ep_exchange_rows_device trait+impl, backward_accumulate_targets trio, is_device_backed, set_backend, zeros_like, serde_json dep | 566 |
| iso_spectrum + dead fns | spec-train | iso_spectrum.rs (387 lines), density(), trained_rows(), ahash dep, 3 no-op features | 422 |
| 56 dead FFI declarations | mlx-sys | qwen35_compiled_* session API (23), superseded mlx_fast_*/mlx_quantize/etc., mlx_metal_kernel_*, mlx_gguf_*, tape replay, GDR, misc math | 556 |
| tensor.rs + kv_quant.rs | cuda-kernels | 22 dead methods (quantized constructors, predicates, copy/slice, device/context, alloc-trace query), 7 dead kv_quant fns + 2 consts, 2 dead tests, cascades | 1,677 |
| cross-crate scattered | infer-server/api/cuda/seam/cli/train/chat/kv-native | collect_timeout, 5 relay dead methods, has_media_content ×2, from_model_dir_without_chat, 5 dead re-exports, with_sampling/with_stop/as_openai_str/with_images/from_chain, set_bypass, with_admission, fuzzy_filter+nucleo dep, write_file_atomic_cache, aopd_profile enabled(), entropy_weight stub, image-input cluster, --gkd-entropy-weight, --marlin-w4-fp8-prefill, MARLIN_W4_FP8_PREFILL static+setter, vestigial device-ordinal override | 318 |
| remaining + xgrammar wiring | deepep-sys/infer-api/cli/chat/scripts/docs | deepep dead forwarding, gen_arle_longctx_eval.py, INFER_TRITON_PYTHON, vendor/mlx-sys, stale docs; xgrammar real feature wired (cli/grammar → infer-api/grammar → infer-server/grammar → xgrammar-sys/real) | 172 |
| caller fixes | cli/train | aopd_profile env check inline, reject_unimplemented_gkd_objectives signature update | 5 |

## What was NOT deleted (live, not dead)

- `--la-backward-mono` / `--autograd-decode-attn-legacy` — A/B escape hatches, live callers
- `--no-fused-distill` — live A/B flag, has a test
- TokenKVPool core (paged_kv.rs) — live, only dead migrate/accessor methods removed
- TileLang JIT compiler in build.rs — live, compiles CUDA kernels at build time
- GDR generic path (gdr_fq_prep/cumsum/kkt/fwd_cuda) — live, called from infer-cuda
- GDR head-specific FFI declarations (h48/h24g8/h12g4/h16g8/h16g16) — live, referenced by build.rs-generated `FLASHQLA_GDR_TABLE`
- Vulkan/HIP experimental backends — active development
- xgrammar `real` feature — wired per user request, not deleted

## Verification

### Local (Mac, Metal + cpu features)

```
cargo check -p infer-api --release --no-default-features --features metal,no-cuda --lib  → clean
cargo check -p cli -p train --release --no-default-features --features metal,no-cuda     → clean
cargo check -p autograd --release --no-default-features --features no-cuda               → clean
cargo check -p spec-train --release --no-default-features                                → clean
cargo check -p mlx-sys --release                                                         → clean
cargo check -p infer-server --release --no-default-features                              → clean
cargo clippy -p <all changed crates> -- -D warnings                                      → zero warnings
cargo test -p arle --profile release-fast --features cpu,no-cuda,cli                     → 5/5 pass
cargo test -p cli --release --features metal,no-cuda                                     → 4/4 pass
```

### Remote (CUDA build + parity) — pending

- `cargo build --release --features cuda` on pod
- `scripts/needle_gate.py` ×3 + `scripts/lever_gate.sh`
- Expected: zero behavior change (all deletions are zero-caller code)

## Problems

- **GDR FFI declarations wrongly deleted (restored, 337 lines)**: The 27 GDR
  FFI declarations in `ffi/recurrent.rs` appeared dead on Mac — their only
  caller is `flashqla_gdr_generated.rs`, a build.rs-generated lookup table
  compiled under `#[cfg(feature = "cuda")]`, invisible in Mac no-cuda builds.
  V100 CUDA build failed with 27× E0425. Restored all 27 declarations
  (`62349f463`). Lesson: generated-code callers behind cfg gates don't show up
  in `rg` on the host platform.
- Plan estimated 78 dead FFI declarations in cuda-kernels; actual count was 25
  (the plan's scan included functions with C++ internal callers). Verified each
  individually before deletion.
- `--no-fused-distill` was flagged as a no-op in the plan; it is actually a live
  A/B flag with a test. Kept.
- Pre-existing bug found and fixed: `kv_types` re-export in cuda-kernels/lib.rs
  was ungated while the module is `#[cfg(feature = "cuda")]` — broke no-cuda
  builds. Fixed by gating the re-export.
- Two follow-ups from the tensor.rs subagent: MARLIN_W4_FP8_PREFILL was
  write-only after reader deletion (removed the full chain: CLI flag → config
  field → setter → static); device-ordinal override was vestigial after
  with_device_ordinal_override deletion (collapsed to direct env parse).

## Learnings

PASS (local). All deletions are zero-caller code — no runtime change expected.
Remote CUDA build + parity gate is the final verification.
