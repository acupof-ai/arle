# 2026-05-05 · TileLang Phase 2a (GDR kernel skeleton) — pending-remote regression check

## Status: pending-remote (build-only, runtime path unchanged)

The Phase 2a commit adds `crates/cuda-kernels/tools/tilelang/gated_delta_rule.py`
— a literal 7-stage TileLang translation of the existing chunk-wise GDR Triton
kernels for the Qwen3.5 hybrid path. The commit is **build-only**: it does not
touch `build.rs`, the FFI declarations in
`crates/cuda-kernels/src/ffi/recurrent.rs`, or the call sites in
`infer/src/ops/recurrent.rs`. The Triton AOT pipeline remains the active
runtime path; the new `.py` is a parked artifact awaiting the AOT-generator
generalization in Phase 2b.

Per `CLAUDE.md` §Benchmarks, build-only diffs that don't change the runtime
path are eligible for the `bench: exempt` claim. This entry stands as the
formal regression-check artifact in line with Phase 0's discipline.

## Why scope-reduced from the original Phase 2 brief

The original Phase 2 brief (Decision D2 option A) asked for the literal
7-stage port plus the build.rs / FFI / call-site swap to land Triton-→-TileLang
in one round. Two upstream-survey constraints emerged during execution:

1. **`tools/tilelang/gen_tilelang_aot.py` is hard-coded for the 18-arg
   attention wrapper.** The generator's `TENSOR_NAME_TO_USER_INPUT` map only
   lists the 8 attention tensor names (Q, K_pool, V_pool, Q_indptr,
   KV_indptr, KV_indices, KV_last_page_len, Output) and `WRAPPER_FILL_RULES`
   only lists the attention symbolic scalars (batch_size, max_qlen,
   num_pages, total_pages, total_q_tokens). The 7 GDR stages have 7
   different signatures with 10-15 tensors each and different scalar mixes
   (`num_key_heads`, `num_value_heads`, `qkv_dim`, `seq_len`, `scale`).
   Generalizing the generator to dispatch wrapper emission by kernel
   family is a non-trivial engineering task that needs GPU validation
   per-stage to confirm TVM-FFI argument-order stability.

2. **FlashQLA (the closest upstream "TileLang GDR" reference) is sm_90 only.**
   FlashQLA uses Hopper-specific `T.gemm_v1`, `T.alloc_barrier`, warp-spec
   producer/consumer split, and TMA — none of which lower to ARLE's sm_75
   through sm_89 fat-build targets without major rewriting. The TileLang
   `.py` in this commit borrows the *algorithmic structure* (4-block
   solve_tril decomposition, fused chunk-state recurrence) but uses
   SM-portable primitives (`T.gemm`, `T.alloc_shared`, `T.alloc_fragment`,
   `T.Pipelined`) instead of FlashQLA's Hopper-only ones.

The honest conclusion: a full Phase 2 (skeleton + build swap + numerical-parity
validation) cannot land in one commit without GPU access. Splitting into
**Phase 2a — kernel skeleton** (this commit) and **Phase 2b — AOT generator
generalization + build swap + Triton deletion** (next round, GPU-required)
mirrors the Phase 0 / Phase 1 cadence and keeps Triton live until 2b ships.

## Pending bench (Phase 2b prerequisite)

When Phase 2b lands, the regression-check is a Qwen3.5-4B hybrid sweep against
the 2026-04-29 hybrid baseline (Phase 1 verdict pending — see
`memory/project_phase1_verdict_2026-04-30.md`). The acceptance bar from the
plan §3 Phase 2 gate is:

- `cargo test --release --test e2e_qwen35` green on L4 sm_89.
- Qwen3.5-4B JSON substring match (`infer/test_data/`).
- guidellm Qwen3.5-4B c=16 — Δ ≤ 5% out tok/s vs the most recent baseline.

```bash
source /tmp/arle-env.sh
CUDA_HOME=/usr/local/cuda cargo build --release -p infer --features cuda --bin infer
./target/release/infer \
  --model-path infer/models/Qwen3.5-4B \
  --port 8000 \
  --num-slots 16 \
  --max-seq-len 8192

# Separate terminal:
scripts/bench_guidellm.sh tilelang-phase2-qwen35 \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3.5-4B \
  --processor infer/models/Qwen3.5-4B \
  --concurrencies 16 \
  --max-seconds 120
```

## Hypothesis

Phase 2a alone: zero perf delta (build-only). The new `.py` file is not yet
referenced by `build.rs::compile_tilelang_aot_kernels`, so no extra cubins are
compiled and no extern symbols change.

Phase 2b (forecast): TileLang chunk-wise GDR within ±5% of Triton on
Qwen3.5-4B c=16 prefill is achievable based on:

- Triton AOT and TileLang AOT both lower through nvcc with similar tile-shape
  lattices for Qwen3.5's fixed (BT=64, KEY_DIM=128, VALUE_DIM=128).
- The 2026-04-28 HD128 prefill bench (`docs/experience/wins/2026-04-28-bench-guidellm-cuda-l4-tilelang-prefill-causal-bound.md`)
  showed TileLang reaching parity with FlashInfer on a more demanding kernel
  family; chunk-wise GDR has a smaller surface to tune.
- Stage 4 solve_tril is the highest-risk piece — the Triton implementation
  uses fp32 IEEE-precision dot products (`input_precision="ieee"`) for
  numerical stability; TileLang's default GEMM accum is fp32, so the
  numerics should match without an explicit precision flag.

## Reviews summary

- Round 1 self-review (Claude): see commit body
- Round 2 `cargo check --no-default-features --features no-cuda`: PASS
- Round 3 `codex review --uncommitted`: deferred to Phase 2b (this commit's
  diff is one new `.py` file plus this entry plus a plan update — review
  surface is small and the runtime path is unchanged)

## Pointers

- Plan: `docs/plans/2026-05-05-cuda-kernel-tilelang-unification.md` §3 Phase 2
- New TileLang skeleton: `crates/cuda-kernels/tools/tilelang/gated_delta_rule.py`
- Existing Triton source (still active): `crates/cuda-kernels/tools/triton/gated_delta_rule_chunkwise_kernels.py`
- Upstream FLA (Apache-2.0): https://github.com/fla-org/flash-linear-attention
- Upstream FlashQLA (MIT, e88d71a1): https://github.com/QwenLM/FlashQLA
- Phase 0 stub for reference: `docs/experience/wins/2026-05-05-bench-tilelang-phase0-pending-remote.md`
- AOT-generator constraint: `crates/cuda-kernels/tools/tilelang/gen_tilelang_aot.py`
- TileLang 0.1.9 AOT recipe: `memory/project_tilelang_0p1p9_aot_blocker.md`
