# Qwen3.5-MoE / Qwen3.6 CUDA SOTA-grouped forward (BF16) — 2026-05-30

**Status: pending-remote** (CUDA path; cannot run on Mac dev box — needs a
V100/sm_70 smoke pod). Bench is a stub per CLAUDE.md §Benchmarks
("can't run locally → cite remote ticket, stub with `pending-remote`").

## Context

The CUDA Qwen3.6/Qwen3.5-MoE forward was a `todo!()` stub
(`infer/src/backend/cuda/bootstrap.rs::load_qwen35_moe_components`). Metal had
its own MoE path (`mlx_qwen35_moe_block.cpp`); CUDA panicked on MoE load.

This change implements the single-GPU **SOTA-grouped** MoE forward
(permute → grouped GEMM → combine; NOT a per-expert loop), reusing the
DeepSeek-V4 grouped-MoE kernel pipeline. Correctness-first BF16; the W4
grouped GEMM + perf tuning is an explicit follow-up.

## What landed

Pipeline (mirrors DSv4 compact grouped path + the Metal reference):
`route (plain softmax + top-k) → optional norm_topk_prob renorm →
per-expert count → exclusive-scan offsets → permute/pack grouped-by-expert →
grouped gate+up GEMM (paired) → SwiGLU → grouped down GEMM →
scale-by-route-weight + scatter-accumulate to token rows →
+ shared expert (dense SwiGLU) × sigmoid(x @ shared_expert_gate)`.

**New BF16 CUDA kernels** (sm_70-safe — CUDA-core warp-reduce, no mma/FP8):
- `crates/cuda-kernels/csrc/gemm/moe_grouped_gemm.cu` —
  `moe_bf16_grouped_gemm_pair_batch_cuda` (gate+up) and
  `moe_bf16_grouped_gemm_batch_cuda` (down). BF16 mirror of the DSv4 FP8
  grouped GEMM structure (M-grouping by expert via offsets/counts,
  DSV4_BATCH_TILE=32-way M reuse) with no quant decode — just
  `__bfloat162float` MAC.
- `crates/cuda-kernels/csrc/moe/qwen36_route.cu` —
  `qwen36_renorm_topk_weights_cuda` (`norm_topk_prob` renorm over the
  dsv4_route weight buffer) and `qwen36_add_shared_expert_gated_cuda`
  (shared-expert sigmoid gate + accumulate).

**Reused DSv4 kernels** (dtype-agnostic on BF16 activations): `dsv4_route_cuda`
(scoring_kind=0 softmax + routing_kind=1 block-argmax top-k, zeroed bias),
`dsv4_count_local_experts_cuda`, `dsv4_exclusive_scan_i32_cuda`,
`dsv4_pack_local_experts_cuda`, `dsv4_scatter_packed_expert_cuda`.

**Rust** (`infer/src/model/qwen35/`):
- `moe.rs` (new) — `MoeMlp` weight struct + `Mlp { Dense, Moe }` enum +
  `load_moe_mlp` loader + grouped `forward`.
- `weights.rs` — `TransformerBlock35.mlp` is now `Mlp`; loader branches on
  `Config35::is_moe_layer(i)` (MoE vs dense). Offload/reload/marlin/parity
  paths MoE-guarded (dense-only OPD).
- `prefill.rs` / `batch_decode.rs` / `diagnostics.rs` — 7 MLP call sites
  branch `as_moe()` → grouped MoE, else the **unchanged** dense gemm-into
  chain via `mlp.dense()`. Dense path is bit-identical.
- `bootstrap.rs` — stub → real `Qwen35Model::from_safetensors` load.

**Tensor naming supported** (HF safetensors, prefix
`model.language_model.layers.N`): router `mlp.gate(.weight)`; experts
`mlp.experts.{i}.{gate_proj,up_proj,down_proj}(.weight)`; shared
`mlp.shared_expert.{gate,up,down}_proj(.weight)`; shared gate
`mlp.shared_expert_gate(.weight)`. Both `.weight`-suffixed and bare names
resolve. The stacked `switch_mlp.*` convention is detected and rejected with
a clear error (expert-axis slicing is the follow-up; the BF16 smoke model
`tiny-random/qwen3.5-moe` uses the per-expert layout).

## Done vs TODO

- **Done**: BF16 grouped forward, end-to-end, typechecks under `cuda,no-cuda` +
  `metal,no-cuda`; dense path bit-identical; per-expert HF tensor layout.
- **TODO (follow-up, scoped, not half-state)**:
  - W4 nibble-decode grouped GEMM variant (Qwen3.6-35B-A3B-4bit production
    weights). The BF16 path is the correctness reference.
  - Perf: scratch-buffer reuse (current `forward` allocates per-call),
    `max_count` per-expert tightening, CUDA-graph capture (MoE forces
    graphsafe=false today), W4 Marlin.
  - Stacked `switch_mlp.*` expert-axis slicing.
  - TP/EP for MoE (single-GPU only today; multi-GPU MoE errors at load).
  - GGUF MoE load (errors; use safetensors).

## Verification (Mac, no nvcc)

- `cargo check -p infer --no-default-features --features cuda,no-cuda` — clean.
- `cargo check -p infer --no-default-features --features metal,no-cuda` — clean.
- `cargo check -p cuda-kernels --no-default-features --features cuda,no-cuda` — clean.
- `cargo test -p qwen35-spec --lib` — 30 passed (MoE config parse).
- `.cu` kernels not compiled (no nvcc on Mac) — pending-remote V100 build.

## Remote bench TODO

On a CUDA pod (sm_70+): build with nvcc, load `tiny-random/qwen3.5-moe`
(BF16) for a correctness smoke (greedy-decode a known prompt, compare tokens
vs the Metal MoE path or HF reference), then a `scripts/bench_guidellm.sh`
run vs the dense Qwen3.5 baseline. W4 grouped GEMM is the perf follow-up.

## Rule

Reuse the DSv4 grouped kernel pipeline for any new single-GPU MoE model —
route/pack/swiglu/scatter are dtype-agnostic on BF16 activations; only the
grouped GEMM (per quant format) and the route-scoring branch are new.
