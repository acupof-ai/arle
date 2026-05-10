# DSv4 1B + Qwen3.6 CUDA Substrate Audit

Date: 2026-05-10
Status: Phase 0 audit for human review
Scope: ARLE-native CUDA `infer` substrate only. No llama.cpp, ollama, Triton,
or external runtime.

## Goal

Land the substrate that can eventually run both:

1. `infer/models/dsv4-mini-1B-init/` DeepSeek V4 1B init checkpoint.
2. Qwen3.6 / Qwen3.5-MoE 35B-A3B 4-bit checkpoint under CUDA on a 16 GB
   RTX 4070 Ti SUPER by expert offload plus 8-bit KV.

This document is the Phase 0 gate. It does not license implementation work.
Implementation must stop here until the phase plan below is reviewed.

## Sources Read

- Root `AGENTS.md` contract and module guides:
  - `infer/src/model/AGENTS.md`
  - `crates/cuda-kernels/AGENTS.md`
- Global docs:
  - `docs/index.md`
  - `docs/codebase-map.md`
  - `docs/architecture.md`
  - `docs/support-matrix.md`
  - `ROADMAP.md`
  - `docs/bench-and-trace-spec.md`
- DeepSeek docs:
  - `docs/projects/2026-05-01-deepseek-v4-readiness.md`
  - `docs/plans/2026-05-01-mla-kernel-design.md`
  - `docs/plans/2026-05-05-deepseek-v4-small-substrate.md`
- Qwen3.6 prior evidence:
  - `docs/experience/wins/2026-05-07-bench-qwen36-baseline.md`
  - `docs/experience/wins/2026-05-07-bench-qwen36-moe-swiglu-fusion.md`
  - `docs/experience/wins/2026-05-07-bench-qwen36-topk-sweep.md`
- Local code and assets:
  - `infer/models/dsv4-mini-1B-init/{config.json,model.safetensors,README.md}`
  - `crates/deepseek-spec/src/{lib.rs,v4.rs}`
  - `infer/src/model/deepseek/*`
  - `infer/tests/dsv4_nano_smoke.rs`
  - `infer/src/model_registry.rs`
  - `infer/src/backend/cuda/bootstrap.rs`
  - `infer/src/backend/metal/config.rs`
  - `crates/mlx-sys/src/mlx_qwen35_moe_block.cpp`
  - `crates/cuda-kernels/{AGENTS.md,csrc/,src/ffi/,src/paged_kv.rs,src/kv_quant.rs}`

Plan agent used first, per mission request. Its read-only conclusion is folded
into this document: the starting phase outline contains a DSv4 architecture
mismatch and must be corrected before any kernel work starts.

## Workspace Guard

`git status -sb` before this document showed unrelated dirty files:

- `infer/src/model.rs`
- `infer/src/model/qwen35/forward.rs`
- `infer/src/model/qwen35/recurrent_state.rs`
- `infer/src/scheduler/cuda/spec_path.rs`

Those paths are not touched by Phase 0. In particular,
`infer/src/model/qwen35/recurrent_state.rs` is owned by the concurrent Medusa
work and remains out of scope.

## Executive Finding

Phase 0 kills one core assumption from the starting outline:

**The local DSv4 1B checkpoint is not the old dense/no-MoE MLA nano fixture.**

The actual checkpoint at `infer/models/dsv4-mini-1B-init/config.json` declares:

- `architectures = ["DeepseekV4ForCausalLM"]`
- `model_type = "deepseek_v4"`
- `hidden_size = 1024`
- `num_hidden_layers = 24`
- `num_key_value_heads = 1`
- `q_lora_rank = 384`
- `o_lora_rank = 384`
- `o_groups = 4`
- `qk_rope_head_dim = 32`
- `n_routed_experts = 16`
- `n_shared_experts = 1`
- `num_experts_per_tok = 2`
- `scoring_func = "sqrtsoftplus"`
- `topk_method = "noaux_tc"`
- `num_hash_layers = 2`
- `sliding_window = 64`
- `num_nextn_predict_layers = 1`
- `vocab_size = 129280`

The safetensors header confirms V4-style tensor names such as:

- `embed.weight`
- `hc_head_base`
- `layers.0.attn.wq_a.weight`
- `layers.0.attn.wkv.weight`
- `layers.0.ffn.gate.weight`
- `layers.0.ffn.experts.<id>.w{1,2,3}.weight`
- `layers.0.ffn.shared_experts.w{1,2,3}.weight`

The existing runtime scaffold under `infer/src/model/deepseek/*` still uses
`deepseek_spec::DeepSeekConfig`, which is the V3-era MLA config. Its comments
and method bodies assume "MLA + dense MLP"; `from_config`, `from_safetensors`,
decode, batch decode, and sampling are still `todo!()` stubs.

Therefore:

- The original "Phase 2A - DS3 MLA kernel + DSv4 1B smoke pass" cannot be
  licensed for the named 2.0 GB checkpoint.
- `docs/plans/2026-05-01-mla-kernel-design.md` remains useful prior art for
  V2/V3 MLA and compressed attention kernel structure, but it is not sufficient
  for this DSv4 1B target.
- The next implementation phase must first align runtime DeepSeek code with
  `DeepSeekV4Config`, not add MLA kernels against the wrong scaffold.

Qwen3.6 has a separate but clear blocker:

- `ModelArch::Qwen3_5_Moe` exists.
- CUDA `load_qwen35_moe_components` is an explicit
  `todo!("GPU required: Qwen3.6 CUDA not yet implemented")`.
- `docs/support-matrix.md` marks Qwen3.6 as "Beta (Metal), CUDA stub".
- No Qwen3.6 checkpoint is present under `infer/models/` or the local
  Hugging Face cache on this host. Existing config evidence comes from the
  2026-05-07 Metal audit: 40 layers, hidden 2048, 256 experts, top_k 8,
  `moe_intermediate_size = 512`, no MTP tensors, multimodal weight footprint.

## Hardware Evidence

Observed local GPU:

- NVIDIA GeForce RTX 4070 Ti SUPER
- 16,376 MiB total memory
- 1,323 MiB used at audit time
- 0% GPU utilization at audit time

Local model sizes:

- `infer/models/dsv4-mini-1B-init`: 2.0 GB
- largest local Qwen3 4B BF16 path: 15 GB
- Qwen3.6 35B-A3B 4-bit: not cached locally

Qwen3.6 cannot be loaded monolithically on 16 GB once weights, KV, scratch,
runtime allocations, and the multimodal footprint are counted. Expert offload
is a hard dependency, not an optimization.

## Current Support State

### DeepSeek V4

What exists:

- `crates/deepseek-spec/src/v4.rs` parses the actual `deepseek_v4` checkpoint
  shape and tensor-name contract.
- `infer/src/model_registry.rs` maps `DeepseekV4ForCausalLM` to
  `ModelArch::DeepSeekV4` and classifies its attention as
  `DeepSeekV4Hybrid`.
- `infer/src/model/deepseek/*` provides a skeleton that implements the
  `ModelForward` type shape.
- `infer/tests/dsv4_nano_smoke.rs` exists but is ignored.

What is blocked:

- CUDA runtime bootstrap does not accept DeepSeek V4 as a loadable model.
- `infer/src/model/deepseek/config.rs` wraps the old `DeepSeekConfig`, not
  `DeepSeekV4Config`.
- `infer/src/model/deepseek/weights.rs` describes "MLA + dense MLP" and
  defers MoE.
- `infer/src/model/deepseek/forward.rs` returns MLA TODOs for decode, batch
  decode, and sampling.
- `infer/tests/dsv4_nano_smoke.rs` uses `DeepSeekConfig::nano()`, a synthetic
  2-layer old MLA fixture, not the 2.0 GB checkpoint named in this mission.

### Qwen3.6 / Qwen3.5-MoE

What exists:

- Architecture detection for `Qwen3_5MoeForCausalLM` and
  `Qwen3_5MoeForConditionalGeneration`.
- Metal config and model code for `mlx-community/Qwen3.6-35B-A3B-4bit`.
- Metal MoE semantic reference in `mlx_qwen35_moe_block.cpp`.
- CUDA Qwen3/Qwen3.5 dense paths and W4/W8 quantized GEMM kernels.

What is blocked:

- CUDA MoE loader is a stub.
- CUDA model dispatch still routes `Qwen35Moe` through `Qwen35Model` component
  type as a placeholder.
- The actual Qwen3.6 CUDA checkpoint format is not locally present. The plan
  must inspect HF native / AWQ / GPTQ / MLX 4-bit metadata before committing to
  loader code.
- Qwen3.6 4-bit weights exceed practical 16 GB VRAM residency, so weight-tier
  infrastructure is mandatory before end-to-end decode.

## Kernel Reuse Map

| Need | Reuse level | Existing surface | Audit verdict |
|---|---:|---|---|
| CUDA tensor ownership, streams, device buffers | High | `crates/cuda-kernels/src/tensor.rs`, prelude | Reuse directly. |
| Runtime scheduler / slots / sampling shell | High | `infer/src/scheduler/cuda/*`, `ModelForward`, `sampling.cu` | Reuse once model modules implement the contract. |
| RMSNorm / fused add norm | High | `crates/cuda-kernels/csrc/misc/norm.cu` | Reuse directly for both families. |
| BF16 GEMM / GEMV | High | `ffi/gemm.rs`, `gemm/gemv.cu`, `gemm/marlin_kernel.cu` | Reuse for correctness baselines and dense/shared experts. |
| W4/W8 quantized GEMM | Medium | Marlin W4A16/W4A8, W4+FP8, quantized GEMV | Reuse for Qwen3.6 experts only after checkpoint quant format is proven compatible. |
| Fused SwiGLU MLP | Medium | `misc/fused_mlp.cu` | Reuse dense/shared-expert pieces; MoE routing/dispatch still new. |
| Paged KV pool metadata | Medium | `TokenKVPool`, `PagedKVBatchMeta` | Reuse allocator/page-table ideas. DSv4 V4 attention needs custom payload semantics. |
| INT8 / FP8 KV storage | Medium | `kv_quant.rs`, `csrc/kv/kv_quant.cu` | Reuse for "8-bit KV" if exact q8_0 is not required. Exact q8_0 needs new format work. |
| Varlen split-KV scheduling | Pattern only | `decode_attention_varlen_fp8.cu` | Reuse launch/partial-merge pattern, not inner math. |
| V3 MLA ABI | Low | `ffi/mla.rs`, `attention/mla_decode.cu` | Placeholder returns not-supported. Not enough for DSv4 1B. |
| DSv4 V4 hybrid attention | None | N/A | New kernels/model glue required. |
| CUDA MoE router + dispatch + combine | None | N/A | New shared primitive required. |
| Weight offload / expert tier | None | Existing KV tier only | New weight-tier substrate required for Qwen3.6. |

## MoE Commonality and Divergence

A common MoE primitive is still justified, but only if it is parameterized.

Shared shape:

- Compute router logits.
- Select top-k experts per token.
- Compute expert SwiGLU for selected routes.
- Weight and combine selected expert outputs.
- Add shared expert output where the architecture has one.

Qwen3.6 specifics:

- `num_experts = 256`, `top_k = 8`, hidden 2048, MoE hidden 512.
- Router semantics in Metal reference: softmax, top-k argpartition, optional
  top-k normalization.
- Router and shared gate are 8-bit in the MLX 4-bit checkpoint; switch experts
  and shared expert are 4-bit.
- Existing Metal semantic reference uses `SwitchGLU` / `gather_qmm` style
  expert selection.

DSv4 1B specifics:

- `n_routed_experts = 16`, `top_k = 2`, hidden 1024, MoE hidden 512.
- Router config is `scoring_func = "sqrtsoftplus"` and
  `topk_method = "noaux_tc"`.
- Early hash-routed layers exist (`num_hash_layers = 2`) and safetensors include
  `layers.N.ffn.gate.weight` plus expert `w1/w2/w3`.
- Shared expert is present.

Conclusion:

- A Qwen3.6-only MoE kernel is not enough.
- A DSv4-only MoE kernel is not enough.
- Phase 1 should implement a router-policy parameter and expert-layout
  abstraction first, then license optimized grouped/per-expert kernels against
  a BF16 correctness baseline.

## Revised Phase Plan

### Phase 0 - Audit and Architecture Freeze

Deliverable:

- This document.

License gate:

- Human review explicitly accepts the corrected phase tree.
- Human either licenses or rejects the DSv4 pivot away from "DS3 MLA first".

Verdict:

- **KILL** the assumption "the named DSv4 1B checkpoint is a dense/no-MoE MLA
  nano fixture".
- **LICENSE FOR REVIEW ONLY** the revised V4Config-based path below.

Stop:

- Stop after this document and local docs commit.

### Phase 0.5 - DSv4 Truth Alignment

Purpose:

- Convert the runtime DeepSeek surface from old `DeepSeekConfig` assumptions to
  an explicit V4 shape before kernels are added.

Expected work:

- Add or split a V4 runtime config that wraps `DeepSeekV4Config`.
- Update the smoke-test plan so the acceptance target is the 2.0 GB
  `infer/models/dsv4-mini-1B-init/` checkpoint, not only
  `DeepSeekConfig::nano()`.
- Add a clean CUDA unsupported/error path if runtime loading is intentionally
  still blocked at this phase.
- Do not implement kernels yet.

Estimated LOC:

- 500-900 LOC.

License gate:

- Config parse against the local checkpoint.
- Tensor-name manifest coverage against `DeepSeekV4Config`.
- `cargo check` gates green.
- No qwen35 file touches.

Stop:

- Stop for review before CUDA kernel work.

### Phase 1 - Shared CUDA MoE Primitive

Purpose:

- Build the shared router + top-k + expert dispatch + combine substrate used by
  DSv4 and Qwen3.6.

Expected work:

- New CUDA MoE kernel/API surface under `crates/cuda-kernels`.
- Use `csrc/moe/` only if the implementation has at least three source/header
  files; otherwise follow `crates/cuda-kernels/AGENTS.md` and place the first
  kernel under the closest existing domain.
- Rust API/FFI for BF16 correctness baseline.
- Synthetic test: 4 experts x 8 tokens, top-2, BF16 weights, deterministic
  expected shape and finite output.
- Bench against naive per-expert loop.

Estimated LOC:

- 1,200-2,200 LOC for BF16 primitive, tests, and bench doc.
- Add 700-1,500 LOC if W4 expert GEMM support is included in the same phase.

Formula-predict:

- BF16 optimized route should be at least 3x faster than a naive per-expert
  loop for the synthetic bench shape.

License gate:

- Synthetic correctness test passes.
- Bench shows >=3x over naive loop, or Phase 1 is killed/pivoted with an
  errors entry.
- Numbers are evaluated on wall-clock kernel/API time, not only a narrow
  internal launch window.

Stop:

- Stop after wins/errors entry and local commit.

### Phase 2A - DSv4 1B V4 CUDA Smoke

Purpose:

- Make the named DSv4 1B checkpoint load and run enough CUDA forward/decode to
  satisfy mission acceptance (A).

Expected work:

- Wire `DeepSeekV4Config` into `infer/src/model/deepseek/*` or split V3/V4
  modules cleanly.
- Implement safetensors loading for V4 tensor names.
- Implement V4 attention path needed by the 1B checkpoint:
  Q-LoRA, single KV head, O-LoRA grouping, compression/indexer streams,
  sliding-window / hybrid attention, and hyperconnection state as required by
  the config.
- Use Phase 1 MoE for `DeepseekMoE`.
- Handle MTP tensors as loaded/deferred based on the forward path; do not let
  MTP tensors silently mismatch.
- Un-ignore and update `infer/tests/dsv4_nano_smoke.rs` or add a correctly
  named test for the 2.0 GB checkpoint.

Estimated LOC:

- 2,500-5,000 LOC.

Formula-predict:

- DSv4 1B fits on the 16 GB GPU with no offload.
- Greedy 8-token decode is deterministic and finite, but quality is not
  asserted because the checkpoint is randomly initialized.

License gate:

- `cargo test --release -p infer --features cuda --test dsv4_nano_smoke`
  passes without `#[ignore]`.
- Logits shape is `[seq, vocab]`.
- Greedy 8-token decode is deterministic across two runs and has no NaN/Inf.
- Wins entry lands with runtime logs.

Stop:

- Stop after Phase 2A report.

### Phase 2B - Qwen3.6 CUDA Model Shell and Loader

Purpose:

- Load Qwen3.6/Qwen3.5-MoE config and run a 1-token forward shape test under
  CUDA before offload.

Expected work:

- Inspect actual checkpoint config and safetensors/quant metadata first.
- Add CUDA-side Qwen3.5-MoE model module without editing qwen35 files unless
  the owner explicitly clears the overlap.
- Reuse Qwen3.5 hybrid attention semantics where possible through stable
  interfaces, not by modifying dirty qwen35 work.
- Use Phase 1 MoE primitive for sparse MoE block.
- Support either a HF native FP16/BF16 path or a supported AWQ/GPTQ 4-bit path.

Estimated LOC:

- 1,500-3,000 LOC excluding offload.

Formula-predict:

- Full 35B-A3B 4-bit weights will not fit on 16 GB without offload, so this
  phase must not claim end-to-end viability.

License gate:

- Config and tensor manifest audited from the actual checkpoint.
- 1-token forward shape succeeds on a reduced/synthetic or staged subset, with
  non-NaN logits.
- No accidental edits to `infer/src/model/qwen35/*`.

Stop:

- Stop after Phase 2B report.

### Phase 3A - Qwen3.6 Expert Weight Offload

Purpose:

- Add the CPU/GPU expert weight tier needed to run Qwen3.6 35B-A3B 4-bit on
  16 GB VRAM.

Expected work:

- New weight-tier abstraction analogous in discipline to KV tier, but for
  immutable model weights.
- CPU storage for inactive experts.
- GPU staging buffer for active experts.
- Measured prefetch/compute overlap where possible.
- Runtime counters for staged bytes, active experts, stalls, and GPU memory.

Estimated LOC:

- 1,800-3,200 LOC.

Formula-predict:

- Qwen3.6 should fit because only active expert slices plus shared weights,
  attention weights, KV, and staging buffers occupy GPU memory at a time.
- Decode target remains speculative until measured; 5-15 tok/s is a hypothesis,
  not evidence.

License gate:

- Model loads within 16 GB measured GPU memory.
- Prompt `"Hello, world!"` greedy decode emits finite tokens.
- Bench wins entry reports tok/s, TTFT/ITL, GPU memory, staged bytes, and stalls.

Stop:

- Stop after Phase 3A report.

### Phase 3B - 8-bit KV

Purpose:

- Reduce KV memory pressure before long-context Qwen3.6 benching.

Expected work:

- Prefer reusing existing INT8 or FP8 paged KV infrastructure first.
- Only implement exact q8_0 if the project requires that specific block
  format rather than generic 8-bit KV.
- Validate on Qwen3-4B before applying to Qwen3.6.

Estimated LOC:

- 300-700 LOC if existing INT8/FP8 paged KV is sufficient.
- 800-1,500 LOC if exact q8_0 layout and kernels are required.

Formula-predict:

- 8-bit KV should reduce KV storage by at least 40% versus BF16 after scale
  overhead.

License gate:

- Qwen3-4B quality drift <0.5% on the chosen perplexity/eval gate.
- KV memory reduction >=40%.
- Qwen3.6 long-context run uses the licensed KV format.

Stop:

- Stop after Phase 3B report.

### Phase 4 - End-to-End Bench and Docs Integration

Purpose:

- Close the initiative with measured end-to-end evidence and update global
  truth surfaces.

Expected work:

- DSv4 1B single-prompt latency/decode smoke bench.
- Qwen3.6 35B-A3B 4-bit conc=1 bench with offload and 8-bit KV.
- `docs/experience/wins/` entries following `docs/bench-and-trace-spec.md`.
- Update `ROADMAP.md`, `docs/support-matrix.md`, and relevant project docs.

Estimated LOC:

- 300-600 LOC, mostly docs and bench snapshots.

License gate:

- Bench entries include Goal, Hypothesis, Params, Env, Results, Problems, and
  Learnings.
- Claims are based on wall-clock/request-level metrics.
- Industry comparisons are literature-only and explicitly labeled as such.

Stop:

- Stop after final report.

## Dependency Tree

```text
Phase 0 audit
  -> Phase 0.5 DSv4 V4 truth alignment
      -> Phase 1 shared MoE primitive
          -> Phase 2A DSv4 1B V4 CUDA smoke
          -> Phase 2B Qwen3.6 CUDA loader/model shell
              -> Phase 3A expert weight offload
                  -> Phase 3B 8-bit KV
                      -> Phase 4 e2e bench/docs
```

Rationale:

- DSv4 actual checkpoint and Qwen3.6 both need MoE.
- DSv4 actual checkpoint must align to V4 config before any MLA/V4-attention
  kernel choice is made.
- Qwen3.6 cannot license end-to-end on 16 GB until expert offload exists.
- 8-bit KV is useful but should not hide the larger weight-residency blocker.

## File Touch Map

Phase 0:

- `docs/plans/2026-05-10-dsv4-qwen36-substrate-audit.md`

Likely Phase 0.5:

- `crates/deepseek-spec/src/v4.rs`
- `infer/src/model/deepseek/*`
- `infer/tests/dsv4_nano_smoke.rs`
- `infer/src/model_registry.rs`
- `infer/src/backend/cuda/bootstrap.rs`

Likely Phase 1:

- `crates/cuda-kernels/csrc/moe/*` or closest existing domain if fewer than
  three files
- `crates/cuda-kernels/src/ffi/*`
- `crates/cuda-kernels/src/moe.rs`
- focused kernel tests/benches

Likely Phase 2A:

- `infer/src/model/deepseek/*`
- `infer/tests/dsv4_nano_smoke.rs`
- DeepSeek CUDA attention/MoE FFI and kernels

Likely Phase 2B/3A:

- new Qwen3.5-MoE CUDA module files
- CUDA bootstrap/model dispatch
- weight-tier module files

Explicit non-touch until reauthorized:

- `infer/src/model/qwen35/recurrent_state.rs`
- other dirty qwen35 files owned by concurrent Medusa work
- `crates/cuda-kernels/csrc/quant/marlin*` unless a later phase explicitly
  needs and coordinates Marlin changes

## License-Or-Kill Decisions From Phase 0

1. KILL: "DSv4 1B nano is dense/no-MoE MLA and can be unlocked by only landing
   DS3 MLA kernels."
2. LICENSE FOR REVIEW: "DSv4 acceptance should target `DeepSeekV4Config` and
   the local 2.0 GB checkpoint tensor contract."
3. LICENSE FOR REVIEW: "A shared MoE primitive is worth Phase 1, but only with
   router-policy and layout parameterization."
4. KILL: "Qwen3.6 4-bit can fit monolithically on 16 GB."
5. DEFER: "Exact q8_0 KV is required." Existing INT8/FP8 paged KV may satisfy
   the mission's "8-bit KV" unless human review requires q8_0 specifically.

## Open Review Questions

1. Should Phase 0.5 be inserted before Phase 1 to align the DSv4 runtime with
   `DeepSeekV4Config`, or should Phase 1 MoE proceed first while DSv4 alignment
   is reviewed separately?
2. Should the DSv4 test keep the name `dsv4_nano_smoke` while targeting the
   2.0 GB checkpoint, or should it be renamed to avoid confusing the old
   `DeepSeekConfig::nano()` fixture with the actual 1B checkpoint?
3. For Qwen3.6 CUDA, is the target checkpoint HF native FP16/BF16, AWQ/GPTQ
   4-bit, or an MLX 4-bit checkpoint converted into an ARLE-supported format?
4. Does "8-bit KV" mean existing ARLE INT8/FP8 paged KV is acceptable, or is
   exact llama.cpp-style q8_0 required?
5. Is it acceptable to defer Qwen3.6 CUDA work that would touch qwen35 files
   until the concurrent Medusa owner clears the dirty state?

## Phase 0 Verification

Commands run before the local docs commit:

| Command | Result | Notes |
|---|---:|---|
| `cargo fmt --all --check` | PASS | Formatting is clean in the shared workspace. |
| `git diff --check -- docs/plans/2026-05-10-dsv4-qwen36-substrate-audit.md` | PASS | No whitespace errors in the Phase 0 doc. |
| `cargo check -p infer --no-default-features --features cuda,no-cuda` | PASS | no-cuda build path skips CUDA/TileLang kernel compilation as intended. |
| `CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 NVCC_CCBIN=/usr/bin/g++-14 INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python cargo check --release -p infer --features cuda` | FAIL | Existing CUDA build blocker: `cuda-kernels` build script panics while running nvcc for `csrc/attention/decode_attention_quantized.cu`. This Phase 0 doc does not touch CUDA sources. |
| `CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 NVCC_CCBIN=/usr/bin/g++-14 INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python cargo clippy --release -p infer --features cuda -- -D warnings` | FAIL | Same nvcc blocker as release CUDA check. |
| `cargo test --release -p infer` | FAIL | Unit tests compile and 566 lib tests pass, but `infer/tests/metal_eval_audit.rs::metal_materialize_boundaries_stay_classified` fails because `infer/src/backend/metal/kv_pool.rs` is now an unclassified Metal materialize boundary. This Phase 0 doc does not touch Metal sources. |

## Stop Condition

Phase 0 is complete when this document is committed locally.

Do not start Phase 0.5, Phase 1, or any code implementation until human review
green-lights the next phase.
