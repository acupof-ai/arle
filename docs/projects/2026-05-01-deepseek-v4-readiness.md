# DeepSeek V4 readiness assessment

Date: 2026-05-01 (assessment); 2026-05-06 priority refresh.
Status: **#1 next-model priority** (per
[`ROADMAP.md` §Next-Model Priority Order](../../ROADMAP.md#next-model-priority-order)).
Substrate landing: DS0 spec crate, runtime model skeleton, and nano autograd
training shipped 2026-05-05; MLA forward kernels + DS4 CUDA MoE forward + DS5
NCCL collectives in forward are the active runtime blockers. Scope:
CUDA/runtime readiness for the DeepSeek V4-family serving path.

This document treats the public V4 information as a readiness input, not a
frozen implementation contract. The stable architectural baseline remains the
DeepSeek-V3/R1 line: MLA, sparse MoE, MTP, and FP8 training/inference
techniques. The V4-specific references below are current public integration
signals and must be re-checked against the actual checkpoint/config before
implementation.

External references used:

- vLLM model executor surface for DeepSeek V4 MTP:
  `https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/deepseek_v4_mtp/`
- LMSYS Day-0 notes for DeepSeek-V3.2-exp / V4-adjacent serving:
  `https://lmsys.org/blog/2025-09-29-deepseek-v3-2-exp/`
- Hugging Face engineering notes on DeepSeek V3.2-exp and DeepSeek-V4 kernels:
  `https://huggingface.co/blog/vllm-ds-v3-2`
- DeepSeek-V3 technical report baseline:
  `https://arxiv.org/abs/2412.19437`

## Required Capabilities

| Capability | Why V4 needs it | ARLE dependency |
|---|---|---|
| MoE architecture | DeepSeek V3-family models are sparse MoE; V4 readiness must support expert routing, shared/dense experts where present, and distributed expert placement. | F2-F8 multi-GPU axes: EP, MoE-TP, MoE-DP, router metadata, expert dispatch. |
| Multi-Token Prediction (MTP) | V3 introduced MTP modules; V4 public surfaces expose MTP-specific model executors. MTP can become either a training-head compatibility path or a serving acceleration path. | Phase 2 speculative decode framework, `DraftMode`, persistent draft state, verifier accounting. |
| Block-FP8 quantization | V3 reports FP8 training; V4 serving stacks call out block-FP8 kernels. This differs from ARLE's current KV-only FP8 scale model. | New block-wise FP8 weight/KV/activation loaders and kernels, separate from `KVFormat::FP8E4M3`. |
| MLA attention | DeepSeek uses Multi-head Latent Attention, not Qwen-style GQA. It changes cache layout, projection names, attention math, and decode kernels. | New model family plus MLA prefill/decode kernels; cannot reuse Qwen3 GQA kernels directly. |
| Fine-grained expert routing | Efficient MoE needs top-k routing, grouped expert batching, per-rank token dispatch, and combine/scatter semantics. | CUDA MoE forward, `ParallelState` MoE groups, `GroupCoordinator`, scheduler batch metadata. |

## Current ARLE State

Ready or partially ready:

- Distributed group metadata exists in `infer/src/distributed/parallel_state.rs`
  with world, TP, PP, EP, attention-TP/DP/CP, and MoE-TP/EP/DP groups.
- Group collective metadata exists in
  `infer/src/distributed/group_coordinator.rs`; single-rank paths are no-op
  and NCCL wrapping is feature-gated.
- NCCL smoke exists behind the `nccl` feature in `infer/src/distributed/nccl.rs`
  and `infer/src/distributed.rs`.
- `LayerCommunicator` exists in `infer/src/model/layer_communicator.rs`; it is
  wired into Qwen3/Qwen3.5 call sites, but production multi-rank is still
  guarded until real collectives land.
- Qwen3/Qwen3.5 sharded-load math and loader helpers are staged in
  `infer/src/model/qwen3/weights.rs`, `infer/src/model/qwen35/weights.rs`, and
  `infer/src/weight_loader.rs`; TP>1 load currently fails fast because
  collectives are not complete.
- The speculative decode framework exists in `infer/src/speculative.rs`,
  `infer/src/speculative/cuda.rs`, and `infer/src/scheduler/cuda/spec_path.rs`.
  It includes `DraftMode`, external draft state, acceptance tracking, and
  verifier plumbing, but its current external-draft data showed throughput
  regression and is not yet an acceleration win.
- KV FP8 exists through `KVFormat::FP8E4M3` and kernels under
  `crates/cuda-kernels/csrc/kv/kv_quant.cu`; this is useful prior art for scale
  handling and dispatch, not block-FP8 coverage.
- Architecture detection now names DeepSeek V2/V3, DeepSeek V4, and the V4 MTP
  draft surface in `infer/src/model_registry.rs`; all DeepSeek variants remain
  marked unimplemented until model/kernel work lands.
- Metal has Qwen3.5/Qwen3.6 MoE prior art under `infer/src/backend/metal/` and
  `crates/mlx-sys/src/mlx_qwen35_moe_block.cpp`; this is not a CUDA DeepSeek
  implementation but helps reason about router/expert tensor shapes.

Missing:

- No CUDA DeepSeek safetensors loader, scheduler registration, or HTTP/runtime
  dispatch arm.
- No MLA KV cache layout, prefill kernel, decode kernel, or paged-cache planner.
- No CUDA sparse MoE expert forward, router top-k, expert batching, all-to-all
  dispatch, or EP runtime.
- No block-FP8 tensor format abstraction, safetensors loader, dequant GEMM, or
  block-wise scale storage.
- No MTP-head model support or MTP-aware verifier path; current spec decode is
  draft-model based rather than checkpoint-native MTP.

## Proposed Spec Crate

Add `crates/deepseek-spec/`, modeled after `crates/qwen3-spec/` and
`crates/qwen35-spec/`.

Responsibilities:

- Parse DeepSeek V3/V4 `config.json` variants and normalize architecture names
  such as `DeepseekV3ForCausalLM` and any V4-specific successor.
- Own tensor-name contracts for embeddings, MLA projections, router, shared
  experts, routed experts, MTP modules, final norm, and LM head.
- Provide `Shard` annotations for dense projections, MLA projections, router
  weights, expert stacks, shared experts, and MTP heads.
- Expose feature flags/metadata for `has_mtp`, `num_experts`,
  `num_experts_per_tok`, expert group size, MLA latent dimensions, rope/nope
  head dimensions, and quantization block shape.
- Keep this crate CPU-only and shared between `infer` and future train-side
  support, following the existing qwen spec-crate pattern.

## Gap Matrix

| Gap | Description | Current state | Required commit sequence | Est. size/time |
|---|---|---|---|---:|
| DS0 spec crate | DeepSeek V4 needs a canonical config and tensor-name contract before any loader or kernel work. | Landed: `crates/deepseek-spec` owns config parsing, tensor names, shard annotations, MTP names, and MoE forward planning. | Keep extending it as V4 checkpoint metadata stabilizes; do not fork tensor-name truth into the runtime. | landed |
| DS1 model registry | Runtime must detect DeepSeek V4/MTP configs and route them to an explicit unsupported or experimental path. | Landed: `model_registry.rs` has `DeepSeekV4` and `DeepSeekV4Mtp`, maps verified architecture strings, exposes a V4-specific attention variant, and keeps `is_implemented=false`. | Wire runtime loading only after DS3/DS4/DS5 produce a real serving path. | landed |
| DS2 Block-FP8 format | V4 serving requires block-wise FP8 weights/activations, not only per-token FP8 KV. | ARLE has `KVFormat::FP8E4M3` and KV quant kernels; weight loader has non-block quant paths. | 1. Add block-FP8 metadata structs in loader/spec. 2. Parse safetensors quant metadata. 3. Add CPU decode/dequant tests. 4. Add CUDA dequant-GEMM kernel entry points. | 800-1600 LoC, 3-5 days |
| DS3 MLA cache and kernels | MLA changes K/V cache contents and attention math. GQA kernels cannot be reused as-is. | `AttentionVariant::Mla` exists only as registry metadata. Qwen kernels cover GQA/hybrid GQA. | 1. Define MLA state/cache layout. 2. Add BF16 MLA prefill reference path. 3. Add paged MLA decode kernel. 4. Add FP8/block-FP8 variants. 5. Add numerical baselines. | 1800-3500 LoC, 1-2 weeks |
| DS4 CUDA MoE forward | Sparse routed experts are the core DeepSeek throughput path. | CUDA Qwen3.5 MoE is a stub; Metal has separate Qwen MoE prior art. Distributed MoE groups exist as metadata. | 1. Add router top-k CPU/GPU tests. 2. Add single-GPU expert batching and combine. 3. Add shared expert support. 4. Add EP dispatch using group metadata. 5. Add MoE-TP/MoE-DP collectives. | 2000-4500 LoC, 1-2 weeks |
| DS5 NCCL collectives in forward | DeepSeek V4 likely cannot fit useful deployments without TP/EP. | `LayerCommunicator` and `GroupCoordinator` exist; F2 currently guards TP>1 production load. | 1. Wire BF16/f16 all-reduce/all-gather/broadcast into `GroupCoordinator`. 2. Feed communicators into model instances. 3. Remove F2 fail-fast only after TP=2 smoke passes. | 700-1400 LoC, 3-5 days |
| DS6 MTP integration | V4 MTP modules need a native serving story, not only external draft. | Spec decode framework exists, but current real external-draft path regressed; no checkpoint-native MTP head exists. | 1. Extend `DraftMode` with `MtpHead`. 2. Add MTP tensor names to spec crate. 3. Add MTP head loader/state. 4. Reuse verifier/acceptance metrics. 5. Bench MTP vs no-MTP. | 900-1800 LoC, 4-7 days |
| DS7 scheduler routing metadata | Fine-grained expert routing needs per-step token/expert placement and possibly all-to-all traffic accounting. | Scheduler has continuous batching and spec path; no expert token routing metadata. | 1. Add per-batch expert routing buffers. 2. Account expert dispatch workspace. 3. Integrate EP admission limits. 4. Add stats for expert imbalance. | 700-1300 LoC, 3-5 days |
| DS8 validation and baselines | DeepSeek V4 needs reproducible correctness before performance tuning. | Qwen tests compare against JSON baselines; no DeepSeek fixtures exist. | 1. Add tiny config fixtures in spec crate. 2. Add safetensors shape-only loader test. 3. Add CPU/reference logits fixture if license permits. 4. Add CUDA smoke once kernels land. | 400-900 LoC, 2-4 days |

## Recommended Priority

1. **DS2 block-FP8 metadata next.** DS0/DS1 now give shape/routing truth; the
   next low-risk code tranche is CPU-side quant metadata parsing before any
   CUDA kernel ABI is committed.
2. **DS5 collectives before claiming DeepSeek multi-GPU serving.** F2 now
   correctly refuses TP>1 production load until collectives are real. DeepSeek
   MoE/MLA work should not bypass that guard.
3. **DS3 MLA kernel and DS4 MoE forward are the largest new work.** These are
   the true model-family deltas. Borrow design from SGLang, vLLM, and
   TensorRT-LLM DeepSeek V3/V4 paths, but implement ARLE-native Rust/CUDA
   contracts.
4. **DS2 block-FP8 should run in parallel with DS3/DS4 design, not after.**
   MLA/MoE kernels need to know scale layout early; retrofitting block-FP8 late
   will churn loader and kernel ABIs.
5. **DS6 MTP reuses Phase 2 surfaces but should not reuse the external-draft
   performance assumption.** Treat MTP as checkpoint-native draft heads wired
   through existing verifier metrics, then bench honestly.

## Readiness Verdict

DS4 is now the #1 next-model priority (canonical in
[`ROADMAP.md` §Next-Model Priority Order](../../ROADMAP.md#next-model-priority-order)).
Substrate landed 2026-05-05; full serving is not ready immediately, but the
staged prep sequence below is the active execution path, not a hypothetical
plan:

1. Keep `crates/deepseek-spec/` and registry detection as the single source of
   truth for checkpoint family routing.
2. Finish F2 collectives so TP/EP paths can run instead of fail-fast.
3. Prototype MLA BF16 prefill/decode correctness.
4. Prototype single-GPU MoE forward correctness.
5. Add block-FP8 only once tensor scale layout is pinned by the spec crate.
6. Add MTP head support through the existing spec-decode/acceptance metrics
   surface.

The highest-risk items are MLA kernels and CUDA MoE forward/EP runtime. The
most reusable existing ARLE pieces are distributed group metadata, NCCL smoke
foundation, `LayerCommunicator`, speculative decode tracking, and FP8 KV prior
art.
