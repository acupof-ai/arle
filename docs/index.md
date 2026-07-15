# Maintainer Doc Index

> **Looking for getting-started, install, or HTTP API docs?** Go to
> [README.md](../README.md), [docs/install.md](install.md),
> [docs/onboarding.md](onboarding.md),
> [docs/troubleshooting.md](troubleshooting.md), or
> [docs/http-api.md](http-api.md) instead. This file is for ARLE maintainers
> tracking canonical truth surfaces, active plans, and experience logs.

**This file is a pure index — no narrative state.** Current phase table and
in-flight model items: [`ROADMAP.md`](../ROADMAP.md). Strategic master
(positioning, evolution path, kill registry):
[`projects/2026-06-10-arle-master-strategy-v2.md`](projects/2026-06-10-arle-master-strategy-v2.md).
Chronological progress spine (phase exits, default flips, license-or-kill
verdicts): [`CHANGELOG.md`](../CHANGELOG.md). The status snapshots that used
to live here rotted against ROADMAP.md and were removed 2026-07-02; recover
any of them with `git log -- docs/index.md`.

## Canonical Truth Surfaces

| Concern | Canonical source | Notes |
| --- | --- | --- |
| **New contributor onboarding (30 min)** | [onboarding.md](onboarding.md) | Current truth, paths, feature flags, verify checklist. |
| Strategic master (positioning, evolution path, kill criteria) | [projects/2026-06-10-arle-master-strategy-v2.md](projects/2026-06-10-arle-master-strategy-v2.md) | Cited by [`ROADMAP.md`](../ROADMAP.md) as strategic master. v1 (2026-05-07) is SUPERSEDED, kept as history. |
| Support status of backends / APIs / model families | [support-matrix.md](support-matrix.md) | README and roadmap summarize only. |
| Quantization deep map (KV + weights, kernels, status, tests) | [quantization.md](quantization.md) | Canonical for every quant path; support-matrix §4 mirrors a one-glance view. |
| Stability levels and compatibility posture | [stability-policy.md](stability-policy.md) | Do not redefine tiers elsewhere. |
| Workspace topology and module entry points | [codebase-map.md](codebase-map.md) | Source of truth for "what exists today". |
| Architecture ownership and boundaries | [architecture.md](architecture.md) | The `infer-*` rewrite crates (`infer-core`/`-seam`/`-cuda`/`-metal`/`-server`/`-api`) own runtime truth. |
| DSv4/GLM prefill+decode paths & kernels | [architecture-dsv4.md](architecture-dsv4.md) | Mechanism-level map: FlashMLA prefill/decode, DeepGEMM grouped MoE, DSA indexer, MTP spec-decode + rollback; DSpark survey. |
| Benchmark and trace process | [bench-and-trace-spec.md](bench-and-trace-spec.md) | Native runner is the canonical e2e benchmark path. |
| Canonical e2e bench tool + parameter set | [plans/native-bench.md](plans/native-bench.md) | `scripts/bench_throughput.py` uses this contract. |
| Capability and agent-code evals | [eval.md](eval.md) | MMLU and SWE-bench Pro workflows; ARLE engine owns the candidate answer/patch, deterministic graders own only scoring. |
| OPD/QAT capability curve + serving blocker | [opd-capability-curve.md](opd-capability-curve.md) | Δ-vs-baseline curve driver (`scripts/opd_capability_curve.py`) over the eval lanes; the exact adapter-serving blocker (file:line) + smallest fix for the non-baseline curve points. |
| OPD mainline execution queue | [projects/2026-05-24-opd-mainline-task-backlog.md](projects/2026-05-24-opd-mainline-task-backlog.md) | Historical artifact ledger; superseded by master strategy v2 Phase 3 (OPD GPU work queued behind Phase 1–2). |
| Contributor operating contract | [../AGENTS.md](../AGENTS.md) | Use with the canonical docs above. |

## Current Positioning

`ARLE` is a runtime-first Rust workspace.

- The `infer-*` rewrite crates are the primary serving/runtime surface. The
  monolithic `infer` crate is deleted; the stack is now `infer-plan` (IR) →
  `infer-seam` (host-only traits) → `infer-core` (Engine/scheduler/RadixCache)
  → `infer-cuda`/`infer-metal` (executors) → `infer-server`/`infer-api`, with
  `infer-topo`/`infer-moe`/`infer-util` as shared leaves. `infer-api` is the
  single front door (`LoadedInferenceEngine`, OPD-teacher surface).
- `arle` is the unified local front door binary (`arle serve` for HTTP) for
  agent, train, eval, and data workflows built on that runtime.
- Train/RL work is strategic because it strengthens the runtime loop; it does
  not create a second equal project identity.

If a plan or project note disagrees with that framing and is not explicitly
marked as the current source of truth, treat it as historical context.

## Active Projects

| Path | Status | Use this when |
| --- | --- | --- |
| [projects/2026-06-04-qwen35-dsv4-final-report.md](projects/2026-06-04-qwen35-dsv4-final-report.md) | Active — rewrite verification + perf | The question is the post-rewrite (`infer-*` crates, branch `arch/ideal-inference-engine`) verification & performance verdict across Metal + CUDA, TP/EP, FP8 DeepGEMM MoE, DeepEP — including the honest DSv4 incremental-decode-broken correction and the SGLang-class perf push. Companion detail: [projects/2026-06-04-rewrite-completion-verification-report.md](projects/2026-06-04-rewrite-completion-verification-report.md). |
| [projects/2026-05-24-opd-mainline-task-backlog.md](projects/2026-05-24-opd-mainline-task-backlog.md) | Historical — superseded by strategy v2 | The question is the past OPD mainline task order or session artifact ledger; current OPD sequencing is master strategy v2 Phase 3. |
| [projects/2026-05-18-opd-only-pivot.md](projects/2026-05-18-opd-only-pivot.md) | Active — product boundary | The question is why training scope is OPD-only and why scratch pretrain/SFT/GRPO/multi-turn surfaces stay deleted. |
| [projects/2026-05-01-deepseek-v4-readiness.md](projects/2026-05-01-deepseek-v4-readiness.md) | Active — DSv4 serving; GLM-5.2 verify pending-remote | The question is DeepSeek V4 readiness, the DS0–DS8 gap matrix, and current 8xH20 DeepEP decode hot path. (Qwen3.6 now serves on CUDA + Metal; the in-flight model additions are GLM-5.2 — wired on the DSv4 path, verification pending-remote — and the Metal VLMs Gemma4 / DeepSeek-OCR.) |
| [projects/2026-04-30-longctx-32k-128k-leadership.md](projects/2026-04-30-longctx-32k-128k-leadership.md) | Paused — restarts at strategy v2 Phase 3 | The question is the 32k–128k longctx world-#1 mission (4 phase plan, baseline panel, hardware tiers, Phase 1 SGLang-row close + the Phase 2 spec-decode regression that the frozen-KV redesign now addresses). |
| [projects/2026-05-02-agent-load-mission-expansion.md](projects/2026-05-02-agent-load-mission-expansion.md) | Paused — restarts at strategy v2 Phase 3 | The question is the agent-load world-#1 expansion: W3 short-prompt multi-turn, W4 tool-call resume, session affinity, prefix-cache reuse, four-engine baseline gates (cross-engine baseline still never run). |
| [projects/2026-05-01-multi-gpu-f0-readiness.md](projects/2026-05-01-multi-gpu-f0-readiness.md) | Active | The question is single-node multi-GPU F0 readiness, TP/PP/EP axes, NCCL smoke, the gap matrix to real multi-rank serving. |
| [projects/2026-05-01-spec-decode-integration-design.md](projects/2026-05-01-spec-decode-integration-design.md) | Active | The question is how Phase 2 spec decode plumbing integrates with the CUDA scheduler, verifier, and external draft state. |
| [projects/tiered-kv-cache.md](projects/tiered-kv-cache.md) | Historical — monolith-era design spec | The question is the original tiered-KV design rationale (industry survey, §13 corrections). Current tier status: [support-matrix §4b](support-matrix.md); runtime truth: the `infer-*` code. |
| [projects/mlx-backend-roadmap.md](projects/mlx-backend-roadmap.md) | Active | The question is Metal serving closure, MLX runtime direction, Qwen3.5 GGUF decode hot path. |
| [projects/agent-rl-self-evolving.md](projects/agent-rl-self-evolving.md) | Active | The question is how train/RL/self-evolution work strengthens the runtime spine. |
| [projects/agent-first-architecture.md](projects/agent-first-architecture.md) | Active but secondary | The question is long-horizon agent-serving priorities outside the current KV plan. |

## Active Plans

| Path | Status | Use this when |
| --- | --- | --- |
| [plans/2026-07-10-operator-artifact-dev-release-system.md](plans/2026-07-10-operator-artifact-dev-release-system.md) | Active — scope-reduced operator truth and artifacts | The question is how ARLE proves one operator family end to end, persists canonical performance evidence, exposes engagement stats, removes proven-dead kernels, or publishes exact immutable GitHub Release bundles without overbuilding an artifact platform. |
| [plans/2026-06-15-cuda-quant-subsystem-plan.md](plans/2026-06-15-cuda-quant-subsystem-plan.md) | Active implementation plan — CUDA native resident weight quant | The question is Qwen3.6-35B-A3B FP8/NVFP4 on CUDA without dense BF16 materialization: checkpoint codec, resident `WeightFormat`, loader wiring, Qwen35 GEMV/GEMM/MoE quant kernels, A100 serve/eval gates, and the no-DSv4-ABI boundary. |
| [plans/2026-06-12-architecture-refactor-roadmap.md](plans/2026-06-12-architecture-refactor-roadmap.md) | Active — structural steering brief (tranches R0–R6) | The question is the architecture problem inventory (model×backend matrix, lateral backend deps, truth-surface drift, seam capability governance), the per-tranche agent briefs with scope/non-scope/exit conditions, or the pending xgrammar / AIPC-ratification verdicts. |
| [plans/2026-06-10-hip-backend-mvp.md](plans/2026-06-10-hip-backend-mvp.md) | Active — AIPC HIP lane (#76/#77; phase ratification pending) | The question is the HIP/ROCm DSv4 GGUF 2-bit shim-portable backend: pinned mode flags, residency tiers, on-box bring-up (see also [plans/2026-06-11-hip-onbox-runbook.md](plans/2026-06-11-hip-onbox-runbook.md)). |
| [plans/2026-06-07-unified-batched-kvpool-abstraction.md](plans/2026-06-07-unified-batched-kvpool-abstraction.md) | Active — authoritative (DSv4 campaign forward plan) | The question is the engine-generic batched-decode / paged-KV abstraction: `KvBatchDescriptor` over the `KvPool` seam + per-model `ModelKvAdapter` (DSv4 first, then Qwen/Gemma), the 7-phase refactor-first map, and the Metal convergence end state. The throughput axis of the DSv4 perf campaign. |
| [plans/2026-06-06-dsv4-pd-systematic-analysis.md](plans/2026-06-06-dsv4-pd-systematic-analysis.md) | Active — root-cause anchor | The question is the DSv4 P/D bottleneck at the 4096 SLO shape from a wall-clock end-to-end trace (csa_select #1), the operator-integration audit, and the single-row-executor throughput ceiling. The doc that overturned the smoke-shape lever plans. |
| [plans/2026-06-06-dsv4-h20-reference-baseline.md](plans/2026-06-06-dsv4-h20-reference-baseline.md) | Active — reference baseline | The question is "how fast SHOULD DSv4-Flash be on H20" (base decode ~20-35ms, prefill ~1.5s warm @1024; 6ms requires MTP/EAGLE spec) — the should-be ground truth for any DSv4 perf license. |
| [plans/2026-06-07-dsv4-code-cleanup-audit.md](plans/2026-06-07-dsv4-code-cleanup-audit.md) | Active — cleanup task list | The question is the DSv4 session code-cleanup queue: `ARLE_DSV4_*` env flags → CLI `--flags`, legacy fallbacks now default-off (csa_select, masked/pooled decode), dead hand-rolled paths from the official-kernel swaps, parked-MTP code, with safe-now vs wait-for-batched-decode per item. |
| [plans/2026-06-06-dsv4-handrolled-kernel-audit.md](plans/2026-06-06-dsv4-handrolled-kernel-audit.md) | Active — kernel adoption map | The question is which hand-rolled CUDA kernels duplicate a vendored/official one (DELETE+ADOPT), which wrap already-adopted libs (KEEP), and which are irreducible ARLE glue (KEEP) — the per-operator license-or-kill map. |
| [plans/2026-06-06-dsv4-frozen-kv-mtp-redesign.md](plans/2026-06-06-dsv4-frozen-kv-mtp-redesign.md) | Active design — MTP parked | The question is the SGLang frozen-KV MTP approach for DSv4 (freeze compressor + reuse selection during verify), which un-killed the s_q=K conclusion; MTP remains parked at the draft-quality wall pending a coherent-workload acceptance measurement. |
| [plans/2026-06-04-dsv4-decode-sglang-class-perf.md](plans/2026-06-04-dsv4-decode-sglang-class-perf.md) | Active roadmap — gated on decode fix | The question is the DSv4-Flash TP=8/EP=8 decode path to SGLang-class 5–6 ms/token: on-device MoE routing (kill per-layer H2D/D2H), DeepEP low-latency decode mode, graph-capturable all-reduce, full decode-graph capture. Sequenced behind the incremental-decode correctness fix (#23). |
| [plans/2026-05-27-flashinfer-paged-prefill-migration.md](plans/2026-05-27-flashinfer-paged-prefill-migration.md) | Active design | The question is whether/how to drop TileLang HD128 paged prefill for FlashInfer on sm_80. Driven by two TileLang 0.1.10 regressions in one week. |
| [plans/2026-06-02-metal-mtp-sglang-alignment.md](plans/2026-06-02-metal-mtp-sglang-alignment.md) | Active control plan | The question is Metal Qwen3.6 MTP after the SGLang survey: frozen-KV invariants, parity-first gates, packed verify order, why MTP remains opt-in, or the bottom-level acceptance compute note at [research/2026-06-02-metal-mtp-acceptance-compute.md](research/2026-06-02-metal-mtp-acceptance-compute.md). |
| [plans/2026-05-24-sglang-pipeline-cuda-mlx-gap-analysis.md](plans/2026-05-24-sglang-pipeline-cuda-mlx-gap-analysis.md) | Active — OPD/runtime gap queue | The question is G1–G7 license-or-kill work, SGLang parity gaps, or GPU/Metal-deferred runtime experiments. |
| [plans/2026-05-25-kv-storage-transport-library-design.md](plans/2026-05-25-kv-storage-transport-library-design.md) | Active design | The question is storage/transport substrate direction for SSD↔HBM, DRAM↔HBM, T2/T3 KV movement, or the proposed transport crate boundary. |
| [plans/2026-04-28-single-node-multi-gpu.md](plans/2026-04-28-single-node-multi-gpu.md) | Active | The question is the single-node multi-GPU plan (F0–F8 phases) for TP/PP/EP scaffolding and forward collectives. |
| [plans/2026-04-28-multi-gpu-f0-verification.md](plans/2026-04-28-multi-gpu-f0-verification.md) | Active | The question is the F0 verification protocol (NCCL link, rendezvous, all-reduce smoke, single-rank no-regression gate). |
| [plans/2026-05-01-longctx-spec-decode-phase2.md](plans/2026-05-01-longctx-spec-decode-phase2.md) | Active | The question is Phase 2 long-context speculative decode integration on top of the closed Phase 1 W1 c=4 SGLang row. |
| [plans/M_medusa-phase1b-qwen35-v2-snapshot-ring-redesign.md](plans/M_medusa-phase1b-qwen35-v2-snapshot-ring-redesign.md) | Active gate | The question is how to make Qwen3.5 safe for Medusa/spec verification. Start here for Qwen3.5 Medusa work. |
| [plans/2026-05-01-mla-kernel-design.md](plans/2026-05-01-mla-kernel-design.md) | Design only | The question is the DeepSeek-family MLA CUDA kernel design (DS3) — formula, cache layout, prefill/decode dispatch. |
| [plans/2026-05-02-agent-load-bench-spec.md](plans/2026-05-02-agent-load-bench-spec.md) | Active | The question is the W3/W4 agent-load benchmark contract: short-prompt multi-turn, tool-call resume, session affinity, cache metrics, four-engine baseline evidence. |
| [plans/2026-05-03-a8-gpu-sm-kv-io-kernel.md](plans/2026-05-03-a8-gpu-sm-kv-io-kernel.md) | Pending — gated on W4 close | The question is whether to swap `cudaMemcpyAsync` for an SM-driven kernel on T0↔T1 paged-block transfers (LMSYS 3× claim). Read before touching `crates/infer-cuda/src/kv_tier.rs`. |
| [plans/cpu-gpu-pipeline-sync-stream.md](plans/cpu-gpu-pipeline-sync-stream.md) | Design plan | The question is how to make CPU/GPU serving pipeline stages explicit, with CUDA stream/event fences and Metal async-eval or command-buffer completion semantics. |
| [plans/infer-observability-v1.md](plans/infer-observability-v1.md) | Active | The question is operator-facing observability, traces, or profiling flow. |
| [plans/tiered-kv-hicache-readmission.md](plans/tiered-kv-hicache-readmission.md) | Active | The question is staged KV readmission or remote/shared backend follow-up. |
| [plans/rust-agent-rl-single-node.md](plans/rust-agent-rl-single-node.md) | Active | The question is the Phase 6 execution path under the runtime-first rule. |
| [plans/train-runtime-architecture-v1.md](plans/train-runtime-architecture-v1.md) | Active | The question is today's train-side runtime / control-plane factoring. |
| [plans/train-observability-v1.md](plans/train-observability-v1.md) | Active | The question is train-side events, MLflow, OTLP, or W&B export flow. |
| [plans/train-eval-infer-dx-v1.md](plans/train-eval-infer-dx-v1.md) | Active | The question is unified operator DX across train, eval, and infer. |

## Reference Plans

| Path | Role |
| --- | --- |
| [plans/2026-04-20-project-constitution-and-refactor-plan.md](plans/2026-04-20-project-constitution-and-refactor-plan.md) | SSOT identity, project boundaries, doc/release governance (Tranches T0/T3 completed 2026-04-25). |
| [plans/cuda-kernel-crate-extraction.md](plans/cuda-kernel-crate-extraction.md) | Reference (extraction landed; trip wires govern future splits). |
| [plans/native-bench.md](plans/native-bench.md) | Canonical native benchmark contract. |

## Multi-SM / Hardware Coverage

| Path | Role |
| --- | --- |
| [plans/sm-coverage.md](plans/sm-coverage.md) | SM tier policy (T1/T2), per-SM cubin contract; referenced from CLAUDE.md build section. |
| [plans/sm-coverage-verification.md](plans/sm-coverage-verification.md) | Runbook for retiring `pending-remote` bench stubs across A100/A10/L4/H100. |

## Operator And Policy References

| Path | Role |
| --- | --- |
| [http-api.md](http-api.md) | HTTP contract and streaming behavior |
| [environment.md](environment.md) | Environment variables and runtime knobs |
| [release-checklist.md](release-checklist.md) | Release prep and artifact verification |
| [perf-and-correctness-gates.md](perf-and-correctness-gates.md) | Lightweight validation expectations by change type |
| [resources/profiling-guide.md](resources/profiling-guide.md) | GPU profiling playbook |
| [resources/kv-cache-quantization.md](resources/kv-cache-quantization.md) | KV-cache quantization formats and operator-side guidance |
| [resources/infer-cuda-profiling-wrappers.md](resources/infer-cuda-profiling-wrappers.md) | `nsys` / `ncu` wrapper scripts |

## Archived / Historical (kept for evidence + cross-refs)

These plans and project notes are not active source of truth, but stay
in tree because active docs link to them or they capture audit history
worth preserving. Treat them as historical context unless a current plan
brings them back.

### Plans (archived)

| Path | Why kept |
| --- | --- |
| [plans/2026-05-05-multi-backend-tilelang-rocm-vulkan.md](plans/2026-05-05-multi-backend-tilelang-rocm-vulkan.md) | Strix Halo / ROCm / Vulkan exploration; referenced from `backend-unification.md` and `cuda-kernel-tilelang-unification.md`. |
| [plans/M3.5-collapse-scheduler-loops.md](plans/M3.5-collapse-scheduler-loops.md) | Structural follow-up to M3; cited by `m6-cuda-vllm-gap-followups.md`. |
| [plans/M5-P0-modelforward-survey.md](plans/M5-P0-modelforward-survey.md) | Pre-plan survey behind the landed `m5-modelarch-trait.md`. |
| [plans/M_medusa-phase1b-substrate-brief.md](plans/M_medusa-phase1b-substrate-brief.md) | PAUSED Qwen3/Qwen3.6 Medusa brief; superseded by `M_medusa-phase1b-qwen35-v2-snapshot-ring-redesign.md` which links back to it. |
| [plans/2026-05-10-dsv4-qwen36-substrate-audit.md](plans/2026-05-10-dsv4-qwen36-substrate-audit.md) | Phase 0 audit for DSv4 1B + Qwen3.6 CUDA substrate; predates current DSv4 readiness project. |
| [plans/2026-06-06-dsv4-decode-6ms-remaining-levers.md](plans/2026-06-06-dsv4-decode-6ms-remaining-levers.md) | **Superseded** — smoke-shape decode lever ranking (comm 32.4% / GEMV / mhc); overturned by the wall-clock @4096 trace (csa_select #1) + official-kernel adoption. Marked at top; kept for the profiling-method record. |
| [plans/2026-06-06-dsv4-decode-6ms-dag.md](plans/2026-06-06-dsv4-decode-6ms-dag.md) | **Superseded** — 6ms-via-EAGLE-now DAG on the smoke-shape ranking; overturned (csa_select root cause + H20 baseline re-anchor + MTP parked). Marked at top; kept for the §0.1 decomposition. |
| [plans/2026-06-06-dsv4-decode-residual-gemv-fusion.md](plans/2026-06-06-dsv4-decode-residual-gemv-fusion.md) | **Superseded** — "GEMV is the #2 decode lever (14.4%)" is a smoke-shape artifact; the real fix was the official DSA indexer. Marked at top; the fused-wo design kept as a candidate. |
| [plans/2026-06-06-dsv4-prefill-profile-levers.md](plans/2026-06-06-dsv4-prefill-profile-levers.md) | Partially superseded — profile valid (compute-bound), but lever #3 "FlashMLA-prefill killed" was wrong (official kernel default-on, faster) and #2 csa-reuse killed. Marked at top. |
| [plans/2026-06-06-dsv4-prefill-fused-wqkv-extend.md](plans/2026-06-06-dsv4-prefill-fused-wqkv-extend.md) | Shipped — fused `wq_a\|wkv` → FP8 DeepGEMM, now default-on (`FP8_LINEAR_DEEPGEMM`); realized −5% (fused slice only). Status pointer at top. |
| [plans/2026-06-06-dsv4-eagle-mtp-phase2-verify-loop.md](plans/2026-06-06-dsv4-eagle-mtp-phase2-verify-loop.md) | **Superseded** — verify loop landed correct (default-off) but the "1.9× lever" goal didn't hold; MTP parked at the draft-quality wall. Marked at top. |
| [plans/2026-06-06-dsv4-a2-sqk-verify-detail.md](plans/2026-06-06-dsv4-a2-sqk-verify-detail.md) | **Superseded** — s_q=K verify killed then un-killed by frozen-KV; MTP parked regardless. Marked at top. |
| [plans/2026-06-06-dsv4-wholesale-kernel-adopt.md](plans/2026-06-06-dsv4-wholesale-kernel-adopt.md) | Mostly shipped — the adopt-official sequencing arc (correct); the EAGLE step is overturned (MTP parked). Status pointer at top. |

### Projects (archived)

| Path | Why kept |
| --- | --- |
| [projects/2026-05-07-metal-world-first-strategy.md](projects/2026-05-07-metal-world-first-strategy.md) | Consolidated 2026-05-07 Metal strategy synthesis (SOTA gap audit + unification recalibration + sequencing). Folds in three earlier same-day notes; current state pointer is ROADMAP P3 and `mlx-backend-roadmap.md`. |
| [projects/2026-04-29-scheduler-pipeline-map.md](projects/2026-04-29-scheduler-pipeline-map.md) | End-to-end CUDA scheduler walk-through with file:line cites; referenced from `mla-kernel-design.md` and the longctx project. |
| [projects/2026-04-29-perf-bug-roundup.md](projects/2026-04-29-perf-bug-roundup.md) | SGLang-parity perf bug ledger; cited by `bench-and-trace-spec.md` and the throughput-gap analysis. |
| [projects/2026-04-29-throughput-gap-analysis.md](projects/2026-04-29-throughput-gap-analysis.md) | "Why we're 28% behind SGLang at c=16" snapshot; cited by the longctx project. |
| [projects/2026-04-30-arle-vs-sglang-admission.md](projects/2026-04-30-arle-vs-sglang-admission.md) | Admission policy gap matrix; sibling to active SGLang admission research note. |
| [projects/2026-05-02-tilekernels-integration-decision.md](projects/2026-05-02-tilekernels-integration-decision.md) | Decision record (don't-submodule, port-selectively) for `cklxx/TileKernels`; referenced from the multi-backend plan. |
| [projects/2026-05-07-eli-arle-native-provider-design.md](projects/2026-05-07-eli-arle-native-provider-design.md) | Layer-2 nexil ↔ arle native-provider design; shipped 2026-05-07 (`session_id` forwarding, now in `infer-api` + `infer-server`). Kept as post-implementation reference. |

### Resources (archived)

| Path | Why kept |
| --- | --- |
| [resources/metal-dflash.md](resources/metal-dflash.md) | **Historical** — written against deleted `metal_request`/`metal_bench`/`metal_serve` binaries. DFlash survives only as the `mlx-sys` draft-model FFI substrate; rewrite Metal serve uses MTP. |
| [resources/metal-dflash-params.md](resources/metal-dflash-params.md) | **Historical** — DFlash CLI param reference for the deleted binaries; pairs with `metal-dflash.md`. |
| [resources/eli-integration.md](resources/eli-integration.md) | **Historical** — eli drove the deleted `metal_serve`; rewrite entry point is `arle serve --backend metal`. |

## Historical Material

- `docs/experience/wins/` and `docs/experience/errors/` are the curated
  evidence log. The latest three of each are always-loaded per `AGENTS.md`;
  earlier entries are kept only when they are referenced from a KEEP file or
  document a milestone (M0–M5 tiered-kv, hybrid Qwen3.5 acceptance, c-sweep
  SGLang closure, RoPE YARN scaling landing, train-side milestone snapshots).
- `docs/experience/reviews/` is one Codex code-review snapshot retained as
  reference for the cuda-link audit.
- `docs/trace-artifacts/` holds dated nsys / GPU trace artifacts (DSv4 decode
  + DeepEP, 2026-05-14 onwards).
- Plans / projects / research / reviews not listed above (active or archived)
  are historical session notes. Anything not on this index is not a source
  of truth.

## Truth-surface invariant

Per [`plans/2026-04-20-project-constitution-and-refactor-plan.md`](plans/2026-04-20-project-constitution-and-refactor-plan.md)
§2: every concern in the canonical-truth-surfaces table above has exactly
one definition. Adding a second one (a new index, a parallel `*/docs/`
tree, a sibling status matrix) is a regression and must be rejected at PR
time.
