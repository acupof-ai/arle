# Maintainer Doc Index

> **Looking for getting-started, install, or HTTP API docs?** Go to
> [README.md](../README.md), [docs/install.md](install.md),
> [docs/troubleshooting.md](troubleshooting.md), or
> [docs/http-api.md](http-api.md) instead. This file is for ARLE maintainers
> tracking canonical truth surfaces, active plans, and experience logs.

**Current status (2026-05-25):** OPD mainline execution is active while the
long-running P5 pure-OPD 5k CUDA run owns the local GPU. The live queue,
CPU-only shipped tasks, GPU-deferred gates, and session artifact ledger are in
[`projects/2026-05-24-opd-mainline-task-backlog.md`](projects/2026-05-24-opd-mainline-task-backlog.md).
The OPD-only product boundary remains
[`projects/2026-05-18-opd-only-pivot.md`](projects/2026-05-18-opd-only-pivot.md).

**DSv4 status snapshot (2026-05-26):** DSv4 DeepEP decode remains the active
next-model hot path. User-facing target framing is input 32K / output 1.5K,
H20 qps 8 at concurrency 8, SLO TTFT <= 5000 ms and TPOT <= 30 ms; current
target baseline is TTFT 4800 ms, TPOT 18 ms, total throughput 8402. Default
B=1 padded BF16 reduce-scatter combine, fused local-expert prepare, broad
scratch reuse, DeepEP-style dispatch/combine, and the DeepGEMM auto local
expert backend have landed. Required DeepGEMM remains a fail-fast validation
toolchain gate after the 2026-05-26 wall-clock and correctness KILL;
route-grouped experts remain diagnostic-only. Remaining blockers: replacing the
DeepEP-style NCCL fallback with native DeepEP low-latency kernels,
byte-identical grouped expert GEMM, launch churn. `deepep` is the default MoE
backend today; native DeepEP is now the top DSv4 communication axis, but it is
not yet the default because the official multi-process DeepEP LL/intranode
gates pass while ARLE's same-process 8-thread LL/intranode gates fail. The next
DeepEP step is a process-per-rank transport design, not another same-process
drop-in attempt. A2.0 fused the B=1 decode attention
window-cache update into the attention kernels, removing 9504 standalone update
kernel launches from the measured H20 `max_tokens=32` nsys request. A2.1 fused
Q/K prepare into one launch, cutting `cudaLaunchKernel` runtime calls 490244 →
479574 in the same filtered decode framing. Evidence:
[`experience/errors/2026-05-14-dsv4-decode-nccl-bottleneck.md`](experience/errors/2026-05-14-dsv4-decode-nccl-bottleneck.md),
[`experience/errors/2026-05-26-dsv4-a3-phase2-route-grouped-kill.md`](experience/errors/2026-05-26-dsv4-a3-phase2-route-grouped-kill.md),
[`experience/errors/2026-05-26-dsv4-a3-phase2-deepgemm-kill.md`](experience/errors/2026-05-26-dsv4-a3-phase2-deepgemm-kill.md),
[`experience/errors/2026-05-26-dsv4-native-deepep-ll-sameprocess-timeout.md`](experience/errors/2026-05-26-dsv4-native-deepep-ll-sameprocess-timeout.md),
[`experience/errors/2026-05-26-dsv4-native-deepep-process-model-gate.md`](experience/errors/2026-05-26-dsv4-native-deepep-process-model-gate.md),
[`plans/2026-05-26-dsv4-deepep-process-per-rank.md`](plans/2026-05-26-dsv4-deepep-process-per-rank.md) (next step — sidecar transport),
[`experience/wins/2026-05-26-dsv4-deepep-child-process-spike.md`](experience/wins/2026-05-26-dsv4-deepep-child-process-spike.md) (phase 0 PASS — child-process buffer reuse),
[`experience/wins/2026-05-26-dsv4-deepep-ipc-roundtrip-measurement.md`](experience/wins/2026-05-26-dsv4-deepep-ipc-roundtrip-measurement.md) (phase 1 budget reframed to 250 us / layer; C++ sidecar locked),
[`experience/wins/2026-05-26-dsv4-default-deepep-deepgemm.md`](experience/wins/2026-05-26-dsv4-default-deepep-deepgemm.md),
[`experience/wins/2026-05-26-dsv4-a20-fused-attn-window-update.md`](experience/wins/2026-05-26-dsv4-a20-fused-attn-window-update.md),
[`experience/wins/2026-05-26-dsv4-a21-fused-qk-prep.md`](experience/wins/2026-05-26-dsv4-a21-fused-qk-prep.md),
[`experience/wins/2026-05-26-dsv4-deepgemm-device-prop-cache.md`](experience/wins/2026-05-26-dsv4-deepgemm-device-prop-cache.md),
[`experience/wins/2026-05-26-dsv4-deepgemm-device-counts.md`](experience/wins/2026-05-26-dsv4-deepgemm-device-counts.md),
`trace-artifacts/2026-05-15-dsv4-deepep/`.

**Qwen3.5 Medusa is not pickup-ready** — recurrent-state accepted-length
commit/rollback contract is the gate. Active plan:
[`plans/M_medusa-phase1b-qwen35-v2-snapshot-ring-redesign.md`](plans/M_medusa-phase1b-qwen35-v2-snapshot-ring-redesign.md);
Step 0 audit:
[`research/2026-05-10-medusa-phase1b-qwen35-step0-audit.md`](research/2026-05-10-medusa-phase1b-qwen35-step0-audit.md).

For older session retros, run `git log -- docs/index.md` — they no longer
live in this file.

## Canonical Truth Surfaces

| Concern | Canonical source | Notes |
| --- | --- | --- |
| Strategic master (positioning, axes, kill criteria) | [projects/2026-05-07-arle-master-strategy.md](projects/2026-05-07-arle-master-strategy.md) | Cited by [`ROADMAP.md`](../ROADMAP.md) as strategic master. |
| Support status of backends / APIs / model families | [support-matrix.md](support-matrix.md) | README and roadmap summarize only. |
| Quantization deep map (KV + weights, kernels, status, tests) | [quantization.md](quantization.md) | Canonical for every quant path; support-matrix §4 mirrors a one-glance view. |
| Stability levels and compatibility posture | [stability-policy.md](stability-policy.md) | Do not redefine tiers elsewhere. |
| Workspace topology and module entry points | [codebase-map.md](codebase-map.md) | Source of truth for "what exists today". |
| Architecture ownership and boundaries | [architecture.md](architecture.md) | `infer` owns runtime truth. |
| Benchmark and trace process | [bench-and-trace-spec.md](bench-and-trace-spec.md) | `guidellm` is the canonical e2e benchmark path. |
| Canonical e2e bench tool + parameter set | [plans/guidellm-integration.md](plans/guidellm-integration.md) | Wrapper script `scripts/bench_guidellm.sh` uses these params verbatim. |
| OPD mainline execution queue | [projects/2026-05-24-opd-mainline-task-backlog.md](projects/2026-05-24-opd-mainline-task-backlog.md) | Live CPU/GPU task order, deferred gates, and fdb021c→HEAD artifact ledger. |
| Contributor operating contract | [../AGENTS.md](../AGENTS.md) | Use with the canonical docs above. |

## Current Positioning

`ARLE` is a runtime-first Rust workspace.

- `infer` is the primary serving/runtime surface.
- `arle` is the unified local front door for agent, train, eval, and data
  workflows built on that runtime.
- Train/RL work is strategic because it strengthens the runtime loop; it does
  not create a second equal project identity.

If a plan or project note disagrees with that framing and is not explicitly
marked as the current source of truth, treat it as historical context.

## Active Projects

| Path | Status | Use this when |
| --- | --- | --- |
| [projects/2026-05-24-opd-mainline-task-backlog.md](projects/2026-05-24-opd-mainline-task-backlog.md) | Active — live queue | The question is the current OPD mainline task order, CPU-only shipped work, GPU-deferred gates, or session artifact ledger. |
| [projects/2026-05-18-opd-only-pivot.md](projects/2026-05-18-opd-only-pivot.md) | Active — product boundary | The question is why training scope is OPD-only and why scratch pretrain/SFT/GRPO/multi-turn surfaces stay deleted. |
| [projects/2026-05-01-deepseek-v4-readiness.md](projects/2026-05-01-deepseek-v4-readiness.md) | Active — #1 next-model | The question is DeepSeek V4 readiness, the DS0–DS8 gap matrix, and current 8xH20 DeepEP decode hot path. |
| [projects/2026-04-30-longctx-32k-128k-leadership.md](projects/2026-04-30-longctx-32k-128k-leadership.md) | Active — P0 mission | The question is the 32k–128k longctx world-#1 mission (4 phase plan, baseline panel, hardware tiers, current Phase 1 SGLang-row close + Phase 2 plumbing/regression status). |
| [projects/2026-05-02-agent-load-mission-expansion.md](projects/2026-05-02-agent-load-mission-expansion.md) | Active — mission expansion | The question is the agent-load world-#1 expansion: W3 short-prompt multi-turn, W4 tool-call resume, session affinity, prefix-cache reuse, four-engine baseline gates. |
| [projects/2026-05-01-multi-gpu-f0-readiness.md](projects/2026-05-01-multi-gpu-f0-readiness.md) | Active | The question is single-node multi-GPU F0 readiness, TP/PP/EP axes, NCCL smoke, the gap matrix to real multi-rank serving. |
| [projects/2026-05-01-spec-decode-integration-design.md](projects/2026-05-01-spec-decode-integration-design.md) | Active | The question is how Phase 2 spec decode plumbing integrates with the CUDA scheduler, verifier, and external draft state. |
| [projects/tiered-kv-cache.md](projects/tiered-kv-cache.md) | Active | The question is current KV-tier scope, milestones, or operator-facing status. |
| [projects/tiered-kv-runtime-flow.md](projects/tiered-kv-runtime-flow.md) | Active | The question is how scheduler, RadixCache, and tier coordinator interact at runtime. |
| [projects/mlx-backend-roadmap.md](projects/mlx-backend-roadmap.md) | Active | The question is Metal serving closure, MLX runtime direction, Qwen3.5 GGUF decode hot path. |
| [projects/agent-rl-self-evolving.md](projects/agent-rl-self-evolving.md) | Active | The question is how train/RL/self-evolution work strengthens the runtime spine. |
| [projects/agent-first-architecture.md](projects/agent-first-architecture.md) | Active but secondary | The question is long-horizon agent-serving priorities outside the current KV plan. |

## Active Plans

| Path | Status | Use this when |
| --- | --- | --- |
| [plans/2026-05-27-flashinfer-paged-prefill-migration.md](plans/2026-05-27-flashinfer-paged-prefill-migration.md) | Active design | The question is whether/how to drop TileLang HD128 paged prefill for FlashInfer on sm_80. Driven by two TileLang 0.1.10 regressions in one week. |
| [plans/2026-05-24-sglang-pipeline-cuda-mlx-gap-analysis.md](plans/2026-05-24-sglang-pipeline-cuda-mlx-gap-analysis.md) | Active — OPD/runtime gap queue | The question is G1–G7 license-or-kill work, SGLang parity gaps, or GPU/Metal-deferred runtime experiments. |
| [plans/2026-05-25-kv-storage-transport-library-design.md](plans/2026-05-25-kv-storage-transport-library-design.md) | Active design | The question is storage/transport substrate direction for SSD↔HBM, DRAM↔HBM, T2/T3 KV movement, or the proposed transport crate boundary. |
| [plans/2026-04-28-single-node-multi-gpu.md](plans/2026-04-28-single-node-multi-gpu.md) | Active | The question is the single-node multi-GPU plan (F0–F8 phases) for TP/PP/EP scaffolding and forward collectives. |
| [plans/2026-04-28-multi-gpu-f0-verification.md](plans/2026-04-28-multi-gpu-f0-verification.md) | Active | The question is the F0 verification protocol (NCCL link, rendezvous, all-reduce smoke, single-rank no-regression gate). |
| [plans/2026-05-01-longctx-spec-decode-phase2.md](plans/2026-05-01-longctx-spec-decode-phase2.md) | Active | The question is Phase 2 long-context speculative decode integration on top of the closed Phase 1 W1 c=4 SGLang row. |
| [plans/M_medusa-phase1b-qwen35-v2-snapshot-ring-redesign.md](plans/M_medusa-phase1b-qwen35-v2-snapshot-ring-redesign.md) | Active gate | The question is how to make Qwen3.5 safe for Medusa/spec verification. Start here for Qwen3.5 Medusa work. |
| [plans/2026-05-01-mla-kernel-design.md](plans/2026-05-01-mla-kernel-design.md) | Design only | The question is the DeepSeek-family MLA CUDA kernel design (DS3) — formula, cache layout, prefill/decode dispatch. |
| [plans/2026-05-02-agent-load-bench-spec.md](plans/2026-05-02-agent-load-bench-spec.md) | Active | The question is the W3/W4 agent-load benchmark contract: short-prompt multi-turn, tool-call resume, session affinity, cache metrics, four-engine baseline evidence. |
| [plans/2026-05-03-a8-gpu-sm-kv-io-kernel.md](plans/2026-05-03-a8-gpu-sm-kv-io-kernel.md) | Pending — gated on W4 close | The question is whether to swap `cudaMemcpyAsync` for an SM-driven kernel on T0↔T1 paged-block transfers (LMSYS 3× claim). Read before touching `kv_tier/transport`. |
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
| [plans/guidellm-integration.md](plans/guidellm-integration.md) | Canonical `guidellm` parameter set and bench wrapper contract. |

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
| [resources/metal-dflash.md](resources/metal-dflash.md) | DFlash usage runbook |
| [resources/metal-dflash-params.md](resources/metal-dflash-params.md) | DFlash CLI parameter reference |
| [resources/kv-cache-quantization.md](resources/kv-cache-quantization.md) | KV-cache quantization formats and operator-side guidance |
| [resources/infer-cuda-profiling-wrappers.md](resources/infer-cuda-profiling-wrappers.md) | `nsys` / `ncu` wrapper scripts |
| [resources/eli-integration.md](resources/eli-integration.md) | Eli sibling-repo integration runbook; layer-2 nexil session-id forwarding shipped 2026-05-07. |

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

### Projects (archived)

| Path | Why kept |
| --- | --- |
| [projects/2026-05-07-metal-world-first-strategy.md](projects/2026-05-07-metal-world-first-strategy.md) | Consolidated 2026-05-07 Metal strategy synthesis (SOTA gap audit + unification recalibration + sequencing). Folds in three earlier same-day notes; current state pointer is ROADMAP P3 and `mlx-backend-roadmap.md`. |
| [projects/2026-04-29-scheduler-pipeline-map.md](projects/2026-04-29-scheduler-pipeline-map.md) | End-to-end CUDA scheduler walk-through with file:line cites; referenced from `mla-kernel-design.md` and the longctx project. |
| [projects/2026-04-29-perf-bug-roundup.md](projects/2026-04-29-perf-bug-roundup.md) | SGLang-parity perf bug ledger; cited by `bench-and-trace-spec.md` and the throughput-gap analysis. |
| [projects/2026-04-29-throughput-gap-analysis.md](projects/2026-04-29-throughput-gap-analysis.md) | "Why we're 28% behind SGLang at c=16" snapshot; cited by the longctx project. |
| [projects/2026-04-30-arle-vs-sglang-admission.md](projects/2026-04-30-arle-vs-sglang-admission.md) | Admission policy gap matrix; sibling to active SGLang admission research note. |
| [projects/2026-05-02-tilekernels-integration-decision.md](projects/2026-05-02-tilekernels-integration-decision.md) | Decision record (don't-submodule, port-selectively) for `cklxx/TileKernels`; referenced from the multi-backend plan. |
| [projects/2026-05-07-eli-arle-native-provider-design.md](projects/2026-05-07-eli-arle-native-provider-design.md) | Layer-2 nexil ↔ arle native-provider design; shipped 2026-05-07 (`http_server/openai_v1.rs` session_id forwarding). Kept as post-implementation reference. |

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
