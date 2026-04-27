# Maintainer Doc Index

> **Looking for getting-started, install, or HTTP API docs?** Go to
> [README.md](../README.md), [docs/install.md](install.md),
> [docs/troubleshooting.md](troubleshooting.md), or
> [docs/http-api.md](http-api.md) instead. This file is for ARLE maintainers
> tracking canonical truth surfaces, active plans, and experience logs.

Last refreshed: 2026-04-26 (front-door polish series).

## Canonical Truth Surfaces

| Concern | Canonical source | Notes |
| --- | --- | --- |
| Support status of backends / APIs / model families / quantization | [support-matrix.md](support-matrix.md) | README and roadmap summarize only. |
| Stability levels and compatibility posture | [stability-policy.md](stability-policy.md) | Do not redefine tiers elsewhere. |
| Workspace topology and module entry points | [codebase-map.md](codebase-map.md) | Source of truth for "what exists today". |
| Architecture ownership and boundaries | [architecture.md](architecture.md) | `infer` owns runtime truth. |
| Benchmark and trace process | [bench-and-trace-spec.md](bench-and-trace-spec.md) | `guidellm` is the canonical e2e benchmark path. |
| Canonical e2e bench tool + parameter set | [plans/guidellm-integration.md](plans/guidellm-integration.md) | Wrapper script `scripts/bench_guidellm.sh` uses these params verbatim. |
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
| [projects/tiered-kv-cache.md](projects/tiered-kv-cache.md) | Active | The question is current KV-tier scope, milestones, or operator-facing status. |
| [projects/tiered-kv-runtime-flow.md](projects/tiered-kv-runtime-flow.md) | Active | The question is how scheduler, RadixCache, and tier coordinator interact at runtime. |
| [projects/active-kv-swap-out-unification.md](projects/active-kv-swap-out-unification.md) | Active | The question is closing the SGLang admission gap by extending the existing tier demote/promote machinery to in-flight active KV (PreemptionMode wired end-to-end). |
| [projects/mlx-backend-roadmap.md](projects/mlx-backend-roadmap.md) | Active | The question is Metal serving closure or MLX runtime direction. |
| [projects/agent-rl-self-evolving.md](projects/agent-rl-self-evolving.md) | Active | The question is how train/RL/self-evolution work strengthens the runtime spine. |
| [projects/agent-first-architecture.md](projects/agent-first-architecture.md) | Active but secondary | The question is long-horizon agent-serving priorities outside the current KV plan. |

## Active Plans

| Path | Status | Use this when |
| --- | --- | --- |
| [plans/2026-04-23-cuda-decode-sglang-alignment.md](plans/2026-04-23-cuda-decode-sglang-alignment.md) | Active — current decode truth | The question is CUDA decode alignment vs SGLang `main`. |
| [plans/infer-observability-v1.md](plans/infer-observability-v1.md) | Active | The question is operator-facing observability, traces, or profiling flow. |
| [plans/2026-04-20-project-constitution-and-refactor-plan.md](plans/2026-04-20-project-constitution-and-refactor-plan.md) | Reference (Tranches T0/T3 completed 2026-04-25) | The question is SSOT identity, project boundaries, or doc/release governance — the constitution itself, not its execution status. |
| [plans/tiered-kv-hicache-readmission.md](plans/tiered-kv-hicache-readmission.md) | Active | The question is staged KV readmission or remote/shared backend follow-up. |
| [plans/rust-agent-rl-single-node.md](plans/rust-agent-rl-single-node.md) | Active | The question is the Phase 6 execution path under the runtime-first rule. |
| [plans/train-runtime-architecture-v1.md](plans/train-runtime-architecture-v1.md) | Active current-architecture map | The question is today's train-side runtime/control-plane factoring. |
| [plans/train-observability-v1.md](plans/train-observability-v1.md) | Active | The question is train-side events, MLflow, OTLP, or W&B export flow. |
| [plans/train-eval-infer-dx-v1.md](plans/train-eval-infer-dx-v1.md) | Active | The question is unified operator DX across train, eval, and infer. |
| [plans/cuda-kernel-crate-extraction.md](plans/cuda-kernel-crate-extraction.md) | Reference (extraction landed; trip wires govern future splits) | The question is whether to peel another layer out of `infer` and what bar that has to clear. |
| [plans/guidellm-integration.md](plans/guidellm-integration.md) | Reference (canonical bench parameters) | The question is the exact `guidellm` parameter set or the bench wrapper contract. |

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

## Historical Material

- `docs/experience/wins/` and `docs/experience/errors/` are the curated
  evidence log. The latest three of each are always-loaded per `AGENTS.md`;
  earlier entries are kept only when they are referenced from a KEEP file or
  document a milestone (M0–M5 tiered-kv, hybrid Qwen3.5 acceptance, train-
  side milestone snapshots, c1–c16 SGLang closure summary).
- `docs/experience/reviews/` is one Codex code-review snapshot retained as
  reference for the cuda-link audit.
- Plans / projects / research / reviews not listed in the active section
  above are not historical fallbacks: they were retired during the
  2026-04-25 truth-surface cleanup. Anything not on this index is not a
  source of truth.

## Truth-surface invariant

Per [`plans/2026-04-20-project-constitution-and-refactor-plan.md`](plans/2026-04-20-project-constitution-and-refactor-plan.md)
§2: every concern in the canonical-truth-surfaces table above has exactly
one definition. Adding a second one (a new index, a parallel `*/docs/`
tree, a sibling status matrix) is a regression and must be rejected at PR
time.
