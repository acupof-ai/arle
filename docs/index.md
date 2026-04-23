# Doc Index

Maintainer-facing index for current truth. This is not the public landing page.

Last refreshed: 2026-04-23.

## Canonical Truth Surfaces

| Concern | Canonical source | Notes |
| --- | --- | --- |
| Support status of backends / APIs / model families / quantization | [support-matrix.md](support-matrix.md) | README and roadmap summarize only. |
| Stability levels and compatibility posture | [stability-policy.md](stability-policy.md), [compatibility.md](compatibility.md) | Do not redefine tiers elsewhere. |
| Workspace topology and module entry points | [codebase-map.md](codebase-map.md) | Source of truth for "what exists today". |
| Architecture ownership and boundaries | [architecture.md](architecture.md) | `infer` owns runtime truth. |
| Benchmark and trace process | [bench-and-trace-spec.md](bench-and-trace-spec.md) | `guidellm` is the canonical e2e benchmark path. |
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
| [projects/mlx-backend-roadmap.md](projects/mlx-backend-roadmap.md) | Active | The question is Metal serving closure or MLX runtime direction. |
| [projects/agent-rl-self-evolving.md](projects/agent-rl-self-evolving.md) | Active | The question is how train/RL/self-evolution work strengthens the runtime spine. |
| [projects/agent-first-architecture.md](projects/agent-first-architecture.md) | Active but secondary | The question is long-horizon agent-serving priorities outside the current KV plan. |

## Active Plans

| Path | Status | Use this when |
| --- | --- | --- |
| [plans/2026-04-23-cuda-decode-sglang-alignment.md](plans/2026-04-23-cuda-decode-sglang-alignment.md) | Active — current decode truth | The question is CUDA decode alignment vs SGLang `main`. |
| [plans/infer-observability-v1.md](plans/infer-observability-v1.md) | Active | The question is operator-facing observability, traces, or profiling flow. |
| [plans/2026-04-20-project-constitution-and-refactor-plan.md](plans/2026-04-20-project-constitution-and-refactor-plan.md) | Active | The question is SSOT, project identity, or doc/release/toolchain cleanup. |
| [plans/tiered-kv-hicache-readmission.md](plans/tiered-kv-hicache-readmission.md) | Active | The question is staged KV readmission or remote/shared backend follow-up. |
| [plans/rust-agent-rl-single-node.md](plans/rust-agent-rl-single-node.md) | Active | The question is the Phase 6 execution path under the runtime-first rule. |
| [plans/train-runtime-architecture-v1.md](plans/train-runtime-architecture-v1.md) | Active current-architecture map | The question is today's train-side runtime/control-plane factoring. |
| [plans/train-observability-v1.md](plans/train-observability-v1.md) | Active | The question is train-side events, MLflow, OTLP, or W&B export flow. |
| [plans/train-eval-infer-dx-v1.md](plans/train-eval-infer-dx-v1.md) | Active | The question is unified operator DX across train, eval, and infer. |

## Operator And Policy References

| Path | Role |
| --- | --- |
| [http-api.md](http-api.md) | HTTP contract and streaming behavior |
| [environment.md](environment.md) | Environment variables and runtime knobs |
| [release-checklist.md](release-checklist.md) | Release prep and artifact verification |
| [perf-and-correctness-gates.md](perf-and-correctness-gates.md) | Lightweight validation expectations by change type |

## Historical Material

- `docs/experience/wins/`, `docs/experience/errors/`, and
  `docs/experience/reviews/` are the evidence log and learning archive. The
  latest three wins/errors are always-loaded per `AGENTS.md`; the directories
  themselves are not re-indexed here because that listing decays quickly.
- `docs/archives/` contains inactive architectural proposals kept only as
  historical context.
- Plans not listed in the active section above should be treated as superseded
  or historical unless they explicitly mark themselves as the current source of
  truth for a topic.
