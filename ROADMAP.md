# agent-infer Roadmap

Updated 2026-04-23.

This file is a derived planning surface. If it conflicts with a canonical
document, the canonical document wins:

- support status: [`docs/support-matrix.md`](docs/support-matrix.md)
- workspace topology: [`docs/codebase-map.md`](docs/codebase-map.md)
- architecture boundaries: [`docs/architecture.md`](docs/architecture.md)
- benchmark process: [`docs/bench-and-trace-spec.md`](docs/bench-and-trace-spec.md)
- contributor operating contract: [`AGENTS.md`](AGENTS.md)

## Project Positioning

`agent-infer` is a Rust-native inference runtime with integrated local
agent/train/self-evolution workflows.

- The runtime stays primary.
- `infer` owns serving/runtime truth.
- `arle` is the unified front door for local agent, train, eval, and data
  workflows built on that runtime.
- Train/RL work is strategic because it strengthens the runtime loop; it is not
  a second equal product line with its own competing architecture.

## Current Baseline

As of 2026-04-23, the repository already ships:

- CUDA as the primary serving path for Qwen3 and Qwen3.5-family models, with
  continuous batching, paged KV, radix-backed prefix reuse, FlashInfer-backed
  prefill/decode, and OpenAI-compatible HTTP surfaces.
- Metal as the Apple Silicon serving path, including scheduler-backed serving,
  live prefix reuse, and Beta DFlash work.
- A strong local tiered-KV path (`T0 GPU -> T1 host pinned -> T2 local disk`,
  with a minimal shared backend surface for cluster-shared experiments).
- A runtime-led local agent/train/eval stack: `arle`, `pretrain`, `train_sft`,
  `train_grpo`, `train_multi_turn`, `eval_lm`, and the train-side
  `/v1/train/{status,events,stop,save}` control plane.

Evidence for performance claims lives under
[`docs/experience/wins/`](docs/experience/wins/), produced through the
canonical `guidellm` flow.

## Active Priorities

| Priority | Goal | Current truth / anchor |
| --- | --- | --- |
| P0 | Close the remaining CUDA decode and mixed-batch gap against SGLang `main` without reopening duplicate runtime surfaces. | [`docs/plans/2026-04-23-cuda-decode-sglang-alignment.md`](docs/plans/2026-04-23-cuda-decode-sglang-alignment.md) |
| P1 | Finish the infer-side observability spine so throughput, TTFT, ITL, queue shape, `ncu`, `nsys`, and sampled traces sit on one operator-facing surface. | [`docs/plans/infer-observability-v1.md`](docs/plans/infer-observability-v1.md) |
| P2 | Push tiered KV from a strong local CUDA path toward fully validated staged readmission and remote/shared backends. | [`docs/projects/tiered-kv-cache.md`](docs/projects/tiered-kv-cache.md), [`docs/plans/tiered-kv-hicache-readmission.md`](docs/plans/tiered-kv-hicache-readmission.md) |
| P3 | Finish serving-grade Metal batching and long-context closure without forking runtime truth away from CUDA. | [`docs/projects/mlx-backend-roadmap.md`](docs/projects/mlx-backend-roadmap.md) |
| P4 | Keep Phase 6 train/agent work runtime-led: shared model truth, unified operator surface, and no second independent project identity. | [`docs/projects/agent-rl-self-evolving.md`](docs/projects/agent-rl-self-evolving.md), [`docs/plans/rust-agent-rl-single-node.md`](docs/plans/rust-agent-rl-single-node.md), [`docs/plans/train-runtime-architecture-v1.md`](docs/plans/train-runtime-architecture-v1.md) |
| P5 | Finish the constitution / SSOT / release cleanup so README, roadmap, index, CI, release packaging, and benchmark workflow all describe the same project. | [`docs/plans/2026-04-20-project-constitution-and-refactor-plan.md`](docs/plans/2026-04-20-project-constitution-and-refactor-plan.md) |

## What "Done" Looks Like

This roadmap revision is only useful if all of the following hold:

1. A maintainer can answer "what is current?" without reading multiple stale
   phase lists.
2. README, roadmap, and index summarize the same runtime-first project.
3. Support claims match `docs/support-matrix.md`.
4. Performance claims match dated `guidellm` evidence.
5. Train/agent work strengthens the runtime spine instead of inventing a
   second product boundary.

## Historical Note

The old phase-by-phase long-form roadmap was removed from this file because it
had become a stale second source of truth.

History still lives in:

- `docs/experience/wins/`
- `docs/experience/errors/`
- `docs/projects/`
- `docs/plans/`
- `docs/archives/`

Use [`docs/index.md`](docs/index.md) to find the current documents. Treat plans
or project notes not called out there as historical context unless they
explicitly mark themselves as the current source of truth for their topic.
