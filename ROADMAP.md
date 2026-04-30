# ARLE Roadmap

Updated 2026-04-27.

This file is a derived planning surface. If it conflicts with a canonical
document, the canonical document wins:

- support status: [`docs/support-matrix.md`](docs/support-matrix.md)
- workspace topology: [`docs/codebase-map.md`](docs/codebase-map.md)
- architecture boundaries: [`docs/architecture.md`](docs/architecture.md)
- benchmark process: [`docs/bench-and-trace-spec.md`](docs/bench-and-trace-spec.md)
- contributor operating contract: [`AGENTS.md`](AGENTS.md)

## Released

- **v0.1.1 — 2026-04-27.** Install ergonomics, TileLang / KV-tier
  follow-ups, macOS Metal link cleanup, and the first Qwen3.5 GGUF Q4
  Metal closure round. See
  [GitHub Release](https://github.com/cklxx/arle/releases/tag/v0.1.1)
  and [`CHANGELOG.md` §0.1.1](CHANGELOG.md).
- **v0.1.0 — 2026-04-26.** First tagged release. CUDA Stable, Metal /
  Metal DFlash Beta, Qwen3 + Qwen3.5, unified `arle` front door, Docker
  image on GHCR, prebuilt Linux + macOS tarballs. See
  [GitHub Release](https://github.com/cklxx/arle/releases/tag/v0.1.0)
  and [`CHANGELOG.md` §0.1.0](CHANGELOG.md).

## Project Positioning

`ARLE` is a Rust-native inference runtime with integrated local
agent/train/self-evolution workflows.

- The runtime stays primary.
- `infer` owns serving/runtime truth.
- `arle` is the unified front door for local agent, train, eval, and data
  workflows built on that runtime.
- Train/RL work is strategic because it strengthens the runtime loop; it is not
  a second equal product line with its own competing architecture.

## Current Baseline

As of 2026-04-28, the repository already ships:

- CUDA as the primary serving path for Qwen3 and Qwen3.5-family models, with
  continuous batching, paged KV, radix-backed prefix reuse, FlashInfer-backed
  prefill/decode, and OpenAI-compatible HTTP surfaces.
- Metal as the Apple Silicon serving path, including scheduler-backed serving,
  live prefix reuse, Beta DFlash work, and a measured Qwen3.5-0.8B MLX 4bit
  single-request step-driver result of 305.5 tok/s on M4 Pro 20c for
  `1024/256`. The matched GGUF Q4_K_M exact default is 202.1 tok/s direct;
  the opt-in native-q4 load path is 236.7 tok/s direct / 239.8 tok/s
  step-driver and remains a separate exact-K-quant kernel/format target.
- A strong local tiered-KV path (`T0 GPU -> T1 host pinned -> T2 local disk`,
  with a minimal shared backend surface for cluster-shared experiments).
- A runtime-led local agent/train/eval stack: `arle` as the unified front
  door (`arle run`, `arle serve`, `arle train {pretrain,sft,grpo,multi-turn,eval}`,
  `arle data {download,convert}`), plus the train-side
  `/v1/train/{status,events,stop,save}` control plane.

Evidence for performance claims lives under
[`docs/experience/wins/`](docs/experience/wins/), produced through the
canonical `guidellm` flow.

## Active Priorities

| Priority | Goal | Current truth / anchor |
| --- | --- | --- |
| P0 | **32k–128k 长上下文吞吐 — World #1 by ≥30% mission**：在 W1 max-throughput (32k×c=4) + W2 long-decode (32k+2048×c=4) 两个 workload，于 L4 + H100 (+ Apple) 三档硬件上，对 SGLang / vLLM / TRT-LLM / Mooncake 4 家 baseline 同时领先 ≥30%。4 phase 顺序执行：Phase 1 split-KV varlen FP8 + Mixed wire (catch-up) → Phase 2 long-ctx spec decode (MagicDec/TriForce, ×2-2.5) → Phase 3 disaggregated prefill/decode (Mooncake-aligned, ×1.5) → Phase 4 sparse near-lossless (DuoAttention 可选, ×1.3)。当前 Phase 1 = 50%。 | [`docs/projects/2026-04-30-longctx-32k-128k-leadership.md`](docs/projects/2026-04-30-longctx-32k-128k-leadership.md)，父审计 [`docs/plans/2026-04-23-cuda-decode-sglang-alignment.md`](docs/plans/2026-04-23-cuda-decode-sglang-alignment.md) |
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
had become a stale second source of truth. The 2026-04-25 truth-surface cleanup
also retired the inactive `docs/plans/`, `docs/projects/`, `docs/research/`,
`docs/reviews/`, `docs/archives/`, and `docs/areas/` entries that no longer
described current reality.

Engineering history now lives in:

- `docs/experience/wins/` and `docs/experience/errors/` (curated evidence log)
- `CHANGELOG.md`
- `git log`

Use [`docs/index.md`](docs/index.md) to find current documents. Anything not
listed there is not a source of truth.
