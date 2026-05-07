# ARLE Roadmap

Updated 2026-05-08.

> ⚠️ **Strategic master:** [`docs/projects/2026-05-07-arle-master-strategy.md`](docs/projects/2026-05-07-arle-master-strategy.md)
> defines ARLE 双线产品(coding/agent runtime + DSV4 from-scratch training)
> + 5-cap moat + P0/P1/P2 sequence。本 ROADMAP 是 derived planning surface,
> 与 master 冲突以 master 为准。

This file is a derived planning surface. If it conflicts with a canonical
document, the canonical document wins:

- **战略主文档**: [`docs/projects/2026-05-07-arle-master-strategy.md`](docs/projects/2026-05-07-arle-master-strategy.md)

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
| P0 | **32k–128k 长上下文吞吐 — World #1 by ≥30% mission**：在 W1 max-throughput (32k×c=4) + W2 long-decode (32k+2048×c=4) 两个 workload，于 L4 + H100 (+ Apple) 三档硬件上，对 SGLang / vLLM / TRT-LLM / Mooncake 4 家 baseline 同时领先 ≥30%。4 phase 顺序执行：Phase 1 split-KV varlen FP8 + Mixed wire (catch-up) → Phase 2 long-ctx spec decode (MagicDec/TriForce, ×2-2.5) → Phase 3 disaggregated prefill/decode (Mooncake-aligned, ×1.5) → Phase 4 sparse near-lossless (DuoAttention 可选, ×1.3)。当前状态：Phase 1 SGLang-row closed 2026-05-01, W1/c4 mean `1.609x` SGLang (worst `1.469x`, above the `1.30x` mission margin); remaining vLLM / TRT-LLM / Mooncake baseline panel still required for full world-#1 claim. Phase 2 spec-decode plumbing (external draft request-state lifecycle + K-token proposals + greedy verifier + bonus-token commit) landed but its first end-to-end bench regressed to `9.73` headline tok/s (-62.8% vs Phase 1 close), tracked in `docs/experience/errors/2026-05-01-phase2-real-spec-regression.md`; Phase 2 throughput claims paused until a packed K+1 verifier or MagicDec sparse-KV self-spec path lands. | [`docs/projects/2026-04-30-longctx-32k-128k-leadership.md`](docs/projects/2026-04-30-longctx-32k-128k-leadership.md)，父审计 [`docs/plans/2026-04-23-cuda-decode-sglang-alignment.md`](docs/plans/2026-04-23-cuda-decode-sglang-alignment.md), Phase 2 plan [`docs/plans/2026-05-01-longctx-spec-decode-phase2.md`](docs/plans/2026-05-01-longctx-spec-decode-phase2.md) |
| P0' | **Multi-GPU single-node F0–F4 scaffold (TP/PP/EP axes)** — F0 NCCL group-coordinator smoke + 2-thread `all_reduce` proven; F1 `parallel_state` + shard-aware BF16 `TpLoadContext` landed; F2 Qwen3 + Qwen3.5 TP forward sharding wired through `LayerCommunicator` (TP=1 no-op, TP>1 fails fast until production NCCL forward collectives land); F3 pipeline-parallel scaffold + F4 expert-parallel scaffold present. Production multi-rank serving still gated on the F2 collective wire-up + a TP=2 throughput bench on the H20 box. | [`docs/projects/2026-05-01-multi-gpu-f0-readiness.md`](docs/projects/2026-05-01-multi-gpu-f0-readiness.md), [`docs/plans/2026-04-28-single-node-multi-gpu.md`](docs/plans/2026-04-28-single-node-multi-gpu.md), [`docs/plans/2026-04-28-multi-gpu-f0-verification.md`](docs/plans/2026-04-28-multi-gpu-f0-verification.md) |
| P0'' | **DeepSeek V4 readiness prep (parallel product line, #1 next-model priority)** — readiness assessment landed and DS0 spec crate scaffolded under `crates/deepseek-spec/` (config + tensor names + `Shard` annotations). MLA kernel design landed. Runtime substrate scaffold + nano autograd training landed 2026-05-05 (`infer/src/model/deepseek/*`, `arle train pretrain-dsv4 --deepseek-config nano`). Implementation (DS1 registry, DS2 block-FP8, DS3 MLA cache/kernels, DS4 CUDA MoE forward, DS5 NCCL collectives in forward, DS6 MTP, DS7 scheduler routing, DS8 baselines) is gated on the F0–F4 multi-GPU scaffold completing real collectives in forward. | [`docs/projects/2026-05-01-deepseek-v4-readiness.md`](docs/projects/2026-05-01-deepseek-v4-readiness.md), [`docs/plans/2026-05-01-mla-kernel-design.md`](docs/plans/2026-05-01-mla-kernel-design.md), [`docs/plans/2026-05-05-deepseek-v4-small-substrate.md`](docs/plans/2026-05-05-deepseek-v4-small-substrate.md) |
| P1 | Finish the infer-side observability spine so throughput, TTFT, ITL, queue shape, `ncu`, `nsys`, and sampled traces sit on one operator-facing surface. | [`docs/plans/infer-observability-v1.md`](docs/plans/infer-observability-v1.md) |
| P2 | Push tiered KV from a strong local CUDA path toward fully validated staged readmission and remote/shared backends. | [`docs/projects/tiered-kv-cache.md`](docs/projects/tiered-kv-cache.md), [`docs/plans/tiered-kv-hicache-readmission.md`](docs/plans/tiered-kv-hicache-readmission.md) |
| P3 | Finish serving-grade Metal batching and long-context closure without forking runtime truth away from CUDA. | [`docs/projects/mlx-backend-roadmap.md`](docs/projects/mlx-backend-roadmap.md) |
| P4 | Keep Phase 6 train/agent work runtime-led: shared model truth, unified operator surface, and no second independent project identity. | [`docs/projects/agent-rl-self-evolving.md`](docs/projects/agent-rl-self-evolving.md), [`docs/plans/rust-agent-rl-single-node.md`](docs/plans/rust-agent-rl-single-node.md), [`docs/plans/train-runtime-architecture-v1.md`](docs/plans/train-runtime-architecture-v1.md) |
| P5 | Finish the constitution / SSOT / release cleanup so README, roadmap, index, CI, release packaging, and benchmark workflow all describe the same project. | [`docs/plans/2026-04-20-project-constitution-and-refactor-plan.md`](docs/plans/2026-04-20-project-constitution-and-refactor-plan.md) |

## Next-Model Priority Order

Currently shipped: Qwen3 + Qwen3.5-family. Going forward the model-coverage
queue is ranked, not parallel:

1. **DeepSeek V4 (DS4)** — highest-priority next model. Substrate scaffold has
   landed (DS0 spec crate, runtime model skeleton, nano autograd training);
   DS3 MLA kernels + DS4 CUDA MoE forward + DS5 NCCL collectives in forward
   are the work this lane is converging on. Tracked under P0'' above.
2. **Qwen 3.6** — second priority, planned / scoping. The Metal path already
   loads `mlx-community/Qwen3.6-35B-A3B-4bit` for diagnostic use
   ([`docs/support-matrix.md` §3](docs/support-matrix.md#3-model-family-matrix));
   CUDA serving, kernel coverage, and the DFlash long-context evidence bar
   land after DS4's runtime substrate is producing benches.

Other planned families listed in the support matrix (Llama 3 / 4,
DeepSeek-V3/R1, Mistral / Mixtral / Gemma / Phi) sit behind these two and
are not actively scheduled.

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
