# ARLE Roadmap

Updated 2026-05-15. Derived planning surface. On any conflict the canonical
doc wins:

- Strategic master: [`docs/projects/2026-05-07-arle-master-strategy.md`](docs/projects/2026-05-07-arle-master-strategy.md)
- Support status: [`docs/support-matrix.md`](docs/support-matrix.md)
- Workspace topology: [`docs/codebase-map.md`](docs/codebase-map.md)
- Architecture boundaries: [`docs/architecture.md`](docs/architecture.md)
- Benchmark process: [`docs/bench-and-trace-spec.md`](docs/bench-and-trace-spec.md)
- Contributor contract: [`AGENTS.md`](AGENTS.md)

## Positioning

ARLE is a Rust-native inference runtime with integrated local agent / train
/ self-evolution workflows. Runtime stays primary: `infer` owns serving truth;
`arle` is the unified front door; train/RL strengthens the runtime loop, it
does not fork a second product identity.

## Active Priorities

| # | Goal | Anchor |
| --- | --- | --- |
| **P0** | **DeepSeek V4 8xH20 readiness** — DSv4 DeepEP decode is the active hot path. Default B=1 padded BF16 reduce-scatter combine + fused local-expert prepare + broad scratch reuse landed; `decode64` 12.05 post-first tok/s, isolated single-token wave 87.7 ms. Remaining: NCCL SendRecv/AllReduce, FP8/FP4 expert GEMV (awaits true grouped GEMM / DeepGEMM), launch churn. | [`docs/projects/2026-05-01-deepseek-v4-readiness.md`](docs/projects/2026-05-01-deepseek-v4-readiness.md), [`docs/experience/errors/2026-05-14-dsv4-decode-nccl-bottleneck.md`](docs/experience/errors/2026-05-14-dsv4-decode-nccl-bottleneck.md) |
| **P0'** | **32k–128k long-context world-#1 mission** — Phase 1 SGLang-row closed (`1.609x` mean at W1/c4); Phase 2 spec-decode plumbing landed but first end-to-end bench regressed (9.73 tok/s, -62.8% vs Phase 1), blocked on packed K+1 verifier / MagicDec sparse-KV self-spec / Qwen3.5 recurrent-state rollback. vLLM / TRT-LLM / Mooncake baseline panel still required for full world-#1 claim. | [`docs/projects/2026-04-30-longctx-32k-128k-leadership.md`](docs/projects/2026-04-30-longctx-32k-128k-leadership.md), [`docs/plans/2026-05-01-longctx-spec-decode-phase2.md`](docs/plans/2026-05-01-longctx-spec-decode-phase2.md) |
| **P0''** | **Single-node multi-GPU F0–F4 scaffold** — F0 NCCL group-coordinator smoke + 2-thread `all_reduce` proven; F1 shard-aware BF16 `TpLoadContext` landed; F2 Qwen3 + Qwen3.5 TP forward sharding wired through `LayerCommunicator` (TP=1 no-op, TP>1 fails fast until production NCCL forward collectives land); F3/F4 PP + EP scaffolds present. Gated on F2 collective wire-up + TP=2 H20 throughput bench. | [`docs/projects/2026-05-01-multi-gpu-f0-readiness.md`](docs/projects/2026-05-01-multi-gpu-f0-readiness.md) |
| **P1** | Finish the `infer`-side observability spine: throughput, TTFT, ITL, queue shape, `ncu`, `nsys`, sampled traces on one operator surface. | [`docs/plans/infer-observability-v1.md`](docs/plans/infer-observability-v1.md) |
| **P2** | Push tiered KV from a strong local CUDA path toward validated staged readmission + remote/shared backends. | [`docs/projects/tiered-kv-cache.md`](docs/projects/tiered-kv-cache.md), [`docs/plans/tiered-kv-hicache-readmission.md`](docs/plans/tiered-kv-hicache-readmission.md) |
| **P3** | Finish serving-grade Metal batching and long-context closure without forking runtime truth away from CUDA. | [`docs/projects/mlx-backend-roadmap.md`](docs/projects/mlx-backend-roadmap.md) |
| **P4** | Keep Phase 6 train/agent work runtime-led: shared model truth, unified operator surface, no second project identity. | [`docs/projects/agent-rl-self-evolving.md`](docs/projects/agent-rl-self-evolving.md), [`docs/plans/train-runtime-architecture-v1.md`](docs/plans/train-runtime-architecture-v1.md) |

## Next-Model Priority Order

Currently shipped: Qwen3 + Qwen3.5-family. Going forward the model-coverage
queue is ranked, not parallel:

1. **DeepSeek V4 (DS4)** — highest priority. V4-only spec, registry, train
   bootstrap, and CPU reference HTTP smoke have landed against
   `infer/models/dsv4-mini-1B-init`; CUDA V4 hybrid attention + MoE + MTP
   kernels are the active substrate (see P0).
2. **Qwen 3.6** — second priority, planned. Metal already loads
   `mlx-community/Qwen3.6-35B-A3B-4bit` for diagnostic use; CUDA serving and
   DFlash long-context evidence land after DS4's runtime substrate is
   producing benches.

Other families in the support matrix (Llama 3 / 4, DeepSeek V3/R1, Mistral /
Mixtral / Gemma / Phi) sit behind these two and are not actively scheduled.

## History

Released tags + bench evidence live in:

- [`CHANGELOG.md`](CHANGELOG.md) — per-version notes (latest: v0.1.5)
- [`docs/experience/wins/`](docs/experience/wins/), [`docs/experience/errors/`](docs/experience/errors/) — curated evidence log
- [GitHub Releases](https://github.com/cklxx/arle/releases) — tagged binaries
- `git log` — full history

Use [`docs/index.md`](docs/index.md) to find current documents. Anything not
listed there is not a source of truth.
