# Maintainer Doc Index

> **Looking for getting-started, install, or HTTP API docs?** Go to
> [README.md](../README.md), [docs/install.md](install.md),
> [docs/troubleshooting.md](troubleshooting.md), or
> [docs/http-api.md](http-api.md) instead. This file is for ARLE maintainers
> tracking canonical truth surfaces, active plans, and experience logs.

Last refreshed: 2026-05-10 EOD+310 (**🚫🔬 Path B-Phase2' Phase 0 P0.A KILLED `67f18b9` + architectural synthesis `61c9666`** — codex cutlass DIRECT FP8 GEMM smoke (post Claude `d5a6679` sm_89 template unstick: `GemmUniversalWithAbsMax` + `arch::Sm89` + `LinearCombinationGenericWithScalingAndAbsMax` epilogue) ran with `Status::Success` on Qwen3-4B linear shapes. **Decode (M=1 N=4096 K=2560)**: BF16 0.427ms TFLOPS=0.05 vs FP8-fast 0.229ms TFLOPS=0.09 = **1.86× speedup BELOW 2× kill threshold** = decode KILL. **Prefill (M=2048 N=4096 K=2560)**: 5.21× speedup BUT only 159.8 TFLOPS = 22.6% of 706 theoretical (codex flagged below absolute 50% gate); separate TTFT axis. **ARCHITECTURAL INSIGHT**: W4 decode HBM-bound on weight read (already 4× smaller than BF16); FP8 mma helps compute not bandwidth; activation is 0.2% of memory traffic in W4 GEMM → user's "-20-40% ITL via FP8 path" **structurally infeasible** on sm_89 W4 decode; same memory-bound ceiling explains why Machete (sm_90) wouldn't help on sm_89 even if backportable; PushNotification dispatched with revised priority: P0=#28 spec decoding (-50%+ ITL via amortized weight read, blocked on #34 HF Hub), P1=W3/W2 quant research, P2=Phase 1 dequant.h port (-3-8% ITL fallback, e59beb5), P3=NEW prefill-only FP8 directive (~700 LOC, -8-16% TTFT separate from ITL); P0.B PPL gate skipped per 6a6114d decision matrix; Task #41 completed; **🚫 #36 PrefixAwareAdmission Layer 2 KILLED `9bbc441`** Claude self-driven A/B via `scripts/bench_36_warmmix_direct.py` (3b4c89b + b96a1e7 /v1/stats parse fix): warm p95 +17%, **cold p95 +114%**, throughput -3.4%, starvation ratio 4.56× → **8.33×** at cold_soft_cap=3 + 8 slots + c=8. Substrate works (gate fires 73×, prefix_hit_rate=57.2%) but op-point makes every metric regress; `QueueBound` stays default at types.rs:214, `--admission-policy prefix-aware` retained as opt-in CLI. Three follow-up paths documented (cold_headroom sweep / session_id workload / c=32) — none P0 since #40 Path B.2 already delivered single-stream gap closure; **🛠️ skill kernel-optimization v1.10.0 `b96a1e7`** added anti-pattern #28 (hallucinated tool output overrides peer-agent investigation) from `ee2c5b0` SOLID-critical chain — Claude's prior `0f4d0ae` "correction" of codex on `--max-waiting-requests` flag existence was based on fabricated grep evidence; codex was correct from the start; sediment rule: when correcting peer file-content claim MUST re-run verification + quote raw output literally; **🚫 Machete sm_89 BLOCKER reaffirmed `e65a096`** 5-pt convergent evidence (collective_builder + mainloop + generate.py + Readme + 2026-05-09 prior survey all confirm Hopper-only) — default Path B-Phase2' (3e83741: W4+FP8 sm_89 native = real -20-40% ITL mechanism) absent explicit user "Path A confirmed" ack; **🔍 #36 P0 survey `5e902da` finds substrate FULLY WIRED** — all M_b3 steps LANDED (Step 1 `7c8fd61` admission_allows signature + Step 2 `prefix_aware_admission_allows_plan` real-signal pipeline + fail-open anti-starvation + Step 3 `--admission-policy` CLI flag at main.rs:124); **🛠️ skill kernel-optimization v1.9.0 `08d9b7e`** added anti-patterns #26 (smoke-shape ≠ production-shape capture-key cardinality from `a7a8b94` Path B v1 KILL) + #27 (bucketing without scalar-capture sync = semantic miss from `a56b7a9` Path B.2 win) from #37/#40 chain; **🚫 #33 KV W4A8 KILLED `ddf0615`** scalar inline INT4 unpack 1.12× < 1.5× kill gate; **🎉🎉 #40 Path B.2 TIER 1 STRONG PROCEED LANDED `c44788f`** — Engine TTFT p50 **2000ms → 150ms = -92.5%** + 388 unique keys → **7 unique = 98% reduction** + 98.5% LRU dominant key reuse + **+632% throughput** in matched-control 60s window;codex's "second-order bucketing" insight(captured scalar launch parameters use bucket capacity)load-bearing for win;**closes 4k/c=4 SGLang +76.6% gap as P0 outcome**;guidellm 0.6.0 client-side TTFT broken `e8d82b0` — bench tool bug isolated,server-side ground truth stands; **🎉 #24 W4A8 prefill graph capture hoist LANDED `35fc3cf`** opt-in via `INFER_PREFILL_GRAPH=1` + `INFER_HYBRID_W4A8_PREFILL=1`; **🚫 #37 Path A multi-key throughput KILLED `e462c53`** + **🚫 Path B v1 KILLED `a7a8b94`** Tier 4 100% miss at 4k production,Path B.2 bucketing(`a56b7a9`)recovers via 64-entry/128-row bucket dimensions; **🎉 M_rope-yarn-scaling Phase 1+2 LANDED `e30bffe..da53d81`** YARN/Linear/NtkAware scaling support across qwen3-spec + qwen35-spec + `precompute_rope_with_scaling` weight_loader integration,7 atomic commits + 51 unit tests + 1 wins consolidation `11fca7a`; **🚧 #26 M_xgrammar Phase 1 FFI scaffold WIP** codex-side `crates/xgrammar-sys` C++ shim + Rust safe wrapper(1046 LOC,upstream `mlc-ai/xgrammar` v0.1.34,no HTTP/scheduler/sampler integration yet); **🚫 P1.6/P1.4/P1.3 各 KILL** prior axis(prior log保留)。Phase 3 long-ctx YARN bench plan ready `8466202`(Qwen3-4B 64k YARN×2 / 128k YARN×4+FP8 KV CUDA-side viable,无需 Mac,native ctx=40960 confirmed)。Pre-built post-commit chain ready:`scripts/{validate_p24_phase0v3,setup_qwen3_yarn_config,post_p24_commit_pipeline}.sh`。Anchor:[`docs/experience/wins/2026-05-10-m-rope-yarn-scaling-phase1-phase2-landed.md`](experience/wins/2026-05-10-m-rope-yarn-scaling-phase1-phase2-landed.md),[`docs/research/2026-05-10-37-rescope-post-codex-multikey-impl.md`](research/2026-05-10-37-rescope-post-codex-multikey-impl.md). Priority order canonical in [`ROADMAP.md` §Next-Model Priority Order](../ROADMAP.md#next-model-priority-order). Pickup queue: [`docs/plans/codex-pickup-queue-2026-05-09.md`](plans/codex-pickup-queue-2026-05-09.md).

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
| [projects/2026-04-30-longctx-32k-128k-leadership.md](projects/2026-04-30-longctx-32k-128k-leadership.md) | Active — P0 mission | The question is the 32k–128k longctx world-#1 mission (4 phase plan, baseline panel, hardware tiers, current Phase 1 SGLang-row close + Phase 2 plumbing/regression status in §13/§13.A). |
| [projects/2026-05-02-agent-load-mission-expansion.md](projects/2026-05-02-agent-load-mission-expansion.md) | Active — mission expansion | The question is the agent-load world-#1 expansion: W3 short-prompt multi-turn, W4 tool-call resume, session affinity, prefix-cache reuse, and four-engine baseline gates. |
| [projects/2026-05-01-multi-gpu-f0-readiness.md](projects/2026-05-01-multi-gpu-f0-readiness.md) | Active | The question is single-node multi-GPU F0 readiness, scaffolded TP/PP/EP axes, NCCL smoke, and the gap matrix to real multi-rank serving. |
| [projects/2026-05-01-deepseek-v4-readiness.md](projects/2026-05-01-deepseek-v4-readiness.md) | Active — parallel product line | The question is DeepSeek V4 readiness, the DS0–DS8 gap matrix, and what `crates/deepseek-spec/` already scaffolds. |
| [projects/2026-05-01-spec-decode-integration-design.md](projects/2026-05-01-spec-decode-integration-design.md) | Active | The question is how Phase 2 spec decode plumbing integrates with the CUDA scheduler, verifier, and external draft state. |
| [projects/tiered-kv-cache.md](projects/tiered-kv-cache.md) | Active | The question is current KV-tier scope, milestones, or operator-facing status. |
| [projects/tiered-kv-runtime-flow.md](projects/tiered-kv-runtime-flow.md) | Active | The question is how scheduler, RadixCache, and tier coordinator interact at runtime. |
| [projects/mlx-backend-roadmap.md](projects/mlx-backend-roadmap.md) | Active | The question is Metal serving closure, MLX runtime direction, or the Qwen3.5 GGUF decode hot path. |
| [projects/agent-rl-self-evolving.md](projects/agent-rl-self-evolving.md) | Active | The question is how train/RL/self-evolution work strengthens the runtime spine. |
| [projects/agent-first-architecture.md](projects/agent-first-architecture.md) | Active but secondary | The question is long-horizon agent-serving priorities outside the current KV plan. |

## Active Plans

| Path | Status | Use this when |
| --- | --- | --- |
| [plans/2026-04-28-single-node-multi-gpu.md](plans/2026-04-28-single-node-multi-gpu.md) | Active | The question is the single-node multi-GPU plan (F0–F8 phases) for TP/PP/EP scaffolding and forward collectives. |
| [plans/2026-04-28-multi-gpu-f0-verification.md](plans/2026-04-28-multi-gpu-f0-verification.md) | Active | The question is the F0 verification protocol (NCCL link, rendezvous, all-reduce smoke, single-rank no-regression gate). |
| [plans/2026-05-01-longctx-spec-decode-phase2.md](plans/2026-05-01-longctx-spec-decode-phase2.md) | Active | The question is Phase 2 long-context speculative decode integration on top of the closed Phase 1 W1 c=4 SGLang row. |
| [plans/2026-05-01-mla-kernel-design.md](plans/2026-05-01-mla-kernel-design.md) | Design only | The question is the DeepSeek-family MLA CUDA kernel design (DS3) — formula, cache layout, prefill/decode dispatch. |
| [plans/2026-05-02-agent-load-bench-spec.md](plans/2026-05-02-agent-load-bench-spec.md) | Active | The question is the W3/W4 agent-load benchmark contract: short-prompt multi-turn, tool-call resume, session affinity, cache metrics, and four-engine baseline evidence. |
| [plans/2026-05-03-a8-gpu-sm-kv-io-kernel.md](plans/2026-05-03-a8-gpu-sm-kv-io-kernel.md) | Pending — 待落地 (gated on (A) closing W4) | The question is whether to swap `cudaMemcpyAsync` for an SM-driven kernel on T0↔T1 paged-block transfers (LMSYS 3× claim). Read before touching `kv_tier/transport`. |
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
