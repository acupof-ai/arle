# Maintainer Doc Index

> Getting-started / install / HTTP API: [README.md](../README.md),
> [install.md](install.md), [onboarding.md](onboarding.md),
> [troubleshooting.md](troubleshooting.md), [http-api.md](http-api.md).
> This file is a pure index — no narrative state.

**Current phase / models:** [`ROADMAP.md`](../ROADMAP.md)  
**Progress spine:** [`CHANGELOG.md`](../CHANGELOG.md)  
**Agent contract:** [`../AGENTS.md`](../AGENTS.md) · [`../CLAUDE.md`](../CLAUDE.md)

Pre-2026-07 dated plans/wins/errors/research were purged 2026-07-21
(`git log -- docs/` to recover). Live dated corpus is **2026-07-*** only.

---

## Canonical truth surfaces

| Concern | Source |
| --- | --- |
| Onboarding (30 min) | [onboarding.md](onboarding.md) |
| Backend / model / quant tiers | [support-matrix.md](support-matrix.md) |
| Quantization map | [quantization.md](quantization.md) |
| Stability tiers | [stability-policy.md](stability-policy.md) |
| Workspace topology | [codebase-map.md](codebase-map.md) |
| Package boundaries | [architecture.md](architecture.md) |
| DSv4/GLM path map | [architecture-dsv4.md](architecture-dsv4.md) |
| Bench + trace process | [bench-and-trace-spec.md](bench-and-trace-spec.md) |
| Rolling baselines | [baselines.md](baselines.md) |
| Env / flags | [environment.md](environment.md) |
| Capability evals | [eval.md](eval.md) |
| OPD capability curve | [opd-capability-curve.md](opd-capability-curve.md) |
| Kernel live index | [reviews/kernel-registry.md](reviews/kernel-registry.md) |

---

## Positioning (fact)

- Runtime-first: `infer-plan` → `infer-seam` → `infer-core` →
  `infer-cuda`/`infer-metal` → `infer-server`/`infer-api`. Monolith `infer/` deleted.
- `arle` = CLI front door; `infer-api` (`LoadedInferenceEngine`) = programmatic front door.
- Train = **OPD-only** substrate on the same runtime (pretrain/SFT/GRPO multi-turn retired).
- Metal canonical model: `mlx-community/Qwen3.6-35B-A3B-4bit`.

---

## Live projects

| Path | Use when |
| --- | --- |
| [projects/mlx-backend-roadmap.md](projects/mlx-backend-roadmap.md) | Metal/MLX serving direction |
| [projects/agent-rl-self-evolving.md](projects/agent-rl-self-evolving.md) | train/RL strengthens runtime spine |
| [projects/agent-first-architecture.md](projects/agent-first-architecture.md) | long-horizon agent-serving priorities |
| [projects/tiered-kv-cache.md](projects/tiered-kv-cache.md) | historical tiered-KV design; runtime truth is code |

---

## Live plans (2026-07)

Controlling / high-traffic:

| Path | Use when |
| --- | --- |
| [plans/2026-07-21-kernel-opt-fusion-campaign.md](plans/2026-07-21-kernel-opt-fusion-campaign.md) | remaining kernel/path opt (slots→B≳43 primary, FA3-decode probe, EP) |
| [plans/2026-07-11-dsv4-high-concurrency-throughput-campaign.md](plans/2026-07-11-dsv4-high-concurrency-throughput-campaign.md) | DSv4 high-c thruput |
| [plans/2026-07-11-dspark-dsv4-flash-spec-decode.md](plans/2026-07-11-dspark-dsv4-flash-spec-decode.md) | DSpark on DSv4 |
| [plans/2026-07-09-dspark-dflash-spec-decode-qwen36.md](plans/2026-07-09-dspark-dflash-spec-decode-qwen36.md) | DSpark/DFlash on Qwen3.6 (licensed) |
| [plans/2026-07-19-dspark-train-sidecar.md](plans/2026-07-19-dspark-train-sidecar.md) | DSpark train sidecar |
| [plans/2026-07-02-deepspec-adoption-map.md](plans/2026-07-02-deepspec-adoption-map.md) | DeepSpec map |
| [plans/2026-07-10-operator-artifact-dev-release-system.md](plans/2026-07-10-operator-artifact-dev-release-system.md) | operator artifacts / releases |

Full list: `ls docs/plans/2026-07-*` (36 plans).

---

## Experience / reviews / research

| Path | Role |
| --- | --- |
| [experience/wins/](experience/wins/) | July licensed wins / benches |
| [experience/errors/](experience/errors/) | July kills / regressions |
| [experience/wins/TEMPLATE-bench.md](experience/wins/TEMPLATE-bench.md) | bench entry skeleton |
| [reviews/kernel-registry.md](reviews/kernel-registry.md) | live CUDA operator index |
| [reviews/2026-07-10-operator-platform-m0-readonly-audit.md](reviews/2026-07-10-operator-platform-m0-readonly-audit.md) | operator platform M0 |
| [research/2026-07-11-rl-comfort-zone-difficulty-band.md](research/2026-07-11-rl-comfort-zone-difficulty-band.md) | RL difficulty band |
| [research/2026-07-21-rl-algo-infra-deepresearch.md](research/2026-07-21-rl-algo-infra-deepresearch.md) | RL algo/infra research |
