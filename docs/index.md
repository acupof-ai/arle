# Maintainer Doc Index

> Getting-started / install / HTTP API: [README.md](../README.md),
> [install.md](install.md), [onboarding.md](onboarding.md),
> [troubleshooting.md](troubleshooting.md), [http-api.md](http-api.md).
> This file is a pure index — no narrative state.

**Progress spine:** [`CHANGELOG.md`](../CHANGELOG.md)  
**Agent contract:** [`../AGENTS.md`](../AGENTS.md) (`CLAUDE.md` symlinks to it) ·
working method: [`agent-method.md`](agent-method.md)

The doc tree is reference docs + `plans/` + `research/` + experience wins/errors
+ CHANGELOG. `git log -- docs/` recovers anything removed.

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
| Qwen3.6-27B performance chain | [perf-qwen36-27b.md](perf-qwen36-27b.md) |
| CUDA operator organization | [plans/2026-08-20-cuda-operator-organization.md](plans/2026-08-20-cuda-operator-organization.md) |
| Bench + trace process | [bench-and-trace-spec.md](bench-and-trace-spec.md) |
| Rolling baselines | [baselines.md](baselines.md) |
| Design theses (depth track) | [plans/2026-09-02-design-theses.md](plans/2026-09-02-design-theses.md) |
| Design notes | [design/hybrid-prefix-cache.md](design/hybrid-prefix-cache.md) |
| Env / flags | [environment.md](environment.md) |
| Capability evals | [eval.md](eval.md) |
| OPD capability curve | [opd-capability-curve.md](opd-capability-curve.md) |

---

## Positioning (fact)

- Front door (README / landing, 2026-09-02): the local inference server for
  coding agents — Anthropic `/v1/messages` + OpenAI `/v1/chat/completions`,
  KV cache that survives across turns, one binary on Apple Silicon and NVIDIA.
  H20 / DSv4 / NVFP4 detail lives in `docs/baselines.md`, not the first screen.
- Runtime-first: `infer-plan` → `infer-seam` → `infer-core` →
  `infer-cuda`/`infer-metal` → `infer-server`/`infer-api`. Monolith `infer/` deleted.
- `arle` = CLI front door; `infer-api` (`LoadedInferenceEngine`) = programmatic front door.
- Train = **OPD-only** substrate on the same runtime (pretrain/SFT/GRPO multi-turn retired).
- Metal canonical model: `mlx-community/Qwen3.6-35B-A3B-4bit`.

---

## Experience

| Path | Role |
| --- | --- |
| [experience/wins/](experience/wins/) | licensed wins / benches |
| [experience/errors/](experience/errors/) | kills / regressions |
| [experience/wins/TEMPLATE-bench.md](experience/wins/TEMPLATE-bench.md) | bench entry skeleton |
