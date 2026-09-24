# Maintainer Doc Index

> Getting-started / install / HTTP API: [README.md](../README.md),
> [install.md](install.md), [onboarding.md](onboarding.md),
> [troubleshooting.md](troubleshooting.md), [http-api.md](http-api.md).
> This file is a pure index — no narrative state.

**Release notes:** [`CHANGELOG.md`](../CHANGELOG.md)  
**Agent contract:** [`../AGENTS.md`](../AGENTS.md) (`CLAUDE.md` symlinks to it) ·
working method: [`agent-method.md`](agent-method.md)

The doc tree is reference docs + `design/` + `research/`.

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
| Qwen3.6-27B performance chain | [perf-qwen36-27b.md](perf-qwen36-27b.md) · [per-stage timing](perf-stages-27b.md) |
| Bench + trace process | [bench-and-trace-spec.md](bench-and-trace-spec.md) |
| Performance and correctness gates | [perf-and-correctness-gates.md](perf-and-correctness-gates.md) |
| Rolling baselines | [baselines.md](baselines.md) |
| Release checklist | [release-checklist.md](release-checklist.md) |
| Unsafe code audit | [unsafe-audit.md](unsafe-audit.md) |
| Design notes | [1 prefix cache](design/hybrid-prefix-cache.md) · [2 speculative decoding](design/speculative-decoding.md) · [3 backend seam](design/seam-cost-contract.md) · [4 memory](design/memory-is-the-product.md) · [5 matched measurement](design/matched-measurement.md) · [what-breaks](design/what-breaks.md) |
| Research notes | [`research/`](research/) |
| Env / flags | [environment.md](environment.md) |

---

## Positioning (fact)

- Front door (README / landing): the local inference server for
  coding agents — Anthropic `/v1/messages` + OpenAI `/v1/chat/completions`,
  KV cache that survives across turns, one binary on Apple Silicon and NVIDIA.
  H20 / DSv4 / NVFP4 detail lives in `docs/baselines.md`, not the first screen.
- Runtime-first: `infer-plan` → `infer-seam` → `infer-core` →
  `infer-cuda`/`infer-metal` → `infer-server`/`infer-api`.
- `arle` = CLI front door; `infer-api` (`LoadedInferenceEngine`) = programmatic front door.
- OPD training lives in the separate [arle-opd](https://github.com/acupof-ai/arle-opd)
  repository; this runtime serves as its teacher (`/v1/raw_logits`, `--lora-adapters`).
- Metal canonical model: `mlx-community/Qwen3.6-35B-A3B-4bit`.
