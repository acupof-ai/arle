# 2026-04-15 · Metal Backend Execution Checklist

## Goal

Turn the current Metal backend from "usable serial beta" into a production-grade
Apple Silicon serving path with explicit milestones, acceptance criteria, and a
clear execution order.

This checklist is the execution companion to:

- [../projects/mlx-backend-roadmap.md](../projects/mlx-backend-roadmap.md)
- [2026-04-15-metal-backend-acceptance-plan.md](2026-04-15-metal-backend-acceptance-plan.md)
- [../reviews/2026-04-15-metal-ecosystem-route-correction.md](../reviews/2026-04-15-metal-ecosystem-route-correction.md)

## P0 · Serving Floor

- [x] `M0.1` Make `metal_serve` local-only by default and add optional Bearer auth.
  Exit:
  `metal_serve` binds `127.0.0.1` by default; `--api-key` or `AGENT_INFER_API_KEY`
  protects `/v1/*`; `/metrics` stays scrapeable without auth.
- [ ] `M0.2` Replace serial `BackendRuntimeHandle` serving with a real `MetalScheduler`.
  Exit:
  live HTTP traffic no longer goes through the single-request runtime; scheduler
  owns admission, step selection, and request cleanup.
  Status:
  `2026-04-15`: `M0.2a` local request-state layer landed for Qwen3/Qwen3.5;
  `M0.2b` rewired standard `metal_serve` onto a live scheduler runtime and
  cut `512/256 C=4` TTFT p50 from `7994ms` to `1826ms`, but aggregate
  throughput (`58.7 tok/s`) still trailed the old serial reference (`65.8 tok/s`).
  `M0.2c` then added same-length Qwen3 decode batching and improved focused
  Qwen3 server throughput from `23.30 -> 25.39 tok/s` at `C=4`, but the
  milestone stayed open. `M0.2d` then added same-length Qwen3.5 batched decode
  through the compiled MLX bridge; direct `128/128` improved from `82.0 -> 84.2`
  generation TPS, but the quick HTTP sweep stayed flat (`512/256 C=4`
  `66.4 -> 66.2 tok/s`), so the milestone is still open because variable-length
  decode and per-step batch-state rebuild cost still dominate the serving exit.
- [ ] `M0.3` Wire Metal prefix cache + KV pool into the live scheduler path.
  Exit:
  shared-prefix requests skip matched prefill in the serving path, not only in
  the single-request fallback path.
- [ ] `M0.4` Expose Metal memory and reuse observability.
  Exit:
  `/metrics` and `/v1/stats` surface at least `prefix_hit_rate`, `kv_util`,
  `active_memory`, `peak_memory`, and queue depth.

## P1 · API And DX

- [x] `M1.1` Promote Metal experimental env toggles to documented CLI flags.
  Exit:
  no user-facing Metal serving behavior requires hidden env vars.
- [ ] `M1.2` Add `/v1/models` and `/v1/responses`.
  Exit:
  standard OpenAI SDK flows work without compatibility shims for model discovery
  and the Responses API.
  Status:
  `2026-04-15`: `/v1/models` shipped and `/v1/responses` non-streaming subset
  shipped; streaming parity still pending.
- [ ] `M1.3` Add structured output support.
  Exit:
  `response_format` with JSON-schema constrained decoding is supported for chat
  requests; tool-call reliability no longer depends on post-hoc parsing only.
- [ ] `M1.4` Add a one-command Apple Silicon path.
  Exit:
  documented install/run flow no longer assumes Cargo feature knowledge for a
  first-time Metal user.

## P2 · Product Breadth

- [ ] `M2.1` Generalize DFlash from `Qwen3`-only into a Metal speculative decode framework.
  Exit:
  speculative decode lives behind a backend-level contract instead of a single
  hard-coded model family path.
- [ ] `M2.2` Expand Metal model coverage deliberately.
  Exit:
  GLM4 reaches supported status; Gemma text path has an explicit support decision.
- [ ] `M2.3` Add capture/profiling ergonomics.
  Exit:
  MLX / Metal capture can be enabled from the CLI and documented for regression work.
- [ ] `M2.4` Make Metal KV quantization an explicit go / no-go decision instead of an implied gap.
  Exit:
  docs and roadmap explicitly state that Metal serving does not currently ship
  quantized KV cache; a follow-up implementation only starts when long-context /
  high-concurrency Apple workloads justify it after `M0.2/M0.3/M0.4`.

## Execution Order

1. Ship `M0.1` immediately to reduce accidental exposure risk.
2. Land `M0.2` before spending more effort on single-request-only optimizations.
3. Fold `M0.3` and `M0.4` into the same scheduler integration track so reuse can
   be measured as soon as it exists.
4. Start `M1` only after the live Metal serving path has real batching and reuse.
5. Treat `metal_bench` as a sanity check, not as the serving milestone exit.
   `metal_serve` must pass an HTTP throughput sweep once `M0.2` is in.

## Route Guardrails

These guardrails come from the 2026-04-15 ecosystem review:

- Do not count direct `metal_bench` wins as serving progress.
- Do not start another single-request-only tuning wave before `M0.2/M0.3/M0.4`
  unless it fixes correctness, memory safety, or a build blocker.
- Use `mlx-lm` as the direct execution reference and `vllm-metal` /
  Docker Model Runner as the serving reference.
- Treat install DX as competitive scope, not polish work. One-command Apple
  setup belongs on the product path, not the backlog tail.
