# 2026-04-15 · Metal Backend Execution Checklist

## Goal

Turn the current Metal backend from "usable serial beta" into a production-grade
Apple Silicon serving path with explicit milestones, acceptance criteria, and a
clear execution order.

This checklist is the execution companion to:

- [../projects/mlx-backend-roadmap.md](../projects/mlx-backend-roadmap.md)
- [2026-04-15-metal-backend-acceptance-plan.md](2026-04-15-metal-backend-acceptance-plan.md)

## P0 · Serving Floor

- [x] `M0.1` Make `metal_serve` local-only by default and add optional Bearer auth.
  Exit:
  `metal_serve` binds `127.0.0.1` by default; `--api-key` or `AGENT_INFER_API_KEY`
  protects `/v1/*`; `/metrics` stays scrapeable without auth.
- [ ] `M0.2` Replace serial `BackendRuntimeHandle` serving with a real `MetalScheduler`.
  Exit:
  live HTTP traffic no longer goes through the single-request runtime; scheduler
  owns admission, step selection, and request cleanup.
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

## Execution Order

1. Ship `M0.1` immediately to reduce accidental exposure risk.
2. Land `M0.2` before spending more effort on single-request-only optimizations.
3. Fold `M0.3` and `M0.4` into the same scheduler integration track so reuse can
   be measured as soon as it exists.
4. Start `M1` only after the live Metal serving path has real batching and reuse.
