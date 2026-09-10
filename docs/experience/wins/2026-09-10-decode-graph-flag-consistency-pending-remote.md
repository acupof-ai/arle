# Decode-graph gate reads the DeepGEMM min-routes flag — pending remote validation

Status: **pending-remote** — code landed on `lane/model` (PR #273); the GPU
behavior check below has not run. Do not read the consistency claim as
verified.

## Context

The decode-graph capturability gate (`qwen35_decode_moe_graph_capturable`)
read the hardcoded const `QWEN35_DEEPGEMM_MIN_ROUTES = 1024`, while the
DeepGEMM dispatch it guards reads the runtime flag
`--qwen35-deepgemm-min-routes` (`runtime_flags::qwen35_deepgemm_min_routes()`,
default 1024). At default settings the two agreed; with the flag set to any
other value they forked — the dispatch would route a band to (or away from)
DeepGEMM while the graph gate still decided capture off the const, so a
decode step could be marked graph-capturable that the dispatch no longer ran
as a pure device-kernel sequence, or a capturable shape refused capture.

## What changed

- The gate moves to `infer_model::qwen35::moe_decode_graph_capturable(cfg,
  min_routes)`; the caller passes `qwen35_deepgemm_min_routes()` — the same
  value the dispatch reads. One decision, one input.
- `MoeConfig::device_route_eligible` (greedy top-k, no group limits) moves to
  `infer-moe`; the gate and the router's device path both call it.
- The const `QWEN35_DEEPGEMM_MIN_ROUTES` stays — `qwen35_load.rs` uses it as
  the warm-up route floor, unrelated to the graph gate.

This is a bug fix carried inside the step-3b host-geometry refactor
(PR #273): the refactor moved the gate, and the move switched its input.

## Remote gate (to run on the next GPU window)

1. Serve an MoE model with `--qwen35-deepgemm-min-routes 64` (below
   top_k × B_max): the decode graph must stay disabled (gate says
   not-capturable) while the dispatch takes the hand-kernel band. Check the
   startup graph-mode log line and decode wall-time.
2. Same with the flag at default 1024: graph capture behaves as before this
   change — no regression vs the current baseline.
3. `scripts/needle_gate.py` ×3 same-config vs the baseline envelope.

What a regression looks like: a decode step captured as a graph that is not a
pure device-kernel sequence (capture failure or a silent host fallback
mid-graph), or a capturable shape refused capture (decode ms/token regresses).

## Rule

A gate that guards a flag-tunable dispatch reads the same flag value the
dispatch reads — never a const with the same default.
