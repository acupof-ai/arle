# R6a CUDA port stalled: clean extraction closure is ~25.6k LOC, not Metal-sized

## Context

The Metal backend (R3a–R3e) ported cleanly into `infer-metal` (~2k LOC, no legacy
`infer` dependency) because the MLX forward path is relatively self-contained
(config + loader + mlx wrapper + qwen35_compiled session calls). The R6 plan assumed
CUDA would follow the same shape. R6a (single-slot greedy Qwen3 on V100) was briefed
as "port the minimal Qwen3 CUDA forward into infer-cuda, no legacy dep."

## Root Cause

The real Qwen3 CUDA forward is **deeply coupled**: the dependency closure spans
`infer/src/model/qwen3/* + ops/* + oplib + weight_loader + model_source + quant +
gguf + tensor_parallel + dispatch_policy + the scheduler/model trait glue` —
**~25.6k LOC** measured. The three options all conflict with a constraint:

- Depend on legacy `infer` → violates the clean-crate-graph (the whole point of the
  rewrite; reintroduces the coupling being removed).
- `#[path=]`-include `infer/src` → same coupling, worse.
- Re-derive a tiny forward from kernels → violates "port tested numerics" (would
  re-introduce solved bugs — exactly what the rewrite plan forbids).

So a clean CUDA port is a **large multi-tranche extraction**, an order of magnitude
bigger than Metal. Codex correctly stopped and reported (no fake commit, clean tree).

## Fix / Decision

CUDA clean extraction is a multi-session tranche. The AI-PC **primary** backend
(Metal) is already done + correctness-verified (4 configs incl. Qwen3.6 MoE) +
benched. CUDA (V100/H20) is the server/consumer-NVIDIA backend — not the AI-PC
critical path. Approach chosen with ckl (see the conversation fork); options were:
incremental clean extraction, a temporary legacy bridge, or re-scope CUDA as a
deferred clean tranche.

## Rule

Estimate a backend's clean-extraction **dependency closure** before scoping its port
as "Metal-sized." Self-contained kernel bridges (MLX) port cheaply; a backend whose
forward reaches into the shared model/ops/loader/quant/TP stack (CUDA here, 25.6k
LOC) is a multi-tranche extraction. Measure the closure (Codex did: grep the import
graph) before briefing "port the minimal forward." Faithful reporting beat a fake
commit here — Codex stopping with the LOC number was the right call.
