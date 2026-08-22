# Chat requests with thinking off defaulted to max_tokens=16 — CUDA, 2026-08-22

> Status: Fixed (2907785d5), verified

## Context

Terminal-Bench eval of ThinkingCap-Qwen3.6-27B-NVFP4 (TP2) returned 0/4, all
tasks failing as `unknown_agent_error`. The terminus agent sends chat
completions without `max_tokens` and without a thinking flag.

## Root cause

`crates/infer-server/src/coordinator.rs` had three default sites for
`max_tokens`; the chat and Anthropic paths fell back to `16` when the request
carried no thinking budget. ThinkingCap's template emits reasoning before the
answer, so the whole 16-token budget went to reasoning and the response ended
with `finish_reason=length` before any command JSON. The raw-completions path
already used `THINK_CONTENT_HEADROOM` (4096); the chat paths did not.

A second defect hid behind it: `defaults_thinking_on()` only returns true for
`BuiltinDeepseekV4`, so ThinkingCap's Jinja `enable_thinking` default was not
detected and the think-budget branch never engaged.

## Fix

`2907785d5` — collapsed the three sites to one rule:
`max_tokens = sampling.max_new_tokens.unwrap_or_else(|| think_budget +
THINK_CONTENT_HEADROOM)` (4096 when no budget). The `else { 16 }` fallback is
gone.

## Verification

MMLU 5-shot (50 questions, seed 0, TP2, FP8 KV, spec off): accuracy 0.816
(40/49, 1 invalid, 65.6 s) — completions now run to a real answer instead of
truncating at 16 tokens. Needle ladder was unaffected (it sets
NEEDLE_MAX_TOKENS explicitly).

## Rule

A default generation budget is a generation budget, not a protocol constant —
the fallback when the client omits `max_tokens` must be the same headroom the
raw path uses, and it must never depend on whether the model's thinking flag
was auto-detected.
