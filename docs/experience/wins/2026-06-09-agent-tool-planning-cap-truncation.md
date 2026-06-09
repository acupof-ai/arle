# Agent tool-planning cap 256→49984 — kill pre-tool reply truncation

## Context

Metal `arle` agent CLI (Qwen3.5-9B-MLX-4bit) misbehaved on a plain
"写个矩阵乘法的优化算法并测试一下性能" prompt:

- Turn 1: `out 256 tok`, `finish_reason=Length` — a legitimately long
  text answer (code + explanation) chopped mid-code-block.
- "用你的工具测试下": 0 tool calls, `finish_reason=Length`,
  "(agent finished without a visible reply)".
- "继续": deterministic recovery scraped a 23-char decorative
  `print(f"\n{'='*50}")` fragment and ran it as the "test" → sandbox stderr.

## Root Cause

`crates/agent/src/lib.rs:536` clamps every generation to
`TOOL_PLANNING_MAX_TOKENS` (=256) until the first tool call lands
(`tool_calls_executed == 0 && !tools.is_empty()`). The banner's
`Max tokens: 262144` is `settings.max_tokens`, which only unlocks AFTER
a tool fires. The cap assumes the pre-tool phase is a short "decide to
call a tool" step — false for prompts whose correct response is a long
text answer. 256 tokens isn't enough room to write a code block AND emit
the trailing `<tool_call>`, so the turn dies on `Length`, the
`<tool_call>` never appears, and the loop falls onto the greedy
`extract_balanced_call(text, "print(")` fallback that grabs the first
cosmetic `print(...)`.

## What Worked

Raised `TOOL_PLANNING_MAX_TOKENS` 256 → **49_984** (≈50k, 64-aligned =
781·64). Gives the pre-tool turn room to finish a code block + tool call
instead of truncating at 256. `first_agent_tool_planning_turn_is_capped`
bumped its `AgentSettings.max_tokens` 4096 → 262144 so `min(setting, cap)`
still resolves to the cap and the test exercises the clamp.

Validation: `cargo test -p agent` 33/33 green. guidellm is the wrong
instrument here — it drives the serving path, not the agent tool-planning
loop; this is an agent-loop behavior fix, not a throughput change. The
ground truth is the transcript repro above + the targeted unit test.

## Rule

The pre-tool generation cap is a **planning** budget, not a global one.
Set it high enough to contain a full code-block-plus-tool-call draft
(≥ a few k tokens), or gate the 256-style clamp on the draft actually
*looking like* a tool attempt — never let it truncate a legitimate long
text answer and then scrape a `print(` fragment as a phantom tool call.
