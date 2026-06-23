# arle serve: chat_template_kwargs / enable_thinking pass-through + thinking-token budget

## Context

Agentic-OPD's clean re-gate was blocked: the teacher is served via `arle serve`,
but `render_jinja` never received `enable_thinking`, so the Qwen chat template's
`{% if enable_thinking %}` branch was stuck at its template default. Forced
think-mode made the slow teacher run away / time out, and the eval mis-scored the
timeouts as abstention (the "+42pp teacher abstention" was 14/17 timeouts — the
agentic-OPD structural false-KILL, CLAUDE.md §0). The fix is the vLLM/SGLang-standard
`chat_template_kwargs` pass-through plus a token budget so think-on can't run away.

## What Worked

OpenAI-compatible request → template, end to end, backward-compatible:
- `ChatCompletionRequest.chat_template_kwargs: Option<Map<String,Value>>`
  (`#[serde(default)]`; vLLM/SGLang field name) → threaded through
  `render_chat_with_kwargs` → `render_jinja(..., Some(&kwargs))`, spread-merged
  **under** the reserved HF context keys (`messages`/`add_generation_prompt`/
  `bos_token`/`eos_token` always win) so `enable_thinking` (and any other kwarg)
  becomes a top-level template var. Absent kwargs → original `context!`,
  byte-identical render.
- Thinking budget: `arle serve --max-thinking-tokens N` (default 0 = off) →
  `EngineLoadConfig.max_thinking_tokens` → the chat handler clamps
  `max_tokens = max_tokens.min(N)` when the request enables thinking. Blunt total-output
  bound (not a `</think>`-span cap — that needs an infer-core stop condition), off
  by default, prevents the runaway/timeout.
- CPU unit test `jinja_enable_thinking_kwarg_toggles_branch`: on≠off render, and
  absent == explicit-false (backward-compat). 6 files: `infer-server/{schema,tokenizer,http}.rs`,
  `infer-api/loaded.rs`, `cli/{args,serve}.rs`.

Gates green: `cargo test -p infer-server` 34/34; clippy `infer-api`/`cli`/`infer-server`
(`cpu,no-cuda`) `-D warnings` clean; `cargo check -p infer-api --features cuda,no-cuda` clean.

## pending-remote

Functional/serve change (no perf delta). The payoff — the clean agentic-OPD
re-gate with a bounded think-on teacher (think-on irrelevance-abstention 0.65 vs
no-think 0.28) — runs on the GPU box: serve the teacher with
`--max-thinking-tokens <budget>` + clients passing `chat_template_kwargs:
{enable_thinking: true}`, then re-run the agentic gate (#7) + corrected OPD (#5).

## Rule

A chat-serving runtime MUST pass `chat_template_kwargs` (esp. `enable_thinking`)
from the request through to the Jinja render — omitting it silently forces the
template default, and a forced think-mode on a slow teacher manifests downstream
as timeouts mis-scored as a capability signal (not as a config bug). Spread the
kwargs UNDER the reserved HF context keys so they can't shadow `messages`/tokens.
