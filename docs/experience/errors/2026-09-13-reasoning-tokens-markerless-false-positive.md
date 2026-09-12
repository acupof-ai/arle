# reasoning_tokens counted the whole completion when no think marker appeared — 2026-09-13

> Status: Fixed, CPU-only unit gates incl. a real-tokenizer fixture. No device
> behavior; no pod build. Finding #3 from the false-zero audit — a false
> non-zero, not a false zero.

## Context

`completion_tokens_details.reasoning_tokens` counts tokens the model spent in a
think block. The scanner is `count_reasoning_tokens`
(`crates/infer-server/src/coordinator.rs`), called for chat, n>1 chat,
multimodal chat, and Anthropic responses, plus an inline copy in the streaming
accumulator. A think block is the token interval between a checkpoint-specific
start id and end id (`OpenAiTokenizer::think_token_ids`).

## Phenomenon — concrete input and number

The scanner seeded its `in_thinking` state from the request's thinking *intent*
flag, not from any observed marker:

```rust
let mut in_thinking = start_in_thinking; // the enable_thinking flag
```

Request shape that triggers it:

1. A checkpoint whose template renders **ChatML** — the fallback renderer emits
   `<|im_start|>assistant\n` and never prefills a think-start token.
2. The server configured with a thinking budget, or the request passing
   `chat_template_kwargs.enable_thinking=true` — so the intent flag resolves
   true even though the rendered prompt carries no marker.
3. A tokenizer whose vocab contains the think ids (so the scanner is armed, not
   the safe `None` path). The checked-in Qwen3.5-0.8B tokenizer resolves
   `(248068, 248069)`.

Model output: ordinary content with no think marker, e.g. the tokenized text
`The sky is blue today.` — 6 tokens, neither id present.

Result before the fix: `reasoning_tokens = 6` with `completion_tokens = 6`.
Every content token was billed as reasoning. The visible
`reasoning_content` split on the same response disagrees: `split_reasoning`
requires a closed block and returns the whole body as content, so the response
carried zero reasoning text but a usage object claiming 100% reasoning.

A second trigger with the marker present but delayed: output
`Hi! <start> r1 r2 <end> answer` counted the two pre-marker content tokens as
reasoning too (4 instead of 2).

Demonstrated red-first: two unit probes returned `left: 10 / right: 0` and
`left: 4 / right: 2` against the old scanner; a fixture test using the real
Qwen3.5 tokenizer renders the ChatML prompt (tail
`<|im_start|>assistant\n`), resolves the think ids, and shows the 6-token
answer scanned as all reasoning.

## Root cause

The intent flag answers "did we ask for thinking", not "is the generation
inside a think block". The comment assumed every thinking-enabled render
prefills the start token into the prompt, which is true for the DSv4 builtin
and Jinja templates that honor the flag, and false for ChatML and for any
template that ignores `enable_thinking`. The text-parsing path
(`prompt_prefills_think` / `split_reasoning`) already keyed off rendered truth;
the token counter keyed off the flag.

## Fix

Seed from the rendered prompt: `reasoning_seed(prompt_tokens, think_ids)` is
true only when the prompt's last token is the think-start id (the token-level
equivalent of `prompt_prefills_think`). Otherwise the scanner starts closed and
counts only intervals it sees open with a start marker. The seed is computed
once per handler at the three encode sites and passed to every count call; the
streaming accumulator seeds the same way. Marker-less content now counts 0;
prefill and in-stream-marker cases count exactly the block interior. 37
infer-server tests pass, including a prefill case (2), an in-stream-marker case
(2, excluding pre-marker content), a no-vocab-ids case (0), and the fixture.

## Rule

Measure a token interval from the markers (or the rendered prompt's tail) that
delimit it, never from a request flag expressing intent. A presence/intent flag
can gate whether a field is reported; it cannot supply the measurement. This is
the same honest-set rule as the omitted-zero fixes: publish a number only when
something measured it.

## Not fixed here (separate bug)

Whether the `completion_tokens_details` object is present at all when thinking
was never requested is a separate presence-gating question on the same field.
This lane changes only the count's correctness, so the fix does not hide
behind a gate change.
