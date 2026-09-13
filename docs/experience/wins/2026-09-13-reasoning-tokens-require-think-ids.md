# reasoning_tokens breakdown emitted only when the tokenizer carries think ids

Date: 2026-09-13. Scope: `crates/infer-server` chat/anthropic/streaming
response assembly. Host-only, no GPU.

## Context

The false-zero audit of serving fields found that
`usage.completion_tokens_details.reasoning_tokens` was published for every
chat response with a structural constant 0 whenever the request did not
ask for thinking. For a tokenizer that carries no think markers at all
(non-reasoning checkpoints), reasoning tokens are not a measurable
quantity: the published 0 read as a measured zero.

The discriminator is tokenizer capability, not request intent: the chat
template's `enable_thinking` kwarg says what the caller wants; the
tokenizer's think start/end ids say whether the quantity exists. A Qwen
vocab with think ids that produces an answer outside the think block
yields a genuine measured 0, which must stay present.

## What worked

- `ChatCompletionResponse::from_parts` takes `reasoning_tokens:
  Option<usize>`; callers pass `state.think_token_ids.map(|_| count)`.
  `None` omits the whole breakdown; `Some(0)` serializes a present zero.
- The streaming usage chunks gate on `think_ids.is_some()` instead of the
  `thinking` intent flag, matching the non-streaming paths.
- Wire-visible effect on the canonical models: Qwen and DSv4 tokenizers
  always carry think ids, so a thinking-**off** request on them previously
  emitted no `completion_tokens_details` (gated on intent) and now emits
  `completion_tokens_details: {reasoning_tokens: 0}` — the quantity is
  measurable and zero was measured. Additive on the wire. Non-reasoning
  vocabularies gain nothing and remain key-absent.
- One end-to-end-shaped unit test asserts both cases through
  `from_parts`: no think ids + thinking requested → key absent; think ids
  + measured zero → key present with 0.

## Rule

Publish an unmeasurable quantity as absent, and gate that choice on
whether the measurement is defined for the model, not on whether this
request exercised it. Capability comes from the tokenizer/config; intent
comes from the request.
