# OpenAI logprobs on /v1/completions (CUDA Qwen3.5/3.6)

## Context

Issue #202 part 2 needs the top-2 logprobs of the first sampled position from
`/v1/completions`. The server accepted `logprobs` and returned `null`; the
device argmax fast path never produced alternatives.

## What Worked

- `SamplingParams::top_logprobs: Option<usize>` (cap 8, 400 above) rides the
  existing `is_raw_argmax()` veto: a logprobs request takes the host sampling
  path, where `infer_plan::sampled_top_logprobs` computes entry 0 = sampled
  token's full-softmax logprob + top-N alternatives from the same host copy
  the sampler already reads.
- Capture channel: `Qwen35Workspace::top_logprobs` (single-row paths, drained
  with `mem::take` by the executor right after the sampling call) and a
  widened `BatchSample` return for the two batched decode kernels — no
  signature ripple through the forward chain.
- Spec decode (MTP + DSpark) is vetoed per request when `top_logprobs` is set
  (the verify commits tokens without full per-position distributions); the row
  stays on the warm path, which captures.
- Wire: `SlotToken::top_logprobs` → `StreamItem::Token` /
  `RelayCompletionDelta::top_logprobs` (serde-default, old peers interoperate)
  → coordinator builds the OpenAI `tokens` / `token_logprobs` /
  `top_logprobs` / `text_offset` object with real detokenization. Missing
  capture with `logprobs` requested → 501 (Metal / DSv4 / legacy-Qwen paths),
  never a silent `null`. `stream=true` + `logprobs` → 400 (not wired yet).

## Bench

pending-remote — needs the H20 pod (issue #202 serve). Plan: matched A/B
c=1 greedy decode with and without `logprobs=2` on ThinkingCap-27B-FP8
(the veto only affects requests that ask; the no-logprobs path is untouched —
control expected to be a wash), plus a curl shape check: `top_logprobs[0]`
top-1 equals `token_logprobs[0]`.

## Rule

A per-request feature that needs host logits joins the existing
`is_raw_argmax()` veto (compile-enforced destructure) instead of adding a new
gate; captures ride existing `&mut` step state, not new signatures.
