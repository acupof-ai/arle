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

H20 pod, build `lp202` (a8150bc6b), ThinkingCap-27B-FP8, DSpark serve
(`--spec-type dspark`), 2026-08-14. 13/13 e2e checks PASS:

- Shape: `tokens` / `token_logprobs` / `top_logprobs` / `text_offset` equal
  length; greedy `top_logprobs[i]` max equals `token_logprobs[i]` at every
  position (8/8); all values finite. Chat entry shape + top-1 identity PASS.
- Rejects: `logprobs=9` → 400, `stream=true`+`logprobs` → 400.
- Matched A/B c=1 greedy 64 tok ×6: median 0.589 s without logprobs, 1.045 s
  with (+77.5%) — the expected cost of the per-request spec veto on a DSpark
  serve, paid only by requests that ask; the no-logprobs arm keeps spec.

## Rule

A per-request feature that needs host logits joins the existing
`is_raw_argmax()` veto (compile-enforced destructure) instead of adding a new
gate; captures ride existing `&mut` step state, not new signatures.
