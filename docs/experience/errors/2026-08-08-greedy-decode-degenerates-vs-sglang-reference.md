# ARLE greedy decode degenerates where sglang answers correctly — same checkpoint, same prompt ids

**Date:** 2026-08-08 · **Pod:** 8×H20, ThinkingCap-Qwen3.6-27B-FP8,
ARLE `ee6339fd7` on GPU 2 against sglang 0.5.13.post2 on GPU 3

## Context

GSM8K in the capability eval returned 140 of 500 items unscored (28%). The text
was U+FFFD, which first looked like the streaming detokenizer defect
([entry](2026-08-08-streaming-detokenizer-splits-multibyte-codepoints.md)) and is
not: that response came from a whole-sequence decode, and at token level the
corrupted span is 48 repetitions of token 151353 (`aa bd`, bare UTF-8
continuation bytes) with zero byte-fallback tokens in the generation.

## Phenomenon

Both runtimes were fed the **identical 108 prompt token ids** — the confound
that the first comparison attempt had (raw vs templated prompt) was removed
before these numbers were taken. sglang loaded the checkpoint natively through
its own `qwen3_5.py` at FP8 `e4m3`, `weight_block_size [128,128]`, with no
precision substitution and no `trust_remote_code` fallback, so it is a valid
reference.

| | ARLE | sglang |
|---|---|---|
| prompt token ids | 108 | 108, identical |
| generated tokens | 546 | 644 |
| count of token 151353 | 48, contiguous to the end from position 498 | 0 |
| U+FFFD in text | 96 | 0 |
| finish | `end_turn` | `stop` |
| answer | none | `#### 410`, matches gold |

The first **22 generated token ids agree exactly**, then:

```
ARLE   gen[22:30] = [5289,  2702, 5821, 60766,  539, 43313, 1155, 321]
sglang gen[22:30] = [ 328, 16694, 2702, 5821, 60766,    1,  539, 43313]
```

From there the sequences drift; ARLE ends in the fragment loop, sglang completes
the arithmetic.

Position is not the variable: the divergence is at token 22 while the fragment
run starts at 498, and three separate cases garble at character 25 of a 573-token
generation, character 236 of a 121-token generation, and not at all at 618
tokens. Content-dependent.

## Root Cause

Localized to the logits, not to sampling or decoding luck. Measured on the
identical 130-token prefix (108 prompt + the 22 agreed generated tokens,
round-trip verified as the same ids on both runtimes) with the existing
`INFER_DSV4_DUMP_TOPK_POSITIONS` dump and `ARLE_PROBE_JSONL` +
`ARLE_PROBE_TOKEN_ENTROPY`:

| | ARLE | sglang |
|---|---|---|
| p(5289) `' average'` | **0.350, rank 1** | 0.081, rank 3 |
| p(328) `' "'` | below 0.350, rank 2 | **0.768, rank 1** |
| top-1/top-2 margin | 1.6875 logits | 1.625 nats |
| entropy at the position | **7.31 nats** | ~1 nat, implied by p1 = 0.768 |
| logits vector | 248320 finite, **nan 0, pos_inf 0, neg_inf 0** | |

ARLE's p1 comes from the probe's own `nll` (1.050838 → p = 0.350), which is
already on the probability scale, so the raw-logits-versus-logprobs mismatch does
not affect the comparison.

Two things hold at once. The ranking is **inverted** — ARLE orders
`5289 > 328 > 348`, sglang orders `328 > 348 > 5289` — and the distribution is
**broader**, top-1 holding 35% against 77%. Neither is a near-tie flipped by
reduction order: both runtimes are internally confident with comparable margins.
The entropy figure alone does not establish broadening on a 248k vocab; it is the
pairing with p1 = 0.350 that does.

Entropy is elevated from the first generated token, not flat-then-broken, so the
perturbation is systemic and token 22 is only where it first changed an argmax.

**The remaining question is which path.** The whole divergence happens within 130
tokens, so no long-context, paged-KV, or window mechanism is in play. The split
is prefill forward versus incremental decode: send the same 130 ids as a prompt
with `max_tokens=1` and compare that prefill top-8 against the decode path's.
Prefill agreeing with sglang puts the defect in the decode path; prefill
reproducing `5289` at p 0.35 puts it in the forward (quantization, MoE routing, a
GEMM at this shape). Pending.

## Fix

None yet. Consequences already actionable:

- **The GSM8K invalid rate measures this defect, not model capability.** Of 140
  invalid items in seed 0, ~111 (79%, = **22% of all 500**) end in a repetition
  run: 90 have empty content because the loop fires inside the thinking block
  (79 of them finish with `stop` at a median 304 tokens, so not budget
  exhaustion), 21 carry a replacement run in the content. The loop appears with
  valid tokens too (`' Rory made'` ×8, `私` ×25), so the garble and the loop are
  independent symptoms of one defect. The gold answer is present in the thinking
  text for 44 of the 90 empty items — the model often reaches the right number
  and degenerates before reporting it.
- **There is no extractor bug**: zero invalid items contain an unparseable
  `####`. An earlier decomposition claiming 82 such items was measured on
  `/v1/completions` while the harness uses the chat endpoint, and was retracted.
- The full-1319 GSM8K point is on hold; MMLU stays the capability gate (0.843 /
  0.882 / 0.861 across seeds 0–2 at concurrency 8, invalid 23/26/26). Note the
  **3.9 pp seed-to-seed spread**: a per-round adapter delta must clear that or be
  read as a per-question paired comparison on the same seeds.
- **The `fulltrain11` production run continues.** `rejection-ce` trains only on
  pytest-verified passes, so a degenerate rollout fails its tests and is filtered.
  The defect costs pass yield and rollout budget, not training-data correctness.

## Rule

A reference runtime is the cheapest way to convert "our output looks wrong" into
"our output *is* wrong", but it only works if the comparison is fed identical
token ids and the reference loads the checkpoint natively — a templated-vs-raw
prompt difference or a precision substitution turns the reference into another
unvalidated path. And compare *token ids*, not text: text comparison hides the
distinction between a different token and a different rendering of the same one.
