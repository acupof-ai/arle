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

**Not yet established, and the evidence does not yet support calling it a
numerics defect.** Greedy divergence between two runtimes is expected on its own
— different kernels reduce in different orders, this is an MoE checkpoint, and
once two logits are near-tied argmax flips for free. What the comparison
establishes is narrower and still serious: **on this input the checkpoint is
capable of the correct answer and our runtime does not produce it**, ending in
degenerate output instead.

The measurement that separates the two readings is a forced-prefix logprob
comparison at the divergence point (108 prompt + the 22 agreed tokens): a
near-tie where sglang's winner is ARLE's close second means the divergence is
benign and the defect is downstream degeneracy; a wide margin, or disagreeing
logprob values for the same token, means a numerics or kernel defect at this
shape. Pending.

## Fix

None yet. Consequences already actionable:

- **The 28% GSM8K invalid rate measures this defect, not model capability.** The
  full-1319 GSM8K point is on hold; MMLU stays the capability gate (0.8414 at
  concurrency 1, 0.8428 at 8, 4.6% invalid — short answers leave no room for the
  degeneracy).
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
