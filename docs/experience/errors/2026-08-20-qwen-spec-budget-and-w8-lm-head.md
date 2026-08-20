# Qwen speculative thinking budget and W8A16 lm_head routing — 2026-08-20

> Status: **Thinking budget confirmed on GPU. W8A16 lm_head routing still
> unvalidated — no checkpoint on the box can reach it (see Result).**

## Context

Two Qwen dispatch gaps survived because format and sampling decisions were split
across parallel entry points:

- MTP/DSpark accepted chains cannot apply forced tokens, grammar masks, logit
  bias, penalties, or a strict per-token thinking budget.
- W8A16 Marlin owned batched GEMM, while the single-row `lm_head` dispatch only
  forwarded FP4 and FP8 to their Marlin layouts.

## Root Cause

Qwen speculative routing vetoed penalties only. A thinking budget becomes
`force_next_token = </think>` after the limit, but the speculative verifier
continues through its raw argmax or filtered-probability path. A multi-token
accepted chain can also cross the limit before the engine observes its tokens.

W8A16 repack frees `qweight` and `qscales`. `gemm_batch` consumed the retained
Marlin buffers, while `gemv` fell through to the freed scalar buffers for a
quantized `lm_head`.

## Fix

- One Qwen compatibility predicate now sends every unsupported token rewrite,
  including any configured thinking budget, through one-token plain decode.
- W8A16 batch and single-row dispatch now share one raw-pointer Marlin launcher.

## Result

`834a87aed` on 1xH20, CUDA release, TP=1, FP8 KV, `--spec-type mtp
--mtp-draft-tokens 3 --max-running-requests 4`, greedy, 600 output-token cap.

**1. Thinking budget — confirmed, with a control.** Two servers on the same
binary, `--max-thinking-tokens 8` against `0` (unlimited), same three prompts.
The budget arm alone proves nothing: a prompt that says "think briefly" can
produce 8 reasoning tokens on its own. The unlimited arm is what makes it
evidence.

| prompt | budget 8 | unlimited |
|---|---:|---:|
| `What is 17*23? Think briefly.` | **8** | 439 |
| 17-sheep multi-step word problem | **8** | 600 (hit the output cap) |
| `Prove sqrt(2) is irrational, show every step` | **8** | 600 (hit the output cap) |

`reasoning_tokens` from `usage.completion_tokens_details`. The ceiling holds on
all three, and generation continues correctly past the forced `</think>` —
`17 * 23 = 391`, then a step-by-step answer and a proof. `SERVER_ERRORS=0`.

**2. W8A16 lm_head — NOT validated, and not validatable on this box.** Both
W8A16 checkpoints under `/data00` (`qwen35-08b-w8a16`, `qwen35-08b-w8a16b`) set
`tie_word_embeddings: true`, and the tied `embed_tokens.weight` is BF16
`[248320, 1024]`. The `lm_head` a serve builds from them is therefore never
`WeightFormat::W8A16`, so the new single-row arm cannot be reached.

What the smoke run does show: the model loads and generates correctly (`The
capital of France is` -> ` Paris.`), `SERVER_ERRORS=0`, and the batched
projections still take Marlin — `cuda.w8a16.marlin_tensorcore` 288. That covers
the refactor, not the fix.

`cuda.w8a16.gemv` 36 confirms the arm is being *called* at m=1 and returning
false: those 36 single-row calls are weights `repack_for_marlin_w8a16` declined,
which is the fall-through case, not the case this change exists for. Note also
that `MARLIN_W8A16_HITS` increments inside `marlin_w8a16_gemm_raw`, so the batch
and single-row lanes share one counter and a future run cannot separate them
from `/v1/stats` alone.

Parameters item 2 stays open. Closing it needs a Qwen3.5/3.6 checkpoint with
`lm_head.weight` quantized at group size 128 and untied.

## Environment

There is no Mac typecheck lane for this: the local `cuda,no-cuda` test binary
cannot link because the CUDA C symbols are intentionally absent, so the pod
build is the typecheck.

- Commit `834a87aed`, on top of `a5df06c7c`. Binary sha256 (first 16)
  `bcea08abfe9a87c2`; an earlier run of the same HEAD hashed
  `2e99884cdba05a60` — Rust release links are not bit-reproducible here, so the
  hash identifies a build, not a source.
- 1xH20 (sm_90), GPU 0/1, TP=1, driver as installed on the shared pod.
- Models: `/data00/Qwen3.6-27B-FP8`, `/data00/qwen35-08b-w8a16`.
- The FP8 checkpoint is 128x128 block-scaled, so it has no Marlin layout:
  `cuda.qwen.fp8_pack_deepgemm` 512 and `cuda.qwen.fp8_gemv` 512, with
  `fp8_marlin_tensorcore` absent. Pre-existing for that checkpoint, unchanged by
  this commit.

## Rule

Represent speculative compatibility once at the executor boundary. Route every
consumer of a source-free weight through the retained layout.
