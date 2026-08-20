# Qwen speculative thinking budget and W8A16 lm_head routing — 2026-08-20

> Status: **Both confirmed on GPU.**

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

**2. W8A16 lm_head — confirmed.** Same binary, one prompt, the only difference
being whether `lm_head` is a tied BF16 tensor or a quantised untied one:

| checkpoint | `w8a16.marlin_tensorcore` | `w8a16.gemv` |
|---|---:|---:|
| `qwen35-08b-w8a16` (tied, BF16 lm_head) | 288 | 36 |
| `qwen35-08b-w8a16-lmhead` (untied, W8A16) | **291** | 36 |

The +3 is `lm_head`'s three single-row calls taking Marlin. Without this fix they
fall to the scalar arm, and the W8A16 repack has already freed `qweight` and
`qscales`, so the result is a missing-source error rather than a slow path. Both
arms answer `Paris`. `gemv` is unchanged at 36, so the new arm caught `lm_head`
rather than stealing another format's calls.

**Why this looked unvalidatable.** The first pass reported no checkpoint on the
box could reach the arm, which was true but not the reason. `scripts/quantize.py`
lists `lm_head.weight` in `W8A16_SKIP_ENDINGS` (:246), so the repo's own
quantiser does not produce this input by design — the path had never had one.
That skip's comment says the loader reads those tensors BF16-only and quantising
them crashes serve; it is stale. `load_output_head_quant_aware`
(`loader.rs:1497`) is quant-aware and goes through `load_matrix_quant_aware` +
`marlin_repack_dense`. The loader and the dispatch had already converged; only
the quantiser was still on the old coverage.

`scripts/make_w8a16_lmhead_checkpoint.py` builds the input: untie the embedding
and quantise it into an `lm_head` with the quantiser's own per-row
per-128-group symmetric INT8 (max abs error 9.07e-4 on `[248320, 1024]`).

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
