# DSv4 "broken forward" was a Qwen-vs-DeepSeek prompt-id confounder

## Context

The R6 clean-CUDA DSv4 multi-GPU (TP=8/EP=8) greedy-parity harness emitted
`clean_tokens=[16]` where the captured legacy oracle is
`[11111, 603, 671, 6102, 294, 8760, 344, …]`. This drove a long, escalating
chain of forward-bug hypotheses, all on real 8×H20 hardware:

- the FP8 native-grouped expert bypass is numerically broken;
- the shared expert is loaded whole on every rank and double-counted
  `world_size×` by the MoE `all_reduce_sum`;
- the DeepGEMM native bridge (`cuLibraryGetKernelCount → CUDA_ERROR_UNKNOWN` in
  multi-rank) is the correctness culprit.

Each was plausible from source. None was the bug.

## Root Cause

The parity harness (`crates/infer-cuda/examples/dsv4_parity.rs`) was cloned from
the Qwen greedy-parity harness and **carried the Qwen prompt token ids verbatim**:
`DEFAULT_PROMPT_IDS = "785,6722,315,9625,374"`, commented "Qwen tokenizer ids =
'The capital of France is'". DSv4-Flash uses the **DeepSeek** tokenizer
(vocab 128000), where those ids decode to garbage `" ar造成 thATE v"`. The
harness feeds raw ids (no tokenization), so the rewrite was scored on a
**different prompt** than the legacy oracle — which was produced by the legacy
binary fed the correctly-tokenized text. `'.'` (token 16) is a perfectly
plausible greedy continuation of the garbage prompt.

Decisive check (per the standing lesson "decode the actual tokens"): decoding the
ids on the pod tokenizer showed the prompt was garbage and the oracle was
coherent (`11111=' Paris'`). The correct DeepSeek ids for "The capital of France
is" are `671,6102,294,8760,344` — and those appear verbatim at oracle positions
3-7 (the base model echoing the prompt back). Re-running the **same rewrite
binary** (native bypass, untouched) on the correct ids produced
`prefill argmax token#1 = 11111` = the oracle. The forward was never broken.

Two distinct issues had been conflated: (1) the `[16]` "failure" = this prompt-id
confounder; (2) the DeepGEMM multi-rank bridge failure = a real but **separate**
infra issue that is *not* a correctness blocker (the native path is correct).

## Fix

`a882823b`: default prompt → `671,6102,294,8760,344` (DeepSeek ids, matching the
oracle); corrected the misleading "Qwen tokenizer ids" comments. The infra fixes
surfaced along the way were independently valid and mirrored to the repo
(NCCL file-rendezvous `e91cf0da`, MTP-tolerant config `7a7bd70d`, launcher
`INFER_CUDA_DEVICE=0` `3889ed5d`).

## Rule

A cross-implementation parity comparison must verify **both sides receive the
identical, correctly-tokenized input** before attributing any output difference
to a code bug. When porting a parity harness across model families, **re-tokenize
the prompt with the new model's tokenizer — never carry raw token ids** (a token
id is only meaningful relative to one tokenizer; vocab sizes alone — 151936 Qwen
vs 128000 DeepSeek — should trigger the check). The single cheapest disambiguator
is to **decode every id on both sides**: it would have ended this in one minute
instead of a multi-hour forward hunt. Reinforces "garbage output = config-suspect
first, code-suspect second" and "A/B must be same-binary, same-shell, same-prompt."
