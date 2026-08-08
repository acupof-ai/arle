# Streaming decode replaced every codepoint that spanned a token boundary

**Date:** 2026-08-08 · **Pod:** 8×H20, ThinkingCap-Qwen3.6-27B-FP8 serve on GPU 1,
plus a read-only scan of the `fulltrain11` rollout dumps

## Context

GSM8K in the capability eval returned 140 of 500 items unscored (28%) because
the answer text was U+FFFD replacement characters. The first hypothesis —
sampling temperature (#48) — is refuted by the harness source: both runners pass
`temperature=0.0` literally (`scripts/arle_capability_eval.py:442`, `:632`) and
`--seeds` only shuffles which items are drawn (`:602`), never the generation RNG.
`args.rs:2069` records that `b4b293f0c` already fixed greedy. So this is a
different defect on the greedy path.

## Phenomenon

Two distinct populations in the 3212 U+FFFD occurrences found across
`fulltrain11`'s 3473 dump bodies (318 bodies, 9.2%, in 34 distinct sample tags):

| Run length | Count | Byte context |
|---|---:|---|
| 1 | 1680 | a 3-byte character survives, a 4-byte one does not |
| 2 | 1428 | one 4-byte emoji → exactly two replacements |
| 3 | 58 | |
| 12–14 | 13 each | |
| 96 | 1 | the GSM8K answer block |

The run=1/2 bulk is decisive: in one checklist, item 1's ✅ (3 bytes,
`e2 9c 85`) and a `→` (3 bytes) are intact while item 2's 4-byte emoji is two
replacements. Length is not the trigger — the same prompt at `max_tokens`
64/256/1024/2048 puts the first replacement at the same absolute character
position, and 256/1024/2048 return byte-identical output.

## Root Cause

The three streaming paths decoded **each delta's tokens in isolation** —
`tok.decode(&delta.token_ids)` in `coordinator.rs` (completions, chat,
Anthropic). A codepoint's bytes can span a token boundary, and HF `tokenizers`
decodes lossily, so each orphaned byte became one U+FFFD. A 4-byte codepoint
split 2+2 therefore arrives as two replacements — exactly the observed
signature. Whole-sequence decode (the non-streaming paths and
`split_reasoning`) was never affected, which is why MMLU's 4.6% invalid rate
looks unrelated.

## Fix

`IncrementalDetokenizer` in `crates/infer-server/src/tokenizer.rs`: hold the
tokens whose decode ends in a replacement character, emit the valid prefix, and
retry when the bytes that finish the codepoint arrive. Bounded at 4 pending
tokens — a codepoint is at most 4 bytes and a token at least 1, so a longer run
is genuinely invalid output and is emitted rather than buffered. All three
streaming sites now push through it and flush the tail at `finish` (through the
tool/thinking pipeline, before it closes, so a held tail cannot be dropped).

Two tests, both non-vacuous: the split test first asserts that naive per-token
decode *does* produce a replacement for the chosen string, so it fails if the
vocab ever stops splitting it; the second asserts a run of continuation bytes is
still emitted, so a stream cannot stall holding them.

**Bench gate:** correctness fix in `crates/infer-server/src/`, throughput-neutral
by construction (one extra decode of ≤4 tokens only while a codepoint is
incomplete).

Verified on the pod at `ee6339fd7` (isolated build tree, serve on an idle GPU so
production's binary and endpoints were untouched): the assembled streamed text
carries **zero U+FFFD and is byte-identical to the non-streaming decode of the
same greedy generation**. The old binary on the same class of prompt put the
first replacement at character 236, immediately after a 4-byte emoji, with 97
occurrences in the body.

## The 96-run is a separate defect, on the generation side

Resolved at token level, and it is not a decode problem. The corrupted span is
**48 repetitions of one ordinary token, id 151353, piece `ª½`** — bytes `aa bd`,
both UTF-8 continuation bytes, valid only as the tail of a 4-byte codepoint
(`f0 9f` + `aa bd` = U+1FABD). The generation contains **zero byte-fallback
tokens** (0 of 546) and **zero tokens carrying a 4-byte lead byte**, so the
emoji's tail was emitted 48 times and its head never was. 48 tokens × 2 orphaned
bytes = the 96 replacements: the detokenizer is reporting faithfully, and no
amount of incremental buffering can repair bytes whose lead never arrives.

The token sequence entering the run:

```
'Ċ' '</think>' 'ĊĊ' 'Step' '-by' '-step' ' solution' ':'  then 48× 'ª½'
```

So greedy decode closes `</think>`, writes a heading, and then locks into a
repetition loop on a partial-codepoint fragment token until it stops
(`finish_reason=stop`, 546 tokens). Deterministic and reproducible on the same
item; a neighbouring item re-ran clean, so it is content-specific.

**Cause unknown.** This is an inference-correctness question, tracked separately
from this entry. The decisive next probe is a reference comparison: serve the
same checkpoint under sglang, same prompt, greedy — clean output there makes it
ours.

Also noted while in the same response: `completion_tokens_details.reasoning_tokens`
reports 0 while 1666 characters of thinking were returned.

## Rule

Streaming text output must be detokenized incrementally with a byte carry, never
per delta — a lossy decode turns every multi-byte character that lands on a
chunk boundary into replacement characters, and the damage is invisible in
ASCII-only tests. And when a metric reports mass invalid output, check the
harness source before attributing it to a model setting: the temperature
hypothesis here cost a round trip and was refuted by two literal `0.0`s.
