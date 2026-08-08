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
incomplete). Pod verification of the repro is `pending-remote`.

## Open — the 96-run is not this bug

The GSM8K case is a whole-sequence decode: `split_reasoning` splits the *string*
after one decode of all tokens, so a token-boundary split cannot produce it. In
that response the thinking block is clean for 1666 characters and the answer
block is 96 of 121 characters replaced, `finish_reason=stop`,
`completion_tokens=573`. The model produced a long run of bytes that never form
valid UTF-8. **Cause unknown** — the next step is to read the generated token ids
for that span and check whether they are byte-fallback tokens, which would make
it a generation defect rather than a decode one. Also noted while there:
`completion_tokens_details.reasoning_tokens` reports 0 while 1666 characters of
thinking were returned.

## Rule

Streaming text output must be detokenized incrementally with a byte carry, never
per delta — a lossy decode turns every multi-byte character that lands on a
chunk boundary into replacement characters, and the damage is invisible in
ASCII-only tests. And when a metric reports mass invalid output, check the
harness source before attributing it to a model setting: the temperature
hypothesis here cost a round trip and was refuted by two literal `0.0`s.
