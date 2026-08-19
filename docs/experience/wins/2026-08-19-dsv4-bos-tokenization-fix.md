# DSv4 BOS tokenization fix — thinking mode degeneration — CUDA, 2026-08-19

> Status: Shipped

## Context

DeepSeek-V4-Flash-0731's reasoning degenerated into repetition loops in
thinking mode: the model found the needle but repeated it endlessly without
generating `</think>`, so no answer was ever produced. Simple prompts
(math, logic, recall) worked fine — the failure was specific to retrieval
tasks with the chat template.

## Root cause

The checkpoint's `tokenizer.json` has BOS (`<｜begin▁of▁sentence｜>`, id 0)
in the vocab but **not** in `added_tokens`. The BPE pre-tokenizer splits
the BOS string into 10 mojibake subword tokens:

```
[30, 29429, 8277, 29429, 2154, 29429, 85, 51015, 29429, 32]
```

The model was trained with token 0 as BOS but received 10 subword tokens at
prompt start. This corrupted the prompt structure, causing the model to
behave unpredictably — most visibly, looping in thinking mode on retrieval
tasks.

The `<｜User｜>` / `<｜Assistant｜>` tags are also not in the vocab, but that
is correct: they were trained as subword sequences. Only BOS is broken
because it has a vocab entry (id 0) that the pre-tokenizer bypasses.

## Fix

`OpenAiTokenizer::encode` (`crates/infer-server/src/tokenizer.rs`): strip
the BOS string prefix and prepend token 0 directly.

```rust
let bos = "<｜begin▁of▁sentence｜>";
let (body, prepend_bos) = match text.strip_prefix(bos) {
    Some(rest) => (rest, true),
    None => (text, false),
};
// ... encode body, then ids.insert(0, 0) if prepend_bos
```

Verified: BOS-only prompt = 1 token (was 10). BOS + "Hello" = 2 tokens
(was 11). Token [0] and BOS string now produce identical model output.

## Result

DeepSeek-V4-Flash-0731, 2×H20, TP=2, W4AFP8, build `bos-fix`:

- **Before fix**: thinking mode on needle retrieval → endless repetition,
  no `</think>`, no answer at any context length (1K/8K/32K).
- **After fix**: model finds the needle, generates `</think>`, and answers
  correctly. Verified at 1K with instruction CUE:
  `content: "The secret access code is **738291**."` (ct=111–164).
- Simple prompts (math, logic, recall) unchanged — they already worked.
- Needle gate (RAW=1, no template): unchanged — the BOS fix only affects
  the chat template path.

Commit: `d4477f925`.

## Remaining: thinking-mode repetition on completion CUE

The model still loops when the prompt ends with a completion-style CUE
("The secret access code is") in thinking mode. With an instruction CUE
("Please recall and tell me...") the model succeeds. This is a prompt
format interaction, not a tokenization bug — the model completes the
phrase pattern instead of transitioning to `</think>`.

Mild `repetition_penalty=1.1` or `frequency_penalty=0.5` also helps
without losing the needle. `repetition_penalty=1.2` is too strong — the
model loses the needle entirely.

## Rule

A checkpoint packaging bug (vocab entry missing from `added_tokens`)
produces silent tokenization corruption that manifests as model
degeneration. When a model loops or degrades only through the chat
template path, verify every special token's round-trip:
`encode(decode(id)) == id` and `encode(string) == [id]`.
