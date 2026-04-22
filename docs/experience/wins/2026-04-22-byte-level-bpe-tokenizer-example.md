# Byte-Level BPE Tokenizer Example

## Context

Small-model training in this tree needed a tokenizer path that matches common
industry practice better than the existing whitespace word-level example. The
goal was to keep tokenizer policy out of the training binaries while still
supporting reusable `tokenizer.json` generation from plain-text corpora.

## What Worked

- Added [crates/train/examples/build_bpe_tokenizer.rs](/path/to/code/agent-infer/crates/train/examples/build_bpe_tokenizer.rs:1)
  as a generic byte-level BPE builder.
- Stayed on the official `tokenizers` typed-builder path instead of adding
  train-side tokenizer special cases.
- Defaulted to a mainstream byte-level stack:
  `Strip + NFC` normalizer, `ByteLevel` pre-tokenizer / post-processor, and
  optional `ByteFallback -> ByteLevel` decoder chaining when `--byte-fallback`
  is enabled.
- Kept the surface adjustable but small:
  repeatable `--corpus`, repeatable `--special-token`, `--vocab-size`,
  `--min-frequency`, `--unk-token`, `--add-prefix-space`, and
  `--byte-fallback`.

## Rule

When training needs a non-framework-specific tokenizer, add it as a generic
example that outputs a reusable `tokenizer.json`; do not hardcode corpus or
template policy into the core training binaries.
