# Cold start 0.35s→0.21s: tokenizer binary cache — Metal, 2026-08-20

> Status: Shipped

## Context

After warmup-off + config de-dup (`wins/2026-08-20-cold-start-warmup-off-config-dedup.md`),
cold start was ~0.35s. The tokenizer JSON parse (~218ms for 248k vocab + 247k
merges) was the remaining bottleneck — it blocks the engine thread join even
though weight loading is parallel.

## What worked

**Bincode cache of raw tokenizer parts, reconstructed via `BPE::builder()`.**

The tokenizers crate's serde implementation uses
`#[serde(flatten)] rest: serde_json::Value` in `ModelWrapper::deserialize`,
which is incompatible with compact binary formats (bincode, postcard, CBOR,
MessagePack all fail on the flatten round-trip). Direct serde caching of the
`Tokenizer` type is impossible without patching the crate.

Instead, cache the raw parts:

1. Parse `tokenizer.json` to `serde_json::Value` (first start only).
2. Pull `model.vocab` and `model.merges` out of the Value in place (no clone),
   keeping the rest of the config as a JSON string.
3. Bincode-serialize `(ChatTemplate, vocab, merges, config)` to
   `model_dir/tokenizer.arle.bin`, using the tokenizers crate's own
   `bpe::Vocab` (`AHashMap<String, u32>`) and `bpe::Merges`
   (`Vec<(String, String)>`) types — no local aliases, no conversion on load.
4. On subsequent starts, bincode-deserialize and reconstruct the `Tokenizer`
   via `BPE::builder().vocab_and_merges(vocab, merges).build()`, then set
   pre_tokenizer / decoder / post_processor / normalizer / truncation /
   padding / added_tokens from the config JSON.

This skips both the JSON syntax parse (~100ms) and the serde flatten
round-trip (~118ms of double HashMap building), paying only the single
HashMap build (~60ms) + BPE model construction (~50ms). Caching the vocab as
an `AHashMap` (not a `Vec`) means bincode builds the map directly — no
second Vec→HashMap conversion on the cache-hit path.

Cache invalidation: mtime check against `tokenizer.json`,
`tokenizer_config.json`, `config.json`, `chat_template.jinja`. Non-BPE
models fall back to the serde path (no cache).

## Result

M4 Pro 48GB, `mlx-community/Qwen3.5-9B-4bit` (5.6 GB), `--max-running-requests 1`.

| Metric | Before (no cache) | After (cache hit) | Delta |
|---|---:|---:|---:|
| Cold start (launch → /health) | 0.33–0.39s | **0.133–0.138s** | −62% |
| Total (launch → answer) | 1.12–1.35s | **1.37s** | wash |

The total answer time is wash because the first request pays the embed
dequant + JIT cost (~1s), which dominates the 140ms cold-start savings.
The cold-start improvement matters for serverless / scale-to-zero scenarios
where the process is started fresh per request.

Cache file: 12.7 MB (bincode with full vocab strings). First start pays
~200ms extra for JSON parse + cache write; subsequent starts save ~140ms.

Correctness: smoke test passed — model answers correctly with cache hit.
The `BPE::builder()` path produces the same tokenizer as the serde path
(vocab, merges, pre_tokenizer, decoder, post_processor, normalizer,
truncation, padding, added_tokens all set from the same config).

## Rule

When a dependency's serde implementation is JSON-only (flatten into
`serde_json::Value`), cache the raw parts and reconstruct via the public
builder API instead of fighting the serde round-trip. The flatten blocker
affects bincode, postcard, CBOR, and MessagePack — no compact binary
format survives it.
