# 2026-05-07 · RadixCache silent corruption when tokenizer is hot-swapped

## Context

Surfaced 2026-05-07 during M6 / M_e gauntlet review when the user
flagged the BPE/SentencePiece determinism risk surface
(Unicode normalization, byte-level whitespace, chat-template injection,
fast/slow tokenizer divergence, cross-segment composition, vocab/merges
drift). Mapping that checklist to ARLE exposes a silent-corruption hole.

## What's missing

`grep -rnE "tokenizer.*hash|tokenizer.*fingerprint|sha256.*tokenizer|namespace.*tokenizer|cache_key.*tokenizer" infer/src/`
returns zero hits. Concretely:

- `infer/src/tokenizer.rs::Tokenizer::from_file(path)` loads
  `models/<model>/tokenizer.json` directly; the file is unversioned and
  there is no commit-hash / sha256 capture at load time.
- `Tokenizer::encode(text)` calls `tokenizer.encode(text)` with no
  Unicode normalization (no NFC entry-point gate).
- `infer/src/prefix_cache.rs::RadixCache` keys nodes by
  `&[u32]` token id sequence. The cache namespace does NOT include any
  tokenizer fingerprint. There is no tokenizer-version invalidation
  hook on swap.

## Silent corruption sequence

1. Server starts, loads `tokenizer.json` v1, computes KV for prefixes
   under v1 vocab, publishes blocks to RadixCache keyed by `(v1 token
   id sequence) → BlockId`. The KV bytes in those blocks were produced
   by the model running on **v1 token ids** (which the model interprets
   as v1 embedding vectors).
2. Operator hot-swaps `tokenizer.json` to v2 (vocab/merges drift —
   common when an HF repo author updates without bumping the repo
   name, or when a `add_tokens()` call grows the vocab). Server keeps
   running; the loaded `Tokenizer` is the v1 instance still in memory,
   but a *new connection* via re-init / reload reads v2.
3. **Most BPE / SentencePiece sequences for ASCII-only text are
   identical between v1 and v2** — vocab updates typically only touch
   rarer codepoints. So v2 tokenize produces a `&[u32]` whose long
   prefix matches v1 → RadixCache HIT.
4. The cached KV blocks were computed from v1 token-id-to-embedding
   mapping. The active model is the same physical model. If v1 and
   v2 happen to map the matching prefix tokens to the same embedding
   row (very common for ASCII), the KV is silently correct.
5. **At the prompt suffix or generation step**, v2 emits a token id
   that under v2 means one thing and under v1 meant something else.
   The model embedding lookup uses v2 ids. The downstream attention
   reads the cached v1-prefix KV. The math sees inconsistent
   embeddings. Output drifts subtly. **No log, no error, no test
   failure** — just silently wrong continuations.

The same shape fires in less exotic conditions: byte-level BPE
treating `" hello"` (leading space) vs `"hello"` differently, fast vs
slow tokenizer divergence on emoji or runs of punctuation, chat
template injection running differently across requests.

## Why "cache miss is safe but cache hit is dangerous"

If v1 and v2 produced *different* token id sequences for the same
text (e.g. NFC vs NFD form of `é`), RadixCache would simply not hit
on the second-tokenizer's prompt — performance loss only, no
correctness issue. The dangerous regime is v2 tokenize that **shares
a long prefix with v1 tokenize** (the common case for ASCII), letting
the lookup return blocks that were generated under a different vocab
contract.

## Production mitigations (per the user's 2026-05-07 expert summary)

1. **Lock tokenizer file commit hash into the deployment image**.
   `Tokenizer::from_file` should compute SHA-256 of the file at load
   time and stash it as `tokenizer_fingerprint: [u8; 32]`.
2. **NFC normalization at the entry point** before
   `tokenizer.encode`. Apply `unicode_normalization::nfc` (or
   equivalent) so `e + U+0301` and `U+00E9` produce the same token id
   stream.
3. **Render chat template before tokenize**, and include the chat
   template version (system prompt hash + role template hash) in the
   RadixCache namespace. Two requests with same `user content` but
   different `system content` must NOT share radix nodes.
4. **No token-segment composition reuse**. ARLE today does not
   compose KV blocks from different requests' token id segments
   (RadixCache walks one prefix), so this constraint already holds —
   keep it that way: future "share KV across requests by hash" work
   must operate on text-level fingerprints, not token id slices.
5. **Tokenizer fingerprint as RadixCache namespace prefix**. All
   `RadixCache::lookup` keys should be `(tokenizer_fingerprint,
   chat_template_fingerprint, &[u32])`. On tokenizer swap, the entire
   RadixCache invalidates by namespace — no silent reuse of stale
   blocks.

## Implementation sketch (next milestone — tentatively M3.7 or M_d.1)

| Step | File | Notes |
|---|---|---|
| 1. Compute and store `tokenizer_fingerprint` at load | `infer/src/tokenizer.rs` | `sha256(read(path))`, surface via `pub fn fingerprint(&self) -> [u8;32]` |
| 2. NFC entry-point (optional gate by env `INFER_TOKENIZER_STRICT_NFC=1`) | `infer/src/tokenizer.rs::encode` | use `unicode-normalization` crate |
| 3. RadixCache namespace key = (tokenizer_fp, chat_template_fp, token_ids) | `infer/src/prefix_cache.rs::RadixCache::lookup_or_stage` | small change — the namespace becomes a prefix of the key |
| 4. Server boot logs tokenizer fingerprint at INFO; refuses swap mid-run unless explicitly allowed | `infer/src/main.rs` boot path + `arle serve --doctor` output | block silent corruption by failing loud at the obvious operator action |
| 5. `infer/tests/tokenizer_fingerprint_radix_isolation.rs` | new test | run two server instances with different fake-fingerprinted tokenizers against the same RadixCache; assert that v2 lookup of a v1-published prefix returns MISS, not HIT |

## Hardware / context

Discovered during M6 gauntlet planning, not on hardware. No bench
artifact. Mitigation acceptance is correctness, not performance.

## Rule

- New cross-request KV reuse paths (Tier-KV M_d, MagicDec sparse
  spec, future content-addressable persistence) **must** key on
  `(tokenizer_fp, chat_template_fp, …)` not `&[u32]` alone.
- Operator-facing docs (`docs/environment.md`, `infer/src/main.rs`
  doctor output) state the tokenizer-swap policy: silent swap is
  forbidden; stop-restart-clear-cache is the contract.
- Any RadixCache fork (per-session, per-tenant) inherits the same
  namespace contract; no path may bypass the fingerprint prefix.
