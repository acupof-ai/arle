# Tokenizer refactor crashed every DSv4 serve at startup — CUDA, 2026-08-20

> Status: Fix committed (`a1b5dffe2`), verified on pod.

## Context

Commit `a9f0bab97` (2026-08-20) refactored the tokenizer cache to use upstream
`Vocab`/`Merges` types and switched `extract_cache_parts` from manual field
extraction to `serde_json::from_value`. The refactor passed local typecheck
but crashed every DSv4 serve at startup with "BPE model missing vocab/merges".

## Root Cause

DSv4's `tokenizer.json` stores BPE merges as space-separated strings
(`"Ġ t"`) instead of the HuggingFace-standard `["Ġ", "t"]` arrays.
`serde_json::from_value::<Merges>` (`Vec<(String, String)>`) cannot deserialize
a string, so `extract_cache_parts` returned `None`, and the `.context(...)`
propagated a fatal error.

The pre-refactor code was unaffected because it built the tokenizer from the
full JSON via `build_tokenizer(&value)` (the tokenizers crate's deserializer
handles both formats); `extract_cache_parts` only fed the cache writer, and
its `filter_map` silently skipped string-format merges (producing an empty
merges list in the cache — a latent correctness bug that never surfaced
because the cache was only read on the cold path, and the cold path rebuilt
from the full JSON).

## Fix

Handle both formats in `extract_cache_parts`: arrays deserialize directly;
strings split on the first space.

## Rule

When a refactor changes a fallback path into the primary path, test it
against every production checkpoint format — the old code's silent skip
became the new code's hard crash.
