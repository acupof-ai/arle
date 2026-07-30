# DSpark proposal convention keyed on a field every checkpoint has

## Context

`DsparkConfig::from_dir` derived the proposal convention —
`next_token_heads`, i.e. block drafts `block_size` tokens (DSpark, anchor row
is the first prediction position) vs `block_size − 1` (DFlash, same-position)
— from `dflash_config` presence: nested block ⇒ DFlash. Found while checking
our implementation against the DeepSpec reference (user directive).

## Root Cause

A DSpark checkpoint's backbone IS DFlash, so it carries the same nested
`dflash_config` block. All three checkpoints on the host
(`dspark-fr-native`, `dspark-aeon`, `Qwen3.6-27B-DFlash`) have it —
`next_token_heads` was **permanently false**, and every DSpark checkpoint
drafted `block − 1` tokens, discarding the anchor row's logit (one free draft
position per chain, every chain). The reference never sniffs field nesting:
DeepSpec's `eval.py` dispatches on `architectures[0]`
(`DSparkDraftModel` vs `DFlashDraftModel`), and sglang does the same. The
comment on our detection ("DSpark keeps them top-level") described a
checkpoint layout that does not exist.

Invisible to the correctness gate: spec decode is lossless under either
convention — only the accept rate moves, and no assertion reads it.

## Fix

`92afc4a17` — key `next_token_heads` on `architectures[0].contains("DSpark")`,
matching the reference dispatch; field lookup still falls through
`dflash_config` → top-level for either flavor. Also fixed the batched
confidence-feature copy to offset by `first_row` (it read rows `0..n`, correct
only for the DSpark convention; DFlash checkpoints carry no confidence head,
which is the only reason it never fired wrong). Accept-rate A/B vs the 0.544
baseline: **pending-remote** (GPU deferred on request).

## Rule

**A discriminator must be validated against every class it discriminates.**
The nesting heuristic was only ever checked against DFlash checkpoints — the
one class where it is trivially right. One `python -c 'json.load(...)'` over
each checkpoint's config would have shown the field present in all three.
When a reference implementation exists, adopt its dispatch axis, not a proxy
inferred from one sample.
