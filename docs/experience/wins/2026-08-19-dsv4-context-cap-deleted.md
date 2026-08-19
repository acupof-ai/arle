# DSv4 32K context cap deleted — 1M context accepted — CUDA, 2026-08-19

> Status: Shipped

## Context

`DSV4_AUTO_CONTEXT_CEILING = 32768` capped DSv4's auto-resolved
`max_prompt_tokens` / `max_total_tokens` at 32K, added 2026-07-06 because
FlashMLA pool sizing crashed with large `max_seq_len`. The demand-paged
joint `(num_slots, pool_tokens)` budget solve (#154 Phase 3b, 2026-07-10)
fixed the crash 4 days later: per-slot costs that exceed VRAM reject
startup cleanly (`affordable=0`), and the pool scan degrades `num_slots`
instead of panicking. The cap was obsolete.

## What changed

Deleted the constant, its doc comment, the `cuda_model_is_dsv4` helper
(only caller was the cap), and all re-exports. `serve.rs` now passes the
checkpoint's `max_position_embeddings` through unmodified for every model
— one rule, no DSv4 special case.

Commit: `32670210d`.

## Result

DeepSeek-V4-Flash-0731, 4×H20, TP=4, W4AFP8, build `ctx-cap-v2`:

- Server starts with `max_prompt_tokens=1048576` (was 32768).
- Budget solver at 1M: free 55GB/GPU, per-slot 7.7GB, affordable=5,
  planned 4 slots, shared comp capacity 3.16M tokens. No crash.
- Needle-in-a-haystack (completions API, greedy):

| Prompt tokens | Needle found | Answer |
|--------------:|:------------:|--------|
| 10,520 | ✓ | ARLE-2026 |
| 34,220 | ✗ | ARLE-2028 (last digit wrong) |
| 69,420 | ✓ | ARLE-2026 |
| 139,820 | ✗ | ARLE-2028 (last digit wrong) |
| 280,620 | ✗ | ARLE-12799 (lost) |

The server accepts and serves prompts past the old 32K cap. Needle
accuracy degrades at longer contexts — a model/KV-precision issue, not a
context-cap issue.

## Rule

An artificial cap added as a stopgap before the underlying fix lands
must be deleted once the fix is verified. The demand-paged budget solve
is the fix; the cap was the stopgap.
