# `--max-prompt-tokens` enforced as a ceiling, not a floor (#145) — pod-verified

## Context

`EngineLoadConfig::scheduler_config` (`infer-api/src/loaded.rs`) clamped
`max_prompt_tokens` with `.max(KV_capacity − gen_reserve)`. An explicit
`--max-prompt-tokens 4096` was therefore *raised* to the KV capacity
(114688 on DSv4-Flash TP=4), so an over-length prompt was admitted and could
write past the fixed DSv4 KV bands (memory-unsafe). Default was `32_768`, also a
floor.

## What Worked

One-line clamp flip `.max` → `.min` (capacity is a hard ceiling, never a floor);
default → `usize::MAX` sentinel = "unset → capacity-bound". `f0648c385`.
NOT adding the issue's suggested band-capacity `ensure!` in `flashmla_alloc_append`
— `set_band_cursor` (paged_kv.rs:823) documents that exact check as already tried
and reverted; the bound belongs at ingress, not at the logical cursor.

Pod A/B (DSv4-Flash-FP8, TP=4, `--max-prompt-tokens 4096`), the discriminating
case sits *above* the 4096 cap but *below* KV-capacity−reserve:

| prompt_tokens | result |
|---|---|
| 18002 (>cap, <KV cap) | **rejected** — `finish_reason:"abort"`, `completion_tokens:0` |
| 9 (<cap) | accepted, coherent generation |

Pre-fix `.max` would have raised the cap to 114688 and *accepted-and-generated*
the 18002-token prompt. Rank config log confirms `max_prompt_tokens: 4096` on all
4 TP ranks (not raised). Abort mechanism: length guard `infer-core/src/lib.rs:817`
(`prompt_tokens.len() > config.max_prompt_tokens` → `FinishReason::Abort`).

## Rule

ARLE's admission-reject surfaces as **HTTP 200 + `finish_reason:"abort"` +
`completion_tokens:0`**, not a 4xx — the runtime's empty-completion abort
semantic. A length-cap fix's discriminating test prompt must land *between* the
requested cap and the KV capacity; a prompt merely over the cap but also over
capacity can't distinguish `.min` from `.max`.
